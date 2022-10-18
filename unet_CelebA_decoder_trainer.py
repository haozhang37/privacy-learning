import os

import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib # rqh 0502 add
matplotlib.use('agg') # rqh 0502 add
import matplotlib.pyplot as plt
import random
import math
import argparse

from tools.lib import *
#new decoder
from resnet_decoder.get_model_resnet_decoder import get_model_resnet_decoder
from res_model_generator.get_model_generator import Get_Model_Generator
from res_model.get_model_resnet import Get_Model_ResNet
from encrypt_layer import encrypt
from dataset import ourDataset
from alexnet_model.get_model_alexnet import Get_Model_AlexNet
from unet_model_revise.get_model_unet import Get_Model_Unet


class Decoder:
    def __init__(self, args):
        self.IsEncrypt = args.is_encrypt
        self.IsNoise = args.is_noise
        self.encrypt_location = args.encrypt_position
        if self.encrypt_location == 'a':
            self.train_id = 1
        elif self.encrypt_location == 'b':
            self.train_id = 2
        elif self.encrypt_location == 'c':
            self.train_id = 3
        self.lr = args.decoder_lr
        self.show_steps = args.show_step

        self.load_epoch = args.load_net_epoch

        self.num_classes = args.net_class_num
        self.num_layer = args.net_layer_num
        self.visual_simple_num = args.visual_num
        self.bs = args.batch_size
        if args.device != "cpu":
            self.device = int(args.device)
        else:
            self.device = args.device
        self.num_epoch = args.decoder_epoch
        self.decoder_type = args.decoder_type

        #  model parameters
        self.num_input_channels = args.num_input_channels
        self.num_block = args.num_block
        self.conv_per_block = args.conv_per_block
        self.num_block_upsample = args.num_block_upsample
        self.if_alexnet = args.if_alexnet

        # GAN parameters
        self.G_boardsize = args.G_boardsize
        self.G_channel = args.G_channel
        self.G_block = args.G_block
        self.G_kernel = args.G_kernel
        self.normal_id = args.normal_id

        self.dim = args.dim

        #  paths
        self.data_path = args.data_path
        self.result_path = args.result_path
        if self.decoder_type == 1:
            self.model_path = os.path.join(self.result_path, 'net_origin')
            self.decoder_path = os.path.join(self.result_path, 'unet_decoder_origin')
        elif self.decoder_type == 2:
            self.model_path = os.path.join(self.result_path, 'net_addlayer')
            self.decoder_path = os.path.join(self.result_path, 'unet_decoder_addlayer')
        elif self.decoder_type == 3:
            self.model_path = os.path.join(self.result_path, 'net_privacy_GAN_stage')
            self.decoder_path = os.path.join(self.result_path, 'unet_decoder_privacy_GAN_noencrypt_stage')
        elif self.decoder_type == 4:
            self.model_path = os.path.join(self.result_path, 'net_privacy_GAN_stage')
            self.decoder_path = os.path.join(self.result_path, 'unet_decoder_privacy_GAN_encrypt_stage')
        else:
            print("Illegal decoder type!")
            return

        self.sub_model_path = os.path.join(self.model_path, str(self.train_id))
        self.sub_decoder_path = os.path.join(self.decoder_path, str(self.train_id))

        if not os.path.exists(self.decoder_path):
            os.mkdir(self.decoder_path)
        if not os.path.exists(self.sub_decoder_path):
            os.mkdir(self.sub_decoder_path)

        self.load_part1_path = os.path.join(self.sub_model_path, "model_part1_" + str(self.load_epoch) + ".bin")
        self.load_part2_path = os.path.join(self.sub_model_path, "model_part2_" + str(self.load_epoch) + ".bin")
        self.load_part3_path = os.path.join(self.sub_model_path, "model_part3_" + str(self.load_epoch) + ".bin")
        self.load_G_path = os.path.join(self.sub_model_path, "Generator_" + str(self.load_epoch) + ".bin")

        #dataset param
        self.celeba_mean = [0.5, 0.5, 0.5]
        self.celeba_std = [0.5, 0.5, 0.5]

    def prepare_Model(self):
        #decoder
        self.decoder = Get_Model_Unet(gpu_id=self.device, num_input_channels=self.num_input_channels,
                                      num_classes=3, conv_per_block=self.conv_per_block)
        self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        self.criterion_decoder = torch.nn.MSELoss(reduction="mean")

        #resnet
        self.resnet_part1, self.resnet_part2, self.resnet_part3 = Get_Model_AlexNet(self.device, self.IsEncrypt, self.dim,
                                                                    double_filter=False, double_layer=False,
                                                                    num_classes=self.num_classes, num_layer=self.num_layer,
                                                                    encrypt_location=self.encrypt_location)
        self.G = None

        self.resnet_part1.load_state_dict(torch.load(self.load_part1_path, map_location=torch.device("cpu")))
        self.resnet_part2.load_state_dict(torch.load(self.load_part2_path, map_location=torch.device("cpu")))
        self.resnet_part3.load_state_dict(torch.load(self.load_part3_path, map_location=torch.device("cpu")))

        if self.decoder_type > 1:
            self.G = Get_Model_Generator(board_size=self.G_boardsize,
                                         num_input_channels=self.G_channel,
                                         num_block=self.G_block, gpu_id=self.device,
                                         kernel_size=self.G_kernel)
            self.G.load_state_dict(torch.load(self.load_G_path, map_location=torch.device("cpu")))

    def prepare_Dataset(self):
        normalize = transforms.Normalize(mean=self.celeba_mean, std=self.celeba_std)
        training_set = ourDataset('list_attr_celeba.txt', training=True)
        self.train_loader = torch.utils.data.DataLoader(training_set, batch_size=self.bs, shuffle=True)

        val_set = ourDataset('list_attr_celeba.txt', training=False)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.bs, shuffle=False)

    def save_epoch(self,loss_list,epoch):
        visual_path = os.path.join(self.sub_decoder_path, str(epoch))
        if os.path.exists(visual_path)is False:
            os.mkdir(visual_path)

        save_decoder_path = os.path.join(self.sub_decoder_path, "decoder" + str(epoch) + ".bin")
        save_list_path = os.path.join(self.sub_decoder_path, "list_" + str(epoch) + ".bin")
        torch.save(self.decoder.state_dict(), save_decoder_path)
        torch.save(loss_list, save_list_path)

    def get_privacy_key(self, bs):  # here bs was not divided by dim yet
        temp_tensor = torch.randn(self.dim, self.dim)
        rot_mat = torch.zeros(bs // self.dim, self.dim, self.dim)
        key_num = 0
        while key_num < bs // self.dim:
            nn.init.orthogonal_(temp_tensor)
            if torch.det(temp_tensor) > 0:
                rot_mat[key_num] = temp_tensor
                key_num = key_num + 1
        return rot_mat.to(self.device)

    def train_model(self):
        self.decoder.train().to(self.device)
        self.resnet_part1.eval().to(self.device)
        self.resnet_part2.eval().to(self.device)
        self.resnet_part3.eval().to(self.device)
        if self.decoder_type > 1:
            self.G.eval().to(self.device)
        train_loss = AverageMeter()
        train_loss.reset()

        for i, (image, celeba_label) in enumerate(self.train_loader):
            image_size = image.size()
            batch_use = image.size(0)
            batch_use = batch_use // self.dim * self.dim
            if batch_use == 0:
                continue
            image = image[:batch_use]

            rot_mat = self.get_privacy_key(batch_use)
            if self.IsEncrypt == True:
                if self.IsNoise == True:
                    noise = get_noise(image,mean=self.celeba_mean,std=self.celeba_std)
                    image = torch.cat((image,noise),0)
            image = image.to(self.device)
            with torch.no_grad():
                encrypt_inputs = self.resnet_part1(image)
                if self.decoder_type > 1:
                    G_outputs = self.G(encrypt_inputs.detach())

                for channel_id in range(3):
                    image[:, channel_id, :, :] = image[:, channel_id, :, :] * self.celeba_std[channel_id]
                    image[:, channel_id, :, :] = image[:, channel_id, :, :] + self.celeba_mean[channel_id]
                origin_image = image * 255 - 127

                decoder_image = origin_image
                if self.decoder_type == 1:
                    decoder_feature = encrypt_inputs.detach()
                elif self.decoder_type == 2:
                    decoder_feature = G_outputs.detach()
                elif self.decoder_type == 3:
                    decoder_feature = G_outputs.detach()
                elif self.decoder_type == 4:
                    encrypt_outputs = encrypt.apply(G_outputs.detach(), rot_mat)
                    decoder_feature = encrypt_outputs.detach()
            upsample_method = nn.Upsample(size=image_size[-1], mode='bilinear')
            decoder_feature = upsample_method(decoder_feature)
            decoder_out = self.decoder(decoder_feature)
            loss = self.criterion_decoder(decoder_out, decoder_image)
            train_loss.update(loss.cpu().detach())

            self.optimizer_decoder.zero_grad()
            loss.backward()
            self.optimizer_decoder.step()
            if i % self.show_steps == 0:
                print("[train][" + str(i) + "/" + str(len(self.train_loader)) + "]" + "\n"
                      + "train_loss:" + str(loss) + "\n")

        return train_loss.avg

    def val_model(self):
        self.resnet_part1.eval().to(self.device)
        self.resnet_part2.eval().to(self.device)
        self.resnet_part3.eval().to(self.device)
        self.decoder.eval().to(self.device)
        if self.decoder_type > 1:
            self.G.eval().to(self.device)
        val_loss = AverageMeter()
        val_loss.reset()
        Origins = [[] for _ in range(self.dim)]  # list of length dim
        Decoder = []

        with torch.no_grad():
            for i, (image, celeba_label) in enumerate(self.val_loader):
                image_size = image.size()
                batch_use = image.size(0)
                batch_use = batch_use // self.dim * self.dim
                if batch_use == 0:
                    continue
                image = image[:batch_use]

                rot_mat = self.get_privacy_key(batch_use)
                if self.IsEncrypt == True:
                    if self.IsNoise == True:
                        noise = get_noise(image, mean=self.celeba_mean, std=self.celeba_std)
                        image = torch.cat((image, noise), 0)
                image = image.to(self.device)

                encrypt_inputs = self.resnet_part1(image)
                if self.decoder_type > 1:
                    G_outputs = self.G(encrypt_inputs.detach())

                for channel_id in range(3):
                    image[:,channel_id,:,:] = image[:,channel_id,:,:]*self.celeba_std[channel_id]
                    image[:,channel_id,:,:] = image[:,channel_id,:,:]+self.celeba_mean[channel_id]
                origin_image = image*255-127

                decoder_image = origin_image
                if self.decoder_type == 1:
                    decoder_feature = encrypt_inputs.detach()
                elif self.decoder_type == 2:
                    decoder_feature = G_outputs.detach()
                elif self.decoder_type == 3:
                    decoder_feature = G_outputs.detach()
                elif self.decoder_type == 4:
                    encrypt_outputs = encrypt.apply(G_outputs.detach(), rot_mat)
                    decoder_feature = encrypt_outputs.detach()

                upsample_method = nn.Upsample(size=image_size[-1], mode='bilinear')
                decoder_feature = upsample_method(decoder_feature)
                decoder_out = self.decoder(decoder_feature)
                loss = self.criterion_decoder(decoder_out, decoder_image)
                val_loss.update(loss.cpu().detach())

                if i % self.show_steps == 0:
                    print("[val] [" + str(i) + "/" + str(len(self.val_loader)) + "]" + "\n"
                          + "val_loss:" + str(loss) + "\n")

                if i <self.visual_simple_num:
                    origin_image = origin_image.data.cpu()
                    for d in range(self.dim): # d from 0 to dim-1
                        Origin_img = self.generateImg(origin_image[batch_use * d // self.dim])
                        Origins[d].append(Origin_img)

                    decoder_out_visual = decoder_out.data.cpu()
                    Decoder_img = self.generateImg(decoder_out_visual[0])

                    Decoder.append(Decoder_img)

            return Origins, Decoder, val_loss.avg # Origins: list of (dim, visual_simple_num)

    def generateImg(self, origin_image):
        origin_visual = np.array(origin_image)
        origin_visual = origin_visual + 127
        origin_visual = origin_visual.transpose((1, 2, 0))
        origin_visual = np.uint8(origin_visual)
        origin_img = Image.fromarray(origin_visual, 'RGB')
        return origin_img

    def saveImg(self,epoch, Origin_imgs, Decoder_img):
        visual_path = os.path.join(self.sub_decoder_path, str(epoch))
        if os.path.exists(visual_path) == False:
            os.mkdir(visual_path)
        for i in range(len(Origin_imgs[0])):  # visual_simple_num
            for d in range(self.dim):
                fig_path = os.path.join(visual_path, "origin_d%d_" % (d+1) + str(i) + ".png")
                Origin_imgs[d][i].save(fig_path)
            fig_path = os.path.join(visual_path, 'decoder_out_' + str(i) + ".png")
            Decoder_img[i].save(fig_path)

    def drawFig(self,train_loss_list,val_loss_list):
        x = np.arange(0, len(train_loss_list), 1)
        y1 = np.array(train_loss_list)
        y2 = np.array(val_loss_list)
        plt.figure(str(self.train_id))
        plt.subplot(111)
        plt.plot(x, y1, color='blue')
        plt.plot(x, y2, color='red')
        plt.title('celeba_loss')

        fig_path = os.path.join(self.decoder_path, 'resnet_decoder_' + str(self.train_id) + ".png")
        plt.savefig(fig_path)

    def adjust_lr(self, lr):
        for param_group in self.optimizer_decoder.param_groups:
            param_group["lr"] = lr

    def work(self):
        val_loss_list = []
        train_loss_list = []

        st = round(math.log(self.lr, 10))
        logs = torch.logspace(start=st, end=st - 1, steps=self.num_epoch)
        for epoch in range(0, self.num_epoch):
            self.adjust_lr(logs[epoch])
            train_loss=self.train_model()
            train_loss_list.append(train_loss)
            Origin_imgs, Decoder_img, val_loss = self.val_model()
            val_loss_list.append(val_loss)
            self.saveImg(epoch,Origin_imgs, Decoder_img)
            print(
                "\nSummary of epoch {ep}\ntrain loss: {train_l}, test loss: {test_l}\n".format(
                    ep=epoch, train_l=train_loss, test_l=val_loss
                ))
            if epoch % 10 == 9 or epoch == self.num_epoch - 1:
                self.save_epoch([train_loss_list, val_loss_list], epoch)
            self.drawFig(train_loss_list,val_loss_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="1", type=str)
    parser.add_argument("--data-path", default="./data/cropped_CelebA", type=str)
    # parser.add_argument("--batch-size", default=15, type=int) # original default value: 12
    parser.add_argument("--net-layer-num", default=20, type=int)
    parser.add_argument("--net-class-num", default=40, type=int)
    parser.add_argument("--visual-num", default=1, type=int)
    parser.add_argument("--encrypt-position", default="c", type=str)
    parser.add_argument("--show-step", default=50, type=int)
    parser.add_argument("--if-alexnet", default=0, type=int)
    parser.add_argument("--num-input-channels", default=384, type=int)
    parser.add_argument("--num-block", default=6, type=int)
    parser.add_argument("--num-block-upsample", default=4, type=int)
    parser.add_argument("--is-encrypt", default=True, type=bool)
    parser.add_argument("--is-noise", default=False, type=bool)
    parser.add_argument("--result-path", default="Alexnet_CelebA", type=str)
    parser.add_argument("--decoder-type", default=3, type=int, choices=[1, 2, 3, 4],
                        help="1: origin, 2: add_layer, 3: privacy-noencrypt, 4: privacy-encrypt")

    # parameters for classifier
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)

    # parameters for decoder
    parser.add_argument("--decoder-epoch", default=5, type=int)
    parser.add_argument("--decoder-lr", default=1e-4, type=float)
    parser.add_argument("--conv-per-block", default=6, type=int)
    parser.add_argument("--load-net-epoch", default=9, type=int)

    # parameteer for GAN
    parser.add_argument("--G-boardsize", default=13, type=int)
    parser.add_argument("--G-channel", default=384, type=int)
    parser.add_argument("--G-block", default=1, type=int)
    parser.add_argument("--G-kernel", default=3, type=int)
    parser.add_argument("--normal-id", default="6", type=str)

    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5]) # rqh 0514 add

    args = parser.parse_args()
    args.root = os.getcwd()
    args.batch_size = 15 if args.dim == 5 else 12
    args.result_path = "result_dim%d/" % args.dim + args.result_path

    seed = int(args.net_layer_num + args.net_class_num + args.decoder_type)
    seed_torch(seed)
    if (args.decoder_type > 2 and not args.is_encrypt) or (args.decoder_type <= 2 and args.is_encrypt):
        raise RuntimeError("Invalid choise for the args is_encrypt!")

    decoder = Decoder(args)
    decoder.prepare_Model()
    decoder.prepare_Dataset()

    decoder.work()


if __name__ == "__main__":
    main()
