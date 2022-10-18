import os
import torch
import numpy as np
import argparse
import math
import matplotlib # rqh 0502 add
matplotlib.use('agg') # rqh 0502 add
from matplotlib import pyplot as plt

from torchvision import transforms
from torchvision import datasets
from torch import optim
from torch import nn
from PIL import Image

from res_model.get_model_resnet import Get_Model_ResNet
from alexnet_model.get_model_alexnet import Get_Model_AlexNet
from res_model_generator.get_model_generator import Get_Model_Generator
from res_model_discriminator.get_model_discriminator import Get_Model_Discriminator
from lenet_model.get_model_lenet import Get_Model_Lenet
from encrypt_layer import encrypt
from decrypt_layer_figure import decrypt_figure
from tools.lib import *

from dataset import ourDataset


class Privacy_Trainer:
    def __init__(self, args):
        self.path = {}
        self.path["root"] = os.getcwd()
        self.epoch_num = args.epoch_num
        self.init_lr = args.lr
        self.bs = args.batch_size
        self.start_epoch = args.start_epoch  # 0 means train from beginning, otherwise, continue to train
        self.from_begin = (args.start_epoch == 0)
        if args.device == "cpu":
            self.device = args.device
        else:
            self.device = int(args.device)
        self.num_classes = args.num_classes
        self.num_layers = args.num_layers
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.pretrained = args.pretrained
        self.is_encrypt = args.is_encrypt
        self.double_filter = args.double_filter
        self.double_layer = args.double_layer
        self.encrypt_location = args.encrypt_location
        self.model_name = args.sub_result_path
        self.is_CelebA = args.is_CelebA
        self.milestones = args.milestone_list

        self.dim = args.dim

        # Parameter of GAN
        self.G_boardsize = args.G_boardsize
        self.G_channel = args.G_channel
        self.G_block = args.G_block
        self.G_kernel = args.G_kernel
        self.D_boardsize = args.D_boardsize
        self.D_channel = args.D_channel
        self.D_block = args.D_block
        self.D_kernel = args.D_kernel
        self.G_lr = args.G_lr
        self.D_lr = args.D_lr
        self.GAN_step = args.GAN_step
        self.clip_threshold = args.clip_threshold
        self.GAN_theta = args.GAN_theta

        # ------------------ GAN key ----------------------------------------------
        self.GAN_num = 0
        # -------------------------------------------------------------------------

        self.path["result_path"] = os.path.join(self.path["root"], args.result_path)
        if not os.path.exists(self.path["result_path"]):
            os.makedirs(self.path["result_path"])
        self.path["result_path"] = os.path.join(self.path["result_path"], args.sub_result_path)
        if not os.path.exists(self.path["result_path"]):
            os.makedirs(self.path["result_path"])

    def prepare_Model(self, model, optimizer=None, GAN_optimizer=None, loss_fn=None, scheduler=None):
        # Save the parameters to use in the future, these parameter can initialize in __init__ function as well

        # initialize the models
        self.model_part1, self.model_part2, self.model_part3 = model(self.device, self.is_encrypt, self.dim, self.double_filter,
                                                                     self.double_layer, self.num_classes,
                                                                     self.num_layers, self.encrypt_location)
        self.G = Get_Model_Generator(board_size=self.G_boardsize, num_input_channels=self.G_channel,
                                     num_block=self.G_block, gpu_id=self.device, kernel_size=self.G_kernel)
        self.D = Get_Model_Discriminator(board_size=self.D_boardsize, num_input_channels=self.D_channel,
                                         num_block=self.D_block, gpu_id=self.device, kernel_size=self.D_kernel)
        self.list = [[] for _ in range(4)]  # result list: [train loss, train error, val loss, val error,]
        self.GAN_list = [[] for _ in range(10)]
        # GAN result list: [train_G_loss, train_D_loss, train_D_error, train_D_pos_loss, train_D_neg_loss,
        #                   val_G_loss,   val_D_loss,   val_D_error,   val_D_pos_loss,   val_D_neg_loss]

        if optimizer is not None:
            self.optimizer = optimizer([{"params":self.model_part1.parameters(),'lr':self.init_lr,'initial_lr':self.init_lr},
                                        {"params":self.G.parameters(),'lr':self.init_lr,'initial_lr':self.init_lr},
                                        {"params":self.model_part2.parameters(),'lr':self.init_lr,'initial_lr':self.init_lr},
                                        {"params":self.model_part3.parameters(),'lr':self.init_lr,'initial_lr':self.init_lr}],
                                         momentum=self.momentum, weight_decay=self.weight_decay)
        if GAN_optimizer is not None:
            self.G_optimizer = GAN_optimizer([{"params":self.model_part1.parameters(),'initial_lr':self.G_lr},
                                              {"params":self.G.parameters(),'initial_lr':self.G_lr}],lr=self.G_lr)
            self.D_optimizer = GAN_optimizer([{"params":self.D.parameters(),'initial_lr':self.D_lr}],lr=self.D_lr)
        if loss_fn is not None:
            self.loss_fn = loss_fn(reduction="mean").to(self.device)

        # Load latest epoch if needed
        if not self.from_begin:
            self.load_Latest_Epoch()

    def prepare_Dataset(self, train_set=None, val_set=None, test_set=None, **kwargs):
        # Save the parameters to use in the future, these parameter can initialize in __init__ function as well
        # Instantiate the dataset and dataloader
        if train_set is not None:
            self.path["train_path"] = os.path.join(self.path["root"], kwargs["train_subpath"])
            self.train_transform = kwargs["train_transform"]
            self.train_shuffle = kwargs["train_shuffle"]
            if train_set == datasets.CIFAR10 or train_set == datasets.CIFAR100:
                self.train_set = train_set(self.path["train_path"], transform=self.train_transform, train=True,
                                           download=True)
            elif train_set == datasets.ImageFolder:
                self.train_set = train_set(self.path["train_path"], transform=self.train_transform)
            elif train_set == ourDataset:
                self.train_set = train_set('list_attr_celeba.txt', training=True)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.bs,
                                                            shuffle=self.train_shuffle)

        if val_set is not None:
            self.path["val_path"] = os.path.join(self.path["root"], kwargs["val_subpath"])
            self.val_transform = kwargs["val_transform"]
            self.val_shuffle = kwargs["val_shuffle"]
            if val_set == datasets.CIFAR10 or train_set == datasets.CIFAR100:
                self.val_set = val_set(self.path["val_path"], transform=self.val_transform, train=False, download=True)
            elif val_set == datasets.ImageFolder:
                self.val_set = val_set(self.path["val_path"], transform=self.val_transform)
            elif val_set == ourDataset:
                self.val_set = val_set('list_attr_celeba.txt', training=False)
            self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.bs,
                                                          shuffle=self.val_shuffle)

        if test_set is not None:
            self.path["test_path"] = os.path.join(self.path["root"], kwargs["test_subpath"])
            self.test_transform = kwargs["test_transform"]
            self.test_shuffle = kwargs["test_shuffle"]
            self.test_set = test_set(self.path["test_path"], transform=self.test_transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.bs, shuffle=self.test_shuffle)

    def load_Latest_Epoch(self):
        model_part1_path = os.path.join(self.path["result_path"], "model_part1_" + str(self.start_epoch - 1) + ".bin")
        model_part2_path = os.path.join(self.path["result_path"], "model_part2_" + str(self.start_epoch - 1) + ".bin")
        model_part3_path = os.path.join(self.path["result_path"], "model_part3_" + str(self.start_epoch - 1) + ".bin")
        G_path = os.path.join(self.path["result_path"], "Generator_" + str(self.start_epoch - 1) + ".bin")
        D_path = os.path.join(self.path["result_path"], "Discriminator_" + str(self.start_epoch - 1) + ".bin")
        list_path = os.path.join(self.path["result_path"], "list_" + str(self.start_epoch - 1) + ".bin")
        GAN_list_path = os.path.join(self.path["result_path"], "GAN_list_" + str(self.start_epoch - 1) + ".bin")
        self.model_part1.load_state_dict(torch.load(model_part1_path).state_dict())
        self.model_part2.load_state_dict(torch.load(model_part2_path).state_dict())
        self.model_part3.load_state_dict(torch.load(model_part3_path).state_dict())
        self.G.load_state_dict(torch.load(G_path).state_dict())
        self.D.load_state_dict(torch.load(D_path).state_dict())
        self.list = torch.load(list_path)
        self.GAN_list = torch.load(GAN_list_path)

    def save_Latest_Epoch(self, epoch):
        model_part1_path = os.path.join(self.path["result_path"], "model_part1_" + str(epoch) + ".bin")
        model_part2_path = os.path.join(self.path["result_path"], "model_part2_" + str(epoch) + ".bin")
        model_part3_path = os.path.join(self.path["result_path"], "model_part3_" + str(epoch) + ".bin")
        G_path = os.path.join(self.path["result_path"], "Generator_" + str(epoch) + ".bin")
        D_path = os.path.join(self.path["result_path"], "Discriminator_" + str(epoch) + ".bin")
        list_path = os.path.join(self.path["result_path"], "list_" + str(epoch) + ".bin")
        GAN_list_path = os.path.join(self.path["result_path"], "GAN_list_" + str(epoch) + ".bin")
        torch.save(self.model_part1.state_dict(), model_part1_path)
        torch.save(self.model_part2.state_dict(), model_part2_path)
        torch.save(self.model_part3.state_dict(), model_part3_path)
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        torch.save(self.list, list_path)
        torch.save(self.GAN_list, GAN_list_path)

    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def get_privacy_key(self, bs):
        temp_tensor = torch.randn(self.dim, self.dim)
        rot_mat = torch.zeros(bs // self.dim, self.dim, self.dim)
        key_num = 0
        while key_num < bs // self.dim:
            nn.init.orthogonal_(temp_tensor)
            if torch.det(temp_tensor) > 0:
                rot_mat[key_num] = temp_tensor
                key_num = key_num + 1
        return rot_mat.to(self.device)

    def get_GAN_key(self, bs):
        # ---------------------------------GAN key ---------------------------------------------
        self.GAN_num += 1
        # --------------------------------------------------------------------------------------
        if self.dim == 2:
            theta = torch.from_numpy(np.random.choice(self.GAN_theta, bs // self.dim, replace=True))
            rot_mat = complex2rotmat(theta)
        elif self.dim == 3:
            theta = torch.from_numpy(np.random.choice(self.GAN_theta, bs // self.dim, replace=True))
            axis_z = torch.from_numpy(np.random.choice(self.GAN_theta, bs // self.dim, replace=True))
            axis_angle = torch.from_numpy(np.random.choice(self.GAN_theta, bs // self.dim, replace=True))
            o1 = torch.sin(axis_z) * torch.cos(axis_angle)
            o2 = torch.sin(axis_z) * torch.sin(axis_angle)
            o3 = torch.cos(axis_z)
            rot_mat = quarternion2rotmat(theta, o1, o2, o3)
        else: # dim == 5 or even higher
            rot_mat = self.get_privacy_key(bs)
        return rot_mat.to(self.device)

    def train_GAN(self, i, part1_output):
        batch_use = int(part1_output.size(0) // self.dim)
        D_loss, D_error, D_loss_pos, D_loss_neg = None, None, None, None

        if i % self.GAN_step == 0:
            rot_mat = self.get_GAN_key(batch_use * self.dim)
            G_output = self.G(part1_output.detach())
            G_output_detach = G_output.detach()

            if self.dim == 2 or self.dim == 3:
                neg_sample = encrypt_detach_by_rot_mat_nodiscard(G_output_detach, rot_mat)  # sample_size = batch_use
            else:
                neg_sample = encrypt_detach_by_rot_mat(G_output_detach, rot_mat)  # sample_size <= batch_use, since we discard samples with rotation angle < pi/4
            sample_size = neg_sample.size()[0]
            assert sample_size > 0 # in case that neg_sample has no elements
            pos_sample_list = []
            for d in range(self.dim):
                pos_sample_list.append(G_output_detach[d * batch_use: d * batch_use + sample_size, :, :, :])
            pos_sample = random.choice(pos_sample_list)

            D_neg = self.D(neg_sample)
            D_pos = self.D(pos_sample)
            D_error = get_D_error(D_pos, D_neg)
            D_loss_pos = torch.mean(D_pos)
            D_loss_neg = torch.mean(D_neg)
            D_loss = -(torch.mean(D_pos) - torch.mean(D_neg))
            if self.D.training:
                self.D_optimizer.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

            for p in self.D.parameters():
                p.data.clamp_(-1 * self.clip_threshold, self.clip_threshold)

        rot_mat = self.get_GAN_key(batch_use * self.dim)
        G_sample = self.G(part1_output)
        if self.dim == 2 or self.dim == 3:
            neg_sample = encrypt_detach_by_rot_mat_nodiscard(G_sample, rot_mat)
        else:
            neg_sample = encrypt_detach_by_rot_mat(G_sample, rot_mat)
        assert neg_sample.size()[0] > 0
        D_neg = self.D(neg_sample)
        G_loss = -torch.mean(D_neg)
        # if self.G.training:
        #     self.G_optimizer.zero_grad()
        #     G_loss.backward(retain_graph=True)
        #     self.G_optimizer.step()

        return G_loss, D_loss, D_error, D_loss_pos, D_loss_neg

    def train_Model(self):
        self.model_part1.train()
        self.model_part2.train()
        self.model_part3.train()
        self.G.train()
        self.D.train()
        self.model_part1.to(self.device)
        self.model_part2.to(self.device)
        self.model_part3.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)

        Loss, Error, G_Loss, D_Loss, D_Error, D_Pos_Loss, D_Neg_Loss = 0, 0, 0, 0, 0, 0, 0

        for i, (img, lbl) in enumerate(self.train_loader):
            img = img.to(self.device)
            lbl = lbl.to(self.device)
            batch_use = img.size(0)
            batch_use = batch_use // self.dim * self.dim
            if batch_use == 0:
                continue
            img = img[:batch_use]
            lbl = lbl[:batch_use]

            rot_mat = self.get_privacy_key(batch_use)

            part1_output = self.model_part1(img)
            G_loss, D_loss, D_error, D_loss_pos, D_loss_neg = self.train_GAN(i, part1_output)
            G_output = self.G(part1_output)

            encrypt_output = encrypt.apply(G_output, rot_mat)
            part2_output = self.model_part2(encrypt_output)
            decrypt_output = decrypt_figure.apply(part2_output, rot_mat)
            output = self.model_part3(decrypt_output)
            # predict = output.detach().max(1)[1]

            loss = self.loss_fn(output, lbl)
            loss += G_loss
            if not self.is_CelebA:
                predict = output.detach().max(1)[1]
                error = self.get_error(predict, lbl)
            else:
                error = get_binary_error(output.detach(), lbl, 0)

            self.optimizer.zero_grad()
            self.G_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.G_optimizer.step()

            Loss += loss.cpu().item()
            Error += error
            G_Loss += G_loss.cpu().item()
            if D_loss is not None:
                D_Loss += D_loss.cpu().item()
                D_Error += D_error
                D_Pos_Loss += D_loss_pos.cpu().item()
                D_Neg_Loss += D_loss_neg.cpu().item()
            # Print the result if you want to
            D_step = i // self.GAN_step + 1
            print("[train] batch: %d, loss: %.5f, error: %.5f, G_loss: %.5f, D_loss: %.5f, D_error: %.5f, D_loss_pos: %.5f, D_loss_neg: %.5f" %
                  (i, Loss / (i + 1), Error / (i + 1), G_Loss / (i + 1), D_Loss / D_step, D_Error / D_step,
                   D_Pos_Loss / D_step, D_Neg_Loss / D_step))
        D_step = i // self.GAN_step + 1
        Loss, Error, G_Loss = Loss / (i + 1), Error / (i + 1), G_Loss / (i + 1)
        D_Loss, D_Error, D_Pos_Loss, D_Neg_Loss = D_Loss / D_step, D_Error / D_step, D_Pos_Loss / D_step, D_Neg_Loss / D_step
        return Loss, Error, G_Loss, D_Loss, D_Error, D_Pos_Loss, D_Neg_Loss

    def val_Model(self):
        Loss, Error, G_Loss, D_Loss, D_Error, D_Pos_Loss, D_Neg_Loss = 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            self.model_part1.eval()
            self.model_part2.eval()
            self.model_part3.eval()
            self.G.eval()
            self.D.eval()
            self.model_part1.to(self.device)
            self.model_part2.to(self.device)
            self.model_part3.to(self.device)
            self.G.to(self.device)
            self.D.to(self.device)
            for i, (img, lbl) in enumerate(self.val_loader):
                img = img.to(self.device)
                lbl = lbl.to(self.device)
                batch_use = img.size(0)
                batch_use = batch_use // self.dim * self.dim
                if batch_use == 0:
                    continue
                img = img[:batch_use]
                lbl = lbl[:batch_use]

                rot_mat = self.get_privacy_key(batch_use)

                part1_output = self.model_part1(img)
                G_loss, D_loss, D_error, D_loss_pos, D_loss_neg = self.train_GAN(i, part1_output)
                G_output = self.G(part1_output)
                encrypt_output = encrypt.apply(G_output, rot_mat)
                part2_output = self.model_part2(encrypt_output)
                decrypt_output = decrypt_figure.apply(part2_output, rot_mat)
                output = self.model_part3(decrypt_output)
                # predict = output.detach().max(1)[1]

                loss = self.loss_fn(output, lbl)
                if not self.is_CelebA:
                    predict = output.detach().max(1)[1]
                    error = self.get_error(predict, lbl)
                else:
                    error = get_binary_error(output.detach(), lbl, 0)

                Loss += loss.cpu().item()
                Error += error
                G_Loss += G_loss.cpu().item()
                if D_loss is not None:
                    D_Loss += D_loss.cpu().item()
                    D_Error += D_error
                    D_Pos_Loss += D_loss_pos.cpu().item()
                    D_Neg_Loss += D_loss_neg.cpu().item()

                # Print the result if you want to
                D_step = i // self.GAN_step + 1
                print("[val] batch: %d, loss: %.5f, error: %.5f, G_loss: %.5f, D_loss: %.5f, D_error: %.5f, D_loss_pos: %.5f, D_loss_neg: %.5f" %
                      (i, Loss / (i + 1), Error / (i + 1), G_Loss / (i + 1), D_Loss / D_step, D_Error / D_step,
                       D_Pos_Loss / D_step, D_Neg_Loss / D_step))
        D_step = i // self.GAN_step + 1
        Loss, Error, G_Loss = Loss / (i + 1), Error / (i + 1), G_Loss / (i + 1)
        D_Loss, D_Error, D_Pos_Loss, D_Neg_Loss = D_Loss / D_step, D_Error / D_step, D_Pos_Loss / D_step, D_Neg_Loss / D_step
        return Loss, Error, G_Loss, D_Loss, D_Error, D_Pos_Loss, D_Neg_Loss

    def test_Model(self):
        """
        Test the model
        """
        pass

    def draw_Figure(self):
        x = np.arange(0, len(self.list[0]), 1)
        y0, y1, y2, y3 = np.array(self.list[0]), np.array(self.list[1]), np.array(self.list[2]), np.array(self.list[3])
        z0, z1, z2, z3, z4, z5, z6, z7, z8, z9 = np.array(self.GAN_list[0]), np.array(self.GAN_list[1]),\
        np.array(self.GAN_list[2]), np.array(self.GAN_list[3]), np.array(self.GAN_list[4]), np.array(self.GAN_list[5]),\
        np.array(self.GAN_list[6]), np.array(self.GAN_list[7]), np.array(self.GAN_list[8]), np.array(self.GAN_list[9])

        plt.figure()
        plt.subplot(331)
        plt.plot(x, y0, color="blue")
        plt.plot(x, y2, color="red")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.subplot(332)
        plt.plot(x, y1, color="blue")
        plt.plot(x, y3, color="red")
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.subplot(333)
        plt.plot(x, z0, color='blue')
        plt.plot(x, z5, color='red')
        plt.xlabel("epoch")
        plt.ylabel('G_loss')
        plt.subplot(334)
        plt.plot(x, z1, color='blue')
        plt.plot(x, z6, color='red')
        plt.xlabel("epoch")
        plt.ylabel('D_loss')
        plt.subplot(335)
        plt.plot(x, z2, color='blue')
        plt.plot(x, z7, color='red')
        plt.xlabel("epoch")
        plt.ylabel('D_error')
        plt.subplot(336)
        plt.plot(x, z3, color='blue')
        plt.plot(x, z8, color='red')
        plt.xlabel("epoch")
        plt.ylabel('D_pos_loss')
        plt.subplot(337)
        plt.plot(x, z4, color='blue')
        plt.plot(x, z9, color='red')
        plt.xlabel("epoch")
        plt.ylabel('D_neg_loss')
        plt.savefig(os.path.join(self.path["result_path"], "curve.jpg"))

    def adjust_lr(self, lr, G_lr, D_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in self.G_optimizer.param_groups:
            param_group["lr"] = G_lr
        for param_group in self.D_optimizer.param_groups:
            param_group["lr"] = D_lr

    def work(self):
        st, G_st, D_st = round(math.log(self.init_lr, 10)), round(math.log(self.G_lr, 10)), round(math.log(self.D_lr, 10))
        logs = torch.logspace(start=st, end=st-2, steps=self.epoch_num)
        G_logs = torch.logspace(start=G_st, end=G_st-2, steps=self.epoch_num)
        D_logs = torch.logspace(start=D_st, end=D_st-2, steps=self.epoch_num)
        for epoch in range(self.start_epoch, self.epoch_num):
            # self.adjust_lr(logs[epoch], G_logs[epoch], D_logs[epoch])
            self.adjust_lr(self.init_lr * self.milestones[epoch], self.G_lr * self.milestones[epoch], self.D_lr * self.milestones[epoch])
            # if epoch >= 150:
            #     self.adjust_lr(0.01 * self.init_lr, 0.01 * self.G_lr, 0.01 * self.D_lr)
            # elif epoch >= 100:
            #     self.adjust_lr(0.1 * self.init_lr, 0.1 * self.G_lr, 0.1 * self.D_lr)
            loss, error, G_loss, D_loss, D_error, D_pos_loss, D_neg_loss = self.train_Model()
            self.list[0].append(loss)
            self.list[1].append(error)
            self.GAN_list[0].append(G_loss)
            self.GAN_list[1].append(D_loss)
            self.GAN_list[2].append(D_error)
            self.GAN_list[3].append(D_pos_loss)
            self.GAN_list[4].append(D_neg_loss)
            loss, error, G_loss, D_loss, D_error, D_pos_loss, D_neg_loss = self.val_Model()
            self.list[2].append(loss)
            self.list[3].append(error)
            self.GAN_list[5].append(G_loss)
            self.GAN_list[6].append(D_loss)
            self.GAN_list[7].append(D_error)
            self.GAN_list[8].append(D_pos_loss)
            self.GAN_list[9].append(D_neg_loss)

            if epoch % 10 == 9 or epoch == self.epoch_num - 1:
                self.save_Latest_Epoch(epoch)
            self.draw_Figure()

        self.test_Model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="2", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--epoch-num", default=10, type=int)
    parser.add_argument("--num-classes", default=40, type=int)
    parser.add_argument("--num-layers", default=20, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    # parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--is-encrypt", default=True, type=bool)
    parser.add_argument("--double_filter", default=False, type=bool)
    parser.add_argument("--double_layer", default=False, type=bool)
    parser.add_argument("--encrypt-location", default="c", type=str)
    parser.add_argument("--milestone", default=[5, 7], type=list)
    parser.add_argument("--result-path", default="Alexnet_CelebA", type=str)
    parser.add_argument("--sub-result-path", default="net_privacy_GAN_stage", type=str)

    # Parameter of GAN
    parser.add_argument("--G-boardsize", default=13, type=int)
    parser.add_argument("--G-channel", default=384, type=int)
    parser.add_argument("--G-block", default=1, type=int)
    parser.add_argument("--G-kernel", default=3, type=int)
    parser.add_argument("--D-boardsize", default=13, type=int)
    parser.add_argument("--D-channel", default=384, type=int)
    parser.add_argument("--D-block", default=1, type=int)
    parser.add_argument("--D-kernel", default=3, type=int)
    parser.add_argument("--G-lr", default=1e-3, type=float)
    parser.add_argument("--D-lr", default=1e-5, type=float)
    parser.add_argument("--GAN-step", default=50, type=int)
    parser.add_argument("--clip-threshold", default=0.01, type=float)

    parser.add_argument("--train-id", default="5", type=str)

    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5]) # rqh 0514 add

    args = parser.parse_args()
    args.batch_size = 100 // (2*args.dim) * (2*args.dim) # set batch size close to 100
    args.result_path = "result_dim%d/" % args.dim + args.result_path

    if args.encrypt_location == "a":
        args.sub_result_path = args.sub_result_path + "/1"
    elif args.encrypt_location == "b":
        args.sub_result_path = args.sub_result_path + "/2"
    elif args.encrypt_location == "c":
        args.sub_result_path = args.sub_result_path + "/3"
    else:
        raise RuntimeError("Invalid encrypt location, please choose from a, b, and c.")
    if "CelebA" in args.result_path:
        args.is_CelebA = True
    else:
        args.is_CelebA = False

    seed_torch(args.epoch_num + args.num_layers + args.batch_size)
    args.GAN_theta = np.array([0.25*math.pi,0.75*math.pi,1.25*math.pi,1.75*math.pi])
    # ------------------ Milestones List ---------------------------------------------------------
    scalar = 1.
    stones = []
    last = 0
    for i in range(len(args.milestone)):
        stones.extend([scalar for _ in range(args.milestone[i] - last)])
        last = args.milestone[i]
        scalar *= 0.1
    stones.extend([scalar for _ in range(args.epoch_num - last)])
    args.milestone_list = stones
    # --------------------------------------------------------------------------------------------

    trainer = Privacy_Trainer(args)
    trainer.prepare_Model(Get_Model_AlexNet, optim.SGD, optim.RMSprop, nn.BCEWithLogitsLoss)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop((224, 224), 32), # actually no use.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # actually no use
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # actually no use
    ])
    trainer.prepare_Dataset(train_set=ourDataset, val_set=ourDataset,
                            train_subpath=os.path.join("data", "cropped_CelebA"),
                            val_subpath=os.path.join("data", "cropped_CelebA"),
                            train_shuffle=True, val_shuffle=False,
                            train_transform=train_transform, val_transform=val_transform
                            )
    trainer.work()


if __name__ == "__main__":
    main()
