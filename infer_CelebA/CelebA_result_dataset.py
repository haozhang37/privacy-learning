import os
root = os.getcwd()
import sys
sys.path.append(root)
import argparse
import torch
import torchvision
import numpy as np
import math
import torchvision.transforms as transforms

from PIL import Image as Image
from res_model.get_model_resnet import Get_Model_ResNet
from res_model_generator.get_model_generator import Get_Model_Generator
from res_model_discriminator.get_model_discriminator import Get_Model_Discriminator
from resnet_decoder.get_model_resnet_decoder import get_model_resnet_decoder
from encrypt_layer import encrypt
from decrypt_layer_figure import decrypt_figure
from tools.lib import *
from tools.Sphere_Sampling import Sphere_Sampling
from dataset import ourDataset
from alexnet_model.get_model_alexnet import Get_Model_AlexNet
from unet_model_revise.get_model_unet import Get_Model_Unet


# def generate_key(n):
#     samples = Sphere_Sampling(n)
#     for i in range(len(samples)):
#         for j in range(len(samples[i])):
#             samples[i][j] = samples[i][j] * 2 * math.pi / 360
#     torch.save(samples, "./data/CelebA_theta_large.bin")


def generate_best_sample(args, feature, D):
    batch_use = feature.size(0) // args.dim
    rot_mat = gen_rot_mat(args.sample_num, args.dim)
    rot_mat = rot_mat.to(args.device)
    max_index = torch.Tensor(0).to(args.device)
    with torch.no_grad():
        for i in range(rot_mat.size(0)):
            rot_mat_use = rot_mat[i].expand(batch_use, args.dim, args.dim)
            decrypt_output = encrypt_detach_by_rot_mat_nodiscard(feature, rot_mat_use)
            D_output = D(decrypt_output)
            max_index = torch.cat((max_index, D_output), 1)
    max_index = max_index.max(1)[1]
    max_index_use = torch.zeros(batch_use, args.dim, args.dim, device=args.device)
    max_index_use = max_index_use.long()
    for i in range(max_index.size(0)):
        max_index_use[i, :, :] = torch.full([args.dim, args.dim], max_index[i])

    rot_mat_use = torch.gather(rot_mat, 0, max_index_use)
    best_rot_mat = []
    for i in range(batch_use):
        best_rot_mat.append(rot_mat[max_index[i]])
    best_sample = decrypt_figure.apply(feature,rot_mat_use)
    return best_sample, best_rot_mat


def get_models(args):
    model_part1, model_part2, model_part3 = Get_Model_AlexNet(args.device, args.is_encrypt, args.dim, args.double_filter,
                                                             args.double_layer, args.net_class_num,
                                                             args.net_layer_num, args.encrypt_position)
    G = Get_Model_Generator(board_size=args.G_boardsize, num_input_channels=args.G_channel,
                            num_block=args.G_block, gpu_id=args.device, kernel_size=args.G_kernel)
    D = Get_Model_Discriminator(board_size=args.D_boardsize, num_input_channels=args.D_channel,
                                num_block=args.D_block, gpu_id=args.device, kernel_size=args.D_kernel)
    decoder = Get_Model_Unet(gpu_id=args.device, num_input_channels=args.num_input_channels,
                                      num_classes=3, conv_per_block=args.conv_per_block)
    return model_part1, model_part2, model_part3, G, D, decoder


def get_paths(args):
    if args.decoder_type == 1:
        sub_folder = "net_origin"
        decoder_folder = "decoder_origin"
    elif args.decoder_type == 2:
        sub_folder = "net_addlayer"
        decoder_folder = "decoder_addlayer"
    elif args.decoder_type == 3 or args.decoder_type == 5:
        sub_folder = "net_normal3"
        decoder_folder = args.is_unet + "decoder_normal_nonencrypt"
    elif args.decoder_type == 4:
        sub_folder = "net_privacy_GAN"
        decoder_folder = args.is_unet + "decoder_normal"
    else:
        raise RuntimeError("Invalid decoder type.")
    train_id = str(ord(args.encrypt_position) - ord("a") + 1)

    if args.decoder_type < 5:
        visual_path = "%s/visual_result/%s/%s/" % (args.result_path, decoder_folder, train_id)
    else:
        visual_path = "%s/visual_result/%s/%s/" % (args.result_path, "best_sample" + args.train_id, train_id)
    model_part1_path = "%s/%s/%s/model_part1_%d.bin" % (args.result_path, sub_folder, train_id, args.load_net_epoch)
    model_part2_path = "%s/%s/%s/model_part2_%d.bin" % (args.result_path, sub_folder, train_id, args.load_net_epoch)
    model_part3_path = "%s/%s/%s/model_part3_%d.bin" % (args.result_path, sub_folder, train_id, args.load_net_epoch)
    G_path = "%s/%s/%s/Generator_%d.bin" % (args.result_path, sub_folder, train_id, args.load_net_epoch)
    D_path = "%s/%s/%s/Discriminator_%d.bin" % (args.result_path, sub_folder, train_id, args.load_net_epoch)

    decoder_path = "%s/%s/%s/decoder%d.bin" % (args.result_path, decoder_folder, train_id, args.load_decoder_epoch)

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    return model_part1_path, model_part2_path, model_part3_path, G_path, D_path, decoder_path, visual_path


def get_dataset(args):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.cifar10_mean, std=args.cifar10_std)
    ])
    train_dataset = ourDataset('list_attr_celeba.txt', training=True)
    val_dataset = ourDataset('list_attr_celeba.txt', training=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def generateImg(origin_image):
    origin_visual = np.array(origin_image)
    origin_visual = origin_visual + 127
    origin_visual = origin_visual.transpose((1, 2, 0))
    origin_visual = np.uint8(origin_visual)
    origin_img = Image.fromarray(origin_visual, 'RGB')
    return origin_img


def get_results(args):
    train_set_len = 162770
    model_part1, model_part2, model_part3, G, D, decoder = get_models(args)
    model_part1_path, model_part2_path, model_part3_path, G_path, D_path, decoder_path, visual_path = get_paths(args)
    train_dataloader, val_dataloader = get_dataset(args)
    model_part1.to(args.device).eval()
    model_part2.to(args.device).eval()
    model_part3.to(args.device).eval()
    G.to(args.device).eval()
    D.to(args.device).eval()
    decoder.to(args.device).eval()
    img_dataset = "./data/CelebA_best_img_dim%d/" % args.dim  # discard train_id here
    fea_dataset = "./data/CelebA_best_feature_dim%d/" % args.dim  # discard train_id here

    model_part1.load_state_dict(torch.load(model_part1_path, map_location=torch.device("cpu")))
    model_part2.load_state_dict(torch.load(model_part2_path, map_location=torch.device("cpu")))
    model_part3.load_state_dict(torch.load(model_part3_path, map_location=torch.device("cpu")))
    decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device("cpu")))
    if args.decoder_type < 5:
        raise RuntimeError("This program only works for decoder type 5!")

    G.load_state_dict(torch.load(G_path, map_location=torch.device("cpu")))
    D.load_state_dict(torch.load(D_path, map_location=torch.device("cpu")))

    rot_mat = gen_rot_mat(args.sample_num_for_all_train_sample, args.dim)
    rot_mat = rot_mat.to(args.device)
    best_sample_theta = [[], []]  # first: ground truth,  second: predict result
    mse_error = 0.
    mse_num = 0

    best_train_img = img_dataset + "train/"
    best_test_img = img_dataset + "test/"
    feat_train = fea_dataset + "train/"
    feat_test = fea_dataset + "test/"
    if not os.path.exists(best_train_img):
        os.makedirs(best_train_img)
    if not os.path.exists(best_test_img):
        os.makedirs(best_test_img)
    if not os.path.exists(feat_train):
        os.makedirs(feat_train)
    if not os.path.exists(feat_test):
        os.makedirs(feat_test)

    with torch.no_grad():
        for i, (img, lbl) in enumerate(train_dataloader):

            print("train:%d" % i)
            img_size = img.size()
            img = img.to(args.device)
            lbl = lbl.to(args.device)

            batch_use = img.size(0) // args.dim
            if batch_use < 1:
                break

            img = img[:batch_use * args.dim]
            lbl = lbl[:batch_use * args.dim]

            part1_output = model_part1(img)
            G_output = G(part1_output)

            rot_mat_use = rot_mat[i * batch_use: (i + 1) * batch_use]
            encrypt_output = encrypt.apply(G_output, rot_mat_use)
            # sample_use = rot_mat[i * batch_use: (i + 1) * batch_use] #todo:redundant
            feature, best_rot_mat = generate_best_sample(args, encrypt_output, D)
            best_sample_theta[0].extend(rot_mat_use)
            best_sample_theta[1].extend(best_rot_mat)

            upsample_method = nn.Upsample(size=img_size[-1], mode='bilinear')
            up_feature = upsample_method(feature)

            decoder_out = decoder(up_feature)

            for j in range(batch_use * args.dim):
                number = i * args.batch_size + j + 1
                feat_name = feat_train + str(number).zfill(6) + ".bin"
                img_name = best_train_img + str(number).zfill(6) + ".jpg"
                torch.save(feature[j].data.cpu(), feat_name)
                img = generateImg(decoder_out[j].cpu())
                img.save(img_name)
        # -------------------------------------------------------------------------
        for i, (img, lbl) in enumerate(val_dataloader):

            print("val:%d" % i)
            img_size = img.size()
            img = img.to(args.device)
            lbl = lbl.to(args.device)

            batch_use = img.size(0) // args.dim
            if batch_use < 1:
                break

            img = img[:batch_use * args.dim]
            lbl = lbl[:batch_use * args.dim]

            part1_output = model_part1(img)
            G_output = G(part1_output)


            rot_mat_use = rot_mat[i * batch_use: (i + 1) * batch_use]
            encrypt_output = encrypt.apply(G_output, rot_mat_use)
            # sample_use = rot_mat[i * batch_use: (i + 1) * batch_use] #todo:redundant
            feature, best_rot_mat = generate_best_sample(args, encrypt_output, D)
            best_sample_theta[0].extend(rot_mat_use)
            best_sample_theta[1].extend(best_rot_mat)

            upsample_method = nn.Upsample(size=img_size[-1], mode='bilinear')
            up_feature = upsample_method(feature)

            decoder_out = decoder(up_feature)

            for j in range(batch_use * args.dim):
                number = i * args.batch_size + j + 1 + train_set_len
                feat_name = feat_test + str(number).zfill(6) + ".bin"
                img_name = best_test_img + str(number).zfill(6) + ".jpg"
                torch.save(feature[j].data.cpu(), feat_name)
                img = generateImg(decoder_out[j].cpu())
                img.save(img_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="1", type=str)
    parser.add_argument("--data-path", default="./data/cropped_CelebA", type=str)
    # parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument("--net-layer-num", default=44, type=int)
    parser.add_argument("--net-class-num", default=30, type=int)
    parser.add_argument("--sample-num", default=1000, type=int)
    parser.add_argument("--sample-num-for-all-train-sample", default=250000, type=int)# generate 250000 rotation matrices, since CelebA only has 160000 train images, 250000 is enough
    parser.add_argument("--visual-num", default=30, type=int)
    parser.add_argument("--encrypt-position", default="c", type=str)
    parser.add_argument("--if-alexnet", default=0, type=int)
    parser.add_argument("--num-input-channels", default=384, type=int)
    parser.add_argument("--num-block", default=6, type=int)
    parser.add_argument("--num-block-upsample", default=3, type=int)
    parser.add_argument("--is-encrypt", default=1, type=int)
    parser.add_argument("--is-noise", default=False, type=bool)
    parser.add_argument("--double-filter", default=False, type=bool)
    parser.add_argument("--double-layer", default=False, type=bool)
    parser.add_argument("--result-path", default="Alexnet_CelebA_infer", type=str)
    parser.add_argument("--decoder-type", default=5, type=int, choices=[1, 2, 3, 4, 5],
                        help="1: origin, 2: add_layer, 3: privacy-noencrypt, 4: privacy-encrypt, 5: best_sample")
    # parameters for decoder
    parser.add_argument("--decoder-epoch", default=30, type=int)
    parser.add_argument("--decoder-lr", default=1e-4, type=float)
    parser.add_argument("--conv-per-block", default=6, type=int)
    parser.add_argument("--load-net-epoch", default=9, type=int)
    parser.add_argument("--load-decoder-epoch", default=4, type=int)
    # parameter for GAN
    parser.add_argument("--D-boardsize", default=13, type=int)
    parser.add_argument("--D-channel", default=384, type=int)
    parser.add_argument("--D-block", default=1, type=int)
    parser.add_argument("--D-kernel", default=3, type=int)
    parser.add_argument("--G-boardsize", default=13, type=int)
    parser.add_argument("--G-channel", default=384, type=int)
    parser.add_argument("--G-block", default=1, type=int)
    parser.add_argument("--G-kernel", default=3, type=int)

    parser.add_argument("--train-id", default="5", type=str)
    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5])

    args = parser.parse_args()
    args.batch_size = 50 if args.dim == 5 else 36
    seed_torch(args.num_block + args.net_layer_num + args.batch_size)
    args.result_path = "result_dim%d/" % args.dim + args.result_path
    args.root = os.getcwd()
    args.is_unet = "unet_"
    args.is_encrypt = False if args.is_encrypt == 0 else True
    if args.device != "cpu":
        args.device = int(args.device)
    args.cifar10_std = [0.5, 0.5, 0.5]
    args.cifar10_mean = [0.5, 0.5, 0.5]

    get_results(args)


# def fix_val_set():
#     our_val_num = 39828
#     ori_train_num = 162770
#     pth1 = "./data/CelebA_best_img/test/"
#     pth2 = "./data/CelebA_feature/test/"
#     for i in range(our_val_num):
#         print(i)
#         num = i + 1
#         img_name1 = str(num).zfill(6) + ".jpg"
#         img_name2 = str(num + ori_train_num).zfill(6) + ".jpg"
#         fea_name1 = str(num).zfill(6) + ".bin"
#         fea_name2 = str(num + ori_train_num).zfill(6) + ".bin"
#         os.system("mv %s %s" % (pth1 + img_name1, pth1 + img_name2))
#         os.system("mv %s %s" % (pth2 + fea_name1, pth2 + fea_name2))

if __name__ == "__main__":
    main()

