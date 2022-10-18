import os
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
#     torch.save(samples, "./data/CelebA_theta.bin")


def generate_best_sample(args, feature, D):

    batch_use = feature.size(0) // args.dim
    rot_mat = gen_rot_mat(args.sample_num, args.dim)
    rot_mat = rot_mat.to(args.device)
    max_index = torch.Tensor(0).to(args.device)
    with torch.no_grad():
        for i in range(rot_mat.size(0)):
            rot_mat_use = rot_mat[i].expand(batch_use,args.dim,args.dim)
            decrypt_output = encrypt_detach_by_rot_mat_nodiscard(feature, rot_mat_use)
            D_output = D(decrypt_output)
            max_index = torch.cat((max_index, D_output), 1)
    max_index = max_index.max(1)[1]
    max_index_use = torch.zeros(batch_use, args.dim, args.dim, device=args.device)
    max_index_use = max_index_use.long()
    for i in range(max_index.size(0)):
        max_index_use[i, :, :] = torch.full([args.dim, args.dim], max_index[i])

    rot_mat_use = torch.gather(rot_mat, 0, max_index_use)
    sample_use = []
    for i in range(batch_use):
        sample_use.append(rot_mat[max_index[i]])
    best_sample = decrypt_figure.apply(feature, rot_mat_use)
    return best_sample, sample_use


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
        sub_folder = "net_privacy_GAN_stage"
        decoder_folder = args.is_unet + "decoder_privacy_GAN_noencrypt_stage"
    elif args.decoder_type == 4:
        sub_folder = "net_privacy_GAN_stage"
        decoder_folder = args.is_unet + "decoder_privacy_GAN_encrypt_stage"
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
        transforms.Normalize(mean=args.celeba_mean, std=args.celeba_std)
    ])
    dataset = ourDataset('list_attr_celeba.txt', training=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader


def generateImg(origin_image):
    origin_visual = np.array(origin_image)
    origin_visual = origin_visual + 127
    origin_visual = origin_visual.transpose((1, 2, 0))
    origin_visual = np.uint8(origin_visual)
    origin_img = Image.fromarray(origin_visual, 'RGB')
    return origin_img


def get_results(args):
    model_part1, model_part2, model_part3, G, D, decoder = get_models(args)
    model_part1_path, model_part2_path, model_part3_path, G_path, D_path, decoder_path, visual_path = get_paths(args)
    dataloader = get_dataset(args)
    model_part1.to(args.device).eval()
    model_part2.to(args.device).eval()
    model_part3.to(args.device).eval()
    G.to(args.device).eval()
    D.to(args.device).eval()
    decoder.to(args.device).eval()

    model_part1.load_state_dict(torch.load(model_part1_path, map_location=torch.device("cpu")))
    model_part2.load_state_dict(torch.load(model_part2_path, map_location=torch.device("cpu")))
    model_part3.load_state_dict(torch.load(model_part3_path, map_location=torch.device("cpu")))
    decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device("cpu")))

    if args.decoder_type > 1:
        G.load_state_dict(torch.load(G_path, map_location=torch.device("cpu")))
    if args.decoder_type > 2:
        D.load_state_dict(torch.load(D_path, map_location=torch.device("cpu")))
    if args.decoder_type >= 4:

        rot_mat = gen_rot_mat(args.sample_num_for_all_test_sample, args.dim) # generate enough matrices for all images in the test set
        rot_mat = rot_mat.to(args.device)
        best_sample_theta = [[], []]  # first: ground truth,  second: predict result
        mse_error = 0.
        mse_num = 0

    with torch.no_grad():
        for i, (img, lbl) in enumerate(dataloader):
            if i > args.visual_num and args.decoder_type != 5 and args.decoder_type != 4:
                break
            print(i)
            img_size = img.size()
            img = img.to(args.device)
            lbl = lbl.to(args.device)

            batch_use = img.size(0) // args.dim
            if batch_use == 0:
                break
            img = img[:batch_use * args.dim]
            lbl = lbl[:batch_use * args.dim]

            part1_output = model_part1(img)

            if args.decoder_type > 1:
                G_output = G(part1_output)

                if args.decoder_type >= 4:
                    rot_mat_use = rot_mat[i * batch_use: (i + 1) * batch_use]
                    encrypt_output = encrypt.apply(G_output, rot_mat_use)
                    if args.decoder_type == 4:
                        feature = encrypt_output
                    elif args.decoder_type == 5:
                        # sample_use = rot_mat[i * batch_use: (i + 1) * batch_use] # todo: delete this line? seems redundant
                        feature, best_rot_mat = generate_best_sample(args, encrypt_output, D) #todo: understand
                        best_sample_theta[0].extend(rot_mat_use) # rqh 0528 reuse the previous variable name
                        best_sample_theta[1].extend(best_rot_mat) # rqh 0528 rename
                        torch.save(best_sample_theta, visual_path + "../best_sample_%s.bin" % args.encrypt_position)
                else:
                    feature = G_output
            else:
                feature = part1_output

            upsample_method = nn.Upsample(size=img_size[-1], mode='bilinear')
            feature = upsample_method(feature)

            if i <= args.visual_num or args.decoder_type == 5 or args.decoder_type == 4:
                for channel_id in range(3):
                    img[:, channel_id, :, :] = img[:, channel_id, :, :] * args.celeba_std[channel_id]
                    img[:, channel_id, :, :] = img[:, channel_id, :, :] + args.celeba_mean[channel_id]
                origin_image = img * 255 - 127
                decoder_out = decoder(feature)
                if args.decoder_type >= 4 and img.size(0) == args.batch_size:
                    mse_error += torch.sqrt(torch.nn.functional.mse_loss(origin_image, decoder_out, reduction="mean") / 255 / 255)
                    mse_num += 1

                decoder_imgs, original_imgs = [], []
                for d in range(args.dim):
                    decoder_imgs.append(generateImg(decoder_out[d * batch_use].cpu()))
                    original_imgs.append(generateImg(origin_image[d * batch_use].cpu()))
                if i <= args.visual_num:
                    for d in range(args.dim):
                        decoder_imgs[d].save(visual_path + "decode_%d_d%d.jpg" % (i, d + 1))
                        original_imgs[d].save(visual_path + "origin_%d_d%d.jpg" % (i, d + 1))

    if args.decoder_type >= 4:
        f = open(visual_path + "../mse_error_%s.txt" % args.encrypt_position, "w")
        f.write(str(mse_error / mse_num))
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="1", type=str)
    parser.add_argument("--data-path", default="./data/cropped_CelebA", type=str)
    # parser.add_argument("--batch-size", default=25, type=int)
    parser.add_argument("--net-layer-num", default=44, type=int)
    parser.add_argument("--net-class-num", default=40, type=int)
    parser.add_argument("--sample-num", default=1000, type=int)
    parser.add_argument("--sample-num-for-all-test-sample", default=100000, type=int) # generate 20000 rotation matrices, since CelebA only has 40000 test images, 20000 is enough
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
    parser.add_argument("--result-path", default="Alexnet_CelebA", type=str)
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

    parser.add_argument("--train-id", default="2", type=str)
    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5]) # rqh 0514 add

    args = parser.parse_args()
    args.batch_size = 25 if args.dim == 5 else 24
    args.result_path = "result_dim%d/" % args.dim + args.result_path
    args.root = os.getcwd()
    args.is_unet = "unet_"
    args.is_encrypt = False if args.is_encrypt == 0 else True
    if args.device != "cpu":
        args.device = int(args.device)
    args.celeba_std = [0.5, 0.5, 0.5]
    args.celeba_mean = [0.5, 0.5, 0.5]
    seed_torch(args.num_block + args.net_layer_num + args.batch_size)
    # args.result_path = "./result/Alexnet_CelebA"  # result_path

    get_results(args)


if __name__ == "__main__":
    main()
