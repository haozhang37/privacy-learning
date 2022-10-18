import os
root = os.getcwd()
import sys
sys.path.append(root)
import torch
import numpy as np
import argparse
import matplotlib # rqh 0502 add
matplotlib.use('agg') # rqh 0502 add
from matplotlib import pyplot as plt

from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torch import optim
from torch import nn
from PIL import Image
from tools.lib import *
from dataset import ourDataset
from dataset import ourFeatureset

from res_model.get_model_resnet import Get_Model_ResNet
from alexnet_model.feature_resnet import feature_resnet50
from alexnet_model.get_model_alexnet import Get_Model_AlexNet
from lenet_model.get_model_lenet import Get_Model_Lenet


class KNN:
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
        self.class_attr = args.class_attr
        self.hide_attr = args.hide_attr
        self.k_nearest = args.k_nearest

        self.path["result_path"] = os.path.join(self.path["root"], args.result_path)
        if not os.path.exists(self.path["result_path"]):
            os.mkdir(self.path["result_path"])
        self.path["result_path"] = os.path.join(self.path["result_path"], args.sub_result_path)
        if not os.path.exists(self.path["result_path"]):
            os.makedirs(self.path["result_path"])

    def prepare_Dataset(self, train_set=None, val_set=None, test_set=None, **kwargs):
        # Save the parameters to use in the future, these parameter can initialize in __init__ function as well
        # Instantiate the dataset and dataloader
        if train_set is not None:
            self.path["train_path"] = os.path.join(self.path["root"], kwargs["train_subpath"])
            self.train_transform = kwargs["train_transform"]
            self.train_shuffle = kwargs["train_shuffle"]
            if train_set == datasets.CIFAR10 or train_set == datasets.CIFAR100:
                self.train_set = train_set(self.path["train_path"], transform=self.train_transform, train=True, download=True)
            elif train_set == datasets.ImageFolder:
                self.train_set = train_set(self.path["train_path"], transform=self.train_transform)
            elif train_set == ourDataset or train_set == ourFeatureset:
                self.train_set = train_set('list_attr_celeba.txt', dst_path=kwargs["train_subpath"], training=True)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.bs,
                                                            shuffle=self.train_shuffle)

        if val_set is not None:
            self.path["val_path"] = os.path.join(self.path["root"], kwargs["val_subpath"])
            self.val_transform = kwargs["val_transform"]
            self.val_shuffle = kwargs["val_shuffle"]
            if val_set == datasets.CIFAR10 or val_set == datasets.CIFAR100:
                self.val_set = val_set(self.path["val_path"], transform=self.val_transform, train=False, download=True)
            elif val_set == datasets.ImageFolder:
                self.val_set = val_set(self.path["val_path"], transform=self.val_transform)
            elif val_set == ourDataset or train_set == ourFeatureset:
                # self.val_set = val_set('list_attr_celeba.txt', training=False)
                self.val_set = val_set('list_attr_celeba.txt', dst_path=kwargs["val_subpath"], training=False)
            self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.bs,
                                                          shuffle=self.val_shuffle, drop_last=True)

        if test_set is not None:
            self.path["test_path"] = os.path.join(self.path["root"], kwargs["test_subpath"])
            self.test_transform = kwargs["test_transform"]
            self.test_shuffle = kwargs["test_shuffle"]
            self.test_set = test_set(self.path["test_path"], transform=self.test_transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.bs, shuffle=self.test_shuffle)

    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def train_Model(self):
        error = 0
        num = 0

        for i, (image, CelebA_label) in enumerate(self.val_loader):
            val_feature = image.to(self.device)
            batch_use = image.size(0)
            nearest = torch.Tensor([[99999 for k in range(batch_use)] for j in range(self.k_nearest)]).to(self.device)
            near_label = torch.Tensor([[-1 for k in range(batch_use)] for j in range(self.k_nearest)]).to(self.device)
            for j in range(len(self.train_set)):
                print(j)
                train_fn = self.train_set.imgs[j][:6] + '.bin'
                train_feature = torch.load('./data/CelebA_best_feature5/train/' + train_fn).to(self.device)

                batch_distance = (val_feature - train_feature) * (val_feature - train_feature)
                batch_distance = torch.sum(batch_distance, dim=(3, 2, 1))
                batch_distance = batch_distance.sqrt()
                batch_label = torch.FloatTensor([j for k in range(batch_use)]).to(self.device)

                for k in range(self.k_nearest):
                    tmp_dist = torch.min(nearest[k], batch_distance)
                    tmp_lbl = batch_label.data
                    lbl1 = (nearest[k] == tmp_dist).float()
                    lbl2 = 1 - lbl1
                    batch_label = batch_label * lbl1 + near_label[k] * lbl2
                    near_label[k] = near_label[k] * lbl1 + tmp_lbl * lbl2
                    batch_distance = torch.max(nearest[k], batch_distance)
                    nearest[k].copy_(tmp_dist)

            for j in range(batch_use):
                class_label = torch.Tensor([0 for i in range(10)]).to(self.device)
                for k in range(self.k_nearest):
                    train_label = torch.Tensor(self.train_set.lbls[int(near_label[k][j].item())]).to(self.device)
                    class_label = class_label + train_label[self.hide_attr]

                truth_label = CelebA_label[j][self.hide_attr].to(self.device)
                er = get_binary_error(class_label, truth_label, self.k_nearest / 2)
                error += er
                num += 1
            print("Batch: %d, avg_error: %.5f" % (i, error / num))
            # if i>=8: #todo: check drop_last=True for val set
            #     break
        return error / num

    def work(self):
        st = math.log(self.init_lr, 10)

        error = self.train_Model()
        f = open(self.path["result_path"] + '/result_%d.txt' % self.k_nearest, 'w')
        f.write("error: %.5f" % (error))
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="3", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--epoch-num", default=10, type=int)
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("--num-layers", default=110, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--batch-size", default=4000, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--is-encrypt", default=False, type=bool)
    parser.add_argument("--double_filter", default=False, type=bool)
    parser.add_argument("--double_layer", default=False, type=bool)
    parser.add_argument("--encrypt-location", default="c", type=str)
    parser.add_argument("--milestone", default=[5, 7], type=list)
    parser.add_argument("--result-path", default="Alexnet_CelebA_infer", type=str)
    parser.add_argument("--sub-result-path", default="knn_best", type=str)
    parser.add_argument("--k-nearest", default=1, type=int)
    parser.add_argument("--normal-id", default="5", type=str)
    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5])

    args = parser.parse_args()
    args.result_path = "result_dim%d/" % args.dim + args.result_path
    args.sub_result_path = args.sub_result_path
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

    args.class_attr = [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 26,
                       27, 28, 29, 30, 31, 32, 33, 34, 35, 39]
    args.hide_attr = [2, 3, 6, 15, 19, 23, 25, 36, 37, 38]
    seed = int(args.num_classes + args.epoch_num)
    seed_torch(seed)
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

    trainer = KNN(args)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop((224, 224), 32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    trainer.prepare_Dataset(train_set=ourFeatureset, val_set=ourFeatureset,
                            train_subpath="/data/CelebA_best_feature_dim%d" % args.dim,
                            val_subpath="/data/CelebA_best_feature_dim%d" % args.dim,
                            train_shuffle=True, val_shuffle=False,
                            train_transform=train_transform, val_transform=val_transform
                            )
    trainer.work()


if __name__ == "__main__":
    main()
