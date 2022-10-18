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


class Origin_Trainer:
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
        self.dim = args.dim

        self.path["result_path"] = os.path.join(self.path["root"], args.result_path)
        if not os.path.exists(self.path["result_path"]):
            os.mkdir(self.path["result_path"])
        self.path["result_path"] = os.path.join(self.path["result_path"], args.sub_result_path)
        if not os.path.exists(self.path["result_path"]):
            os.makedirs(self.path["result_path"])

    def prepare_Model(self, model, optimizer=None, loss_fn=None, scheduler=None):
        # Save the parameters to use in the future, these parameter can initialize in __init__ function as well

        # initialize the models
        self.model = model(num_classes=self.num_classes)
        self.list = [[] for _ in range(4)]
        # result list: [train loss, train error, val loss, val error, ]
        if optimizer is not None:
            self.optimizer = optimizer([{"params":self.model.parameters(),'lr':self.init_lr,'initial_lr':self.init_lr}],
                                         momentum=self.momentum, weight_decay=self.weight_decay)
        if loss_fn is not None:
            self.loss_fn = loss_fn(reduction="mean").to(self.device)
        # if scheduler is not None:
        #     self.scheduler = scheduler(optimizer=self.optimizer, factor=self.factor)

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

    def load_Latest_Epoch(self):
        model_path = os.path.join(self.path["result_path"], "model_" + str(self.start_epoch - 1) + ".bin")
        list_path = os.path.join(self.path["result_path"], "list_" + str(self.start_epoch - 1) + ".bin")
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")).state_dict())
        self.list = torch.load(list_path)

    def save_Latest_Epoch(self, epoch):
        model_path = os.path.join(self.path["result_path"], "model_" + str(epoch) + ".bin")
        list_path = os.path.join(self.path["result_path"], "list_" + str(epoch) + ".bin")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.list, list_path)

    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def train_Model(self):
        self.model.train()
        self.model.to(self.device)
        Loss = 0
        Error = 0
        for i, (img, lbl) in enumerate(self.train_loader):
            img = img.to(self.device)
            lbl = lbl.to(self.device)
            batch_use = img.size(0)
            batch_use = batch_use // self.dim * self.dim
            if batch_use < 1:
                break
            img = img[:batch_use]
            lbl = lbl[:batch_use]
            lbl = lbl[:, self.hide_attr]

            output = self.model(img)
            # predict = output.detach().max(1)[1]

            loss = self.loss_fn(output, lbl)

            error = get_binary_error(output.detach(), lbl, 0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            Loss += loss.cpu().item()
            Error += error
            # Print the result if you want to
            print("[train] batch: %d, loss: %.5f, error: %.5f" % (i, Loss / (i + 1), Error / (i + 1)))
        Loss, Error = Loss / (i + 1), Error / (i + 1)
        return Loss, Error

    def val_Model(self):
        Loss = 0
        BestLoss = 0
        Error = 0
        BestError = 0
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            for i, (img, lbl) in enumerate(self.val_loader):
                img = img.to(self.device)
                lbl = lbl.to(self.device)
                batch_use = img.size(0)
                batch_use = batch_use // self.dim * self.dim
                if batch_use < 1:
                    break
                img = img[:batch_use]
                lbl = lbl[:batch_use]
                lbl = lbl[:, self.hide_attr]

                output = self.model(img)

                loss = self.loss_fn(output, lbl)
                error = get_binary_error(output.detach(), lbl, 0)

                Loss += loss.cpu().item()
                Error += error

                # Print the result if you want to
                print("[val] batch: %d, loss: %.5f, error: %.5f" % (i, Loss / (i + 1), Error / (i + 1)))
                # if i>=397: #todo:check
                #     break
            Loss, Error = Loss / (i + 1), Error / (i + 1)

        return Loss, Error

    def test_Model(self):
        """
        Test the model
        """
        pass

    def draw_Figure(self):
        x = np.arange(0, len(self.list[0]), 1)
        y0 = np.array(self.list[0])
        y1 = np.array(self.list[1])
        y2 = np.array(self.list[2])
        y3 = np.array(self.list[3])

        plt.figure()
        plt.subplot(211)
        plt.plot(x, y0, color="blue")
        plt.plot(x, y2, color="red")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.subplot(212)
        plt.plot(x, y1, color="blue")
        plt.plot(x, y3, color="red")
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.savefig(os.path.join(self.path["result_path"], "curve.jpg"))

    def adjust_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def work(self):
        st = math.log(self.init_lr, 10)
        for epoch in range(self.start_epoch, self.epoch_num):
            self.adjust_lr(self.init_lr * self.milestones[epoch])
            loss, error = self.train_Model()
            self.list[0].append(loss)
            self.list[1].append(error)
            loss, error = self.val_Model()
            self.list[2].append(loss)
            self.list[3].append(error)

            if epoch % 10 == 9 or epoch == self.epoch_num - 1:
                self.save_Latest_Epoch(epoch)
            self.draw_Figure()

        self.test_Model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="2", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--epoch-num", default=10, type=int)
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("--num-layers", default=110, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    # parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--is-encrypt", default=False, type=bool)
    parser.add_argument("--double_filter", default=False, type=bool)
    parser.add_argument("--double_layer", default=False, type=bool)
    parser.add_argument("--encrypt-location", default="c", type=str)
    parser.add_argument("--milestone", default=[5, 7], type=list)
    parser.add_argument("--result-path", default="Alexnet_CelebA_infer", type=str)
    parser.add_argument("--sub-result-path", default="eval_best_feature", type=str)
    parser.add_argument("--train-id", default="5", type=str)
    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5])

    args = parser.parse_args()
    args.batch_size = 100 // (2*args.dim) * (2*args.dim) # set batch size close to 100
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

    trainer = Origin_Trainer(args)
    trainer.prepare_Model(feature_resnet50, optim.SGD, nn.BCEWithLogitsLoss)
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
