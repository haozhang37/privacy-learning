import os
root = os.getcwd()
import sys
sys.path.append(root)
#import scipy.io as scio
import argparse
import torch
import torchvision
from torch import nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib # rqh 0502 add
matplotlib.use('agg') # rqh 0502 add
import matplotlib.pyplot as plt
import random
import math

from tools.lib import *

from res_model.resnet import resnet56

class eval_baseline_1:
    def __init__(self,args):
        self.path = {}
        self.path["root"] = os.getcwd()
        self.num_epoch = args.num_epoch
        self.bs = args.batch_size
        self.init_lr = args.lr
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        if args.device == "cpu":
            self.device = args.device
        else:
            self.device = int(args.device)
        self.cifar100_mean = [0.507, 0.487, 0.441]
        self.cifar100_std = [0.267, 0.256, 0.276]
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cifar100_mean, std=self.cifar100_std)
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cifar100_mean, std=self.cifar100_std)
        ])
        self.train_id = args.train_id
        self.num_classes = args.num_classes
        self.train_shuffle = True
        self.val_shuffle = False
        self.load_epoch = args.load_epoch
        self.dim = args.dim

        self.path["data_path"] = os.path.join(self.path["root"], args.data_path)
        # self.path["noloss_decode"] = self.path["root"] + args.noloss_decode
        self.path["result_path"] = os.path.join(self.path["root"], args.result_path)
        if not os.path.exists(self.path["result_path"]):
            os.mkdir(self.path["result_path"])
        self.path["best_decode"] = os.path.join(self.path["root"], args.best_decode)

    def prepare_model(self,optimizer, loss_fn):
        self.model = resnet56(num_classes=self.num_classes)
        self.optimizer = optimizer([{"params":self.model.parameters(),'lr':self.init_lr,'initial_lr':self.init_lr}]
                                    ,weight_decay=self.weight_decay,momentum=self.momentum)
        self.loss_fn = loss_fn(reduction="mean").to(self.device)
        self.list = [[] for _ in range(8)] #result list [tran_loss, train_err, val_loss, val_err, val_loss_noloss, val_err_noloss, val_loss_best, val_err_best]

        if self.load_epoch!=0:
            self.load_latest_epoch()

    def prepare_dataset(self):
        self.train_set = datasets.CIFAR100(self.path["data_path"],train=True,transform=self.train_transform,download=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_set,batch_size=self.bs, shuffle=self.train_shuffle)

        self.val_set = datasets.CIFAR100(self.path["data_path"],train=False,transform=self.val_transform,download=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_set,batch_size=self.bs, shuffle=self.val_shuffle)



        self.val_set_best = datasets.ImageFolder(self.path["best_decode"] + "/test", transform=self.val_transform)
        self.val_loader_best = torch.utils.data.DataLoader(self.val_set_best, batch_size=self.bs,
                                                             shuffle=self.val_shuffle)

    def save_latest_epoch(self,epoch):
        model_path = self.path["result_path"]+ "resnet56_" + str(epoch)+".bin"
        list_path = self.path["result_path"] + "list_" + str(epoch)+".bin"
        torch.save(self.model.state_dict(),model_path)
        torch.save(self.list,list_path)

    def load_latest_epoch(self):
        model_path = self.path["result_path"] + "resnet56_" + str(self.load_epoch)+".bin"
        list_path = self.path["result_path"] + "list_" + str(self.load_epoch)+".bin"
        self.model.load_state_dict(torch.load(model_path))
        self.list = torch.load(list_path)

    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def train_model(self):
        self.model.train()
        self.model.to(self.device)
        train_loss, train_err = 0, 0

        for i,(img,lbl) in enumerate(self.train_loader):
            img = img.to(self.device)
            lbl = lbl.to(self.device)
            batch_use = img.size(0)
            batch_use = batch_use // self.dim * self.dim
            if batch_use == 0:
                continue
            img = img[:batch_use]
            lbl = lbl[:batch_use]

            model_output = self.model(img)

            loss = self.loss_fn(model_output, lbl)
            predict = model_output.detach().max(1)[1]
            error = self.get_error(predict,lbl)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.cpu().item()
            train_err += error

            print("[train] batch: %d, loss: %f, err: %f" %(i, train_loss/(i+1), train_err/(i+1)))
        return train_loss/(i+1), train_err/(i+1)

    def val_model(self):
        self.model.eval()
        self.model.to(self.device)
        val_loss, val_err, val_loss_noloss, val_err_noloss, val_loss_best, val_err_best = 0, 0, 0, 0, 0, 0

        with torch.no_grad():
            for i,(img,lbl) in enumerate(self.val_loader):
                img = img.to(self.device)
                lbl = lbl.to(self.device)
                batch_use = img.size(0)
                batch_use = batch_use // self.dim * self.dim
                if batch_use == 0:
                    continue
                img = img[:batch_use]
                lbl = lbl[:batch_use]

                model_output = self.model(img)

                loss = self.loss_fn(model_output, lbl)
                predict = model_output.detach().max(1)[1]
                error = self.get_error(predict,lbl)

                val_loss += loss.cpu().item()
                val_err += error
                print("[val] batch: %d, loss: %f, err: %f" % (i, val_loss / (i + 1), val_err / (i + 1)))
            val_loss = val_loss/(i+1)
            val_err = val_err/(i+1)



            for j,(img,lbl) in enumerate(self.val_loader_best):
                img = img.to(self.device)
                lbl = lbl.to(self.device)
                batch_use = img.size(0)
                batch_use = batch_use // self.dim * self.dim
                if batch_use == 0:
                    continue
                img = img[:batch_use]
                lbl = lbl[:batch_use]

                output = self.model(img)

                loss = self.loss_fn(output, lbl)
                predict = output.detach().max(1)[1]
                error = self.get_error(predict,lbl)

                val_loss_best += loss.cpu().item()
                val_err_best += error
                print("[val_best] batch: %d, loss: %f, err: %f" % (j, val_loss_best / (j + 1), val_err_best / (j + 1)))
            val_loss_best = val_loss_best / (j + 1)
            val_err_best = val_err_best / (j + 1)

            return val_loss,val_err,val_loss_noloss,val_err_noloss,val_loss_best,val_err_best

    def draw_fig(self):
        x = np.arange(0,len(self.list[0]),1)
        y1 = np.array(self.list[0])
        y2 = np.array(self.list[1])
        y3 = np.array(self.list[2])
        y4 = np.array(self.list[3])

        y5 = np.array(self.list[4])
        y6 = np.array(self.list[5])
        y7 = np.array(self.list[6])
        y8 = np.array(self.list[7])

        plt.figure(str(self.train_id))
        plt.subplot(211)
        plt.plot(x, y1, color='blue')
        plt.plot(x, y3, color='red')
        plt.plot(x, y5, color='green')
        plt.plot(x, y7, color='black')
        plt.title('loss')
        plt.subplot(212)
        plt.plot(x, y2, color='blue')
        plt.plot(x, y4, color='red')
        plt.plot(x, y6, color='green')
        plt.plot(x, y8, color='black')
        plt.title('error')
        plt.savefig(self.path["result_path"] + 'baseline_' + str(self.train_id) + ".png")

    def adjust_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def work(self):
        for epoch in range(self.load_epoch, self.num_epoch):
            if epoch>= 150:
                self.adjust_lr(0.01 * self.init_lr)
            elif epoch>=100:
                self.adjust_lr(0.1*self.init_lr)
            train_loss, train_err = self.train_model()
            self.list[0].append(train_loss)
            self.list[1].append(train_err)

            val_loss,val_err,val_loss_noloss,val_err_noloss,val_loss_best,val_err_best = self.val_model()
            self.list[2].append(val_loss)
            self.list[3].append(val_err)
            self.list[4].append(val_loss_noloss)
            self.list[5].append(val_err_noloss)
            self.list[6].append(val_loss_best)
            self.list[7].append(val_err_best)

            if epoch % 10 == 9 or epoch == self.num_epoch-1:
                self.save_latest_epoch(epoch)
            self.draw_fig()
        # self.val_model()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="1", type=str)
    # parser.add_argument("--batch-size", default = 100, type= int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num-epoch", default=200, type=int)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9,type=float)
    parser.add_argument("--train-id",default=1,type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--load-epoch", default=0, type=int)

    parser.add_argument("--result-path", default="resnet56_cifar100_infer/eval_baseline_1/", type=str)
    parser.add_argument("--data-path", default="data/cifar")
    parser.add_argument("--noloss-decode",default="data/infer_cifar100/noloss_decode/test")
    parser.add_argument("--best-decode", default="data/infer_cifar100/best_decode")
    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5])

    args = parser.parse_args()
    args.batch_size = 100 // (2 * args.dim) * (2 * args.dim)  # set batch size close to 100
    args.best_decode = args.best_decode + "_dim%d" % args.dim

    seed = int(args.num_classes + args.num_epoch)
    seed_torch(seed)
    args.result_path = "result_dim%d/" % args.dim + args.result_path
    baseline = eval_baseline_1(args)
    baseline.prepare_model(optimizer=torch.optim.SGD,loss_fn=torch.nn.CrossEntropyLoss)
    baseline.prepare_dataset()
    baseline.work()

if __name__ == '__main__':
    main()



