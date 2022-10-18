import sys

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np
from alexnet_model.our_pool import our_mpool

from tools.lib import *

class alexnet_part1(nn.Module):
    def __init__(self):
        super(alexnet_part1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        return x
    def copy_params_from_alexnet(self, alexnet):
        features = [
            self.conv1, self.relu1,
            self.pool1,
            self.conv2, self.relu2
        ]
        for l1, l2 in zip(alexnet.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

class alexnet_part2(nn.Module):
    def __init__(self, isEncrypt, dim):
        super(alexnet_part2, self).__init__()
        self.IsEncrypt = isEncrypt
        self.dim = dim
        self.pool2 = our_mpool(3, 2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=not isEncrypt)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=not isEncrypt)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=not isEncrypt)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = our_mpool(3, 2)

    def forward(self, x):
        x = self.pool2(x, self.IsEncrypt, self.dim)
        x = our_relu(self.conv3(x), self.IsEncrypt, self.dim)
        x = our_relu(self.conv4(x), self.IsEncrypt, self.dim)
        x = our_relu(self.conv5(x), self.IsEncrypt, self.dim)
        x = self.pool3(x, self.IsEncrypt, self.dim)
        return x

    def copy_params_from_alexnet(self, alexnet, isEncrypt):
        features = [
            self.pool2,
            self.conv3, self.relu3,
            self.conv4, self.relu4,
            self.conv5, self.relu5,
            self.pool3
        ]
        for l1, l2 in zip(alexnet.features[5:], features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                l2.weight.data = l1.weight.data
                if not isEncrypt:
                    assert l1.bias.size() == l2.bias.size()
                    l2.bias.data = l1.bias.data

class alexnet_part3(nn.Module):
    def __init__(self, num_classes=40):
        super(alexnet_part3, self).__init__()
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        batchSize = x.size()[0]
        x = x.view(batchSize, -1)
        x = self.dropout1(x)
        x = self.relu1(self.linear1(x))
        x = self.dropout2(x)
        x = self.relu2(self.linear2(x))
        x = self.linear3(x)
        return x

    def copy_params_from_alexnet(self, alexnet):
        features = [
            self.dropout1,
            self.linear1, self.relu1,
            self.dropout2,
            self.linear2, self.relu2,
            self.linear3
        ]
        for l1, l2 in zip(alexnet.classifier, features):
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
                assert l1.weight.size() == l2.weight.size()
                l2.weight.data = l1.weight.data
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data = l1.bias.data

