import sys

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# sys.path.append('tools')
from tools.lib import *

from encrypt_layer import encrypt
from decrypt_layer_figure import decrypt_figure
# from decrypt_layer_noise import decrypt_noise

#part1: lenet before the encryption
class LeNet_part1(nn.Module):
    def __init__(self):
        super(LeNet_part1, self).__init__()
        self.conv = nn.Conv2d(3,32,5,padding = 2)
        self.mpool = nn.MaxPool2d(3,stride = 2)
    def forward(self, inputs):
        x = inputs
        pad = (0,1,0,1)
        x = F.relu(self.conv(x))
        x = self.mpool(F.pad(x, pad, "replicate", 0))
        return x

#part2: lenet between encryption and decryption(including encryption)
class LeNet_part2(nn.Module):
    def __init__(self, isEncrypt, dim, num_classes):
        super(LeNet_part2, self).__init__()
        self.IsEncrypt = isEncrypt
        self.dim = dim
        self.conv1 = nn.Conv2d(32, 64, 5, padding = 2, bias = not isEncrypt)
        self.conv2 = nn.Conv2d(64, 128, 5, padding = 2, bias = not isEncrypt)
        self.conv3 = nn.Conv2d(128, 128, 4, bias = not isEncrypt)
        self.conv4 = nn.Conv2d(128, num_classes, 1, bias = not isEncrypt)
        self.apool = nn.AvgPool2d(3, stride = 2)

    def forward(self,inputs):
        x = inputs
        pad = (0,1,0,1)
        x = our_relu(self.conv1(x),self.IsEncrypt, self.dim)
        x = self.apool(F.pad(x, pad, "replicate", 0))
        x = our_relu(self.conv2(x),self.IsEncrypt, self.dim)
        x = self.apool(F.pad(x, pad, "replicate", 0))
        x = our_relu(self.conv3(x),self.IsEncrypt, self.dim)
        x = our_relu(self.conv4(x),self.IsEncrypt, self.dim)
        # x = x.view(-1, 10)
        return x

#part3: lenet behind decryption(including decryption)
class LeNet_part3(nn.Module):
    def __init__(self, num_classes):
        super(LeNet_part3, self).__init__()
        self.num_classes = num_classes

    def forward(self,inputs):
        x = inputs
        x = x.view(-1, self.num_classes)
        return x
        
