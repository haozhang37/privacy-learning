import os
import sys

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchvision import utils
import torch.nn.init as init

class BasicBlockDecoder(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,if_upsample=False,if_alexnet=0):
        super(BasicBlockDecoder, self).__init__()
        if if_upsample:
            if if_alexnet == 1:
                self.conv1 = nn.ConvTranspose2d(in_planes,planes, kernel_size=4, stride=stride, padding=0,output_padding=0,bias=False).cuda()
            else:
                self.conv1 = nn.ConvTranspose2d(in_planes,planes, kernel_size=4, stride=stride, padding=1,output_padding=0,bias=False).cuda()
            # the kernel of upconv should be 4.
        else:
            self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3, stride=stride, padding=1, bias=False).cuda()
        self.bn1 = nn.BatchNorm2d(planes).cuda()
        self.conv2 = nn.Conv2d(in_planes,in_planes, kernel_size=3, stride=1, padding=1, bias=False).cuda()
        self.bn2 = nn.BatchNorm2d(in_planes).cuda()
        if if_upsample:
            if if_alexnet == 1:
                self.conv3 = nn.ConvTranspose2d(in_planes,planes, kernel_size=4, stride=stride, padding=0,output_padding=0,bias=False).cuda()
            else:
                self.conv3 = nn.ConvTranspose2d(in_planes,planes, kernel_size=4, stride=stride, padding=1,output_padding=0,bias=False).cuda()
        else:
            self.conv3 = nn.Conv2d(in_planes,planes, kernel_size=3, stride=stride, padding=1, bias=False).cuda()
        self.bn3 = nn.BatchNorm2d(planes).cuda()

    def forward(self, x):
        residual = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        out += residual
        out = F.relu(out)
        # print(out.size())
        return out

# Decoder
class ResNetDecoder(nn.Module):
    def __init__(self,block,num_input_channels,num_block,num_block_upsample,planes=64,if_alexnet=0):
        super(ResNetDecoder, self).__init__()
        self.planes = planes
        self.inputconv = nn.Conv2d(num_input_channels, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.inputbn = nn.BatchNorm2d(self.planes)
        self.relu = nn.ReLU()
        self.resnet_decoder = self._get_resnet_decoder(block,num_block,num_block_upsample,if_alexnet)
        self.outputconv = nn.Conv2d(self.planes, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def _get_resnet_decoder(self,block,num_block,num_block_upsample,if_alexnet):
        strides = [2]*int(num_block_upsample) + [1] * int(num_block-num_block_upsample)
        if_upsample = [True]*int(num_block_upsample) + [False]*int(num_block-num_block_upsample)
        resnet_decoder = []
        for block_id in range(0,num_block):
            if if_alexnet == 1 and block_id == 0:
                resnet_decoder.append(block(self.planes, self.planes, strides[block_id], if_upsample[block_id],if_alexnet))
            else:
                resnet_decoder.append(block(self.planes, self.planes, strides[block_id], if_upsample[block_id]))
        return nn.Sequential(*resnet_decoder)

    def forward(self, x):
        out = self.inputconv(x)
        out = self.inputbn(out)
        out = self.relu(out)
        out = self.resnet_decoder(out)
        out = self.outputconv(out)
        return out


