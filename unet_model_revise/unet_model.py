# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn

from unet_model_revise.unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,conv_per_block,add_bn=False):
        super(UNet, self).__init__()
        self.add_bn = add_bn
        if add_bn:
            self.bn = nn.BatchNorm2d(n_channels)
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64,128,conv_per_block)
        self.down2 = down(128,256,conv_per_block)
        self.down3 = down(256,512,conv_per_block)
        self.down4 = down(512,512,conv_per_block)
        self.up1 = up(1024,256,conv_per_block)
        self.up2 = up(512,128,conv_per_block)
        self.up3 = up(256,64,conv_per_block)
        self.up4 = up(128,64,conv_per_block)
        self.outc = outconv(64,n_classes)

    def forward(self, x):
        x0 = self.bn(x) if self.add_bn else x
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
