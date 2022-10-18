
import torch.nn as nn

from res_model_discriminator.model_base81 import Model81
from res_model_discriminator.block81 import Block81

class GoResNet81(Model81):
    # Class for resnet, which is used in Class Model_Valuenetwork.
    def __init__(self,num_block,kernel_size):
        super().__init__()
        self.blocks = []
        for _ in range(num_block):
            self.blocks.append(Block81(kernel_size))
        self.resnet = nn.Sequential(*self.blocks)

    def forward(self, s):
        return self.resnet(s)
