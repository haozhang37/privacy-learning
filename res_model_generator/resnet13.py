
import torch.nn as nn

from res_model_generator.model_base13 import Model13
from res_model_generator.block13 import Block13

class GoResNet13(Model13):
    # Class for resnet, which is used in Class Model_Valuenetwork.
    def __init__(self,num_block,kernel_size):
        super().__init__()
        self.blocks = []
        for _ in range(num_block):
            self.blocks.append(Block13(kernel_size))
        self.resnet = nn.Sequential(*self.blocks)

    def forward(self, s):
        return self.resnet(s)
