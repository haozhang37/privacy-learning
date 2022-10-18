
import torch.nn as nn

from res_model_discriminator.model_base81 import Model81

class Block81(Model81):
    # Class for block. 20 blocks is included in resnet. One block has two convlution layers. It is used in Class GoresNet
    def __init__(self,kernel_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_lower = self._conv_layer(kernel=kernel_size)
        self.conv_upper = self._conv_layer(relu=False,kernel=kernel_size)

    def _conv_layer(
            self,
            input_channel=None,
            output_channel=None,
            kernel=3,
            relu=True,
            ):
        if input_channel is None:
            input_channel = 224
        if output_channel is None:
            output_channel = 224

        layers = []
        layers.append(nn.Conv2d(
            input_channel,
            output_channel,
            kernel,
            padding=(kernel // 2),
        ))
        layers.append(nn.BatchNorm2d(output_channel, momentum=0.1))
        if relu:
            layers.append(self.relu)

        return nn.Sequential(*layers)

    def forward(self, s):
        s1 = self.conv_lower(s)
        s1 = self.conv_upper(s1)
        s1 = s1 + s
        s = self.relu(s1)
        return s
