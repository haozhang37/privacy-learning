
import torch.nn as nn

from res_model_discriminator.model_base81 import Model81
from res_model_discriminator.resnet81 import GoResNet81

class Model_Resnet81(Model81):
    # Class for value network model.

    def __init__(self,board_size,num_input_channels,num_block,kernel_size):
        super().__init__()

        self.board_size = board_size
        self.num_input_channels = num_input_channels
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init_conv = self._conv_layer(self.num_input_channels,kernel=kernel_size)
        self.value_final_conv = self._conv_layer(224, 1, 1)

        d = self.board_size ** 2

        self.value_linear1 = nn.Linear(d, 256)
        self.value_linear2 = nn.Linear(256, 1)

        # Softmax as the final layer
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.resnet = GoResNet81(num_block,kernel_size)

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
            padding=(kernel // 2)
        ))
        layers.append(nn.BatchNorm2d(output_channel, momentum=0.1))

        if relu:
            layers.append(self.relu)

        return nn.Sequential(*layers)

    def forward(self, x):

        s = x

        s = self.init_conv(s)

        s = self.resnet(s)

        d = self.board_size ** 2

        V = self.value_final_conv(s)

        V = self.relu(self.value_linear1(V.view(-1, d)))

        V = self.value_linear2(V)

        return V
