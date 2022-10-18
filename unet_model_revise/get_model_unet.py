
import os
import torch
import torch.nn as nn

from unet_model_revise.unet_model import UNet

def Get_Model_Unet(num_input_channels,num_classes,conv_per_block,gpu_id,add_bn=False):
    model = UNet(num_input_channels,num_classes,conv_per_block,add_bn)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("###########Using GPU!!!##########")

    return model


