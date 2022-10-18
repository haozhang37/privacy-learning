
import os
import torch
import torch.nn as nn

from resnet_decoder.resnet_decoder import BasicBlockDecoder
from resnet_decoder.resnet_decoder import ResNetDecoder

def get_model_resnet_decoder(gpu_id,num_input_channels,num_block,num_block_upsample,if_alexnet=0):

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    model = ResNetDecoder(BasicBlockDecoder,num_input_channels,num_block,num_block_upsample,planes=64,if_alexnet=if_alexnet)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("###########Using GPU!!!##########")

    return model
