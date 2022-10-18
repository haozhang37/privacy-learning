import os
import torch
import torch.nn as nn

from alexnet_model.alexnet_b import alexnet_part1
from alexnet_model.alexnet_b import alexnet_part2
from alexnet_model.alexnet_b import alexnet_part3

def Get_Model_AlexNet_b(gpu_id, isEncrypt, dim, num_classes=40):
    alex_part1 = alexnet_part1()
    alex_part2 = alexnet_part2(isEncrypt, dim)
    alex_part3 = alexnet_part3(num_classes)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("###########Using GPU!!!##########")

    return [alex_part1,alex_part2,alex_part3]

def Get_Model_State(board_size,num_input_channels,num_block,gpu_id):

    model = get_model(board_size,num_input_channels,num_block,gpu_id)
    model_state = model.state_dict()
    key_list = list(model_state.keys())
    value_list = list(model_state.values())

    return key_list,value_list
