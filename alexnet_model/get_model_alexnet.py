import os
import torch
import torch.nn as nn

from alexnet_model.get_model_alexnet_a import Get_Model_AlexNet_a
from alexnet_model.get_model_alexnet_b import Get_Model_AlexNet_b
from alexnet_model.get_model_alexnet_c import Get_Model_AlexNet_c

def Get_Model_AlexNet(gpu_id, isEncrypt, dim, double_filter=False, double_layer=False, num_classes=40, num_layer=None, encrypt_location="a"):

    if encrypt_location == "a":
        alex_part1, alex_part2, alex_part3 = Get_Model_AlexNet_a(gpu_id, isEncrypt, dim, num_classes)
    elif encrypt_location == "b":
        alex_part1, alex_part2, alex_part3 = Get_Model_AlexNet_b(gpu_id, isEncrypt, dim, num_classes)
    elif encrypt_location == "c":
        alex_part1, alex_part2, alex_part3 = Get_Model_AlexNet_c(gpu_id, isEncrypt, dim, num_classes)
    else:
        raise RuntimeError("Invalid encrypt location. Please use a, b, or c. ")

    return [alex_part1,alex_part2,alex_part3]

# def Get_Model_State(board_size,num_input_channels,num_block,gpu_id):
#
#     model = get_model(board_size,num_input_channels,num_block,gpu_id)
#     model_state = model.state_dict()
#     key_list = list(model_state.keys())
#     value_list = list(model_state.values())
#
#     return key_list,value_list
