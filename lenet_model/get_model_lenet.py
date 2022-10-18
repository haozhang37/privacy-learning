
import os
import torch
import torch.nn as nn

from lenet_model.lenet_model import LeNet_part1
from lenet_model.lenet_model import LeNet_part2
from lenet_model.lenet_model import LeNet_part3

def Get_Model_Lenet(gpu_id, isEncrypt, dim, double_filter=False, double_layer=False, num_classes=40, num_layer=None, encrypt_location="a"):
    lenet_part1 = LeNet_part1()
    lenet_part2 = LeNet_part2(isEncrypt, dim, num_classes)
    lenet_part3 = LeNet_part3(num_classes)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # lenet_part1 = lenet_part1.cuda()
        # lenet_part2 = lenet_part2.cuda()
        # lenet_part3 = lenet_part3.cuda()
        print("###########Using GPU!!!##########")

    return [lenet_part1,lenet_part2,lenet_part3]

# def Get_Model_State(board_size,num_input_channels,num_block,gpu_id):
#
#     model = get_model(board_size,num_input_channels,num_block,gpu_id)
#     model_state = model.state_dict()
#     key_list = list(model_state.keys())
#     value_list = list(model_state.values())
#
#     return key_list,value_list
