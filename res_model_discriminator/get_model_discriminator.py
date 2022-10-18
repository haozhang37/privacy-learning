
import os
import torch
import torch.nn as nn

from res_model_discriminator.value_network81 import Model_Resnet81

def Get_Model_Discriminator(board_size,num_input_channels,num_block,gpu_id,kernel_size):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    model = Model_Resnet81(board_size,num_input_channels,num_block,kernel_size)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("###########Using GPU!!!##########")

    return model

def Get_Model_State(board_size,num_input_channels,num_block,gpu_id):

    model = get_model(board_size,num_input_channels,num_block,gpu_id)
    model_state = model.state_dict()
    key_list = list(model_state.keys())
    value_list = list(model_state.values())

    return key_list,value_list
