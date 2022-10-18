
import os
import torch
import torch.nn as nn

from res_model.resnet import BasicBlock
from res_model.resnet import OurBasicBlock
from res_model.resnet import ResNet_part1
from res_model.resnet import ResNet_part2
from res_model.resnet import ResNet_part3

def Get_Model_ResNet(gpu_id,IsEncrypt, dim, double_filter,double_layer,num_classes,num_layer,encrypt_location):

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if (num_layer-2) % 3 != 0:
        print('Error: wrong num of layers!')
    else:
        num_block = (num_layer-2)/2
        num_block_per_group = int(num_block/3)
        if encrypt_location == 'a':
            resnet_part1 = ResNet_part1(BasicBlock, [num_block_per_group, 1, 0], num_block_per_group)
            if IsEncrypt == True:
                resnet_part2 = ResNet_part2(dim, OurBasicBlock, [0, num_block_per_group-1, 1], num_block_per_group, double_filter=double_filter, double_layer=double_layer)
            else:
                resnet_part2 = ResNet_part2(dim, BasicBlock, [0, num_block_per_group-1, 1], num_block_per_group, double_filter=double_filter, double_layer=double_layer)
            resnet_part3 = ResNet_part3(BasicBlock, [0, 0, num_block_per_group-1], num_block_per_group, num_classes=num_classes)
        elif encrypt_location == 'b':
            resnet_part1 = ResNet_part1(BasicBlock, [num_block_per_group, 1, 0], num_block_per_group)
            if IsEncrypt == True:
                resnet_part2 = ResNet_part2(dim, OurBasicBlock, [0, num_block_per_group-1, num_block_per_group-1], num_block_per_group, double_filter=double_filter, double_layer=double_layer)
            else:
                resnet_part2 = ResNet_part2(dim, BasicBlock, [0, num_block_per_group-1, num_block_per_group-1], num_block_per_group, double_filter=double_filter, double_layer=double_layer)
            resnet_part3 = ResNet_part3(BasicBlock, [0, 0, 1], num_block_per_group, num_classes=num_classes)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            # resnet_part1 = resnet_part1.cuda()
            # resnet_part2 = resnet_part2.cuda()
            # resnet_part3 = resnet_part3.cuda()
            print("###########Using GPU!!!##########")

        return [resnet_part1,resnet_part2,resnet_part3]

# def Get_Model_State(board_size,num_input_channels,num_block,gpu_id):
#
#     model = get_model(board_size,num_input_channels,num_block,gpu_id)
#     model_state = model.state_dict()
#     key_list = list(model_state.keys())
#     value_list = list(model_state.values())
#
#     return key_list,value_list
