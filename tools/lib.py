import os
import sys
import scipy.io as scio
import copy

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from PIL import Image




def complex_rotate(real_part,imaginary_part,theta):

    input_size = real_part.size()
    complex = torch.zeros(input_size[0],input_size[1]*2,input_size[2],input_size[3])
    rotate_real = real_part*math.cos(theta) - imaginary_part*math.sin(theta)
    rotate_imaginary = imaginary_part*math.cos(theta) + real_part*math.sin(theta)
    complex[:,0:input_size[1],:,:] = rotate_real
    complex[:,input_size[1]:2*input_size[1],:,:] = rotate_imaginary
    complex = complex.cuda()

    return complex


def quaternion_multiply(q,p):
    s = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3]
    x = q[0] * p[1] + q[1] * p[0] + q[2] * p[3] - q[3] * p[2]
    y = q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1]
    z = q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0]
    result = [s,x,y,z]
    return result


def get_binary_error(output,label,binary_threshold):

    output = np.array(output.detach().cpu())
    label = np.array(label.detach().cpu())
    output = output - binary_threshold
    output = np.sign(output)
    label = (label - 0.5)*2
    binary_error = np.multiply(output,label)
    binary_error = (binary_error+1)/2
    binary_error = np.mean(binary_error)
    binary_error = 1 - binary_error

    return binary_error


# def for_hook(module, input):
#
#         for val in input:
#             layer.append(val.data.cpu().numpy())
#
#
# def get_input(model, input, layer_name, index=-1):
#
#     layer = []
#     handle_feat = layer_name.register_forward_pre_hook(for_hook)
#     model(input)
#     input_layer = torch.from_numpy(layer[index]).cuda()
#     handle_feat.remove()
#
#     return input_layer


def im_show(image,folderid,epoch,id):

    image = np.array(image)
    size = image.shape
    new_image = np.zeros((3,size[0],size[1]))
    new_image[0] = image
    new_image[1] = image
    new_image[2] = image
    visual = new_image.transpose((1,2,0))
    visual_max = float(np.max(visual))
    visual_min = float(np.min(visual))
    visual = (visual - visual_min)/(visual_max-visual_min)
    visual = 1 - visual
    visual = visual*255
    visual = np.uint8(visual)
    img = Image.fromarray(visual, 'RGB')
    img.save('./result/debug/'+str(folderid)+'/'+str(epoch)+'_'+str(id)+'.png')
    # img.show()


def get_revise_criterion(input):

    size = input.size()
    output = copy.deepcopy(input)
    output[:,0:int(size[1]/2),:,:] = input[:,int(size[1]/2):size[1],:,:]
    output[:,int(size[1]/2):size[1],:,:] = input[:,0:int(size[1]/2),:,:]
    output = output.cuda()

    return output


def get_noise(image,mean,std):

    image_size = image.size()
    noise = torch.rand(image_size).to(image.device)
    for i in range(len(noise[1])):
        noise[:,i,:,:] = mean[i]-noise[:,i,:,:]
        noise[:,i,:,:] = noise[:,i,:,:]/std[i]

    return noise


def our_bn(x,epsilon):

    xd = x.detach()
    inSize = xd.size()
    xd2 = xd * xd
    mean2 = xd2.permute(1,0,2,3).contiguous().view(inSize[1],-1).mean(dim=1).cuda()
    coefficient = torch.sqrt(mean2 + epsilon).view(1,inSize[1],1,1).cuda()
    y = torch.div(x, coefficient)

    return y


def our_relu(x,IsEncrypt,dim):

    if IsEncrypt == True:
        x_detach = x.detach()
        inSize = x_detach.size()
        batch_size = inSize[0] // dim
        mod_square = 0
        for d in range(dim):
            mod_square = mod_square + x_detach[d * batch_size: (d+1) * batch_size,:,:,:] ** 2
        mod = torch.sqrt(mod_square).to(x.device)
        threshold = torch.mean(mod)
        threshold_mod = copy.deepcopy(mod)
        threshold_mod[:,:,:,:] = threshold
        after_thre_mod= torch.max(threshold_mod,mod)
        coefficient = torch.div(mod,after_thre_mod)
        coefficient = torch.cat([coefficient] * dim,0)
        y = torch.mul(coefficient, x)

    else:

        y = F.relu(x)

    return y

def get_GANtheta(batch_use,GANtheta_angle): # no use

    GANtheta = torch.zeros(batch_use,len(GANtheta_angle))
    for angle_id in range(len(GANtheta_angle)):
            GANtheta[:,angle_id] = GANtheta_angle[angle_id]
    GANtheta = GANtheta.float().cuda()

    return GANtheta


def get_cifar10_error(outputs,labels): # no use

    outputs_size = outputs.size()
    batch_size = outputs_size[0]
    cifar10_error = 0
    for batch_id in range(batch_size):
        outputs_batch = outputs[batch_id]
        max_id = torch.argmax(outputs_batch)
        label_id = labels[batch_id]
        if max_id != label_id:
           cifar10_error += 1
    cifar10_error = cifar10_error/batch_size

    return cifar10_error


def get_D_error(D_pos,D_neg):

    batch_size = D_pos.size()[0]
    error = torch.zeros(batch_size,1)
    for batch_id in range(batch_size):
        error[batch_id] = D_pos[batch_id] - D_neg[batch_id]
    error = torch.sign(error)
    error = (error + 1)/2   # if pos > neg, it will be 1, else 0. It can be the accuracy.
    error = 1 - error  # 1 = accuracy + error
    error = torch.mean(error)

    return error


def get_norm_coefficient(inputs): # no use

    inputs_size = inputs.size()
    real_part = inputs[0:int(inputs_size[0]/2),:,:,:]
    imaginary_part = inputs[int(inputs_size[0]/2):inputs_size[0],:,:,:]
    size_coefficient = math.sqrt(inputs_size[0]*inputs_size[1]*inputs_size[2]*inputs_size[3])
    mold_coefficient = torch.sqrt(torch.sum(real_part.mul(real_part)+imaginary_part.mul(imaginary_part)))
    norm_coefficient = float(size_coefficient/mold_coefficient)
    print(norm_coefficient)

    return norm_coefficient


def complex_encrypt_detach(inputs,encrypt_theta): # no use

    inSize = inputs.size()
    batch_size = inSize[0] // 2

    outputs = copy.deepcopy(inputs[0:batch_size,:,:,:].detach())

    for i in range(batch_size):
        outputs[i,:,:,:] = inputs[i,:,:,:] * torch.cos(encrypt_theta[i]) - \
                     inputs[i+batch_size,:,:,:] * torch.sin(encrypt_theta[i])

    outputs = outputs.cuda()

    return outputs

def quaternion_encrypt_detach(inputs, encrypt_theta, o1, o2, o3): # no use
    inSize = inputs.size()
    batch_size = inSize[0] // 3
    encrypt_theta = encrypt_theta / 2

    outputs = torch.zeros(batch_size, inSize[1], inSize[2], inSize[3]).to(inputs.device)

    for i in range(batch_size):
        tensor_x = [0, inputs[i, :, :, :], inputs[i + batch_size, :, :, :], inputs[i + 2 * batch_size, :, :, :]]
        tensor_R = [torch.cos(encrypt_theta[i]), torch.sin(encrypt_theta[i]) * o1[i],
                    torch.sin(encrypt_theta[i]) * o2[i], torch.sin(encrypt_theta[i]) * o3[i]]
        tensor_R_c = [torch.cos(encrypt_theta[i]), -torch.sin(encrypt_theta[i]) * o1[i],
                      -torch.sin(encrypt_theta[i]) * o2[i], -torch.sin(encrypt_theta[i]) * o3[i]]

        tensor_y = quaternion_multiply(tensor_R, tensor_x)
        tensor_y = quaternion_multiply(tensor_y, tensor_R_c)

        outputs[i, :, :, :] = random.choice([tensor_y[1], tensor_y[2], tensor_y[3]])
    return outputs.detach()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_feature(feature,visual_path):

    feature_size = feature.size()
    new_feature = np.zeros((feature_size[0],feature_size[1],3))
    feature = np.array(feature)
    new_feature[:,:,0] = feature
    new_feature[:,:,1] = feature
    new_feature[:,:,2] = feature
    max = np.max(new_feature)
    min = np.min(new_feature)
    new_feature = 255*(new_feature-min)/(max-min)
    new_feature = np.uint8(new_feature)
    feature_visual = Image.fromarray(new_feature, 'RGB')
    feature_visual.save(visual_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class encrypt_GAN(Function): # no use

    @staticmethod
    def forward(ctx,inputs,encrypt_GANtheta,GAN_encrypt_zeros):

        inSize = inputs.size()
        batch_size = inSize[0] // 2

        outputs = copy.deepcopy(inputs[0:batch_size,:,:,:].detach())

        for i in range(batch_size):
            outputs[i,:,:,:] = inputs[i,:,:,:] * torch.cos(encrypt_GANtheta[i]) - \
                         inputs[i+batch_size,:,:,:] * torch.sin(encrypt_GANtheta[i])

        outputs = outputs.cuda()
        ctx.save_for_backward(encrypt_GANtheta,GAN_encrypt_zeros)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):

        encrypt_GANtheta,GAN_encrypt_zeros = ctx.saved_tensors
        # print(ctx.needs_input_grad)

        # print(torch.sum(torch.abs(grad_outputs)))
        # print(encrypt_GANtheta)

        inSize = grad_outputs.size()
        batch_size = inSize[0]
        grad_inputs = copy.deepcopy(GAN_encrypt_zeros)

        for i in range(batch_size):
            grad_inputs[i,:,:,:] = grad_outputs[i,:,:,:] * torch.cos(encrypt_GANtheta[i])
            grad_inputs[i+batch_size,:,:,:] = - grad_outputs[i,:,:,:] * torch.sin(encrypt_GANtheta[i])

        grad_inputs = grad_inputs.cuda()

        return grad_inputs,encrypt_GANtheta,GAN_encrypt_zeros

class quaternion_encrypt_GAN(Function): # no use

    @staticmethod
    def forward(ctx,inputs,encrypt_theta,o1,o2,o3):

        inSize = inputs.size()
        batch_size = inSize[0] // 3
        encrypt_theta = encrypt_theta / 2

        outputs = torch.zeros(batch_size * 3, inSize[1], inSize[2], inSize[3]).to(inputs.device)

        for i in range(batch_size):

            tensor_x = [0,inputs[i,:,:,:],inputs[i+batch_size,:,:,:],inputs[i+2*batch_size,:,:,:]]
            tensor_R = [torch.cos(encrypt_theta[i]), torch.sin(encrypt_theta[i])*o1[i],
                        torch.sin(encrypt_theta[i])*o2[i], torch.sin(encrypt_theta[i])*o3[i]]
            tensor_R_c = [torch.cos(encrypt_theta[i]), -torch.sin(encrypt_theta[i]) * o1[i],
                          -torch.sin(encrypt_theta[i]) * o2[i], -torch.sin(encrypt_theta[i]) * o3[i]]

            tensor_y = quaternion_multiply(tensor_R, tensor_x)
            tensor_y = quaternion_multiply(tensor_y, tensor_R_c)

            outputs[i, :, :, :] = tensor_y[1]
            outputs[i + batch_size, :, :, :] = tensor_y[2]
            outputs[i + 2 * batch_size, :, :, :] = tensor_y[3]

        ctx.save_for_backward(encrypt_theta,o1,o2,o3)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        encrypt_theta,o1,o2,o3 = ctx.saved_tensors

        inSize = grad_outputs.size()
        batch_size = inSize[0] // 3
        # grad_inputs = copy.deepcopy(grad_outputs)
        grad_inputs = torch.zeros(batch_size * 3, inSize[1], inSize[2], inSize[3]).to(encrypt_theta.device)

        for i in range(batch_size):
            tensor_x = [0, grad_outputs[i, :, :, :], grad_outputs[i+batch_size, :, :, :],
                        grad_outputs[i+2*batch_size, :, :, :]]
            tensor_R = [torch.cos(encrypt_theta[i]), torch.sin(encrypt_theta[i]) * o1[i],
                        torch.sin(encrypt_theta[i]) * o2[i], torch.sin(encrypt_theta[i]) * o3[i]]
            tensor_R_c = [torch.cos(encrypt_theta[i]), -torch.sin(encrypt_theta[i]) * o1[i],
                          -torch.sin(encrypt_theta[i]) * o2[i], -torch.sin(encrypt_theta[i]) * o3[i]]

            tensor_y = quaternion_multiply(tensor_R_c, tensor_x)
            tensor_y = quaternion_multiply(tensor_y, tensor_R)
            grad_inputs[i, :, :, :] = tensor_y[1]
            grad_inputs[i + batch_size, :, :, :] = tensor_y[2]
            grad_inputs[i + 2 * batch_size, :, :, :] = tensor_y[3]

        return grad_inputs,encrypt_theta,o1,o2,o3



def encrypt_detach_by_rot_mat(inputs, rot_mat):
    inSize = inputs.size()
    dim = rot_mat.size()[-1] # todo: check
    # print("rot mat dim=", dim)
    batch_size = inSize[0] // dim

    ouputs_list=[]
    v1 = torch.zeros(dim, device=rot_mat.device)
    v1[0] = 1. # unit vector of dimension dim
    for i in range(batch_size):
        v1_rot = torch.matmul(rot_mat[i], v1)
        rot_cosine = torch.dot(v1_rot, v1)
        if rot_cosine > math.cos(math.pi*0.25):
            continue

        tensor_x_component_list = []
        for d in range(dim):
            tensor_x_component_list.append(inputs[i + d * batch_size, :, :, :])
        tensor_x = torch.stack(tensor_x_component_list, 0)
        tensor_y = torch.matmul(rot_mat[i], tensor_x.view(dim, -1)).view(dim, inSize[1], inSize[2], inSize[3]) # rqh 0525 delete cpu()
        choice = random.choice(np.arange(dim))
        ouputs_list.append(tensor_y[choice])
        #outputs[i, :, :, :] = random.choice([tensor_y[0],tensor_y[1], tensor_y[2], tensor_y[3],tensor_y[4]])
    sample_len = len(ouputs_list)
    outputs = torch.zeros(sample_len, inSize[1], inSize[2], inSize[3]).to(inputs.device)
    for i in range(sample_len):
        outputs[i]=ouputs_list[i]
    return outputs.detach()

def encrypt_detach_by_rot_mat_nodiscard(inputs, rot_mat):
    inSize = inputs.size()
    dim = rot_mat.size()[-1]
    # print("rot mat dim=", dim)
    batch_size = inSize[0] // dim

    outputs = torch.zeros(batch_size, inSize[1], inSize[2], inSize[3]).to(inputs.device)
    for i in range(batch_size):
        tensor_x_component_list = []
        for d in range(dim):
            tensor_x_component_list.append(inputs[i + d * batch_size, :, :, :])
        tensor_x = torch.stack(tensor_x_component_list, 0)
        tensor_y = torch.matmul(rot_mat[i], tensor_x.view(dim, -1)).view(dim, inSize[1], inSize[2], inSize[3])  # rqh 0525 delete cpu()
        choice = random.choice(np.arange(dim))
        outputs[i, :, :, :] = tensor_y[choice]
    return outputs.detach()

def gen_rot_mat(n, dim):
    temp_tensor = torch.randn(dim, dim)
    rot_mat = torch.zeros(n, dim, dim)
    key_num = 0
    while key_num < n:
        nn.init.orthogonal_(temp_tensor)
        if torch.det(temp_tensor) > 0:
            rot_mat[key_num] = temp_tensor
            key_num = key_num + 1
    return rot_mat

def complex2rotmat(theta):
    """
    Input:
        theta: (B,) tensor
    Return:
        rot_mat: (B,2,2) tensor
    """
    B = theta.size()[0]
    rot_mat = torch.zeros(B,2,2)
    rot_mat[:, 0, 0] = torch.cos(theta)
    rot_mat[:, 0, 1] = -torch.sin(theta)
    rot_mat[:, 1, 0] = torch.sin(theta)
    rot_mat[:, 1, 1] = torch.cos(theta)
    return rot_mat

def quarternion2rotmat(theta, o1, o2, o3):
    """
        Input:
            theta, o1, o2, o3: all are (B,) tensors
        Return:
            rot_mat: (B,3,3) tensor
    """
    B = theta.size()[0]
    cos, sin = torch.cos(theta), torch.sin(theta)
    rot_mat = torch.zeros(B, 3, 3)
    rot_mat[:, 0, 0] = cos + (1 - cos) * (o1 ** 2)
    rot_mat[:, 0, 1] = (1 - cos) * o1 * o2 - sin * o3
    rot_mat[:, 0, 2] = (1 - cos) * o1 * o3 + sin * o2
    rot_mat[:, 1, 0] = (1 - cos) * o1 * o2 + sin * o3
    rot_mat[:, 1, 1] = cos + (1 - cos) * (o2 ** 2)
    rot_mat[:, 1, 2] = (1 - cos) * o2 * o3 - sin * o1
    rot_mat[:, 2, 0] = (1 - cos) * o1 * o3 - sin * o2
    rot_mat[:, 2, 1] = (1 - cos) * o2 * o3 + sin * o1
    rot_mat[:, 2, 2] = cos + (1 - cos) * (o3 ** 2)

    return rot_mat # no problem, 0605 verified
