import torch
from torch.autograd import Function
import copy

from tools.lib import quaternion_multiply

class encrypt(Function):

    @staticmethod
    def forward(ctx,inputs,rot_mat):

        inSize = inputs.size()
        dim = rot_mat.size()[-1]
        batch_size = inSize[0] // dim

        outputs = torch.zeros(batch_size * dim, inSize[1], inSize[2], inSize[3]).to(inputs.device)

        for i in range(batch_size):
            tensor_x_component_list = []
            for d in range(dim):
                tensor_x_component_list.append(inputs[i + d * batch_size, :, :, :])
            tensor_x = torch.stack(tensor_x_component_list, 0)
            tensor_x = tensor_x.view(dim,-1)
            tensor_y = torch.matmul(rot_mat[i], tensor_x).view(dim, inSize[1], inSize[2], inSize[3]) #rqh 0526 delete cpu()

            for d in range(dim):
                outputs[i + d * batch_size, :, :, :] = tensor_y[d]

        # outputs = outputs.cuda()
        ctx.save_for_backward(rot_mat)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):

        # print('En------------------')
        # print(grad_outputs[0][0][0])

        rot_mat, = ctx.saved_tensors
        dim = rot_mat.size()[-1]
        # print(ctx.needs_input_grad)
        # print(encrypt_theta)

        inSize = grad_outputs.size()
        batch_size = inSize[0] // dim

        grad_inputs = torch.zeros(batch_size * dim, inSize[1], inSize[2], inSize[3]).to(rot_mat.device)

        for i in range(batch_size):
            tensor_x_component_list = []
            for d in range(dim):
                tensor_x_component_list.append(grad_outputs[i + d * batch_size, :, :, :])
            tensor_x = torch.stack(tensor_x_component_list, 0)
            tensor_x = tensor_x.view(dim, -1)
            tensor_y = torch.matmul(torch.inverse(rot_mat[i]), tensor_x).view(dim, inSize[1], inSize[2], inSize[3])#rqh 0526 delete cpu()

            for d in range(dim):
                grad_inputs[i + d * batch_size, :, :, :] = tensor_y[d]

        grad_inputs = grad_inputs.cuda()

        return grad_inputs,rot_mat
