import torch
from torch.autograd import Function
import copy

class encrypt(Function):

    @staticmethod
    def forward(ctx,inputs,encrypt_theta):

        inSize = inputs.size()
        batch_size = inSize[0] // 2

        outputs = copy.deepcopy(inputs.detach())

        for i in range(batch_size):
            outputs[i,:,:,:] = inputs[i,:,:,:] * torch.cos(encrypt_theta[i]) - \
                         inputs[i+batch_size,:,:,:] * torch.sin(encrypt_theta[i])
            outputs[i+batch_size,:,:,:] = inputs[i,:,:,:] * torch.sin(encrypt_theta[i]) + \
                         inputs[i+batch_size,:,:,:] * torch.cos(encrypt_theta[i])

        outputs = outputs.cuda()
        ctx.save_for_backward(encrypt_theta)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):

        encrypt_theta, = ctx.saved_tensors

        inSize = grad_outputs.size()
        batch_size = inSize[0] // 2
        A = grad_outputs[0:batch_size,:,:,:]
        B = grad_outputs[batch_size:2*batch_size,:,:,:]
        grad_inputs = copy.deepcopy(grad_outputs)

        for i in range(batch_size):
            grad_inputs[i,:,:,:] = A[i,:,:,:] * torch.cos(encrypt_theta[i]) + B[i,:,:,:] * torch.sin(encrypt_theta[i])
            grad_inputs[i+batch_size,:,:,:] = B[i,:,:,:]*torch.cos(encrypt_theta[i]) - A[i,:,:,:] * torch.sin(encrypt_theta[i])

        return grad_inputs,encrypt_theta
