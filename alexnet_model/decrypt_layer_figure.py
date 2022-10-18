import torch
from torch.autograd import Function
import copy

class decrypt_figure(Function):

    @staticmethod
    def forward(ctx,inputs,decrypt_theta):

        inSize = inputs.size()
        batch_size = inSize[0] // 2
        A = inputs[0:batch_size,:,:,:]
        B = inputs[batch_size:2*batch_size,:,:,:]

        outputs = copy.deepcopy(inputs.detach())
        for i in range(batch_size):
            outputs[i,:,:,:] = A[i,:,:,:] * torch.cos(decrypt_theta[i]) + B[i,:,:,:] * torch.sin(decrypt_theta[i])
            outputs[i+batch_size,:,:,:] = B[i,:,:,:] * torch.cos(decrypt_theta[i]) - A[i,:,:,:] * torch.sin(decrypt_theta[i])

        outputs = outputs.cuda()
        ctx.save_for_backward(decrypt_theta)
        
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        
        decrypt_theta, = ctx.saved_tensors
        
        inSize = grad_outputs.size()
        batch_size = inSize[0]

        grad_inputs = copy.deepcopy(grad_outputs)
        A = copy.deepcopy(grad_outputs[0:int(batch_size/2),:,:,:])
        B = copy.deepcopy(grad_outputs[0:int(batch_size/2),:,:,:])
        for i in range(int(batch_size/2)):
            A[i,:,:,:] = grad_outputs[i,:,:,:] * torch.cos(decrypt_theta[i]) - \
                         grad_outputs[i+int(batch_size/2),:,:,:] * torch.sin(decrypt_theta[i])
            B[i,:,:,:] = grad_outputs[i,:,:,:] * torch.sin(decrypt_theta[i]) + \
                         grad_outputs[i+int(batch_size/2),:,:,:] * torch.cos(decrypt_theta[i])

        grad_inputs_size = grad_inputs.size()
        grad_inputs[0:int(grad_inputs_size[0]/2),:,:,:] = A
        grad_inputs[int(grad_inputs_size[0]/2):grad_inputs_size[0],:,:,:] = B

        grad_inputs = grad_inputs.cuda()
        
        return grad_inputs,decrypt_theta
