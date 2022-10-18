import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import gradcheck
import copy

class our_mpool2d(Function):
    @staticmethod
    def forward(ctx, x, dim, kernel_size = torch.IntTensor([2]), stride = torch.IntTensor([2])):
        x_detach = x.detach()

        inSize = x_detach.size()
        inSize = torch.IntTensor([inSize[0], inSize[1], inSize[2], inSize[3]])
        batch_size = inSize[0] // dim

        mod_square = 0
        for d in range(dim):
            mod_square += x_detach[d * batch_size: (d + 1) * batch_size, :, :, :] ** 2
        mod = torch.sqrt(mod_square).to(x.device)
        mod, indices = F.max_pool2d(mod, kernel_size.item(), stride.item(), return_indices = True)
        modSize = mod.size()
        ctx.save_for_backward(x, indices, dim)
        x = x.view(inSize[0], inSize[1], -1)
        indices = indices.view(modSize[0], modSize[1], -1)
        indices = torch.cat([indices] * dim, 0)
        y = torch.gather(x, 2, indices)
        y = y.view(modSize[0] * dim, modSize[1], modSize[2], modSize[3])
        return y
    @staticmethod
    def backward(ctx, grad_outputs):
        # grad_outputs = grad_outputs.cuda()
        x, indices, dim = ctx.saved_variables
        inSize = x.size()

        grad_inputs = copy.deepcopy(x.detach())
        grad_inputs.zero_()
        grad_inputs = grad_inputs.to(grad_outputs.device)
        grad_inputs = grad_inputs.view(inSize[0], inSize[1], -1)
        grad_outputs = grad_outputs.view(inSize[0], inSize[1], -1)
        indices = indices.view(inSize[0] // dim, inSize[1], -1)
        indices = torch.cat([indices] * dim, 0)
        grad_inputs = grad_inputs.scatter_(2, indices, grad_outputs)
        grad_inputs = grad_inputs.view(inSize[0],inSize[1],inSize[2],inSize[3])
        grad_inputs = grad_inputs.to(grad_outputs.device)
        # print(grad_inputs)
        return grad_inputs, None, None

class our_mpool(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2):
        super(our_mpool, self).__init__()
        self.kernel_size = torch.IntTensor([kernel_size])
        self.stride = torch.IntTensor([stride])

    def forward(self, x, isEncrypt, dim):
        if not isEncrypt:
            x = F.max_pool2d(x, self.kernel_size.item(), self.stride.item())
            return x
        else:
            x = our_mpool2d.apply(x, dim, self.kernel_size, self.stride)
            return x

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
    device = torch.device(2)
    x = torch.randn(2, 2, 5, 5, requires_grad = True).cuda()
    pool = our_mpool(3, 2).cuda()
    y = pool(x)
    print(x)
    print(y)
    z = y.mean()
    z.backward()



