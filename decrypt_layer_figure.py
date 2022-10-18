import torch
from torch.autograd import Function
import copy

from tools.lib import quaternion_multiply

class decrypt_figure(Function):

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
            tensor_x = tensor_x.view(dim, -1)
            tensor_y = torch.matmul(torch.inverse(rot_mat[i]), tensor_x).view(dim, inSize[1], inSize[2], inSize[3])

            for d in range(dim):
                outputs[i + d * batch_size, :, :, :] = tensor_y[d]

        ctx.save_for_backward(rot_mat)
        
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        
        rot_mat, = ctx.saved_tensors
        dim = rot_mat.size()[-1]

        inSize = grad_outputs.size()
        batch_size = inSize[0] // dim

        grad_inputs = torch.zeros(batch_size * dim, inSize[1], inSize[2], inSize[3]).to(rot_mat.device)
        
        for i in range(batch_size):
            tensor_x_component_list = []
            for d in range(dim):
                tensor_x_component_list.append(grad_outputs[i + d * batch_size, :, :, :])
            tensor_x = torch.stack(tensor_x_component_list, 0)
            tensor_x = tensor_x.view(dim, -1)
            tensor_y = torch.matmul(rot_mat[i], tensor_x).view(dim, inSize[1], inSize[2], inSize[3])

            for d in range(dim):
                grad_inputs[i + d * batch_size, :, :, :] = tensor_y[d]


        grad_inputs = grad_inputs.cuda()
        
        return grad_inputs,rot_mat


# class decrypt_noise(Function):
#
#     @staticmethod
#     def forward(ctx, inputs, decrypt_theta, o1, o2, o3):
#
#         inSize = inputs.size()
#         batch_size = inSize[0] // 3
#         decrypt_theta = decrypt_theta / 2
#         # print('-------------------')
#         # print(inputs[0][0])
#
#         # outputs = copy.deepcopy(inputs.detach())
#         outputs = torch.zeros(batch_size, inSize[1], inSize[2], inSize[3]).to(inputs.device)
#
#         for i in range(batch_size):
#             tensor_x = [0, inputs[i, :, :, :], inputs[i + batch_size, :, :, :], inputs[i + 2 * batch_size, :, :, :]]
#             tensor_R = [torch.cos(decrypt_theta[i]), torch.sin(decrypt_theta[i]) * o1[i],
#                         torch.sin(decrypt_theta[i]) * o2[i], torch.sin(decrypt_theta[i]) * o3[i]]
#             tensor_R_c = [torch.cos(decrypt_theta[i]), -torch.sin(decrypt_theta[i]) * o1[i],
#                           -torch.sin(decrypt_theta[i]) * o2[i], -torch.sin(decrypt_theta[i]) * o3[i]]
#
#             tensor_y = quaternion_multiply(tensor_R_c, tensor_x)
#             tensor_y = quaternion_multiply(tensor_y, tensor_R)
#
#             outputs[i, :, :, :] = tensor_y[1]
#             # outputs[i + batch_size, :, :, :] = tensor_y[2]
#             # outputs[i + 2 * batch_size, :, :, :] = tensor_y[3]
#             # outputs[i + 3 * batch_size, :, :, :] = quaternion_multiply(tensor_x, tensor_y)[3]
#
#         ctx.save_for_backward(decrypt_theta, o1, o2, o3)
#
#         return outputs
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#
#         decrypt_theta, o1, o2, o3 = ctx.saved_tensors
#         # print('De---------')
#         # print(grad_outputs[0][0][0])
#
#         inSize = grad_outputs.size()
#         batch_size = inSize[0]
#         # print(ctx.needs_input_grad)
#         # print(decrypt_theta)
#
#         # grad_inputs = copy.deepcopy(grad_outputs)
#         grad_inputs = torch.zeros(batch_size * 3, inSize[1], inSize[2], inSize[3]).to(decrypt_theta.device)
#
#         for i in range(batch_size):
#             tensor_x = [0, grad_outputs[i, :, :, :], 0, 0]
#             tensor_R = [torch.cos(decrypt_theta[i]), torch.sin(decrypt_theta[i]) * o1[i],
#                         torch.sin(decrypt_theta[i]) * o2[i], torch.sin(decrypt_theta[i]) * o3[i]]
#             tensor_R_c = [torch.cos(decrypt_theta[i]), -torch.sin(decrypt_theta[i]) * o1[i],
#                           -torch.sin(decrypt_theta[i]) * o2[i], -torch.sin(decrypt_theta[i]) * o3[i]]
#
#             tensor_y = quaternion_multiply(tensor_R, tensor_x)
#             tensor_y = quaternion_multiply(tensor_y, tensor_R_c)
#
#             # grad_inputs[i, :, :, :] = tensor_y[0]
#             # grad_inputs[i + batch_size, :, :, :] = tensor_y[1]
#             # grad_inputs[i + 2 * batch_size, :, :, :] = tensor_y[2]
#             # grad_inputs[i + 3 * batch_size, :, :, :] = tensor_y[3]
#
#             grad_inputs[i, :, :, :] = tensor_y[1]
#             grad_inputs[i + batch_size, :, :, :] = tensor_y[2]
#             grad_inputs[i + 2 * batch_size, :, :, :] = tensor_y[3]
#             # grad_inputs[i + 3 * batch_size, :, :, :] = tensor_y[3]
#
#         grad_inputs = grad_inputs.cuda()
#
#         return grad_inputs, decrypt_theta, o1, o2, o3