import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Normal residual block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', double_layer = False, dim=2): # dim as a dummy argument
        super(BasicBlock, self).__init__()
        self.double_layer = double_layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if double_layer:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes)
            self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],  # TODO: Here performs a pooling?
                                                  (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2),
                                                  "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.double_layer:
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.relu(self.bn3(self.conv3(out)))
            out = self.bn4(self.conv4(out))
        else:
            out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# The buliding block used between encrypt layer and decrypt layer
class OurBasicBlock(nn.Module):
    expansion = 1
    epsilon = 1e-5
    double_layer = False

    def __init__(self, in_planes, planes, stride=1, option='A', double_layer=False, dim=2):
        super(OurBasicBlock, self).__init__()
        self.double_layer = double_layer
        self.dim = dim

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if double_layer:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # TODO: Will it change phase?
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0, (planes-in_planes)//2, (planes-in_planes)//2),
                                                  "constant", 0))
            # discard option B for now
            assert option == 'A'
            # elif option == 'B':
            #     self.shortcut = nn.Sequential(
            #          nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            #          nn.BatchNorm2d(self.expansion * planes)
            #     )

    def our_relu(self,x):
        x_detach = x.detach()
        inSize = x_detach.size()
        batch_size = inSize[0] // self.dim
        mod_square = 0
        for d in range(self.dim):
            mod_square += x_detach[d * batch_size: (d + 1) * batch_size, :, :, :] ** 2
        mod = torch.sqrt(mod_square).to(x.device)
        # threshold = torch.mean(mod)
        threshold = 1
        threshold_mod = copy.deepcopy(mod)
        threshold_mod[:,:,:,:] = threshold
        after_thre_mod= torch.max(threshold_mod,mod)
        coefficient = torch.div(mod,after_thre_mod)
        coefficient = torch.cat([coefficient]*self.dim,0)
        y = torch.mul(coefficient, x)
        return y

    def our_bn(self, x):
        xd = x.detach()
        inSize = xd.size()
        xd2 = xd * xd
        mean2 = xd2.permute(1,0,2,3).contiguous().view(inSize[1],-1).mean(dim=1).to(x.device)
        coefficient = torch.sqrt(mean2 + self.epsilon).view(1,inSize[1],1,1).to(x.device)
        y = torch.div(x, coefficient)
        return y

    def forward(self, x):
        out = self.our_relu(self.our_bn(self.conv1(x)))
        if self.double_layer:
            out = self.our_relu(self.our_bn(self.conv2(out)))
            out = self.our_relu(self.our_bn(self.conv3(out)))
            out = self.our_bn(self.conv4(out))
        else:
            out = self.our_bn(self.conv2(out))
        out += self.shortcut(x)
        out = self.our_relu(out)
        return out


class ResNet_part1(nn.Module):
    def __init__(self, block, num_blocks, group_size, num_classes=10):
        super(ResNet_part1, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.group1 = self._make_layer(block, 16, num_blocks[0], [1]*num_blocks[0])
        self.group2 = self._make_layer(block, 32, num_blocks[1], [2]+[1]*(num_blocks[1]-1))

        # TODO: Initialize weights here?
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        # TODO: Add decrypt
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.group1(out)
        out = self.group2(out)
        return out

class ResNet_part2(nn.Module):

    def __init__(self, dim, block, num_blocks, group_size, double_filter=False, double_layer=False, num_classes=10, epsilon = 1e-5):
        super(ResNet_part2, self).__init__()
        # assert isinstance(block, OurBasicBlock)
        self.groups = []
        self.is_encrypt = (block == OurBasicBlock)
        self.double_filter = double_filter
        self.double_layer = double_layer
        self.epsilon = epsilon
        self.dim = dim

        self.in_planes = 32 if num_blocks[1] != group_size else 16
        strides = [1]*num_blocks[1] if num_blocks[1]!=group_size else [2]+[1]*(num_blocks[1]-1)

        out_planes = 32
        if num_blocks[1] != 0:
            self.groups.append(self._make_layer(block, 32, num_blocks[1], strides,
                                                double_filter=double_filter, double_layer=double_layer))
            out_planes = 32
        if num_blocks[2] != 0:
            self.groups.append(self._make_layer(block, 64, num_blocks[2], [2]+[1]*(num_blocks[2]-1),
                                                double_filter=double_filter, double_layer=double_layer))
            out_planes = 64
        self.groups = nn.ModuleList(self.groups)
        if double_filter:
            self.conv = nn.Conv2d(self.in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False).cuda()
            if not self.is_encrypt:
                self.bn = nn.BatchNorm2d(out_planes).cuda()
        # TODO: Initialize weights here?
        for sec in self.groups:
            sec.apply(_weights_init)
        self.apply(_weights_init)
            
    def _make_layer(self, block, planes, num_blocks, strides, double_filter=False, double_layer=False):
        layers = []
        if double_filter:
            planes = planes*2
            # TODO: If encrypt before first layer of a group, channel expanded by 4, will it affect trainning?
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, double_layer=double_layer, dim=self.dim))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def our_bn(self, x):
        xd = x.detach()
        inSize = xd.size()
        xd2 = xd * xd
        mean2 = xd2.permute(1, 0, 2, 3).contiguous().view(inSize[1], -1).mean(dim=1).cuda()
        coefficient = torch.sqrt(mean2 + self.epsilon).view(1, inSize[1], 1, 1).cuda()
        y = torch.div(x, coefficient)
        return y
    
    def forward(self,x):
        out = x
        for sec in self.groups:
            out = sec(out)
        if self.double_filter:
            if self.is_encrypt:
                out = self.our_bn(self.conv(out))
            else:
                out = self.bn(self.conv(out))
        return out

class ResNet_part3(nn.Module):
    def __init__(self, block, num_blocks, group_size, num_classes=10):
        super(ResNet_part3, self).__init__()
        self.in_planes = 64 if num_blocks[2]!=group_size else 32
        strides = [1]*num_blocks[2] if num_blocks[2]!=group_size else [2]+[1]*(num_blocks[2]-1)
        self.layers = self._make_layer(block, 64, num_blocks[2], strides)

        self.linear = nn.Linear(64, num_classes)

        # TODO: Initialize weights here?
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out = x
        out = self.layers(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Original
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# def resnet20():
#     return ResNet(BasicBlock, [3, 3, 3])
#
#
# def resnet32():
#     return ResNet(BasicBlock, [5, 5, 5])
#
#
# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])
#
#
def resnet56(num_classes):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

#
# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])
#
#
# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])

# def resnet20():
#     return (ResNet_part1(BasicBlock, [3, 3, 0], 3),
#             ResNet_part2(BasicBlock, [0, 0, 2], 3),
#             ResNet_part3(BasicBlock, [0, 0, 1], 3))
#
# def resnet32():
#     return (ResNet_part1(BasicBlock, [5, 1, 0], 5),
#             ResNet_part2(BasicBlock, [0, 4, 4], 5),
#             ResNet_part3(BasicBlock, [0, 0, 1], 5))
#
# def resnet44():
#     return (ResNet_part1(BasicBlock, [6, 0, 0], 7),
#             ResNet_part2(BasicBlock, [1, 7, 6], 7),
#             ResNet_part3(BasicBlock, [0, 0, 1], 7))
#
# def resnet56():
#     return (ResNet_part1(BasicBlock, [9, 9, 0], 9),
#             ResNet_part2(BasicBlock, [0, 0, 6], 9),
#             ResNet_part3(BasicBlock, [0, 0, 3], 9))
#
# def resnet110():
#     return (ResNet_part1(BasicBlock, [18, 0, 0], 18),
#             ResNet_part2(BasicBlock, [0, 18, 15], 18),
#             ResNet_part3(BasicBlock, [0, 0, 3], 18))


# def test(net):
#     import numpy as np
#     total_params = 0
#
#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.data.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    net = resnet56(100).cuda()
    x = torch.randn(2, 3, 32, 32).cuda()
    y = net(x)
