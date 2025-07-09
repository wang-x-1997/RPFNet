import torch
from torch import nn
import torch.nn.functional as F
import math
# from function import adaptive_instance_normalization as adain
# from s import fused_leaky_relu
import numpy as np
# from extractor import VitExtractor

def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    return (x - min1) / (max1 - min1 + 1e-10)

import torch
from torch import nn
from torch.nn import functional as F



def sumPatch(x,k):
    dim = x.shape
    kernel = np.ones((2*k+1,2*k+1))
    kernel = kernel/(1.0*(2*k+1)*(2*k+1))
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1],dim[1],1,1)
    weight = nn.Parameter(data=kernel,requires_grad=False)
    weight = weight.cuda()
    gradMap = F.conv2d(x,weight=weight,stride=1,padding=k)
    return gradMap

def sqt(F_ir,F_vis):
    F_1 = F_ir
    F_2 = F_vis
    # F_1 = torch.sqrt((F_1 - torch.mean(F_1)) ** 2)
    # F_2 = torch.sqrt((F_2 - torch.mean(F_2)) ** 2)
    # F_1 = torch.split(F_ir, 4, 1)
    # F_2 = torch.split(F_vis, 4, 1)
    F_1 = torch.sqrt((F_1 - torch.mean(F_1)) ** 2)
    F_2 = torch.sqrt((F_2 - torch.mean(F_2)) ** 2)
    g_ir = F_1.sum(dim=1, keepdim=True) / 64
    g_vi = F_2.sum(dim=1, keepdim=True) / 64
    w1 = g_ir.greater(g_vi)
    w2 = ~w1
    w1 = w1.to(torch.int)
    w2 = w2.to(torch.int)
    # w1 = g_ir / (g_vi + g_ir + 1e-10)
    # w2 = 1 - w1
    return  w1,w2

def act(F_ir,F_vis):
    F_1 = torch.split(F_ir, 4, 1)
    F_2 = torch.split(F_vis, 4, 1)
    # F_1 = F_ir
    # F_2 = F_vis
    g_ir = F_1[0].sum(dim=1, keepdim=True) / 16
    g_vi = F_2[0].sum(dim=1, keepdim=True) / 16
    g_ir = sumPatch(g_ir, 3)
    g_vi = sumPatch(g_vi, 3)
    w1 = g_ir.greater(g_vi)
    w2 = ~w1
    w1 = w1.to(torch.int)
    w2 = w2.to(torch.int)
    # w1 = g_ir/(g_vi+g_ir+1e-10)
    # w2 = 1-w1
    return w1,w2

def cc(tensor,tensor1):

    c, d, h, w = tensor.size()
    tensor = tensor.view(c, d * h*w)
    tensor1 = tensor1.view(c, d * h * w)
    gram = torch.mm(tensor , tensor1.t())
    return gram.mean()

def en_ac(x,y):
    F_1 = torch.split(x, 4, 1)
    F_1 = torch.cat([F_1[1], F_1[2], F_1[3]], 1)
    D_fea = torch.cat([y[0], y[1], y[2]], 1)
    c_d = D_fea.size(1)
    g_ir = D_fea.sum(dim=1, keepdim=True)/c_d
    g_vi = F_1.sum(dim=1, keepdim=True) / 48
    w1 = norm_1(g_ir - g_vi)
    return w1
def en_w(x,y):
    w1 = en_ac(x[0], y[0])
    w2 = en_ac(x[1], y[1])
    w1 = w1.greater(w2)
    w2 = ~w1
    return w1,w2
