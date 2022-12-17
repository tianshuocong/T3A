import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib
import torchvision
from collections import OrderedDict
import heapq
import torchvision.transforms as transforms
import numpy as np
from models.resnet import resnet18
from PIL import Image




def getEmbedding(model, data):
    modules = list(model.children())[:-1]
    generator = nn.Sequential(*modules)
    #print(generator)
    for p in generator.parameters():
        p.requires_grad = False
    z = generator(data)
    z = torch.squeeze(z)
    return z


def my_norm(x):
    l2 = torch.linalg.norm(x, dim=0, ord=2)
    return x/l2


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(0) * x.log_softmax(0)).sum(0)


def Filter(Set, W, M):
    entropy_list = []
    for j in range(Set.size(0)):
        logits = torch.mm(W, Set[j,:].unsqueeze(1))  # [10, 1]
        entropy = softmax_entropy(logits)
        entropy_list.append(entropy.item())

    indices = np.argsort(entropy_list)[-M:].tolist()
    Set_filtered = Set[indices, :]
    return Set_filtered



def Set_num(Set):
    n_list = []
    for i in range(10):
        n = Set[i].size(0)
        n_list.append(n)
    print(n_list)



def S_update(S, z, y_dot, fc_weight, M):
    # ## update support set
    S[y_dot.item()] = torch.cat((S[y_dot.item()], my_norm(z).unsqueeze(0)),0)
    # # ## Filter S
    S[y_dot.item()] = Filter(S[y_dot.item()], fc_weight, M)
    return S



def T3A_predictor(S, z):
    c = {}
    o = []
    for i in range(10):
        c[i] = torch.mean(S[i], 0)  ##[512]
        o.append(torch.mm(c[i].unsqueeze(0), z.unsqueeze(1)).item())  ## [1,512] x [512x1]
    predict_t3a =  o.index(max(o))
    return predict_t3a




def Set_adapt(S, base_model, inputs):
    M = 99999
    # NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
    # te_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), transforms.Normalize(*NORM)])
    # inputs = te_transforms(image_in).unsqueeze(0)
    inputs = inputs.cuda()
    fc_weight = base_model.state_dict()['fc.weight']   ## ResNet ## [10, 512]
    z = getEmbedding(base_model, inputs)  ## [32, 512]
    logits = torch.mm(fc_weight, z.unsqueeze(1)) # [32,512] x [512,10]

    # x = torch.topk(logits[:,0],2).indices[0].item()
    # y = torch.topk(logits[:,0],2).indices[1].item()
    # Matrix[x,y] = Matrix[x,y] + 1

    _, y_dot = logits.max(0)
    S = S_update(S, z, y_dot, fc_weight, M)
    return S




