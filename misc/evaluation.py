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
import copy
from misc.t3a import *

def test_all(model, dataset):

    correct = []
    NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
    te_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), transforms.Normalize(*NORM)])


    for id in range(9000,10000):
        _, labels = dataset[id]
        image = Image.fromarray(dataset.data[id])
        inputs = te_transforms(image).unsqueeze(0)
        inputs = inputs.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())

    correct1 = torch.cat(correct).numpy()
    acc = correct1.mean()
    return acc


def test_all_loader(model, dataset):

    correct = []

    for batch_idx, (inputs, labels) in enumerate(dataset):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())

    correct1 = torch.cat(correct).numpy()
    acc = correct1.mean()
    return acc




# def T3A_eval(S_orig, base_model, dataset):
    
#     correct = []

#     NORM = ((0.4913, 0.4821 ,0.4465), (0.2470, 0.2434, 0.261))
#     te_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), transforms.Normalize(*NORM)])
#     # print("Start point:")
#     # Set_num(S_orig)

#     for id in range(9000, 10000):

#         _, labels = dataset[id]
#         image = Image.fromarray(dataset.data[id])
#         inputs = te_transforms(image).unsqueeze(0)
#         inputs = inputs.cuda()

#         S_copy = copy.deepcopy(S_orig)
#         #Set_num(S_copy)
#         S = Set_adapt(S_copy, base_model, image)
#         #print("ID:", id)
#         #Set_num(S)

#         z = getEmbedding(base_model, inputs)
#         predicted = T3A_predictor(S, z)
#         if labels == predicted:
#             correct.append(1)
#         else:
#             correct.append(0)
#     acc = sum(correct)/len(correct)

#     return acc

def T3A_eval(S_orig, base_model, dataset):
    
    correct = []

    for batch_idx, (inputs, labels) in enumerate(dataset):
        inputs, labels = inputs.cuda(), labels.cuda()
        S_copy = copy.deepcopy(S_orig)
        S = Set_adapt(S_copy, base_model, inputs)
        z = getEmbedding(base_model, inputs)
        predicted = T3A_predictor(S, z)
        if labels == predicted:
            correct.append(1)
        else:
            correct.append(0)

    acc = sum(correct)/len(correct)
    return acc
