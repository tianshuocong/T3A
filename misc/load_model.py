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




def load_target():
    path = "/home/c01tico/CISPA-projects/attacking_testtime_training_models-2022/SP-MR/"
    base_model = resnet18(pretrained=False).cuda()
    cp_base = torch.load(path+"bn-net-target/resnet18-train.ckpt", map_location=torch.device("cuda:0"))['state_dict']
    new_state_dict = OrderedDict()
    for key, value in cp_base.items():
        key = key[6:] # remove `model.`
        new_state_dict[key] = value
    base_model.load_state_dict(new_state_dict, strict=True)
    base_model.eval()
    base_model = base_model.cuda()
    return base_model
