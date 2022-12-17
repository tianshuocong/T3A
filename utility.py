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

from misc.t3a import *
from misc.load_model import *
from misc.load_dataset import *
from misc.evaluation import *

logger = logging.getLogger(__name__)


def evaluate(description):


    base_model = load_target()
    corruption = "glass_blur"
    level = 5

    loader1, loader2 = prepare_test_data(corruption, level)

    acc_base = test_all_loader(base_model, loader2)
    print(f"Base Acc: {100*acc_base:.2f}")

    fc_weight = base_model.state_dict()['fc.weight']   ## ResNet ## [10, 512]

    S = {}
    for i in range(10):
        S[i] = my_norm(fc_weight[i]).unsqueeze(0)  # [1,512]

    correct = []

    for idx, (inputs, labels) in enumerate(loader1):
        if idx < 1000:
            inputs, labels = inputs.cuda(), labels.cuda()
            Set_adapt(S, base_model, inputs)

    Set_num(S)

    acc = T3A_eval(S, base_model, loader2)
    print(f"{acc*100:.2f}")


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
