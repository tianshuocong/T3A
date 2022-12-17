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
    level = 3

    te_set = prepare_test_data(corruption, level)

    acc_base = test_all(base_model, te_set)
    print(f"Base Acc: {100*acc_base:.2f}")

    fc_weight = base_model.state_dict()['fc.weight']   ## ResNet ## [10, 512]

    S = {}
    for i in range(10):
        S[i] = my_norm(fc_weight[i]).unsqueeze(0)  # [1,512]

    correct = []

    M_dist = np.zeros((10,10))
    print(M_dist)

    for id in range(1000):

        image = Image.fromarray(te_set.data[id])
        S,M = Set_adapt(S, base_model, image)

    print(M)

    Set_num(S)

    acc = T3A_eval(S, base_model, te_set)
    print(f"{acc*100:.2f}")


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
