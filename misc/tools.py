import torch
import torch.nn as nn
import numpy as np
import random
import os 


def setseed(manualSeed) :
    # manualSeed = 1
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    # torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def my_makedir(name):
	try:
		os.makedirs(name)
	except OSError:
		pass