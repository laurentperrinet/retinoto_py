"""Top-level package for Foveated Retinotopy."""

__author__ = """Laurent U Perrinet"""
__email__ = 'laurent.perrinet@cnrs.fr'

all_models = ['resnet18', 'resnet50', 'resnet101'] 
all_datasets = ['full', 'bbox']

#####################################################
from .params import Params
from .utils import get_device, set_seed
from .torch_utils import imshow, get_idx_to_label, get_loader, imgs_to_np
from .torch_utils import load_model, count_parameters, count_layers
from .retinoto_py import get_validation_accuracy, train_model
#############################################################
# Importing libraries

import time


import pandas as pd # to store results
import torch
import torch.nn.functional as nnf
import torchvision
from torchvision.io import read_image
# https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html
from torchvision.transforms.functional import crop
# from torchvision import datasets, models, transforms
# from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
import torch.nn as nn
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)

import seaborn as sns
import matplotlib.pyplot as plt
