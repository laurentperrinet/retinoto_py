"""Top-level package for Foveated Retinotopy."""

__author__ = """Laurent U Perrinet"""
__email__ = 'laurent.perrinet@cnrs.fr'


#####################################################
from .utils import Params
#############################################################
# Importing libraries
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


