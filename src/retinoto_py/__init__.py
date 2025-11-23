"""Top-level package for Foveated Retinotopy."""

__author__ = """Laurent U Perrinet"""
__email__ = 'laurent.perrinet@cnrs.fr'

all_model_names = ['resnet18', 'resnet50', 'resnet101'] 
all_datasets = ['full', 'bbox']

#####################################################
from .params import Params
from .utils import get_device, set_seed, savefig
from .torch_utils import imshow, get_idx_to_label, get_loader, imgs_to_np
from .torch_utils import load_model, count_parameters, count_layers
from .retinoto_py import get_validation_accuracy, train_model, make_mask
#############################################################
# Importing libraries
from tqdm.auto import tqdm

import time
import numpy as np

from PIL import Image

import pandas as pd # to store results
import torch
import torch.nn.functional as nnf
import torchvision
from torchvision.io import read_image
# https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html
# from torchvision.transforms.functional import crop
import torchvision.transforms.functional as TF
# from torchvision import datasets, models, transforms
# from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
import torch.nn as nn
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)

import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
import matplotlib.colors

# --- Define your defaults in a dictionary ---
plt.rcParams.update({
    'font.size': 14,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    
    # Figure settings
    'figure.dpi': 200,
    'figure.figsize': (12, 12/1.618039), # (width, height)
    'savefig.dpi': 'figure', #200,
    # 'savefig.bbox_inches': 'tight',
    # 'savefig.pad_inches': 0.1,
    # 'savefig.edgecolor': 'none',

    # Subplot spacing
    'figure.subplot.left': 0.125,
    'figure.subplot.right': 0.95,
    'figure.subplot.bottom': 0.25,
    'figure.subplot.top': 0.95,
    'figure.subplot.wspace': 0.05,
    'figure.subplot.hspace': 0.05,
})
opts_savefig = dict(
    bbox_inches='tight',
    pad_inches=0.1,
    edgecolor=None
)
import seaborn as sns
# https://seaborn.pydata.org/generated/seaborn.set_theme.html
# https://seaborn.pydata.org/tutorial/color_palettes.html
sns.set_theme(style="whitegrid")
# sns.despine(offset=10, trim=True);
# sns.set_context("talk")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#############################################################

