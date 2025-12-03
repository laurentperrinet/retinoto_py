"""Top-level package for Foveated Retinotopy."""

__author__ = """Laurent U Perrinet"""
__email__ = 'laurent.perrinet@cnrs.fr'

all_model_names = ['resnet18', 'resnet50', 'resnet101'] 
all_model_names_ls = [':', '-.', '-'] 
all_model_names_color = ['blue', 'blue', 'blue']
all_cn_model_names = ['convnext_tiny', 'convnext_base', 'convnext_large'] #'convnext_small', 
all_cn_model_names_color = ['blue', 'blue', 'blue']
all_cn_model_names_ls = [':', '-.', '-'] 
all_datasets = ['full', 'bbox']
all_datasets_color = ['blue', 'orange']
all_datasets_ls = ['-', '-']


#####################################################
from .params import Params
from .utils import get_device, set_seed, savefig, make_mp4, plot_model_comparison
from .torch_utils import imshow, get_idx_to_label, get_label_to_idx, get_loader, get_dataset, imgs_to_np
from .torch_utils import load_model, count_parameters, count_layers, apply_weights
from .torch_utils import make_mask, get_preprocess, TF
from .retinoto_py import get_validation_accuracy, train_model, do_learning
from .retinoto_py import compute_likelihood_map
#############################################################
# Importing libraries
import warnings
warnings.filterwarnings("ignore", message=".*application/vnd.jupyter.widget-view+json.*")
from tqdm.auto import tqdm

import time
import numpy as np

# from PIL import Image
# from PIL import ImageFile
# ImageFile.MAX_TEXT_CHUNK = 10 * 1024 * 1024   # 10 MiB (choose a value > largest chunk)

from PIL import Image, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024   # 10 MiB (choose a value > largest chunk)

import pandas as pd # to store results
import torch

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
import seaborn as sns
# https://seaborn.pydata.org/generated/seaborn.set_theme.html
# https://seaborn.pydata.org/tutorial/color_palettes.html
sns.set_theme(style="whitegrid")
# sns.despine(offset=10, trim=True);
# sns.set_context("talk")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#############################################################

