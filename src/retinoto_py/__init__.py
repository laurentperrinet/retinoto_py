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


def get_idx_to_label(args):
    ##############
    LABELS_FILE = args.data_cache / 'imagenet_class_index.json' # Local cache file name

    try:
        import json # Don't forget to import json
        # Check if we already have the file
        if not LABELS_FILE.exists():
            import requests

            # --- 4. Download and Load the ImageNet Class Index (with caching) ---
            LABELS_URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

            print(f"Downloading labels to {LABELS_FILE}...")
            response = requests.get(LABELS_URL)
            response.raise_for_status()
            with open(LABELS_FILE, 'w') as f:
                json.dump(response.json(), f)
        else:
            print(f"Loading labels from local cache {LABELS_FILE}...")
            
        # In both cases, load from the local file
        with open(LABELS_FILE, 'r') as f:
            class_idx = json.load(f)

        # Create a simple mapping from index to class name for easy lookup
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

        return idx2label

    except requests.exceptions.RequestException as e:
        print(f"Error downloading labels: {e}")
        exit()
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error handling local label file: {e}")
        exit()
    if args.verbose: print(f'Got a list with {len(idx2label)} labels in {LABELS_FILE} ')