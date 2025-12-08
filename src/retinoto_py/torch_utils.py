"""

Useful torch snippets to use in the main module.

"""

#############################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torchvision
import torch
from torchvision import datasets
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
import torch.nn.functional as nnf
# https://pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html
# from torchvision.transforms.functional import crop
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as TF
# from torchvision import datasets, models, transforms
# from torchvision.datasets import ImageFolder
import torch.nn as nn
from torchvision.transforms import InterpolationMode
from .utils import set_seed
import random
#############################################################
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"iCCP: profile",
    category=UserWarning,
)


def get_idx_to_label(args, verbose=False):
    ##############
    LABELS_FILE = args.data_cache / 'imagenet_class_index.json' # Local cache file name

    try:
        import json # Don't forget to import json
        # Check if we already have the file
        if not LABELS_FILE.exists():
            import requests

            # --- 4. Download and Load the ImageNet Class Index (with caching) ---
            LABELS_URL = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

            if verbose: print(f"Downloading labels to {LABELS_FILE}...")
            response = requests.get(LABELS_URL)
            response.raise_for_status()
            with open(LABELS_FILE, 'w') as f:
                json.dump(response.json(), f)
        else:
            if verbose: print(f"Loading labels from local cache {LABELS_FILE}...")
            
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

def get_label_to_idx(args):
    idx2label = get_idx_to_label(args)  # Get the list of labels
    label2idx = {label: idx for idx, label in enumerate(idx2label)}
    return label2idx

# https://github.com/laurentperrinet/2024-12-09-normalizing-images-in-convolutional-neural-networks
im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225]) 

def get_smaller_balanced_dataset(dataset, subset_factor=10, seed=42):
    """
    Create a smaller, balanced subset of the dataset.

    Args:
        dataset: The full ImageFolder dataset.
        fraction: Fraction of the dataset to use (default: 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Subset of the dataset with balanced classes.
    """
    np.random.seed(seed)

    # Get all targets
    targets = np.array(dataset.targets)

    # Get unique classes and their indices
    classes, counts = np.unique(targets, return_counts=True)

    # Calculate the number of samples per class in the subset
    n_per_class = min(counts) // subset_factor

    # Sample indices for each class
    subset_indices = []
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        subset_indices.extend(cls_indices[:n_per_class])

    # Shuffle the subset indices
    np.random.shuffle(subset_indices)
    subset_indices = [int(idx) for idx in subset_indices]

    assert len(subset_indices) > 0, "Subset is empty!"
    assert max(subset_indices) < len(dataset), "Index out of bounds"

    return subset_indices

class InMemoryImageDataset(Dataset):
    """Load entire ImageFolder dataset into memory"""
    def __init__(self, dataset,  seed=None):
        self.images = []
        self.labels = []

        np.random.seed(seed)
        n_total = len(dataset)
        for idx in tqdm(range(n_total), desc='Putting images in memory', total=n_total, leave=False):
            self.images.append(dataset[idx][0])
            self.labels.append(dataset[idx][1])

        # print(f"Loaded {len(self.images)} images into memory")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        return img, label


# https://github.com/laurentperrinet/2024-12-09-normalizing-images-in-convolutional-neural-networks
im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225]) 

def make_mask(image_size: int, radius: float = 1.0):
    """
    Create a circular mask for the image.
    
    image_size: int, size of the image (height and width)
    radius: float, radius of the circle (0.5 means half the image size)"""
    
    X, Y = np.meshgrid(np.linspace(-1, 1, image_size), # Coordonnées normalisées de -1 à 1
                       np.linspace(-1, 1, image_size),
                       indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    mask = (R <= radius).astype(np.float32) # 1.0 pour un cercle complet
    return torch.from_numpy(mask).unsqueeze(0) # Ajoute la dimension du canal

class ApplyMask(object):
    """Applique un masque circulaire à un tenseur d'image."""
    def __init__(self, mask: torch.Tensor):
        # On stocke le masque. Le .clone() est une bonne pratique pour éviter
        # des modifications inattendues du masque original.
        self.mask = mask.clone()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applique le masque à un tenseur d'image.
        Args:
            tensor (torch.Tensor): Tenseur d'image de forme (C, H, W).
        Returns:
            torch.Tensor: Tenseur masqué.
        """
        return tensor * self.mask

def squarify(image):
    """
    Takes an image and pad it to make it square
    
    """
    three, H, W = image.shape
    assert three == 3

    square_image_size = max(H, W)
    pad_height = (square_image_size - H) // 2
    pad_width = (square_image_size - W) // 2

    # If a sequence of length 4 is provided
    #     this is the padding for the left, top, right and bottom borders respectively.
    transform = transforms.Pad((pad_width, 
                                pad_height, 
                                square_image_size - W - pad_width, 
                                square_image_size - H - pad_height), padding_mode='reflect')
    image = transform(image)     
    return image.squeeze(0)

# Prefer direct module import to avoid static analysis issues in some environments
def get_grid(args, angle_start=0, endpoint=False):
    """
    Generate a grid for the log-polar mapping
    """

    rs_ = torch.logspace(args.rs_min, args.rs_max, args.image_size, base=2) # Radial distances (log scale)
    # adds a margin in angles in order to get an overrepresentation
    ts_ = torch.linspace(args.angle_start, args.angle_start+torch.pi*2+args.angle_margin, args.image_size+1)[:-1] 
    grid_xs = torch.outer(rs_, torch.cos(ts_)) # X-coordinates
    grid_ys = torch.outer(rs_, torch.sin(ts_)) # Y-coordinates	
    
    return torch.stack((grid_xs, grid_ys), 2) # (H_scaled, W_scaled, 2)

class transform_apply_grid(object): 
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    def __init__(self, logPolar_grid, padding_mode, mode):
        self.grid = logPolar_grid
        self.padding_mode = padding_mode
        self.mode = mode

    def __call__(self, images):
        result =  nnf.grid_sample(images.unsqueeze(dim=0), 
                                    self.grid.unsqueeze(dim=0), 
                                    padding_mode=self.padding_mode, align_corners=False, 
                                    mode=self.mode)
        return result.squeeze(0)


def get_preprocess(args, do_full_preprocess=True, angle_min=None, angle_max=None, 
                   interpolation=InterpolationMode.BILINEAR, mode='bilinear'):
    # --- 5. Define Image Pre-processing ---
    # The images must be pre-processed in the exact same way the model was trained on.
    # This includes resizing, cropping, and normalizing.
    transform_list = []
 
    transform_list.append(transforms.ToImage())  
    transform_list.append(transforms.ToDtype(torch.float32, scale=True)) 
    if do_full_preprocess:
        # Si les deux angles ne sont pas None, on applique la rotation
        if angle_min is not None and angle_max is not None:
            transform_list.append(transforms.RandomRotation(degrees=(angle_min, angle_max), interpolation=interpolation))
        transform_list.append(transforms.RandomHorizontalFlip())

        if args.do_fovea: # apply log-polar mapping to the image
            grid_polar = get_grid(args)
            transform_list.append(transform_apply_grid(grid_polar, padding_mode=args.padding_mode, mode=mode))
        else:
            # transform_list.append(PadAndResize(args.image_size, interpolation=interpolation))
            transform_list.append(transforms.Resize(args.image_size, interpolation=interpolation, antialias=True))
            transform_list.append(transforms.CenterCrop((args.image_size, args.image_size)))

        transform_list.append(transforms.Normalize(mean=im_mean, std=im_std))

        if args.do_mask:
            if args.do_fovea: raise(BaseException, 'Something is wrong here')
            # Créer le masque une seule fois avec la taille de l'image
            mask = make_mask(image_size=args.image_size)
            # Ajouter notre transform personnalisée à la liste
            transform_list.append(ApplyMask(mask))

    # Créer la chaîne de prétraitement finale
    preprocess = transforms.Compose(transform_list)
    return preprocess

def get_dataset(args, DATA_DIR, do_full_preprocess=True, angle_min=None, angle_max=None, in_memory=None):
    preprocess = get_preprocess(args, do_full_preprocess=do_full_preprocess, angle_min=angle_min, angle_max=angle_max)
    
    is_valid_file = lambda p: p.lower().endswith(('.png', '.jpg', '.jpeg'))
    # --- 2. Create Dataset and DataLoader using ImageFolder ---
    # ImageFolder automatically infers class names from directory names
    # and maps them to integer indices.
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=preprocess, is_valid_file=is_valid_file)
    if args.subset_factor > 1:
        subset_indices = get_smaller_balanced_dataset(dataset, subset_factor=args.subset_factor, seed=args.seed)
        dataset = Subset(dataset, subset_indices)

    if in_memory is None: in_memory = args.in_memory
    if in_memory:
        dataset = InMemoryImageDataset(dataset, is_valid_file=is_valid_file, seed=args.seed)

    if args.subset_factor > 1:
        dataset.class_to_idx = dataset.dataset.class_to_idx
        dataset.classes = dataset.dataset.classes
        dataset.targets = [dataset.dataset.targets[i] for i in subset_indices]

    dataset.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    dataset.idx2label = get_idx_to_label(args)
    dataset.label2idx = get_label_to_idx(args)

    return dataset


def get_loader(args, dataset, drop_last=True, seed=None):

    if seed is None:
        seed = getattr(args, "seed", 0)
        
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        # worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed),  # deterministic shuffling
        # pin_memory=False,               # unified memory → no need for pinned host memory
        # persistent_workers=False,       # workers are spawned each epoch (safer for transform changes)
        # prefetch_factor=2,              # a small pre‑fetch queue is enough on M‑series
    )
    return loader

import torchvision.models as models
def load_model(args, model_filename=None):
    """
    Load the model from the torchvision library.
    
    """

    if args.model_name=='resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.model_name=='resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif args.model_name=='resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif args.model_name=='convnext_tiny':
        # https://github.com/facebookresearch/ConvNeXt/
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif args.model_name=='convnext_small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    elif args.model_name=='convnext_base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif args.model_name=='convnext_large':
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f'Unknown model {args.model_name}')
    
    model = model.to(args.device)

    # if args.model_name=='convnext_base': # HACK
    #     model_filename_fb = args.data_cache /  'convnext_base_1k_224_ema.pth'  # Remplacez par le chemin réel
    #     model = apply_weights(model, model_filename, args.device, verbose=args.verbose)

    if not model_filename is None:
        model = apply_weights(model, model_filename, args.device, verbose=args.verbose)

    return model

def apply_weights(model, model_filename, device, verbose=True):
    """
    Apply the weights to the model.
    Args:
        model: torch model, the model to apply the weights to
        model_filename: str, path to the weights file
        verbose: bool, whether to print the loading message or not
    Returns:    
        model: torch model, the model with the weights applied
        """
    if verbose: print(f'loading .... {model_filename}')
    model.load_state_dict(torch.load(model_filename, map_location=torch.device(device)), strict=True)
    return model


def count_parameters(model):
    """Counts the total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }

def count_layers(model, layer_type=None):
    """
    Counts the layers in a PyTorch model.
    If `layer_type` is None, counts all nn.Module children.
    If `layer_type` is specified (e.g., nn.Conv2d), counts only those.
    """
    if layer_type is None:
        return sum(1 for _ in model.modules())
    else:
        return sum(1 for module in model.modules() if isinstance(module, layer_type))

#############################################################

def imgs_to_np(img_list, im_mean:np.array=im_mean, im_std:np.array=im_std):
    images = torchvision.utils.make_grid(img_list, nrow=11)
    """Imshow for Tensor."""
    inp = images.numpy().transpose((1, 2, 0))
    inp = im_std * inp + im_mean
    inp = np.clip(inp, 0, 1)
    return(inp)

from .utils import savefig
def imshow(img_list, im_mean:np.array=im_mean, im_std:np.array=im_std, 
           title:str=None, fig_height:float=7., fig=None, ax:matplotlib.axes.Axes=None, 
           figures_folder:str='figures', fontsize=14, save:bool=False, name:str=None, 
           dpi = 'figure', exts:list=['pdf', 'png']):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_height*len(img_list), fig_height))

    inp = imgs_to_np(img_list, im_mean=im_mean, im_std=im_std)
    ax.imshow(inp)
    ax.set_xticks([])
    ax.set_yticks([])
    if title != None: 
        fig.suptitle(title, fontsize=fontsize)
    fig.set_facecolor(color='white')
    #plt.tight_layout()

    if save:
        savefig(fig=fig, name=name, exts=exts, dpi=dpi, figures_folder=figures_folder)
    else:
        return fig, ax