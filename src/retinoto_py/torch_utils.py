"""

Useful torch snippets to use in the main module.

"""

#############################################################
import numpy as np
# https://github.com/laurentperrinet/2024-12-09-normalizing-images-in-convolutional-neural-networks
im_mean = np.array([0.485, 0.456, 0.406])
im_std = np.array([0.229, 0.224, 0.225]) 
#############################################################
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
# import warnings
# warnings.filterwarnings(
#     "ignore",
#     message=r"iCCP: profile",
#     category=UserWarning,
# )


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
    image = transform(image.unsqueeze(0))
    return image.squeeze(0)

def fixate(image, h, w, box_size, padding_mode='reflect'):
    three, H, W = image.shape
    assert three == 3
    assert 0 <= h < H
    assert 0 <= w < W
    # assert box_size <= H
    # assert box_size <= W

    radius_minus, radius_plus = box_size//2, box_size-box_size//2

    h_min, h_max = max((0, h-radius_minus)), min((h+radius_plus, H))
    w_min, w_max = max((0, w-radius_minus)), min((w+radius_plus, W))
    box = image[:, h_min:h_max, w_min:w_max]

    # Calcul du padding nécessaire pour atteindre (box_size, box_size)
    current_height = h_max - h_min
    current_width = w_max - w_min

    # Padding à gauche/droite et haut/bas
    pad_left = max(radius_minus - (w - w_min), 0)
    pad_right = max(radius_plus - (w_max - w), 0)
    pad_top = max(radius_minus - (h - h_min), 0)
    pad_bottom = max(radius_plus - (h_max - h), 0)

    # Correction pour garantir box_size x box_size
    total_pad_width = box_size - current_width
    total_pad_height = box_size - current_height

    # Répartition du padding
    pad_left = max(pad_left, 0)
    pad_right = max(total_pad_width - pad_left, 0)
    pad_top = max(pad_top, 0)
    pad_bottom = max(total_pad_height - pad_top, 0)

    # padding for the left, top, right and bottom borders
    transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), padding_mode=padding_mode)
    box_padded = transform(box)

    # Vérification de la taille
    assert box_padded.shape[1:] == (box_size, box_size), f"Expected {(box_size, box_size)}, got {box_padded.shape[1:]}"

    return box_padded    

# Prefer direct module import to avoid static analysis issues in some environments
def get_grid(args):
    """
    Generate a grid for the log-polar mapping

    """
    rs_ = torch.logspace(args.rs_min, args.rs_max, args.grid_size, base=2) # Radial distances (log scale)
    # adds a margin in angles in order to get an overrepresentation
    ts_ = torch.linspace(args.angle_start, args.angle_start+torch.pi*2+args.angle_margin, args.grid_size+1)[:-1] 
    grid_xs = torch.outer(rs_, torch.cos(ts_)) # X-coordinates
    grid_ys = torch.outer(rs_, torch.sin(ts_)) # Y-coordinates	
    
    return torch.stack((grid_xs, grid_ys), 2) # (H_scaled, W_scaled, 2)


def get_grid_hexagonal(args):
    """
    Generate a hexagonally-packed grid for the log-polar mapping
    with staggered rows for better sampling efficiency.

    This creates a pattern similar to hexagonal packing where every
    second eccentricity row is shifted by half the angular resolution.
    
    Args:
        args: Parameters object containing grid configuration
        
    Returns:
        torch.Tensor: Grid tensor of shape (grid_size, grid_size, 2) containing (x,y) coordinates
    """
    rs_ = torch.logspace(args.rs_min, args.rs_max, args.grid_size, base=2)  # Radial distances
    angular_resolution = 2 * torch.pi / args.grid_size  # Base angular step

    # Create staggered angular coordinates
    grid_xs_list = []
    grid_ys_list = []
    for i, r in enumerate(rs_):
        # Calculate phase shift: alternate between 0 and half angular resolution
        phase_shift = 0 if i % 2 == 0 else angular_resolution / 2

        # Create angular coordinates for this radial ring
        ts_i = torch.linspace(args.angle_start + phase_shift,
                              args.angle_start + 2*torch.pi + args.angle_margin + phase_shift,
                              args.grid_size + 1)[:-1]

        # Convert to Cartesian coordinates
        grid_xs_i = r * torch.cos(ts_i)
        grid_ys_i = r * torch.sin(ts_i)

        grid_xs_list.append(grid_xs_i)
        grid_ys_list.append(grid_ys_i)

    # Stack all rings together
    grid_xs = torch.stack(grid_xs_list, dim=0)
    grid_ys = torch.stack(grid_ys_list, dim=0)

    return torch.stack((grid_xs, grid_ys), 2)  # Shape: (grid_size, grid_size, 2)

class transform_apply_grid(object): 
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    def __init__(self, logPolar_grid, padding_mode, mode):
        self.grid = logPolar_grid
        self.padding_mode = padding_mode
        self.mode = mode

    def __call__(self, images):
        result =  nnf.grid_sample(images.unsqueeze(dim=0), 
                                  self.grid.unsqueeze(dim=0), 
                                  padding_mode=self.padding_mode, align_corners=True, 
                                  mode=self.mode)
        return result.squeeze(0)


def get_preprocess(args, do_full_preprocess=True, angle_min=None, angle_max=None, 
                   interpolation=InterpolationMode.BILINEAR, mode='bilinear', do_augment=None):
    """
    Defines get_preprocess for the preprocessing 
    
    :param args: A containaer for all parameters
    :param do_full_preprocess: set to FAlse to bypass the full preprocessing and use that for getting a dataloader providing raw images
    :param angle_min: Description
    :param angle_max: Description
    :param interpolation: Description
    :param mode: Description
    :param do_augment: Description
    """


    if do_augment is None: do_augment = args.do_augment # sets to dafault value

    # The images must be pre-processed in the exact same way the model was trained on.
    # This includes resizing, cropping, and normalizing.
    transform_list = []
 
    transform_list.append(transforms.ToImage())  
    transform_list.append(transforms.ToDtype(torch.float32, scale=True)) 


    if do_full_preprocess:

        if do_augment:
            # transform_list.append(transforms.RandomHorizontalFlip())

            # ----- Augmentations spécifiques au training -----
            # 1. RandAugment
            # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandAugment.html#torchvision.transforms.v2.RandAugment
            # --------------------------------------------------------------
            #  Bloc d’augmentations « full preprocess » (activé seulement
            #  pendant l’entraînement).  Toutes les opérations ci‑dessous
            #  conservent la résolution de l’image (pas de crop ou de zoom)
            #  puisqu’on a déjà fixé la taille avec `Resize` plus haut.
            # --------------------------------------------------------------

            # 1️⃣ RandAugment – applique N opérations de façon aléatoire
            #    parmi le catalogue torchvision (rotation, shear, contrast, …)
            #    * num_ops=2  →  deux opérations sont composées pour chaque image
            #    * magnitude=9 → intensité élevée (0‑30 ° de rotation, forte
            #      contraste, etc.) mais toujours dans les limites du même shape
            #    * interpolation=interpolation → on passe le même mode d’interpolation
            #      qui a été utilisé pour les éventuelles rotations précédentes
            transform_list.append(
                transforms.RandAugment(
                    num_ops=2,
                    magnitude=args.augment_magnitude,
                    interpolation=interpolation
                )
            )

        # Si les deux angles ne sont pas None, on applique la rotation
        if angle_min is not None and angle_max is not None:
            transform_list.append(transforms.RandomRotation(degrees=(angle_min, angle_max), interpolation=interpolation))

        if args.do_fovea: # apply log-polar mapping to the image
            # Choose between regular or hexagonal grid
            # Priority: explicit parameter > args setting > default (False)
            if args.use_hexagonal_grid:
                grid_polar = get_grid_hexagonal(args)
            else:
                grid_polar = get_grid(args)
            # grid_polar = grid_polar.to(args.device)
            transform_list.append(transform_apply_grid(grid_polar, padding_mode=args.padding_mode, mode=mode))
        else:
            # transform_list.append(PadAndResize(args.image_size, interpolation=interpolation))
            transform_list.append(transforms.Resize(args.image_size, interpolation=interpolation, antialias=True))
            transform_list.append(transforms.CenterCrop((args.image_size, args.image_size)))

        # 3. Add ColorJitter BEFORE RandomGrayscale
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )
        )
        # 3️⃣ RandomGrayscale – avec probabilité 0.2
        #    - transforme l’image en niveaux de gris (R=G=B) pour forcer le
        #      réseau à ne pas dépendre uniquement de la chrominance.
        transform_list.append(transforms.RandomGrayscale(p=args.augment_proba))


        transform_list.append(transforms.Normalize(mean=im_mean, std=im_std))

        if do_augment:
            # RandomErasing should be applied AFTER normalization
            transform_list.append(
                transforms.RandomErasing(
                    p=args.augment_proba,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3)
                )
            )
        if args.do_mask:
            if args.do_fovea: raise(BaseException, 'Something is wrong here')
            # Créer le masque une seule fois avec la taille de l'image
            mask = make_mask(image_size=args.image_size)#.to(args.device)
            # Ajouter notre transform personnalisée à la liste
            transform_list.append(ApplyMask(mask))

    # Créer la chaîne de prétraitement finale
    preprocess = transforms.Compose(transform_list)
    return preprocess

def _core_dataset(ds):
    """Retourne le dataset le plus interne qui possède les attributs classiques."""
    while isinstance(ds, (torch.utils.data.Subset,
                         torch.utils.data.ConcatDataset,
                         torch.utils.data.WeightedRandomSampler)):
        # Subset possède l’attribut .dataset ; les autres variantes sont gérées de façon similaire
        ds = ds.dataset
    return ds

is_valid_file = lambda p: p.lower().endswith(('.png', '.jpg', '.jpeg'))

def get_dataset(args, DATA_DIR, do_full_preprocess=True, angle_min=None, angle_max=None, do_augment=None):
    
    # defines preprocessing from the raw image to the input to the network
    preprocess = get_preprocess(args, do_full_preprocess=do_full_preprocess, angle_min=angle_min, angle_max=angle_max, do_augment=do_augment)
    
    # --- 2. Create Dataset and DataLoader using ImageFolder ---
    # ImageFolder automatically infers class names from directory names
    # and maps them to integer indices.
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=preprocess, is_valid_file=is_valid_file)

    # -----------------------------------------------------------------
    # Sous‑échantillonnage (debug) – on crée un Subset
    # -----------------------------------------------------------------
    if args.subset_factor > 1:
        # choisir aléatoirement les indices à garder
        subset_indices = np.random.choice(len(dataset), size=len(dataset)//args.subset_factor,
                                         replace=False)
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    # -----------------------------------------------------------------
    # Récupération robuste des métadonnées (class_to_idx, classes, targets)
    # -----------------------------------------------------------------
    core = _core_dataset(dataset)   # le dataset réel derrière le Subset éventuel

    # Copie des dictionnaires de classe → indice
    dataset.class_to_idx = getattr(core, "class_to_idx", {})
    dataset.classes      = getattr(core, "classes", [])

    # Si le dataset possède .targets (ImageFolder) on les reconstruit pour le Subset
    if hasattr(core, "targets"):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset.targets = [core.targets[i] for i in subset_indices]
        else:
            dataset.targets = core.targets

    dataset.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    dataset.idx2label   = get_idx_to_label(args)
    dataset.label2idx   = get_label_to_idx(args)

    return dataset

def get_loader(args, dataset, drop_last=True, seed=None):

    if seed is None: seed = args.seed
        
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(seed),  # deterministic shuffling
        pin_memory=False, # unified memory → no need for pinned host memory
        persistent_workers=False, # workers are spawned each epoch (safer for transform changes)
        prefetch_factor= None if args.prefetch_factor==0 else args.prefetch_factor,  # a small pre‑fetch queue is enough on M‑series
    )
    return loader

def load_model(args, model_filename=None):
    """
    Load the model from the torchvision library.
    
    """
    import os, torchvision.models as models
    os.environ.setdefault('TORCH_HOME', str(args.data_cache))

    opts_convnext = dict(stochastic_depth_prob=args.stochastic_depth_prob)

    if args.model_name=='resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.model_name=='resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif args.model_name=='resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif args.model_name=='convnext_tiny':
        # https://github.com/facebookresearch/ConvNeXt/
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, **opts_convnext)
    elif args.model_name=='convnext_small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1, **opts_convnext)
    elif args.model_name=='convnext_base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1, **opts_convnext)
    elif args.model_name=='convnext_large':
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1, **opts_convnext)
    else:
        raise ValueError(f'Unknown model {args.model_name}')
    
    model = model.to(args.device)

    # if args.model_name=='convnext_base': # HACK
    #     model_filename_fb = args.data_cache /  'convnext_base_1k_224_ema.pth'  # Remplacez par le chemin réel
    #     model = apply_weights(model, model_filename, args.device, verbose=args.verbose)

    if model_filename is not None:
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
    try:
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model.load_state_dict(torch.load(model_filename, map_location=torch.device(device)), strict=True, weights_only=False)
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

def imgs_to_np(img_list, im_mean=im_mean, im_std=im_std, nrow=11):
    images = torchvision.utils.make_grid(img_list, nrow=nrow)
    inp = images.numpy().transpose((1, 2, 0))
    inp = im_std * inp + im_mean
    inp = np.clip(inp, 0, 1)
    return(inp)

from .utils import savefig
def imshow(img_list, nrow=11, im_mean=im_mean, im_std=im_std, 
           title=None, fig_height=7., fig=None, ax=None, 
           fontsize=14):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_height*len(img_list), fig_height))

    inp = imgs_to_np(img_list, im_mean=im_mean, im_std=im_std, nrow=nrow)
    ax.imshow(inp)
    ax.set_xticks([])
    ax.set_yticks([])
    if title != None: fig.suptitle(title, fontsize=fontsize)
    fig.set_facecolor(color='white')
    #plt.tight_layout()
    return fig, ax