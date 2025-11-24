import torch
import platform
from pathlib import Path
import numpy as np

#############################################################

def get_device(verbose):

    if verbose: 
        print('Welcome on', platform.platform(), end='\t')
        print(f" user {Path.home().owner() if hasattr(Path.home(), 'owner') else Path.home().name}", end='\t')


    if torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print('Running on MPS device (Apple Silicon/MacOS)', end='\t')
            try:
                if hasattr(torch.backends.mps, 'metal_version'):
                    print(f' - metal_version = {torch.backends.mps.metal_version()}', end='\t')
            except:
                pass
            print(f' - macos_version = {platform.mac_ver()[0]}', end='\t')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose: print('Running on GPU : ', torch.cuda.get_device_name(), '#GPU=', torch.cuda.device_count(), end='\t')
        torch.cuda.empty_cache()
        print_gpu_memory()
    else:
        device = torch.device('cpu')

    if verbose: 
        print(f' with device {device}, pytorch=={torch.__version__}')
    return device


# set seed function
def set_seed(seed=None, seed_torch=True, verbose=False):
    "Define a random seed or use a predefined seed for repeatability"
    if seed is None:
        seed = np.random.choice(2 ** 32)

    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)

    if verbose: print(f'Random seed {seed} has been set.')    

def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", end='\t')
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB", end='\t')
    print(f">> Scanning variables occupying GPU memory...", end='\t')
    for var_name, var in globals().items():
        if torch.is_tensor(var) and var.is_cuda:
            print(f"{var_name}: {var.element_size() * var.nelement() / 1024**2:.2f} MB", end='\t')


def savefig(fig, name:str, exts:list=['pdf', 'png'], figures_folder=Path('figures'), opts_savefig = dict(    bbox_inches='tight', pad_inches=0.1, edgecolor=None)):
    for ext in exts: 
        fig.savefig(figures_folder / f'{name}.{ext}', **opts_savefig)

