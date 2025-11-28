import torch
import platform
from pathlib import Path
import numpy as np
import imageio
from tqdm.auto import tqdm

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


def make_mp4(moviename, fnames, fps, do_delete=True):
    """Create an MP4 video from a sequence of image files using pathlib paths.

    Args:
        moviename (str | Path): Path to the output video file.
        fnames (Iterable[str | Path]): Iterable of input image file paths.
        fps (int): Frames per second for the output video.
        do_delete (bool): Whether to delete the input images after making the video.

    Returns:
    Path: Path to the created video file.
    """

    moviename = Path(moviename)
    fnames = [Path(f) for f in fnames]  # materialize and convert to Path
    moviename.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(moviename, fps=fps, macro_block_size=1) as writer:
        for fname in tqdm(fnames, desc="Creating video"):
            writer.append_data(imageio.v2.imread(fname))

    if do_delete:
        for fname in fnames:
            try:
                fname.unlink()
            except Exception as e:
                print('Could not unlink', fname, ' error', e)

    return moviename

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results, model_names, datasets, do_masks=[True, False], 
                          fig=None, axes=None,
                          figures_folder=None, save_name=None, exts=['pdf']):
    """
    Plot model comparison: inference time vs accuracy and model size vs accuracy.
    
    Parameters
    ----------
    results : DataFrame
        Results dataframe with columns: dataset, do_mask, accuracy, wall_clock_time, 
        total_parameters, model_name
    model_names : list
        List of model names to plot
    datasets : list
        List of dataset names to plot
    figures_folder : str, optional
        Folder to save figures
    save_name : str
        Name for saved figure
    """
    
    # Sort results
    results_sorted = results.sort_values(['dataset', 'do_mask', 'accuracy'])
    
    # Setup styles
    linestyles = {True: '--', False: '-'}
    markers = {name: marker for name, marker in zip(model_names, 
                                                     ['o', 's', '^', 'D', 'v', 'p', '*', 'H'][:len(model_names)])}
    palette = sns.color_palette("husl", len(datasets))
    dataset_colors = {dataset: palette[i] for i, dataset in enumerate(datasets)}
    
    if fig is None: fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    
    # Plot both subplots with same logic
    plot_configs = [
        (axes[0], 'wall_clock_time', 'Wall Clock Time (s/image)', 'Inference Time vs Accuracy'),
        (axes[1], 'total_parameters', 'Total Parameters', 'Model Size vs Accuracy')
    ]
    
    for ax, y_col, y_label, title in plot_configs:
        for dataset in datasets:
            for do_mask in do_masks:
                data_subset = results_sorted[
                    (results_sorted['dataset'] == dataset) & 
                    (results_sorted['do_mask'] == do_mask)
                ].copy()
                
                if len(data_subset) == 0:
                    continue
                
                # Order by model_names
                data_subset['model_name'] = data_subset['model_name'].astype('category')
                data_subset['model_name'] = data_subset['model_name'].cat.set_categories(model_names)
                data_subset = data_subset.sort_values('model_name')
                
                # Plot line
                ax.plot(data_subset['accuracy'], 
                       data_subset[y_col],
                       linestyle=linestyles[do_mask],
                       color=dataset_colors[dataset],
                       linewidth=2.5)
                
                # Add markers
                for model_name in model_names:
                    model_data = data_subset[data_subset['model_name'] == model_name]
                    if len(model_data) > 0:
                        ax.scatter(model_data['accuracy'], 
                                  model_data[y_col],
                                  marker=markers[model_name],
                                  color=dataset_colors[dataset],
                                  s=100,
                                  zorder=5)
        
        ax.set_xlabel('Accuracy', fontsize=18)
        ax.set_xscale('logit')
        ax.set_ylabel(y_label, fontsize=18)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Create legend
    legend_elements = []
    
    # Datasets
    if len(datasets) > 1:
        legend_elements.append(Line2D([0], [0], color='none', label='Datasets:', lw=0))
        for dataset in datasets:
            legend_elements.append(Line2D([0], [0], color=dataset_colors[dataset], lw=2.5, label=f'  {dataset}'))
        
        legend_elements.append(Line2D([0], [0], color='none', label='', lw=0))
    
    # Masking
    legend_elements.append(Line2D([0], [0], color='none', label='Masking:', lw=0))
    legend_elements.append(Line2D([0], [0], color='black', linestyle='-', lw=2.5, label='  do_mask=False'))
    legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=2.5, label='  do_mask=True'))
    
    legend_elements.append(Line2D([0], [0], color='none', label='', lw=0))
    
    # Models
    legend_elements.append(Line2D([0], [0], color='none', label='Models:', lw=0))
    for model_name in model_names:
        legend_elements.append(Line2D([0], [0], marker=markers[model_name], color='black', 
                                     linestyle='none', markersize=8, label=f'  {model_name}'))
    
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(.9, 0.5), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    if not(save_name is None):
        savefig(fig, name=save_name, figures_folder=figures_folder, exts=exts)
    
    plt.show()
    return fig, axes

def savefig(fig, name, exts=['pdf', 'png'], figures_folder=Path('figures'), opts_savefig = dict(bbox_inches='tight', pad_inches=0.1, edgecolor=None)):
    for ext in exts:
        fig.savefig(figures_folder / f'{name}.{ext}', **opts_savefig)

