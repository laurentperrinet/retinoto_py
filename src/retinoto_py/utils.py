import torch
import platform
from pathlib import Path
import numpy as np
import imageio
from tqdm.auto import tqdm
from time import strftime, gmtime

#############################################################
def get_device(verbose):

    if verbose: 
        print('Welcome on', platform.platform(), '- Timestamp (UTC) ', strftime("%Y-%m-%d_%H-%M-%S", gmtime()), f" user {Path.home().owner() if hasattr(Path.home(), 'owner') else Path.home().name}", f'>  pytorch=={torch.__version__}')

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print(f'> device (Apple Silicon/MacOS) - macos_version = {platform.mac_ver()[0]}')

    elif torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose: print('Running on GPU : ', torch.cuda.get_device_name(), '#GPU=', torch.cuda.device_count())
        torch.cuda.empty_cache()
        print_gpu_memory()
    else:
        device = torch.device('cpu')

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

    with imageio.get_writer(moviename, fps=fps) as writer:
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
                                                     ['o', 's', 'D', '^', 'v', '>', 'p', '*', 'H', '+'][:len(model_names)])}
    palette = sns.color_palette("husl", len(datasets))
    dataset_colors = {dataset: palette[i] for i, dataset in enumerate(datasets)}
    
    if fig is None: fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    
    # Plot both subplots with same logic
    plot_configs = [
        (axes[0], 'total_parameters', 'Total Parameters', 'Model Size vs Accuracy'),
        (axes[1], 'wall_clock_time', 'Wall Clock Time (s/image)', 'Inference Time vs Accuracy'),
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
    
    # plt.show()
    return fig, axes

def savefig(fig, name, exts=['pdf', 'png'], figures_folder=Path('figures'), opts_savefig = dict(bbox_inches='tight', pad_inches=0.1, edgecolor=None)):
    for ext in exts:
        fig.savefig(figures_folder / f'{name}.{ext}', **opts_savefig)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
def visualize_likelihood_map(likelihood_map, sigma=0, color='blue', fig=None, axes=None):

    resolution = (likelihood_map.shape[0], likelihood_map.shape[1])
    # Lisser la heatmap
    if sigma >0: likelihood_map = gaussian_filter(likelihood_map, sigma=sigma)

    # Calcul des marginales
    marginal_H = likelihood_map.mean(axis=1)  # Marginale verticale
    marginal_W = likelihood_map.mean(axis=0)  # Marginale horizontale

    # Créer la figure et l'axe principal
    fig, ax = plt.subplots(figsize=(10, 8))

    # Afficher la heatmap
    contour = ax.contourf(likelihood_map, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label='Likelihood')
    # ax.set_title("Carte de probabilité de vraisemblance")
    ax.set_xlabel("Position (W)")
    ax.set_ylabel("Position (H)")

    # Ajouter une grille légère
    ax.grid(alpha=0.3, linestyle='--')

    # Ajouter les lignes centrales (verticale et horizontale)
    mid_w = likelihood_map.shape[1] / 2
    mid_h = likelihood_map.shape[0] / 2
    ax.axvline(x=mid_w, color='white', linestyle='--', linewidth=2)
    ax.axhline(y=mid_h, color='white', linestyle='--', linewidth=2)

    # Ajouter des axes pour les marginales
    divider = make_axes_locatable(ax)

    # Axe pour la marginale horizontale (en haut)
    ax_top = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    # ax_top.plot(np.linspace(-1, 1, resolution[1]), marginal_W, color='red')
    ax_top.plot(range(resolution[1]), marginal_W, color=color)
    # ax_top.set_title("Probabilité marginale (axe horizontal)")
    # ax_top.set_ylabel("Probabilité")
    ax_top.set_xticks([])  # Pas de labels sur l'axe x pour éviter la redondance

    # Axe pour la marginale verticale (à droite)
    ax_right = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    # ax_right.plot(marginal_H, np.linspace(-1, 1, resolution[0]), color='blue')
    ax_right.plot(marginal_H, range(resolution[0]), color=color)
    # ax_right.set_title("Probabilité marginale (axe vertical)")
    # ax_right.set_xlabel("Probabilité")
    ax_right.set_yticks([])  # Pas de labels sur l'axe y pour éviter la redondance

    plt.tight_layout()
    axes = (ax, ax_top, ax_right)
    return fig, axes


import numpy as np
import lmfit
from lmfit import Parameters, minimize, report_fit
import pandas as pd

def gaussian_2d_lmfit(params, x, y, data):
    """
    2D Gaussian function for lmfit.
    """
    amp = params['amplitude']
    x0 = params['x0']
    y0 = params['y0']
    sx = params['sigma_x']
    sy = params['sigma_y']
    theta = params['theta']

    a = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
    b = -(np.sin(2*theta))/(4*sx**2) + (np.sin(2*theta))/(4*sy**2)
    c = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)

    model = amp * np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return model - data  # Residuals for least-squares minimization

def fit_gaussian_lmfit(likelihood_map):
    """
    Fit a 2D Gaussian to a likelihood map using lmfit, constraining x0 and y0 to [-1, 1].
    Returns: amplitude, x0, y0, sigma_x, sigma_y, theta (rotation angle)
    """
    # Create a grid of coordinates
    ny, nx = likelihood_map.shape
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    x, y = np.meshgrid(x, y)

    # Flatten the likelihood map and coordinates
    data = likelihood_map.flatten()
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Create Parameters object with initial guesses and bounds
    params = lmfit.Parameters()
    params.add('amplitude', value=np.max(likelihood_map), min=0, max=1)  # Amplitude must be positive
    params.add('x0', value=0, min=-1, max=1)  # x0 in [-1, 1]
    params.add('y0', value=0, min=-1, max=1)  # y0 in [-1, 1]
    params.add('sigma_x', value=0.5, min=0.01, max=4)  # sigma_x bounds
    params.add('sigma_y', value=0.5, min=0.01, max=4)  # sigma_y bounds
    params.add('theta', value=0, min=-np.pi/2, max=np.pi/2)  # theta bounds

    # Minimize the residuals using least-squares
    result = minimize(
        gaussian_2d_lmfit,
        params,
        args=(x_flat, y_flat, data),
        # method='leastsq',  # Levenberg-Marquardt algorithm
        method='lbfgsb',
        tol=1e-8,
        # xtol=1e-8,
        max_nfev=20000
    )

    # Check if the fit succeeded
    if not result.success:
        return np.full(6, np.nan)

    fit_result = {}
    for name, param in result.params.items():
        fit_result[name] = param.value
        # fit_result[f"{name}_error"] = param.stderr

    return fit_result

from scipy.ndimage import gaussian_filter
def compute_gaussian_params(likelihood_maps, sigma=.5):
    """
    Compute Gaussian parameters for each likelihood map in `likelihood_maps`.
    `likelihood_maps` shape: (height, width, n_images)
    Returns: pandas DataFrame with columns for each parameter
    """
    n_images = likelihood_maps.shape[-1]
    results = []

    for i_image in tqdm(range(n_images)):
        likelihood_map = likelihood_maps[:, :, i_image]
        # Smooth the map to reduce noise
        likelihood_map = gaussian_filter(likelihood_map, sigma=sigma)
        # Fit the Gaussian
        results.append(fit_gaussian_lmfit(likelihood_map))


    # Convert the dictionary to a pandas DataFrame
    gaussian_df = pd.DataFrame(results)
    gaussian_df["theta_deg"] = np.rad2deg(gaussian_df["theta"])
    gaussian_df['sigma_major'] = np.where(gaussian_df['sigma_x'] > gaussian_df['sigma_y'], 
                                          gaussian_df['sigma_x'], gaussian_df['sigma_y'])
    gaussian_df['sigma_minor'] = np.where(gaussian_df['sigma_x'] < gaussian_df['sigma_y'], 
                                          gaussian_df['sigma_x'], gaussian_df['sigma_y'])

    gaussian_df["elongation"] = gaussian_df['sigma_major'] / gaussian_df['sigma_minor']
    gaussian_df["sigma"] = np.sqrt(gaussian_df['sigma_major'] * gaussian_df['sigma_minor'])
    return gaussian_df