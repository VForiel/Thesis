"""Cartographie de la contribution des noyaux sur le ciel.

Fournit des fonctions pour calculer et afficher les cartes de
transmission des différents kernels et la zone de contribution totale.
"""
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from copy import deepcopy as copy
from astropy import units as u
from tqdm import tqdm
from phise.classes import Context
from phise.modules import coordinates, utils
from phise.classes.context import project_position_jit, get_unique_source_input_fields_jit
from phise.classes.archs.superkn import expected_outputs_jit

def get_contribution_map(ctx: Context, resolution: int = 100, n: int = 100, map_func=np.median):
    """
    Compute the contribution zones of the kernels in the sky.

    Parameters
    ----------
    ctx : Context
        The context to use for the calculation.
    resolution : int
        The resolution of the map.
    n : int
        Number of observations per hour angle.
    map_func : function
        The function to use for mapping the images.

    Returns
    -------
    dict: containing 'images' (raw kernel maps) and 'stack' (combined map)
    """
    if ctx is None:
        ref_ctx = Context.get_VLTI()
        ref_ctx.interferometer.chip.σ = np.zeros(14) * u.nm
    else:
        ref_ctx = copy(ctx)
        
    raw_images = np.zeros((9, resolution, resolution)) # 6 Darks + 3 Kernels
    # Note: get_maps returns u.Quantity objects for rho/theta, but here we just need normalized rho map value
    (_, _, ρ_map, _) = coordinates.get_maps(N=resolution, fov=ref_ctx.interferometer.fov)
    ρ_map = ρ_map.value / np.max(ρ_map.value)
    
    h_range = ref_ctx.get_h_range()
    
    for (i, h) in enumerate(tqdm(h_range, desc="Computing contribution map")):
        ctx = copy(ref_ctx)
        ctx.h = h
        
        # Data storage for this hour angle
        darks_data = np.empty((n, 6))
        kernels_data = np.empty((n, 3))
        
        for j in range(n):
            # observe returns intensities; extract kernels via process_outputs
            outs = ctx.observe()
            # outs[0] is bright, outs[1:7] are darks
            darks_data[j] = outs[1:7]
            # process_outputs computes kernels from darks
            k = ctx.interferometer.chip.process_outputs(outs)
            kernels_data[j] = k
            
        trans_maps_result = ctx.get_transmission_maps(N=resolution)
        dark_maps = trans_maps_result[0][1:7] # Raw maps 1-6 are Darks
        kernel_maps = trans_maps_result[1]     # Processed maps are Kernels
        
        # Process Darks (Indices 0-5 in raw_images)
        for d in range(6):
            data = map_func(darks_data[:, d])
            t_map = dark_maps[d]
            raw_images[d] += t_map * data / len(h_range)
            
        # Process Kernels (Indices 6-8 in raw_images)
        for k in range(3):
            data = map_func(kernels_data[:, k])
            t_map = kernel_maps[k]
            # Note: Kernel maps/signals can be negative, but product is positive at source
            raw_images[6+k] += t_map * data / len(h_range)
            
    # Clamp negative values (artifacts) to 0 before stacking
    raw_images[raw_images < 0] = 0
    
    # Geometric mean of all 9 maps
    stack = np.prod(raw_images, axis=0) ** (1 / 9)
    
    return {"images": raw_images, "stack": stack, "ctx": ref_ctx}

def plot(ctx: Context=None, resolution: int=100, n=100, map=np.median, save_as=None):
    """
    Plot the contribution zones of the kernels in the sky.

    Parameters
    ----------
    ctx : Context, optional
        The context to use for the plot. If None, a default context is used.
    resolution : int, optional
        The resolution of the plot. Default is 100.
    map : function, optional
        The function to use for mapping the images. Default is np.median.
    save_as : str, optional
        Path to save the plot.
    """
    res = get_contribution_map(ctx, resolution, n, map)
    images = res["images"]
    stack = res["stack"]
    ref_ctx = res["ctx"] # Using the context used in calculation (which might be default if ctx was None)

    max_im = np.max(images)
    (_, axs) = plt.subplots(1, 4, figsize=(25, 5))
    fov = ref_ctx.interferometer.fov.to(u.mas)
    extent = [-fov.value / 2, fov.value / 2, -fov.value / 2, fov.value / 2]
    for k in range(3):
        img = images[k]
        img[img < 0] = 0
        im = axs[k].imshow(img, cmap='hot', vmax=max_im, extent=extent)
        axs[k].set_title(f'Kernel {k + 1}')
        plt.colorbar(im, ax=axs[k])
        for companion in ref_ctx.target.companions:
            (planet_x, planet_y) = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=fov)
            axs[k].scatter(planet_x * fov / 2, planet_y * fov / 2, color='tab:blue', edgecolors='black')
    
    im = axs[3].imshow(stack, cmap='hot', extent=extent)
    axs[3].set_title('Contribution zones')
    plt.colorbar(im, ax=axs[3])
    for companion in ref_ctx.target.companions:
        (planet_x, planet_y) = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=fov)
        axs[3].scatter(planet_x * fov / 2, planet_y * fov / 2, color='tab:blue', edgecolors='black')
    if save_as:
        utils.save_plot(save_as, "sky_contribution.png")

    plt.show()

def kernels_modulation(ctx: Context, h_range: np.ndarray, ρ: float, θ: float):
    """
    Calculate the modulation of kernels for a planet at a given position.

    Parameters:
    -----------
    ctx: Context
        The context containing interferometer and target info.
    h_range: np.ndarray
        Array of hour angles (in hourangle values, e.g. 0 to 24).
    ρ: float
        Angular separation in radians.
    θ: float
        Azimuthal angle in radians.

    Returns:
    --------
    kernels: np.ndarray
        Array of shape (3, len(h_range)) containing the 3 kernel outputs.
    """
    # Prepare constants
    l = ctx.interferometer.l.to(u.rad).value
    δ = ctx.target.δ.to(u.rad).value
    r = np.array([t.r.to(u.m).value for t in ctx.interferometer.telescopes])
    λ = ctx.interferometer.λ.to(u.m).value
    
    kernels = np.empty((3, len(h_range)))
    
    # Pre-compute input amplitude (normalized or scaled, here relative modulation matters)
    # We use 1.0 for amplitude to get the modulation shape
    a = np.ones(4) / 4 # Split flux eqally? or 1.0. expected_outputs_jit expects fields.
    
    for i, h in enumerate(h_range):
        # Convert hour angle from standard units (hourangle value) to radians
        # 1 HA = 15 degrees = pi/12 radians
        h_rad = h * (np.pi / 12)
        
        p = project_position_jit(r, h_rad, l, δ)
        
        # Calculate input fields for the source at (ρ, θ)
        ψ = get_unique_source_input_fields_jit(a=a, ρ=ρ, θ=θ, λ=λ, p=p)
        
        # Calculate analytical outputs
        _, _, k = expected_outputs_jit(ψ)
        
        kernels[:, i] = k
        
    return kernels

def get_correlation_map(ctx: Context, data: np.ndarray, h_range: np.ndarray, resolution: int = 20):
    """
    Compute the correlation map between observed data and theoretical kernel modulations.

    Parameters:
    -----------
    ctx: Context
        The observation context.
    data: np.ndarray
        Observed kernel data of shape (3, len(h_range)).
    h_range: np.ndarray
        Range of hour angles used for observation.
    resolution: int
        Resolution of the map (resolution x resolution pixels).
        
    Returns:
    --------
    correl_map: np.ndarray
        Correlation map of shape (resolution, resolution).
    """
    # Get coordinate maps
    # get_maps returns x, y, theta (angle), rho (separation)
    # theta is in rad, rho is in fov units
    _, _, theta_map, rho_map = coordinates.get_maps(N=resolution, fov=ctx.interferometer.fov)
    
    # Convert maps to values for JIT functions
    theta_vals = theta_map.to(u.rad).value
    rho_vals = rho_map.to(u.rad).value
    
    correl_map = np.zeros((resolution, resolution))
    
    print(f"Computing correlation map ({resolution}x{resolution})...")
    
    for x in tqdm(range(resolution), desc="Computing correlation map"):
        for y in range(resolution):
            ρ = rho_vals[x, y]
            θ = theta_vals[x, y]
            
            # Predict modulation for a planet at this position
            km = kernels_modulation(ctx, h_range, ρ=ρ, θ=θ)
            
            # Compute correlation with observed data
            # We average the correlation coefficients of the 3 kernels
            corr_sum = 0
            for k in range(3):
                # np.corrcoef returns 2x2 matrix, [0,1] is the correlation
                if np.std(km[k]) == 0 or np.std(data[k]) == 0:
                    c = 0 # Avoid division by zero
                else:
                    c = np.corrcoef(data[k], km[k])[0, 1]
                corr_sum += c
                
            correl_map[x, y] = corr_sum / 3

    return correl_map

def correlation_map(ctx: Context, data: np.ndarray, h_range: np.ndarray, resolution: int = 20, save_as: str = None):
    """
    Compute and plot the correlation map between observed data and theoretical kernel modulations.

    Parameters:
    -----------
    ctx: Context
        The observation context.
    data: np.ndarray
        Observed kernel data of shape (3, len(h_range)).
    h_range: np.ndarray
        Range of hour angles used for observation.
    resolution: int
        Resolution of the map (resolution x resolution pixels).
    save_as: str, optional
        Path to save the plot.
    """
    correl_map = get_correlation_map(ctx, data, h_range, resolution)
            
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fov_val = ctx.interferometer.fov.to(u.mas).value
    extent = [-fov_val/2, fov_val/2, -fov_val/2, fov_val/2]
    
    im = ax.imshow(correl_map, extent=extent, origin='lower', cmap='seismic')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Plot real companion positions
    for companion in ctx.target.companions:
        px, py = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=ctx.interferometer.fov)
        # ρθ_to_xy returns normalized coordinates in [-1, 1]
        # Multiply by fov/2 to get physical coordinates
        ax.scatter(px * fov_val/2, py * fov_val/2, color='lime', marker='x', s=100, label='Real Position')
        
    ax.set_title("Correlation Map")
    ax.set_xlabel(f"$\\alpha$ [mas]")
    ax.set_ylabel(f"$\\delta$ [mas]")
    if ctx.target.companions:
        ax.legend()
        
    if save_as:
        utils.save_plot(save_as, "correlation_map.png")
        
    plt.show()

def plot_filtered_contribution(ctx: Context, data: np.ndarray, h_range: np.ndarray, resolution: int = 20, n: int = 100, map_func=np.median, save_as: str = None):
    """
    Plot the sky contribution map filtered by positive correlation values.

    Parameters:
    -----------
    ctx: Context
        Observed context.
    data: np.ndarray
        Observed data (kernels).
    h_range: np.ndarray
        Hour angle range.
    resolution: int
        Resolution for both maps.
    n: int
        Number of samples for contribution calculation.
    map_func: function
        Mapping function for contribution.
    save_as: str, optional
        Path to save.
    """
    # 1. Compute contribution map
    print("Step 1/2: Computing Sky Contribution...")
    contrib_res = get_contribution_map(ctx, resolution, n, map_func)
    sky_contrib = contrib_res["stack"]
    
    # 2. Compute correlation map
    print("Step 2/2: Computing Correlation Map...")
    corr_map = get_correlation_map(ctx, data, h_range, resolution)
    
    # 3. Create mask (weighted)
    # Set negative values to 0
    mask = np.maximum(corr_map, 0)
    # Normalize to [0, 1]
    max_corr = np.max(mask)
    if max_corr > 0:
        mask = mask / max_corr
    
    # 4. Filter
    filtered_map = sky_contrib * mask
    
    # 5. Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fov_val = ctx.interferometer.fov.to(u.mas).value
    extent = [-fov_val/2, fov_val/2, -fov_val/2, fov_val/2]
    
    # Original Contribution
    im1 = axs[0].imshow(sky_contrib, extent=extent, origin='lower', cmap='hot')
    plt.colorbar(im1, ax=axs[0], label='Contribution')
    axs[0].set_title("Original Sky Contribution")
    
    # Correlation Mask
    im2 = axs[1].imshow(corr_map, extent=extent, origin='lower', cmap='seismic')
    plt.colorbar(im2, ax=axs[1], label='Correlation')
    axs[1].set_title("Correlation Map")
    
    # Filtered
    im3 = axs[2].imshow(filtered_map, extent=extent, origin='lower', cmap='hot')
    plt.colorbar(im3, ax=axs[2], label='Filtered Contribution')
    axs[2].set_title("Filtered Contribution (Pos. Corr. Only)")
    
    # Add markers
    for ax in axs:
        ax.set_xlabel(f"$\\alpha$ [mas]")
        ax.set_ylabel(f"$\\delta$ [mas]")
        for companion in ctx.target.companions:
            px, py = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=ctx.interferometer.fov)
            ax.scatter(px * fov_val/2, py * fov_val/2, color='lime' if ax != axs[0] and ax != axs[2] else 'tab:blue', 
                       marker='x', s=100, edgecolors='black')

    if save_as:
        utils.save_plot(save_as, "filtered_contribution.png")
        
    plt.show()