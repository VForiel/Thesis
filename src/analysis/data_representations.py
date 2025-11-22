"""Visualisations et transformations des distributions issues des
observations du kernel-nuller.

Fonctions pour calculer et tracer la distribution instantanée, les
évolutions temporelles et autres représentations des données.
"""
import numpy as np
import astropy.units as u
from copy import deepcopy as copy
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from phise import Context

π = np.pi

def get_B(Γ, α=1, β=1, φ2=π/3, φ3=π/4, φ4=π/6, n=10_000):

    # tirage bruit
    σ2 = np.random.normal(0, Γ, n)
    σ3 = np.random.normal(0, Γ, n)
    σ4 = np.random.normal(0, Γ, n)

    # champs étoile
    Bs = np.abs(np.sqrt(α/4) * (1
               + np.exp(1j*(σ2))
               + np.exp(1j*(σ3))
               + np.exp(1j*(σ4))))**2

    # champs planète (avec phases relatives)
    Bp = np.abs(np.sqrt(β/4) * (1
               + np.exp(1j*(σ2 + φ2))
               + np.exp(1j*(σ3 + φ3))
               + np.exp(1j*(σ4 + φ4))))**2

    return Bs, Bp, Bs + Bp
    
def get_K1(Γ, α=1, β=1, φ2=π/3, φ3=π/4, φ4=π/6, n=10_000):

    # tirage bruit
    σ2 = np.random.normal(0, Γ, n)
    σ3 = np.random.normal(0, Γ, n)
    σ4 = np.random.normal(0, Γ, n)

    # champs étoile
    S1s = np.sqrt(α/4) * (1
               + np.exp(1j*(  π/2 + σ2))
               + np.exp(1j*(3*π/2 + σ3))
               + np.exp(1j*(  π   + σ4)))

    S2s = np.sqrt(α/4) * (1
               + np.exp(1j*(3*π/2 + σ2))
               + np.exp(1j*(  π/2 + σ3))
               + np.exp(1j*(  π   + σ4)))

    # champs planète (avec phases relatives)
    S1p = np.sqrt(β/4) * (1
               + np.exp(1j*(  π/2 + σ2 + φ2))
               + np.exp(1j*(3*π/2 + σ3 + φ3))
               + np.exp(1j*(  π   + σ4 + φ4)))

    S2p = np.sqrt(β/4) * (1
               + np.exp(1j*(3*π/2 + σ2 + φ2))
               + np.exp(1j*(  π/2 + σ3 + φ3))
               + np.exp(1j*(  π   + σ4 + φ4)))
    
    S1 = np.abs(S1s)**2 + np.abs(S1p)**2
    S2 = np.abs(S2s)**2 + np.abs(S2p)**2

    Ks = np.abs(S1s)**2 - np.abs(S2s)**2
    Kp = np.abs(S1p)**2 - np.abs(S2p)**2
    K = S1 - S2

    return Ks, Kp, K

def get_K2(Γ, α=1, β=1, φ2=π/3, φ3=π/4, φ4=π/6, n=10_000):

    # tirage bruit
    σ2 = np.random.normal(0, Γ, n)
    σ3 = np.random.normal(0, Γ, n)
    σ4 = np.random.normal(0, Γ, n)

    # champs étoile
    S1s = np.sqrt(α) * (1
               + np.exp(1j*(  π/2 + σ2))
               + np.exp(1j*(  π   + σ3))
               + np.exp(1j*(3*π/2 + σ4)))

    S2s = np.sqrt(α) * (1
               + np.exp(1j*(3*π/2 + σ2))
               + np.exp(1j*(  π   + σ3))
               + np.exp(1j*(  π/2 + σ4)))

    # champs planète (avec phases relatives)
    S1p = np.sqrt(β) * (1
               + np.exp(1j*(  π/2 + σ2 + φ2))
               + np.exp(1j*(  π   + σ3 + φ3))
               + np.exp(1j*(3*π/2 + σ4 + φ4)))

    S2p = np.sqrt(β) * (1
               + np.exp(1j*(3*π/2 + σ2 + φ2))
               + np.exp(1j*(  π   + σ3 + φ3))
               + np.exp(1j*(  π/2 + σ4 + φ4)))

    Ks = np.abs(S1s)**2 - np.abs(S2s)**2
    Kp = np.abs(S1p)**2 - np.abs(S2p)**2

    return Ks, Kp, Ks + Kp

def get_K3(Γ, α=1, β=1, φ2=π/3, φ3=π/4, φ4=π/6, n=10_000):

    # tirage bruit
    σ2 = np.random.normal(0, Γ, n)
    σ3 = np.random.normal(0, Γ, n)
    σ4 = np.random.normal(0, Γ, n)

    # champs étoile
    S1s = np.sqrt(α) * (1
               + np.exp(1j*(  π   + σ2))
               + np.exp(1j*(  π/2 + σ3))
               + np.exp(1j*(3*π/2 + σ4)))

    S2s = np.sqrt(α) * (1
               + np.exp(1j*(  π   + σ2))
               + np.exp(1j*(3*π/2 + σ3))
               + np.exp(1j*(  π/2 + σ4)))

    # champs planète (avec phases relatives)
    S1p = np.sqrt(β) * (1
               + np.exp(1j*(  π   + σ2 + φ2))
               + np.exp(1j*(  π/2 + σ3 + φ3))
               + np.exp(1j*(3*π/2 + σ4 + φ4)))

    S2p = np.sqrt(β) * (1
               + np.exp(1j*(  π   + σ2 + φ2))
               + np.exp(1j*(3*π/2 + σ3 + φ3))
               + np.exp(1j*(  π/2 + σ4 + φ4)))

    Ks = np.abs(S1s)**2 - np.abs(S2s)**2
    Kp = np.abs(S1p)**2 - np.abs(S2p)**2

    return Ks, Kp, Ks + Kp

#==============================================================================
# Instantaneous distribution
#==============================================================================

def instant_distribution(ctx: Context=None, n=10000, stat=np.median, figsize=(10, 10), compare=True, r=1, log=False, density = False, 
    sync_plots = True) -> np.ndarray:
    """
    Get the instantaneous distribution of the kernel nuller.

    Parameters
    ----------
    ctx : Context
        The context to use.
    n : int, optional
        The number of samples to take, by default 1000
    stat : function, optional
        The function to use to compute the statistic, by default np.median.

    Returns
    -------
    np.ndarray
        The instantaneous distribution of the kernel nuller.
    """

    # Set up contexts ---------------------------------------------------------

    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.um
        ctx.target.companions[0].c = 0.1
    else:
        ctx = copy(ctx)
        if ctx.target.companions == []:
            raise ValueError('No companion in the context. Please add a companion to the target.')
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle

    ctx_so = copy(ctx)
    ctx_so.target.companions = []

    ctx_po = copy(ctx)
    scale = 1e12
    ctx_po.target.f /= scale # Scale down the star flux to make it negligible
    ctx_po.target.companions[0].c *= scale

    # Numerical model ---------------------------------------------------------

    # Prepare data arrays
    data = np.empty((n, 3))
    data_so = np.empty((n, 3))
    data_po = np.empty((n, 3))

    brights = np.empty(n)
    brights_so = np.empty(n)
    brights_po = np.empty(n)

    # Sample data
    for i in range(n):

        # Generate noise
        opd_error = np.random.normal(0, ctx.Γ.value, size=len(ctx.interferometer.telescopes)) * ctx.Γ.unit

        # Distrib with companion(s)
        outs = ctx.observe(opd_error=opd_error)
        data[i, :] = ctx.interferometer.chip.process_outputs(outs)
        brights[i] = np.sum(np.abs(outs)**2)
        
        # Distrib with star only
        outs_so = ctx_so.observe(opd_error=opd_error)
        data_so[i, :] = ctx_so.interferometer.chip.process_outputs(outs_so)
        brights_so[i] = np.sum(np.abs(outs_so)**2)

        # Distrib with planet only
        outs_po = ctx_po.observe(opd_error=opd_error)
        data_po[i, :] = ctx_po.interferometer.chip.process_outputs(outs_po)
        brights_po[i] = np.sum(np.abs(outs_po)**2)

    print("Numerical model:")
    print(f'   Median brightness (star + planet): {np.median(brights):.3e}')
    print(f'   Median brightness (star only): {np.median(brights_so):.3e}')
    print(f'   Median brightness (planet only): {np.median(brights_po):.3e}')

    # Analytical model --------------------------------------------------------

    # Prepare data arrays
    analytical_data = np.empty((n,3))
    analytical_data_so = np.empty((n,3))
    analytical_data_po = np.empty((n,3))

    # Get parameters (amplitude and phases)
    ψi = ctx.get_input_fields()
    φi1, φi2, φi3, φi4 = np.angle(ψi[1])
    α = (np.mean(ctx.pf)).value
    β = α * ctx.target.companions[0].c
    Γ = ctx.Γ.to(u.nm).value / ctx.interferometer.λ.to(u.nm).value * 2 * np.pi

    # Get kernels
    brights_so, brights_po, brights = get_B(Γ=Γ, α=α, β=β, φ2=φi2-φi1, φ3=φi3-φi1, φ4=φi4-φi1, n=n)

    print("Analytical model:")
    print(f'   Median brightness (star + planet): {np.median(brights):.3e}')
    print(f'   Median brightness (star only): {np.median(brights_so):.3e}')
    print(f'   Median brightness (planet only): {np.median(brights_po):.3e}')

    tmp_so, tmp_po, tmp = get_K1(Γ=Γ, α=α, β=β, φ2=φi2-φi1, φ3=φi3-φi1, φ4=φi4-φi1, n=n)
    analytical_data[:, 0] = tmp * r
    analytical_data_so[:, 0] = tmp_so * r
    analytical_data_po[:, 0] = tmp_po * r
    tmp_so, tmp_po, tmp = get_K2(Γ=Γ,  α=α, β=β, φ2=φi2-φi1, φ3=φi3-φi1, φ4=φi4-φi1, n=n)
    analytical_data[:, 1] = tmp * r
    analytical_data_so[:, 1] = tmp_so * r
    analytical_data_po[:, 1] = tmp_po * r
    tmp_so, tmp_po, tmp = get_K3(Γ=Γ,  α=α, β=β, φ2=φi2-φi1, φ3=φi3-φi1, φ4=φi4-φi1, n=n)
    analytical_data[:, 2] = tmp * r
    analytical_data_so[:, 2] = tmp_so * r
    analytical_data_po[:, 2] = tmp_po * r

    # Sum and convolution of distribs -----------------------------------------

    data_comb = data_so + data_po
    analytical_comb = analytical_data_so + analytical_data_po

    data_conv = np.empty((2 * int(np.sqrt(n)) + 1, 3))
    analytical_conv = np.empty((2 * int(np.sqrt(n)) + 1, 3))

    for k in range(3):
        hist_so, bin_edges = np.histogram(data_so[:,k], bins=2 * int(np.sqrt(n)) + 1, density=density)
        hist_po, _ = np.histogram(data_po[:,k], bins=bin_edges, density=density)
        data_conv[:,k] = np.convolve(hist_so, hist_po, mode='same') / n

        hist_so, bin_edges = np.histogram(analytical_data_so[:,k], bins=2 * int(np.sqrt(n)) + 1, density=density)
        hist_po, _ = np.histogram(analytical_data_po[:,k], bins=bin_edges, density=density)
        analytical_conv[:,k] = np.convolve(hist_so, hist_po, mode='same') / n

        
        if density:
            data_conv[:,k] /= np.sum(data_conv[:,k]) * (bin_edges[1]-bin_edges[0])
            analytical_conv[:,k] /= np.sum(analytical_conv[:,k]) * (bin_edges[1]-bin_edges[0])
    
    # Plotting ----------------------------------------------------------------
    
    scenario_full = [data, analytical_data]
    scenario_star_only = [data_so, analytical_data_so]
    scenario_planet_only = [data_po, analytical_data_po]

    for i, (numerical_data, analytical_data) in enumerate([scenario_star_only, scenario_planet_only, scenario_full]):

        (_, axs) = plt.subplots(3, 3, figsize=figsize, constrained_layout=True, sharex=sync_plots)
        plt.suptitle(['Star only', 'Planet only', 'Star + planet', 'Convolution'][i], fontsize=16)
        
        # Get plot limits
        lim = 0
        for k in range(3):

            keep = 75
            percentil = (100 - keep) / 2

            # Get x limits
            (xmin_plot, xmax_plot) = np.percentile(numerical_data[:, k], [percentil, keep + percentil])
            lim = max(lim, abs(xmin_plot), abs(xmax_plot))
            (xmin_plot, xmax_plot) = np.percentile(analytical_data[:, k], [percentil, keep + percentil])
            lim = max(lim, abs(xmin_plot), abs(xmax_plot))

        # Get bins
        if sync_plots:
            bins = np.linspace(-lim, lim, 2 * int(np.sqrt(n)) + 1)
        else:
            bins = 2 * int(np.sqrt(n)) + 1

        # Plot histograms
        for k in range(3):
            for j in range(3):
                if j in [1,2]:
                    axs[k, j].hist(analytical_data[:,k], label='Analytic', bins=bins, alpha=0.5, color='green', density=density, log=log)
                    axs[k, j].axvline(stat(analytical_data[:,k]), color='green', linestyle='--')
                    if i==2:
                        axs[k, j].hist(analytical_comb[:,k], label='Analytic (sum)', bins=bins, alpha=1, color='red', density=density, histtype='step', linewidth=2, log=log)
                        axs[k, j].axvline(stat(analytical_comb[:,k]), color='red', linestyle='--')
                        Δbins = bins[1]-bins[0]
                        edges = bins - Δbins/2
                        edges = np.append(edges, edges[-1] + Δbins)
                        axs[k, j].stairs(analytical_conv[:,k], edges=edges, label='Analytic (conv)', alpha=1, color='darkred', edgecolor='darkred', fill=False, linewidth=2)
                if j in [0,2]:
                    axs[k, j].hist(numerical_data[:, k], label='Numeric', bins=bins, alpha=0.5, color='blue', density=density, log=log)
                    axs[k, j].axvline(stat(numerical_data[:,k]), color='blue', linestyle='--')
                    if i==2:
                        axs[k, j].hist(data_comb[:,k], label='Numeric (sum)', bins=bins, alpha=1, color='orange', density=density, histtype='step', linewidth=2, log=log)
                        axs[k, j].axvline(stat(data_comb[:,k]), color='orange', linestyle='--')
                        Δbins = bins[1]-bins[0]
                        edges = bins - Δbins/2
                        edges = np.append(edges, edges[-1] + Δbins)
                        axs[k, j].stairs(data_conv[:,k], edges=edges, label='Numeric (conv)', alpha=1, color='darkorange', edgecolor='darkorange', fill=False, linewidth=2)
                # Set labels
                axs[k,j].set_ylabel('Occurrences (%)')
                axs[k,j].set_title(f'Kernel {k + 1}')
                axs[k,j].legend()
            if sync_plots:
                axs[k,0].set_xlim(-lim, lim)
                axs[k,2].set_xlim(-lim, lim)
        axs[2,0].set_xlabel('Kernel output')
        axs[2,1].set_xlabel('Kernel output')
        axs[2,2].set_xlabel('Kernel output')
        plt.show()

    return (data, data_so)

def time_evolution(ctx: Context=None, n=100, map=np.median) -> np.ndarray:
    """
    Get the time evolution of the kernel nuller.

    Parameters
    ----------
    ctx : Context
        The context to use.
    n : int, optional
        The number of samples to take at a given time, by default 1000.
    map : function, optional
        The function to use to map the data, by default np.median.

    Returns
    -------
    np.ndarray
        The time evolution of the kernel nuller. (n_h, 3)
    np.ndarray
        The reference time evolution of the kernel nuller (without input perturbation). (n_h, 3)
    """

    # Bluid context
    if ctx is None:
        ctx:Context = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.um
        ctx.Γ = 10 * u.nm
    else:
        ctx:Context = copy(ctx)
    ctx_so = copy(ctx)
    ctx_so.Γ = 0 * u.nm

    # Prepare data arrays
    nh = len(ctx.get_h_range())
    data = np.empty((nh, 3))
    data_so = np.empty((nh, 3))

    # Sample data
    outs = ctx.observation_serie(n=n)
    b = outs[:,:,0]
    k = np.empty((n, nh, 3))
    for i in range(n):
        for j in range(nh):
            k[i, j, :] = ctx.interferometer.chip.process_outputs(outs[i, j, :])
    outs = ctx_so.observation_serie(n=1)
    b = outs[:,:,0]
    k = np.empty((nh, 3))
    for j in range(nh):
        k[j, :] = ctx.interferometer.chip.process_outputs(outs[j, :])

    k_depth = np.empty_like(k)
    ref_k_depth = np.empty_like(ref_k)
    for i in range(n):
        for h in range(len(ctx.get_h_range())):
            k_depth[i, h] = k[i, h] / b[i, h]
        for h in range(len(ctx_so.get_h_range())):
            ref_k_depth[0, h] = ref_k[0, h] / ref_b[0, h]
    for h in range(len(ctx.get_h_range())):
        for k in range(3):
            data[h, k] = map(k_depth[:, h, k])
    for h in range(len(ctx_so.get_h_range())):
        for k in range(3):
            data_so[h, k] = ref_k_depth[0, h, k]
    (_, axs) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for k in range(3):
        axs[k].scatter(ctx.get_h_range(), data[:, k], label='Data')
        axs[k].plot(ctx_so.get_h_range(), data_so[:, k], label='Reference', linestyle='--')
        axs[k].set_ylabel(f'Kernel output')
        axs[k].set_xlabel('Time (hourangle)')
        axs[k].set_title(f'Kernel {k + 1}')
        axs[k].legend()
    plt.show()
    return (data, data_so)