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
from phise.classes.archs.superkn import expected_outputs_jit, expected_outputs_batch_jit

π = np.pi

def compute_analytical_distrib(n, ctx, opd_errors, α, β, φ1, φ2, φ3, φ4):
    """
    Compute analytical distribution using SuperKN.expected_outputs_batch_jit.
    """
    if ctx.monochromatic:
        λ_range = np.array([ctx.interferometer.λ.to(u.m).value])
    else:
        λ0 = ctx.interferometer.λ.to(u.m).value
        Δλ = ctx.interferometer.Δλ.to(u.m).value
        λ_range = np.linspace(λ0 - Δλ/2, λ0 + Δλ/2, 5)
    
    λ0 = ctx.interferometer.λ.to(u.m).value
    
    # Pre-allocate accumulators
    brights = np.zeros(n)
    darks = np.zeros((n, 6))
    kernels = np.zeros((n, 3))
    
    geometric_phases = np.array([φ1, φ2, φ3, φ4])
    
    # opd_errors is (n, 4)
    
    for λ in λ_range:
        # Compute phases for all samples
        # ph is (4,)
        ph = geometric_phases * (λ0 / λ)
        
        # sig is (n, 4)
        sig = 2 * np.pi * opd_errors / λ
        
        # Construct ψ (input fields)
        # Star field per input j: sqrt(α)/4 * exp(1j * sig)
        # We need to broadcast sqrt(α)/4 to (n, 4)
        
        # Star
        # ψ_s is (n, 4)
        ψ_s = np.sqrt(α/4) * np.exp(1j * sig)
        
        b_s, d_s, k_s = expected_outputs_batch_jit(ψ_s)

        # Planet
        # ψ_p is (n, 4)
        # ph is (4,) -> broadcast to (1, 4)
        ψ_p = np.sqrt(β/4) * np.exp(1j * (sig + ph[np.newaxis, :]))
        
        b_p, d_p, k_p = expected_outputs_batch_jit(ψ_p)
        
        # Sum intensities
        brights += b_s + b_p
        darks += d_s + d_p
        kernels += k_s + k_p
        
    brights /= len(λ_range)
    darks /= len(λ_range)
    kernels /= len(λ_range)
        
    return brights, darks, kernels

#==============================================================================
# Instantaneous distribution
#==============================================================================

def instant_distribution(ctx: Context=None, n=10000, stat=np.median, figsize=(10, 10), compare=True, r=1, log=False, density = False, 
    sync_plots = True, save_path=None, show=True) -> np.ndarray:
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
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show : bool, optional
        Whether to show the plot. Default is True.

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

    print(f"Star: Signal: {np.sum(ctx_po.pf) * ctx_po.interferometer.camera.e:.2e}, flux : {ctx_po.target.f:.2e}")
    print(f"Planet: Contrast: {ctx_po.target.companions[0].c:.2e} Signal: {np.sum(ctx_po.pf) * ctx_po.target.companions[0].c * ctx_po.interferometer.camera.e:.2e}")

    # Numerical model ---------------------------------------------------------

    # Sample data
    # Generate noise for all samples at once
    upstream_pistons = np.random.normal(0, ctx.Γ.value, size=(n, len(ctx.interferometer.telescopes))) * ctx.Γ.unit
    errors = upstream_pistons

    # Distrib with companion(s)
    outs = ctx.observe_batch(upstream_pistons=upstream_pistons)
    brights = outs[:, 0]
    darks = outs[:, 1:]
    k1 = darks[:, 0] - darks[:, 1]
    k2 = darks[:, 2] - darks[:, 3]
    k3 = darks[:, 4] - darks[:, 5]
    data = np.stack([k1, k2, k3], axis=1)
    
    # Distrib with star only
    outs_so = ctx_so.observe_batch(upstream_pistons=upstream_pistons)
    brights_so = outs_so[:, 0]
    darks_so = outs_so[:, 1:]
    k1_so = darks_so[:, 0] - darks_so[:, 1]
    k2_so = darks_so[:, 2] - darks_so[:, 3]
    k3_so = darks_so[:, 4] - darks_so[:, 5]
    data_so = np.stack([k1_so, k2_so, k3_so], axis=1)

    # Distrib with planet only
    outs_po = ctx_po.observe_batch(upstream_pistons=upstream_pistons)
    brights_po = outs_po[:, 0]
    darks_po = outs_po[:, 1:]
    k1_po = darks_po[:, 0] - darks_po[:, 1]
    k2_po = darks_po[:, 2] - darks_po[:, 3]
    k3_po = darks_po[:, 4] - darks_po[:, 5]
    data_po = np.stack([k1_po, k2_po, k3_po], axis=1)

    # Analytical model --------------------------------------------------------

    # Get parameters (amplitude and phases) for combined scenario
    ψi = ctx.get_input_fields()
    φ1, φ2, φ3, φ4 = np.angle(ψi[1])
    e = ctx.interferometer.camera.e.to(u.s).value
    α = np.sum(ctx.pf).value * e
    β = α * ctx.target.companions[0].c
    
    opd_errors = errors.to(u.m).value

    # Get parameters for star-only scenario
    α_so = np.sum(ctx_so.pf).value * e
    β_so = 0  # No planet

    # Get parameters for planet-only scenario (scaled)
    α_po = np.sum(ctx_po.pf).value * e # Star flux scaled down by 1e12
    β_po = α_po * ctx_po.target.companions[0].c  # Planet flux scaled up by 1e12

    # Combined
    b_comb, d_comb, k_comb = compute_analytical_distrib(n, ctx, opd_errors, α, β, φ1, φ2, φ3, φ4)
    analytical_brights = b_comb
    analytical_darks = d_comb
    analytical_data = k_comb * r
    
    # Star only
    b_so, d_so, k_so = compute_analytical_distrib(n, ctx_so, opd_errors, α_so, β_so, φ1, φ2, φ3, φ4)
    analytical_brights_so = b_so
    analytical_darks_so = d_so
    analytical_data_so = k_so * r
    
    # Planet only
    b_po, d_po, k_po = compute_analytical_distrib(n, ctx_po, opd_errors, α_po, β_po, φ1, φ2, φ3, φ4)
    analytical_brights_po = b_po
    analytical_darks_po = d_po
    analytical_data_po = k_po * r

    # Sum of distributions -----------------------------------------

    data_comb = data_so + data_po
    analytical_comb = analytical_data_so + analytical_data_po

    # Note: the convolution (data_conv and analytical_conv) needs the
    # histogram bins to be defined (dependent on plot limits). We compute
    # the convolutions later inside the plotting loop once `bins` is set,
    # so they are aligned with the plotted histograms.
    
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

        # If we are plotting the combined case (i==2), compute the
        # convolution of the star-only and planet-only distributions
        # on the same `bins` grid so stairs() aligns with hist().
        if i == 2:
            Nbins = len(bins)
            Δbins = bins[1] - bins[0]
            data_conv = np.zeros((Nbins, 3))
            analytical_conv = np.zeros((Nbins, 3))
            for kk in range(3):
                # Numeric: use density=True to get PDFs, then convolve
                hist_so, _ = np.histogram(data_so[:, kk], bins=bins, density=True)
                hist_po, _ = np.histogram(data_po[:, kk], bins=bins, density=True)
                conv = np.convolve(hist_so, hist_po, mode='full') * Δbins
                start = (len(conv) - Nbins) // 2
                data_conv[:, kk] = conv[start:start + Nbins]

                # Analytical
                ahist_so, _ = np.histogram(analytical_data_so[:, kk], bins=bins, density=True)
                ahist_po, _ = np.histogram(analytical_data_po[:, kk], bins=bins, density=True)
                aconv = np.convolve(ahist_so, ahist_po, mode='full') * Δbins
                analytical_conv[:, kk] = aconv[start:start + Nbins]

        # Plot histograms
        for k in range(3):
            for j in range(3):
                if j in [1,2]:
                    axs[k, j].hist(analytical_data[:,k], label='Analytic', bins=bins, alpha=0.5, color='green', density=density, log=log)
                    axs[k, j].axvline(stat(analytical_data[:,k]), color='green', linestyle='--')
                    if i==2:
                        axs[k, j].hist(analytical_comb[:,k], label='Analytic (sum)', bins=bins, alpha=0.5, color='red', density=density, histtype='step', linewidth=2, log=log)
                        axs[k, j].axvline(stat(analytical_comb[:,k]), color='red', linestyle='--')
                        Δbins = bins[1]-bins[0]
                        edges = bins - Δbins/2
                        edges = np.append(edges, edges[-1] + Δbins)
                        axs[k, j].stairs(analytical_conv[:,k], edges=edges, label='Analytic (conv)', alpha=0.5, color='purple', edgecolor='purple', fill=False, linewidth=2)
                if j in [0,2]:
                    axs[k, j].hist(numerical_data[:, k], label='Numeric', bins=bins, alpha=0.5, color='blue', density=density, log=log)
                    axs[k, j].axvline(stat(numerical_data[:,k]), color='blue', linestyle='--')
                    if i==2:
                        axs[k, j].hist(data_comb[:,k], label='Numeric (sum)', bins=bins, alpha=0.5, color='orange', density=density, histtype='step', linewidth=2, log=log)
                        axs[k, j].axvline(stat(data_comb[:,k]), color='orange', linestyle='--')
                        Δbins = bins[1]-bins[0]
                        edges = bins - Δbins/2
                        edges = np.append(edges, edges[-1] + Δbins)
                        axs[k, j].stairs(data_conv[:,k], edges=edges, label='Numeric (conv)', alpha=0.5, color='sienna', edgecolor='sienna', fill=False, linewidth=2)
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
        
        if save_path:
            # If multiple figures, append suffix
            fname = save_path
            if i > 0:
                import os
                base, ext = os.path.splitext(save_path)
                fname = f"{base}_{i}{ext}"
            plt.savefig(fname)
            
        if show:
            plt.show()
        else:
            plt.close()

    # Debug info --------------------------------------------------------------

    # Convolution Scenario (Simulated via shuffling)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Numerical
    data_conv_sim = data_so + data_po[indices]
    brights_conv_sim = brights_so + brights_po[indices]
    darks_conv_sim = darks_so + darks_po[indices]
    
    # Analytical
    analytical_data_conv_sim = analytical_data_so + analytical_data_po[indices]
    analytical_brights_conv_sim = analytical_brights_so + analytical_brights_po[indices]
    analytical_darks_conv_sim = analytical_darks_so + analytical_darks_po[indices]

    # Collect data for all scenarios
    scenarios = {
        "Étoile seule": {
            "bright": (brights_so, analytical_brights_so),
            "darks": (darks_so, analytical_darks_so),
            "kernels": (data_so, analytical_data_so)
        },
        "Planète seule": {
            "bright": (brights_po, analytical_brights_po),
            "darks": (darks_po, analytical_darks_po),
            "kernels": (data_po, analytical_data_po)
        },
        "Complet": {
            "bright": (brights, analytical_brights),
            "darks": (darks, analytical_darks),
            "kernels": (data, analytical_data)
        },
        "Somme": {
            "bright": (brights_so + brights_po, analytical_brights_so + analytical_brights_po),
            "darks": (darks_so + darks_po, analytical_darks_so + analytical_darks_po),
            "kernels": (data_so + data_po, analytical_data_so + analytical_data_po)
        },
        "Convolution": {
            "bright": (brights_conv_sim, analytical_brights_conv_sim),
            "darks": (darks_conv_sim, analytical_darks_conv_sim),
            "kernels": (data_conv_sim, analytical_data_conv_sim)
        }
    }

    print("\n# Comparaison des Statistiques")
    
    print("(Analytique | Numerique)")

    # Total Photons
    print("\n## Total Photons (Bright + Sum(Darks))")
    for name, data_dict in scenarios.items():
        num_b, ana_b = data_dict["bright"]
        num_d, ana_d = data_dict["darks"]
        
        total_num = np.mean(num_b) + np.sum(np.mean(num_d, axis=0))
        total_ana = np.mean(ana_b) + np.sum(np.mean(ana_d, axis=0))
        
        print(f"- {name}: {total_ana:.3e} | {total_num:.3e}")

    # Bright
    print("\n## Bright")
    for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
        print(f"\n### {stat_name}")
        for name, data_dict in scenarios.items():
            num, ana = data_dict["bright"]
            print(f"- {name}: {stat_func(ana):.3e} | {stat_func(num):.3e}")

    # Darks
    for k in range(6):
        print(f"\n## Dark {k+1}")
        for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
            print(f"\n### {stat_name}")
            for name, data_dict in scenarios.items():
                num, ana = data_dict["darks"]
                print(f"- {name}: {stat_func(ana[:, k]):.3e} | {stat_func(num[:, k]):.3e}")

    # Kernels
    for k in range(3):
        print(f"\n## Kernel {k+1}")
        for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
            print(f"\n### {stat_name}")
            for name, data_dict in scenarios.items():
                num, ana = data_dict["kernels"]
                print(f"- {name}: {stat_func(ana[:, k]):.3e} | {stat_func(num[:, k]):.3e}")

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
    data = np.empty((len(ctx.get_h_range()), 3))
    data_so = np.empty((len(ctx.get_h_range()), 3))
    data_po = np.empty((len(ctx.get_h_range()), 3))
    
    ref_data = np.empty((len(ctx.get_h_range()), 3))
    ref_data_so = np.empty((len(ctx.get_h_range()), 3))
    ref_data_po = np.empty((len(ctx.get_h_range()), 3))

    # Sample data
    for i, h in enumerate(ctx.get_h_range()):
        ctx.h = h * u.rad
        ctx_so.h = h * u.rad
        ctx_po.h = h * u.rad

        # Generate noise
        upstream_pistons = np.random.normal(0, ctx.Γ.value, size=(n, len(ctx.interferometer.telescopes))) * ctx.Γ.unit

        # Distrib with companion(s)
        outs = ctx.observe(upstream_pistons=upstream_pistons)
        data[i, :] = map(ctx.interferometer.chip.process_outputs(outs), axis=0)
        
        # Distrib with star only
        outs_so = ctx_so.observe(upstream_pistons=upstream_pistons)
        data_so[i, :] = map(ctx_so.interferometer.chip.process_outputs(outs_so), axis=0)

        # Distrib with planet only
        outs_po = ctx_po.observe(upstream_pistons=upstream_pistons)
        data_po[i, :] = map(ctx_po.interferometer.chip.process_outputs(outs_po), axis=0)

        # Reference data
        outs = ctx.observe()
        ref_data[i, :] = ctx.interferometer.chip.process_outputs(outs)
        
        outs_so = ctx_so.observe()
        ref_data_so[i, :] = ctx_so.interferometer.chip.process_outputs(outs_so)

        outs_po = ctx_po.observe()
        ref_data_po[i, :] = ctx_po.interferometer.chip.process_outputs(outs_po)

    return data, ref_data