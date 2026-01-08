"""ABCD loop calibration for photonic interferometers.

This algorithm iteratively tunes phase shifters using four-point ABCD sampling
(per shifter) across several loops, then performs a fine sweep to refine each
shifter with dense sampling. It targets kernel-null depth minimization while
keeping the API aligned with other calibration utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from ..phase import bound


def _kernel_null_metric(ctx) -> float:
    """Compute kernel-null-depth metric for current chip phases.

    The metric is defined as the mean processed kernel magnitude divided by the
    bright output intensity. Lower is better.
    """
    outs = ctx.observe()
    bright = outs[0]
    kernels = ctx.interferometer.chip.process_outputs(outs)
    k_mean = np.mean(np.abs(kernels))
    return float(k_mean / bright)


def calibrate_abcd(
    ctx,
    n_loops: int = 2,
    n_final_samples: int = 64,
    plot: bool = False,
    verbose: bool = False,
    figsize: tuple = (14, 8),
    save_as=None,
):
    """ABCD multi-loop calibration followed by a fine sweep.

    For each phase shifter, four measurements (A,B,C,D) are taken at phases
    0, π/2, π, 3π/2 (in radians). The ABCD relation

    .. math:: \phi_\mathrm{bias} = \mathrm{atan2}(D-B, A-C)

    provides the phase offset that minimizes the sinusoidal kernel-null-depth.
    After iterating over all shifters for ``n_loops`` passes, a final dense
    sweep with ``n_final_samples`` points refines each shifter using a direct
    minimum search on the metric.

    Parameters
    ----------
    ctx : Context
        Observation context holding interferometer and target.
    n_loops : int, optional
        Number of ABCD passes over all shifters (default: 2).
    n_final_samples : int, optional
        Number of samples for the final dense sweep per shifter (default: 64).
    plot : bool, optional
        If True, plot metric evolution across loops and per-shifter updates.
    verbose : bool, optional
        If True, print per-shifter ABCD estimates and metrics.
    figsize : tuple, optional
        Figure size used when ``plot=True`` (default: (14, 8)).
    save_as : str, optional
        Directory path where plots will be saved using ``save_plot`` if provided.

    Returns
    -------
    dict
        Calibration summary with keys:
        - ``context``: the updated context (same object, returned for chaining)
        - ``history``: list of metrics after each shifter update across all loops
        - ``history_shifters``: array of shape (steps, n_shifters) tracking phase values
        - ``final_phi``: final phase shifter values (Quantity)
    """

    chip = ctx.interferometer.chip
    λ = ctx.interferometer.λ
    n_shifters = len(chip.φ)

    # Start from zeroed phases as specified
    chip.φ = np.zeros(n_shifters) * λ.unit

    # Phases for ABCD sampling in length units
    phases_rad = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    phases_len = phases_rad / (2 * np.pi) * λ

    # Tracking arrays for plotting
    metric_history = []  # metric after each shifter update (step)
    shifter_history = []  # phase values after each shifter update
    step_count = 0
    loop_boundaries = []  # indices marking loop boundaries

    for loop_idx in range(n_loops):
        if verbose:
            print(f"Loop {loop_idx + 1}/{n_loops}")

        for idx in range(n_shifters):
            # Sample ABCD
            samples = []
            for phase_len in phases_len:
                chip.φ[idx] = phase_len
                samples.append(_kernel_null_metric(ctx))

            A, B, C, D = samples
            phi_bias = np.arctan2(D - B, A - C)  # radians
            target_phase = np.mod(np.pi - phi_bias, 2 * np.pi)
            chip.φ[idx] = target_phase / (2 * np.pi) * λ

            metric_after = _kernel_null_metric(ctx)
            metric_history.append(metric_after)
            shifter_history.append(np.copy(chip.φ.value))
            step_count += 1

            if verbose:
                print(
                    f"  Shifter {idx+1:02d}: ABCD=[{A:.3e},{B:.3e},{C:.3e},{D:.3e}] "
                    f"-> φ_bias={phi_bias:.3f} rad, set φ={chip.φ[idx]:.3e}, metric={metric_after:.3e}"
                )

        loop_boundaries.append(step_count)

    # Final fine sweep per shifter
    x_rad = np.linspace(0, 2 * np.pi, n_final_samples, endpoint=False)
    x_len = x_rad / (2 * np.pi) * λ
    fine_sweep_start = step_count

    for idx in range(n_shifters):
        metrics = np.empty_like(x_rad)
        for j, phase_len in enumerate(x_len):
            chip.φ[idx] = phase_len
            metrics[j] = _kernel_null_metric(ctx)
        best_idx = int(np.argmin(metrics))
        chip.φ[idx] = x_len[best_idx]
        
        metric_after = _kernel_null_metric(ctx)
        metric_history.append(metric_after)
        shifter_history.append(np.copy(chip.φ.value))
        step_count += 1

        if verbose:
            print(
                f"  Fine sweep shifter {idx+1:02d}: best metric={metrics[best_idx]:.3e} "
                f"at phase={chip.φ[idx]:.3e}"
            )

    chip.φ = bound(chip.φ, λ)

    # Plotting
    if plot:
        metric_history = np.array(metric_history)
        shifter_history = np.array(shifter_history)

        fig, ax = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

        # Plot 1: Kernel-null depth metric per step
        steps = np.arange(len(metric_history))
        ax[0].plot(steps, metric_history, 'o-', markersize=4, linewidth=1.5, color='tab:blue')
        
        # Add vertical lines at loop boundaries and colored background for fine sweep
        for boundary in loop_boundaries:
            ax[0].axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Highlight fine sweep region
        ax[0].axvspan(fine_sweep_start - 0.5, len(metric_history) - 0.5, 
                      alpha=0.15, color='orange', label='Fine sweep')
        
        ax[0].set_xlabel("Step (each shifter update)")
        ax[0].set_ylabel("Kernel-null depth metric")
        ax[0].set_title("Metric evolution after each ABCD optimization")
        ax[0].set_yscale("log")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        # Plot 2: Phase shifter evolution
        for i in range(n_shifters):
            ax[1].plot(steps, shifter_history[:, i], label=f"Shifter {i+1}", linewidth=1.2)
        
        # Add vertical lines at loop boundaries and colored background for fine sweep
        for boundary in loop_boundaries:
            ax[1].axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Highlight fine sweep region
        ax[1].axvspan(fine_sweep_start - 0.5, len(metric_history) - 0.5, 
                      alpha=0.15, color='orange', label='Fine sweep')
        
        ax[1].set_xlabel("Step (each shifter update)")
        ax[1].set_ylabel("Phase shift value")
        ax[1].set_title("Phase shifter convergence")
        ax[1].grid(True, alpha=0.3)
        if n_shifters <= 8:
            ax[1].legend(loc='best', fontsize=9)

        if save_as:
            from ..utils import save_plot
            save_plot(save_as, "calibration_abcd.png")
        plt.show()

    return {
        "context": ctx,
        "history": np.array(metric_history),
        "history_shifters": np.array(shifter_history),
        "final_phi": chip.φ.copy(),
        "loop_boundaries": loop_boundaries,
        "fine_sweep_start": fine_sweep_start,
    }
