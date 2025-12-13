
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

import numpy as np
from scipy.optimize import curve_fit
import time
import astropy.units as u
from phise.classes.context import Context
from copy import deepcopy

# --- Original Function ---
def sine_func_simple(x, A, C, E):
    return A * np.sin(x + C) + E

def get_phase_visibility_map_full(context: Context, n_steps: int = 20):
    # (Simplified version of the one in neural_calibration.py for benchmarking logic)
    # We will just simulate the "fitting" part because that's what we want to optimize.
    # The generation part (get_output_fields) is matrix algebra, likely fast.
    
    # Let's simulate data
    n_shifters = 14
    n_outputs = 4
    n_combinations = 15
    
    # Simulate fluxes: (15 combos) x (14 shifters) x (20 steps) x (4 outputs)
    # Total fits: 15 * 14 * 4 = 840 fits per sample.
    
    phase_range_rad = np.linspace(0, 2*np.pi, n_steps)
    
    # Generate synthetic noisy sine waves
    # A * sin(x + C) + E
    A = 0.5
    C = 1.0
    E = 0.5
    y_true = sine_func_simple(phase_range_rad, A, C, E)
    
    y_data_batch = np.tile(y_true, (n_combinations * n_shifters * n_outputs, 1))
    # Add noise
    y_data_batch += np.random.normal(0, 0.001, y_data_batch.shape)
    
    # Timing fitting
    start = time.time()
    
    for i in range(y_data_batch.shape[0]):
        y_data = y_data_batch[i]
        visibility = np.ptp(y_data)
        
        A_guess = visibility / 2
        E_guess = np.mean(y_data)
        C_guess = 0.0
        p0 = [A_guess, C_guess, E_guess]
        
        try:
            popt, _ = curve_fit(sine_func_simple, phase_range_rad, y_data, p0=p0, maxfev=2000)
            phase = popt[1] % (2*np.pi)
        except:
            pass
            
    end = time.time()
    return end - start

# --- Fast Function ---
def get_phase_visibility_map_fast(context: Context, n_steps: int = 20):
    # Simulate data same as above
    n_shifters = 14
    n_outputs = 4
    n_combinations = 15
    
    phase_range_rad = np.linspace(0, 2*np.pi, n_steps)
    # Check if endpoint included
    # np.linspace(0, 2pi, 20) -> last point is 2pi.
    # We should exclude it for FFT
    
    A = 0.5
    C = 1.0
    E = 0.5
    y_true = sine_func_simple(phase_range_rad, A, C, E)
    
    y_data_batch = np.tile(y_true, (n_combinations * n_shifters * n_outputs, 1))
    y_data_batch += np.random.normal(0, 0.001, y_data_batch.shape) # (840, 20)
    
    start = time.time()
    
    # Vectorized FFT
    # If endpoint=True (default in original code), we should ignore last point or just use it.
    # Use first N-1 points for FFT if endpoint is 2pi
    y_fft_in = y_data_batch[:, :-1] # (840, 19)
    N = n_steps - 1
    
    # Compute DFT component at k=1
    # Y = fft(y)
    Y = np.fft.rfft(y_fft_in, axis=1) # (840, N//2 + 1)
    Y1 = Y[:, 1] # First harmonic (k=1)
    
    # Phase extraction
    # C - pi/2 = angle(Y1)
    phases = (np.angle(Y1) + np.pi/2) % (2*np.pi)
    
    # Visibility (Amplitude)
    # A = 2/N * |Y1|
    # Visibility = 2*A = 4/N * |Y1|
    visibilities = (4/N) * np.abs(Y1)
    
    # Wait, original uses np.ptp() for visibility
    # We can keep np.ptp for visibility if we want strict equivalence, or use harmonic amplitude.
    # Original: visibility = np.ptp(y_data)
    # Let's perform ptp as well for fairness if we keep it
    visibilities_ptp = np.ptp(y_data_batch, axis=1)
    
    end = time.time()
    return end - start, phases[0], visibilities_ptp[0]

if __name__ == "__main__":
    print("Running benchmark...")
    t_orig = get_phase_visibility_map_full(None)
    t_fast, ph, vis = get_phase_visibility_map_fast(None)
    
    print(f"Original Time (840 fits): {t_orig:.4f} s")
    print(f"Fast Time (840 fits): {t_fast:.4f} s")
    print(f"Speedup: {t_orig/t_fast:.2f}x")
    print(f"Test Phase: {ph:.4f} (Expected ~1.0)")
