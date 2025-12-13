
import sys
from pathlib import Path
import numpy as np
import torch
import astropy.units as u
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from phise.classes.context import Context
from analysis.neural_calibration import get_phase_visibility_map_full

def test_correlation():
    print("--- Setting up Context ---")
    ctx = Context.get_VLTI()
    ctx.Γ = 0 * u.nm # No atmospheric noise for this test
    
    n_shifters = len(ctx.interferometer.chip.φ)
    print(f"Number of shifters: {n_shifters}")
    
    # Generate random true phases
    rng = np.random.default_rng(42)
    true_phases = rng.uniform(0, 2*np.pi, n_shifters)
    
    print(f"True Phases: {true_phases}")
    
    # Apply to chip (via base_phases arg in get_phase_visibility_map_full? 
    # No, the function takes base_phases argument now)
    
    wavelength = ctx.interferometer.λ
    true_opd = (true_phases / (2*np.pi)) * wavelength.to(u.m).value * u.m
    
    print("--- Generating Map ---")
    # This calls the scanning logic (FFT etc)
    phases_vec, vis_vec = get_phase_visibility_map_full(ctx, n_steps=20, base_phases=true_opd)
    
    print(f"Extracted Phases Vector Shape: {phases_vec.shape}")
    
    # features structure:
    # 1-input combos: 4 combos * 14 shifters * 4 outputs = 224
    # 2-input combos: 6 combos * 14 shifters * 4 outputs = 336
    # ...
    
    # Let's look at 2-input combos.
    # Inputs: [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]
    # For combination [0,1], we expect interference.
    # Phase difference should be related to phi[0] - phi[1] (if shifters map to inputs simply)
    # The shifters are likely inside the chip.
    # But usually shifters are on the input arms?
    
    # Let's look at correlation between the vector and the true phases
    # simple scatter plot logic
    
    # We don't know the exact mapping, but if there is information, 
    # some features should correlate with some true phases.
    
    # Let's check max correlation for each true phase against the feature vector
    correlations = []
    
    print("\n--- Correlation Analysis ---")
    for i in range(n_shifters):
        # Create a "probe" vector that is just this phase
        # We can't do simple correlation because features are non-linear combos?
        # But for 2-input, phase is phi_i - phi_j.
        # So "feature - true_phase" should be constant-ish?
        pass

    # A simpler check: 
    # If we change ONE phase slightly, does the feature vector change?
    # And specifically, do the features corresponding to that shifter change?
    
    # Let's verify sensitivity.
    
    # 1. Base 
    f1, v1 = get_phase_visibility_map_full(ctx, n_steps=20, base_phases=true_opd)
    
    # 2. Perturb shifter 0
    delta = 0.1 # radians
    perturbed_phases = true_phases.copy()
    perturbed_phases[0] += delta
    perturbed_opd = (perturbed_phases / (2*np.pi)) * wavelength.to(u.m).value * u.m
    
    f2, v2 = get_phase_visibility_map_full(ctx, n_steps=20, base_phases=perturbed_opd)
    
    diff = f2 - f1
    # Wrap diff to [-pi, pi]
    diff = (diff + np.pi) % (2*np.pi) - np.pi
    
    print(f"\nSensitivity to Shifter 0 (delta={delta}):")
    print(f"Max absolute changes in features: {np.max(np.abs(diff))}")
    print(f"Number of changing features (>0.01): {np.sum(np.abs(diff) > 0.01)}")
    
    if np.max(np.abs(diff)) < 0.01:
        print("CRITICAL: Features did not change when input phase changed!")
    else:
        print("SUCCESS: Features responded to phase change.")
        
    # Check if the change matches delta in some features
    # Ideally, for some features, the change should be exactly delta (or -delta)
    near_delta = np.isclose(np.abs(diff), delta, atol=0.05)
    print(f"Features changing by exactly delta: {np.sum(near_delta)}")
    
    if np.sum(near_delta) > 0:
        print("GOOD: Some features track the phase change directly.")
        print(f"Indices: {np.where(near_delta)[0]}")
    else:
        print("WARNING: No feature tracks the phase change 1:1.")
        print(f"Distribution of changes: {diff[np.abs(diff)>0.01]}")

if __name__ == "__main__":
    test_correlation()
