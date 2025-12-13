
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

import numpy as np
import astropy.units as u
from phise.classes.archs import SuperKN
from phise.modules import phase

def test_vectorization():
    print("Testing vectorization...")
    λ = 1.55e-6 * u.m
    
    # Initialize SuperKN directly
    n_shifters = 14
    phi_init = np.zeros(n_shifters) * u.m
    sigma_init = np.zeros(n_shifters) * u.m
    chip = SuperKN(φ=phi_init, σ=sigma_init, λ0=λ)
    
    # Random phases
    np.random.seed(42)
    chip.φ = np.random.uniform(0, 1e-6, 14) * u.m
    chip.σ = np.random.normal(0, 1e-9, 14) * u.m
    
    # Input field (single)
    ψ = np.array([1+0j, 1+0j, 1+0j, 1+0j]) * 0.5
    
    # 1. Single run
    print("Running single...")
    out_single = chip.get_output_fields(ψ, λ)
    print("Single output (first 3):", out_single[:3])
    
    # 2. Batch run (size 1)
    print("Running batch (N=1)...")
    phi_batch = chip.φ.to(u.m).value.reshape(1, 14) * u.m
    # We need to explicitly call the batch path? 
    # Chip.get_output_fields logic: if φ has batch dim...
    
    out_batch_1 = chip.get_output_fields(ψ, λ, φ=phi_batch)
    print("Batch (N=1) output shape:", out_batch_1.shape)
    print("Batch output (first 3):", out_batch_1[0, :3])
    
    diff = np.abs(out_single - out_batch_1[0])
    print("Max diff (Single vs Batch N=1):", np.max(diff))
    
    if np.max(diff) > 1e-10:
        print("FAIL: Batch N=1 result mismatches Single result!")
    else:
        print("PASS: Batch N=1 matches.")
        
    # 3. Batch run (N=10) with varying phi
    print("\nRunning batch (N=10)...")
    n_batch = 10
    phi_batch_var = np.tile(chip.φ.to(u.m).value, (n_batch, 1))
    # Perturb each
    for i in range(n_batch):
        phi_batch_var[i, 0] += i * 1e-7 # Change 1st shifter
        
    phi_batch_var = phi_batch_var * u.m
    
    out_batch_var = chip.get_output_fields(ψ, λ, φ=phi_batch_var)
    print("Batch output shape:", out_batch_var.shape)
    
    # Check if outputs vary
    start_vals = out_batch_var[0]
    end_vals = out_batch_var[-1]
    print("First item output:", start_vals[:3])
    print("Last item output: ", end_vals[:3])
    
    if np.allclose(start_vals, end_vals):
        print("FAIL: Outputs are identical despite varying inputs!")
    else:
        print("PASS: Outputs vary with input.")

if __name__ == "__main__":
    test_vectorization()
