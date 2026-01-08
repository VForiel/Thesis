#!/usr/bin/env python
"""Test updated calibrate_abcd with improved plotting."""

import sys
sys.path.insert(0, r'd:\THESIS\src')

from phise.modules.calibration import calibrate_abcd
from phise.classes import Context, Interferometer, Target, Camera
from phise.classes.archs import SuperKN
from phise.classes import telescope
import astropy.units as u
import numpy as np

print("Creating 4-telescope context...")
interf = Interferometer(
    l=0*u.deg,
    λ=1.55*u.um,
    Δλ=0.1*u.um,
    fov=10*u.mas,
    η=0.02,
    telescopes=telescope.get_VLTI_UTs(),
    chip=SuperKN(
        φ=np.zeros(14)*u.nm,
        σ=np.zeros(14)*u.nm,
        λ0=1.55*u.um,
    ),
    camera=Camera(e=5*u.min),
)
target = Target(f=1e-12*u.W/u.m**2/u.nm, δ=0*u.deg, companions=[])

ctx = Context(
    interferometer=interf,
    target=target,
    h=0*u.hourangle,
    Δh=1*u.hourangle,
    Γ=10*u.nm,
)

print("Running calibrate_abcd with n_loops=1, n_final_samples=16...")
res = calibrate_abcd(ctx, n_loops=1, n_final_samples=16, plot=True, verbose=False)

print("\nResult summary:")
print(f"  Context name: {res['context'].name}")
print(f"  Metric history shape: {np.array(res['history']).shape}")
print(f"  Shifter history shape: {res['history_shifters'].shape}")
print(f"  Final phi shape: {res['final_phi'].shape}")
print(f"  Loop boundaries: {res['loop_boundaries']}")
print(f"  Fine sweep start: {res['fine_sweep_start']}")
print(f"  Total steps: {len(res['history'])}")
print(f"  n_loops * n_shifters + n_shifters = 1*14 + 14 = {1*14 + 14}")
print("\n[PASSED] Test passed!")
