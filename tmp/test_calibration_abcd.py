#!/usr/bin/env python
"""Smoke test for calibrate_abcd algorithm."""

import sys
sys.path.insert(0, r'd:\THESIS\src')

from phise.modules.calibration import calibrate_abcd
from phise.classes import Context, Interferometer, Target, Telescope, Camera
from phise.classes.archs import SuperKN
from phise.classes import telescope
import astropy.units as u
import numpy as np

# Build minimal 4-telescope context

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

res = calibrate_abcd(ctx, n_loops=1, n_final_samples=8, plot=False, verbose=False)

print("Result keys:", res.keys())
print("Final phi shape:", res["final_phi"].shape)
print("History shape:", res["history"].shape)
print("Context name:", res["context"].name)
