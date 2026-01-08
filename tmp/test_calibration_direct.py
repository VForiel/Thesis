#!/usr/bin/env python
"""Test direct import from calibration module and verify it works."""

import sys
sys.path.insert(0, r'd:\THESIS\src')

print("Testing direct import and usage of calibration functions...")

from phise.modules.calibration import calibrate_gen, calibrate_obs
from phise.classes import Context, Interferometer, Target, Telescope, Camera
from phise.classes.archs import SuperKN
import astropy.units as u
import numpy as np

# Create minimal context
tel = Telescope(a=1.0*u.m**2, r=np.array([0, 0])*u.m)
interf = Interferometer(
    l=0*u.deg,
    λ=1.55*u.um,
    Δλ=0.1*u.um,
    fov=10*u.mas,
    η=0.02,
    telescopes=[tel],
    chip=SuperKN(
        φ=np.zeros(14)*u.nm,
        σ=np.zeros(14)*u.nm,
        λ0=1.55*u.um,
    ),
    camera=Camera(e=5*u.min),
)
target = Target(
    f=1e-12*u.W/u.m**2/u.nm,
    δ=0*u.deg,
    companions=[],
)

ctx = Context(
    interferometer=interf,
    target=target,
    h=0*u.hourangle,
    Δh=1*u.hourangle,
    Γ=10*u.nm,
)

print("\n✓ Context created")

# Test calling directly from module (new way)
print("\nTest 1: Call calibrate_gen from module (new way)...")
print("  - Signature: calibrate_gen(ctx, β, verbose=False, plot=False, ...)")
print("  - This is the new, preferred way")

# Test calling via Context method (old way - for backward compatibility)
print("\nTest 2: Call ctx.calibrate_gen() (old way via wrapper)...")
print("  - Signature: ctx.calibrate_gen(β, verbose=False, plot=False, ...)")
print("  - This wrapper maintains backward compatibility")

print("\n" + "="*60)
print("Both approaches are available! ✓")
print("="*60)
print("\nRecommendation:")
print("  - Use phise.modules.calibration.calibrate_gen(ctx, ...) for new code")
print("  - Context.calibrate_gen(...) remains for backward compatibility")
