#!/usr/bin/env python
"""Quick test to verify calibration module refactoring works correctly."""

import sys
sys.path.insert(0, r'd:\THESIS\src')

# Test 1: Import calibration functions from module
print("Test 1: Importing calibration functions from phise.modules.calibration...")
try:
    from phise.modules.calibration import calibrate_gen, calibrate_obs
    print("✓ Successfully imported calibrate_gen and calibrate_obs from phise.modules.calibration")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Import Context and verify it has the wrapper methods
print("\nTest 2: Verifying Context has wrapper methods...")
try:
    from phise.classes import Context
    assert hasattr(Context, 'calibrate_gen'), "Context missing calibrate_gen method"
    assert hasattr(Context, 'calibrate_obs'), "Context missing calibrate_obs method"
    print("✓ Context has both calibrate_gen and calibrate_obs methods")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Create a simple context and verify the wrappers work
print("\nTest 3: Creating a simple context and testing wrapper methods...")
try:
    from phise.classes import Interferometer, Target, Telescope, Camera
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
    
    print("✓ Context created successfully")
    print(f"  - Context.calibrate_gen is callable: {callable(ctx.calibrate_gen)}")
    print(f"  - Context.calibrate_obs is callable: {callable(ctx.calibrate_obs)}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
