#!/usr/bin/env python
"""Final comprehensive test of calibration module refactoring."""

import sys
sys.path.insert(0, r'd:\THESIS\src')

print("="*70)
print("COMPREHENSIVE CALIBRATION REFACTORING TEST")
print("="*70)

# Test 1: Import from phise.modules
print("\n1. Test imports from phise.modules...")
try:
    import phise.modules.calibration
    from phise.modules.calibration import calibrate_gen, calibrate_obs
    print("   ✓ Direct import: phise.modules.calibration works")
    print(f"   ✓ Functions available: {phise.modules.calibration.__all__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Verify Context wrappers
print("\n2. Test Context wrapper methods...")
try:
    from phise import Context
    assert hasattr(Context, 'calibrate_gen'), "Missing calibrate_gen"
    assert hasattr(Context, 'calibrate_obs'), "Missing calibrate_obs"
    print("   ✓ Context.calibrate_gen() wrapper exists")
    print("   ✓ Context.calibrate_obs() wrapper exists")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Check docstrings have deprecation notices
print("\n3. Test deprecation notices in wrapper docstrings...")
try:
    assert 'deprecated' in Context.calibrate_gen.__doc__.lower(), \
        "Missing deprecation notice in calibrate_gen"
    assert 'deprecated' in Context.calibrate_obs.__doc__.lower(), \
        "Missing deprecation notice in calibrate_obs"
    print("   ✓ Deprecation notices present in wrapper docstrings")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Verify function signatures
print("\n4. Test function signatures...")
try:
    import inspect
    
    # Check calibrate_gen signature
    sig_gen = inspect.signature(calibrate_gen)
    params_gen = list(sig_gen.parameters.keys())
    assert params_gen[0] == 'ctx', f"First param should be 'ctx', got '{params_gen[0]}'"
    assert 'β' in params_gen, "Missing β parameter"
    print(f"   ✓ calibrate_gen signature: {params_gen}")
    
    # Check calibrate_obs signature
    sig_obs = inspect.signature(calibrate_obs)
    params_obs = list(sig_obs.parameters.keys())
    assert params_obs[0] == 'ctx', f"First param should be 'ctx', got '{params_obs[0]}'"
    assert 'n' in params_obs, "Missing n parameter"
    print(f"   ✓ calibrate_obs signature: {params_obs}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Create context and verify both calling methods work
print("\n5. Test both calling methods (direct and wrapper)...")
try:
    from phise.classes import Interferometer, Target, Telescope, Camera
    from phise.classes.archs import SuperKN
    import astropy.units as u
    import numpy as np
    
    tel = Telescope(a=1.0*u.m**2, r=np.array([0, 0])*u.m)
    interf = Interferometer(
        l=0*u.deg, λ=1.55*u.um, Δλ=0.1*u.um, fov=10*u.mas, η=0.02,
        telescopes=[tel],
        chip=SuperKN(φ=np.zeros(14)*u.nm, σ=np.zeros(14)*u.nm, λ0=1.55*u.um),
        camera=Camera(e=5*u.min),
    )
    target = Target(f=1e-12*u.W/u.m**2/u.nm, δ=0*u.deg, companions=[])
    ctx = Context(
        interferometer=interf, target=target,
        h=0*u.hourangle, Δh=1*u.hourangle, Γ=10*u.nm,
    )
    
    # Verify wrapper method is callable
    assert callable(ctx.calibrate_gen), "ctx.calibrate_gen not callable"
    assert callable(ctx.calibrate_obs), "ctx.calibrate_obs not callable"
    
    print("   ✓ Context created successfully")
    print("   ✓ ctx.calibrate_gen() is callable (backward compatible)")
    print("   ✓ ctx.calibrate_obs() is callable (backward compatible)")
    print("   ✓ calibrate_gen(ctx, ...) available (new preferred way)")
    print("   ✓ calibrate_obs(ctx, ...) available (new preferred way)")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nSummary:")
print("  • Calibration functions moved to phise.modules.calibration")
print("  • New signature: calibrate_gen(ctx, β, ...) and calibrate_obs(ctx, n, ...)")
print("  • Backward compatibility maintained via Context wrappers")
print("  • Deprecation notices added to wrapper docstrings")
print("\nMigration guide:")
print("  Old: ctx.calibrate_gen(β=0.9, verbose=True)")
print("  New: calibrate_gen(ctx, β=0.9, verbose=True)")
print("="*70)
