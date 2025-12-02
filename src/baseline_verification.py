
import sys
from unittest.mock import MagicMock

# Mock ipywidgets and sympy to avoid import error
sys.modules["ipywidgets"] = MagicMock()
sys.modules["sympy"] = MagicMock()
sys.modules["fitter"] = MagicMock()

print(f"DEBUG: sympy in sys.modules: {'sympy' in sys.modules}")
try:
    import sympy
    print(f"DEBUG: Imported sympy: {sympy}")
except ImportError as e:
    print(f"DEBUG: Failed to import sympy directly: {e}")

import numpy as np
import astropy.units as u
from analysis.data_representations import compute_analytical_distrib
from phise import Context

def test_analytical_new():
    # Setup parameters
    Γ = 100 * u.nm
    λ0 = 1.55e-6
    α = 1.0
    β = 0.01
    φ1, φ2, φ3, φ4 = 0, np.pi/3, np.pi/4, np.pi/6
    n = 100
    
    # Generate random errors for consistency
    opd_errors = np.random.normal(0, Γ.to(u.m).value, (n, 4))
    
    # Mock context for wavelength info
    ctx = Context.get_VLTI()
    ctx.monochromatic = True 
    ctx.interferometer.λ = 1.55 * u.um
    
    # Run new function
    brights, kernels = compute_analytical_distrib(n, ctx, opd_errors, α, β, φ1, φ2, φ3, φ4)
    
    print(f"New Bright mean: {np.mean(brights)}")
    print(f"New Kernels mean: {np.mean(kernels, axis=0)}")
    
    # Check shapes
    assert brights.shape == (n,)
    assert kernels.shape == (n, 3)
    
    print("Verification successful!")

if __name__ == "__main__":
    test_analytical_new()
