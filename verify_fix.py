
import numpy as np
import astropy.units as u
from phise import Context
import sys
from unittest.mock import MagicMock
sys.modules['ipywidgets'] = MagicMock()
sys.modules['sympy'] = MagicMock()
from src.analysis.data_representations import instant_distribution
from src.analysis.data_representations import instant_distribution
import matplotlib.pyplot as plt

def verify_fix():
    print("Verifying fix for Planet Flux Discrepancy...")
    
    # Setup context similar to numerical_simulation.ipynb
    ctx = Context.get_VLTI()
    ctx.monochromatic = True
    ctx.Γ = 10 * u.nm
    ctx.target.companions[0].c = 0.01
    
    # Run instant_distribution
    # We capture stdout to check the printed signal value
    import io
    import sys
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        # Run with show=False to avoid blocking plot
        instant_distribution(ctx=ctx, n=100, show=False)
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"Error running instant_distribution: {e}")
        return

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    print("Output captured from instant_distribution:")
    print(output)
    
    # Parse output for Planet Signal
    import re
    match = re.search(r"Planet: Contrast: .* Signal: (.*)", output)
    if match:
        signal_str = match.group(1)
        print(f"Found Planet Signal: {signal_str}")
        
        try:
            signal_val = float(signal_str)
            
            # Calculate expected signal
            # Initial Flux * Contrast * Efficiency * Area * Bandwidth * Wavelength / (h*c) * Exposure
            # But simpler: Initial Photon Flux (sum) * Contrast * Exposure
            
            # Re-calculate initial PF
            ctx._update_pf()
            initial_pf_sum = np.sum(ctx.pf).value
            exposure = ctx.interferometer.camera.e.to(u.s).value
            contrast = 0.01
            
            expected_signal = initial_pf_sum * contrast * exposure
            print(f"Expected Signal: {expected_signal:.2e}")
            
            # Check if close
            if np.isclose(signal_val, expected_signal, rtol=0.1): # Allow 10% tolerance for float formatting
                print("SUCCESS: Planet Signal matches Expected Signal.")
            else:
                print(f"FAILURE: Planet Signal mismatch. Ratio: {signal_val/expected_signal:.2e}")
                
        except ValueError:
            print("Could not parse signal value.")
    else:
        print("Could not find Planet Signal in output.")

if __name__ == "__main__":
    verify_fix()
