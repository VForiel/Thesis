
import numpy as np
import astropy.units as u
from copy import deepcopy as copy
from phise import Context

def reproduce():
    print("Initializing Context...")
    ctx = Context.get_VLTI()
    # Set known values
    ctx.target.f = 1e11 * u.W / u.m**2 / u.nm # Dummy value to track
    ctx.target.companions[0].c = 0.01
    ctx.monochromatic = True
    ctx.Γ = 10 * u.nm
    
    # Force update to be sure
    ctx._update_pf()
    initial_pf = np.sum(ctx.pf).value
    print(f"Initial PF (sum): {initial_pf:.2e}")
    
    # Create Planet Only context as in data_representations.py
    print("\nCreating Planet Only Context (ctx_po)...")
    ctx_po = copy(ctx)
    scale = 1e12 # Test with 1e12 as in time_evolution
    # scale = 1 # Test with 1 as in instant_distribution (current file)
    
    print(f"Scaling: f /= {scale:.0e}, c *= {scale:.0e}")
    ctx_po.target.f /= scale
    ctx_po.target.companions[0].c *= scale
    
    print(f"ctx_po.target.f: {ctx_po.target.f:.2e}")
    print(f"ctx_po.target.companions[0].c: {ctx_po.target.companions[0].c:.2e}")

    # Check Companion Identity
    if ctx.target.companions[0] is ctx_po.target.companions[0]:
        print("-> Companion objects are IDENTICAL (Shallow Copy of elements?)")
    else:
        print("-> Companion objects are DIFFERENT (Deep Copy)")

    # Check if Original Companion was modified
    print(f"Original Companion c: {ctx.target.companions[0].c:.2e}")
    if ctx.target.companions[0].c != 0.01:
        print("-> Original Companion WAS modified (Side Effect!)")
    else:
        print("-> Original Companion was NOT modified.")
    
    # Check if pf is updated automatically
    po_pf = np.sum(ctx_po.pf).value
    print(f"ctx_po.pf (sum) [Immediate]: {po_pf:.2e}")
    
    if po_pf == initial_pf:
        print("-> pf was NOT updated automatically.")
    else:
        print("-> pf WAS updated automatically.")
        
    # Check what get_input_fields uses
    print("\nChecking get_input_fields...")
    # This calls get_input_fields which uses self.pf
    # We want to see the amplitude of the companion field
    
    # In get_input_fields:
    # pf = self.pf
    # pf_c = pf * c.c
    # returns s * sqrt(pf_c)
    
    # If pf is NOT updated: pf = Initial PF
    # c is updated: c = Initial C * Scale
    # pf_c = Initial PF * (Initial C * Scale) = (Initial PF * Initial C) * Scale
    # This is HUGE.
    
    # If pf IS updated: pf = Initial PF / Scale
    # c is updated: c = Initial C * Scale
    # pf_c = (Initial PF / Scale) * (Initial C * Scale) = Initial PF * Initial C
    # This is EXPECTED.
    
    input_fields = ctx_po.get_input_fields()
    # input_fields[0] is Star (should be negligible)
    # input_fields[1] is Planet
    
    star_field = input_fields[0]
    planet_field = input_fields[1]
    
    star_intensity = np.sum(np.abs(star_field)**2)
    planet_intensity = np.sum(np.abs(planet_field)**2)
    
    print(f"Star Intensity (sum |psi|^2): {star_intensity:.2e}")
    print(f"Planet Intensity (sum |psi|^2): {planet_intensity:.2e}")
    
    expected_planet_intensity = initial_pf * 0.01
    print(f"Expected Planet Intensity (Initial PF * 0.01): {expected_planet_intensity:.2e}")
    
    if np.isclose(planet_intensity, expected_planet_intensity):
        print("-> Planet Intensity matches Expected.")
    else:
        print(f"-> Planet Intensity MISMATCH. Ratio: {planet_intensity/expected_planet_intensity:.2e}")

    # Check if observe() triggers update
    print("\nCalling observe()...")
    try:
        ctx_po.observe()
        po_pf_after = np.sum(ctx_po.pf).value
        print(f"ctx_po.pf (sum) [After observe]: {po_pf_after:.2e}")
        
        if po_pf_after != po_pf:
             print("-> observe() triggered pf update.")
        else:
             print("-> observe() did NOT trigger pf update.")
             
    except Exception as e:
        print(f"observe() failed: {e}")

if __name__ == "__main__":
    reproduce()
