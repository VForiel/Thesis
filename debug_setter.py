
import astropy.units as u
import numpy as np
from phise import Context, Target
import copy

def debug_setter():
    print("Initializing Context...")
    ctx = Context.get_VLTI()
    print(f"Context initialized. Target parent_ctx: {ctx.target.parent_ctx}")
    
    if ctx.target.parent_ctx is ctx:
        print("ctx.target.parent_ctx IS ctx (Correct)")
    else:
        print(f"ctx.target.parent_ctx IS NOT ctx. It is {ctx.target.parent_ctx}")

    initial_pf = ctx.pf
    print(f"Initial pf: {initial_pf}")

    print("\nModifying target flux using /= ...")
    old_f = ctx.target.f
    ctx.target.f /= 100
    new_f = ctx.target.f
    
    print(f"Old f: {old_f}")
    print(f"New f: {new_f}")
    
    if old_f != new_f:
        print("Target flux value updated.")
    else:
        print("Target flux value NOT updated.")

    new_pf = ctx.pf
    print(f"New pf: {new_pf}")

    if np.all(initial_pf != new_pf):
        print("Photon flux (pf) updated automatically.")
    else:
        print("Photon flux (pf) NOT updated automatically.")

    print("\nChecking deepcopy behavior...")
    ctx_copy = copy.deepcopy(ctx)
    print(f"Copy created. Copy target parent_ctx: {ctx_copy.target.parent_ctx}")
    
    if ctx_copy.target.parent_ctx is ctx_copy:
        print("ctx_copy.target.parent_ctx IS ctx_copy (Correct)")
    else:
        print(f"ctx_copy.target.parent_ctx IS NOT ctx_copy. It is {ctx_copy.target.parent_ctx}")

    print("\nModifying copy target flux...")
    initial_copy_pf = ctx_copy.pf
    ctx_copy.target.f /= 10
    new_copy_pf = ctx_copy.pf
    
    print(f"Initial copy pf: {initial_copy_pf}")
    print(f"New copy pf: {new_copy_pf}")
    
    if np.all(initial_copy_pf != new_copy_pf):
        print("Copy photon flux updated automatically.")
    else:
        print("Copy photon flux NOT updated automatically.")

if __name__ == "__main__":
    debug_setter()
