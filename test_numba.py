
import sys
import os
sys.path.append(os.path.abspath("src"))

from phise.classes.context import Context
from astropy import units as u
import numpy as np

def test_observe():
    print("Setting up context...")
    ctx = Context.get_VLTI()
    ctx.h = 0 * u.hourangle
    ctx.Δh = 1 * u.hourangle
    ctx.Γ = 10 * u.nm
    
    print("Observing...")
    outs = ctx.observe()
    print("Observation successful.")
    print("Outputs:", outs)

if __name__ == "__main__":
    test_observe()
