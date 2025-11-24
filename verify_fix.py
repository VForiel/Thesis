
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from unittest.mock import MagicMock
sys.modules['ipywidgets'] = MagicMock()
sys.modules['sympy'] = MagicMock()
sys.modules['fitter'] = MagicMock()

import numpy as np
import astropy.units as u
from phise import Context
from analysis.data_representations import instant_distribution

# Setup context
ctx = Context.get_VLTI()
ctx.interferometer.chip.σ = np.zeros(14) * u.um
ctx.target.companions[0].c = 0.1
ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle

# Run instant_distribution
print("Running instant_distribution...")
data, data_so = instant_distribution(ctx=ctx, n=5, compare=False, sync_plots=False)

print("Verification complete.")
