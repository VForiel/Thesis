
import sys
from unittest.mock import MagicMock
sys.modules['ipywidgets'] = MagicMock()
sys.modules['sympy'] = MagicMock()
sys.modules['fitter'] = MagicMock()

import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from phise import Context
from analysis.data_representations import instant_distribution

# Create a context
ctx = Context.get_VLTI()

# Run instant_distribution
# We use a small n to make it fast
instant_distribution(ctx, n=100, show=False)
