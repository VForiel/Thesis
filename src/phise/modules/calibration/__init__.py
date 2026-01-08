"""Calibration modules for photonic interferometers.

Provides trial-and-error and obstruction-based calibration algorithms
for optimizing phase shifter values in kernel nulling interferometers.
"""

from .trialerror import calibrate_gen
from .obstruction import calibrate_obs
from .abcd import calibrate_abcd

__all__ = ["calibrate_gen", "calibrate_obs", "calibrate_abcd"]
