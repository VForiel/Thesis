"""Outils d'analyse et d'affichage pour la bibliothèque PHISE.

Ce module expose les sous-modules d'analyse fournis dans le package
`src.analysis`. Chaque sous-module est organisé en dossier séparé et
contient des utilitaires pour visualiser, mesurer et analyser les 
performances du nuller à noyau et des outils de calibration/demo.
"""
from . import projected_telescopes
from . import transmission_maps
from . import sky_contribution
from . import calibration
from . import manual_control
from . import neural_calibration
from . import data_representations
from . import noise_sensitivity
from . import distribution_model
from . import distrib_test_statistics
from . import temporal_response
from . import demonstration
from . import wavelength_scan

__all__ = [
    'projected_telescopes',
    'transmission_maps',
    'sky_contribution',
    'calibration',
    'manual_control',
    'neural_calibration',
    'data_representations',
    'noise_sensitivity',
    'distribution_model',
    'distrib_test_statistics',
    'temporal_response',
    'demonstration',
    'wavelength_scan',
]