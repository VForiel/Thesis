"""Script de test pour valider les corrections du modèle analytique."""

from phise import Context
import astropy.units as u
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analysis import data_representations

print('=== Test du modèle analytique corrigé ===\n')

# Configuration du contexte
ctx = Context.get_VLTI()
ctx.interferometer.chip.σ = np.zeros(14) * u.nm
ctx.interferometer.camera.e = 1 * u.s
ctx.target.companions[0].c = 1e-2
ctx.Γ = 100 * u.nm
ctx.interferometer.camera.ideal = True
ctx.monochromatic = True
ctx.interferometer.Δλ = 0.01 * u.um

print('Test 1: Validation des cartes de transmission analytiques')
print('Vérification que toutes les valeurs sont ≤ 1...\n')

bright_map, kernel_maps = ctx.get_analytical_transmission_maps(N=50)

print(f'Bright map:')
print(f'  Min: {bright_map.min():.4f}')
print(f'  Max: {bright_map.max():.4f}')
print(f'  Toutes valeurs ≤ 1: {np.all(bright_map <= 1)}')

for i in range(3):
    print(f'\nKernel {i+1} map:')
    print(f'  Min: {kernel_maps[i].min():.4f}')
    print(f'  Max: {kernel_maps[i].max():.4f}')

print('\n' + '='*60)
print('Test 2: Comparaison modèle numérique vs analytique')
print('='*60 + '\n')

# Test avec un petit échantillon pour aller plus vite
_ = data_representations.instant_distribution(ctx, n=1000, r=1, figsize=(18,6), 
                                               log=True, density=True, sync_plots=True)

print('\n=== Tests terminés ===')
