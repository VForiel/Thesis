"""
Script de diagnostic pour comprendre le facteur manquant entre
les modèles numérique et analytique.
"""

import warnings
warnings.filterwarnings('ignore')

from phise import Context
import astropy.units as u
import numpy as np

# Configuration
ctx = Context.get_VLTI()
ctx.interferometer.chip.σ = np.zeros(14) * u.nm
ctx.target.companions[0].c = 1e-2
ctx.Γ = 0 * u.nm  # Pas de bruit
ctx.monochromatic = True
ctx.interferometer.camera.ideal = True

print("=" * 80)
print("ANALYSE DU FACTEUR MANQUANT")
print("=" * 80)

# --- 1. Paramètres du contexte ---
print("\n1. PARAMÈTRES DU CONTEXTE")
print(f"   Longueur d'onde λ: {ctx.interferometer.λ}")
print(f"   Bande passante Δλ: {ctx.interferometer.Δλ}")
print(f"   Temps d'exposition e: {ctx.interferometer.camera.e}")
print(f"   Mode monochromatiqu: {ctx.monochromatic}")

# --- 2. Flux d'entrée ---
print("\n2. FLUX D'ENTRÉE")
α = np.sum(ctx.pf).value
β = α * ctx.target.companions[0].c
print(f"   Flux étoile α: {α:.3e} photons/s")
print(f"   Flux planète β: {β:.3e} photons/s")

# --- 3. Champs d'entrée ---
print("\n3. CHAMPS D'ENTRÉE")
ψi = ctx.get_input_fields()
print(f"   Champs étoile: {ψi[0]}")
print(f"   |ψ[0]|²: {np.abs(ψi[0][0])**2:.3e} (devrait être α/4 = {α/4:.3e})")
print(f"   Somme |ψ_étoile|²: {np.sum(np.abs(ψi[0])**2):.3e} (devrait être α = {α:.3e})")

# --- 4. Propagation numérique ---
print("\n4. PROPAGATION NUMÉRIQUE")
out_fields_star = ctx.interferometer.chip.get_output_fields(ψ=ψi[0], λ=ctx.interferometer.λ)
out_fields_planet = ctx.interferometer.chip.get_output_fields(ψ=ψi[1], λ=ctx.interferometer.λ)

print(f"   Champ bright étoile: {out_fields_star[0]}")
print(f"   |E_bright_étoile|²: {np.abs(out_fields_star[0])**2:.3e}")
print(f"   Attendu (si 4 télescopes en phase): 4α = {4*α:.3e}")
print(f"   Ratio: {np.abs(out_fields_star[0])**2 / (4*α):.3f}")

# --- 5. Observation complète ---
print("\n5. OBSERVATION COMPLÈTE (avec caméra)")
outs = ctx.observe()
print(f"   Bright observé: {outs[0]:.3e} photons détectés")
print(f"   Bright attendu (4α × e): {4*α * ctx.interferometer.camera.e.to(u.s).value:.3e}")
print(f"   Facteur: {outs[0] / (4*α * ctx.interferometer.camera.e.to(u.s).value):.3f}")

# --- 6. Analyse de l'intégration spectrale ---
print("\n6. ANALYSE DE L'INTÉGRATION SPECTRALE")
print(f"   Mode monochrome: {ctx.monochromatic}")

# Test avec observe() qui peut faire une intégration spectrale
ctx_poly = Context.get_VLTI()
ctx_poly.interferometer.chip.σ = np.zeros(14) * u.nm
ctx_poly.target.companions[0].c = 1e-2
ctx_poly.Γ = 0 * u.nm
ctx_poly.monochromatic = False  # Mode polychromatique
ctx_poly.interferometer.camera.ideal = True

outs_poly = ctx_poly.observe(spectral_samples=5)
print(f"\n   Bright (monochrome): {outs[0]:.3e}")
print(f"   Bright (polychrome, 5 samples): {outs_poly[0]:.3e}")
print(f"   Ratio poly/mono: {outs_poly[0] / outs[0]:.3f}")

# --- 7. Modèle analytique ---
print("\n7. MODÈLE ANALYTIQUE")
from src.analysis.data_representations import get_B

φ1, φ2, φ3, φ4 = np.angle(ψi[1])
Γ_rad = 0
n_samples = 1

Bs, Bp, B_total = get_B(Γ=Γ_rad, α=α, β=β, φ1=φ1, φ2=φ2, φ3=φ3, φ4=φ4, n=n_samples, ctx=ctx)

print(f"   Bright analytique (étoile): {Bs[0]:.3e}")
print(f"   Bright analytique (planète): {Bp[0]:.3e}")
print(f"   Bright analytique (total): {B_total[0]:.3e}")
print(f"\n   Différence avec numérique:")
print(f"      Numérique: {outs[0]:.3e}")
print(f"      Analytique: {B_total[0]:.3e}")
print(f"      Facteur: {outs[0] / B_total[0]:.3f}")

# --- 8. Analyse du facteur ---
print("\n8. DÉCOMPOSITION DU FACTEUR")
e = ctx.interferometer.camera.e.to(u.s).value
facteur_total = outs[0] / B_total[0]
print(f"   Facteur total observé: {facteur_total:.3f}")
print(f"   Temps d'exposition e: {e:.1f} s")
print(f"   Facteur / e: {facteur_total / e:.3f}")

# Vérification du calcul analytique manuel
σ = np.zeros(4)
E_star = np.sqrt(α/16) * (np.exp(1j*σ[0]) + np.exp(1j*σ[1]) + np.exp(1j*σ[2]) + np.exp(1j*σ[3]))
I_star_correct = np.abs(E_star)**2
print(f"\n   Calcul analytique correct (avec √(α/16)):")
print(f"      |E_étoile|²: {I_star_correct:.3e} (devrait être α = {α:.3e})")
print(f"      Ratio: {I_star_correct / α:.3f}")

E_star_wrong = np.sqrt(α/4) * (np.exp(1j*σ[0]) + np.exp(1j*σ[1]) + np.exp(1j*σ[2]) + np.exp(1j*σ[3]))
I_star_wrong = np.abs(E_star_wrong)**2
print(f"\n   Calcul analytique INCORRECT (avec √(α/4)):")
print(f"      |E_étoile|²: {I_star_wrong:.3e} (= 4α)")
print(f"      Ratio par rapport à α: {I_star_wrong / α:.3f}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("Le modèle analytique utilise √(α/16) qui est correct.")
print("Le facteur manquant est probablement le temps d'exposition × autre chose.")
print("=" * 80)
