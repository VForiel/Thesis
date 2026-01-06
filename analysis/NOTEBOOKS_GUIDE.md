# Analysis Notebook Navigation Guide

## Overview

Thirteen executable notebooks live in `/analysis` and mirror the analytical modules in `/src/analysis/`. Use this guide to pick the right entry point and keep a coherent learning path.

---

## Recommended Paths

**Fast orientation (30 min)**
```
demonstration.ipynb
    ↓ end-to-end flow
data_representations.ipynb
    ↓ distributions and quick stats
neural_calibration.ipynb (overview cells)
```

**Physics-first (≈2h)**
```
transmission_maps.ipynb
    ↓ null depth vs angle
sky_contribution.ipynb
    ↓ thermal background and noise
distrib_test_statistics.ipynb
    ↓ detection metrics and ROC curves
neural_calibration.ipynb
    ↓ calibration comparison (classical vs ML)
```

**Calibration & control (≈1.5h)**
```
calibration.ipynb
    ↓ piston correction loop
neural_calibration.ipynb
    ↓ ML alternative
manual_control.ipynb
    ↓ commissioning-style operations
```

**Geometry & observation planning (≈1h)**
```
projected_telescopes.ipynb
    ↓ projected baselines
transmission_maps.ipynb
    ↓ transmission impact
temporal_response.ipynb
    ↓ time evolution
wavelength_scan.ipynb
    ↓ spectral behavior
```

---

## Thematic Classification

**Foundational analysis (4)**
| Notebook | Topic | Level |
|----------|-------|-------|
| data_representations.ipynb | Output distributions and trends | Intermediate |
| distrib_test_statistics.ipynb | Detection statistics, ROC | Advanced |
| transmission_maps.ipynb | Null depth maps, chromaticity | Advanced |
| sky_contribution.ipynb | Thermal background and noise budget | Intermediate |

**Calibration and control (3)**
| Notebook | Topic | Level |
|----------|-------|-------|
| calibration.ipynb | Piston correction loop | Intermediate |
| neural_calibration.ipynb | ML-based calibration | Advanced |
| manual_control.ipynb | Interactive control surface | Beginner |

**Geometry and observation (4)**
| Notebook | Topic | Level |
|----------|-------|-------|
| projected_telescopes.ipynb | Baseline projection and (u,v) coverage | Beginner |
| temporal_response.ipynb | Time evolution of kernel outputs | Intermediate |
| wavelength_scan.ipynb | Spectral response and chromatic trends | Intermediate |
| transmission_maps.ipynb | (see Foundational) | - |

**Pedagogy and demo (2)**
| Notebook | Topic | Level |
|----------|-------|-------|
| demonstration.ipynb | End-to-end walkthrough | Beginner |
| distribution_model.ipynb | Statistical modeling overview | Intermediate |

**Noise focus (1)**
| Notebook | Topic | Level |
|----------|-------|-------|
| noise_sensitivity.ipynb | Noise sources and mitigation levers | Beginner |

---

## Notebook Glossary (high level)

- **data_representations** — Instantaneous and time-evolving distributions; helper routines `instant_distribution`, `time_evolution`.
- **distrib_test_statistics** — Binary detection metrics, ROC plotting, SNR definitions.
- **transmission_maps** — Interferometric transmission grids; null depth formula $N_{depth} = -2.5 \log_{10}(T)$.
- **sky_contribution** — Planck radiance, thermal background modeling, SNR impact.
- **neural_calibration** — Network architecture, training loops, robustness checks.
- **calibration** — Classical OPD correction loop and residual errors.
- **projected_telescopes** — Baseline projection on sky and hour angle evolution; rotation matrices.
- **temporal_response** — Time-series generation and stability metrics.
- **wavelength_scan** — Spectral sweep of null depth and performance.
- **demonstration** — Start-to-finish workflow for the kernel nuller.
- **noise_sensitivity** — Comparative table of noise sources and mitigation options.
- **distribution_model** — Statistical modeling of outputs (analytical vs empirical).
- **manual_control** — Interactive control patterns for commissioning exercises.

---

## FAQs — Which Notebook?

- How does the kernel nuller work? → `demonstration.ipynb`, then `transmission_maps.ipynb`.
- What limits detection? → `distrib_test_statistics.ipynb` + `sky_contribution.ipynb`.
- How to improve calibration? → `neural_calibration.ipynb`.
- Why does performance change with wavelength? → `wavelength_scan.ipynb` + `transmission_maps.ipynb`.
- What are the dominant noise terms? → `noise_sensitivity.ipynb`.
- Which notebook for commissioning? → `manual_control.ipynb` + `calibration.ipynb`.

---

## Usage Tips

- Keep imports consistent: rely on `phise` installed in editable mode; avoid modifying `sys.path` when possible.
- Respect units with `astropy.units`; never pass raw floats for physical quantities.
- Random processes vary run to run; set `np.random.seed(...)` for reproducibility.
- For heavy grids or long time series, reduce sampling or wavelength count to keep runtime reasonable.
- Save figures explicitly with `plt.savefig(..., dpi=150, bbox_inches='tight')` when building reports.

---

## Quick Summary

| Need | Notebook(s) | Time |
|------|-------------|------|
| Overview | demonstration.ipynb | 5 min |
| Full physics tour | transmission_maps + sky_contribution + distrib_test_statistics | ~2h |
| Calibration focus | calibration + neural_calibration + manual_control | ~1.5h |
| Noise review | noise_sensitivity | 15 min |
| Geometry planning | projected_telescopes + transmission_maps | ~45 min |

---

Last updated: 2025-11-26  
Status: Maintained
