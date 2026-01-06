# THESIS Analysis Notebooks

## Overview

This folder contains **13 executable Jupyter notebooks** for PHISE kernel-nuller analysis. Each notebook mirrors a module in `/src/analysis/` and is intended for reproducible, physics-grounded exploration.

## Purpose

Provide clear, runnable examples to:
- Explain kernel-nulling physics and performance
- Evaluate detection statistics and noise sources
- Demonstrate calibration workflows (classical and ML)
- Support commissioning-style manual control

## Notebook Inventory

```
analysis/
├── README.md                        ← This file
├── calibration.ipynb                ← Classical piston calibration loop
├── data_representations.ipynb       ← Output distributions and moments
├── demonstration.ipynb              ← End-to-end workflow (start here)
├── distribution_model.ipynb         ← Statistical modeling of outputs
├── distrib_test_statistics.ipynb    ← Detection metrics and ROC curves
├── manual_control.ipynb             ← Interactive control patterns
├── neural_calibration.ipynb         ← ML-based calibration
├── noise_sensitivity.ipynb          ← Noise sources and impact
├── projected_telescopes.ipynb       ← Baseline geometry and projections
├── sky_contribution.ipynb           ← Thermal background and sky noise
├── temporal_response.ipynb          ← Time evolution of kernels
├── transmission_maps.ipynb          ← Null depth maps vs angle
└── wavelength_scan.ipynb            ← Spectral response sweeps
```

## Quick Start

1. Launch Jupyter inside the project root: `jupyter notebook analysis/demonstration.ipynb`.
2. Run cells in order; adjust parameters (`Context.get_VLTI()`, `ctx.Γ`, `ctx.h`, companion contrast) to explore sensitivity.
3. For a physics-first path: `transmission_maps.ipynb` → `sky_contribution.ipynb` → `distrib_test_statistics.ipynb`.
4. For calibration: `calibration.ipynb` → `neural_calibration.ipynb` → `manual_control.ipynb`.

## Software Prereqs

```bash
pip install -e .          # from repository root
pip install jupyter
```

Dependencies used in notebooks: `phise`, `astropy`, `numpy`, `matplotlib` (already pulled by the project).

## Tips

- All physical parameters use `astropy.units`; avoid raw floats for lengths, angles, or wavelengths.
- Save figures with `plt.savefig('figure.png', dpi=150, bbox_inches='tight')` when needed.
- For heavy runs (`n > 1e4` samples), reduce grid resolution or wavelength sampling; numba JIT is already enabled in PHISE core routines.

## Support

- Documentation: [docs/](../docs)
- Navigation guide: [NOTEBOOKS_GUIDE.md](../analysis/NOTEBOOKS_GUIDE.md)
- Agent logs: [.github/agent-logs/](../.github/agent-logs)

---

Created: 2025-11-26  
Last updated: 2025-11-26  
Status: Maintained  
Version: PHISE 0.1.0
