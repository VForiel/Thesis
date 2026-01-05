# Tunable Kernel Nulling

**Author:** Vincent Foriel
**Supervisor:** Dr. Frantz Martinache, Prof. David Mary

This repository contains the source code, simulation tools, and analysis scripts developed in the context of my PhD thesis.
The research focuses on **Tunable Kernel Nulling** and **Photonic Interferometry** for the direct detection of exoplanets.

## Project Overview

The core of this work is the development of **PHISE** (PHotonic Interferometry Simulation for Exoplanets), a Python package designed for the simulation and analysis of interferometric instruments using layered photonic chips.

PHISE provides:
- High-level classes to model the complete instrument chain (`TelescopeArray`, `Interferometer`, `Chip`, `Camera`).
- Numerical modules for signal propagation, MMI recombiners, and statistical analysis.
- Tools for visualizing instrument responses (transmission maps, projected baselines, nulling outputs).

While the package name `phise` remains used in the codebase for import (e.g., `import phise`), this repository (`THESIS`) encompasses the broader scope of the research project, including experimental scripts, optimization logs, and thesis-specific analysis notebooks.

> ⚠️ **Important Note**
>
> This code is part of an active PhD research project. APIs and functionalities are subject to change as the research evolves.
> The package is currently in a pre-release state `v0.1.1`.

## Requirements and Installation

This project requires **Python 3.11+**.

Dependencies (managed automatically): `numpy`, `astropy`, `scipy`, `matplotlib`, `numba`, `ipywidgets`, `sympy`, etc.

### Installation

**1. Conda Environment (Recommended)**

```powershell
conda env create -f environment.yml
conda activate phise
```

**2. Pip Editable Install (Dev Mode)**

```powershell
pip install -e .
```

## Documentation

Full documentation for the PHISE package is available at: [https://phise.readthedocs.io/](https://phise.readthedocs.io/)

## Design Philosophy

- **Physical Consistency**: All physical quantities use `astropy.units` and are validated to ensure consistency.
- **Performance**: Computationally intensive tasks utilize `numpy` and `numba`.
- **Automation**: Parameter changes in high-level contexts automatically trigger re-computations of derived values (e.g., projected baselines).

## Citation

If you use this code or findings in your scientific work, please cite the related PhD thesis or publications by Vincent Foriel.

## Contact & Contributions

Questions, issues, and contributions are welcome. Please open an issue or pull request for improvements.
