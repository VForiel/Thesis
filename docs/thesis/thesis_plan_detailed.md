# Thesis Plan: Tunable Kernel Nulling for High-Contrast Exoplanet Imaging

**Author:** Vincent Foriel
**Working Title:** Tunable Kernel Nulling for High-Contrast Exoplanet Imaging: From Theory to Experimental Validation

## Abstract

This thesis explores the theoretical and experimental development of **Tunable Kernel Nulling**, a robust interferometric technique for the direct detection of exoplanets. It details the design, simulation, and validation of novel photonic beam combiners based on Multi-Mode Interferometers (MMI) within the **PHISE** simulation framework and their experimental implementation on the **PHOBos** test bench.

---

## 1. Introduction

*   **Context**: The quest for other worlds and the challenge of high-contrast imaging.
*   **Problem**: Detecting faint planets next to bright stars (contrast ratios $10^{-6}$ to $10^{-10}$).
*   **Limitations**:
    *   Coronagraphy (requires single aperture, sensitive to wavefront errors).
    *   Classical Interferometry (requires precise path length control).
    *   Nulling Interferometry limitations (instability).
*   **Solution**: **Kernel Nulling** as a robust alternative using phase closure properties.

## 2. Theoretical Foundations

### 2.1 Optical Interferometry & Nulling
*   Coherence, Visibility, Phase.
*   The Bracewell Nuller concept.
*   Effect of phase perturbations on null depth.

### 2.2 Kernel Nulling (KN)
*   **Linear algebraic formalism**: Mapping inputs to outputs via a transfer matrix $\mathbf{M}$.
*   **Kernel Nulls**: Observables lying in the left null space of the perturbations matrix.
*   Robustness to phase noise (2nd order vs 1st order for raw nulls).
*   *Reference: Martinache et al.*

### 2.3 Integrated Photonics for Interferometry
*   Advantages: Stability, compactness, filtering.
*   **Multi-Mode Interferometers (MMI)**: Physics of self-imaging.
*   Transfer matrix of an MMI coupler ($N \times M$).
*   *Reference: Soldano & Pennings.*

### 2.4 Tunable Kernel Nulling
*   The concept of reconfigurable photonic chips.
*   Using thermal phase shifters to tune the combiner matrix.
*   Generalizing KN to arbitrary architectures.

---

## 3. Numerical Simulation: The PHISE Framework

### 3.1 Software Architecture
*   Object-Oriented Design in Python.
*   **Core Classes**: `Context`, `Telescope`, `Interferometer`, `Chip`, `Camera`.
*   Handling physical units with `astropy.units`.

### 3.2 Signal Propagation Model
*   Monochromatic vs Polychromatic light.
*   Matrix propagation through the optical chain.
*   Shot noise and detector noise simulation.

### 3.3 Optimization
*   Use of `numba` for JIT compilation of critical loops.
*   Benchmarking results.

### 3.4 Validating the Physics
*   Comparison with analytical models.
*   Flux conservation checks.

---

## 4. Experimental Implementation: The PHOBos Bench

### 4.1 Bench Overview
*   Optical layout: Source -> Turbulence Simulator -> Telescope Array -> Delay Lines -> Photonic Chip -> Camera.
*   Key components:
    *   **Deformable Mirror (DM)** for wavefront control.
    *   **C-RED3 Camera** for high-speed detection.

### 4.2 PHOBos Control Software
*   Why a custom OS? (Hardware abstraction, modularity).
*   **Architecture**: Device drivers, server-client model.
*   **Sandbox Mode**: Developing without hardware.

### 4.3 Calibration Strategies
*   **Phase Shifters calibration**: Determining the V-pi curve.
*   **Flux balancing**: Equalizing injection into the chip.

---

## 5. Characterization & First Results

### 5.1 Photonic Chip Characterization
*   Measuring the Transfer Matrix $\mathbf{V}$.
*   Spectral response characterization.
*   Inter-channel cross-talk analysis.

### 5.2 Experimental Kernel Nulling
*   Constructing kernels from experimental data.
*   Null depth histograms: Comparison with theory.
*   Stability measurements over time (Allan variance).

### 5.3 On-Sky Performance estimation
*   Extrapolating bench results to a full instrument (e.g., GLINT at Subaru).
*   Predicted sensitivity limits for varying stellar magnitudes.

---

## 6. Conclusion & Perspectives

*   Summary of achievements: A verified simulation tool + A working prototype bench.
*   Outlook: Moving to on-sky commensal observations.
*   Future work: 3D photonic chips, polychromatic kernel nulling.

---

## 7. Appendices

*   A. PHISE API Documentation excerpts.
*   B. PHOBos Hardware protocols.
*   C. Derivation of the 4x4 MMI Matrix.
