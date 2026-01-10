"""
PHISE Web Application Hub
Landing page to route toward each Thesis analysis section.
"""

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Interactive Thesis 🏠")

st.markdown(
    """
    Welcome to the interactive companion of my PhD thesis on **kernel-nulling
    interferometry with photonic beam combiners**. This app surfaces the core
    analyses, figures, and exploratory tools used throughout the manuscript so
    you can reproduce, tweak, and extend the results.
    """
)

st.info("Use the sidebar to open each analysis or visualization page.")

st.markdown(
    """
    ## Abstract
    
    **Tunable Kernel-Nulling interferometry using active photonics for direct exoplanet detection**
    
    Detecting Earth-like exoplanets demands extreme contrast ratios, high angular resolution, and long-term stability.
    Nulling interferometry is a promising technique to meet these requirements but it relies on extreme phase control.
    With the PHOTONICS project, we present one of the first implementations of an adaptive photonic nulling interferometer
    with a 4-telescope beam-combiner architecture, featuring 14 real-time thermo-optic phase shifters integrated within
    a silicon nitride photonic chip.
    
    We demonstrate two calibration algorithms which retrieve a near-ideal Kernel-Nuller device, using the integrated
    phase shifters to compensate for fabrication imperfections and upstream systematic piston errors. Furthermore, we
    show that the same calibration strategy can be used to adjust on-chip phase settings for wavelengths slightly
    different (but close) to the design wavelength λ₀. This fine-tuning enables obtaining a deep null at a chosen λ,
    allowing spectral scanning to target specific features such as biosignatures in exoplanet atmospheres. Additionally,
    we present advanced data-treatment and statistical analysis approaches to enable robust exoplanet detection and
    characterization, including precise determination of companion position and contrast.
    
    We simulate the imaging capabilities of a Kernel-Nuller with VLTI and LIFE layouts. These findings are further
    supported by laboratory testbed experiments, providing validation of our detection sensitivities across diverse
    instrumental configurations.
    """
)

st.markdown(
    """
    ## Thesis focus (what this app is about)
    - **Science case**: High-contrast detection of exoplanets via nulling interferometry.
    - **Instrument concept**: Photonic kernel nullers (MMI-based SuperKN) fed by arrays like VLTI/LIFE.
    - **Key questions**: uv-coverage, null depth stability, throughput vs. bandwidth, sensitivity to phase errors.
    - **Methodology**: Start from observation contexts (telescope geometry, wavelength band, target), propagate through photonic chip, analyze null outputs and kernel combinations.

    ## How to navigate the pages
    - **Base context dropdown**: collapsed by default; defines the starting interferometer/target/camera.
    - **Visible page controls**: override only the parameters relevant to the current visualization or test.
    - This mirrors the pattern used in the thesis: a reference setup + controlled variations.

    ## About the simulation engine (PHISE)
    - `PHISE` is the **numerical backend** for interferometric propagation and photonic chip modeling.
    - The thesis pages focus on the **analyses built on top of PHISE**, not on the library itself.
    - All physical quantities honor `astropy.units`; chip models follow the SuperKN architecture used in the manuscript.
    """
)
