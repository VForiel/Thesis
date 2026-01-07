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
    This web application makes my thesis **interactive**. It hosts the analyses,
    visualizations, and exploratory tools built for the research work. The
    `PHISE` simulation library is the computational engine powering the pages,
    but the thesis itself is about the **analyses and results** that leverage
    this engine.
    """
)

st.info(
    "Use the sidebar to open each analysis or visualization page."
)

st.markdown(
    """
    ### What you can do here
    - Configure observation contexts and interferometers
    - Explore projected telescope positions and UV coverage
    - Inspect transmission maps, temporal responses, and sensitivity analyses
    - Run scenario-specific simulations tied to the thesis chapters

    ### About the engine
    The pages call into the `PHISE` library for interferometric simulations, but the
    focus of this app is the **thesis workflows** built on top of it: comparing
    architectures, testing observing strategies, and validating analysis methods.
    """
)
