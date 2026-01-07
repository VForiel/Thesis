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
    page_title="PHISE Thesis Companion",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PHISE Thesis Companion")
st.write(
    "Use the left navigation to open each section of Thesis.ipynb as a dedicated page."
)
st.write(
    "Each page keeps parameters on the main view with the optional right-hand context panel for VLTI/LIFE presets."
)

st.markdown("---")
st.subheader("Sections available")
st.markdown(
    "- Abstract\n"
    "- Hypotheses\n"
    "- Contexte & Détection\n"
    "- Nulling & Kernel\n"
    "- Géométrie projetée\n"
    "- Réponse temporelle\n"
    "- Contribution du ciel\n"
    "- Sensibilité au bruit"
)

st.info("Switch pages via the sidebar; this home page is informational only.")
