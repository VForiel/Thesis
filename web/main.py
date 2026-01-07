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
    page_title="🏠 Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 Home")

st.markdown(
    "Welcome to the PHISE thesis interactive app."
)

st.info(
    "Use the sidebar to navigate between pages."
)
