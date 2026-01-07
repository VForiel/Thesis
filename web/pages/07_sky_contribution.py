import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.context_panel import context_panel

st.set_page_config(page_title="Contribution du ciel", page_icon="🌌", layout="wide")

st.title("🌌 Contribution du ciel")
main, ctx = st.columns([4, 1])
ctx_state = context_panel()

with main:
    st.subheader("Background thermique et zodiaque")
    sky_temp = st.slider("Température ciel (K)", 150.0, 300.0, 220.0, 5.0)
    zodi = st.slider("Zodiaque (MJy/sr)", 0.1, 10.0, 1.0, 0.1)
    st.write("Impact sur le rapport signal/bruit et la profondeur de null.")
    st.latex(r"N_{sky} \propto B_{\lambda}(T_{sky}) \\cdot \Omega_{\text{instr}}")
    st.caption("Renseignez RA/Dec/λ dans le panneau de droite pour presets VLTI/LIFE.")

if ctx_state:
    st.caption(
        f"Preset={ctx_state['preset']} | λ={ctx_state['wavelength']} µm | RA={ctx_state['ra']}° | "
        f"Dec={ctx_state['dec']}° | HA={ctx_state['hour_angle']} h"
    )
