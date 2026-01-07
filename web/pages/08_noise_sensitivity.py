import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.context_panel import context_panel

st.set_page_config(page_title="Sensibilité au bruit", page_icon="🔊", layout="wide")

st.title("🔊 Sensibilité au bruit")
main, ctx = st.columns([4, 1])
ctx_state = context_panel()

with main:
    st.subheader("Budget de bruit")
    read = st.slider("Bruit de lecture (e-)", 0.0, 30.0, 5.0, 0.5)
    dark = st.slider("Courant d'obscurité (e-/s)", 0.0, 1.0, 0.05, 0.01)
    jitter = st.slider("Jitter OPD (nm RMS)", 0.0, 200.0, 50.0, 5.0)
    st.write("Évalue l'effet sur le contraste de null et la détection.")
    st.latex(r"\text{SNR} = \frac{S}{\sqrt{S + N_{sky} + N_{dark} + N_{read}^2}}")
    st.caption("Ajustez λ, RA, Dec, HA via panneau de contexte.")

if ctx_state:
    st.caption(
        f"Preset={ctx_state['preset']} | λ={ctx_state['wavelength']} µm | RA={ctx_state['ra']}° | "
        f"Dec={ctx_state['dec']}° | HA={ctx_state['hour_angle']} h"
    )
