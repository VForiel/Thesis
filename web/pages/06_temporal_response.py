import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.context_panel import context_panel

st.set_page_config(page_title="Réponse temporelle", page_icon="⏱️", layout="wide")

st.title("⏱️ Réponse temporelle")
main, ctx = st.columns([4, 1])
ctx_state = context_panel()

with main:
    st.subheader("Temps cohérence et bande passante")
    st.latex(r"\\tau_c = \\frac{\\lambda^2}{c \\Delta \\lambda}")
    bw = st.slider("Bande passante Δλ (nm)", 10.0, 500.0, 50.0, 5.0)
    st.write("Temps de cohérence plus court quand la bande passante augmente.")
    st.latex(r"f_{3dB} \approx \frac{1}{2\\pi\\tau_c}")
    st.caption("Paramètres instrument via panneau de contexte.")

if ctx_state:
    st.caption(
        f"Preset={ctx_state['preset']} | λ={ctx_state['wavelength']} µm | RA={ctx_state['ra']}° | "
        f"Dec={ctx_state['dec']}° | HA={ctx_state['hour_angle']} h"
    )
