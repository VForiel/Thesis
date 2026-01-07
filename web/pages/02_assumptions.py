import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.context_panel import context_panel

st.set_page_config(page_title="Hypothèses", page_icon="❗", layout="wide")

st.title("❗ Hypothèses")
main, ctx = st.columns([4, 1])
ctx_state = context_panel()

with main:
    st.subheader("Bande passante étroite")
    st.write("Δλ < 0.1 µm ⇒ flux et indice constants, phase shifters ≈ pistons (OPD).")
    st.subheader("Source ponctuelle")
    st.write("L'étoile est non résolue et modélisée comme point source.")

if ctx_state:
    st.caption(
        f"Contexte: {ctx_state['preset']} | λ={ctx_state['wavelength']} µm | "
        f"RA={ctx_state['ra']}°, Dec={ctx_state['dec']}°, HA={ctx_state['hour_angle']} h"
    )
