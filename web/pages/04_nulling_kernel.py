import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.context_panel import context_panel

st.set_page_config(page_title="Nulling & Kernel", page_icon="🌀", layout="wide")

st.title("🌀 Nulling et Kernel")
main, ctx = st.columns([4, 1])
ctx_state = context_panel()

with main:
    st.subheader("Nuller 2 entrées")
    st.latex("\\begin{pmatrix}1 & 1 \\ 1 & -1\\end{pmatrix}")
    st.write("Une sortie constructive, une sortie nulle pour annuler l'étoile.")
    st.subheader("Kernel")
    st.write("Combinaison symétrique des recombineurs pour annuler les aberrations de phase d'ordre 1.")
    st.write("Réponse asymétrique utile pour contraindre la position de la planète.")

if ctx_state:
    st.caption(
        f"Preset={ctx_state['preset']} | λ={ctx_state['wavelength']} µm | RA={ctx_state['ra']}° | "
        f"Dec={ctx_state['dec']}° | HA={ctx_state['hour_angle']} h"
    )
