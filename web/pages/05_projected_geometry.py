import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.context_panel import context_panel

st.set_page_config(page_title="Géométrie projetée", page_icon="🗺️", layout="wide")

st.title("🗺️ Géométrie projetée (u,v)")
main, ctx = st.columns([4, 1])
ctx_state = context_panel()

with main:
    st.subheader("Formule de projection")
    st.latex(
        "\\begin{pmatrix}u \\ v\\end{pmatrix} ="
        "\\begin{pmatrix}-\\sin l \\sin h & \\cos h\\\\"
        "\\sin l \\cos h \\sin \\delta + \\cos l \\cos \\delta & \\sin h \\sin \\delta\\end{pmatrix}"
        "\\begin{pmatrix}B_{N} \\ B_{E}\\end{pmatrix}"
    )
    st.write("Dépend de la latitude du site, de l'angle horaire h et de la déclinaison δ.")
    col_a, col_b = st.columns(2)
    with col_a:
        ha = st.slider("Angle horaire (h)", -6.0, 6.0, 0.0, 0.25)
    with col_b:
        dec = st.slider("Déclinaison (deg)", -80.0, 80.0, 0.0, 1.0)
    st.caption("Utilisez le panneau de droite pour VLTI/LIFE.")

if ctx_state:
    st.caption(
        f"Preset={ctx_state['preset']} | λ={ctx_state['wavelength']} µm | RA={ctx_state['ra']}° | "
        f"Dec={ctx_state['dec']}° | HA={ctx_state['hour_angle']} h"
    )
