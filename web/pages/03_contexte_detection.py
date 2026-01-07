import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.context_panel import context_panel

st.set_page_config(page_title="Contexte & Détection", page_icon="🛰️", layout="wide")

st.title("🤔 Contexte & Détection")
main, ctx = st.columns([4, 1])
ctx_state = context_panel()

with main:
    st.subheader("Objectif")
    st.write("Détection directe d'exoplanètes avec contraste 10⁻⁶–10⁻¹⁰ et faible séparation angulaire.")
    st.subheader("Méthodes de détection")
    st.write("Radiale, transit, microlentille, astrométrie, coronographie.")
    st.subheader("Apport du kernel-nulling")
    st.write("Suppression stellaire par interférences destructives, sensible à la phase, robuste via kernels.")

if ctx_state:
    st.caption(
        f"Preset={ctx_state['preset']} | λ={ctx_state['wavelength']} µm | "
        f"RA={ctx_state['ra']}°, Dec={ctx_state['dec']}°, HA={ctx_state['hour_angle']} h"
    )
