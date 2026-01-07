import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.context_panel import context_panel

st.set_page_config(page_title="Abstract", page_icon="📜", layout="wide")

st.title("📜 Abstract")
main, ctx = st.columns([4, 1])
ctx_state = context_panel()

with main:
    st.write(
        "Tunable kernel-nulling interferometry with a 4-telescope photonic chip and"
        " on-chip thermo-optic phase shifters enables spectral fine-tuning around"
        " the design wavelength. Two calibration algorithms retrieve near-ideal"
        " nulls and adjust for fabrication and piston errors, targeting biosignature"
        " scans (VLTI, LIFE)."
    )
    st.write(
        "Validations combine simulations and lab testbed data to assess detection"
        " sensitivity across configurations."
    )

if ctx_state:
    st.info(
        f"Contexte: preset={ctx_state['preset']}, λ={ctx_state['wavelength']} µm, "
        f"RA={ctx_state['ra']}°, Dec={ctx_state['dec']}°, HA={ctx_state['hour_angle']} h"
    )
