"""Shared context panel with quick VLTI/LIFE presets."""
import streamlit as st

PRESETS = {
    "VLTI": {"ra": 0.0, "dec": -30.0, "wavelength": 3.8, "hour_angle": 0.0},
    "LIFE": {"ra": 0.0, "dec": 30.0, "wavelength": 10.0, "hour_angle": 0.0},
}


def context_panel():
    """Render a collapsible context configuration panel on the right.

    Returns a dict with current values or None if panel hidden.
    """
    show = st.toggle("Afficher la config contexte", value=False, key="ctx_toggle")
    if not show:
        return None

    if "ctx_state" not in st.session_state:
        st.session_state.ctx_state = PRESETS["VLTI"].copy()

    st.markdown("**Configuration rapide**")
    preset = st.radio("Préréglage", ["Custom", "VLTI", "LIFE"], index=1, key="ctx_preset")
    if preset in PRESETS:
        st.session_state.ctx_state = PRESETS[preset].copy()

    st.markdown("**Paramètres de base**")
    col_a, col_b = st.columns(2)
    with col_a:
        ra = st.number_input(
            "Ascension droite (deg)",
            value=st.session_state.ctx_state.get("ra", 0.0),
            key="ctx_ra",
        )
        dec = st.number_input(
            "Déclinaison (deg)",
            value=st.session_state.ctx_state.get("dec", 0.0),
            key="ctx_dec",
        )
    with col_b:
        wl = st.number_input(
            "Longueur d'onde (µm)",
            value=st.session_state.ctx_state.get("wavelength", 1.55),
            min_value=0.4,
            max_value=20.0,
            key="ctx_wl",
        )
        ha = st.number_input(
            "Angle horaire (h)",
            value=st.session_state.ctx_state.get("hour_angle", 0.0),
            min_value=-12.0,
            max_value=12.0,
            key="ctx_ha",
        )

    st.caption("Préréglages VLTI ou LIFE appliquent des valeurs types.")
    return {
        "preset": preset,
        "ra": ra,
        "dec": dec,
        "wavelength": wl,
        "hour_angle": ha,
    }
