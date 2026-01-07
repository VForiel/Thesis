"""
Shared utilities for PHISE Streamlit applications.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from phise import Context
from phise.classes.companion import Companion
import astropy.units as u


def setup_path():
    """Add src directory to Python path."""
    ROOT = Path(__file__).parent.parent.parent
    if (ROOT / 'src').exists() and str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))


def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        'ctx': None,
        'cached_contexts': {},
        'last_params': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_vlti_context(
    companion_sep_alpha: float = 150.0,
    companion_sep_delta: float = 100.0,
    companion_contrast: float = 0.01,
    piston_error: float = 5.0,
    spectral_width: float = 0.0,
    reset: bool = False,
) -> Context:
    """
    Get or create a VLTI context with caching.
    
    Parameters
    ----------
    companion_sep_alpha : float
        Companion separation in RA (mas)
    companion_sep_delta : float
        Companion separation in Dec (mas)
    companion_contrast : float
        Companion contrast (0-1)
    piston_error : float
        Piston error in nm
    spectral_width : float
        Spectral width in nm (0 = monochromatic)
    reset : bool
        If True, don't use cached context
    
    Returns
    -------
    Context
        PHISE Context object ready for analysis
    """
    # Create cache key
    cache_key = (companion_sep_alpha, companion_sep_delta, companion_contrast, 
                 piston_error, spectral_width)
    
    if not reset and cache_key in st.session_state.cached_contexts:
        return st.session_state.cached_contexts[cache_key]
    
    # Create new context
    ctx = Context.get_VLTI()
    
    # Configure companion
    if companion_contrast > 0:
        companion = Companion(
            Δα=companion_sep_alpha * u.mas,
            Δδ=companion_sep_delta * u.mas,
            c=companion_contrast
        )
        ctx.target.companions = [companion]
    
    # Set observation parameters
    ctx.Γ = piston_error * u.nm
    if spectral_width > 0:
        ctx.interferometer.Δλ = spectral_width * u.nm
    
    # Cache it
    st.session_state.cached_contexts[cache_key] = ctx
    
    return ctx


def plot_with_streamlit(fig=None, use_container_width=True, **kwargs):
    """
    Display matplotlib figure in Streamlit with proper sizing.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to display. If None, uses current figure.
    use_container_width : bool
        Use full container width
    **kwargs
        Additional arguments for st.pyplot()
    """
    st.pyplot(fig, use_container_width=use_container_width, **kwargs)


def create_param_slider(label: str, min_val: float, max_val: float, 
                       default: float, step: float = None, 
                       format_str: str = None, unit: str = "") -> float:
    """
    Create a styled parameter slider.
    
    Parameters
    ----------
    label : str
        Slider label
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    default : float
        Default value
    step : float, optional
        Step size
    format_str : str, optional
        Format string for display
    unit : str
        Unit string to display
    
    Returns
    -------
    float
        Selected value
    """
    if step is None:
        step = (max_val - min_val) / 100
    
    if format_str is None:
        format_str = "%.2f"
    
    value = st.slider(
        f"{label} ({unit})" if unit else label,
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=step,
        format=format_str
    )
    
    return value


def display_statistics(data: np.ndarray, name: str = "Data"):
    """
    Display basic statistics of data.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    name : str
        Data name for display
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{np.mean(data):.2e}")
    with col2:
        st.metric("Std Dev", f"{np.std(data):.2e}")
    with col3:
        st.metric("Min", f"{np.min(data):.2e}")
    with col4:
        st.metric("Max", f"{np.max(data):.2e}")


def add_header(title: str, subtitle: str = ""):
    """
    Add styled header to page.
    
    Parameters
    ----------
    title : str
        Main title
    subtitle : str, optional
        Subtitle
    """
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"_{subtitle}_")
    st.markdown("---")


def export_figure(fig, filename: str, formats: list = None):
    """
    Create download button for figure.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to export
    filename : str
        Base filename (without extension)
    formats : list, optional
        List of formats to offer (pdf, png, svg)
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    import io
    
    tabs = st.tabs(formats)
    for tab, fmt in zip(tabs, formats):
        with tab:
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label=f"Download as {fmt.upper()}",
                data=buf.getvalue(),
                file_name=f"{filename}.{fmt}",
                mime=f"image/{fmt}"
            )


def reset_context():
    """Reset cached contexts."""
    st.session_state.cached_contexts = {}
    st.session_state.ctx = None
