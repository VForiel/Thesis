"""Outils de visualisation pour positions projetées des télescopes.

Contient une interface graphique minimale pour afficher la géométrie
du réseau de télescopes projetée sur le plan du ciel.
"""
import numpy as np
import astropy.units as u
from ipywidgets import widgets
from IPython.display import display
from io import BytesIO
from copy import deepcopy as copy
from phise import Telescope
from phise import Context

from phise.modules import utils

def gui(ctx: Context=None, n=10, save_as=None) -> None:
    """
    GUI to visualize the projected positions of the telescopes in the array.

    Parameters
    ----------
    ctx: Context
        Context object containing the interferometer and target information.
        If None, a default context is used.
    n: int
        Number of telescopes in the interferometer. Default is 10.
    save_as: str
        Path to save the initial plot.
    """
    if ctx is None:
        ref_ctx = Context.get_VLTI()
    else:
        ref_ctx = copy(ctx)


        
    l_slider = widgets.FloatSlider(value=ref_ctx.interferometer.l.to(u.deg).value, min=-90, max=90, step=0.01, description='Latitude (deg):')
    δ_slider = widgets.FloatSlider(value=ref_ctx.target.δ.to(u.deg).value, min=-90, max=90, step=0.01, description='Declination (deg):')
    reset = widgets.Button(description='Reset to default')
    export = widgets.Button(description='Export')
    plot = widgets.Image(width=500, height=500)

    def update_plot(*_):
        ctx = copy(ref_ctx)
        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg
        plot.value = ctx.plot_projected_positions(N=n, return_image=True)

    def reset_values(*_):
        l_slider.value = ref_ctx.interferometer.l.to(u.deg).value
        δ_slider.value = ref_ctx.target.δ.to(u.deg).value
    
    def export_plot(*_):
        ctx = copy(ref_ctx)
        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg
        ctx.plot_projected_positions(N=n, save_as=save_as)

    reset.on_click(reset_values)
    export.on_click(export_plot)
    l_slider.observe(update_plot, 'value')
    δ_slider.observe(update_plot, 'value')
    display(widgets.VBox([l_slider, δ_slider, widgets.HBox([reset, export]), plot]))
    update_plot()
    return