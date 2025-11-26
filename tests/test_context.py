"""
Unit tests for the Context class.

Each test is commented to explain the goal. When an exact numeric
expectation is not obvious (depends on internal constants or random
generation), a #TODO is left so the maintainer can fill the expected value
later.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u

import pytest

from phise.classes.context import Context
from phise import context


def test_factory_creates_context():
    """Ensure the `get_VLTI` factory returns a Context instance.

    Goal: make sure a full construction (with Interferometer, Camera,
    Chip...) does not raise and that basic attributes are initialized.
    """
    ctx = Context.get_VLTI()
    assert isinstance(ctx, Context)
    assert isinstance(ctx.name, str)
    
    ctx = Context.get_LIFE()
    assert isinstance(ctx, Context)
    assert isinstance(ctx.name, str)

def test_str_and_repr_contains_name():
    """Ensure __str__ and __repr__ include the context name.

    This helps with debugging and readability.
    """
    ctx = Context.get_VLTI()
    s = str(ctx)
    r = repr(ctx)
    assert ctx.name in s
    assert ctx.name in r

def test_property_types_and_units():
    """Check types/units of main properties (h, Δh, Γ, pf, p).

    We assert these properties return Quantities (when appropriate) and
    have consistent units.
    """
    ctx = Context.get_VLTI()

    assert hasattr(ctx, 'h') and hasattr(ctx, 'Δh') and hasattr(ctx, 'Γ')
    assert isinstance(ctx.h, u.Quantity)
    assert isinstance(ctx.Δh, u.Quantity)
    assert isinstance(ctx.Γ, u.Quantity)

    # p and pf should be Quantities (p in meters, pf in photons/s)
    assert isinstance(ctx.p, u.Quantity)
    assert isinstance(ctx.pf, u.Quantity)


def test_setters_type_validation_and_errors():
    """Verify that setters validate types and raise errors.

    - interferometer and target expect specific objects -> TypeError
    - h and Δh must be Quantities with compatible units
    - Γ must be a length Quantity
    - p is read-only and assigning to it should raise ValueError
    """
    ctx = Context.get_VLTI()

    # wrong type for interferometer/target
    with pytest.raises(TypeError):
        ctx.interferometer = object()

    with pytest.raises(TypeError):
        ctx.target = object()

    # wrong type for h
    with pytest.raises(TypeError):
        ctx.h = 1  # not a Quantity

    # Quantity but wrong unit for h
    with pytest.raises(ValueError):
        ctx.h = (1 * u.m)  # should be an hour-angle

    # Δh must be a Quantity
    with pytest.raises(TypeError):
        ctx.Δh = 1

    # Δh too small (smaller than camera.e exposure time) -> ValueError
    with pytest.raises(ValueError):
        ctx.Δh = (1 * u.s)

    # Γ must be a Quantity
    with pytest.raises(TypeError):
        ctx.Γ = 1

    # Γ wrong unit (e.g. second) -> ValueError
    with pytest.raises(ValueError):
        ctx.Γ = (1 * u.s)

    # p is read-only
    with pytest.raises(ValueError):
        ctx.p = 0

    with pytest.raises(TypeError):
        ctx.monochromatic = 'yes'

    with pytest.raises(TypeError):
        ctx.name = 123

def test_get_input_fields_shape_and_dtype():
    """Check that `get_input_fields` returns a complex array shaped
    (n_objects, n_telescopes).

    We verify shape and type; numeric values depend on random draws (noise),
    so we don't assert exact values here.
    """
    ctx = Context.get_VLTI()
    fields = ctx.get_input_fields()

    nb_objects = 1 + len(ctx.target.companions)
    nb_tel = len(ctx.interferometer.telescopes)

    assert isinstance(fields, np.ndarray)
    assert fields.dtype == np.complex128
    assert fields.shape == (nb_objects, nb_tel)

    fields = context.get_unique_source_input_fields_jit(
        a=(np.array([1,2,3,4])/u.s).value, #photons/s
        ρ=(2*u.mas).to(u.rad).value,
        θ=(45*u.deg).to(u.rad).value,
        λ=(1.55*u.um).to(u.m).value,
        p=(np.array([[0,0],[10,0],[0,10],[10,10]])*u.m).value
    )

    print(fields)


def test_get_h_range_properties():
    """Ensure `get_h_range` returns a numpy array of values in radians and
    length >= 1.
    """
    ctx = Context.get_VLTI()
    h_range = ctx.get_h_range()
    assert isinstance(h_range, np.ndarray)
    assert h_range.ndim == 1
    assert h_range.size >= 1


def test_plot_projected_positions_returns_image_bytes():
    """Light test for `plot_projected_positions` requesting the buffer.

    Goal: ensure that calling with `return_image=True` returns a bytes buffer
    (PNG image). We don't inspect binary content.
    """
    ctx = Context.get_VLTI()
    img = ctx.plot_projected_positions(N=5, return_image=True)
    assert isinstance(img, (bytes, bytearray))


def test_pf_is_quantity_and_length_matches_telescopes():
    """Ensure `pf` is a Quantity and its length matches the number of
    telescopes.
    """
    ctx = Context.get_VLTI()
    pf = ctx.pf
    assert isinstance(pf, u.Quantity)
    assert pf.shape[0] == len(ctx.interferometer.telescopes)
    # Exact numeric value depends on instrumental parameters -> #TODO


def test_get_analytical_transmission_maps_shapes():
    """Test that `get_analytical_transmission_maps` returns arrays with correct shapes.
    
    The method should return:
    - bright_map: (N, N) array for the bright output
    - kernel_maps: (3, N, N) array for the 3 kernel outputs
    """
    ctx = Context.get_VLTI()
    N = 20
    bright_map, kernel_maps = ctx.get_analytical_transmission_maps(N=N)
    
    assert bright_map.shape == (N, N), f"Expected bright_map shape ({N}, {N}), got {bright_map.shape}"
    assert kernel_maps.shape == (3, N, N), f"Expected kernel_maps shape (3, {N}, {N}), got {kernel_maps.shape}"


def test_analytical_transmission_maps_physical_coherence():
    """Test that analytical transmission maps produce physically coherent results.
    
    Physical expectations:
    1. Bright output should be non-negative (intensity is always positive)
    2. Kernel outputs should be antisymmetric (roughly zero mean since they are
       differences of dark outputs)
    3. No NaN or infinite values
    """
    ctx = Context.get_VLTI()
    bright_map, kernel_maps = ctx.get_analytical_transmission_maps(N=30)
    
    # Check no NaN or inf
    assert np.all(np.isfinite(bright_map)), "Bright map contains NaN or inf"
    assert np.all(np.isfinite(kernel_maps)), "Kernel maps contain NaN or inf"
    
    # Bright output should be non-negative
    assert np.all(bright_map >= 0), "Bright output should be non-negative"
    
    # Kernel outputs should have approximately zero mean (antisymmetric)
    for i in range(3):
        mean_abs = np.abs(kernel_maps[i].mean())
        max_abs = np.abs(kernel_maps[i]).max()
        # Mean should be much smaller than max (at least 1000x smaller)
        assert mean_abs < max_abs / 1000, f"Kernel {i+1} is not antisymmetric: mean={mean_abs}, max={max_abs}"


def test_analytical_vs_numerical_correlation():
    """Test that analytical and numerical transmission maps are highly correlated.
    
    The analytical model is a simplified version of the numerical model. When
    the numerical model has no manufacturing errors (σ=0) and no injected
    phase shifts (φ=0), both models should produce the same transmission patterns
    (differing only by a constant scaling factor).
    """
    ctx = Context.get_VLTI()
    ctx.interferometer.chip.σ = np.zeros(14) * u.nm  # No manufacturing errors
    ctx.interferometer.chip.φ = np.zeros(14) * u.nm  # No injected phase shifts
    
    N = 30
    
    # Get numerical transmission maps
    raw_num, proc_num = ctx.get_transmission_maps(N=N)
    
    # Get analytical transmission maps
    bright_ana, kernel_ana = ctx.get_analytical_transmission_maps(N=N)
    
    # Check correlation for bright output
    corr_bright = np.corrcoef(raw_num[0].flatten(), bright_ana.flatten())[0, 1]
    assert corr_bright > 0.99, f"Bright correlation too low: {corr_bright}"
    
    # Check correlation for kernel outputs
    for i in range(3):
        corr_kernel = np.corrcoef(proc_num[i].flatten(), kernel_ana[i].flatten())[0, 1]
        assert corr_kernel > 0.99, f"Kernel {i+1} correlation too low: {corr_kernel}"


def test_plot_analytical_transmission_maps_returns_bytes():
    """Test that `plot_analytical_transmission_maps` with return_plot=True
    returns a tuple of (PNG bytes, HTML string).
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    ctx = Context.get_VLTI()
    result = ctx.plot_analytical_transmission_maps(N=20, return_plot=True)
    
    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Should return (bytes, str)"
    assert isinstance(result[0], bytes), "First element should be PNG bytes"
    assert isinstance(result[1], str), "Second element should be HTML string"
    assert len(result[0]) > 0, "PNG bytes should not be empty"

