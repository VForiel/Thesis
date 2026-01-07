# Streamlit Web Application Guidelines

## 📐 Simulation Page Architecture Pattern

### Design Philosophy

Each simulation page follows a **two-layer configuration pattern**:

1. **Base Context Layer** (dropdown, collapsed by default)
   - Provides full access to the underlying `Context` configuration
   - Uses the reusable `context_widget()` from `web/utils/context_widget.py`
   - Typically starts with a preset (VLTI, LIFE, etc.)
   - Allows advanced users to customize all parameters

2. **Simulation-Specific Parameters Layer** (visible by default)
   - Shows only the parameters most relevant to the specific simulation
   - These parameters **override** corresponding values from the base context
   - Provides a simplified, focused interface for quick experimentation

### Implementation Pattern

```python
import streamlit as st
from phise import Context
from web.utils.context_widget import context_widget

st.title("🔭 My Simulation")

# Layer 1: Base context configuration (collapsed dropdown)
base_ctx = context_widget(
    key_prefix="mysim",
    presets={"VLTI": Context.get_VLTI(), "LIFE": Context.get_LIFE()},
    default_preset="VLTI",
    expanded=False,  # COLLAPSED by default
    show_advanced=True
)

# Layer 2: Simulation-specific parameters (visible controls)
st.subheader("Simulation Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    h_override = st.slider("Hour angle h (hours)", -12.0, 12.0, 0.0)
with col2:
    lambda_override = st.number_input("Wavelength λ (µm)", 0.5, 20.0, 3.8)
with col3:
    fov_override = st.number_input("FOV (mas)", 1.0, 1000.0, 100.0)

# Apply overrides to create working context
working_ctx = copy(base_ctx)
working_ctx.h = h_override * u.hourangle
working_ctx.interferometer.λ = lambda_override * u.um
working_ctx.interferometer.fov = fov_override * u.mas

# Run simulation with working_ctx
# ...
```

### Key Principles

✅ **DO**:
- Start with a preset base context (VLTI/LIFE)
- Keep the dropdown **collapsed by default** (`expanded=False`)
- Place simulation-specific parameters **prominently visible**
- Accept parameter duplication between layers (intentional design)
- Document which base context parameters are overridden

❌ **DON'T**:
- Force users to expand the dropdown to use the simulation
- Hide critical simulation parameters inside the dropdown
- Try to "deduplicate" parameters across layers

### Example Pages

- **Projected Telescope Positions**: Overrides h, Γ, λ, fov
- **Transmission Maps**: Overrides λ, fov, companion parameters
- **Temporal Response**: Overrides h, Δh, integration time

### Rationale

This two-layer pattern provides:
- **Simplicity**: Most users only interact with visible simulation controls
- **Flexibility**: Advanced users can reconfigure the entire base context
- **Clarity**: Clear separation between "what context am I starting from?" vs "what am I varying in this simulation?"
- **Discoverability**: Users can explore the full context configuration without it cluttering the main interface

---

## 🎨 UI/UX Standards

### Language
- All code, documentation, comments, and UI text in **English**

### Structure
```
web/
├── main.py              # Home page
├── pages/              # Simulation pages
│   ├── Projected_Telescope_Positions.py
│   └── Context_Configurator.py
└── utils/
    └── context_widget.py  # Reusable context configuration widget
```

### Naming Conventions
- Page files: `Descriptive_Name.py` (no numeric prefixes)
- Widget keys: Use `key_prefix` parameter to avoid collisions
- Session state: `f"{key_prefix}_context"` for context storage

---

*Last updated: 2025-01-07*
