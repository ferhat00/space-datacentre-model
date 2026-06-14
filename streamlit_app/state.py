"""
streamlit_app.state — session state and model wiring
====================================================
The 37 slider fields are stored directly in widget-keyed session state (so preset/orbit
buttons can write them before the widgets render); the remaining fixed P fields (name,
eclipse_frac_daily, packing, pointing, ...) live in ``st.session_state.fixed``. The
current ``ODCModel`` is rebuilt from that state each rerun.
"""
from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from odc.core import P, ODCModel
from odc.schema import slider_fields
from odc.scenarios import TODAY, EARLY, MATURE, SA26, OPTIMIST, SKEPTIC
from odc import orbits, sizes, workloads

PRESETS = {"Today": TODAY, "Early": EARLY, "Mature": MATURE,
           "SA26": SA26, "Optimist": OPTIMIST, "Skeptic": SKEPTIC}
ORBITS = orbits.CATALOG
SIZES = sizes.CATALOG                       # key -> SizeClass
WORKLOADS = workloads.CATALOG

SLIDER_FIELDS = slider_fields()
DEFAULT_ORBIT = orbits.DDSS.name
DEFAULT_SIZE = "d"                          # 1 MW node (the v2 basis)
DEFAULT_WORKLOAD = workloads.FRONTIER_TRAINING.name


def _wk(field: str) -> str:
    """Widget/session key for a slider field."""
    return f"p_{field}"


def _load_preset_values(preset: P) -> None:
    d = asdict(preset)
    for f in SLIDER_FIELDS:
        st.session_state[_wk(f)] = float(d[f])     # float so st.slider types are consistent
    st.session_state.fixed = {k: v for k, v in d.items() if k not in SLIDER_FIELDS}


def init_state() -> None:
    if st.session_state.get("_initialized"):
        return
    _load_preset_values(TODAY)
    st.session_state.preset_name = "Today"
    st.session_state.orbit_name = DEFAULT_ORBIT
    st.session_state.size_key = DEFAULT_SIZE
    st.session_state.workload_name = DEFAULT_WORKLOAD
    st.session_state.scarcity = False
    st.session_state._initialized = True


def mark_custom() -> None:
    """Slider on_change callback: any manual edit drops the preset label to 'Custom'."""
    st.session_state.preset_name = "Custom"


def apply_preset(name: str) -> None:
    """Load a preset's full parameter vector. Leaves the orbit selector independent; the
    caller re-stamps a non-default orbit so a DDSS preset stays numerically exact
    (eclipse_frac_daily=None)."""
    _load_preset_values(PRESETS[name])
    st.session_state.preset_name = name


def apply_orbit(name: str) -> None:
    """Stamp an orbit's eclipse profile onto the current parameters (independent of preset)."""
    o = ORBITS[name]
    st.session_state[_wk("eclipse_min_day")] = float(o.eclipse_min_orbit)
    st.session_state.fixed["eclipse_frac_daily"] = o.eclipse_frac_daily
    st.session_state.orbit_name = name


def current_params() -> dict:
    """Rebuild the full 49-field P-parameter dict from session state."""
    d = dict(st.session_state.fixed)
    for f in SLIDER_FIELDS:
        d[f] = st.session_state[_wk(f)]
    return d


def size_mw() -> float:
    return SIZES[st.session_state.size_key].it_mw


def current_model() -> ODCModel:
    return ODCModel(P(**current_params()), size_mw())


def params_key(params: dict | None = None) -> tuple:
    """A hashable key for st.cache_data (dicts with None/str/float values)."""
    p = params if params is not None else current_params()
    return tuple(sorted(p.items()))
