"""Tab 2 — parameter & scenario builder (the ~37 tunable sliders)."""
from __future__ import annotations

import streamlit as st

from odc.schema import GROUPS
from .. import state


def render() -> None:
    st.subheader("Parameter & scenario builder")
    st.caption(
        f"Active preset: **{st.session_state.preset_name}** · orbit: "
        f"**{st.session_state.orbit_name}** · size: "
        f"**{state.SIZES[st.session_state.size_key].name} ({state.size_mw():g} MW)**. "
        "Preset / orbit / size / workload selectors are in the sidebar. Dragging any "
        "slider switches to *Custom*. The remaining P fields (geometry & efficiency "
        "constants) stay at the preset value.")

    cols = st.columns(2)
    for i, (group_label, fields) in enumerate(GROUPS):
        with cols[i % 2]:
            with st.expander(group_label, expanded=(i < 3)):
                for field, label, unit, lo, hi, step, kind, note in fields:
                    _slider(field, label, unit, lo, hi, step, note)


def _slider(field, label, unit, lo, hi, step, note) -> None:
    key = state._wk(field)
    val = st.session_state[key]
    lo2, hi2 = min(float(lo), val), max(float(hi), val)   # never clamp a preset value
    disp = label + (f"  [{unit}]" if unit else "")
    st.slider(disp, lo2, hi2, step=float(step), key=key,
              on_change=state.mark_custom, help=note or None)
