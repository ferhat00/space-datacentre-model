"""
streamlit_app.main — page shell: sidebar global controls + 7-tab dispatch.
Run via the repo-root entrypoint:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from . import state, compute
from .tabs import methods, builder, results, sensitivity, ladder, workloads, provenance


def _apply_preset_cb(name: str) -> None:
    state.apply_preset(name)
    if st.session_state.orbit_name != state.DEFAULT_ORBIT:
        state.apply_orbit(st.session_state.orbit_name)


def _on_orbit_change() -> None:
    state.apply_orbit(st.session_state.orbit_name)


def _sidebar() -> None:
    sb = st.sidebar
    sb.title("🛰️ ODC controls")

    sb.caption("Preset")
    names = list(state.PRESETS)
    cols = sb.columns(3)
    for i, name in enumerate(names):
        cols[i % 3].button(name, key=f"preset_{name}", use_container_width=True,
                           on_click=_apply_preset_cb, args=(name,))
    sb.markdown(f"**Active:** `{st.session_state.preset_name}`")

    sb.divider()
    sb.selectbox("Orbit", list(state.ORBITS), key="orbit_name", on_change=_on_orbit_change)
    sb.selectbox("Size class", list(state.SIZES), key="size_key",
                 format_func=lambda k: f"{state.SIZES[k].name} ({state.SIZES[k].it_mw:g} MW)")
    sb.selectbox("Headline workload", list(state.WORKLOADS), key="workload_name")
    sb.toggle("Scarcity mode (queue delays in NPV)", key="scarcity")

    sb.divider()
    r = compute.evaluate(state.params_key(), state.size_mw(), st.session_state.scarcity)
    ratio = r["ratio"]
    sb.metric("LCOC ratio (space/ground)", f"{ratio:.2f}×")
    sb.metric("LCOC space", f"${r['lcoc_s']:.2f}/kWh")
    sb.metric("Dry mass", f"{r['pt']['M_dry']/1e3:.1f} t/MW")
    sb.caption("Built on the calibrated `odc` package (OO kernel). SA26 reproduces the "
               "SemiAnalysis 2026 anchors within ~2%.")


def run() -> None:
    st.set_page_config(page_title="Orbital Data Centre Viability", page_icon="🛰️",
                       layout="wide", initial_sidebar_state="expanded")
    state.init_state()

    st.title("Orbital Data Centre Viability Model")
    st.caption("A power → thermal → mass → launch → capex → LCOC/NPV system model of an "
               "orbital AI data centre vs a terrestrial one. Ambitious where the physics and "
               "economics allow; skeptical where they don't. Tune everything in the sidebar and "
               "the **Parameters** tab.")

    _sidebar()

    tab_objs = st.tabs(["🧪 Methods", "🎛️ Parameters", "📊 Results", "🌪️ Sensitivity",
                        "🪜 Size & orbits", "🔌 Workloads & energy", "📚 Provenance"])
    renderers = [methods.render, builder.render, results.render, sensitivity.render,
                 ladder.render, workloads.render, provenance.render]
    for tab, render in zip(tab_objs, renderers):
        with tab:
            render()
