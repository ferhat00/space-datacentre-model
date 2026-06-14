"""Tab 3 — results: KPIs, LCOC, capex/mass anatomy, launch crossing curve."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import state, compute, charts


def render() -> None:
    st.subheader("Results")
    pk = state.params_key()
    mw = state.size_mw()
    r = compute.evaluate(pk, mw, include_delay=st.session_state.scarcity)

    ratio = r["ratio"]
    verdict = ("✅ space at/below parity" if ratio <= 1.0
               else "🟠 within ~2x of parity" if ratio <= 2.0 else "🔴 space far from parity")
    st.markdown(f"**Verdict:** {verdict} — LCOC ratio **{ratio:.2f}×** "
                f"(space/ground) at {mw:g} MW.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LCOC space", f"${r['lcoc_s']:.2f}/kWh")
    c2.metric("LCOC ground", f"${r['lcoc_g']:.2f}/kWh")
    c3.metric("Dry mass", f"{r['pt']['M_dry']/1e3:.1f} t/MW")
    be = r["breakeven_launch"]
    c4.metric("Breakeven launch", f"${be:,.0f}/kg" if be > 0 else "negative")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Capex space", f"${r['cap_s']['total']:.0f}M/MW")
    c6.metric("Capex ground", f"${r['cap_g']['total']:.0f}M/MW")
    c7.metric("NPV space", f"${r['npv_s']:.0f}M/MW")
    c8.metric("NPV ground", f"${r['npv_g']:.0f}M/MW")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.lcoc_bar(r), use_container_width=True, key="res_lcoc")
        st.plotly_chart(charts.mass_anatomy(r), use_container_width=True, key="res_mass")
    with right:
        curve = compute.launch_curve(pk, mw)
        st.plotly_chart(
            charts.launch_curve_fig(curve, st.session_state[state._wk("launch_kg")],
                                    r["lcoc_s"]), use_container_width=True, key="res_curve")
        st.plotly_chart(charts.capex_anatomy(r), use_container_width=True, key="res_capex")

    st.markdown("##### SemiAnalysis cross-check ($/GPU-hr etc. vs published 2026 anchors)")
    rows = compute.sa_anchor_comparison(r)
    df = pd.DataFrame(rows)
    df["model"] = df["model"].map(lambda v: f"{v:,.2f}")
    df["anchor"] = df["anchor"].map(lambda v: f"{v:,.2f}" if v is not None else "—")
    df["delta_pct"] = df["delta_pct"].map(lambda v: f"{v:+.1f}%" if v is not None else "—")
    st.dataframe(df.rename(columns={"metric": "Metric", "model": "This model",
                 "anchor": "SemiAnalysis", "delta_pct": "Δ"}),
                 hide_index=True, use_container_width=True)
    st.caption("SA26 preset reproduces the published anchors within ~2% (regression-guarded).")
