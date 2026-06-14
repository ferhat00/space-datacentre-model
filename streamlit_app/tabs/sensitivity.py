"""Tab 4 — sensitivity (tornado) and provenance-weighted Monte Carlo."""
from __future__ import annotations

import streamlit as st

from .. import state, compute, charts


def render() -> None:
    st.subheader("Sensitivity & Monte Carlo")
    pk, mw = state.params_key(), state.size_mw()

    st.markdown("##### One-at-a-time tornado")
    st.caption("Each contested parameter swung from its skeptic to its optimist end "
               "(odc.provenance ranges), holding the rest at the current scenario.")
    st.plotly_chart(charts.tornado_fig(compute.tornado(pk, mw)), use_container_width=True,
                    key="sens_tornado")

    st.markdown("##### Provenance-weighted Monte Carlo")
    c1, c2 = st.columns(2)
    n = c1.slider("draws", 500, 10000, 4000, 500, key="mc_n")
    seed = c2.number_input("seed", 0, 9999, 0, 1, key="mc_seed")
    mc = compute.monte_carlo(pk, mw, int(n), int(seed))
    st.plotly_chart(charts.mc_hist(mc), use_container_width=True, key="sens_mc")
    c3, c4, c5 = st.columns(3)
    c3.metric("P(space ≤ ground)", f"{mc['p_parity']:.0%}")
    c4.metric("Median ratio", f"{mc['median']:.2f}×")
    c5.metric("P10–P90", f"{mc['p10']:.2f}–{mc['p90']:.2f}×")
