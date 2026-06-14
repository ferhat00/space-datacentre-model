"""Tab 1 — modeling methods: analytical approaches AND physical sub-models, each live.

This is the dedicated "different modeling types" tab. It renders the literature review
(docs/MODELING_METHODS.md) and lets the user pick either an *analytical method* (how to
judge viability) or a *physical sub-model* (the engineering that feeds it) and run it
against the current scenario.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from odc import comms, workloads
from odc.scenarios import SA_ANCHORS
from .. import state, compute, charts

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIT_DOC = _REPO_ROOT / "docs" / "MODELING_METHODS.md"


def render() -> None:
    st.subheader("Modeling methods")
    st.caption("Two complementary families: *analytical methods* (ways to judge viability) "
               "and *physical sub-models* (the engineering that feeds them). Pick one and it "
               "runs live against the current scenario.")

    with st.expander("📚 Literature review of the modelling methods", expanded=False):
        if _LIT_DOC.exists():
            st.markdown(_LIT_DOC.read_text(encoding="utf-8"))
        else:
            st.info("`docs/MODELING_METHODS.md` not found yet — the web-backed literature "
                    "search produces it. The methods below still run live.")

    family = st.radio("Method family", ["Analytical methods", "Physical sub-models"],
                      horizontal=True, key="methods_family")
    pk, mw = state.params_key(), state.size_mw()
    r = compute.evaluate(pk, mw, include_delay=st.session_state.scarcity)

    if family == "Analytical methods":
        choice = st.selectbox("Analytical method", list(_ANALYTICAL), key="methods_analytical")
        _ANALYTICAL[choice](pk, mw, r)
    else:
        choice = st.selectbox("Physical sub-model", list(_SUBMODELS), key="methods_submodel")
        _SUBMODELS[choice](r)


# ----------------------------------------------------------------- analytical methods

def _m_lcoc(pk, mw, r):
    st.markdown("**Levelized cost of compute / TCO** (SemiAnalysis basis). Annualises capex "
                "with a capital-recovery factor under *mixed asset lives* — the whole station "
                "over its short (5–8 yr, FCC-capped) life vs ground IT (5 yr) split from the "
                "facility (15 yr) — then divides by annual sellable compute.")
    st.latex(r"\mathrm{LCOC}=\frac{\mathrm{CRF}(r,n)\cdot \mathrm{Capex}+\mathrm{Opex}}"
             r"{\text{annual sellable kWh}},\quad "
             r"\mathrm{CRF}=\frac{12m(1+m)^{12n}}{(1+m)^{12n}-1},\ m=\tfrac{r}{12}")
    st.plotly_chart(charts.lcoc_bar(r), use_container_width=True, key="m_lcoc")
    df = pd.DataFrame(compute.sa_anchor_comparison(r))
    st.dataframe(df, hide_index=True, use_container_width=True)


def _m_gamma(pk, mw, r):
    st.markdown("**Communication-intensity / Γ-ceiling** (Turyshev; Bhattacherjee et al.). A "
                "workload moves Γ gigabytes of data per kWh of compute; the downlink can only "
                "sustain a bounded rate, so above the ceiling (≈14.8/P_MW GB/kWh) the workload "
                "is *link-bound*, not compute-bound — regardless of cost economics.")
    st.latex(r"\Gamma_{\text{ceiling}}(P_{\mathrm{MW}})\approx \frac{14.8}{P_{\mathrm{MW}}}"
             r"\ \mathrm{GB/kWh};\quad \text{closes if } \Gamma_{\text{workload}}\le "
             r"\Gamma_{\text{ceiling}}")
    st.plotly_chart(charts.gamma_fig(mw), use_container_width=True, key="m_gamma")
    rows = [dict(workload=w.name, gamma=w.gamma_gb_per_kwh,
                 headroom=f"{workloads.gamma_headroom(w, mw):.2f}×",
                 closes="✅" if workloads.closes_in_orbit(w, mw) else "❌",
                 orbital_revenue="✅" if w.revenue_in_orbit else "—")
            for w in workloads.CATALOG.values()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _m_oec(pk, mw, r):
    st.markdown("**Orbital-edge-computing** (Denby & Lucia, ASPLOS 2020; Kodan, ASPLOS 2023). "
                "Bandwidth, not FLOPs, historically bounds orbital compute. Here: the max IT "
                "power each link can keep fed for each revenue workload.")
    rows = []
    for link in comms.CATALOG.values():
        for w in workloads.ORBITAL_REVENUE_WORKLOADS:
            rows.append(dict(link=link.name, workload=w.name,
                             net_availability=f"{link.network_availability:.0%}",
                             feasible_MW=f"{comms.downlink_feasible_mw(link, w):.1f}"))
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _m_montecarlo(pk, mw, r):
    st.markdown("**Monte-Carlo / probabilistic** uncertainty quantification. Each contested "
                "parameter is sampled triangular(min, central, max) using the provenance "
                "ranges (skeptic ↔ optimist ends, SA-central mode); the LCOC ratio distribution "
                "gives P(space ≤ ground).")
    c1, c2 = st.columns(2)
    n = c1.slider("draws", 500, 10000, 4000, 500, key="m_mc_n")
    seed = c2.number_input("seed", 0, 9999, 0, 1, key="m_mc_seed")
    mc = compute.monte_carlo(pk, mw, int(n), int(seed))
    st.plotly_chart(charts.mc_hist(mc), use_container_width=True, key="m_mc")
    c3, c4, c5 = st.columns(3)
    c3.metric("P(space ≤ ground)", f"{mc['p_parity']:.0%}")
    c4.metric("Median ratio", f"{mc['median']:.2f}×")
    c5.metric("P10–P90", f"{mc['p10']:.2f}–{mc['p90']:.2f}×")


def _m_bracket(pk, mw, r):
    st.markdown("**Scenario bracketing** — report the headline as a band: optimist "
                "(Starcloud/Suncatcher/Gaalema) vs SA-central vs skeptic (Turyshev mass, "
                "non-financeable WACC, ISS-class radiators).")
    st.plotly_chart(charts.bracket_fig(compute.bracket_ratios(mw)), use_container_width=True,
                    key="m_bracket")


def _m_tornado(pk, mw, r):
    st.markdown("**Sensitivity / tornado** — one-at-a-time swing of the LCOC ratio as each "
                "provenance parameter moves from its skeptic to optimist end. Identifies the "
                "load-bearing assumptions.")
    st.plotly_chart(charts.tornado_fig(compute.tornado(pk, mw)), use_container_width=True,
                    key="m_tornado")


def _m_breakeven(pk, mw, r):
    be = r["breakeven_launch"]
    st.markdown("**Break-even solve** — the launch price (closed-form, linear in $/kg) at which "
                "space LCOC equals ground LCOC, holding everything else fixed.")
    st.latex(r"L_{\mathrm{be}}=\frac{\left(\frac{\mathrm{LCOC}_g\,\mathrm{kWh}_s}{10^6}"
             r"-\mathrm{Opex}_s\right)/\mathrm{CRF}_s - A}{M_{\mathrm{dry}}(1+f_{\mathrm{ins}})}\cdot 10^6")
    if be > 0:
        st.success(f"Break-even launch price ≈ **${be:,.0f}/kg** (current LCOC ratio "
                   f"{r['ratio']:.2f}×).")
    else:
        st.error("Break-even launch is **negative** — even free launch can't reach parity at "
                 "these inputs; the binding constraints are platform mass, station life and WACC.")


_ANALYTICAL = {
    "Levelized cost / TCO (SemiAnalysis)": _m_lcoc,
    "Communication-intensity / Γ-ceiling (Turyshev)": _m_gamma,
    "Orbital-edge-computing (Denby & Lucia)": _m_oec,
    "Monte-Carlo probabilistic": _m_montecarlo,
    "Scenario bracketing (optimist/skeptic)": _m_bracket,
    "Sensitivity / tornado": _m_tornado,
    "Break-even solve": _m_breakeven,
}


# ----------------------------------------------------------------- physical sub-models

def _s_power(r):
    pt = r["pt"]
    st.markdown("**Power / solar array.** Array sized to carry the bus while sunlit *and* "
                "recharge the eclipse battery, derated for end-of-life degradation and pointing.")
    st.latex(r"P_{\mathrm{BOL}}=\frac{P_{\mathrm{bus}}\cdot\frac{\text{sunlit}+"
             r"\text{ecl}/\eta_{rt}}{\text{sunlit}}}{(1-d)^{L}\,\eta_{\mathrm{point}}},\quad "
             r"A=\frac{P_{\mathrm{BOL}}}{S\,\eta_{\mathrm{cell}}\,\text{packing}}")
    _metrics([("Array BOL", f"{pt['arr_BOL_kW']:,.0f} kW"), ("Array area", f"{pt['A_array']:,.0f} m²"),
              ("Array mass", f"{pt['M_array']/1e3:.1f} t"), ("Array cost", f"${pt['C_array']:.1f}M")])


def _s_thermal(r):
    pt = r["pt"]
    st.markdown("**Thermal / radiator.** All consumed power becomes heat, rejected by two-sided "
                "panels via the Stefan–Boltzmann law. The implied W/m² is the single biggest "
                "model sensitivity.")
    st.latex(r"Q=2\,\varepsilon\,\eta_{\mathrm{fin}}\,\sigma\,(T_{\mathrm{rad}}^4-"
             r"T_{\mathrm{sink}}^4)\,A_{\mathrm{rad}}")
    _metrics([("Heat rejected", f"{pt['Q_kW']:,.0f} kW"),
              ("Per-side flux", f"{pt['q_net_side']:.0f} W/m²"),
              ("Radiator area", f"{pt['A_rad']:,.0f} m²"), ("Radiator mass", f"{pt['M_rad']/1e3:.1f} t")])


def _s_battery(r):
    pt = r["pt"]
    st.markdown("**Battery / eclipse storage.** Sized to carry full bus power through the single "
                "longest eclipse at the usable depth of discharge.")
    st.latex(r"E_{\mathrm{batt}}=\frac{P_{\mathrm{bus}}\cdot(t_{\mathrm{eclipse}}/60)}"
             r"{\mathrm{DoD}}")
    _metrics([("Battery energy", f"{pt['E_batt_kWh']:,.0f} kWh"),
              ("Battery mass", f"{pt['M_batt']/1e3:.2f} t"),
              ("Battery cost", f"${pt['C_batt']:.1f}M"), ("Sunlit fraction", f"{pt['sunlit']:.1%}")])


def _s_mass(r):
    st.markdown("**Mass budget.** Subsystem roll-up (array + radiator + IT + shield + battery) "
                "grossed up by a structure/ADCS/propulsion fraction. Launch cost scales directly "
                "with this dry mass.")
    st.plotly_chart(charts.mass_anatomy(r), use_container_width=True, key="m_sub_mass")


def _s_capex(r):
    st.markdown("**Launch & capex.** IT + platform + launch (= dry mass × $/kg) + integration + "
                "insurance, vs the terrestrial IT + facility.")
    st.plotly_chart(charts.capex_anatomy(r), use_container_width=True, key="m_sub_capex")


def _s_finance(r):
    st.markdown("**Finance / LCOC.** Capital-recovery levelization with split WACCs and mixed "
                "lives, plus a discounted NPV over each horizon. The 5-yr station vs 15-yr "
                "facility life mismatch is the structural driver.")
    _metrics([("LCOC space", f"${r['lcoc_s']:.2f}/kWh"), ("LCOC ground", f"${r['lcoc_g']:.2f}/kWh"),
              ("NPV space", f"${r['npv_s']:.0f}M"), ("NPV ground", f"${r['npv_g']:.0f}M")])


def _metrics(pairs):
    cols = st.columns(len(pairs))
    for col, (label, val) in zip(cols, pairs):
        col.metric(label, val)


_SUBMODELS = {
    "Power / solar array": _s_power,
    "Thermal / radiator": _s_thermal,
    "Battery / eclipse storage": _s_battery,
    "Mass budget": _s_mass,
    "Launch & capex": _s_capex,
    "Finance / LCOC": _s_finance,
}
