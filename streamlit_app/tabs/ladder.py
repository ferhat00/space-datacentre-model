"""Tab 5 — spacecraft × GPU size ladder and orbit families."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from odc import sizes, orbits
from odc.core import power_thermal_mass, P
from .. import state, charts


def render() -> None:
    st.subheader("Size ladder & orbit families")

    scenario = state.PRESETS.get(st.session_state.preset_name)
    if scenario is None:                                # "Custom" -> use current params
        scenario = P(**state.current_params())

    st.markdown("##### Spacecraft × GPU size ladder")
    st.caption("Six reference classes from a single-GPU ESPA edge node to a GW constellation, "
               "each run through the calibrated model at its IT power under the active scenario. "
               "The LCOC ratio is scale-invariant (the core scales linearly per MW); rungs "
               "differ in GPU count and anchor-vs-model mass.")
    rows = sizes.ladder_table(scenario)
    df = pd.DataFrame(rows)[["key", "name", "it_mw", "gpu", "gpu_count", "anchor_mass_t",
                             "anchor_kg_per_kw", "model_dry_t", "model_kg_per_kw", "source"]]
    df.columns = ["", "Class", "IT (MW)", "GPU", "# GPU", "Anchor t", "Anchor kg/kW",
                  "Model t", "Model kg/kW", "Source"]
    st.dataframe(df.style.format({"IT (MW)": "{:.3f}", "# GPU": "{:,.0f}", "Anchor t": "{:.1f}",
                 "Anchor kg/kW": "{:.1f}", "Model t": "{:.1f}", "Model kg/kW": "{:.1f}"}),
                 hide_index=True, use_container_width=True)
    st.plotly_chart(charts.ladder_mass_fig(rows), use_container_width=True, key="ladder_mass")

    st.markdown("##### Orbit families")
    st.caption("The orbit sets eclipse (→ battery/array), radiation dose (→ shielding), latency "
               "(→ which workloads are viable) and FCC deorbit compliance. Dry mass below is the "
               "active scenario stamped with each orbit's eclipse profile.")
    orows = []
    for o in orbits.CATALOG.values():
        pt = power_thermal_mass(o.apply(scenario))
        orows.append(dict(Orbit=o.name, Alt_km=o.altitude_km,
                          Sun_pct=o.sun_frac_annual, Dose_x=o.tid_dose_mult,
                          RTT_ms=o.latency_rtt_ms, FCC_ok="✅" if o.fcc_deorbit_ok else "❌",
                          Dry_t_MW=pt["M_dry"] / 1e3))
    odf = pd.DataFrame(orows)
    st.dataframe(odf.style.format({"Alt_km": "{:,.0f}", "Sun_pct": "{:.0%}", "Dose_x": "{:.1f}",
                 "RTT_ms": "{:.0f}", "Dry_t_MW": "{:.1f}"}),
                 hide_index=True, use_container_width=True)
