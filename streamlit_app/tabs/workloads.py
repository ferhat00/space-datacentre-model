"""Tab 6 — workload Γ gate and the ground-energy comparator."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from odc import workloads, ground_energy
from .. import state, charts


def render() -> None:
    st.subheader("Workloads & ground energy")
    mw = state.size_mw()

    st.markdown("##### Workload Γ gate")
    st.caption("Does the workload's communication intensity sit under the Turyshev downlink "
               "ceiling at this platform power? Training + batch are the orbital-revenue "
               "workloads by decision; latency chat is ground-only.")
    st.plotly_chart(charts.gamma_fig(mw), use_container_width=True, key="wl_gamma")
    wrows = [dict(Workload=w.name, Value_rank=w.value_rank, Gamma=w.gamma_gb_per_kwh,
                  Latency=w.latency_class, Space_fit=w.space_suitable,
                  Headroom=f"{workloads.gamma_headroom(w, mw):.2f}×",
                  Closes="✅" if workloads.closes_in_orbit(w, mw) else "❌",
                  Orbital_rev="✅" if w.revenue_in_orbit else "—")
             for w in workloads.CATALOG.values()]
    st.dataframe(pd.DataFrame(wrows), hide_index=True, use_container_width=True)

    st.markdown("##### Ground energy comparator")
    st.caption("Space's strongest argument isn't cheaper electrons — it's time-to-power. US "
               "interconnect queues run ~40 mo to >7 yr; off-grid solar+battery and gas turbines "
               "are the real terrestrial fast paths.")
    st.plotly_chart(charts.ground_energy_fig(), use_container_width=True, key="wl_ground")
    grows = [dict(Source=s.name, USD_per_MWh=s.usd_per_mwh,
                  Time_to_power_mo=s.time_to_power_months, CF=s.capacity_factor,
                  kgCO2_per_MWh=s.carbon_kg_per_mwh,
                  Dispatchable="✅" if s.dispatchable else "—")
             for s in ground_energy.CATALOG.values()]
    gdf = pd.DataFrame(grows)
    st.dataframe(gdf.style.format({"USD_per_MWh": "{:.0f}", "Time_to_power_mo": "{:.0f}",
                 "CF": "{:.0%}", "kgCO2_per_MWh": "{:.0f}"}),
                 hide_index=True, use_container_width=True)
