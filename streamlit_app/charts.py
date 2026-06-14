"""
streamlit_app.charts — Plotly figure builders
=============================================
Pure functions: given a result dict (or model inputs) return a Plotly figure. No Streamlit
or session-state dependency, so they are unit-testable on their own.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from odc import workloads, ground_energy

SPACE = "#e8542f"     # orbital
GROUND = "#2f6fe8"    # terrestrial
ACCENT = "#1b9e77"
GREY = "#8a8f98"
_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=360,
               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
               legend=dict(orientation="h", y=1.12, x=0))


def _fig(**over):
    f = go.Figure()
    lay = {**_LAYOUT, **over}
    f.update_layout(**lay)
    return f


def lcoc_bar(result: dict) -> go.Figure:
    f = _fig(title="Levelized cost of compute ($/kW-h)")
    f.add_bar(x=["Space", "Ground"], y=[result["lcoc_s"], result["lcoc_g"]],
              marker_color=[SPACE, GROUND],
              text=[f"${result['lcoc_s']:.2f}", f"${result['lcoc_g']:.2f}"],
              textposition="outside")
    f.update_yaxes(title="$/kW-h", rangemode="tozero")
    return f


def capex_anatomy(result: dict) -> go.Figure:
    cs, cg = result["cap_s"], result["cap_g"]
    f = _fig(title="Capex anatomy ($M per MW)", barmode="stack")
    space_parts = [("IT", cs["C_it"]), ("Platform", cs["C_platform"]),
                   ("Launch", cs["C_launch"]), ("Integration", cs["C_int"]),
                   ("Insurance", cs["C_ins"])]
    ground_parts = [("IT", cg["C_it"]), ("Facility", cg["C_fac"])]
    palette = {"IT": "#e8542f", "Platform": "#f29e4c", "Launch": "#c0392b",
               "Integration": "#9b59b6", "Insurance": "#7f8c8d", "Facility": "#2f6fe8"}
    for label, _ in dict(space_parts + ground_parts).items():
        sv = dict(space_parts).get(label, 0.0)
        gv = dict(ground_parts).get(label, 0.0)
        f.add_bar(name=label, x=["Space", "Ground"], y=[sv, gv],
                  marker_color=palette.get(label, GREY))
    f.update_yaxes(title="$M / MW", rangemode="tozero")
    return f


def mass_anatomy(result: dict) -> go.Figure:
    pt = result["pt"]
    m_sub = pt["M_array"] + pt["M_rad"] + pt["M_it"] + pt["M_shield"] + pt["M_batt"]
    structure = pt["M_dry"] - m_sub
    parts = [("Array", pt["M_array"]), ("Radiator", pt["M_rad"]), ("IT", pt["M_it"]),
             ("Battery", pt["M_batt"]), ("Shield", pt["M_shield"]),
             ("Structure/ADCS", structure)]
    f = _fig(title=f"Dry-mass anatomy — {pt['M_dry']/1e3:.1f} t/MW", barmode="stack")
    for label, v in parts:
        f.add_bar(name=label, x=["Dry mass"], y=[v / 1e3])
    f.update_yaxes(title="tonnes / MW", rangemode="tozero")
    return f


def launch_curve_fig(curve: dict, current_launch: float, current_lcoc_s: float) -> go.Figure:
    f = _fig(title="Space LCOC vs launch price")
    f.add_scatter(x=curve["x"], y=curve["lcoc_s"], mode="lines",
                  name="Space LCOC", line=dict(color=SPACE, width=3))
    f.add_hline(y=curve["lcoc_g"], line=dict(color=GROUND, dash="dash"),
                annotation_text=f"Ground ${curve['lcoc_g']:.2f}", annotation_position="top left")
    f.add_scatter(x=[current_launch], y=[current_lcoc_s], mode="markers",
                  name="Current", marker=dict(color="black", size=11, symbol="x"))
    f.update_xaxes(title="Launch price ($/kg)", type="log")
    f.update_yaxes(title="$/kW-h", type="log")
    return f


def tornado_fig(tornado_data: dict) -> go.Figure:
    base = tornado_data["base"]
    rows = tornado_data["rows"]
    f = _fig(title=f"Sensitivity of LCOC ratio (base = {base:.2f}x)", height=460)
    for r in rows:
        lo, hi = r["low"], r["high"]
        f.add_bar(y=[r["field"]], x=[hi - lo], base=[min(lo, hi)],
                  orientation="h", marker_color=ACCENT, showlegend=False,
                  hovertemplate=f"{r['field']}: {min(lo,hi):.2f} → {max(lo,hi):.2f}<extra></extra>")
    f.add_vline(x=base, line=dict(color="black", dash="dot"))
    f.add_vline(x=1.0, line=dict(color=GROUND, dash="dash"),
                annotation_text="parity", annotation_position="top")
    f.update_xaxes(title="LCOC ratio (space / ground)")
    return f


def gamma_fig(p_it_mw: float) -> go.Figure:
    mw = np.logspace(-2, 3, 80)
    ceiling = workloads.gamma_ceiling_gb_per_kwh
    f = _fig(title="Workload Γ gate — comms-intensity ceiling vs demand", height=400)
    f.add_scatter(x=mw, y=[ceiling(m) for m in mw], mode="lines",
                  name="Turyshev ceiling", line=dict(color="black", width=3))
    for w in workloads.CATALOG.values():
        col = ACCENT if w.revenue_in_orbit else GREY
        f.add_hline(y=w.gamma_gb_per_kwh, line=dict(color=col, dash="dot"),
                    annotation_text=w.name.split(" (")[0], annotation_position="right")
    f.add_vline(x=p_it_mw, line=dict(color=SPACE, dash="dash"),
                annotation_text=f"{p_it_mw:g} MW", annotation_position="top")
    f.update_xaxes(title="Platform IT power (MW)", type="log")
    f.update_yaxes(title="GB moved per kWh of IT", type="log")
    return f


def ground_energy_fig() -> go.Figure:
    srcs = list(ground_energy.CATALOG.values())
    f = _fig(title="Ground energy: time-to-power vs $/MWh (bubble = capacity factor)",
             height=440)
    f.add_scatter(
        x=[s.time_to_power_months for s in srcs],
        y=[s.usd_per_mwh for s in srcs],
        mode="markers+text",
        text=[s.name.split(" (")[0] for s in srcs],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=[12 + 26 * s.capacity_factor for s in srcs],
                    color=[s.carbon_kg_per_mwh for s in srcs],
                    colorscale="RdYlGn_r", showscale=True,
                    colorbar=dict(title="kgCO₂/MWh"), line=dict(width=1, color="white")),
        hovertext=[f"{s.name}<br>${s.usd_per_mwh:.0f}/MWh · {s.time_to_power_months:.0f} mo · "
                   f"CF {s.capacity_factor:.0%}" for s in srcs],
        hoverinfo="text", showlegend=False)
    f.update_xaxes(title="Time to power (months)")
    f.update_yaxes(title="$/MWh", rangemode="tozero")
    return f


def mc_hist(mc: dict) -> go.Figure:
    f = _fig(title=f"Monte Carlo — P(space ≤ ground) = {mc['p_parity']:.0%}", height=400)
    f.add_histogram(x=mc["ratios"], nbinsx=60, marker_color=SPACE, opacity=0.8)
    f.add_vline(x=1.0, line=dict(color=GROUND, dash="dash"),
                annotation_text="parity", annotation_position="top")
    f.add_vline(x=mc["median"], line=dict(color="black", dash="dot"),
                annotation_text=f"median {mc['median']:.2f}x", annotation_position="top right")
    f.update_xaxes(title="LCOC ratio (space / ground)")
    f.update_yaxes(title="draws")
    return f


def bracket_fig(rows: list[dict]) -> go.Figure:
    f = _fig(title="Literature brackets — LCOC ratio", height=320)
    f.add_bar(x=[r["name"].split(" (")[0] for r in rows], y=[r["ratio"] for r in rows],
              marker_color=[ACCENT, "black", SPACE],
              text=[f"{r['ratio']:.2f}x" for r in rows], textposition="outside")
    f.add_hline(y=1.0, line=dict(color=GROUND, dash="dash"), annotation_text="parity")
    f.update_yaxes(title="space / ground", rangemode="tozero")
    return f


def ladder_mass_fig(rows: list[dict]) -> go.Figure:
    f = _fig(title="Size ladder — anchor vs model dry mass", barmode="group", height=380)
    names = [r["name"] for r in rows]
    f.add_bar(name="Anchor (literature)", x=names, y=[r["anchor_mass_t"] for r in rows],
              marker_color=GREY)
    f.add_bar(name="Model", x=names, y=[r["model_dry_t"] for r in rows], marker_color=SPACE)
    f.update_yaxes(title="tonnes", type="log")
    f.update_xaxes(tickangle=-30)
    return f
