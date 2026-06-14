"""
odc.schema — single source of truth for the tunable-parameter UI schema
=======================================================================
Both front-ends (the Chart.js dashboard built by ``build_dashboard.py`` and the Streamlit
app) bind their sliders to ``GROUPS`` below, so the two UIs can never drift apart.

``GROUPS`` is a list of ``[group_label, fields]`` where each field is
``[field, label, unit, lo, hi, step, kind, note]`` using the exact ``odc.core.P`` field
names. ``kind`` is one of ``None`` (linear number), ``"log"`` (log slider) or ``"pct"``
(stored as a fraction, shown as a percent).
"""
from __future__ import annotations

from dataclasses import fields as _dc_fields

from .model import P

# [field, label, unit, min, max, step, kind, note]  -- Python field names throughout.
GROUPS = [
 ["Launch & platform", [
  ["launch_kg", "Launch price", "$/kg", 50, 6000, 10, "log", "F9 actual '26 ~ $1.4-1.8k (SemiAnalysis); Starship target ~$250; verified $3,245/kg is naive list/expendable"],
  ["integration_M_MW", "Integration & NRE", "$M/MW", 1, 25, 0.5, None, None],
  ["insurance_frac", "Insurance", "% of hw+launch", 0, 0.15, 0.005, "pct", "v3: market caps ~$300M/risk -> self-insure via redundancy"],
  ["structure_frac", "Structure/ADCS/prop", "% of subsystem mass", 0.08, 0.30, 0.01, "pct", None],
  ["avionics_comms_M_MW", "Avionics & optical comms", "$M/MW", 1, 15, 0.5, None, None],
 ]],
 ["Orbit & storage", [
  ["eclipse_min_day", "Longest eclipse", "min", 0, 80, 1, None, "Dawn-dusk SSO ~35 min/day; the orbit selector sets this + daily fraction"],
  ["batt_Wh_kg", "Battery specific energy", "Wh/kg pack", 100, 350, 5, None, None],
  ["batt_cost_kWh", "Battery cost", "$/kWh installed", 80, 800, 10, None, None],
  ["batt_dod", "Usable depth of discharge", "%", 0.5, 0.95, 0.05, "pct", None],
 ]],
 ["Power", [
  ["sp_array", "Array specific power", "W/kg BOL", 50, 500, 5, None, "iROSA flight ~75 (verified); ROSA-class ~110; Mega-ROSA target 300+ (unproven)"],
  ["array_cost_W", "Array cost", "$/W BOL", 2, 100, 1, None, None],
  ["cell_eff", "Cell efficiency", "", 0.24, 0.40, 0.005, "pct", None],
  ["degr_rate", "Degradation", "%/yr", 0.005, 0.04, 0.0025, "pct", None],
  ["overhead_frac", "Bus overhead (pumps, comms)", "% of IT", 0.03, 0.15, 0.005, "pct", None],
 ]],
 ["Thermal", [
  ["T_rad", "Radiating temperature", "K", 285, 350, 2.5, None, "Liquid cold plates allow 45-60 C -> 318-333 K; W/m^2 is the headline sensitivity"],
  ["T_sink", "Effective sink", "K", 200, 260, 5, None, None],
  ["rad_areal_kg_m2", "Radiator areal density", "kg/m2 panel", 1.5, 14, 0.25, None, "ISS ~14 today (verified); mature target ~6; the '3.5' MARVL figure is a units conflation"],
  ["rad_cost_m2", "Radiator cost", "$/m2", 300, 6000, 100, None, None],
 ]],
 ["IT & reliability", [
  ["it_kg_per_kW", "IT specific mass (system)", "kg/kW", 6, 60, 0.5, None, "AI1 ~14; Starcloud ~19.5; Turyshev 34-59 (verified spread)"],
  ["it_cost_W", "IT capex (both sides)", "$/W", 15, 50, 1, None, None],
  ["overprovision", "Overprovision (no repair)", "%", 0.03, 0.30, 0.01, "pct", None],
  ["compute_overhead", "Checkpoint/SEU overhead", "%", 0, 0.12, 0.005, "pct", None],
  ["rad_availability", "Radiation availability", "%", 0.88, 1.0, 0.005, "pct", "SemiAnalysis: 95% net of solar-event downtime"],
  ["shield_t_per_MW", "Spot shielding", "t/MW", 0, 5, 0.25, None, None],
  ["life_yr", "Station life", "yr", 3, 12, 0.5, None, "SemiAnalysis: 5 yr now -> 10 post-2032; FCC 5-yr deorbit caps it"],
 ]],
 ["Economics", [
  ["rev0_kWh", "Compute price, year 0", "$/kW-h", 0.8, 5, 0.1, None, None],
  ["rev_decline", "Price erosion", "%/yr", 0.05, 0.30, 0.01, "pct", "~13%/yr Starlink ARPU; raw token deflation is far faster"],
  ["util", "Utilisation", "%", 0.6, 1.0, 0.02, "pct", None],
  ["wacc_space", "WACC . space", "%", 0.08, 0.20, 0.0025, "pct", "SemiAnalysis 15%->10.3%; 20% = non-financeable case (most fragile param)"],
  ["wacc_ground", "WACC . ground", "%", 0.07, 0.14, 0.0025, "pct", None],
  ["ops_M_MW_yr", "Space ops", "$M/MW/yr", 0.2, 3, 0.1, None, None],
 ]],
 ["Terrestrial benchmark", [
  ["g_facility_M_MW", "Facility capex", "$M/MW", 6, 25, 0.5, None, "Grid $12-15; crypto-conv $10-15; BTM $15-20; industrial >$20"],
  ["g_facility_life_yr", "Facility life", "yr", 8, 20, 1, None, "15-yr building vs 5-yr station -- the biggest single driver"],
  ["g_elec_MWh", "Electricity", "$/MWh", 40, 250, 5, None, None],
  ["g_pue", "PUE", ">1", 1.05, 1.6, 0.01, None, "Hyperscale now 1.10-1.15; little headroom for space"],
  ["g_ops_M_MW_yr", "Ground ops", "$M/MW/yr", 0.3, 3, 0.1, None, None],
  ["g_delay_yr", "Interconnect queue", "yr", 0, 6, 0.25, None, "Used only in scarcity mode"],
 ]],
]


def slider_fields() -> list[str]:
    """Flat list of every P field that appears as a slider, in display order."""
    return [f[0] for _, fields in GROUPS for f in fields]


def field_meta() -> dict:
    """Map field name -> dict(label, unit, lo, hi, step, kind, note, group)."""
    out = {}
    for group_label, fields in GROUPS:
        for field, label, unit, lo, hi, step, kind, note in fields:
            out[field] = dict(label=label, unit=unit, lo=lo, hi=hi, step=step,
                              kind=kind, note=note, group=group_label)
    return out


def _p_field_names() -> set[str]:
    return {f.name for f in _dc_fields(P)}
