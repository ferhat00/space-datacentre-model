"""
Orbital Data Centre (ODC) viability model — v3 (modular)
========================================================
A coupled power -> thermal -> mass -> launch -> capex -> LCOC/NPV system model of an
orbital AI data centre, benchmarked against a terrestrial data centre running identical
silicon, and extended with:

  * workloads   - AI workload taxonomy (training / batch / latency / embeddings) with
                  value-per-kWh and Turyshev communication-intensity (Gamma) gating.
  * hardware    - GPU/accelerator matrix (power, $/W, kg/kW, radiation tolerance, tok/s).
  * sizes       - six-rung spacecraft x GPU size ladder, from ESPA edge node to GW.
  * orbits      - orbit families (dawn-dusk SSO, equatorial LEO, high LEO, MEO/GEO).
  * comms       - inter-satellite + downlink capacity/cost/availability + Gamma feasibility.
  * ground_energy - terrestrial energy-source comparator ($/MWh, time-to-power, CF).
  * finance     - WACC trajectory, non-financeable case, self-insurance, scarcity NPV.
  * provenance  - every default carries a source + citation strength.

v3 keeps v2's calibrated physics/finance in `core` unchanged (the SemiAnalysis 2026
reproduction remains a regression guard, within ~2%), and applies the verified
literature corrections from the 2026 adversarial review as documentation + new
bracketing scenarios rather than by retuning the calibrated presets.

Basis: 1 MW of *sellable* IT load, dusk-dawn SSO ~550 km, scaled linearly.
Units SI unless noted. Money USD; $M = 1e6.
"""
from .core import (
    P, SIGMA, SOLAR_CONST,
    B300_KW_PER_GPU, B300_PFLOPS_FP4, B300_TOKS_PER_S,
    crf, power_thermal_mass, space_capex, ground_capex,
    annual_compute_kWh, lcoc_and_npv,
)
from .scenarios import TODAY, EARLY, MATURE, SA26, SCENARIOS, SA_ANCHORS

__all__ = [
    "P", "SIGMA", "SOLAR_CONST",
    "B300_KW_PER_GPU", "B300_PFLOPS_FP4", "B300_TOKS_PER_S",
    "crf", "power_thermal_mass", "space_capex", "ground_capex",
    "annual_compute_kWh", "lcoc_and_npv",
    "TODAY", "EARLY", "MATURE", "SA26", "SCENARIOS", "SA_ANCHORS",
]

__version__ = "3.0.0"
