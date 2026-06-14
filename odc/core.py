"""
odc.core — backward-compatible functional API over the OO kernel (odc.model)
============================================================================
Historically this module WAS the calibrated kernel (a procedural port of v2). In v3.1 the
kernel was refactored into an object-oriented form in :mod:`odc.model` (the ``ODCModel``
class). The arithmetic moved *verbatim* — identical floating-point operations in the same
order — so the SemiAnalysis-2026 calibration (``SA26``) and the JS dashboard parity badge
are unchanged (regression-guarded in ``tests/``).

This module is now a thin compatibility shim. It re-exports the parameter record (``P``),
the physical constants, ``crf`` and the ``ODCModel``/``Spacecraft`` classes, and keeps the
original free functions (``power_thermal_mass``, ``space_capex``, ``ground_capex``,
``annual_compute_kWh``, ``lcoc_and_npv``) as wrappers that delegate to ``ODCModel`` and
return the dict-compatible result objects in :mod:`odc.results`. Existing callers that index
results like ``pt["A_rad"]`` or ``r["lcoc_s"]`` keep working unchanged; new code can use the
``ODCModel`` class and attribute access (``r.lcoc_s``) directly.
"""
from .model import (
    P, SIGMA, SOLAR_CONST,
    B300_KW_PER_GPU, B300_PFLOPS_FP4, B300_TOKS_PER_S,
    crf, ODCModel, Spacecraft,
)
from .results import (
    PowerThermalMass, SpaceCapex, GroundCapex, SAUnits, LCOCResult,
)

__all__ = [
    "P", "SIGMA", "SOLAR_CONST",
    "B300_KW_PER_GPU", "B300_PFLOPS_FP4", "B300_TOKS_PER_S",
    "crf", "ODCModel", "Spacecraft",
    "PowerThermalMass", "SpaceCapex", "GroundCapex", "SAUnits", "LCOCResult",
    "power_thermal_mass", "space_capex", "ground_capex",
    "annual_compute_kWh", "lcoc_and_npv",
]


def power_thermal_mass(p: P, P_it_MW: float = 1.0) -> PowerThermalMass:
    """Power/thermal/mass budget. Delegates to :meth:`ODCModel.power_thermal_mass`."""
    return ODCModel(p, P_it_MW).power_thermal_mass()


def space_capex(p: P, pt, P_it_MW: float = 1.0) -> SpaceCapex:
    """Orbital capex breakdown. Delegates to :meth:`ODCModel.space_capex`."""
    return ODCModel(p, P_it_MW).space_capex(pt)


def ground_capex(p: P, P_it_MW: float = 1.0) -> GroundCapex:
    """Terrestrial capex breakdown. Delegates to :meth:`ODCModel.ground_capex`."""
    return ODCModel(p, P_it_MW).ground_capex()


def annual_compute_kWh(p: P, side: str, P_it_MW: float = 1.0) -> float:
    """Sellable compute kW-h/yr for ``side`` ('space' | 'ground')."""
    return ODCModel(p, P_it_MW).annual_compute_kWh(side)


def lcoc_and_npv(p: P, P_it_MW: float = 1.0, include_delay: bool = False) -> LCOCResult:
    """Full LCOC/NPV evaluation. Delegates to :meth:`ODCModel.evaluate`."""
    return ODCModel(p, P_it_MW).evaluate(include_delay)
