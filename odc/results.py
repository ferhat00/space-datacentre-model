"""
odc.results — typed, dict-compatible result objects for the OO model
====================================================================
The calibrated kernel historically returned plain ``dict``s. These frozen dataclasses
replace those dicts with typed objects that ALSO behave like the old dicts — they
support ``obj["key"]`` indexing, ``"key" in obj``, ``.get``, ``.keys``/``.values``/
``.items``, ``**obj`` unpacking and ``dict(obj)`` — so every existing consumer
(``tests/``, ``build_dashboard.py``, ``build_notebook.py``) keeps working unchanged,
while new code can use attribute access (``r.lcoc_s``) and static typing.

Field names and order mirror the original ``dict(...)`` literals in the v2 kernel
verbatim, so ``Result(**legacy_dict)`` round-trips and ``result.as_dict()`` reproduces
the legacy nested-dict shape exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields


class _DictResult:
    """Mixin that makes a dataclass a drop-in replacement for the legacy result dict.

    Indexing, membership, and mapping-protocol methods are delegated to the dataclass
    fields, so `pt["A_rad"]`, `"A_rad" in pt`, `dict(pt)` and `**pt` all behave like the
    plain dict the kernel used to return.
    """

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as exc:  # present a dict-like KeyError, not AttributeError
            raise KeyError(key) from exc

    def __contains__(self, key) -> bool:
        return key in self._field_names()

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return list(self._field_names())

    def values(self):
        return [getattr(self, k) for k in self._field_names()]

    def items(self):
        return [(k, getattr(self, k)) for k in self._field_names()]

    def as_dict(self) -> dict:
        """Legacy nested-dict form (deep): identical to the v2 kernel's return shape."""
        return asdict(self)

    @classmethod
    def _field_names(cls):
        return tuple(f.name for f in fields(cls))


@dataclass(frozen=True)
class PowerThermalMass(_DictResult):
    """Power / thermal / mass budget for a platform (one IT-power basis)."""
    sell_kW: float
    gross_kW: float
    bus_kW: float
    sunlit: float
    E_batt_kWh: float
    M_batt: float
    C_batt: float
    arr_BOL_kW: float
    A_array: float
    M_array: float
    C_array: float
    Q_kW: float
    q_net_side: float
    A_rad: float
    M_rad: float
    C_rad: float
    M_it: float
    M_shield: float
    M_dry: float


@dataclass(frozen=True)
class SpaceCapex(_DictResult):
    """Orbital data-centre capex breakdown ($M)."""
    C_it: float
    C_platform: float
    C_launch: float
    C_int: float
    C_ins: float
    total: float


@dataclass(frozen=True)
class GroundCapex(_DictResult):
    """Terrestrial data-centre capex breakdown ($M)."""
    C_it: float
    C_fac: float
    total: float


@dataclass(frozen=True)
class SAUnits(_DictResult):
    """SemiAnalysis cross-check units (wall-clock convention)."""
    gpu_hr_s: float
    gpu_hr_g: float
    pflop_hr_s: float
    pflop_hr_g: float
    btok_s: float
    btok_g: float


@dataclass(frozen=True)
class LCOCResult(_DictResult):
    """Full levelized-cost / NPV evaluation for space vs ground."""
    pt: PowerThermalMass
    cap_s: SpaceCapex
    cap_g: GroundCapex
    lcoc_s: float
    lcoc_g: float
    ratio: float
    npv_s: float
    npv_g: float
    breakeven_launch: float
    kwh_s: float
    kwh_g: float
    ann_s: float
    ann_g: float
    sa: SAUnits
