"""
streamlit_app.compute — cached analysis functions over the OO model
===================================================================
Every function takes a hashable ``params_key`` (tuple of sorted P items) plus scalars, so
``st.cache_data`` can memoize across reruns. The ``ODCModel``/``P`` objects are built
*inside* each function (dataclasses aren't hashable). These wrap the same algorithms the
notebook uses, so the app, notebook and dashboard agree by construction.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from odc.core import P, ODCModel
from odc.scenarios import OPTIMIST, MATURE, SKEPTIC, SA_ANCHORS
from odc.provenance import monte_carlo_ranges
from odc.schema import _p_field_names


def _model(params_key: tuple, p_it_mw: float) -> ODCModel:
    return ODCModel(P(**dict(params_key)), p_it_mw)


@st.cache_data(show_spinner=False)
def evaluate(params_key: tuple, p_it_mw: float, include_delay: bool = False) -> dict:
    """Full LCOC/NPV evaluation as a nested plain dict (cache-friendly)."""
    return _model(params_key, p_it_mw).evaluate(include_delay).as_dict()


@st.cache_data(show_spinner=False)
def launch_curve(params_key: tuple, p_it_mw: float, lo: float = 50.0,
                 hi: float = 6000.0, n: int = 60) -> dict:
    """Space LCOC as a function of launch price, vs the (flat) ground LCOC."""
    base = dict(params_key)
    xs = np.logspace(np.log10(lo), np.log10(hi), n)
    ys = []
    lcoc_g = None
    for x in xs:
        r = ODCModel(P(**{**base, "launch_kg": float(x)}), p_it_mw).evaluate()
        ys.append(r.lcoc_s)
        lcoc_g = r.lcoc_g
    return dict(x=xs.tolist(), lcoc_s=ys, lcoc_g=lcoc_g)


# Provenance parameters that are real, tunable P fields (radiator_W_m2 is conceptual).
def _provenance_p_fields() -> dict:
    pf = _p_field_names()
    return {k: v for k, v in monte_carlo_ranges().items() if k in pf}


@st.cache_data(show_spinner=False)
def tornado(params_key: tuple, p_it_mw: float) -> list[dict]:
    """One-at-a-time sensitivity of the LCOC ratio to each provenance parameter."""
    base = dict(params_key)
    base_ratio = ODCModel(P(**base), p_it_mw).evaluate().ratio
    rows = []
    for field, (low, central, high) in _provenance_p_fields().items():
        r_lo = ODCModel(P(**{**base, field: float(low)}), p_it_mw).evaluate().ratio
        r_hi = ODCModel(P(**{**base, field: float(high)}), p_it_mw).evaluate().ratio
        rows.append(dict(field=field, low=r_lo, high=r_hi,
                         swing=abs(r_hi - r_lo)))
    rows.sort(key=lambda d: d["swing"])
    return dict(base=base_ratio, rows=rows)


@st.cache_data(show_spinner=False)
def monte_carlo(params_key: tuple, p_it_mw: float, n: int = 4000, seed: int = 0) -> dict:
    """Provenance-weighted triangular Monte Carlo over the LCOC ratio.

    Each provenance param is sampled triangular(min, central, max) with the SA-central
    value as the mode (low/high are the skeptic/optimist ends, not numerical bounds)."""
    base = dict(params_key)
    rng = np.random.default_rng(seed)
    fields = _provenance_p_fields()
    draws = {}
    for field, (low, central, high) in fields.items():
        lo, hi = min(low, central, high), max(low, central, high)
        mode = min(max(central, lo), hi)
        draws[field] = (rng.triangular(lo, mode, hi, size=n) if hi > lo
                        else np.full(n, central))
    ratios = np.empty(n)
    for i in range(n):
        over = {f: float(draws[f][i]) for f in fields}
        ratios[i] = ODCModel(P(**{**base, **over}), p_it_mw).evaluate().ratio
    return dict(ratios=ratios.tolist(),
                p_parity=float(np.mean(ratios <= 1.0)),
                median=float(np.median(ratios)),
                p10=float(np.percentile(ratios, 10)),
                p90=float(np.percentile(ratios, 90)))


@st.cache_data(show_spinner=False)
def bracket_ratios(p_it_mw: float) -> list[dict]:
    """LCOC ratio for the optimist / central / skeptic literature brackets."""
    out = []
    for sc in (OPTIMIST, MATURE, SKEPTIC):
        r = ODCModel(sc, p_it_mw).evaluate()
        out.append(dict(name=sc.name, ratio=r.ratio, lcoc_s=r.lcoc_s, lcoc_g=r.lcoc_g))
    return out


def sa_anchor_comparison(result: dict) -> list[dict]:
    """Rows comparing the current $/GPU-hr etc. to the SemiAnalysis published anchors."""
    sa = result["sa"]
    keys = [("gpu_hr_s", "$/GPU-hr (space)"), ("gpu_hr_g", "$/GPU-hr (ground)"),
            ("pflop_hr_s", "$/PFLOP-hr (space)"), ("pflop_hr_g", "$/PFLOP-hr (ground)"),
            ("btok_s", "$/B-tok (space)"), ("btok_g", "$/B-tok (ground)")]
    rows = []
    for k, label in keys:
        model = sa[k]
        anchor = SA_ANCHORS.get(k)
        rows.append(dict(metric=label, model=model, anchor=anchor,
                         delta_pct=(100.0 * (model - anchor) / anchor) if anchor else None))
    return rows
