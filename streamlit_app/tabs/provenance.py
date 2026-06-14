"""Tab 7 — provenance ledger and citation anchors."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from odc import provenance

_VERDICT_COLOR = {"confirmed": "#1b9e77", "partial": "#d9a441",
                  "refuted": "#d1495b", "unverifiable": "#8a8f98"}


def render() -> None:
    st.subheader("Provenance & citations")
    st.caption("Every load-bearing default carries a source, a skeptic↔optimist range, a "
               "confidence level and the adversarial-verification verdict from the 2026 review. "
               "These ranges also drive the Monte Carlo.")

    rows = [dict(Parameter=p.field, Central=p.central, Low=p.low, High=p.high, Unit=p.unit,
                 Confidence=p.confidence, Verdict=p.verdict, Source=p.source, Note=p.note)
            for p in provenance.REGISTRY.values()]
    df = pd.DataFrame(rows)

    def _color(v):
        return f"background-color: {_VERDICT_COLOR.get(v, '#fff')}33"

    styled = df.style.map(_color, subset=["Verdict"]).format(
        {"Central": "{:.3g}", "Low": "{:.3g}", "High": "{:.3g}"})
    st.dataframe(styled, hide_index=True, use_container_width=True)

    st.markdown("##### Academic citation anchors (Semantic Scholar)")
    cites = sorted(provenance.CITATIONS.items(), key=lambda kv: -kv[1])
    cdf = pd.DataFrame(cites, columns=["Work", "Citations"])
    st.bar_chart(cdf.set_index("Work"), horizontal=True)
    st.caption(f"SemiAnalysis anchor (verified, LCOC basis): "
               f"${provenance.SA_VERIFIED['gpu_hr_s']}/{provenance.SA_VERIFIED['gpu_hr_g']} "
               f"per GPU-hr — {provenance.SA_VERIFIED['note']}")
