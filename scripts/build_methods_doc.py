"""
scripts/build_methods_doc.py — render docs/MODELING_METHODS.md from the literature search
=========================================================================================
Reads the web-backed, adversarially-verified research data in
``docs/modeling_methods_research.json`` (produced by the multi-agent literature-search
workflow) and renders a clean, cited Markdown review. The Streamlit "Methods" tab embeds
the output; regenerate with::

    python scripts/build_methods_doc.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "modeling_methods_research.json"
OUT = ROOT / "docs" / "MODELING_METHODS.md"

VERDICT = {"confirmed": "✅ confirmed", "partial": "🟡 partial",
           "refuted": "❌ refuted", "unverifiable": "⚪ unverifiable"}


def _cell(s: str, limit: int = 220) -> str:
    s = (s or "").replace("\n", " ").replace("|", "∕").strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _verified_map(method: dict) -> dict:
    vm = {}
    for vs in method.get("verified_sources", []) or []:
        src = vs.get("source", {}) or {}
        vm[src.get("url", "")] = vs.get("verification", {}) or {}
    return vm


def _sources_table(method: dict) -> list[str]:
    vm = _verified_map(method)
    rows = ["| Source | Year | Key claim | Verdict |", "|---|---|---|---|"]
    for s in method.get("key_sources", []):
        v = vm.get(s.get("url", ""))
        verdict = (v.get("corrected_verdict") if v else None) or s.get("verdict", "")
        url = (v.get("corrected_url") if v and v.get("corrected_url") else s.get("url", ""))
        title = _cell(s.get("title", "source"), 120)
        link = f"[{title}]({url})" if url else title
        rows.append(f"| {link} | {_cell(str(s.get('year','')),8)} | "
                    f"{_cell(s.get('key_claim',''))} | {VERDICT.get(verdict, verdict)} |")
    return rows


def render(data: dict) -> str:
    res = data["result"]
    methods, amb, sk = res["methods"], res.get("ambitious", {}), res.get("skeptic", {})
    n_agents = data.get("agentCount", "?")

    L: list[str] = []
    L.append("# Modelling methods for orbital data centre viability\n")
    L.append(
        "A literature review of the methods used to judge whether an orbital AI data centre "
        "can compete with a terrestrial one — written from a space-systems-engineering "
        "viewpoint that holds the **ambitious** gigawatt-constellation case (Starcloud, "
        "Google Project Suncatcher, Thales ASCEND) against the **skeptical / realistic** case "
        "(Turyshev mass, the 5-year FCC deorbit life vs 15-year terrestrial facility, "
        "non-financeable cost of capital, radiator W/m² that are not flight-demonstrated).\n")
    L.append(
        f"> **Provenance.** Compiled by a {n_agents}-agent web-backed search with adversarial "
        "citation verification: each load-bearing source was fetched and its claim "
        "independently checked (verdict ∈ confirmed / partial / refuted / unverifiable). "
        "Where verification corrected a source's URL or verdict, the corrected value is shown. "
        "Raw data: `docs/modeling_methods_research.json`. This file is generated — edit "
        "`scripts/build_methods_doc.py`, not the Markdown.\n")

    L.append("## How the methods combine\n")
    L.append(
        "Viability is a *conjunction*, so the methods stack rather than compete. "
        "**Physics-based subsystem sizing** turns an IT-power target into mass, area and cost; "
        "**parametric cost / CERs** and **levelized-cost (LCOC/TCO)** turn those into a $/compute "
        "number; the **communication-intensity (Γ) gate** and **orbital-edge-computing** models "
        "decide which workloads can actually run; **break-even / trade-space** finds the price or "
        "design point of parity; and **sensitivity** + **Monte-Carlo** quantify how much the "
        "verdict depends on the most contested inputs. The repo implements all of these over one "
        "calibrated kernel (`odc/`), so the app, notebook and dashboard agree by construction.\n")

    for i, m in enumerate(methods, 1):
        L.append(f"## {i}. {m['method']}\n")
        L.append(f"*{m['one_liner']}*\n")
        L.append(m["how_it_works"].strip() + "\n")
        L.append(f"**Ambitious read.** {m['ambitious_view'].strip()}\n")
        L.append(f"**Skeptical read.** {m['skeptic_view'].strip()}\n")
        L.append(f"**In this model.** {m['application_to_odc'].strip()}\n")
        L.extend(_sources_table(m))
        L.append("")

    if amb:
        L.append("## Ambitious programmes (the bull case)\n")
        if amb.get("summary"):
            L.append(amb["summary"].strip() + "\n")
        L.append("| Programme | Target | Status | Source |")
        L.append("|---|---|---|---|")
        for p in amb.get("programs", []):
            link = f"[link]({p['url']})" if p.get("url") else "—"
            L.append(f"| {_cell(p.get('name',''),60)} | {_cell(p.get('target',''),90)} | "
                     f"{_cell(p.get('status',''),90)} | {link} |")
        L.append("")

    if sk:
        L.append("## Skeptical / realistic critiques (the bear case)\n")
        if sk.get("summary"):
            L.append(sk["summary"].strip() + "\n")
        L.append("| Critique | Detail | Verdict | Source |")
        L.append("|---|---|---|---|")
        for c in sk.get("critiques", []):
            link = f"[link]({c['url']})" if c.get("url") else "—"
            L.append(f"| {_cell(c.get('point',''),90)} | "
                     f"{_cell(c.get('quantitative_detail',''),120)} | "
                     f"{VERDICT.get(c.get('verdict',''), c.get('verdict',''))} | {link} |")
        L.append("")

    L.append("---\n")
    L.append("*Generated from `docs/modeling_methods_research.json` by "
             "`scripts/build_methods_doc.py`. Citation counts and anchors are also tracked "
             "in `odc/provenance.py`.*")
    return "\n".join(L) + "\n"


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    OUT.write_text(render(data), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes) from {len(data['result']['methods'])} methods")


if __name__ == "__main__":
    main()
