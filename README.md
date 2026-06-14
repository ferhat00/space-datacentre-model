# Space Data Centre Viability Model

A coupled **power → thermal → mass → launch → capex → LCOC/NPV** system model of an
orbital AI data centre (1 MW sellable IT, ~550 km dusk-dawn sun-synchronous orbit),
benchmarked against a terrestrial data centre running identical silicon.

## v3 (modular package, in progress)

`v3` generalises the single-file v2 into a literature-grounded modular package, `odc/`,
backed by a 24-agent review of 130 sources (academic citation counts checked on Semantic
Scholar; 12 load-bearing numbers adversarially verified). The v2 physics/finance are
ported **unchanged** into `odc/core.py`, so the SemiAnalysis 2026 reproduction (`SA26`)
still calibrates within ~2% — guarded by `tests/test_regression.py` (22 tests).

| Module | Adds |
|---|---|
| `odc/core.py` | v2 physics + finance (calibrated), with verified-correction provenance comments |
| `odc/scenarios.py` | TODAY / EARLY / MATURE / SA26 **+ OPTIMIST / SKEPTIC** literature brackets |
| `odc/workloads.py` | training / batch / latency / embeddings taxonomy + Turyshev comms-intensity **Γ gate** (training + batch are the orbital-revenue workloads) |
| `odc/hardware.py` | GPU/accelerator matrix (Jetson → H100 → B300 → GB300 NVL72 → Trillium TPU) |
| `odc/sizes.py` | six-rung spacecraft × GPU **size ladder** (ESPA edge node → GW constellation) with anchored masses |
| `odc/orbits.py` | DDSS / equatorial LEO / high LEO / MEO / GEO — eclipse, dose, latency, drag, FCC |
| `odc/comms.py` | ISL + downlink capacity/availability + Γ feasibility |
| `odc/ground_energy.py` | terrestrial energy comparator (grid/gas/SMR/solar+storage/geothermal/fuel-cell) — space competes on **time-to-power**, not $/MWh |
| `odc/finance.py` | WACC trajectory, non-financeable 20% case, self-insurance, scarcity NPV |
| `odc/provenance.py` | every default carries source + citation strength + verification verdict |

```bash
python -m odc.presets       # full v3 summary: eras, brackets, size ladder, orbits, Γ gate
python -m odc.scenarios     # calibrated table + SA26 reproduction + brackets
python -m pytest tests/     # 22 tests incl. the SA26 calibration guard
python build_notebook.py    # rebuild & execute notebooks/orbital_datacentre_viability.ipynb (v3, 18 sections)
```

The **notebook is regenerated from the package** (`build_notebook.py` now imports `odc`
rather than embedding source) — 18 sections including the literature brackets, the
workload Γ gate, the size ladder, orbit families, the ground-energy comparator, and the
provenance/citation ledger, alongside the original headline table, SA calibration,
launch curves, mass/capex anatomy, tornado, viability map, and a provenance-weighted
Monte Carlo.

The interactive **dashboard is regenerated from the package** too: `build_dashboard.py`
exports presets (a direct `asdict` dump), the slider schema, and all v3 data to one JSON
baked into the HTML; the JavaScript keeps **only the arithmetic** and **self-checks against
Python-computed golden results on load** (a parity badge — currently 0.000% drift across
all 6 presets, which retires the old hand-sync risk). It adds era/orbit/size/workload chip
rows, the Γ-gate chart, the size-ladder table, and the ground-energy scatter.

```bash
python build_dashboard.py   # writes dashboard/*.html (CDN + offline standalone) from odc
```

---

## v3.1 — object-oriented kernel + Streamlit app + literature review

**Object-oriented kernel.** The calibrated physics/finance now lives in an `ODCModel`
class (`odc/model.py`); the parameter record `P`, the constants and `crf` moved there too.
The arithmetic is a *verbatim* port of the v2 functions (identical float operations in the
same order), so SA26 still reproduces the SemiAnalysis anchors within ~2% and the dashboard
parity badge stays at 0.000% drift. `odc/core.py` is now a thin backward-compatible shim:
the old free functions (`power_thermal_mass`, `lcoc_and_npv`, ...) still work and return
typed, **dict-compatible** result objects (`odc/results.py`), so `r["lcoc_s"]` and
`r.lcoc_s` are equivalent and every existing consumer keeps working unchanged.

```python
from odc import ODCModel, SA26
from odc.orbits import EQUATORIAL_LEO
from odc.sizes import STATION_16MW

r = ODCModel.from_scenario(SA26).with_orbit(EQUATORIAL_LEO).with_size(STATION_16MW).evaluate()
print(r.lcoc_s, r.ratio, r.pt.M_dry)        # attribute access; r["lcoc_s"] also works
```

**Streamlit app** (`app.py` + `streamlit_app/`). A 7-tab cockpit that tweaks every
parameter live: **Methods** (the modelling approaches *and* physical sub-models, each
runnable), **Parameters** (the ~37 sliders), **Results**, **Sensitivity & Monte Carlo**,
**Size & orbits**, **Workloads & energy**, **Provenance**. It is additive — it shares the
`odc` package and the slider schema (`odc/schema.py`, now also used by `build_dashboard.py`)
with the Chart.js dashboard, which is untouched.

```bash
pip install -e .[app]        # or: pip install -r requirements.txt
streamlit run app.py
```

**Literature review** (`docs/MODELING_METHODS.md`). A web-backed, adversarially-verified
survey of the modelling methods (LCOC/TCO, Γ-ceiling, orbital-edge-computing, Monte Carlo,
parametric CERs, sensitivity, break-even, physics sizing), balancing the ambitious
GW-constellation case against the skeptical/realistic one. Regenerate from the saved
research data with `python scripts/build_methods_doc.py`.

---

## v2 (single-file model)

**v2 (12 Jun 2026)** is recalibrated against the SemiAnalysis *AI Space Datacenter
TCO Model* introduction — *"To Boldly Go: The Case for Space Datacenters"*
(Nishball, Myana, Holbrook et al., 3 Jun 2026) — and carries a preset that
reproduces their published 2026 numbers to within ~2%.

## Headline results (per MW sellable IT)

| Scenario | Launch $/kg | Dry t/MW | Capex space vs ground | LCOC ratio | $/GPU-hr (B300, SA conv.) | Breakeven launch |
|---|---|---|---|---|---|---|
| **Today (2026, Falcon)** | 1,600 | 56.8 | $211M vs $46M | **5.58×** | $14.89 / $2.67 | **negative** (platform-bound) |
| **Early Starship (~2028-30)** | 700 | 37.8 | $94M vs $41M | 2.57× | $6.24 / $2.43 | negative |
| **Mature Starship (~2033-35)** | 250 | 22.6 | $46M vs $37M | **0.95× (parity)** | $2.08 / $2.19 | +$367/kg |
| **SemiAnalysis 2026 repro** | 1,600 | 39.7 | $158M vs $46M | 4.33× | $10.73 / $2.48 | negative |

**Key finding:** the breakeven launch price in 2026 is *negative under both my central
inputs and SemiAnalysis's own* — a station amortised over **5 years at a 15% WACC**
cannot beat a building amortised over **15 years at 10.3%** at any rocket price.
The frontier is platform hardware, station life and cost of capital, not launch.

## Calibration vs SemiAnalysis (2026, B300 basis)

| Metric | SemiAnalysis (published) | This model (SA26 preset) |
|---|---|---|
| Space $/GPU-hr | 10.91 | 10.73 |
| Ground $/GPU-hr | 2.49 | 2.48 |
| Ratio | 4.38× | 4.33× |
| $/PFLOP-hr | 0.73 / 0.17 | 0.72 / 0.17 |
| $ per B tokens | 590 / 135 | 584 / 135 |

Structural mechanics adopted in v2: ~35 min/day eclipse even in dawn-dusk SSO with a
battery sized for full bus power ride-through; split WACCs (space 15% → 10.3%,
ground 10.3%); monthly-annuity levelization with **mixed lives** (5-yr station vs
5-yr IT + 15-yr building); reliability gross-ups (95% radiation availability ×
20% whole-chain redundancy in space, ~5% on the ground); Falcon 9 at its actual
$1,400–1,800/kg.

Earlier calibration anchors retained from v1: a 5 GW constellation needs ≈16.4 km²
of array (Starcloud's published concept is 16 km²), and the mature-era breakeven
launch price lands in the $110–370/kg band consistent with Google's "<$200/kg"
parity analysis.

## Repository contents

| Path | What it is |
|---|---|
| `odc_model.py` | The v2 system model — dataclass parameters, physics, finance, four scenario presets, SemiAnalysis anchors. Run directly for the scenario table. |
| `build_notebook.py` | Builds and **executes** the analysis notebook with nbformat/nbclient. |
| `notebooks/orbital_datacentre_viability.ipynb` | 27-cell executed notebook: invariant checks, headline table, SemiAnalysis calibration section, launch-price curves, mass/capex anatomy, sensitivity tornado, viability map, Monte Carlo, GW scale-up logistics, feasibility audit, conclusions. |
| `dashboard/orbital_datacentre_dashboard.html` | Interactive FT-styled cockpit (CDN fonts/Chart.js). ~28 sliders, preset chips, to-scale station plan view, breakeven curve, tornado, GW scale-up panel, power-scarcity NPV mode. |
| `dashboard/orbital_datacentre_dashboard_standalone.html` | Same dashboard with Chart.js + all fonts base64-inlined — **2 MB, zero external dependencies, works fully offline**. Share this one. |
| `archive/odc_model_v1.py` | Pre-SemiAnalysis v1 for reference. |

## Quickstart

```bash
pip install -r requirements.txt
python odc_model.py        # prints the four-scenario comparison + SA anchors
python build_notebook.py   # rebuilds & executes notebooks/orbital_datacentre_viability.ipynb
```

Open `dashboard/orbital_datacentre_dashboard_standalone.html` in any browser — no
internet required.

## Sources

SemiAnalysis, *To Boldly Go: The Case for Space Datacenters* + AI Space Datacenter
TCO Model (3 Jun 2026); Starcloud-1 flight data (H100 on orbit, Nov 2025) and
Starcloud whitepaper; Google Project Suncatcher radiation & cost analysis; ISS
power/thermal records; SpaceX S-1 (May 2026) and June 2026 launch pricing.
