"""Build the v3 interactive dashboard by exporting the odc package to JSON and baking it
into a single self-contained HTML file.

Design (per the v3 decision): Python is the single source of truth. This script dumps
presets (full `asdict` of each scenario, so they can never drift), the slider schema, and
all the new v3 data (size ladder, workloads, orbits, ground energy, provenance, SA
anchors) into one `MODEL` object injected into the page. The page's JavaScript keeps
ONLY the model arithmetic (a faithful port of odc.core, using the Python field names) and
self-checks itself against Python-computed `golden` results on load (parity badge).

Run from the repo root:  python build_dashboard.py
Output: dashboard/orbital_datacentre_dashboard.html
"""
import os
import json
from dataclasses import asdict

from odc.core import P, lcoc_and_npv, B300_KW_PER_GPU, B300_PFLOPS_FP4, B300_TOKS_PER_S
from odc.scenarios import TODAY, EARLY, MATURE, SA26, OPTIMIST, SKEPTIC, SA_ANCHORS
from odc import workloads, orbits, ground_energy, provenance
from odc.sizes import LADDER
from odc.hardware import gpus_for_power

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "dashboard", "orbital_datacentre_dashboard.html")

# --------------------------------------------------------------- presets + golden
PRESET_SCENARIOS = [TODAY, EARLY, MATURE, SA26, OPTIMIST, SKEPTIC]
presets = []
for sc in PRESET_SCENARIOS:
    d = asdict(sc)
    r = lcoc_and_npv(sc)
    d["_golden"] = dict(
        lcoc_s=r["lcoc_s"], lcoc_g=r["lcoc_g"], ratio=r["ratio"],
        M_dry=r["pt"]["M_dry"], gpu_hr_s=r["sa"]["gpu_hr_s"], gpu_hr_g=r["sa"]["gpu_hr_g"],
        breakeven=r["breakeven_launch"], npv_s=r["npv_s"], npv_g=r["npv_g"],
    )
    presets.append(d)

# --------------------------------------------------------------- slider schema
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

# --------------------------------------------------------------- size ladder
ladder = []
for s in LADDER:
    ladder.append(dict(
        key=s.key, name=s.name, it_mw=s.it_mw, gpu=s.gpu.name.split("(")[0].strip(),
        gpu_count=s.gpu_count(), anchor_mass_t=s.anchor_mass_t,
        anchor_kg_per_kw=s.anchor_kg_per_kw(), launch=s.launch_class,
        workloads=s.workloads, source=s.source,
    ))

# --------------------------------------------------------------- workloads
wl = []
for w in workloads.CATALOG.values():
    wl.append(dict(name=w.name, value_rank=w.value_rank, j_per_token=w.j_per_token,
                   latency=w.latency_class, gamma=w.gamma_gb_per_kwh,
                   space=w.space_suitable, revenue=w.revenue_in_orbit, note=w.note))

# --------------------------------------------------------------- orbits
orb = []
for o in orbits.CATALOG.values():
    orb.append(dict(name=o.name, alt_km=o.altitude_km, eclipse_min=o.eclipse_min_orbit,
                    eclipse_frac=o.eclipse_frac_daily, sun_frac=o.sun_frac_annual,
                    dose=o.tid_dose_mult, rtt=o.latency_rtt_ms, reboost=o.reboost_factor,
                    fcc=o.fcc_deorbit_ok, note=o.note))

# --------------------------------------------------------------- ground energy
ge = []
for s in ground_energy.CATALOG.values():
    ge.append(dict(name=s.name, usd_mwh=s.usd_per_mwh, ttp=s.time_to_power_months,
                   cf=s.capacity_factor, carbon=s.carbon_kg_per_mwh,
                   dispatch=s.dispatchable, note=s.note))

# --------------------------------------------------------------- provenance + citations
prov = []
for pr in provenance.REGISTRY.values():
    prov.append(dict(field=pr.field, central=pr.central, low=pr.low, high=pr.high,
                     unit=pr.unit, confidence=pr.confidence, verdict=pr.verdict,
                     source=pr.source, note=pr.note))
citations = [dict(work=k, cites=v) for k, v in
             sorted(provenance.CITATIONS.items(), key=lambda kv: -kv[1])]

MODEL = dict(
    constants=dict(SIGMA=5.670374419e-8, SOLAR=1361.0,
                   B300_KW_PER_GPU=B300_KW_PER_GPU, B300_PFLOPS_FP4=B300_PFLOPS_FP4,
                   B300_TOKS_PER_S=B300_TOKS_PER_S, GAMMA_NUM=14.8),
    presets=presets, groups=GROUPS, ladder=ladder, workloads=wl, orbits=orb,
    ground=ge, provenance=prov, citations=citations, saAnchors=SA_ANCHORS,
)

MODEL_JSON = json.dumps(MODEL, indent=None, separators=(",", ":"))

# --------------------------------------------------------------- HTML template
TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Orbital Data Centre - Viability Cockpit v3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --paper:#FFF1E5; --panel:#FCE9D8; --panel2:#F8E2CC; --ink:#26211C; --slate:#66605C;
  --teal:#0D7680; --claret:#990F3D; --oxford:#0F5499; --wheat:#9A6A28; --rule:#E0CDBA;
  --win:#0D7680; --lose:#990F3D;
}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%}
body{background:var(--paper);color:var(--ink);font-family:'IBM Plex Sans',-apple-system,Segoe UI,sans-serif;font-size:14px;line-height:1.45}
.wrap{max-width:1240px;margin:0 auto;padding:18px 20px 60px}
.serif{font-family:'Source Serif 4',Georgia,serif}
.mono{font-family:'IBM Plex Mono',ui-monospace,Menlo,monospace}
.dotrule{border:none;border-top:2px dotted var(--ink);opacity:.55;margin:10px 0 16px}
header .eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--slate)}
header h1{font-family:'Source Serif 4',Georgia,serif;font-weight:700;font-size:clamp(26px,4.4vw,40px);line-height:1.06;margin:6px 0 4px}
header .standfirst{color:var(--slate);max-width:820px;font-size:14.5px}
.chiprow{margin:14px 0 4px}
.chiprow .lab{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--slate);margin-bottom:5px}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{font-family:'IBM Plex Mono',monospace;font-size:12px;border:1.5px solid var(--ink);background:transparent;color:var(--ink);padding:6px 11px;cursor:pointer;border-radius:2px}
.chip:hover{background:var(--panel)}
.chip.active{background:var(--ink);color:var(--paper)}
.chip.opt.active{background:var(--teal);border-color:var(--teal)}
.chip.skp.active{background:var(--claret);border-color:var(--claret)}
.chip:focus-visible,input[type=range]:focus-visible{outline:2px solid var(--oxford);outline-offset:2px}
.parity{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:11px;padding:3px 8px;border:1px solid var(--teal);color:var(--teal);margin-left:8px}
.parity.bad{border-color:var(--claret);color:var(--claret)}
.verdict{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap;padding:14px 16px;border:2px solid var(--ink);background:var(--panel);margin:12px 0 14px}
.verdict .ratio{font-family:'Source Serif 4',Georgia,serif;font-weight:700;font-size:clamp(34px,5vw,52px);line-height:1}
.verdict .vtext{font-size:13.5px;max-width:660px;color:var(--ink)}
.verdict .vtag{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.1em;text-transform:uppercase;padding:3px 8px;color:var(--paper)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 18px}
.kpi{border-top:3px solid var(--ink);padding:8px 2px 2px}
.kpi .lab{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--slate)}
.kpi .val{font-family:'IBM Plex Mono',monospace;font-size:21px;font-weight:600;margin-top:2px}
.kpi .sub{font-size:11px;color:var(--slate)}
.kpi.teal{border-top-color:var(--teal)} .kpi.claret{border-top-color:var(--claret)} .kpi.oxford{border-top-color:var(--oxford)}
.grid{display:grid;grid-template-columns:330px 1fr;gap:26px}
@media(max-width:980px){.grid{grid-template-columns:1fr}}
.rail h2{font-family:'Source Serif 4',Georgia,serif;font-size:18px;margin-bottom:6px}
details{border-top:1px dotted var(--slate);padding:8px 0}
details:last-of-type{border-bottom:1px dotted var(--slate)}
summary{cursor:pointer;font-weight:600;font-size:13px;letter-spacing:.02em;list-style:none;display:flex;justify-content:space-between;align-items:center}
summary::after{content:"+";font-family:'IBM Plex Mono',monospace;color:var(--slate)}
details[open] summary::after{content:"-"}
.ctl{margin:10px 0 4px}
.ctl .row{display:flex;justify-content:space-between;align-items:baseline;font-size:12.5px}
.ctl .row b{font-weight:500}
.ctl .row .v{font-family:'IBM Plex Mono',monospace;font-weight:600}
input[type=range]{width:100%;accent-color:var(--teal);height:20px;background:transparent}
.note{font-size:11px;color:var(--slate);margin-top:2px}
.toggle{display:flex;align-items:center;gap:10px;margin:14px 0;padding:10px;border:1.5px dashed var(--wheat);background:var(--panel2)}
.toggle label{font-size:12.5px;cursor:pointer}
.toggle b{color:var(--claret)}
.card{border:1.5px solid var(--ink);background:#FFF7EE;padding:14px 14px 10px;margin-bottom:20px}
.card h3{font-family:'Source Serif 4',Georgia,serif;font-size:17px;margin-bottom:2px}
.card .cap{font-size:12px;color:var(--slate);margin-bottom:8px}
.chartbox{position:relative;height:280px}
.chartbox.tall{height:300px}.chartbox.short{height:210px}
.split{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:820px){.split{grid-template-columns:1fr}}
.bequote{font-family:'IBM Plex Mono',monospace;font-size:12.5px;margin-top:6px;color:var(--ink)}
.bequote b{color:var(--claret)}
#stationSvg{width:100%;height:auto;display:block;background:
 repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(38,33,28,.05) 40px),
 repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(38,33,28,.05) 40px),#FFF7EE}
.legendline{display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:var(--slate);margin-top:6px}
.sw{display:inline-block;width:10px;height:10px;margin-right:5px;vertical-align:-1px}
.scaleup{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;align-items:end}
.scaleup .big{font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:600}
.scaleup .lab{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--slate)}
.gwin{display:flex;align-items:baseline;gap:8px}
.gwin input{font-family:'IBM Plex Mono',monospace;font-size:20px;width:84px;border:none;border-bottom:2px solid var(--ink);background:transparent;color:var(--ink);padding:2px 4px}
table.dt{width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace}
table.dt th,table.dt td{text-align:left;padding:5px 7px;border-bottom:1px solid var(--rule)}
table.dt th{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--slate);font-weight:600}
table.dt tr.on{background:var(--panel)}
.vd-confirmed{color:var(--teal)} .vd-partial{color:var(--wheat)} .vd-refuted{color:var(--claret)}
.infogrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:8px}
.infogrid .b{font-family:'IBM Plex Mono',monospace;font-size:19px;font-weight:600}
.infogrid .l{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--slate)}
.pill{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:10.5px;padding:2px 7px;border-radius:10px;border:1px solid var(--slate);color:var(--slate)}
.pill.ok{border-color:var(--teal);color:var(--teal)} .pill.no{border-color:var(--claret);color:var(--claret)}
footer{margin-top:26px;font-size:11.5px;color:var(--slate)}
footer .src{border-top:1px solid var(--ink);padding-top:8px;margin-top:8px}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="eyebrow">Space Systems . Interactive Model v3 . June 2026</div>
  <h1>Can a data centre in orbit pay for itself?</h1>
  <p class="standfirst">A coupled power-thermal-mass-launch-economics model of an orbital AI data centre, benchmarked against identical silicon on the ground. <b>v3</b> is generated from the <span class="mono">odc/</span> Python package and backed by a 130-source, citation-checked literature review. Move the assumptions; the physics and the P&amp;L respond together. New: optimist/skeptic brackets, a workload communication-intensity (Gamma) gate, a spacecraft x GPU size ladder, orbit families, and a ground-energy comparator. <span id="parityBadge" class="parity">checking JS/Python parity...</span></p>
</header>
<hr class="dotrule">

<div class="chiprow"><div class="lab">Era preset (resets all sliders)</div>
  <div class="chips" id="presetChips"></div></div>
<div class="chiprow"><div class="lab">Orbit family (sets eclipse, dose, latency)</div>
  <div class="chips" id="orbitChips"></div></div>
<div class="chiprow"><div class="lab">Spacecraft x GPU size class (sets scale &amp; Gamma gate)</div>
  <div class="chips" id="sizeChips"></div></div>
<div class="chiprow"><div class="lab">Orbital workload (training + batch earn revenue in orbit)</div>
  <div class="chips" id="wlChips"></div></div>

<div class="verdict">
  <div class="ratio mono" id="vRatio">-</div>
  <div>
    <span class="vtag" id="vTag" style="background:var(--claret)">-</span>
    <p class="vtext" id="vText"></p>
  </div>
</div>

<div class="kpis">
  <div class="kpi"><div class="lab">Dry mass</div><div class="val" id="kMass">-</div><div class="sub" id="kMassSub">t per MW IT</div></div>
  <div class="kpi teal"><div class="lab">Capex . space</div><div class="val" id="kCapS">-</div><div class="sub" id="kCapSub">$M per MW IT</div></div>
  <div class="kpi claret"><div class="lab">Capex . ground</div><div class="val" id="kCapG">-</div><div class="sub">$M per MW IT</div></div>
  <div class="kpi teal"><div class="lab">LCOC . space</div><div class="val" id="kLcS">-</div><div class="sub">$ / IT-kW-hour</div></div>
  <div class="kpi claret"><div class="lab">LCOC . ground</div><div class="val" id="kLcG">-</div><div class="sub">$ / IT-kW-hour</div></div>
  <div class="kpi oxford"><div class="lab">Breakeven launch</div><div class="val" id="kBE">-</div><div class="sub" id="kBEsub">$ / kg to SSO</div></div>
  <div class="kpi"><div class="lab">$ / GPU-hr . B300</div><div class="val" id="kGpu" style="font-size:17px">-</div><div class="sub">space / ground, SA convention</div></div>
</div>
<p class="cap" style="margin:-8px 0 18px;font-size:11.5px;color:var(--slate)">Anchors - SemiAnalysis TCO model, Jun '26 (LCOC basis, verified): 2026 B300 = <b class="mono">$10.91 vs $2.49</b>/GPU-hr (4.4x); parity <b>~2040</b> base case, <b>early-2030s</b> in their terrestrial-constrained case. The optimist-skeptic band is ~14x wide: which literature you believe decides the answer.</p>

<div class="grid">
  <aside class="rail">
    <h2>Assumptions</h2>
    <p class="note" style="margin-bottom:8px">Grouped by subsystem. Presets reset every slider; touching any slider switches to <span class="mono">Custom</span>.</p>
    <div id="controls"></div>
    <div class="toggle">
      <input type="checkbox" id="scarcity">
      <label for="scarcity"><b>Power-scarcity mode</b> - ground waits <span id="gdel" class="mono">3.0</span> yr in the interconnect queue; orbit deploys in <span id="sdel" class="mono">1.0</span> yr. Applies to NPV only.</label>
    </div>
    <div class="kpi" style="border-top-color:var(--teal)"><div class="lab">NPV . space</div><div class="val" id="kNpvS">-</div><div class="sub">$M per MW over life</div></div>
    <div class="kpi" style="border-top-color:var(--claret);margin-top:8px"><div class="lab">NPV . ground</div><div class="val" id="kNpvG">-</div><div class="sub">$M per MW over life</div></div>
  </aside>

  <main>
    <div class="card">
      <h3>The machine, to scale</h3>
      <div class="cap">Plan view of one 1 MW node as your sliders define it - solar wings sized for end-of-life power, radiator cross sized by the T&#8308; law. A football pitch (105 x 68 m) for scale.</div>
      <svg id="stationSvg" viewBox="0 0 760 300" role="img" aria-label="To-scale plan view"></svg>
      <div class="legendline">
        <span><span class="sw" style="background:#0D7680"></span>Solar array <span class="mono" id="lArr"></span></span>
        <span><span class="sw" style="background:#990F3D"></span>Radiators <span class="mono" id="lRad"></span></span>
        <span><span class="sw" style="background:#26211C"></span>Server core</span>
        <span><span class="sw" style="background:none;border:1.5px dashed #66605C"></span>Football pitch 105 m</span>
      </div>
    </div>

    <div class="card">
      <h3>Selected orbit &amp; size class</h3>
      <div class="cap" id="orbitCap"></div>
      <div class="infogrid" id="orbitInfo"></div>
    </div>

    <div class="card">
      <h3>Where the curves cross</h3>
      <div class="cap">Space LCOC vs launch price (log scale), against the terrestrial benchmark. The dot is your current launch price.</div>
      <div class="chartbox tall"><canvas id="chCurve"></canvas></div>
      <div class="bequote" id="beNote"></div>
    </div>

    <div class="split">
      <div class="card"><h3>Capex anatomy</h3><div class="cap">$M per MW of sellable IT.</div>
        <div class="chartbox short"><canvas id="chCapex"></canvas></div></div>
      <div class="card"><h3>What you launch</h3><div class="cap">Dry mass per MW, by subsystem.</div>
        <div class="chartbox short"><canvas id="chMass"></canvas></div></div>
    </div>

    <div class="card">
      <h3>What moves the answer</h3>
      <div class="cap">One-at-a-time tornado on the space/ground LCOC ratio, around your current settings. Dashed line = parity.</div>
      <div class="chartbox"><canvas id="chTorn"></canvas></div>
    </div>

    <div class="card">
      <h3>Communication-intensity gate (Turyshev Gamma)</h3>
      <div class="cap">An orbital link can only move so much data per kWh of IT: ceiling = 14.8 / P[MW] GB/kWh. A workload closes in orbit only where its headroom (ceiling / demand) exceeds 1. The marker is your selected size class.</div>
      <div class="chartbox"><canvas id="chGamma"></canvas></div>
      <div class="bequote" id="gammaNote"></div>
    </div>

    <div class="card">
      <h3>Spacecraft x GPU size ladder</h3>
      <div class="cap">Six reference classes, ESPA edge node to GW constellation. Anchor mass is the literature figure; model mass is the calibrated engine run at that IT power under your current sliders. LCOC ratio is scale-invariant (linear core).</div>
      <div style="overflow-x:auto"><table class="dt" id="ladderTable"></table></div>
    </div>

    <div class="card">
      <h3>Terrestrial-first: the ground energy options space must beat</h3>
      <div class="cap">Energy is only ~7% of ground TCO - the real pain is time-to-power. Space competes in the bottom-left (fast, cheap), but gas/fuel-cells/off-grid solar already sit there. Bubble size = capacity factor; colour = carbon (green = clean).</div>
      <div class="chartbox tall"><canvas id="chGround"></canvas></div>
    </div>

    <div class="card">
      <h3>Scaling to a gigawatt campus</h3>
      <div class="cap">Logistics for a constellation at your current mass budget - Starship V3 ~ 150 t usable to SSO.</div>
      <div class="scaleup">
        <div class="gwin"><input id="gwTarget" class="mono" type="number" value="5" min="0.1" step="0.5" inputmode="decimal" aria-label="Constellation size in gigawatts"><span class="lab" style="font-size:16px">GW</span></div>
        <div><div class="big" id="suMass">-</div><div class="lab">tonnes to orbit</div></div>
        <div><div class="big" id="suFlights">-</div><div class="lab">Starship flights</div></div>
        <div><div class="big" id="suRepl">-</div><div class="lab">flights / yr replacement</div></div>
        <div><div class="big" id="suArea">-</div><div class="lab">km&#178; solar array</div></div>
      </div>
    </div>

    <div class="card">
      <h3>Provenance &amp; verification ledger</h3>
      <div class="cap">Every load-bearing default's source, literature band, and verdict from the 12-claim adversarial review. This is what separates the model from a spreadsheet of guesses - and flags where it is fragile.</div>
      <div style="overflow-x:auto"><table class="dt" id="provTable"></table></div>
      <div class="note" id="citeNote" style="margin-top:10px"></div>
    </div>

    <footer>
      <p><b>Reading guide.</b> LCOC = levelised cost of compute: monthly-annuity capital charges (station over its own life at the space WACC; ground IT over 5 yr and the building over its facility life at the ground WACC) plus opex, divided by IT-kW-hours sold. The $/GPU-hr card uses SemiAnalysis's wall-clock convention. "Platform-bound" means even free launch cannot reach parity. Counterintuitively, cheaper AI silicon worsens the space case: it shrinks the shared IT capex while the orbital premium stays fixed.</p>
      <p class="src">Generated from the <b>odc/</b> package (single source of truth; presets are a direct asdict dump, JS keeps only the arithmetic and self-checks against Python golden results). Literature: SemiAnalysis "To Boldly Go" (3 Jun 2026); Google Project Suncatcher (arXiv:2511.19468, 2025); Turyshev (arXiv:2604.27197, 2026); Denby &amp; Lucia Orbital Edge Computing (ASPLOS 2020, 292 cites); Forethought (2026); Epoch AI (2025); 130 sources, 12 numbers adversarially verified. Built for Ferhat, June 2026; v3.</p>
    </footer>
  </main>
</div>
</div>

<script>
"use strict";
const MODEL = __MODEL_JSON__;
const K = MODEL.constants, SIGMA = K.SIGMA, SOLAR = K.SOLAR, KWGPU = K.B300_KW_PER_GPU;

/* ============ MODEL (port of odc.core, Python field names) ============ */
function physMass(p, MW=1){
  const sell=1000*MW, gross=sell*(1+p.overprovision), bus=gross*(1+p.overhead_frac);
  const ecl=(p.eclipse_frac_daily!=null)?p.eclipse_frac_daily:p.eclipse_min_day/1440, sunlit=1-ecl;
  const E_batt=bus*(p.eclipse_min_day/60)/p.batt_dod;
  const M_batt=E_batt*1000/p.batt_Wh_kg, C_batt=E_batt*p.batt_cost_kWh/1e6;
  const daily=(sunlit+ecl/p.batt_rt_eff)/sunlit;
  const eol=Math.pow(1-p.degr_rate,p.life_yr);
  const arrBOL=bus*daily/(eol*p.pointing);
  const A_arr=arrBOL*1000/(SOLAR*p.cell_eff*p.packing);
  const M_arr=arrBOL*1000/p.sp_array, C_arr=arrBOL*1000*p.array_cost_W/1e6;
  const Q=bus;
  const q=p.emissivity*p.fin_eff*SIGMA*(Math.pow(p.T_rad,4)-Math.pow(p.T_sink,4));
  const A_rad=Q*1000/(2*q), M_rad=A_rad*p.rad_areal_kg_m2, C_rad=A_rad*p.rad_cost_m2/1e6;
  const M_it=gross*p.it_kg_per_kW, M_sh=p.shield_t_per_MW*1000*MW;
  const M_dry=(M_arr+M_rad+M_it+M_sh+M_batt)*(1+p.structure_frac);
  return {sell,gross,bus,sunlit,E_batt,M_batt,C_batt,arrBOL,A_arr,M_arr,C_arr,Q,q,A_rad,M_rad,C_rad,M_it,M_sh,M_dry};
}
function spaceCapex(p,pt,MW=1){
  const C_it=pt.gross*1000*p.it_cost_W/1e6;
  const C_pl=pt.C_arr+pt.C_rad+pt.C_batt+p.avionics_comms_M_MW*MW;
  const C_l=pt.M_dry*p.launch_kg/1e6, C_int=p.integration_M_MW*MW;
  const C_ins=p.insurance_frac*(C_it+C_pl+C_l);
  return {C_it,C_pl,C_l,C_int,C_ins,total:C_it+C_pl+C_l+C_int+C_ins};
}
function groundCapex(p,MW=1){
  const C_it=1000*MW*(1+p.g_overprovision)*1000*p.it_cost_W/1e6, C_fac=p.g_facility_M_MW*MW;
  return {C_it,C_fac,total:C_it+C_fac};
}
function crfm(r,n){if(r<=0)return 1/n;const m=r/12,k=12*n;return 12*m*Math.pow(1+m,k)/(Math.pow(1+m,k)-1);}
function evaluate(p,MW=1,withDelay=false){
  const pt=physMass(p,MW), cs=spaceCapex(p,pt,MW), cg=groundCapex(p,MW);
  const kwhS=1000*MW*8760*p.util*(1-p.compute_overhead)*p.rad_availability;
  const kwhG=1000*MW*8760*p.util*(1-p.g_compute_overhead)*p.g_rad_availability;
  const opS=p.ops_M_MW_yr*MW;
  const elec=p.g_pue*1000*MW*8760*p.util*p.g_elec_MWh/1e3/1e6;
  const opG=p.g_ops_M_MW_yr*MW+elec;
  const crfS=crfm(p.wacc_space,p.life_yr), crfGit=crfm(p.wacc_ground,p.it_life_yr), crfGfc=crfm(p.wacc_ground,p.g_facility_life_yr);
  const annS=crfS*cs.total+opS, annG=crfGit*cg.C_it+crfGfc*cg.C_fac+opG;
  const lcS=annS*1e6/kwhS, lcG=annG*1e6/kwhG;
  const npv=(side)=>{
    const r=side=="s"?p.wacc_space:p.wacc_ground;
    const yrs=Math.round(side=="s"?p.life_yr:p.it_life_yr);
    const d0=withDelay?(side=="s"?p.deploy_delay_yr:p.g_delay_yr):0;
    const cap=side=="s"?cs.total:cg.C_it;
    const op=side=="s"?opS:(opG+crfGfc*cg.C_fac);
    const kwh=side=="s"?kwhS:kwhG;
    let acc=0;
    for(let i=0;i<yrs;i++){const t=i+0.5+d0;
      acc+=(kwh*p.rev0_kWh*Math.pow(1-p.rev_decline,t)/1e6-op)/Math.pow(1+r,t);}
    return acc-cap;};
  const npvS=npv("s"), npvG=npv("g");
  const A=cs.C_it+cs.C_pl+cs.C_int+p.insurance_frac*(cs.C_it+cs.C_pl);
  const targetCap=(lcG*kwhS/1e6-opS)/crfS;
  const Lbe=(targetCap-A)*1e6/(pt.M_dry*(1+p.insurance_frac));
  const elecLife=elec*((1-Math.pow(1+p.wacc_ground,-p.life_yr))/p.wacc_ground);
  const gpuS=lcS*p.util*KWGPU, gpuG=lcG*p.util*KWGPU;
  return {pt,cs,cg,lcS,lcG,ratio:lcS/lcG,npvS,npvG,Lbe,kwhS,elecLife,gpuS,gpuG};
}
const gammaCeiling=(mw)=>K.GAMMA_NUM/Math.max(mw,1e-6);
const gammaHead=(wl,mw)=>gammaCeiling(mw)/Math.max(wl.gamma,1e-9);

/* ============ State ============ */
function loadPreset(i){const o={...MODEL.presets[i]}; delete o._golden; return o;}
let p = loadPreset(0);
let sizeMW = 1.0;          // selected size class IT power
let orbitIdx = 0;          // DDSS default
let wlIdx = 0;             // workload index into revenue workloads

/* ============ Parity self-check (JS vs Python golden) ============ */
(function parityCheck(){
  let worst=0, worstName="";
  MODEL.presets.forEach(ps=>{
    const o={...ps}; delete o._golden;
    const r=evaluate(o,1,false);
    const g=ps._golden;
    const err=Math.max(Math.abs(r.ratio-g.ratio)/g.ratio, Math.abs(r.lcS-g.lcoc_s)/g.lcoc_s,
                       Math.abs(r.pt.M_dry-g.M_dry)/g.M_dry);
    if(err>worst){worst=err; worstName=ps.name;}
  });
  const b=document.getElementById("parityBadge");
  if(worst<0.005){b.textContent=`JS/Python parity OK (max err ${(worst*100).toFixed(3)}% across ${MODEL.presets.length} presets)`;}
  else{b.classList.add("bad"); b.textContent=`PARITY DRIFT ${(worst*100).toFixed(2)}% at ${worstName}`;}
})();

/* ============ Build slider controls from schema ============ */
const ctlHost=document.getElementById("controls");
const sliders={};
MODEL.groups.forEach(([g,items],gi)=>{
  const d=document.createElement("details"); if(gi<2)d.open=true;
  d.innerHTML=`<summary>${g}</summary>`;
  items.forEach(arr=>{
    const [key,lab,unit,min,max,step,kind,note]=arr;
    const isLog=kind==="log";
    const div=document.createElement("div"); div.className="ctl";
    div.innerHTML=`<div class="row"><b>${lab}</b><span class="v" id="v_${key}"></span></div>
      <input type="range" id="s_${key}" min="${isLog?Math.log10(min):min}" max="${isLog?Math.log10(max):max}" step="${isLog?0.01:step}" aria-label="${lab} ${unit}">
      ${note?`<div class="note">${note}</div>`:`<div class="note">${unit}</div>`}`;
    d.appendChild(div);
    const el=div.querySelector("input");
    sliders[key]={el,isLog,kind,unit};
    el.addEventListener("input",()=>{
      p[key]=isLog?Math.pow(10,parseFloat(el.value)):parseFloat(el.value);
      setCustom(); refresh();
    });
  });
  ctlHost.appendChild(d);
});
function fmtVal(k,v){
  const s=sliders[k];
  if(s.kind==="pct")return (v*100).toFixed(1)+"%";
  if(k==="launch_kg"||k==="rad_cost_m2")return "$"+Math.round(v).toLocaleString();
  if(v>=100)return Math.round(v).toLocaleString();
  return (Math.round(v*100)/100).toString();
}
function syncSliders(){
  for(const k in sliders){const s=sliders[k];
    if(p[k]==null)continue;
    s.el.value=s.isLog?Math.log10(p[k]):p[k];
    document.getElementById("v_"+k).textContent=fmtVal(k,p[k]);}
  document.getElementById("gdel").textContent=p.g_delay_yr.toFixed(1);
  document.getElementById("sdel").textContent=p.deploy_delay_yr.toFixed(1);
}
function setCustom(){
  document.querySelectorAll("#presetChips .chip").forEach(c=>c.classList.remove("active"));
  const cc=document.getElementById("customChip");
  cc.classList.add("active");
}

/* ============ Chip rows ============ */
const presetHost=document.getElementById("presetChips");
MODEL.presets.forEach((ps,i)=>{
  const b=document.createElement("button"); b.className="chip"+(i===0?" active":"");
  if(ps.name.indexOf("Optimist")>=0)b.classList.add("opt");
  if(ps.name.indexOf("Skeptic")>=0)b.classList.add("skp");
  b.textContent=ps.name.replace(" (B300, repro)","").replace("-central","");
  b.addEventListener("click",()=>{
    p=loadPreset(i);
    document.querySelectorAll("#presetChips .chip").forEach(c=>c.classList.remove("active"));
    b.classList.add("active");
    // re-apply current orbit overlay on top of preset eclipse
    applyOrbit(orbitIdx,false);
    syncSliders(); refresh();
  });
  presetHost.appendChild(b);
});
const customChip=document.createElement("button");
customChip.className="chip"; customChip.id="customChip"; customChip.textContent="Custom";
presetHost.appendChild(customChip);

const orbitHost=document.getElementById("orbitChips");
MODEL.orbits.forEach((o,i)=>{
  const b=document.createElement("button"); b.className="chip"+(i===0?" active":"");
  b.textContent=o.name.split("(")[0].trim();
  b.addEventListener("click",()=>{applyOrbit(i,true);});
  orbitHost.appendChild(b);
});
function applyOrbit(i,doRefresh){
  orbitIdx=i; const o=MODEL.orbits[i];
  p.eclipse_min_day=o.eclipse_min; p.eclipse_frac_daily=o.eclipse_frac;
  document.querySelectorAll("#orbitChips .chip").forEach((c,j)=>c.classList.toggle("active",j===i));
  if(doRefresh){syncSliders(); refresh();}
}

const sizeHost=document.getElementById("sizeChips");
MODEL.ladder.forEach((s,i)=>{
  const b=document.createElement("button"); b.className="chip"+(Math.abs(s.it_mw-1.0)<1e-9?" active":"");
  const pw=s.it_mw<1?(s.it_mw*1000)+" kW":s.it_mw+" MW";
  b.textContent=`(${s.key}) ${pw}`;
  b.addEventListener("click",()=>{
    sizeMW=s.it_mw;
    document.querySelectorAll("#sizeChips .chip").forEach(c=>c.classList.remove("active"));
    b.classList.add("active"); refresh();
  });
  sizeHost.appendChild(b);
});

const wlHost=document.getElementById("wlChips");
const REVWL=MODEL.workloads.filter(w=>w.revenue);
REVWL.forEach((w,i)=>{
  const b=document.createElement("button"); b.className="chip"+(i===0?" active":"");
  b.textContent=w.name.split("/")[0].trim();
  b.addEventListener("click",()=>{
    wlIdx=i; document.querySelectorAll("#wlChips .chip").forEach(c=>c.classList.remove("active"));
    b.classList.add("active"); refresh();
  });
  wlHost.appendChild(b);
});

document.getElementById("scarcity").addEventListener("change",refresh);
document.getElementById("gwTarget").addEventListener("input",refresh);

/* ============ Charts ============ */
Chart.defaults.font.family="'IBM Plex Mono',monospace";
Chart.defaults.font.size=10.5; Chart.defaults.color="#66605C"; Chart.defaults.animation=false;
const gridCol="#E5DCD2";

const chCurve=new Chart(document.getElementById("chCurve"),{type:"line",
 data:{datasets:[
  {label:"Space LCOC",data:[],borderColor:"#0D7680",borderWidth:2.4,pointRadius:0,tension:.25},
  {label:"Ground benchmark",data:[],borderColor:"#66605C",borderDash:[5,4],borderWidth:1.6,pointRadius:0},
  {label:"Your launch price",data:[],borderColor:"#990F3D",backgroundColor:"#990F3D",pointRadius:5,showLine:false},
 ]},
 options:{maintainAspectRatio:false,
  scales:{x:{type:"logarithmic",min:50,max:6000,grid:{color:gridCol},title:{display:true,text:"launch price $/kg (log)"},
            ticks:{callback:v=>[50,100,200,500,1000,2000,4000].includes(v)?"$"+v:""}},
          y:{min:0,grid:{color:gridCol},title:{display:true,text:"$ per IT-kW-hour"}}},
  plugins:{legend:{labels:{boxWidth:14,boxHeight:2}},tooltip:{callbacks:{label:c=>` ${c.dataset.label}: $${c.parsed.y.toFixed(2)}/kWh @ $${Math.round(c.parsed.x)}/kg`}}}}});

const capexLabels=["IT hardware","Platform hw","Launch","Integration","Insurance","Facility","Electricity (life, disc.)"];
const capexCols=["#0F5499","#0D7680","#990F3D","#9A6A28","#66605C","#CDA46F","#E3B98F"];
const chCapex=new Chart(document.getElementById("chCapex"),{type:"bar",
 data:{labels:["Space","Ground"],datasets:capexLabels.map((l,i)=>({label:l,data:[0,0],backgroundColor:capexCols[i],stack:"s"}))},
 options:{maintainAspectRatio:false,indexAxis:"y",
  scales:{x:{stacked:true,grid:{color:gridCol},title:{display:true,text:"$M per MW IT"}},y:{stacked:true,grid:{display:false}}},
  plugins:{legend:{labels:{boxWidth:12,boxHeight:8,font:{size:9.5}}},tooltip:{callbacks:{label:c=>` ${c.dataset.label}: $${c.parsed.x.toFixed(0)}M`}}}}});

const massLabels=["Solar array","Radiators","IT hardware","Battery","Shielding","Structure/ADCS"];
const massCols=["#0D7680","#990F3D","#0F5499","#E3B98F","#9A6A28","#CDC4B9"];
const chMass=new Chart(document.getElementById("chMass"),{type:"bar",
 data:{labels:["t / MW"],datasets:massLabels.map((l,i)=>({label:l,data:[0],backgroundColor:massCols[i],stack:"m"}))},
 options:{maintainAspectRatio:false,indexAxis:"y",
  scales:{x:{stacked:true,grid:{color:gridCol},title:{display:true,text:"tonnes per MW IT"}},y:{stacked:true,grid:{display:false}}},
  plugins:{legend:{labels:{boxWidth:12,boxHeight:8,font:{size:9.5}}},tooltip:{callbacks:{label:c=>` ${c.dataset.label}: ${c.parsed.x.toFixed(1)} t`}}}}});

const chTorn=new Chart(document.getElementById("chTorn"),{type:"bar",
 data:{labels:[],datasets:[{label:"ratio range",data:[],backgroundColor:"#990F3D",borderSkipped:false}]},
 options:{maintainAspectRatio:false,indexAxis:"y",
  scales:{x:{grid:{color:gridCol},title:{display:true,text:"space / ground LCOC ratio"}},y:{grid:{display:false},ticks:{font:{size:10},autoSkip:false}}},
  plugins:{legend:{display:false}}},
 plugins:[{id:"tornadoLines",afterDraw(ch){
   const {ctx,chartArea,scales}=ch; if(ch.$base==null)return;
   const draw=(x,col,dash,lab)=>{const px=scales.x.getPixelForValue(x);
     if(px<chartArea.left||px>chartArea.right)return;
     ctx.save();ctx.strokeStyle=col;ctx.setLineDash(dash);ctx.lineWidth=1.3;
     ctx.beginPath();ctx.moveTo(px,chartArea.top);ctx.lineTo(px,chartArea.bottom);ctx.stroke();
     ctx.fillStyle=col;ctx.font="10px 'IBM Plex Mono'";ctx.fillText(lab,px+3,chartArea.top+10);ctx.restore();};
   draw(1.0,"#0D7680",[4,3],"parity"); draw(ch.$base,"#26211C",[],"base");
 }}]});

const chGamma=new Chart(document.getElementById("chGamma"),{type:"line",
 data:{datasets:[
  {label:"Frontier training",data:[],borderColor:"#0D7680",borderWidth:2.4,pointRadius:0},
  {label:"Batch inference",data:[],borderColor:"#0F5499",borderWidth:2.4,pointRadius:0},
  {label:"Latency chat (ground-only)",data:[],borderColor:"#990F3D",borderDash:[5,4],borderWidth:1.6,pointRadius:0},
  {label:"Your size",data:[],borderColor:"#26211C",backgroundColor:"#26211C",pointRadius:5,showLine:false},
 ]},
 options:{maintainAspectRatio:false,
  scales:{x:{type:"logarithmic",min:0.01,max:1000,grid:{color:gridCol},title:{display:true,text:"platform IT power (MW, log)"},
            ticks:{callback:v=>[0.01,0.1,1,10,100,1000].includes(v)?v:""}},
          y:{type:"logarithmic",grid:{color:gridCol},title:{display:true,text:"Gamma headroom (>1 closes)"}}},
  plugins:{legend:{labels:{boxWidth:14,boxHeight:2}}}},
 plugins:[{id:"gammaLine",afterDraw(ch){
   const {ctx,chartArea,scales}=ch; const py=scales.y.getPixelForValue(1.0);
   if(py<chartArea.top||py>chartArea.bottom)return;
   ctx.save();ctx.strokeStyle="#26211C";ctx.setLineDash([4,3]);ctx.lineWidth=1.2;
   ctx.beginPath();ctx.moveTo(chartArea.left,py);ctx.lineTo(chartArea.right,py);ctx.stroke();
   ctx.fillStyle="#26211C";ctx.font="10px 'IBM Plex Mono'";ctx.fillText("closes above here",chartArea.left+4,py-4);ctx.restore();
 }}]});

const chGround=new Chart(document.getElementById("chGround"),{type:"bubble",
 data:{datasets:[]},
 options:{maintainAspectRatio:false,
  scales:{x:{grid:{color:gridCol},title:{display:true,text:"time-to-power (months)"},min:0},
          y:{grid:{color:gridCol},title:{display:true,text:"effective $/MWh"},min:0}},
  plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>` ${c.raw.name}: $${c.raw.y}/MWh, ${c.raw.x} mo, CF ${(c.raw.cf*100).toFixed(0)}%`}}}}});
(function fillGround(){
  chGround.data.datasets=MODEL.ground.map(s=>({
    label:s.name,
    data:[{x:s.ttp,y:s.usd_mwh,r:6+s.cf*16,name:s.name,cf:s.cf}],
    backgroundColor:s.carbon<50?"rgba(13,118,128,.65)":(s.carbon<380?"rgba(154,106,40,.6)":"rgba(153,15,61,.6)"),
    borderColor:"#26211C",borderWidth:.6}));
  chGround.update();
})();

/* ============ Station SVG ============ */
function drawStation(pt){
  const svg=document.getElementById("stationSvg");
  const wingA=pt.A_arr/2, wingW=Math.sqrt(wingA/4), wingL=4*wingW;
  const radA=pt.A_rad/4, radW=Math.sqrt(radA/3), radL=3*radW;
  const span=2*wingL+14, vspan=Math.max(2*radL+14,68,wingW*2);
  const scale=Math.min(720/Math.max(span,110),270/Math.max(vspan,80));
  const cx=380, cy=150, S=v=>v*scale;
  const pitchW=S(105), pitchH=S(68); const e=[];
  e.push(`<rect x="${cx-pitchW/2}" y="${cy-pitchH/2}" width="${pitchW}" height="${pitchH}" fill="none" stroke="#66605C" stroke-width="1.4" stroke-dasharray="5 4"/>`);
  for(const s of[-1,1]){const y0=s<0?cy-7-S(radL):cy+7;
    for(const o of[-1,1]){const x0=cx+(o<0?-4-S(radW):4);
      e.push(`<rect x="${x0}" y="${y0}" width="${S(radW)}" height="${S(radL)}" fill="#990F3D" opacity=".82" stroke="#26211C" stroke-width="1"/>`);}}
  for(const s of[-1,1]){const x0=s<0?cx-7-S(wingL):cx+7, y0=cy-S(wingW)/2;
    e.push(`<rect x="${x0}" y="${y0}" width="${S(wingL)}" height="${S(wingW)}" fill="#0D7680" stroke="#26211C" stroke-width="1.2"/>`);
    const n=Math.max(4,Math.round(wingL/12));
    for(let k=1;k<n;k++)e.push(`<line x1="${x0+S(wingL)*k/n}" y1="${y0}" x2="${x0+S(wingL)*k/n}" y2="${y0+S(wingW)}" stroke="#FFF1E5" stroke-width=".7" opacity=".55"/>`);}
  e.push(`<rect x="${cx-6}" y="${cy-6}" width="12" height="12" fill="#26211C"/>`);
  const sx=cx-S(span)/2, ex=cx+S(span)/2, dy=cy+Math.max(pitchH/2,S(radL)+12)+18;
  e.push(`<line x1="${sx}" y1="${dy}" x2="${ex}" y2="${dy}" stroke="#26211C" stroke-width="1"/>`);
  e.push(`<text x="${cx}" y="${dy+14}" text-anchor="middle" font-family="IBM Plex Mono" font-size="11" fill="#26211C">tip-to-tip span ~ ${Math.round(span)} m</text>`);
  svg.innerHTML=e.join("");
  document.getElementById("lArr").textContent=` ${Math.round(pt.A_arr).toLocaleString()} m²`;
  document.getElementById("lRad").textContent=` ${Math.round(pt.A_rad).toLocaleString()} m² (2-sided)`;
}

/* ============ Static tables ============ */
(function buildProvTable(){
  const t=document.getElementById("provTable");
  let h="<tr><th>Parameter</th><th>Central</th><th>Skeptic</th><th>Optimist</th><th>Conf</th><th>Verdict</th></tr>";
  MODEL.provenance.forEach(p=>{
    h+=`<tr><td>${p.field}</td><td>${p.central}</td><td>${p.low}</td><td>${p.high}</td><td>${p.confidence}</td><td class="vd-${p.verdict}">${p.verdict}</td></tr>`;
  });
  t.innerHTML=h;
  document.getElementById("citeNote").innerHTML="Academic citation anchors (Semantic Scholar, 12 Jun 2026): "+
    MODEL.citations.slice(0,6).map(c=>`<b>${c.cites}</b> ${c.work.split("(")[0].trim()}`).join(" . ");
})();

/* ============ Refresh ============ */
const $=id=>document.getElementById(id);
const f1=v=>v.toFixed(1), f2=v=>v.toFixed(2);
function refresh(){
  for(const k in sliders)if(p[k]!=null)$("v_"+k).textContent=fmtVal(k,p[k]);
  const withDelay=$("scarcity").checked;
  const r=evaluate(p,1,withDelay);                 // per-MW basis for the established panels
  const rs=evaluate(p,sizeMW,withDelay);           // selected size class (absolute)

  $("kMass").textContent=f1(r.pt.M_dry/1e3);
  $("kCapS").textContent="$"+Math.round(r.cs.total)+"M";
  $("kCapG").textContent="$"+Math.round(r.cg.total)+"M";
  $("kLcS").textContent="$"+f2(r.lcS); $("kLcG").textContent="$"+f2(r.lcG);
  $("kGpu").textContent="$"+f2(r.gpuS)+" / $"+f2(r.gpuG);
  $("kNpvS").textContent=(r.npvS>=0?"+":"−")+"$"+Math.abs(r.npvS).toFixed(0)+"M";
  $("kNpvS").style.color=r.npvS>=0?"var(--teal)":"var(--claret)";
  $("kNpvG").textContent=(r.npvG>=0?"+":"−")+"$"+Math.abs(r.npvG).toFixed(0)+"M";
  $("kNpvG").style.color=r.npvG>=0?"var(--teal)":"var(--claret)";
  if(r.Lbe>0){$("kBE").textContent="$"+Math.round(r.Lbe).toLocaleString();$("kBEsub").textContent="$ / kg to SSO";}
  else{$("kBE").textContent="-";$("kBEsub").textContent="platform-bound: free launch can't reach parity";}

  $("vRatio").textContent=f2(r.ratio)+"×";
  const tag=$("vTag"), vt=$("vText");
  if(r.ratio<0.95){tag.textContent="SPACE WINS";tag.style.background="var(--teal)";
    vt.textContent="At these assumptions orbital compute undercuts the terrestrial benchmark per kW-hour.";}
  else if(r.ratio<=1.12){tag.textContent="AT PARITY";tag.style.background="var(--wheat)";
    vt.textContent="Space and ground are within ~10% per kW-hour. Site-specific power prices, queues and sovereignty now decide.";}
  else{tag.textContent="GROUND WINS";tag.style.background="var(--claret)";
    vt.textContent=r.Lbe>0?
     `Orbit costs ${f2(r.ratio)}× ground. Launch below ~$${Math.round(r.Lbe).toLocaleString()}/kg would close the gap.`:
     `Orbit costs ${f2(r.ratio)}× ground - and no launch price fixes it: the space-rated platform alone exceeds a ground build plus its electricity.`;}
  $("vRatio").style.color=r.ratio<0.95?"var(--teal)":(r.ratio<=1.12?"var(--wheat)":"var(--claret)");

  // orbit + size info panel
  const o=MODEL.orbits[orbitIdx];
  const sizeRow=MODEL.ladder.find(s=>Math.abs(s.it_mw-sizeMW)<1e-9)||MODEL.ladder[3];
  $("orbitCap").innerHTML=`<b>${o.name}</b> - ${o.note}`;
  const wlSel=REVWL[wlIdx], closes=gammaHead(wlSel,sizeMW)>=1;
  $("orbitInfo").innerHTML=
    `<div><div class="b">${(o.sun_frac*100).toFixed(0)}%</div><div class="l">annual sun</div></div>`+
    `<div><div class="b">${o.dose}×</div><div class="l">TID dose vs DDSS</div></div>`+
    `<div><div class="b">${o.rtt} ms</div><div class="l">round-trip latency</div></div>`+
    `<div><div class="b"><span class="pill ${o.fcc?'ok':'no'}">${o.fcc?'FCC ok':'>5yr decay'}</span></div><div class="l">deorbit rule</div></div>`+
    `<div><div class="b">${sizeRow.gpu_count.toLocaleString(undefined,{maximumFractionDigits:0})}</div><div class="l">${sizeRow.gpu} GPUs</div></div>`+
    `<div><div class="b">${(rs.pt.M_dry/1e3).toLocaleString(undefined,{maximumFractionDigits:1})} t</div><div class="l">total dry mass</div></div>`+
    `<div><div class="b"><span class="pill ${closes?'ok':'no'}">${closes?'closes':'link-bound'}</span></div><div class="l">${wlSel.name.split('/')[0].trim()} @ this size</div></div>`;

  // curve
  const pts=[],N=70;
  for(let i=0;i<=N;i++){const L=Math.pow(10,Math.log10(50)+(Math.log10(6000)-Math.log10(50))*i/N);
    pts.push({x:L,y:evaluate({...p,launch_kg:L},1,false).lcS});}
  chCurve.data.datasets[0].data=pts;
  chCurve.data.datasets[1].data=[{x:50,y:r.lcG},{x:6000,y:r.lcG}];
  chCurve.data.datasets[2].data=[{x:p.launch_kg,y:r.lcS}];
  chCurve.options.scales.y.max=Math.max(3,Math.ceil(Math.max(pts[N].y,r.lcG)*1.15));
  chCurve.update();
  $("beNote").innerHTML=r.Lbe>0
    ? `Breakeven where teal meets dashed: <b>$${Math.round(r.Lbe).toLocaleString()}/kg</b>. Falcon 9 actual ~ $1,400-1,800; Starship target ~$250.`
    : `<b>No crossing.</b> Even at $50/kg the teal curve sits above the benchmark - platform hardware is the binding constraint.`;

  // capex + mass
  const sp=[r.cs.C_it,r.cs.C_pl,r.cs.C_l,r.cs.C_int,r.cs.C_ins,0,0];
  const gd=[r.cg.C_it,0,0,0,0,r.cg.C_fac,r.elecLife];
  chCapex.data.datasets.forEach((d,i)=>{d.data=[sp[i],gd[i]];}); chCapex.update();
  const mvals=[r.pt.M_arr,r.pt.M_rad,r.pt.M_it,r.pt.M_batt,r.pt.M_sh,
    r.pt.M_dry-(r.pt.M_arr+r.pt.M_rad+r.pt.M_it+r.pt.M_batt+r.pt.M_sh)].map(v=>v/1e3);
  chMass.data.datasets.forEach((d,i)=>{d.data=[mvals[i]];}); chMass.update();

  // tornado
  const sw=[
   ["Launch $/kg","launch_kg",250,1600],["WACC space","wacc_space",0.103,0.20],
   ["Array $/W","array_cost_W",4,35],["IT $/W","it_cost_W",24,36],
   ["Ground $/MWh","g_elec_MWh",60,160],["IT kg/kW","it_kg_per_kW",8,40],
   ["Array W/kg","sp_array",75,400],["Radiator kg/m2","rad_areal_kg_m2",2.5,14],
   ["T_rad K","T_rad",305,333],["Station life yr","life_yr",5,10],
   ["Radiation avail","rad_availability",0.92,0.98]];
  const trows=sw.map(([lab,k,lo,hi])=>{
    const a=evaluate({...p,[k]:lo},1,false).ratio, b=evaluate({...p,[k]:hi},1,false).ratio;
    return [lab,Math.min(a,b),Math.max(a,b)];}).sort((x,y)=>(x[2]-x[1])-(y[2]-y[1]));
  chTorn.data.labels=trows.map(r=>r[0]);
  chTorn.data.datasets[0].data=trows.map(r=>[r[1],r[2]]);
  chTorn.data.datasets[0].backgroundColor=trows.map(r=>(r[2]-r[1])<0.3?"#0D7680":"#990F3D");
  chTorn.$base=r.ratio; chTorn.update();

  // gamma gate
  const gp=[],N2=60;
  const tr=[],ba=[],la=[];
  const WT=MODEL.workloads.find(w=>w.name.indexOf("training")>=0||w.name.indexOf("Frontier")>=0);
  const WB=MODEL.workloads.find(w=>w.name.indexOf("Batch")>=0);
  const WL=MODEL.workloads.find(w=>w.name.indexOf("Latency")>=0||w.name.indexOf("chat")>=0);
  for(let i=0;i<=N2;i++){const mw=Math.pow(10,-2+5*i/N2);
    tr.push({x:mw,y:gammaHead(WT,mw)}); ba.push({x:mw,y:gammaHead(WB,mw)}); la.push({x:mw,y:gammaHead(WL,mw)});}
  chGamma.data.datasets[0].data=tr; chGamma.data.datasets[1].data=ba; chGamma.data.datasets[2].data=la;
  chGamma.data.datasets[3].data=[{x:sizeMW,y:gammaHead(REVWL[wlIdx],sizeMW)}];
  chGamma.update();
  $("gammaNote").innerHTML=`At your ${sizeMW<1?(sizeMW*1000)+' kW':sizeMW+' MW'} size, `+
    `<b>${REVWL[wlIdx].name.split('/')[0].trim()}</b> headroom is <b>${gammaHead(REVWL[wlIdx],sizeMW).toFixed(1)}×</b> `+
    `(${gammaHead(REVWL[wlIdx],sizeMW)>=1?"closes in orbit":"link-bound - cannot close"}). Training scales furthest; batch tops out around a few MW.`;

  // size ladder table
  const lt=$("ladderTable");
  let h="<tr><th>Class</th><th>IT</th><th>GPU</th><th>#GPU</th><th>Anchor t</th><th>Model t</th><th>Train closes?</th></tr>";
  MODEL.ladder.forEach(s=>{
    const md=physMass(p,s.it_mw).M_dry/1e3;
    const cl=gammaHead(WT,s.it_mw)>=1;
    const on=Math.abs(s.it_mw-sizeMW)<1e-9?" class='on'":"";
    const pw=s.it_mw<1?(s.it_mw*1000)+" kW":s.it_mw+" MW";
    h+=`<tr${on}><td>(${s.key}) ${s.name}</td><td>${pw}</td><td>${s.gpu}</td>`+
       `<td>${s.gpu_count.toLocaleString(undefined,{maximumFractionDigits:0})}</td>`+
       `<td>${s.anchor_mass_t}</td><td>${md.toLocaleString(undefined,{maximumFractionDigits:1})}</td>`+
       `<td><span class="pill ${cl?'ok':'no'}">${cl?'yes':'link-bound'}</span></td></tr>`;
  });
  lt.innerHTML=h;

  // scale-up (uses gw input)
  const gw=Math.max(0.1,parseFloat($("gwTarget").value)||5);
  const mass=physMass(p,gw*1000).M_dry/1e3;
  $("suMass").textContent=Math.round(mass).toLocaleString();
  $("suFlights").textContent=Math.round(mass/150).toLocaleString();
  $("suRepl").textContent=Math.round(mass/150/p.life_yr).toLocaleString();
  $("suArea").textContent=(physMass(p,gw*1000).A_arr/1e6).toFixed(2);

  drawStation(r.pt);
}

applyOrbit(0,false);
syncSliders();
refresh();
</script>
</body>
</html>'''

html = TEMPLATE.replace("__MODEL_JSON__", MODEL_JSON)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Dashboard written -> {OUT}  ({len(html):,} bytes, {len(MODEL['presets'])} presets, "
      f"{len(MODEL['ladder'])} size classes, {len(MODEL['orbits'])} orbits)")

# --------------------------------------------------------------- standalone (offline) variant
# Inline Chart.js and drop the Google Fonts <link> so the file works with zero network.
# Fonts gracefully fall back to the system mono/serif already named in the CSS.
OUT_STANDALONE = os.path.join(ROOT, "dashboard", "orbital_datacentre_dashboard_standalone.html")
CHARTJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
try:
    import urllib.request
    with urllib.request.urlopen(CHARTJS_URL, timeout=30) as resp:
        chartjs = resp.read().decode("utf-8")
    standalone = html.replace(
        '<script src="' + CHARTJS_URL + '"></script>',
        "<script>" + chartjs + "</script>")
    # remove the Google Fonts preconnect + stylesheet links (3 lines collapsed by markers)
    for frag in [
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n',
        '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">\n',
    ]:
        standalone = standalone.replace(frag, "")
    with open(OUT_STANDALONE, "w", encoding="utf-8") as f:
        f.write(standalone)
    print(f"Standalone written -> {OUT_STANDALONE}  ({len(standalone):,} bytes, Chart.js inlined, "
          f"fonts fall back to system; fully offline)")
except Exception as e:
    print(f"Standalone NOT regenerated (could not fetch Chart.js: {e}). The CDN dashboard above is current.")

