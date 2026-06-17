"""
odc.sizes — spacecraft x GPU size ladder
========================================
Six realistic reference classes from a single-GPU ESPA edge node to a GW constellation,
each anchored to a real concept or flight. The user asked specifically for "realistic
examples for different spacecraft sizes and gpu sizes from small to large and equivalent
power draw" — this module is that deliverable.

Each SizeClass carries literature-anchored figures (power, mass, launch, GPU, workloads,
source) AND can run the calibrated core model at its IT power under a chosen scenario, so
the table shows both the published anchor and the model's own mass/capex/LCOC.

Sources: Starcloud-1 (flown 2 Nov 2025); Google Suncatcher arXiv:2511.19468 (~28 kW ref
sat, 81-sat cluster); SpaceX "AI1" (Jun 2026, 150 kW/sat, 1 GB300 rack); Gaalema
arXiv:2604.07760 (16 MW, 150 t); ASCEND (2024, ~770 kW block, 1 GW by 2050); Starcloud
whitepaper (5 GW, ~16 km^2 array).
"""
from dataclasses import dataclass
from .core import power_thermal_mass, space_capex, lcoc_and_npv
from .hardware import GPU, H100, B200, B300, GB300_NVL72, JETSON_ORIN, gpus_for_power


@dataclass(frozen=True)
class SizeClass:
    key: str
    name: str
    it_mw: float               # sellable IT power (MW)
    gpu: GPU                   # representative accelerator
    anchor_mass_t: float       # literature-anchored launched mass (tonnes)
    launch_class: str          # launch vehicle / accommodation
    workloads: str             # what fits at this rung
    source: str                # anchoring source

    def gpu_count(self) -> float:
        return gpus_for_power(self.gpu, self.it_mw * 1000.0)

    def anchor_kg_per_kw(self) -> float:
        return self.anchor_mass_t * 1000.0 / (self.it_mw * 1000.0)

    def model_example(self, scenario) -> dict:
        """Run the calibrated core model at this rung's IT power under `scenario`.
        NB: for sub-MW rungs the deployable-radiator/array architecture is an
        extrapolation — small sats use body-mounted panels — so divergence between
        anchor_mass and model mass at rungs (a)/(b) is expected and informative."""
        r = lcoc_and_npv(scenario, P_it_MW=self.it_mw)
        pt = r["pt"]
        return dict(
            model_dry_t=pt["M_dry"] / 1e3,
            model_kg_per_kw=pt["M_dry"] / (self.it_mw * 1000.0),
            array_m2=pt["A_array"], radiator_m2=pt["A_rad"],
            capex_s_M=r["cap_s"]["total"], lcoc_s=r["lcoc_s"], lcoc_g=r["lcoc_g"],
            ratio=r["ratio"], gpu_count=self.gpu_count(),
        )


# (a) ESPA edge node — single GPU, body-mounted, EO-triage / demo class.
ESPA_EDGE = SizeClass("a", "ESPA edge node", 0.001, H100, 0.060, "Falcon 9 rideshare / Transporter",
    "EO triage, single-shot inference, on-orbit demos (first LLM trained in space, NanoGPT)",
    "Starcloud-1 (flown 2 Nov 2025, ~325 km, ~11-mo life)")

# (b) Single sat / small cluster — tens of kW, loosely-coupled jobs.
SINGLE_SAT = SizeClass("b", "Single sat / small cluster", 0.028, GB300_NVL72, 0.575,
    "Falcon 9 / Starship rideshare",
    "Batch inference, EO inference, loosely-coupled jobs",
    "Suncatcher ~28 kW ref sat (2025); Starcloud-2 (~7 kW, B200+H100)")

# (c) Large sat — one frontier rack.
LARGE_SAT = SizeClass("c", "Large sat (1 rack)", 0.150, GB300_NVL72, 2.0, "Starship",
    "One GB300 NVL72 rack; batch / training within a single tensor-parallel domain",
    "SpaceX 'AI1' (Jun 2026, 150 kW peak, ~110 m^2 radiator)")

# (d) 1 MW node — small cluster, the v2 basis.
NODE_1MW = SizeClass("d", "1 MW node", 1.0, GB300_NVL72, 20.0, "Starship",
    "Pre-positioned-data training, large batch inference",
    "Interpolated (Starcloud per-MW budget ~19.5 kg/kW)")

# (e) Assembled / formation station — tens of MW.
STATION_16MW = SizeClass("e", "Assembled station", 16.0, B300, 150.0, "Single Starship bay (Gaalema)",
    "Distributed training (pre-positioned data), large-scale batch inference",
    "Gaalema arXiv:2604.07760 (16 MW, 150 t); Starcloud 40 MW module")

# (f) GW constellation — the end-state.
CONSTELLATION_1GW = SizeClass("f", "GW constellation", 1000.0, GB300_NVL72, 10000.0,
    "~130-280 Starship launches/GW (30-50 AI1 sats/flight)",
    "Frontier training at scale + bulk inference (subject to Gamma/bisection limits)",
    "ASCEND (1 GW by 2050); Starcloud 5 GW on ~16 km^2 array; Suncatcher (scalable)")

LADDER = [ESPA_EDGE, SINGLE_SAT, LARGE_SAT, NODE_1MW, STATION_16MW, CONSTELLATION_1GW]
CATALOG = {s.key: s for s in LADDER}


def ladder_table(scenario):
    """Worked-example rows for every rung under `scenario`."""
    rows = []
    for s in LADDER:
        ex = s.model_example(scenario)
        rows.append(dict(
            key=s.key, name=s.name, it_mw=s.it_mw, gpu=s.gpu.name,
            gpu_count=ex["gpu_count"], anchor_mass_t=s.anchor_mass_t,
            anchor_kg_per_kw=s.anchor_kg_per_kw(),
            model_dry_t=ex["model_dry_t"], model_kg_per_kw=ex["model_kg_per_kw"],
            launch=s.launch_class, workloads=s.workloads, source=s.source,
        ))
    return rows


if __name__ == "__main__":
    from .scenarios import MATURE
    print("SIZE LADDER (model run under MATURE scenario)\n")
    print(f"{'':2} {'class':28} {'IT':>9} {'GPU':>16} {'#GPU':>9} {'anchor t':>9} {'anchor kg/kW':>12} {'model t':>9}")
    for r in ladder_table(MATURE):
        it = f"{r['it_mw']*1000:.0f} kW" if r['it_mw'] < 1 else f"{r['it_mw']:.0f} MW"
        gpu_short = r['gpu'].split('(')[0].strip()
        print(f"{r['key']:2} {r['name']:28} {it:>9} {gpu_short:>16} {r['gpu_count']:9.0f} "
              f"{r['anchor_mass_t']:9.1f} {r['anchor_kg_per_kw']:12.1f} {r['model_dry_t']:9.1f}")
