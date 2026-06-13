"""
odc.provenance — source + citation strength for every default parameter
=======================================================================
Each load-bearing default carries where it came from, a plausible range, a confidence
level, and the adversarial-verification verdict (from the 2026 literature review). This
serves two purposes: (1) auditability — a reader can see why a number is what it is and
how contested it is; (2) it drives Monte Carlo — instead of a flat +/-30% on every
parameter, the sampler can use the provenance ranges and weight by confidence.

Verdicts: confirmed | partial | refuted | unverifiable (from the 12-claim adversarial
ledger). "partial" means the headline survives but with a correction; see note.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    field: str                 # core.P field name (or conceptual key)
    central: float             # SemiAnalysis-central default value
    low: float                 # skeptic end
    high: float                # optimist end
    unit: str
    confidence: str            # high | medium | low
    verdict: str               # adversarial-verification verdict
    source: str
    note: str


REGISTRY = {
    "launch_kg": Param("launch_kg", 1600, 250, 100, "$/kg", "high", "partial",
        "SemiAnalysis 2026 ($1.4-1.8k today); Suncatcher (<=$200/kg by ~2035); Forethought ($100 parity)",
        "$3,245/kg is naive list/expendable, not a quote; SpaceX marginal ~$629/kg; $250 Mature is Starship-reuse-contingent."),
    "it_kg_per_kW": Param("it_kg_per_kW", 14, 59, 9, "kg/kW", "medium", "confirmed",
        "SpaceX AI1 (14.3 peak); Starcloud (19.5); Turyshev (34-59); Gaalema (<10)",
        "System-level launched mass/kW, NOT IT-payload-only. 3-6x spread is the biggest mass uncertainty."),
    "rad_areal_kg_m2": Param("rad_areal_kg_m2", 6.0, 14.0, 2.5, "kg/m^2", "medium", "partial",
        "ISS ~14 kg/m^2 today; NASA NEP ~6.1; mature target ~2.5",
        "Drop the '3.5 kg/m^2 MARVL' figure — MARVL is 3.8 kg/kWe specific mass, not areal density."),
    "radiator_W_m2": Param("radiator_W_m2", 350, 166, 633, "W/m^2", "low", "partial",
        "ISS floor 150-200 @20-27C; Starcloud ceiling 633 @20C",
        "T^4-driven; single biggest model sensitivity. 633 is NOT flight-demonstrated."),
    "sp_array": Param("sp_array", 110, 75, 400, "W/kg", "medium", "refuted",
        "iROSA flight ~75 W/kg; ROSA-class spec ~110; Mega-ROSA target 200-400",
        "REFUTES the 225 W/kg / 4.4 kg/kW claim — iROSA is ~3x heavier. TODAY at 110 is slightly optimistic."),
    "array_cost_W": Param("array_cost_W", 30, 35, 4, "$/W", "low", "partial",
        "Low-tens $/W today (multijunction 3-5x terrestrial); $4/W aspirational",
        "$90/m^2 InP cell-only confirmed; $4/W is a forward target not a price."),
    "life_yr": Param("life_yr", 5, 5, 10, "yr", "high", "partial",
        "SemiAnalysis 5 yr -> 10 yr post-2032; FCC 5-yr deorbit caps it",
        "10-yr post-2032 collides with FCC deorbit licensing -> needs regulatory-extension flag."),
    "g_facility_life_yr": Param("g_facility_life_yr", 15, 15, 15, "yr", "high", "partial",
        "SemiAnalysis 15-yr Earth facility vs 5-yr station",
        "Life mismatch drives the ~17-18x levelized ratio; launch is the largest ABSOLUTE capex line, not life."),
    "wacc_space": Param("wacc_space", 0.15, 0.20, 0.103, "frac", "low", "confirmed",
        "SemiAnalysis 15% -> 10.3%; HFW non-financeable 20%+",
        "MOST FRAGILE parameter. Non-repossessable/non-serviceable/FCC-capped -> add 20%+ case."),
    "rev_decline": Param("rev_decline", 0.15, 0.23, 0.10, "frac/yr", "medium", "confirmed",
        "Starlink ARPU ~13%/yr (S-1 2026); raw token deflation 10-50x/yr (Epoch)",
        "Model exploits the GAP between off-take erosion (~13%) and token deflation (~10-50x). Stress 0.10-0.23."),
    "rad_availability": Param("rad_availability", 0.95, 0.92, 0.98, "frac", "medium", "confirmed",
        "SemiAnalysis 95% net of solar events; Suncatcher TID de-risked to 15 krad",
        "SEUs ECC/scrub-manageable; ~9%/yr attrition (Forethought)."),
    "insurance_frac": Param("insurance_frac", 0.07, 0.0, 0.0, "frac", "medium", "confirmed",
        "Gallagher/Lexology 2025 (market ~$300M/risk, ~$0.5-0.7B/yr pool)",
        "Market too small for GW scale -> reframe as self-insurance via redundancy (see odc.finance)."),
}

# SemiAnalysis published anchors (verified confirmed; LCOC basis, not TCO).
SA_VERIFIED = dict(gpu_hr_s=10.91, gpu_hr_g=2.49, note="LCOC, not TCO ($8.64/$2.37).",
                   verdict="confirmed")

# Academic citation anchors (Semantic Scholar, 12 Jun 2026; verified).
CITATIONS = {
    "Denby & Lucia, Orbital Edge Computing (ASPLOS 2020)": 292,
    "Bhattacherjee et al., In-orbit Computing (HotNets 2020)": 129,
    "Denby & Lucia, Machine Inference in Space (IEEE CAL 2019)": 104,
    "Denby et al., Kodan (ASPLOS 2023)": 67,
    "Google Project Suncatcher (arXiv:2511.19468, 2025)": 20,
    "Jones, Launch Cost (ICES 2018)": 224,
    "Kaplan et al., Scaling Laws (2020)": 3311,
    "Hoffmann et al., Chinchilla (2022)": 3361,   # corrected from a mis-cited 2596
    "Patterson et al., Carbon Footprint (2021)": 1613,
    "Luccioni et al., Inference Energy (FAccT 2024)": 420,
}


def monte_carlo_ranges():
    """Provenance-weighted (low, central, high) triples for the Monte Carlo sampler,
    replacing v2's flat +/-30% triangular draws."""
    return {k: (p.low, p.central, p.high) for k, p in REGISTRY.items()}


if __name__ == "__main__":
    print(f"{'param':20} {'central':>9} {'low':>9} {'high':>9} {'conf':>7} {'verdict':>13}  source")
    for p in REGISTRY.values():
        print(f"{p.field:20} {p.central:9.3g} {p.low:9.3g} {p.high:9.3g} {p.confidence:>7} "
              f"{p.verdict:>13}  {p.source[:48]}")
    print("\nAcademic citation anchors (Semantic Scholar, 12 Jun 2026):")
    for k, v in sorted(CITATIONS.items(), key=lambda kv: -kv[1]):
        print(f"   {v:5d}  {k}")
