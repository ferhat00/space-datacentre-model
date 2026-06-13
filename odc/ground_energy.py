"""
odc.ground_energy — terrestrial energy-source comparator
========================================================
v2 collapsed the ground side to a single $/MWh scalar (g_elec_MWh). But the literature's
strongest pro-space argument is NOT cheaper electrons in orbit — energy is only ~$0.6B of
an ~$8.5B/yr 1 GW ground TCO (Epoch AI 2025). It is TIME-TO-POWER: US interconnection
queues run ~40 months (PJM) to >7 years total, and siting/water fights block $64B of
projects. This module enumerates the real ground options so space competes on speed and
siting, not $/MWh.

Sources: Epoch AI 2025; Lazard LCOE+ Jun 2025; LBNL/Latitude Media 2025-26 (queues);
Handmer/Terraform 2026 (off-grid solar+battery); Fervo 2026 (geothermal); Bloom 2025
(SOFC fuel cells, 50 MW/90 days); Google-Kairos / SMR 2026.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundSource:
    name: str
    usd_per_mwh: float          # effective $/MWh (LCOE or fuel+capex-amortized)
    time_to_power_months: float # lead time from decision to energized
    capacity_factor: float      # achievable CF for a 24/7 AI load
    carbon_kg_per_mwh: float    # lifecycle CO2 (approx; 0 = ~carbon-free)
    dispatchable: bool
    note: str


GRID_US_INDUSTRIAL = GroundSource("Grid (US industrial, DC-weighted)", 83.0, 48.0, 0.95, 370,
    False, "~8.3 c/kWh but the binding constraint: ~40 mo PJM / 36-48 mo DC zones / >7 yr "
           "total interconnect. ERCOT large-load queue ~410 GW (~87% data centres).")
GRID_ERCOT_WHOLESALE = GroundSource("Grid wholesale (ERCOT)", 30.0, 48.0, 0.90, 350,
    False, "$27-34/MWh average (peaks $110-165); same multi-year queue.")
BTM_GAS_TURBINE = GroundSource("Behind-the-meter gas turbine", 75.0, 8.0, 0.92, 450,
    True, "Fast: xAI Colossus 276 MW in ~8 mo. ~$2,500/kW + ~$55/MWh fuel. OEM new-build "
          "slots now gone to 2029-30, so 'fast' assumes turbine availability.")
SOFC_FUEL_CELL = GroundSource("SOFC fuel cell (Bloom)", 110.0, 3.0, 0.95, 380,
    True, "Fastest to power: 50 MW in 90 days, 100 MW in 120 days. Gas SOFC ~680-830 lb "
          "CO2/MWh. Premium $/MWh buys speed.")
SMR = GroundSource("SMR (small modular reactor)", 120.0, 54.0, 0.92, 12,
    True, "FOAK $90-160/MWh, NOAK $50-90. First DC units 2028-2030. Carbon-free baseload "
          "but slow.")
GEOTHERMAL = GroundSource("Geothermal (Fervo-class EGS)", 88.0, 30.0, 0.90, 20,
    True, "~$88/MWh FOAK with credits; ~90% CF carbon-free baseload. Fervo Cape 100 MW 2026.")
UTILITY_SOLAR = GroundSource("Utility solar (standalone)", 58.0, 12.0, 0.25, 30,
    False, "$38-78/MWh but ~25% CF -> needs 4x oversize + firming for a 24/7 load.")
SOLAR_PLUS_STORAGE = GroundSource("Solar + storage", 90.0, 14.0, 0.60, 30,
    False, "$50-131/MWh; the closest ground analog to orbital solar. Improving fast, which "
          "STRENGTHENS the ground baseline against space.")
OFFGRID_SOLAR_BATTERY = GroundSource("Off-grid solar+battery (Handmer)", 70.0, 10.0, 0.95, 25,
    False, "PV ~$200/kW, batt ~$200/kWh, ~15 acres/MW, throttled to 99-99.9%. Bypasses the "
          "grid queue entirely — the real terrestrial answer to space's 'no queue' pitch.")

CATALOG = {s.name: s for s in (
    GRID_US_INDUSTRIAL, GRID_ERCOT_WHOLESALE, BTM_GAS_TURBINE, SOFC_FUEL_CELL, SMR,
    GEOTHERMAL, UTILITY_SOLAR, SOLAR_PLUS_STORAGE, OFFGRID_SOLAR_BATTERY)}


def fastest_sources(max_months: float = 12.0):
    """Ground options that can be energized within `max_months` — the space-competition set."""
    return sorted((s for s in CATALOG.values() if s.time_to_power_months <= max_months),
                  key=lambda s: s.time_to_power_months)


def cheapest_sources(max_carbon: float = None):
    src = CATALOG.values()
    if max_carbon is not None:
        src = [s for s in src if s.carbon_kg_per_mwh <= max_carbon]
    return sorted(src, key=lambda s: s.usd_per_mwh)


if __name__ == "__main__":
    print(f"{'Ground source':36} {'$/MWh':>7} {'lead mo':>8} {'CF':>5} {'kgCO2/MWh':>10} {'disp':>5}")
    for s in CATALOG.values():
        print(f"{s.name:36} {s.usd_per_mwh:7.0f} {s.time_to_power_months:8.0f} "
              f"{s.capacity_factor:5.2f} {s.carbon_kg_per_mwh:10.0f} {str(s.dispatchable):>5}")
    print("\nEnergizable within 12 months (space competes on speed here):")
    for s in fastest_sources(12):
        print(f"   {s.name:36} {s.time_to_power_months:.0f} mo @ ${s.usd_per_mwh:.0f}/MWh")
