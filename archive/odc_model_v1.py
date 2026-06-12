"""
Orbital Data Centre (ODC) viability system model
=================================================
Basis: 1 MW of *sellable* IT load in a dusk-dawn sun-synchronous orbit (~550 km),
scaled linearly. Chains: power -> thermal -> mass -> launch -> capex -> LCOC/NPV,
benchmarked against a terrestrial AI data centre using the same accelerators.

Units: SI unless noted. Money in USD. $M = 1e6 USD.
"""
from dataclasses import dataclass, replace, asdict
import numpy as np

SIGMA = 5.670374419e-8   # Stefan-Boltzmann, W/m^2/K^4
SOLAR_CONST = 1361.0     # W/m^2 at 1 AU

@dataclass
class P:
    name: str = "Today (2026, Falcon-class)"
    # --- Orbit / environment ---
    sunlit_fraction: float = 0.99     # dusk-dawn SSO ~ eclipse-free most of year
    T_sink: float = 230.0             # effective radiative sink (K): mix of deep space + Earth IR/albedo view
    # --- Power chain ---
    cell_eff: float = 0.30            # triple-junction GaAs class
    packing: float = 0.85             # cell packing factor on blanket
    pointing: float = 0.97            # avg cos-loss + harness/PMAD inefficiency lump
    degr_rate: float = 0.02           # array degradation per year (LEO radiation/UV)
    sp_array: float = 100.0           # array specific power, W/kg @ BOL (ROSA-class today)
    array_cost_W: float = 30.0        # $/W BOL installed (space-rated, today)
    overhead_frac: float = 0.08       # bus loads: avionics, comms, pumps, ADCS as frac of IT
    # --- Thermal ---
    T_rad: float = 320.0              # radiating temperature (K) ~ liquid loop 45-55C
    emissivity: float = 0.90
    fin_eff: float = 0.90
    rad_areal_kg_m2: float = 6.0      # kg per m^2 of (two-sided) panel incl. pumped loop share
    rad_cost_m2: float = 3000.0       # $ per m^2 panel
    # --- IT payload ---
    it_kg_per_kW: float = 15.0        # servers + enclosure + cold plates (GB200 NVL72 ~11 kg/kW bare)
    it_cost_W: float = 35.0           # $/W accelerator+server capex (same chips both sides)
    shield_t_per_MW: float = 2.0      # Al-equivalent spot shielding around electronics
    overprovision: float = 0.15       # extra IT capacity launched (no in-orbit repair)
    compute_overhead: float = 0.05    # throughput lost to checkpointing/ECC/SEU scrubbing
    # --- Platform ---
    structure_frac: float = 0.18      # structure+ADCS+harness+prop, frac of subsystem mass
    avionics_comms_M_MW: float = 8.0  # $M/MW: optical terminals, OBC, GNC
    integration_M_MW: float = 12.0    # $M/MW: AIT + amortized NRE
    insurance_frac: float = 0.08      # of (hardware + launch)
    ops_M_MW_yr: float = 1.5          # $M/MW/yr mission ops + ground segment
    life_yr: float = 5.0              # economic life (HW obsolescence-bound)
    # --- Launch ---
    launch_kg: float = 3000.0         # $/kg to SSO (F9 dedicated ~$2.5-4k)
    # --- Economics (shared market) ---
    rev0_kWh: float = 3.00            # $ per sellable IT-kW-hour at t=0 (~H100 @$2/h/0.7kW)
    rev_decline: float = 0.15         # $/GPU-hr erosion per year
    util: float = 0.90                # sold/used fraction
    disc: float = 0.10                # discount rate
    deploy_delay_yr: float = 1.0      # space: build+launch+commission before revenue
    # --- Terrestrial benchmark ---
    g_facility_M_MW: float = 12.0     # $M per MW IT: shell, power, cooling (excl. IT)
    g_pue: float = 1.30
    g_elec_MWh: float = 95.0          # $/MWh all-in industrial
    g_ops_M_MW_yr: float = 1.0
    g_overprovision: float = 0.03     # ground spares (repairable)
    g_compute_overhead: float = 0.01
    g_delay_yr: float = 3.0           # grid interconnect / permitting queue


# ---------------- Physics & mass budget ----------------

def power_thermal_mass(p: P, P_it_MW: float = 1.0):
    sell_kW = 1000.0 * P_it_MW
    gross_kW = sell_kW * (1 + p.overprovision)
    bus_kW = gross_kW * (1 + p.overhead_frac)

    # Array sized so EOL power under degradation+pointing+duty still meets bus demand
    eol = (1 - p.degr_rate) ** p.life_yr
    arr_BOL_kW = bus_kW / (eol * p.pointing * p.sunlit_fraction)
    A_array = arr_BOL_kW * 1000 / (SOLAR_CONST * p.cell_eff * p.packing)        # m^2
    M_array = arr_BOL_kW * 1000 / p.sp_array                                     # kg
    C_array = arr_BOL_kW * 1000 * p.array_cost_W / 1e6                           # $M

    # All consumed electrical power becomes heat (RF/laser out is negligible)
    Q_kW = bus_kW
    q_net = p.emissivity * p.fin_eff * SIGMA * (p.T_rad**4 - p.T_sink**4)        # W/m^2 per side
    A_rad = Q_kW * 1000 / (2 * q_net)                                            # m^2 panel (2-sided)
    M_rad = A_rad * p.rad_areal_kg_m2
    C_rad = A_rad * p.rad_cost_m2 / 1e6

    M_it = gross_kW * p.it_kg_per_kW
    M_shield = p.shield_t_per_MW * 1000 * P_it_MW
    M_sub = M_array + M_rad + M_it + M_shield
    M_dry = M_sub * (1 + p.structure_frac)

    return dict(sell_kW=sell_kW, gross_kW=gross_kW, bus_kW=bus_kW,
                arr_BOL_kW=arr_BOL_kW, A_array=A_array, M_array=M_array, C_array=C_array,
                Q_kW=Q_kW, q_net_side=q_net, A_rad=A_rad, M_rad=M_rad, C_rad=C_rad,
                M_it=M_it, M_shield=M_shield, M_dry=M_dry)


# ---------------- Cost & cash-flow model ----------------

def space_capex(p: P, pt: dict, P_it_MW: float = 1.0):
    C_it = pt["gross_kW"] * 1000 * p.it_cost_W / 1e6
    C_platform = pt["C_array"] + pt["C_rad"] + p.avionics_comms_M_MW * P_it_MW
    C_launch = pt["M_dry"] * p.launch_kg / 1e6
    C_int = p.integration_M_MW * P_it_MW
    C_ins = p.insurance_frac * (C_it + C_platform + C_launch)
    total = C_it + C_platform + C_launch + C_int + C_ins
    return dict(C_it=C_it, C_platform=C_platform, C_launch=C_launch,
                C_int=C_int, C_ins=C_ins, total=total)

def ground_capex(p: P, P_it_MW: float = 1.0):
    sell_kW = 1000 * P_it_MW
    C_it = sell_kW * (1 + p.g_overprovision) * 1000 * p.it_cost_W / 1e6
    C_fac = p.g_facility_M_MW * P_it_MW
    return dict(C_it=C_it, C_fac=C_fac, total=C_it + C_fac)

def _disc_sum(vals, p: P):
    """Mid-year discounting of a per-year series."""
    t = np.arange(len(vals)) + 0.5
    return float(np.sum(np.asarray(vals) / (1 + p.disc) ** t))

def annual_compute_kWh(p: P, side: str, P_it_MW=1.0):
    sell_kW = 1000 * P_it_MW
    ov = p.compute_overhead if side == "space" else p.g_compute_overhead
    return sell_kW * 8760 * p.util * (1 - ov)   # kW-hours of sellable compute / yr

def lcoc_and_npv(p: P, P_it_MW: float = 1.0, include_delay=False):
    pt = power_thermal_mass(p, P_it_MW)
    cap_s = space_capex(p, pt, P_it_MW)
    cap_g = ground_capex(p, P_it_MW)
    yrs = int(round(p.life_yr))

    kwh_s = annual_compute_kWh(p, "space", P_it_MW)
    kwh_g = annual_compute_kWh(p, "ground", P_it_MW)

    opex_s = [p.ops_M_MW_yr * P_it_MW] * yrs
    elec = p.g_pue * (1000 * P_it_MW) * 8760 * p.util * p.g_elec_MWh / 1e3 / 1e6  # $M/yr
    opex_g = [p.g_ops_M_MW_yr * P_it_MW + elec] * yrs

    d_kwh = _disc_sum([1.0] * yrs, p)
    lcoc_s = (cap_s["total"] + _disc_sum(opex_s, p)) * 1e6 / (kwh_s * d_kwh)
    lcoc_g = (cap_g["total"] + _disc_sum(opex_g, p)) * 1e6 / (kwh_g * d_kwh)

    def npv(capex, opex, kwh, delay):
        d0 = delay if include_delay else 0.0
        t = np.arange(yrs) + 0.5 + d0
        rev = kwh * p.rev0_kWh * (1 - p.rev_decline) ** t / 1e6     # $M/yr
        cf = rev - np.asarray(opex)
        return float(np.sum(cf / (1 + p.disc) ** t) - capex)

    npv_s = npv(cap_s["total"], opex_s, kwh_s, p.deploy_delay_yr)
    npv_g = npv(cap_g["total"], opex_g, kwh_g, p.g_delay_yr)

    # Breakeven launch $/kg: capex_s(L) = A + L*M_dry*(1+ins); solve lcoc_s = lcoc_g
    A = cap_s["C_it"] + cap_s["C_platform"] + cap_s["C_int"] \
        + p.insurance_frac * (cap_s["C_it"] + cap_s["C_platform"])
    target_capex = (lcoc_g * kwh_s * d_kwh / 1e6) - _disc_sum(opex_s, p)
    L_be = (target_capex - A) * 1e6 / (pt["M_dry"] * (1 + p.insurance_frac))

    return dict(pt=pt, cap_s=cap_s, cap_g=cap_g, lcoc_s=lcoc_s, lcoc_g=lcoc_g,
                ratio=lcoc_s / lcoc_g, npv_s=npv_s, npv_g=npv_g,
                breakeven_launch=L_be, kwh_s=kwh_s)


# ---------------- Scenario presets ----------------

TODAY = P()
EARLY = replace(TODAY, name="Early Starship (~2028-29)",
    cell_eff=0.32, degr_rate=0.015, sp_array=180, array_cost_W=12, overhead_frac=0.07,
    T_rad=325, rad_areal_kg_m2=4.0, rad_cost_m2=1500,
    it_kg_per_kW=11, it_cost_W=30, shield_t_per_MW=1.5, overprovision=0.12, compute_overhead=0.04,
    structure_frac=0.15, avionics_comms_M_MW=5, integration_M_MW=6, insurance_frac=0.06,
    ops_M_MW_yr=1.0, launch_kg=800, rev0_kWh=2.4, deploy_delay_yr=0.75)
MATURE = replace(TODAY, name="Mature Starship (~2033)",
    cell_eff=0.34, degr_rate=0.01, sp_array=350, array_cost_W=4, overhead_frac=0.06,
    T_rad=330, rad_areal_kg_m2=2.5, rad_cost_m2=600,
    it_kg_per_kW=8, it_cost_W=25, shield_t_per_MW=1.0, overprovision=0.10, compute_overhead=0.03,
    structure_frac=0.12, avionics_comms_M_MW=3, integration_M_MW=3, insurance_frac=0.04,
    ops_M_MW_yr=0.6, life_yr=6, launch_kg=150, rev0_kWh=2.0, deploy_delay_yr=0.5)

SCENARIOS = [TODAY, EARLY, MATURE]


if __name__ == "__main__":
    for sc in SCENARIOS:
        r = lcoc_and_npv(sc)
        pt, cs = r["pt"], r["cap_s"]
        print(f"\n=== {sc.name} ===")
        print(f"  Array: {pt['arr_BOL_kW']:.0f} kW BOL | {pt['A_array']:,.0f} m^2 | {pt['M_array']/1e3:.1f} t | ${pt['C_array']:.1f}M")
        print(f"  Radiator @{sc.T_rad:.0f}K: net {pt['q_net_side']:.0f} W/m^2/side | {pt['A_rad']:,.0f} m^2 | {pt['M_rad']/1e3:.1f} t")
        print(f"  IT mass {pt['M_it']/1e3:.1f} t | shield {pt['M_shield']/1e3:.1f} t | DRY {pt['M_dry']/1e3:.1f} t/MW")
        print(f"  Capex/MW: IT ${cs['C_it']:.0f}M + platform ${cs['C_platform']:.0f}M + launch ${cs['C_launch']:.0f}M "
              f"+ int ${cs['C_int']:.0f}M + ins ${cs['C_ins']:.0f}M = ${cs['total']:.0f}M  (ground ${r['cap_g']['total']:.0f}M)")
        print(f"  LCOC: space ${r['lcoc_s']:.2f} vs ground ${r['lcoc_g']:.2f} per kW-h  -> ratio {r['ratio']:.2f}x")
        print(f"  Breakeven launch: ${r['breakeven_launch']:,.0f}/kg | NPV/MW: space ${r['npv_s']:.0f}M, ground ${r['npv_g']:.0f}M")
