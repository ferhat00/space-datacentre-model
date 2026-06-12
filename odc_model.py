"""
Orbital Data Centre (ODC) viability system model — v2
=====================================================
Recalibrated against the SemiAnalysis "AI Space Datacenter TCO Model"
introduction (Nishball et al., 3 Jun 2026).

Changes vs v1:
  * Eclipse + battery: dawn-dusk SSO still sees up to ~35 min/day eclipse;
    battery sized for full bus power through eclipse (DoD, round-trip eff).
  * Split WACC: space 15% (immature, de-risks to 10.3%) vs ground 10.3%.
  * CRF levelization with MIXED LIVES: space station amortized over its own
    (short) life; ground splits IT (5 yr) from facility (15 yr) — this life
    mismatch is SemiAnalysis's single biggest cost driver (17-18x on DC capex).
  * Reliability gross-up: space radiation availability 95% + 20% redundancy
    (=> ~26% LCOC gross-up, matching SA); ground 5% cold spares.
  * Launch Today: $1,600/kg (F9 actual $1.4-1.8k/kg per SA, was $3,000).
  * Cross-check outputs in SA units: $/GPU-hr (B300), $/PFLOP-hr, $/B tokens.

Basis: 1 MW of *sellable* IT load, dusk-dawn SSO ~550 km, scaled linearly.
Units SI unless noted. Money USD; $M = 1e6.
"""
from dataclasses import dataclass, replace, asdict
import numpy as np

SIGMA = 5.670374419e-8   # Stefan-Boltzmann, W/m^2/K^4
SOLAR_CONST = 1361.0     # W/m^2 at 1 AU

# --- B300 reference for SemiAnalysis cross-checks (30.5 kW / 16 GPU cluster) ---
B300_KW_PER_GPU   = 30.5 / 16          # 1.906 kW critical IT per GPU
B300_PFLOPS_FP4   = 15.0               # PFLOPS/GPU implied by SA arithmetic (10.91/0.73); their prose says 4,500 TFLOPS
B300_TOKS_PER_S   = 5100.0             # DeepSeek R1 FP4 disagg (InferenceX)

@dataclass
class P:
    name: str = "Today (2026, Falcon-class)"
    # --- Orbit / environment ---
    eclipse_min_day: float = 35.0     # dawn-dusk SSO worst-case eclipse (SA: "up to 35 min/day")
    T_sink: float = 230.0             # effective radiative sink (K)
    # --- Power chain ---
    cell_eff: float = 0.30
    packing: float = 0.85
    pointing: float = 0.97
    degr_rate: float = 0.02           # array degradation /yr
    sp_array: float = 110.0           # W/kg BOL (ROSA-class)
    array_cost_W: float = 30.0        # $/W BOL installed
    overhead_frac: float = 0.08       # bus loads as frac of IT
    # --- Energy storage (eclipse ride-through) ---
    batt_Wh_kg: float = 160.0         # space-qualified Li-ion pack level
    batt_cost_kWh: float = 600.0      # $/kWh installed
    batt_dod: float = 0.80            # usable depth of discharge
    batt_rt_eff: float = 0.93         # round-trip efficiency
    # --- Thermal ---
    T_rad: float = 320.0
    emissivity: float = 0.90
    fin_eff: float = 0.90
    rad_areal_kg_m2: float = 6.0
    rad_cost_m2: float = 3000.0
    # --- IT payload ---
    it_kg_per_kW: float = 13.0
    it_cost_W: float = 32.0           # $/W (SA: $986K IT capex / 30.5 kW = $32.3/W)
    shield_t_per_MW: float = 1.8
    overprovision: float = 0.20       # SA: 20% redundancy, no in-orbit repair
    compute_overhead: float = 0.05    # checkpoint/ECC/SEU scrubbing
    rad_availability: float = 0.95    # SA: compute availability net of radiation events
    # --- Platform ---
    structure_frac: float = 0.18
    avionics_comms_M_MW: float = 7.0
    integration_M_MW: float = 10.0
    insurance_frac: float = 0.07
    ops_M_MW_yr: float = 1.5
    life_yr: float = 5.0              # station life (SA: 5 yr -> 10 yr post-2032 robotics)
    # --- Launch ---
    launch_kg: float = 1600.0         # $/kg SSO (SA: F9 today $1,400-1,800/kg)
    # --- Economics ---
    rev0_kWh: float = 3.00
    rev_decline: float = 0.15
    util: float = 0.90
    wacc_space: float = 0.150         # SA: 15% initial, de-risks to 10.3% in ~10 yr
    wacc_ground: float = 0.103        # SA: 7% pre-tax debt, 20% equity, 75/25
    deploy_delay_yr: float = 1.0
    # --- Terrestrial benchmark ---
    g_facility_M_MW: float = 12.0     # grid layer $12-15M/MW (SA layers 2-4: $10-20M+)
    g_facility_life_yr: float = 15.0  # SA: Earth DC facility 15 yr vs space 5 yr
    it_life_yr: float = 5.0           # chips, both sides
    g_pue: float = 1.35               # SA assumption
    g_elec_MWh: float = 87.0          # SA: $0.087/kWh
    g_ops_M_MW_yr: float = 1.0
    g_overprovision: float = 0.05     # SA: ~5% cold spares on Earth
    g_compute_overhead: float = 0.01
    g_rad_availability: float = 1.00
    g_delay_yr: float = 3.0


def crf(r: float, n: float) -> float:
    """Capital recovery factor, monthly compounding (SemiAnalysis convention),
    expressed as annual charge per $1 of capex over n years at annual rate r."""
    if r <= 0:
        return 1.0 / n
    m, k = r / 12.0, 12.0 * n
    return 12.0 * m * (1 + m) ** k / ((1 + m) ** k - 1)


# ---------------- Physics & mass budget ----------------

def power_thermal_mass(p: P, P_it_MW: float = 1.0):
    sell_kW  = 1000.0 * P_it_MW
    gross_kW = sell_kW * (1 + p.overprovision)
    bus_kW   = gross_kW * (1 + p.overhead_frac)

    # Orbit lighting
    ecl_frac = p.eclipse_min_day / 1440.0
    sunlit   = 1.0 - ecl_frac

    # Battery: ride full bus power through eclipse
    E_batt_kWh = bus_kW * (p.eclipse_min_day / 60.0) / p.batt_dod
    M_batt = E_batt_kWh * 1000.0 / p.batt_Wh_kg
    C_batt = E_batt_kWh * p.batt_cost_kWh / 1e6

    # Array: supply bus while sunlit AND recharge battery (round-trip loss)
    daily_factor = (sunlit + ecl_frac / p.batt_rt_eff) / sunlit
    eol = (1 - p.degr_rate) ** p.life_yr
    arr_BOL_kW = bus_kW * daily_factor / (eol * p.pointing)
    A_array = arr_BOL_kW * 1000 / (SOLAR_CONST * p.cell_eff * p.packing)
    M_array = arr_BOL_kW * 1000 / p.sp_array
    C_array = arr_BOL_kW * 1000 * p.array_cost_W / 1e6

    # Thermal: all consumed power -> heat; two-sided panels
    Q_kW = bus_kW
    q_net = p.emissivity * p.fin_eff * SIGMA * (p.T_rad**4 - p.T_sink**4)
    A_rad = Q_kW * 1000 / (2 * q_net)
    M_rad = A_rad * p.rad_areal_kg_m2
    C_rad = A_rad * p.rad_cost_m2 / 1e6

    M_it = gross_kW * p.it_kg_per_kW
    M_shield = p.shield_t_per_MW * 1000 * P_it_MW
    M_sub = M_array + M_rad + M_it + M_shield + M_batt
    M_dry = M_sub * (1 + p.structure_frac)

    return dict(sell_kW=sell_kW, gross_kW=gross_kW, bus_kW=bus_kW, sunlit=sunlit,
                E_batt_kWh=E_batt_kWh, M_batt=M_batt, C_batt=C_batt,
                arr_BOL_kW=arr_BOL_kW, A_array=A_array, M_array=M_array, C_array=C_array,
                Q_kW=Q_kW, q_net_side=q_net, A_rad=A_rad, M_rad=M_rad, C_rad=C_rad,
                M_it=M_it, M_shield=M_shield, M_dry=M_dry)


# ---------------- Cost & cash-flow model ----------------

def space_capex(p: P, pt: dict, P_it_MW: float = 1.0):
    C_it = pt["gross_kW"] * 1000 * p.it_cost_W / 1e6
    C_platform = pt["C_array"] + pt["C_rad"] + pt["C_batt"] + p.avionics_comms_M_MW * P_it_MW
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

def annual_compute_kWh(p: P, side: str, P_it_MW=1.0):
    sell_kW = 1000 * P_it_MW
    if side == "space":
        ov, avail = p.compute_overhead, p.rad_availability
    else:
        ov, avail = p.g_compute_overhead, p.g_rad_availability
    return sell_kW * 8760 * p.util * (1 - ov) * avail   # sellable compute kW-h / yr

def lcoc_and_npv(p: P, P_it_MW: float = 1.0, include_delay=False):
    pt = power_thermal_mass(p, P_it_MW)
    cap_s = space_capex(p, pt, P_it_MW)
    cap_g = ground_capex(p, P_it_MW)

    kwh_s = annual_compute_kWh(p, "space", P_it_MW)
    kwh_g = annual_compute_kWh(p, "ground", P_it_MW)

    opex_s = p.ops_M_MW_yr * P_it_MW                                   # $M/yr
    elec   = p.g_pue * (1000*P_it_MW) * 8760 * p.util * p.g_elec_MWh / 1e3 / 1e6
    opex_g = p.g_ops_M_MW_yr * P_it_MW + elec                           # $M/yr

    # CRF levelization with mixed lives (SA method)
    crf_s   = crf(p.wacc_space,  p.life_yr)            # whole station, short life
    crf_git = crf(p.wacc_ground, p.it_life_yr)         # ground IT: 5 yr
    crf_gfc = crf(p.wacc_ground, p.g_facility_life_yr) # ground facility: 15 yr

    ann_s = crf_s * cap_s["total"] + opex_s
    ann_g = crf_git * cap_g["C_it"] + crf_gfc * cap_g["C_fac"] + opex_g
    lcoc_s = ann_s * 1e6 / kwh_s
    lcoc_g = ann_g * 1e6 / kwh_g

    # NPV over respective horizons (ground facility charged as 15-yr annuity)
    def npv(side):
        if side == "space":
            r, yrs, d0 = p.wacc_space, int(round(p.life_yr)), p.deploy_delay_yr
            cap0, opx, kwh = cap_s["total"], opex_s, kwh_s
        else:
            r, yrs, d0 = p.wacc_ground, int(round(p.it_life_yr)), p.g_delay_yr
            cap0 = cap_g["C_it"]
            opx  = opex_g + crf_gfc * cap_g["C_fac"]
            kwh  = kwh_g
        dd = d0 if include_delay else 0.0
        t = np.arange(yrs) + 0.5 + dd
        rev = kwh * p.rev0_kWh * (1 - p.rev_decline) ** t / 1e6
        return float(np.sum((rev - opx) / (1 + r) ** t) - cap0)

    npv_s, npv_g = npv("space"), npv("ground")

    # Breakeven launch $/kg (closed form; linear in L)
    A = cap_s["C_it"] + cap_s["C_platform"] + cap_s["C_int"] \
        + p.insurance_frac * (cap_s["C_it"] + cap_s["C_platform"])
    target_ann_capex = (lcoc_g * kwh_s / 1e6 - opex_s) / crf_s
    L_be = (target_ann_capex - A) * 1e6 / (pt["M_dry"] * (1 + p.insurance_frac))

    # SemiAnalysis cross-check units (wall-clock convention: capital spread over
    # all 8760 h of deployed capacity; reliability gross-ups apply, commercial
    # utilization does not). My headline LCOC instead charges only sold hours.
    wc_s = lcoc_s * p.util   # remove utilization from the divisor
    wc_g = lcoc_g * p.util
    sa = dict(
        gpu_hr_s = wc_s * B300_KW_PER_GPU,
        gpu_hr_g = wc_g * B300_KW_PER_GPU,
        pflop_hr_s = wc_s * B300_KW_PER_GPU / B300_PFLOPS_FP4,
        pflop_hr_g = wc_g * B300_KW_PER_GPU / B300_PFLOPS_FP4,
        btok_s = wc_s * B300_KW_PER_GPU / (B300_TOKS_PER_S * 3600) * 1e9,
        btok_g = wc_g * B300_KW_PER_GPU / (B300_TOKS_PER_S * 3600) * 1e9,
    )

    return dict(pt=pt, cap_s=cap_s, cap_g=cap_g, lcoc_s=lcoc_s, lcoc_g=lcoc_g,
                ratio=lcoc_s/lcoc_g, npv_s=npv_s, npv_g=npv_g,
                breakeven_launch=L_be, kwh_s=kwh_s, kwh_g=kwh_g,
                ann_s=ann_s, ann_g=ann_g, sa=sa)


# ---------------- Scenario presets ----------------

TODAY = P()
EARLY = replace(TODAY, name="Early Starship (~2028-30)",
    cell_eff=0.32, degr_rate=0.015, sp_array=180, array_cost_W=12, overhead_frac=0.07,
    batt_Wh_kg=200, batt_cost_kWh=300,
    T_rad=325, rad_areal_kg_m2=4.0, rad_cost_m2=1500,
    it_kg_per_kW=11, it_cost_W=28, shield_t_per_MW=1.5,
    overprovision=0.15, compute_overhead=0.04, rad_availability=0.96,
    structure_frac=0.15, avionics_comms_M_MW=5, integration_M_MW=6, insurance_frac=0.06,
    ops_M_MW_yr=1.0, launch_kg=700, rev0_kWh=2.4,
    wacc_space=0.125, deploy_delay_yr=0.75)
MATURE = replace(TODAY, name="Mature Starship (~2033-35)",
    cell_eff=0.34, degr_rate=0.01, sp_array=350, array_cost_W=4, overhead_frac=0.06,
    batt_Wh_kg=260, batt_cost_kWh=150, batt_dod=0.85,
    T_rad=330, rad_areal_kg_m2=2.5, rad_cost_m2=600,
    it_kg_per_kW=8, it_cost_W=24, shield_t_per_MW=1.0,
    overprovision=0.10, compute_overhead=0.03, rad_availability=0.98,
    structure_frac=0.12, avionics_comms_M_MW=3, integration_M_MW=3, insurance_frac=0.04,
    ops_M_MW_yr=0.6, life_yr=8, launch_kg=250, rev0_kWh=2.0,
    wacc_space=0.103, deploy_delay_yr=0.5)

SA26 = replace(TODAY, name="SemiAnalysis 2026 (B300, repro)",
    sp_array=150, array_cost_W=22, overhead_frac=0.06,
    batt_Wh_kg=200, batt_cost_kWh=300,
    T_rad=322, rad_areal_kg_m2=3.5, rad_cost_m2=1100,
    it_kg_per_kW=11, it_cost_W=32.3, shield_t_per_MW=1.0,
    overprovision=0.20, compute_overhead=0.0, rad_availability=0.95,
    structure_frac=0.14, avionics_comms_M_MW=4.5, integration_M_MW=8.0,
    insurance_frac=0.06, ops_M_MW_yr=1.65, launch_kg=1600, wacc_space=0.15,
    g_ops_M_MW_yr=0.15, g_overprovision=0.0, g_compute_overhead=0.048,
    util=0.80)

SCENARIOS = [TODAY, EARLY, MATURE]

# SemiAnalysis published anchors (2026, B300 30.5 kW cluster) for validation
SA_ANCHORS = dict(gpu_hr_s=10.91, gpu_hr_g=2.49, ratio=10.91/2.49,
                  pflop_hr_s=0.73, pflop_hr_g=0.17, btok_s=590.0, btok_g=135.0,
                  capex_total_s_per_W=4.1e6/30500, capex_total_g_per_W=1.4e6/30500,
                  launch_share_dc_capex=1.6/3.1)


if __name__ == "__main__":
    for sc in SCENARIOS:
        r = lcoc_and_npv(sc)
        pt, cs, sa = r["pt"], r["cap_s"], r["sa"]
        print(f"\n=== {sc.name} ===")
        print(f"  Sunlit {pt['sunlit']*100:.1f}% | battery {pt['E_batt_kWh']:,.0f} kWh, {pt['M_batt']/1e3:.2f} t, ${pt['C_batt']:.1f}M")
        print(f"  Array {pt['arr_BOL_kW']:.0f} kW BOL | {pt['A_array']:,.0f} m^2 | {pt['M_array']/1e3:.1f} t | ${pt['C_array']:.1f}M")
        print(f"  Radiator @{sc.T_rad:.0f}K: {pt['q_net_side']:.0f} W/m^2/side | {pt['A_rad']:,.0f} m^2 | {pt['M_rad']/1e3:.1f} t")
        print(f"  DRY {pt['M_dry']/1e3:.1f} t/MW | capex: IT ${cs['C_it']:.0f} + plat ${cs['C_platform']:.0f} + launch ${cs['C_launch']:.0f}"
              f" + int ${cs['C_int']:.0f} + ins ${cs['C_ins']:.0f} = ${cs['total']:.0f}M  (ground ${r['cap_g']['total']:.0f}M)")
        print(f"  LCOC ${r['lcoc_s']:.2f} vs ${r['lcoc_g']:.2f}/kW-h -> {r['ratio']:.2f}x | breakeven launch ${r['breakeven_launch']:,.0f}/kg")
        print(f"  SA units: ${sa['gpu_hr_s']:.2f} vs ${sa['gpu_hr_g']:.2f}/GPU-hr | ${sa['pflop_hr_s']:.2f} vs ${sa['pflop_hr_g']:.2f}/PFLOP-hr"
              f" | ${sa['btok_s']:.0f} vs ${sa['btok_g']:.0f}/B-tok")
        print(f"  NPV/MW: space ${r['npv_s']:.0f}M, ground ${r['npv_g']:.0f}M")
    print(f"\nSemiAnalysis 2026 anchors: ${SA_ANCHORS['gpu_hr_s']:.2f} vs ${SA_ANCHORS['gpu_hr_g']:.2f}/GPU-hr"
          f" ({SA_ANCHORS['ratio']:.2f}x) | ${SA_ANCHORS['pflop_hr_s']:.2f}/{SA_ANCHORS['pflop_hr_g']:.2f}/PFLOP-hr"
          f" | ${SA_ANCHORS['btok_s']:.0f}/{SA_ANCHORS['btok_g']:.0f}/B-tok")
