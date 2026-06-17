"""
odc.scenarios — calibrated presets + literature bracketing cases
================================================================
TODAY / EARLY / MATURE / SA26 are ported verbatim from v2 so the SemiAnalysis 2026
reproduction (SA26) stays within ~2% of SA_ANCHORS. OPTIMIST and SKEPTIC are NEW v3
bracketing cases that encode the contested-parameter ends from the 2026 adversarial
review (Turyshev/Cavalier skeptic vs Starcloud/Suncatcher optimist), so the headline
can be reported as a band around the SemiAnalysis-central default.

Run `python -m odc.scenarios` for the comparison table + SA anchors.
"""
from dataclasses import replace
from .core import P, lcoc_and_npv

# ---------------- Calibrated presets (verbatim from v2 — do not retune) ----------------

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

# ---------------- v3 bracketing cases (literature ends, around SA-central) ----------------

# SKEPTIC: Turyshev (34-59 kg/kW) + Cavalier (radiators 65-70% of mass, ~150-200 W/m^2)
# + non-financeable WACC + launch staying high + iROSA-real ~75 W/kg. "Hard unless
# everything breaks right."
SKEPTIC = replace(TODAY, name="Skeptic-central (Turyshev/Cavalier)",
    sp_array=75,                 # iROSA flight reality, not 110 (claim space-pv-... REFUTED 225)
    array_cost_W=35,
    T_rad=305,                   # cool electronics-side -> low rejection floor
    rad_areal_kg_m2=14.0,        # ISS-class today (claim radiator-areal-density)
    rad_cost_m2=3500,
    it_kg_per_kW=40,             # Turyshev 34-59 kg/kW system-level (claim it-mass-density)
    shield_t_per_MW=2.0,
    overprovision=0.25,
    rad_availability=0.92,
    structure_frac=0.20,
    insurance_frac=0.07,
    life_yr=5,
    launch_kg=1600,              # stays high near-term ($500/kg pessimistic floor at best)
    wacc_space=0.20)             # non-repossessable, non-serviceable, FCC-life-capped

# OPTIMIST: Starcloud/Suncatcher/Gaalema bull case. "Parity reachable mid-2030s."
OPTIMIST = replace(MATURE, name="Optimist-central (Starcloud/Suncatcher)",
    sp_array=400,                # Mega-ROSA-class target (unproven)
    array_cost_W=4,
    T_rad=333,                   # ~600-630 W/m^2 optimistic ceiling @20C electronics
    rad_areal_kg_m2=4.0,
    it_kg_per_kW=9,              # Gaalema <10 kg/kW integrated panels
    overprovision=0.10,
    rad_availability=0.98,
    life_yr=10,                  # 10-yr post-2032 (needs FCC regulatory extension)
    launch_kg=100,              # Forethought cost-parity / Suncatcher <=$200 mid-2030s
    wacc_space=0.103)

# SPACEX_2027: SpaceX's own June-2026 vendor-stated roadmap, dated to its public
# "~1 GW/yr by end-2027" target. NOT a calibrated era — a clearly-attributed bull case.
# Honest tension: SpaceX's headline 70 kW/ton (=14.3 kg/kW WHOLE-sat) is LIGHTER than
# this model's optimist subsystem sum (array+radiator+battery+IT+structure), and the
# 2027 date is vendor-stated only — Musk said "grain of salt"; the binding S-1 commits
# to "as early as 2028" and warns orbital compute "may not achieve commercial viability."
SPACEX_2027 = replace(OPTIMIST, name="SpaceX roadmap (~2027, vendor-stated)",
    launch_kg=150,          # Starship at scale (SpaceX thesis; cf. Suncatcher <=$200/kg target)
    it_kg_per_kW=9,         # AI1 mass-aggressive end (70 kW/ton whole-sat ~ optimist floor)
    life_yr=5,              # SpaceX S-1 ~5-yr hardware life + FCC 5-yr deorbit (HURTS economics)
    deploy_delay_yr=0.5,    # near-term: prototypes early 2027
    wacc_space=0.103)       # bull: financeable on SpaceX balance sheet (skeptic case = 0.20)

SCENARIOS = [TODAY, EARLY, MATURE]
BRACKETS = [OPTIMIST, SKEPTIC, SPACEX_2027]

# SemiAnalysis published anchors (2026, B300 30.5 kW cluster) for validation.
# VERIFIED (claim sa-gpu-hr-anchors, confirmed): these are the LCOC figures
# ($10.91/$2.49), NOT the TCO figures ($8.64/$2.37). Calibrate to LCOC.
SA_ANCHORS = dict(gpu_hr_s=10.91, gpu_hr_g=2.49, ratio=10.91/2.49,
                  pflop_hr_s=0.73, pflop_hr_g=0.17, btok_s=590.0, btok_g=135.0,
                  capex_total_s_per_W=4.1e6/30500, capex_total_g_per_W=1.4e6/30500,
                  launch_share_dc_capex=1.6/3.1)


def scenario_table(scenarios=None):
    """Return a list of dict rows summarizing each scenario (for CLI / notebook / export)."""
    scenarios = scenarios or (SCENARIOS + [SA26])
    rows = []
    for sc in scenarios:
        r = lcoc_and_npv(sc)
        pt, cs, sa = r["pt"], r["cap_s"], r["sa"]
        rows.append(dict(
            name=sc.name, launch_kg=sc.launch_kg,
            dry_t_MW=pt["M_dry"] / 1e3,
            capex_s_M=cs["total"], capex_g_M=r["cap_g"]["total"],
            lcoc_s=r["lcoc_s"], lcoc_g=r["lcoc_g"], ratio=r["ratio"],
            gpu_hr_s=sa["gpu_hr_s"], gpu_hr_g=sa["gpu_hr_g"],
            breakeven_launch=r["breakeven_launch"],
            npv_s=r["npv_s"], npv_g=r["npv_g"],
        ))
    return rows


def _print_scenarios(scenarios):
    for sc in scenarios:
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


if __name__ == "__main__":
    print("CALIBRATED PRESETS (SemiAnalysis-central) + SA26 reproduction")
    _print_scenarios(SCENARIOS + [SA26])
    print("\nLITERATURE BRACKETS (optimist / skeptic ends)")
    _print_scenarios(BRACKETS)
    print(f"\nSemiAnalysis 2026 anchors (LCOC basis): ${SA_ANCHORS['gpu_hr_s']:.2f} vs ${SA_ANCHORS['gpu_hr_g']:.2f}/GPU-hr"
          f" ({SA_ANCHORS['ratio']:.2f}x) | ${SA_ANCHORS['pflop_hr_s']:.2f}/{SA_ANCHORS['pflop_hr_g']:.2f}/PFLOP-hr"
          f" | ${SA_ANCHORS['btok_s']:.0f}/{SA_ANCHORS['btok_g']:.0f}/B-tok")
