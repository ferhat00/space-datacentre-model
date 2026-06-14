"""
odc.model — object-oriented calibrated physics + finance kernel
===============================================================
This is the SemiAnalysis-calibrated heart of the model, expressed object-orientedly.
``ODCModel`` holds a parameter record (``P``) and an IT-power basis, and its methods carry
the coupled power -> thermal -> mass -> launch -> capex -> LCOC/NPV calculation.

The arithmetic is a *verbatim* port of the v2 procedural kernel: every expression is the
same operation in the same order, so the SA26 reproduction stays within ~2% of the
published anchors and the JS dashboard parity badge stays at 0.000% drift
(both regression-guarded in tests/). The backward-compatible free-function API lives in
``odc.core`` and delegates here.

Verified literature corrections from the 2026 adversarial review (Semantic-Scholar
citation checks + 12 adversarially-verified load-bearing claims) are recorded here as
comments and surfaced as provenance (see odc.provenance) and bracketing scenarios
(see odc.scenarios). They are NOT used to retune the calibrated TODAY/EARLY/MATURE/SA26
presets, because doing so would break the SemiAnalysis calibration the model is anchored
to. Where a corrected value matters, it appears as an optimist/skeptic scenario instead.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .results import PowerThermalMass, SpaceCapex, GroundCapex, SAUnits, LCOCResult

SIGMA = 5.670374419e-8   # Stefan-Boltzmann, W/m^2/K^4
SOLAR_CONST = 1361.0     # W/m^2 at 1 AU

# --- B300 reference for SemiAnalysis cross-checks (30.5 kW / 16 GPU cluster) ---
# VERIFIED (claim b300-power-throughput, partially-correct):
#   * 1.906 kW/GPU is the *cluster* critical-IT per GPU (30.5 kW across 16 GPUs, two
#     servers), per SemiAnalysis "To Boldly Go" (3 Jun 2026). Distinct from the 1.4 kW
#     GB300 NVL72 per-GPU TDP. Both used correctly below.
#   * B300_PFLOPS_FP4 = 15.0 is the published *dense* FP4 spec PER GPU (NVIDIA). It is
#     NOT 1,440/72 (=20, the *sparse* per-GPU rate) and NOT 10.91/0.73. The dense rack
#     is ~1,080 PFLOPS (15 x 72); the marketed 1,440 PFLOPS/rack is sparse. Keep 15.0.
#   * B300_TOKS_PER_S = 5,100 is UNVERIFIABLE in any primary source -> treat the
#     resulting $/B-token as the SOFTEST of the three SA conversions.
B300_KW_PER_GPU   = 30.5 / 16          # 1.906 kW cluster critical IT per GPU
B300_PFLOPS_FP4   = 15.0               # dense FP4 PFLOPS per GPU (NVIDIA published spec)
B300_TOKS_PER_S   = 5100.0             # DeepSeek R1 FP4 disagg (InferenceX) -- soft anchor


@dataclass
class P:
    name: str = "Today (2026, Falcon-class)"
    # --- Orbit / environment ---
    eclipse_min_day: float = 35.0     # longest single eclipse for battery ride-through
                                      # (dawn-dusk SSO: one ~35 min/day eclipse in season, SA)
    eclipse_frac_daily: float = None  # total eclipse fraction of the day for array oversizing.
                                      # None => eclipse_min_day/1440 (DDSS: one eclipse/day, so
                                      # the two coincide; calibrated presets keep None). Set by
                                      # odc.orbits for equatorial/high-LEO where many eclipses/day.
    T_sink: float = 230.0             # effective radiative sink (K)
    # --- Power chain ---
    cell_eff: float = 0.30
    packing: float = 0.85
    pointing: float = 0.97
    degr_rate: float = 0.02           # array degradation /yr
    # NOTE (claim space-pv-specific-power-cost): 110 W/kg is a ROSA-class *spec* value.
    # The only flight-demonstrated large arrays (ISS iROSA) deliver ~75 W/kg at the
    # wing/system level -- so the TODAY case is slightly optimistic. The 350 W/kg Mature
    # target is a not-yet-demonstrated Mega-ROSA-class goal. See odc.scenarios skeptic case.
    sp_array: float = 110.0           # W/kg BOL (ROSA-class spec)
    array_cost_W: float = 30.0        # $/W BOL installed (low-tens today; $4/W is aspirational)
    overhead_frac: float = 0.08       # bus loads as frac of IT
    # --- Energy storage (eclipse ride-through) ---
    batt_Wh_kg: float = 160.0         # space-qualified Li-ion pack level
    batt_cost_kWh: float = 600.0      # $/kWh installed
    batt_dod: float = 0.80            # usable depth of discharge
    batt_rt_eff: float = 0.93         # round-trip efficiency
    # --- Thermal ---
    # NOTE (claim radiator-net-rejection): the per-m^2 rejection implied by (T_rad,T_sink)
    # is the single biggest model sensitivity. Proven floor ~150-200 W/m^2 (ISS @20-27C);
    # optimistic ceiling ~600-630 W/m^2 (coated, edge-on, @20C). Do NOT treat 633 as
    # flight-demonstrated.
    T_rad: float = 320.0
    emissivity: float = 0.90
    fin_eff: float = 0.90
    # NOTE (claim radiator-areal-density): ISS is ~14 kg/m^2 today; ~6 kg/m^2 is a mature
    # target. The "3.5 kg/m^2 MARVL" figure circulating online is a conflation -- MARVL is
    # 3.8 kg/kWe specific mass, not areal density. Use ~14 today / ~6 mature.
    rad_areal_kg_m2: float = 6.0
    rad_cost_m2: float = 3000.0
    # --- IT payload ---
    # NOTE (claim it-mass-density): this is a SYSTEM-level launched mass/kW, not
    # IT-payload-only. Anchors: SpaceX AI1 14.3 kg/kW (peak, ~17 avg), Starcloud 19.5,
    # Gaalema <10 (optimistic), Turyshev 34-59 (independent conservative). See scenarios.
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
    insurance_frac: float = 0.07      # NOTE: v3 reframes this as self-insurance; see odc.finance
    ops_M_MW_yr: float = 1.5
    life_yr: float = 5.0              # station life (SA: 5 yr -> 10 yr post-2032 robotics; FCC 5-yr deorbit caps it)
    # --- Launch ---
    # NOTE (claim launch-today): ~$1,600/kg is the SemiAnalysis $1.4-1.8k/kg midpoint
    # (effective/list). $3,245/kg is a naive list/expendable-capacity derivation, not a
    # quote; SpaceX marginal is ~$629/kg. Mature $250/kg is Starship-reuse-contingent.
    launch_kg: float = 1600.0         # $/kg SSO
    # --- Economics ---
    rev0_kWh: float = 3.00
    rev_decline: float = 0.15         # ~13%/yr realized off-take (Starlink ARPU analog); stress 0.10-0.23
    util: float = 0.90
    wacc_space: float = 0.150         # SA: 15% initial, de-risks to 10.3% in ~10 yr; MOST FRAGILE param
    wacc_ground: float = 0.103        # SA: 7% pre-tax debt, 20% equity, 75/25
    deploy_delay_yr: float = 1.0
    # --- Terrestrial benchmark ---
    g_facility_M_MW: float = 12.0     # grid layer $12-15M/MW (SA layers 2-4: $10-20M+)
    g_facility_life_yr: float = 15.0  # SA: Earth DC facility 15 yr vs space 5 yr
    it_life_yr: float = 5.0           # chips, both sides
    g_pue: float = 1.35               # SA assumption (hyperscale now 1.10-1.15; little headroom for space)
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


class ODCModel:
    """Object-oriented façade over the calibrated kernel.

    Holds a parameter record ``P`` and an IT-power basis (``p_it_mw``). The methods carry
    the coupled physics + finance calculation (arithmetic ported verbatim from v2). Build
    one directly, or with the fluent helpers::

        ODCModel.from_scenario(SA26).evaluate()
        ODCModel(TODAY).with_orbit(EQUATORIAL_LEO).with_size(STATION_16MW).evaluate()

    Results are typed, dict-compatible objects (see ``odc.results``), so ``r.lcoc_s`` and
    ``r["lcoc_s"]`` are equivalent.
    """

    def __init__(self, params: P = None, p_it_mw: float = 1.0):
        self.p = params if params is not None else P()
        self.p_it_mw = p_it_mw

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_scenario(cls, scenario: P, p_it_mw: float = 1.0) -> "ODCModel":
        return cls(scenario, p_it_mw)

    def with_orbit(self, orbit) -> "ODCModel":
        """Return a new model with this orbit's eclipse profile stamped onto P."""
        return ODCModel(orbit.apply(self.p), self.p_it_mw)

    def with_size(self, size_class) -> "ODCModel":
        """Return a new model whose IT-power basis is this size class's it_mw."""
        return ODCModel(self.p, size_class.it_mw)

    def with_power(self, p_it_mw: float) -> "ODCModel":
        return ODCModel(self.p, p_it_mw)

    def replace(self, **overrides) -> "ODCModel":
        """Return a new model with the given P fields overridden (what-if / sliders)."""
        return ODCModel(replace(self.p, **overrides), self.p_it_mw)

    # ------------------------------------------------- physics & mass budget
    def power_thermal_mass(self) -> PowerThermalMass:
        p, P_it_MW = self.p, self.p_it_mw
        sell_kW  = 1000.0 * P_it_MW
        gross_kW = sell_kW * (1 + p.overprovision)
        bus_kW   = gross_kW * (1 + p.overhead_frac)

        # Orbit lighting. ecl_frac is the DAILY eclipse fraction (drives array oversizing);
        # eclipse_min_day is the single longest eclipse (drives battery ride-through). For
        # dawn-dusk SSO these coincide (one eclipse/day); for equatorial/high LEO they differ.
        ecl_frac = p.eclipse_frac_daily if p.eclipse_frac_daily is not None else p.eclipse_min_day / 1440.0
        sunlit   = 1.0 - ecl_frac

        # Battery: ride full bus power through the single longest eclipse
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

        return PowerThermalMass(
            sell_kW=sell_kW, gross_kW=gross_kW, bus_kW=bus_kW, sunlit=sunlit,
            E_batt_kWh=E_batt_kWh, M_batt=M_batt, C_batt=C_batt,
            arr_BOL_kW=arr_BOL_kW, A_array=A_array, M_array=M_array, C_array=C_array,
            Q_kW=Q_kW, q_net_side=q_net, A_rad=A_rad, M_rad=M_rad, C_rad=C_rad,
            M_it=M_it, M_shield=M_shield, M_dry=M_dry)

    # ------------------------------------------------- cost & cash-flow model
    def space_capex(self, pt=None) -> SpaceCapex:
        p, P_it_MW = self.p, self.p_it_mw
        if pt is None:
            pt = self.power_thermal_mass()
        C_it = pt["gross_kW"] * 1000 * p.it_cost_W / 1e6
        C_platform = pt["C_array"] + pt["C_rad"] + pt["C_batt"] + p.avionics_comms_M_MW * P_it_MW
        C_launch = pt["M_dry"] * p.launch_kg / 1e6
        C_int = p.integration_M_MW * P_it_MW
        C_ins = p.insurance_frac * (C_it + C_platform + C_launch)
        total = C_it + C_platform + C_launch + C_int + C_ins
        return SpaceCapex(C_it=C_it, C_platform=C_platform, C_launch=C_launch,
                          C_int=C_int, C_ins=C_ins, total=total)

    def ground_capex(self) -> GroundCapex:
        p, P_it_MW = self.p, self.p_it_mw
        sell_kW = 1000 * P_it_MW
        C_it = sell_kW * (1 + p.g_overprovision) * 1000 * p.it_cost_W / 1e6
        C_fac = p.g_facility_M_MW * P_it_MW
        return GroundCapex(C_it=C_it, C_fac=C_fac, total=C_it + C_fac)

    def annual_compute_kWh(self, side: str) -> float:
        p, P_it_MW = self.p, self.p_it_mw
        sell_kW = 1000 * P_it_MW
        if side == "space":
            ov, avail = p.compute_overhead, p.rad_availability
        else:
            ov, avail = p.g_compute_overhead, p.g_rad_availability
        return sell_kW * 8760 * p.util * (1 - ov) * avail   # sellable compute kW-h / yr

    def evaluate(self, include_delay: bool = False) -> LCOCResult:
        p, P_it_MW = self.p, self.p_it_mw
        pt = self.power_thermal_mass()
        cap_s = self.space_capex(pt)
        cap_g = self.ground_capex()

        kwh_s = self.annual_compute_kWh("space")
        kwh_g = self.annual_compute_kWh("ground")

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
        sa = SAUnits(
            gpu_hr_s = wc_s * B300_KW_PER_GPU,
            gpu_hr_g = wc_g * B300_KW_PER_GPU,
            pflop_hr_s = wc_s * B300_KW_PER_GPU / B300_PFLOPS_FP4,
            pflop_hr_g = wc_g * B300_KW_PER_GPU / B300_PFLOPS_FP4,
            btok_s = wc_s * B300_KW_PER_GPU / (B300_TOKS_PER_S * 3600) * 1e9,
            btok_g = wc_g * B300_KW_PER_GPU / (B300_TOKS_PER_S * 3600) * 1e9,
        )

        return LCOCResult(pt=pt, cap_s=cap_s, cap_g=cap_g, lcoc_s=lcoc_s, lcoc_g=lcoc_g,
                          ratio=lcoc_s/lcoc_g, npv_s=npv_s, npv_g=npv_g,
                          breakeven_launch=L_be, kwh_s=kwh_s, kwh_g=kwh_g,
                          ann_s=ann_s, ann_g=ann_g, sa=sa)


# Domain alias: a configured model IS a spacecraft-vs-ground comparison.
Spacecraft = ODCModel
