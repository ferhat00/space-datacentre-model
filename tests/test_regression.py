"""
Regression + invariant tests for the v3 modular model.

The headline guard is SA26: the SemiAnalysis 2026 reproduction must stay within ~2% of
the published anchors. If a refactor breaks this, the model is no longer calibrated.

Run: python -m pytest tests/ -q   (or: python tests/test_regression.py)
"""
from dataclasses import replace
from odc.core import P, power_thermal_mass, lcoc_and_npv
from odc.scenarios import TODAY, EARLY, MATURE, SA26, OPTIMIST, SKEPTIC, SCENARIOS, SA_ANCHORS
from odc.orbits import DDSS, EQUATORIAL_LEO, HIGH_LEO, MEO, GEO
from odc.workloads import (gamma_ceiling_gb_per_kwh, gamma_headroom, closes_in_orbit,
                           ORBITAL_REVENUE_WORKLOADS, FRONTIER_TRAINING, LATENCY_INFERENCE)
from odc.ground_energy import CATALOG as GROUND, fastest_sources
from odc.sizes import LADDER, ladder_table
from odc.hardware import B300, gpus_for_power
from odc.finance import scarcity_npv, non_financeable_case, self_insurance, wacc_trajectory


# ---------------- The calibration guard ----------------

def test_sa26_reproduces_anchors_within_2pct():
    sa = lcoc_and_npv(SA26)["sa"]
    assert abs(sa["gpu_hr_s"] - SA_ANCHORS["gpu_hr_s"]) / SA_ANCHORS["gpu_hr_s"] < 0.02, sa["gpu_hr_s"]
    assert abs(sa["gpu_hr_g"] - SA_ANCHORS["gpu_hr_g"]) / SA_ANCHORS["gpu_hr_g"] < 0.02, sa["gpu_hr_g"]


def test_sa26_pflop_and_token_anchors_reasonable():
    sa = lcoc_and_npv(SA26)["sa"]
    assert abs(sa["pflop_hr_s"] - SA_ANCHORS["pflop_hr_s"]) < 0.05
    assert abs(sa["pflop_hr_g"] - SA_ANCHORS["pflop_hr_g"]) < 0.03
    # $/B-token is the SOFT anchor; allow wider tolerance
    assert abs(sa["btok_s"] - SA_ANCHORS["btok_s"]) / SA_ANCHORS["btok_s"] < 0.05


# ---------------- Physics invariants ----------------

def test_thermal_balance_closes():
    for sc in SCENARIOS + [SA26, OPTIMIST, SKEPTIC]:
        pt = power_thermal_mass(sc)
        assert abs(2 * pt["q_net_side"] * pt["A_rad"] - pt["Q_kW"] * 1000) < 1.0


def test_power_balance_closes():
    for sc in SCENARIOS:
        pt = power_thermal_mass(sc)
        eol = (1 - sc.degr_rate) ** sc.life_yr
        ecl = sc.eclipse_min_day / 1440
        daily = (pt["sunlit"] + ecl / sc.batt_rt_eff) / pt["sunlit"]
        assert abs(pt["arr_BOL_kW"] * eol * sc.pointing - pt["bus_kW"] * daily) < 0.5


def test_battery_sizing_rides_eclipse():
    for sc in SCENARIOS:
        pt = power_thermal_mass(sc)
        assert abs(pt["E_batt_kWh"] * sc.batt_dod - pt["bus_kW"] * sc.eclipse_min_day / 60) < 0.5


def test_launch_price_monotonic():
    assert lcoc_and_npv(replace(TODAY, launch_kg=300))["lcoc_s"] < lcoc_and_npv(TODAY)["lcoc_s"]


def test_t4_radiator_trade():
    hot = power_thermal_mass(replace(TODAY, T_rad=340))
    cold = power_thermal_mass(replace(TODAY, T_rad=300))
    assert cold["A_rad"] > hot["A_rad"]  # cooler radiator needs more area


# ---------------- Backward-compat of the orbit extension ----------------

def test_eclipse_frac_daily_default_is_backward_compatible():
    """A DDSS apply() must not change the calibrated result (eclipse_frac_daily ~ min/1440)."""
    base = power_thermal_mass(TODAY)
    ddss = power_thermal_mass(DDSS.apply(TODAY))
    assert abs(base["M_dry"] - ddss["M_dry"]) / base["M_dry"] < 0.01


def test_equatorial_orbit_costs_more_mass_than_ddss():
    """More daily eclipse -> bigger array + battery -> heavier dry mass."""
    ddss = power_thermal_mass(DDSS.apply(TODAY))["M_dry"]
    equ = power_thermal_mass(EQUATORIAL_LEO.apply(TODAY))["M_dry"]
    assert equ > ddss


def test_orbit_radiation_and_latency_ordering():
    assert GEO.latency_rtt_ms > MEO.latency_rtt_ms > HIGH_LEO.latency_rtt_ms > DDSS.latency_rtt_ms
    assert MEO.tid_dose_mult > GEO.tid_dose_mult > HIGH_LEO.tid_dose_mult > DDSS.tid_dose_mult
    assert DDSS.fcc_deorbit_ok and not HIGH_LEO.fcc_deorbit_ok


# ---------------- Workload Gamma gate ----------------

def test_gamma_ceiling_falls_with_scale():
    assert gamma_ceiling_gb_per_kwh(0.1) > gamma_ceiling_gb_per_kwh(1.0) > gamma_ceiling_gb_per_kwh(10.0)


def test_training_closes_more_easily_than_latency():
    # At 1 MW, low-Gamma training should have more headroom than high-Gamma latency chat.
    assert gamma_headroom(FRONTIER_TRAINING, 1.0) > gamma_headroom(LATENCY_INFERENCE, 1.0)


def test_only_training_and_batch_earn_orbital_revenue():
    names = {w.name for w in ORBITAL_REVENUE_WORKLOADS}
    assert any("training" in n.lower() for n in names)
    assert any("batch" in n.lower() for n in names)
    assert not any("chat" in n.lower() for n in names)


# ---------------- Ground comparator ----------------

def test_ground_sources_have_sane_ordering():
    # Off-grid/gas/fuel-cell beat the grid on time-to-power.
    fast = {s.name for s in fastest_sources(12)}
    assert "Behind-the-meter gas turbine" in fast
    assert "Grid (US industrial, DC-weighted)" not in fast


def test_solar_low_capacity_factor():
    assert GROUND["Utility solar (standalone)"].capacity_factor < 0.3


# ---------------- Size ladder ----------------

def test_size_ladder_spans_six_orders():
    powers = [s.it_mw for s in LADDER]
    assert powers == sorted(powers)
    assert powers[0] <= 0.001 and powers[-1] >= 1000


def test_gpu_count_scales_with_power():
    rows = ladder_table(MATURE)
    counts = [r["gpu_count"] for r in rows]
    assert counts == sorted(counts)


def test_b300_gpus_per_mw_matches_semianalysis():
    # 1 MW / 1.906 kW-per-GPU ~ 525 GPUs (SemiAnalysis cluster basis)
    n = gpus_for_power(B300, 1000)
    assert 500 < n < 550


# ---------------- Finance lenses ----------------

def test_wacc_trajectory_derisks():
    assert wacc_trajectory(2026) > wacc_trajectory(2031) > wacc_trajectory(2036)
    assert abs(wacc_trajectory(2026) - 0.15) < 1e-9
    assert abs(wacc_trajectory(2040) - 0.103) < 1e-9  # clamped to floor


def test_non_financeable_raises_lcoc():
    base = lcoc_and_npv(MATURE)["lcoc_s"]
    nf = lcoc_and_npv(non_financeable_case(MATURE))["lcoc_s"]
    assert nf > base


def test_self_insurance_removes_insurance_line():
    assert self_insurance(TODAY).insurance_frac == 0.0


def test_scarcity_gives_space_a_queue_advantage():
    s = scarcity_npv(MATURE)
    assert s["space_advantage_from_queue"] > 0  # ground's longer queue helps space


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
