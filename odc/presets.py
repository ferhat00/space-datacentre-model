"""
odc.presets — size x era x orbit preset matrix + top-level CLI
==============================================================
Convenience layer that crosses the size ladder (odc.sizes) with the era scenarios
(odc.scenarios) and orbit families (odc.orbits), and a single `summary()` entry point
that prints the whole v3 picture. `python -m odc.presets` is the headline CLI.
"""
from dataclasses import replace
from .core import lcoc_and_npv, power_thermal_mass
from .scenarios import TODAY, EARLY, MATURE, SA26, OPTIMIST, SKEPTIC, SA_ANCHORS, scenario_table
from .sizes import LADDER, ladder_table
from .orbits import DDSS, EQUATORIAL_LEO, HIGH_LEO, MEO, GEO, CATALOG as ORBITS
from .workloads import ORBITAL_REVENUE_WORKLOADS, gamma_ceiling_gb_per_kwh, gamma_headroom


def size_era_matrix(orbit=DDSS):
    """LCOC ratio for each (size rung x era) under a given orbit."""
    eras = [TODAY, EARLY, MATURE]
    out = []
    for sc in eras:
        sc_o = orbit.apply(sc)
        row = {"era": sc.name}
        for s in LADDER:
            r = lcoc_and_npv(sc_o, P_it_MW=s.it_mw)
            row[s.key] = r["ratio"]
        out.append(row)
    return out


def summary():
    print("=" * 78)
    print("ORBITAL DATA CENTRE VIABILITY -- v3 SUMMARY (SemiAnalysis-central default)")
    print("=" * 78)

    print("\n--- 1. Era scenarios (1 MW basis, dawn-dusk SSO) + SA26 reproduction ---")
    print(f"{'scenario':34} {'dry t/MW':>9} {'LCOC s/g':>14} {'ratio':>6} {'$/GPU-hr s/g':>15}")
    for r in scenario_table([TODAY, EARLY, MATURE, SA26]):
        print(f"{r['name']:34} {r['dry_t_MW']:9.1f} "
              f"{r['lcoc_s']:6.2f}/{r['lcoc_g']:<6.2f} {r['ratio']:6.2f} "
              f"{r['gpu_hr_s']:7.2f}/{r['gpu_hr_g']:<6.2f}")
    print(f"   SA anchors (verified, LCOC): ${SA_ANCHORS['gpu_hr_s']}/{SA_ANCHORS['gpu_hr_g']}/GPU-hr "
          f"({SA_ANCHORS['ratio']:.2f}x)")

    print("\n--- 2. Literature brackets (optimist / central / skeptic) ---")
    for sc in (OPTIMIST, MATURE, SKEPTIC):
        r = lcoc_and_npv(sc)
        print(f"   {sc.name:38} LCOC {r['lcoc_s']:6.2f}/{r['lcoc_g']:.2f} -> {r['ratio']:.2f}x")

    print("\n--- 3. Size ladder (MATURE scenario) ---")
    print("   (LCOC ratio is scale-invariant: the core scales linearly per MW. Rung")
    print("    differentiation lives in GPU counts, anchor-vs-model mass, and the Gamma gate.)")
    print(f"{'':2} {'class':26} {'IT':>9} {'#GPU':>9} {'anchor t':>9} {'model t':>9} {'ratio':>6}")
    for r in ladder_table(MATURE):
        it = f"{r['it_mw']*1000:.0f} kW" if r['it_mw'] < 1 else f"{r['it_mw']:.0f} MW"
        ex = next(s for s in LADDER if s.key == r['key']).model_example(MATURE)
        print(f"{r['key']:2} {r['name']:26} {it:>9} {r['gpu_count']:9.0f} "
              f"{r['anchor_mass_t']:9.1f} {r['model_dry_t']:9.1f} {ex['ratio']:6.2f}")

    print("\n--- 4. Orbit families (TODAY, 1 MW) ---")
    print(f"{'orbit':32} {'sun%':>6} {'dose x':>7} {'RTT ms':>7} {'FCC':>5} {'dry t/MW':>9}")
    for o in ORBITS.values():
        pt = power_thermal_mass(o.apply(TODAY))
        print(f"{o.name:32} {o.sun_frac_annual*100:5.0f}% {o.tid_dose_mult:7.1f} "
              f"{o.latency_rtt_ms:7.0f} {str(o.fcc_deorbit_ok):>5} {pt['M_dry']/1e3:9.1f}")

    print("\n--- 5. Workload Gamma gate (does it close in orbit?) ---")
    for mw in (0.1, 1.0, 10.0, 100.0):
        hr = {w.name.split()[0]: f"{gamma_headroom(w, mw):.1f}x" for w in ORBITAL_REVENUE_WORKLOADS}
        print(f"   {mw:6.1f} MW IT: ceiling {gamma_ceiling_gb_per_kwh(mw):6.2f} GB/kWh | headroom {hr}")
    print("   (>1x closes; training+batch are the orbital-revenue workloads by decision.)")

    print("\nBottom line: viability is a CONJUNCTION bet -- launch <=$250/kg AND mass <15 kg/kW")
    print("AND radiators near ~600 W/m^2 AND low-Gamma workloads AND a financing regime that")
    print("does not yet exist. Parity, if it comes, is a 2035-2040 proposition gated on Starship.")


if __name__ == "__main__":
    summary()
