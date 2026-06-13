"""
odc.orbits — orbit families
===========================
The orbit choice sets eclipse statistics (-> battery + array sizing), radiation dose
(-> shielding + availability), comms latency (-> which workloads are viable), and drag
(-> reboost propellant). v2 hard-coded a single dawn-dusk SSO; v3 parameterizes the
family and can stamp the right eclipse numbers onto a core.P via `apply()`.

USER DECISION (v3): model all four families — DDSS (default), equatorial/inclined LEO,
higher LEO ~1,200-1,400 km, and MEO/GEO (mainly a "why LEO wins" contrast).

Sources: Suncatcher arXiv:2511.19468 (650 km DDSS, ~99% sun 9 mo/yr, 1.6 Tbps ISL, 15 krad
TID); tether paper arXiv:2512.09044 (~1,400 km regime); FCC FCC-22-74 (5-yr deorbit <=2,000 km,
verified primary); standard Van Allen / orbital-mechanics references for MEO/GEO.
"""
from dataclasses import dataclass, replace
from .core import P


@dataclass(frozen=True)
class Orbit:
    name: str
    altitude_km: float
    eclipse_min_orbit: float    # longest single eclipse (min) -> battery ride-through
    eclipse_frac_daily: float   # total daily eclipse fraction -> array oversizing
    sun_frac_annual: float      # annual-average sunlit fraction (1 - eclipse_frac_daily, ~)
    tid_dose_mult: float        # shielded TID dose relative to DDSS ~550 km baseline (=1.0)
    latency_ms_oneway: float    # representative one-way ground link latency
    reboost_factor: float       # station-keeping/reboost propellant relative to DDSS (=1.0)
    fcc_deorbit_ok: bool        # complies with FCC 5-yr post-mission deorbit without active dewell
    note: str

    @property
    def latency_rtt_ms(self) -> float:
        return 2.0 * self.latency_ms_oneway

    def apply(self, p: P) -> P:
        """Return a copy of p with this orbit's eclipse statistics stamped on."""
        return replace(p, eclipse_min_day=self.eclipse_min_orbit,
                       eclipse_frac_daily=self.eclipse_frac_daily)


# Dawn-dusk sun-synchronous ~550-650 km — the consensus choice. One eclipse/day in season;
# near-continuous sun ~9 mo/yr. Minimal battery, low drag, benign radiation. Baseline (=1.0).
DDSS = Orbit("Dawn-dusk SSO (~550-650 km)", 600, 35.0, 35.0/1440, 0.976,
             tid_dose_mult=1.0, latency_ms_oneway=2.5, reboost_factor=1.0, fcc_deorbit_ok=True,
             note="Suncatcher/SemiAnalysis consensus. ~99% sun 9 mo/yr; battery only rides a "
                  "single ~35 min/day eclipse. Default for all calibrated presets.")

# Equatorial / inclined LEO ~550 km — eclipse every orbit (~15.5 orbits/day), so a large
# DAILY eclipse fraction even though each single eclipse is ~35 min. Big battery + array
# oversizing penalty, but better ground revisit and lower-latitude launch energy.
EQUATORIAL_LEO = Orbit("Equatorial/inclined LEO (~550 km)", 550, 35.0, 0.36, 0.64,
                       tid_dose_mult=1.1, latency_ms_oneway=2.5, reboost_factor=1.05, fcc_deorbit_ok=True,
                       note="~36% of every day in shadow -> array oversized ~1.6x and battery "
                            "still sized for one 35-min eclipse. Eclipse->battery-mass coupling "
                            "is the headline penalty vs DDSS.")

# Higher LEO ~1,200-1,400 km — longer sun fraction, far less drag, but worse radiation
# (inner proton belt) and it COLLIDES with the FCC 5-yr deorbit rule (decay >> 25 yr).
HIGH_LEO = Orbit("Higher LEO (~1,200-1,400 km)", 1300, 33.0, 0.28, 0.72,
                 tid_dose_mult=6.0, latency_ms_oneway=4.0, reboost_factor=0.3, fcc_deorbit_ok=False,
                 note="ASCEND/tether regime. Less reboost, more sun, but ~6x dose and natural "
                      "decay far exceeds 5 yr -> needs active deorbit + regulatory extension "
                      "(stresses the FCC-life term).")

# MEO ~20,000 km — heart of the Van Allen belts. Severe radiation, heavy shielding, but
# small eclipse and ~67 ms one-way latency that kills interactive use. Contrast case.
MEO = Orbit("MEO (~20,000 km)", 20000, 40.0, 0.07, 0.93,
            tid_dose_mult=80.0, latency_ms_oneway=67.0, reboost_factor=0.0, fcc_deorbit_ok=False,
            note="Van Allen core: ~80x dose, demands heavy shielding. ~134 ms RTT rules out "
                 "latency inference. Mostly a 'why not MEO' contrast.")

# GEO ~35,786 km — near-continuous sun except equinox eclipse seasons (~70 min/day for ~3
# weeks twice a year). Single-station coverage, predictable radiation, negligible drag, but
# ~120 ms one-way latency. The classic comms orbit, wrong for low-latency compute.
GEO = Orbit("GEO (~35,786 km)", 35786, 70.0, 0.02, 0.98,
            tid_dose_mult=15.0, latency_ms_oneway=120.0, reboost_factor=0.05, fcc_deorbit_ok=False,
            note="Continuous sun + single-station coverage, but ~240 ms RTT and equinox eclipses "
                 "up to ~70 min drive the battery. On-orbit servicing heritage (MEV) exists here.")

CATALOG = {o.name: o for o in (DDSS, EQUATORIAL_LEO, HIGH_LEO, MEO, GEO)}


if __name__ == "__main__":
    from .core import power_thermal_mass
    from .scenarios import TODAY
    print(f"{'Orbit':32} {'alt km':>7} {'ecl/day':>8} {'sun%':>6} {'dose x':>7} {'RTT ms':>7} {'FCC ok':>7} {'dry t/MW':>9}")
    for o in CATALOG.values():
        pt = power_thermal_mass(o.apply(TODAY))
        print(f"{o.name:32} {o.altitude_km:7.0f} {o.eclipse_frac_daily*100:7.1f}% {o.sun_frac_annual*100:5.0f}% "
              f"{o.tid_dose_mult:7.1f} {o.latency_rtt_ms:7.1f} {str(o.fcc_deorbit_ok):>7} {pt['M_dry']/1e3:9.1f}")
