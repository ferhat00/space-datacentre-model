"""
odc.comms — inter-satellite + space-to-ground links and the Gamma feasibility gate
==================================================================================
Bandwidth, not compute, is the historically binding constraint on orbital compute
(Denby & Lucia ASPLOS 2020, 292 cites). v3 models three things: inter-satellite optical
links (ISL, to make N satellites one logical cluster), space-to-ground downlink (to move
results), and the workload feasibility gate that combines a platform's downlink with a
workload's communication intensity (Turyshev Gamma).

Sources: Suncatcher arXiv:2511.19468 (1.6 Tbps ISL bench demo; ~10 Tbps/link target);
Starlink laser mesh (~100 Gbps/terminal, mature); NASA TBIRD (~200 Gbps optical downlink
record); AIAA optical-ground-station studies (4+ stations for 98-99% availability due to
cloud cover); Turyshev 2026 (Gamma data ceiling).
"""
from dataclasses import dataclass
from .workloads import Workload, gamma_ceiling_gb_per_kwh


@dataclass(frozen=True)
class LinkBudget:
    name: str
    isl_tbps_per_link: float       # inter-satellite optical link capacity per link
    downlink_gbps_per_terminal: float  # space-to-ground optical per terminal
    ground_stations: int           # number of optical ground stations in the network
    per_station_clear_sky: float   # single-station clear-sky availability (cloud-limited)
    terminal_mass_kg: float        # optical terminal mass (per terminal)
    note: str

    @property
    def network_availability(self) -> float:
        """Probability >=1 ground station is clear, assuming independent weather.
        4+ stations push a ~50-70% single-site number to ~98-99% (AIAA OGS studies)."""
        return 1.0 - (1.0 - self.per_station_clear_sky) ** self.ground_stations

    def effective_downlink_gbps(self) -> float:
        """Availability-weighted aggregate downlink across the ground network."""
        return self.downlink_gbps_per_terminal * self.network_availability


# Mature commercial heritage (Starlink-class): proven but modest per-terminal rates.
MATURE_2026 = LinkBudget("Mature 2026 (Starlink-class)",
    isl_tbps_per_link=0.1, downlink_gbps_per_terminal=100.0, ground_stations=4,
    per_station_clear_sky=0.65, terminal_mass_kg=10.0,
    note="~100 Gbps/terminal ISL flight-proven; ~200 Gbps optical downlink demo'd (TBIRD); "
         "4 stations -> ~98% availability.")

# Suncatcher-class target: bench-demonstrated 1.6 Tbps, ~10 Tbps/link aspiration.
SUNCATCHER_TARGET = LinkBudget("Suncatcher target (~2030)",
    isl_tbps_per_link=10.0, downlink_gbps_per_terminal=200.0, ground_stations=6,
    per_station_clear_sky=0.70, terminal_mass_kg=5.0,
    note="1.6 Tbps ISL bench demo (Google 2025); ~10 Tbps/link target; few-kg terminals. "
         "Bisection bandwidth across a large constellation remains an unquantified gap.")

CATALOG = {l.name: l for l in (MATURE_2026, SUNCATCHER_TARGET)}


def downlink_feasible_mw(link: LinkBudget, workload: Workload, util: float = 0.9) -> float:
    """Max platform IT power (MW) whose data egress this link can sustain for `workload`.

    A workload at IT power P_MW running at utilization `util` generates data at
    rate = energy_rate (kW) * Gamma (GB/kWh). The link must carry it. We invert the
    Turyshev ceiling: the ceiling is 14.8/P_MW GB per kWh, and the link's sustainable
    GB/kWh at power P_MW is (effective_downlink Gb/s) translated to GB per kWh of IT.
    Returns the P_MW at which the workload's Gamma demand equals the link's supply.
    """
    # IT energy per hour at P_MW and util = P_MW*1000*util kWh/h.
    # Link supplies effective_downlink_gbps Gb/s = effective/8 GB/s = effective/8*3600 GB/h.
    gb_per_h_supply = link.effective_downlink_gbps() / 8.0 * 3600.0
    # Demand GB/h = (P_MW*1000*util kWh/h) * workload.gamma_gb_per_kwh.
    # Solve P_MW where supply == demand:
    denom = 1000.0 * util * workload.gamma_gb_per_kwh
    return gb_per_h_supply / denom if denom > 0 else float("inf")


def gamma_supported_power_mw(link: LinkBudget, util: float = 0.9) -> float:
    """The platform IT power (MW) at which the Turyshev ceiling equals this link's
    sustainable GB/kWh — a link-agnostic feasibility scale check."""
    gb_per_h = link.effective_downlink_gbps() / 8.0 * 3600.0
    # link GB/kWh = gb_per_h / (P_MW*1000*util); set equal to ceiling 14.8/P_MW -> P cancels.
    # => link GB/kWh*P = 14.8/1 ... solve: gb_per_h/(1000*util) == 14.8  (P cancels)
    # so feasibility is power-independent at the margin; return the implied GB/kWh vs 14.8.
    link_gb_per_kwh_times_p = gb_per_h / (1000.0 * util)
    return link_gb_per_kwh_times_p  # compare against 14.8 (the @1MW ceiling numerator)


if __name__ == "__main__":
    from .workloads import ORBITAL_REVENUE_WORKLOADS
    for link in CATALOG.values():
        print(f"\n{link.name}: net avail {link.network_availability*100:.1f}%, "
              f"eff downlink {link.effective_downlink_gbps():.0f} Gbps")
        for w in ORBITAL_REVENUE_WORKLOADS:
            print(f"   {w.name:34} downlink-feasible up to {downlink_feasible_mw(link, w):8.2f} MW IT")
