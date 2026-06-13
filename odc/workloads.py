"""
odc.workloads — AI workload taxonomy + communication-intensity (Gamma) gate
===========================================================================
The demand side. Different AI workloads have very different value-per-kWh, energy-per-
token, latency tolerance, and — critically for space — communication intensity. Turyshev
(arXiv:2604.27197, 2026) frames the latter as the data a workload must move per unit
energy; an orbital link can only sustain a bounded data rate, so high-Gamma workloads
structurally cannot "close" in orbit regardless of compute economics.

USER DECISION (v3): training + batch inference are the revenue-earning orbital workloads;
latency chat/agents is modelled as ground-only (it can run in orbit but the round-trip +
ground-station handoff degrades its realizable price); embeddings are marginal.

Sources: Epoch AI 2025 (J/token, price deflation ~50x/yr); Luccioni et al. FAccT 2024
(420 cites, per-prompt energy span); Patterson et al. 2021 (1,613 cites, MoE efficiency);
Turyshev 2026 (Gamma data ceiling: 148 / 14.8 / 1.48 GB/kWh at 0.1 / 1 / 10 MW).
"""
from dataclasses import dataclass

# Turyshev communication-intensity ceiling: usable data moved per kWh of IT energy,
# as a function of platform IT power. Above this, the downlink/ISL cannot keep the
# accelerators fed and the workload is link-bound, not compute-bound.
# Anchored points (GB of data per kWh): 148 @0.1 MW, 14.8 @1 MW, 1.48 @10 MW.
# i.e. the ceiling scales ~ 1/power: GAMMA_CEILING_GB_PER_KWH(P_MW) ~= 14.8 / P_MW.
def gamma_ceiling_gb_per_kwh(p_it_mw: float) -> float:
    """Max data (GB) movable per kWh of IT energy at this platform IT power (Turyshev)."""
    return 14.8 / max(p_it_mw, 1e-6)


@dataclass(frozen=True)
class Workload:
    name: str
    value_rank: int            # 1 = highest revenue per kWh-of-IT, 4 = lowest
    j_per_token: float         # representative energy per token (J); 0 if not token-metered
    latency_class: str         # 'insensitive' | 'tolerant' | 'sub-second'
    gamma_gb_per_kwh: float    # data moved per kWh of IT (the workload's comms intensity)
    space_suitable: str        # 'best' | 'good' | 'marginal' | 'worst'
    revenue_in_orbit: bool     # v3: does this workload earn revenue in the orbital model?
    note: str


FRONTIER_TRAINING = Workload(
    "Frontier training", value_rank=2, j_per_token=0.0, latency_class="insensitive",
    gamma_gb_per_kwh=0.5, space_suitable="best", revenue_in_orbit=True,
    note="FLOP-bound; energy 2-6% of cost. Checkpoint/interruption-tolerant; power swings "
         "fine on a dedicated bus. Datasets pre-positioned (one-time uplink ~tens of TB; "
         "weights ~0.7-0.9 TB FP8). Internal TP needs NVLink-class BW but 99% of GPU pairs "
         "carry no traffic. LOW Gamma -> closes in orbit.")

BATCH_INFERENCE = Workload(
    "Batch / offline inference", value_rank=3, j_per_token=0.6, latency_class="tolerant",
    gamma_gb_per_kwh=4.0, space_suitable="good", revenue_in_orbit=True,
    note="Synthetic-data gen, video gen, data curation. Absorbs orbital round-trip + "
         "bandwidth limits (results downlinked async). Matches China Three-Body (>95% "
         "inference) and Starcloud single-GPU reality.")

LATENCY_INFERENCE = Workload(
    "Latency inference (chat / agents)", value_rank=1, j_per_token=1.5, latency_class="sub-second",
    gamma_gb_per_kwh=40.0, space_suitable="worst", revenue_in_orbit=False,
    note="Highest revenue/kWh but worst space fit: sub-second TTFT + 20-50 tok/s, steady "
         "per-request traffic, orbital RT + ground-station handoff degrade UX. Modelled "
         "ground-only. reasoning ~2.5x base J/token.")

EMBEDDINGS = Workload(
    "Embeddings / classification", value_rank=4, j_per_token=0.02, latency_class="tolerant",
    gamma_gb_per_kwh=8.0, space_suitable="marginal", revenue_in_orbit=False,
    note="Very low energy/prompt (0.002-0.007 Wh, Luccioni 2024) but low value/kWh "
         "undermines the orbital business case.")

CATALOG = {w.name: w for w in (
    FRONTIER_TRAINING, BATCH_INFERENCE, LATENCY_INFERENCE, EMBEDDINGS)}

ORBITAL_REVENUE_WORKLOADS = [w for w in CATALOG.values() if w.revenue_in_orbit]


def closes_in_orbit(workload: Workload, p_it_mw: float) -> bool:
    """Does this workload's comms intensity sit under the Turyshev ceiling at this scale?"""
    return workload.gamma_gb_per_kwh <= gamma_ceiling_gb_per_kwh(p_it_mw)


def gamma_headroom(workload: Workload, p_it_mw: float) -> float:
    """Ratio of ceiling to demand. >1 closes; <1 is link-bound. Useful for the feasibility map."""
    return gamma_ceiling_gb_per_kwh(p_it_mw) / max(workload.gamma_gb_per_kwh, 1e-9)


if __name__ == "__main__":
    print("Workload                          rank  J/tok  latency      Gamma  space   orbital-rev")
    for w in CATALOG.values():
        print(f"{w.name:34} {w.value_rank:>4}  {w.j_per_token:5.2f}  {w.latency_class:11} "
              f"{w.gamma_gb_per_kwh:6.1f}  {w.space_suitable:7} {w.revenue_in_orbit}")
    print("\nGamma ceiling (GB/kWh) and headroom for orbital-revenue workloads:")
    for mw in (0.1, 1.0, 10.0, 100.0):
        ceil = gamma_ceiling_gb_per_kwh(mw)
        hr = {w.name.split()[0]: round(gamma_headroom(w, mw), 1) for w in ORBITAL_REVENUE_WORKLOADS}
        print(f"  {mw:6.1f} MW: ceiling {ceil:7.2f} GB/kWh | headroom {hr}")
