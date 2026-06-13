"""
odc.finance — WACC trajectory, non-financeable case, self-insurance, scarcity NPV
=================================================================================
The 2026 review flagged WACC as the single most fragile model parameter and the
insurance line as structurally mis-specified. This module provides:

  * wacc_trajectory  - WACC declining from an immature start toward a mature floor over
    time, instead of a fixed per-scenario value (SemiAnalysis: 15% -> 10.3% over ~10 yr).
  * non_financeable_case - a P variant at 20%+ WACC reflecting that orbital assets are
    non-repossessable, non-serviceable, and FCC-life-capped (Cape Town Space Protocol not
    in force; all real ODC funding to date is equity).
  * self_insurance - reframes insurance: the space-insurance market (~$300M max/risk,
    ~$0.5-0.7B/yr pool, ~24 insurers) cannot underwrite GW-scale assets, so redundancy IS
    the insurance and uninsurable tail risk sits on the operator balance sheet.
  * scarcity_npv - the time-to-power lens: orbit's value is coming online years before a
    grid-queued ground build (already in core via include_delay; surfaced here as a lens).

Sources: SemiAnalysis 2026 (15->10.3% WACC, verified confirmed); Gallagher/Lexology 2025
(insurance market capacity, verified); HFW / Cape Town Space Protocol; SpaceX S-1 (May
2026, all-equity Starlink); Forethought 2026 (redundancy overbuild).
"""
from dataclasses import replace
from .core import P, lcoc_and_npv


def wacc_trajectory(year: float, start_wacc: float = 0.150, floor_wacc: float = 0.103,
                    derisk_years: float = 10.0, t0: float = 2026.0) -> float:
    """WACC de-risking linearly from start_wacc (at t0) to floor_wacc over derisk_years.
    SemiAnalysis: 15% in 2026 -> 10.3% by ~2036. Clamped to the floor thereafter."""
    frac = max(0.0, min(1.0, (year - t0) / derisk_years))
    return start_wacc + (floor_wacc - start_wacc) * frac


def with_wacc(p: P, wacc_space: float) -> P:
    return replace(p, wacc_space=wacc_space)


def non_financeable_case(p: P, wacc_space: float = 0.20) -> P:
    """A P variant priced as genuinely non-financeable equity (20%+). Use to stress the
    headline: orbital DCs are non-repossessable + non-serviceable + FCC-life-capped, so a
    bank cannot lend against them and the cost of capital is pure equity."""
    return replace(p, name=p.name + " [non-financeable 20%]", wacc_space=wacc_space)


def self_insurance(p: P, redundancy_premium: float = None) -> P:
    """Reframe insurance as self-insurance via redundancy. The dedicated-insurance market
    cannot underwrite GW-scale risk, so set insurance_frac to ~0 and instead carry the
    risk through the (already substantial) overprovision/redundancy line. Optionally bump
    overprovision to represent the self-insured tail."""
    q = replace(p, insurance_frac=0.0)
    if redundancy_premium is not None:
        q = replace(q, overprovision=p.overprovision + redundancy_premium)
    return q


# Dedicated space-insurance market structural limits (Gallagher/Lexology 2025, verified).
INSURANCE_MARKET = dict(
    max_per_risk_usd=325e6,       # ~$300-325M maximum capacity for a single risk
    annual_pool_usd=0.7e9,        # ~$0.5-0.7B/yr total market premium pool
    insurers=24,
    note="A GW-scale orbital DC ($10B+ asset) exceeds single-risk capacity by ~30x; the "
         "whole-market annual pool could not absorb one total loss. Hence self-insure.",
)


def insurable(asset_value_usd: float) -> bool:
    """Can the dedicated market underwrite this asset on a single risk?"""
    return asset_value_usd <= INSURANCE_MARKET["max_per_risk_usd"]


def scarcity_npv(p: P, P_it_MW: float = 1.0) -> dict:
    """Time-to-power lens: NPV with deployment delays applied (orbit ~1 yr vs ground grid
    queue ~3 yr). Returns both the queued (include_delay) and no-queue NPVs and the delta
    that the grid queue confers on the orbital case."""
    queued = lcoc_and_npv(p, P_it_MW, include_delay=True)
    prompt = lcoc_and_npv(p, P_it_MW, include_delay=False)
    return dict(
        npv_s_queued=queued["npv_s"], npv_g_queued=queued["npv_g"],
        npv_s_prompt=prompt["npv_s"], npv_g_prompt=prompt["npv_g"],
        ground_queue_penalty=prompt["npv_g"] - queued["npv_g"],
        space_advantage_from_queue=(queued["npv_s"] - queued["npv_g"]) -
                                   (prompt["npv_s"] - prompt["npv_g"]),
    )


if __name__ == "__main__":
    from .scenarios import TODAY, MATURE
    print("WACC trajectory (SemiAnalysis 15% -> 10.3% over 10 yr):")
    for yr in (2026, 2028, 2031, 2034, 2036, 2040):
        print(f"   {yr}: {wacc_trajectory(yr)*100:.1f}%")
    print("\nScarcity (time-to-power) lens, MATURE preset:")
    s = scarcity_npv(MATURE)
    for k, v in s.items():
        print(f"   {k:28} ${v:7.1f}M")
    print(f"\nInsurance: single-risk cap ${INSURANCE_MARKET['max_per_risk_usd']/1e6:.0f}M; "
          f"a $10B GW asset insurable? {insurable(10e9)}")
