"""
odc.hardware — GPU / accelerator matrix
=======================================
Power, cost, compute, and radiation-tolerance reference data for the accelerators a
space (or ground) data centre might fly, from embedded edge parts to rack-scale systems.
Used by odc.sizes (to populate the size ladder) and odc.workloads (tokens/s -> $/token).

Sources (2026 adversarial review):
  * B300 / GB300 NVL72 power + PFLOPS: NVIDIA DGX/HGX B300 docs (claim b300-power-throughput,
    verified). 1.4 kW/GPU TDP; rack ~132-142 kW; 15 PFLOPS dense FP4 per GPU (1,440/rack is
    SPARSE). Cluster critical-IT 1.906 kW/GPU per SemiAnalysis (30.5 kW / 16 GPU).
  * H100/H200/B200: NVIDIA datasheets.
  * Trillium TPU: Google Project Suncatcher arXiv:2511.19468 (proton-beam tested to 15 krad TID).
  * Throughput figures are regime-dependent (prefill vs decode vs interactive SLA); treat as
    order-of-magnitude. 5,100 tok/s (B300 DeepSeek-R1 FP4) is a SOFT anchor (unverifiable).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    kw_per_gpu: float          # nominal board/GPU TDP (kW)
    critical_kw_per_gpu: float # cluster critical-IT per GPU incl. non-GPU overhead (kW)
    dense_pflops_fp4: float    # dense FP4 PFLOPS per GPU (0 if not an FP4 part)
    gpus_per_rack: int         # GPUs in the reference rack/tray (1 if standalone)
    rack_kw: float             # reference rack critical power (kW)
    approx_cost_usd: float     # rough acquisition cost per GPU (USD)
    tok_per_s_ref: float       # representative serving throughput tok/s/GPU (soft)
    rad_note: str              # radiation-tolerance note from flight/test data


# Ladder of representative accelerators (edge -> rack-scale).
JETSON_ORIN = GPU("Jetson Orin (embedded)", 0.060, 0.075, 0.0, 1, 0.075, 2_000, 50,
                  "COTS edge SoC; EO-triage class (Denby & Lucia OEC, 292 cites).")
H100 = GPU("NVIDIA H100", 0.700, 0.95, 0.0, 8, 7.6, 27_500, 900,
           "Flown on Starcloud-1 (2 Nov 2025); survived months in LEO. HBM is the weak link from ~2 krad.")
H200 = GPU("NVIDIA H200", 0.700, 0.95, 0.0, 8, 7.6, 31_000, 1_100,
           "Same die as H100, larger HBM3e; HBM remains TID-limiting.")
B200 = GPU("NVIDIA B200", 1.000, 1.30, 9.0, 8, 10.4, 35_000, 3_200,
           "Blackwell; flown on Starcloud-2 plan (~Oct 2026, mixed B200/H100).")
B300 = GPU("NVIDIA B300 (Blackwell Ultra)", 1.400, 1.906, 15.0, 16, 30.5, 40_000, 5_100,
           "SemiAnalysis 'To Boldly Go' basis. 15 PFLOPS DENSE FP4/GPU; 1,440/rack is sparse.")
GB300_NVL72 = GPU("GB300 NVL72 (rack)", 1.400, 1.97, 15.0, 72, 142.0, 40_000, 5_100,
                  "72 Blackwell Ultra in one rack; ~132-142 kW; SpaceX 'AI1' flies ~1 rack/sat at 150 kW peak.")
TRILLIUM_TPU = GPU("Google Trillium TPU v6e", 0.300, 0.40, 0.0, 256, 80.0, 12_000, 1_400,
                   "Proton-beam tested to 15 krad(Si) TID with no hard failures (Suncatcher, 2025, 20 cites).")

CATALOG = {g.name: g for g in (
    JETSON_ORIN, H100, H200, B200, B300, GB300_NVL72, TRILLIUM_TPU)}


def gpus_for_power(gpu: GPU, it_kw: float, basis: str = "critical") -> float:
    """How many GPUs of this type fit in `it_kw` of IT power.
    basis='critical' uses cluster critical-IT/GPU (matches SemiAnalysis IT accounting);
    basis='tdp' uses raw board TDP."""
    per = gpu.critical_kw_per_gpu if basis == "critical" else gpu.kw_per_gpu
    return it_kw / per


def power_for_gpus(gpu: GPU, n: float, basis: str = "critical") -> float:
    """IT power (kW) for n GPUs of this type."""
    per = gpu.critical_kw_per_gpu if basis == "critical" else gpu.kw_per_gpu
    return n * per


if __name__ == "__main__":
    print(f"{'Accelerator':32} {'kW/GPU':>7} {'crit kW':>8} {'FP4 PF':>7} {'rack kW':>8} {'GPUs/MW':>8}")
    for g in CATALOG.values():
        print(f"{g.name:32} {g.kw_per_gpu:7.3f} {g.critical_kw_per_gpu:8.3f} "
              f"{g.dense_pflops_fp4:7.1f} {g.rack_kw:8.1f} {gpus_for_power(g, 1000):8.0f}")
