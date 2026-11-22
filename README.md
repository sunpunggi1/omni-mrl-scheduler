# `Omni-MRL` (v11.1) / `Omni-Lindy`: Non-Clairvoyant Spatio-Temporal, Bidirectional Hazard, Heterogeneous & Topology-Aware AI GPU Scheduler

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24442169.svg)](https://doi.org/10.5281/zenodo.24442169)

> **Technical Paper**: [paper/omni_lindy_paper.md](./paper/omni_lindy_paper.md) · **DOI**: [10.5281/zenodo.24442169](https://doi.org/10.5281/zenodo.24442169) · **Cite**: [CITATION.cff](./CITATION.cff) · **License**: [MIT](./LICENSE)

**`Omni-MRL` (v11.1)** is a strictly non-clairvoyant (**Zero-ML, Zero-Migration**) AI/GPU cluster scheduler and **Kubernetes v1.34 HTTP Extender prototype** that unifies discrete GPU boundary preservation, inspection-paradox bidirectional hazard gating (DFR/IFR), same-entry cohort Kaplan-Meier MRL reclamation with guaranteed non-negative tail hazard ($\hat{R}'(u) \ge -1$), heterogeneous `Max-of-N` attenuation, and rack-topology locality at amortized $O(N)$ per decision.

---

## Core Findings (Phase 11.1 Unseen Lockbox 30-Seed & 40.5M-Job Audit)

| Evaluation Axis | Fair Baseline Protocol (`Cohort KM · BPack · Dev-10 Frozen`) | `Omni-MRL` (v11.1) Results | Statistical Significance & Honest Trade-offs |
| :--- | :--- | :--- | :--- |
| **Synthetic IFR (`IFR_Uniform`, Lockbox 30)** | `FirstFit` / `L2Norm` / `NILAS-L2` / `v7` / `NILAS-BPack` | Blk **6.444%**, BF Wait **0.127h** | vs `v7` **+12.46% Blk · +52.02% BF** (`p_holm96<=2.6e-4`); `NILAS-BPack(6.251%)` leads Blk (`-3.09%`, `p_holm=0.011`) |
| **Wear-Out (`IFR_Weibull_k2`, Lockbox 30)** | `FirstFit` / `L2Norm` / `NILAS-L2` / `v7` / `NILAS-BPack` | Blk **7.132%** (Rank-1), BF Wait **0.513h** | vs `v7` **+6.10% Blk · +11.71% BF** (`p_holm96<=0.015`); `FirstFit(0.459h)` is faster in BF Wait |
| **Memoryless & DFR (`W4`, `W1`, `W3`, `W2`)** | `FirstFit` / `L2Norm` / `NILAS-BPack` / `LAVA-BPack` | `W4` Blk **6.383%** (`+7.0% - +8.3%`), `W1` **3.699%/0.074h**, `W3` BF **0.021h** | `W1` vs `L2Norm` n.s. (`p_holm=1`); `W3` Blk `LAVA-BPack(0.894%)` min; `W4` BF `FirstFit(0.422h)` min |
| **Pure MRL Ablation (`Suite 1b`, Lockbox 30)** | `Omni-MRL-NoPristineRelax` vs `Omni-Lindy-v7` | `NoRelax`: `IFR_Uniform` **6.336%/0.127h**, `Weibull` **7.039%/0.517h** | Pure MRL contributes **+13.94%/+7.32% Blk** (`p_W<=7.3e-6`); `pristine_relax=0.35` loses `1.3%-1.7%` Blk |
| **Bias-Corrected Dual-Oracle (`Suite 2`)** | Continuous `NILAS-ML-BPack` ($w=2.0$, $\mathbb{E}[\epsilon]=1$) | `IFR` **6.444% / 0.127h**, `W3` **0.956% / 0.021h** | `IFR` $\sigma=0$ Oracle wins (`6.241%`, `p=3.5e-3`); **Zero-ML takes over at $\sigma\ge 0.5$ (BF +24.3%) & $\sigma\ge 1.0$ (Blk +8.7%-+14.3%)** |
| **Disjoint Real Traces (`Suite 3`)** | `Helios` Low-CV 11 · `Helios_MN 20` · `Alibaba` Disjoint 6 | `Helios Low-CV`: **12.641%** (IFR **61.98%**, `+6.46%` vs `v7`) · `Helios_MN`: **11.354h** | `Helios_MN` n.s. at cluster Holm-16 (`p>=0.161`), window-level sig.; `Alibaba GPUSpec33` 6-win favors `LAVA/DVLA` BF (`3.20-3.40h`) |
| **40.5M-Job Empirical Audit (4 Traces, 982 Win)** | `Alibaba GPU-v2026` (`40,522,321`) · `AcmeTrace` · `Helios` · `Alibaba 2023` | All strata DFR (`Alibaba v2026` CV **9.29**; `AcmeTrace` CV **10.16**) | Only **12/982 (1.22%)** windows show `CV < 0.9` — validating global DFR/IFR gating over per-size splitting |
| **K8s HTTP Extender (v11.1 Prototype)** | `[0, 10]` · `spec.resources` · `400m` case-strict · idempotent · UID-isolated | 128-node active-KM adapter **1.621 ms** / HTTP **6.238 ms** | **61/61** tests pass (`-W error`, 0 warnings); live Informer/atomic Gang/MIG E2E left as future work |

---

## Quick Start

### 1. Install & Run 61 Unit/Math/HTTP Tests
Requires only **Python 3.10+ and NumPy** (zero external web frameworks or Go toolchains).

```bash
pip install -e .
python -W error test_suite.py
```

### 2. One-Click Reproducibility & Paper Cross-Verification
```bash
# Quick check (< 25s): 61 tests + active-KM HTTP latency + Alibaba CSV + 10 JSON reports & 251+ paper cells
python reproduce_all.py --mode quick

# Optional: download raw public trace CSVs (Alibaba PAI 2023 / SenseTime Helios)
python download_traces.py

# Full pipeline rerun (download traces + re-execute Phase 1-11)
python reproduce_all.py --mode full
```

### 3. Run the Kubernetes Scheduler HTTP Extender
```bash
# Start local Extender server (default engine: omni_mrl v11.1, port 8888)
python -m k8s_extender.server --host 127.0.0.1 --port 8888 --nodes-per-rack 8 --engine omni_mrl
```

```bash
# Health check & live online metrics (Linux/macOS)
curl http://127.0.0.1:8888/healthz
curl http://127.0.0.1:8888/metrics?format=prometheus
```

```powershell
# PowerShell health check (Windows)
Invoke-RestMethod http://127.0.0.1:8888/healthz
```

### 4. Kubernetes Cluster Manifests (`deploy/k8s/`)
* Extender Deployment & Service: [extender-deployment.yaml](./deploy/k8s/extender-deployment.yaml)
* `kube-scheduler` Extender config: [scheduler-config.yaml](./deploy/k8s/scheduler-config.yaml)
* Container image: [Dockerfile](./Dockerfile)

---

## Repository Structure

| File / Directory | Purpose |
| :--- | :--- |
| [simulator.py](./simulator.py) | Non-clairvoyant dynamic GPU cluster event simulator (fractional, heterogeneous, Gang, cohort tracking) |
| [omni_mrl.py](./omni_mrl.py) | `Omni-MRL` (`v8.0-v11.1`): bidirectional clock-invariant hazard + cohort KM MRL + topology/spec scarcity defense |
| [adaptive_lindy.py](./adaptive_lindy.py) | `Adaptive B-LindyPack` (v5.0) and heterogeneous/topology engine `Omni-Lindy` (v7.0) |
| [bigtech_baselines.py](./bigtech_baselines.py) | Same-entry cohort KM, $C^0$/valid-hazard `tanh` MRL, GPU-adapted baselines (`Google-NILAS`, `Google-LAVA`, `Alibaba-DVLA`), ML oracles |
| [baselines.py](./baselines.py) | Classical & spatial SOTA baselines (`FirstFit`, `BestFit`, `DotProduct`, `L2Norm`, `FGD`, `FGD-Adaptive`, `RackBestFit`) |
| [k8s_extender/](./k8s_extender/__init__.py) | `kube-scheduler` v1 HTTP Extender prototype (`/filter`, `/prioritize`, `/gang_schedule`, `/events/pod_completed`, `/healthz`, `/metrics`) |
| [paper/omni_lindy_paper.md](./paper/omni_lindy_paper.md) | Full technical paper covering Phases 1-11.1, Lockbox 30-seed factorials, 40.5M-job audit, and K8s Extender |
| [run_phase11_eval.py](./run_phase11_eval.py) | Phase 11.1 Lockbox 30-seed, Suite 1b ablation, bias-corrected Dual-Oracle, and independent trace runner |
| [phase10_report.json](./phase10_report.json) | Phase 10 cohort KM, Dual-Oracle, and `101k` vs `Fresh` reproducibility JSON report |
| [phase11_report.json](./phase11_report.json) | Phase 11.1 Lockbox 30-seed raw data, cluster/window Holm stats, Helios low-CV 11 windows, 16-file SHA-256 JSON |
| [empirical_regime_audit.json](./empirical_regime_audit.json) | Stratified DFR & 982-window scan across `Alibaba GPU-v2026` (40.5M), `AcmeTrace Kalos`, `Helios`, and `Alibaba 2023` |
| [reproduce_all.py](./reproduce_all.py) | One-click verification CLI (`--mode quick` / `--mode full`) |
| [test_suite.py](./test_suite.py) | 61 unit, invariant, MRL math, cohort KM, audit, and live K8s v1.34 HTTP integration tests |
| [trace_loader.py](./trace_loader.py) | Parsers for Alibaba 2023 (`1,213 nodes / 6,212 GPUs`) and SenseTime Helios logs |
| [stats_utils.py](./stats_utils.py) | SciPy-free Student-$t$, Wilcoxon signed-rank, binomial sign test, Holm-Bonferroni, and hierarchical cluster bootstrap |
| [CITATION.cff](./CITATION.cff) | Standard CFF v1.2.0 citation metadata |

---

## Citation

```bibtex
@software{sunpunggi1_omni_mrl_2026,
  author  = {sunpunggi1},
  title   = {Omni-MRL (v11.1) / Omni-Lindy: Non-Clairvoyant Spatio-Temporal, Bidirectional Hazard, Heterogeneous, and Topology-Aware AI GPU Cluster Scheduler},
  year    = {2026},
  version = {11.1.0},
  doi     = {10.5281/zenodo.24442169},
  url     = {https://github.com/sunpunggi1/omni-mrl-scheduler}
}
```
