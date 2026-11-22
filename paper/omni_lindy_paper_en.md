# `Omni-Lindy` / `Omni-MRL` (v11.1): A Non-Clairvoyant AI Cluster Scheduler Unifying Spatio-Temporal Fragmentation, Heterogeneous Resource Hoarding, Multi-Node Topology, and Bidirectional Hazard Rates

> [한국어 원문 (Korean Version)](./omni_lindy_paper.md)

> **Abstract**
>
> In large-scale AI/GPU clusters where fractional-GPU inference pods (0.1-0.5 GPU) dynamically coexist with multi-GPU and multi-node distributed training jobs (1-64 GPUs), bottlenecks arise not only from spatial fragmentation but also from **spatio-temporal fragmentation driven by wide job-lifetime disparities**. Spatial packers such as `FGD` (USENIX ATC '23) suffer from a **"negative-gradient dispersion trap"** under dynamic arrivals, while lifetime-aware cloud schedulers such as Google `LAVA`/`NILAS` (MLSys '25) and Alibaba `DVLA` (OSDI) rely on ML duration predictors or live migration.
>
> We present **`Omni-Lindy` (v7.0)** and its bidirectional hazard extension **`Omni-MRL` (v11.1)**, a **strictly non-clairvoyant (Zero-ML, Zero-Migration)** framework uniting evolutionary formula discovery, online dual gating, heterogeneous/topology-aware placement, and empirical Mean Residual Life (MRL) synchronized reclamation. The framework integrates: (1) **`B-LindyPack`**, an 8-constant discrete-boundary spatial-temporal scorer; (2) **clock-unit-invariant bidirectional hazard gates ($\alpha, \gamma_{\text{DFR}}, \gamma_{\text{IFR}}$)** driven by online log-correlation $\hat{\rho}_t$ and a renewal-theory Inspection Paradox CV heuristic $\hat{c}_t$; (3) **same-entry cohort Kaplan-Meier estimation** (mitigating steady-state length bias under IFR/fixed-epoch workloads) paired with a $C^0$ boundary-continuous, smooth `tanh`-tail ($C^1$ in $\text{CV}$) MRL model guaranteeing a non-negative tail hazard rate ($\hat{R}'(u) \ge -1$); and (4) heterogeneous `Max-of-N` Lindy attenuation, bounded `Live Spec-Scarcity`, `Rack-Preservation`, and `Tightest-Fit Rack` placement.
>
> Evaluated on **Alibaba PAI 2023 (1,213 nodes, 6,212 physical GPUs)**, **SenseTime Helios (3.36M raw rows, up to 64-GPU multi-node)**, a **multi-trace audit across 4 public datasets including Alibaba GPU-v2026 (40,522,321 jobs) and AcmeTrace Kalos (19,907 jobs)**, and **Phase 11 unseen Lockbox 30-seed factorials (`[20011 + 103i]`)** with expanded Dev tuning grids ($w \in [0.05, 4.0]$), bias-corrected ML noise, and Holm-Bonferroni correction across 96 hypotheses:
> 1. **Unseen Lockbox 30-Seed Full Factorial (`FirstFit`/`NILAS-L2` restored, Pareto frontier)**: With all `Omni-MRL` parameters frozen, `Omni-MRL-v11` enters the **Blocking–BF Wait Pareto frontier on 5 of 6 workloads** (`W2` excluded): in **`IFR_Uniform`**, it beats `v7` by `+12.46%` Blk and `+52.02%` BF Wait ($p_{\text{holm96}} \le 2.57\times 10^{-4}$), ties `NILAS-BPack` in BF Wait (`+1.18%`, n.s.), while trailing `NILAS-BPack` in Blk (`6.251%` vs `6.444%`, `-3.09%`, $p_{\text{holm96}}=0.0114$); in **`IFR_Weibull_k2`**, it achieves rank-1 Blocking (`7.132%`, `+6.10%` Blk / `+11.71%` BF vs `v7`, $p_{\text{holm96}}\le 0.0150$; `FirstFit` remains faster in BF Wait at `0.459h` vs `0.513h`); in **`W4_Memoryless_OOD`**, it outperforms `L2Norm`/`BPack` variants by `+6.97% - +8.32%` Blk ($p_{\text{holm96}}\le 0.0122$); in **`W2_Regime_Shift`**, it beats `NILAS-BPack` by `+4.57%` BF Wait ($p_{\text{holm96}}=1.20\times 10^{-3}$). Pure-mechanism ablation (`Omni-MRL-NoPristineRelax`) confirms the MRL reclamation term alone explains over 100% of the IFR Blk gains (`+13.94%`/`+7.32%` Blk, $p_W \le 7.33\times 10^{-6}$). We openly report unfavorable results (`W1` `L2Norm 3.806%` vs `v11 3.699%`, $p_{\text{holm96}}=1.0$; `W3` `LAVA-BPack 0.894%` vs `v11 0.956%`).
> 2. **Bias-Corrected Dual-Oracle ML Crossover (`IFR_Uniform`)**: Under zero-mean-bias lognormal prediction error ($\mathbb{E}[\exp(N(-\sigma^2/2, \sigma^2))]=1$) and expanded grid ($w_{\text{nilas-ml}}=2.0$), `NILAS-ML` only leads in Blocking at zero error ($\sigma=0.0$: `6.241%` vs `6.444%`, $p_{\text{holm48}}=3.48\times 10^{-3}$), and is **monotonically overtaken by Zero-ML `Omni-MRL-v11` from $\sigma \ge 0.50$ in BF Wait (`+24.31%`, $p_{\text{holm48}}=9.86\times 10^{-5}$) and from $\sigma \ge 1.00$ in both Blocking (`+8.72% - +14.32%`, $p_{\text{holm48}}\le 1.75\times 10^{-4}$) and BF Wait**.
> 3. **Independent Trace Windows & 40.5M-Job Empirical Regime Audit**: On `Helios_MN 20`, `Omni-MRL-v11` wins `4/4` clusters numerically (`GW-JCT 11.354h`) against `L2Norm`/`FirstFit`—not significant at primary cluster level (`n=4`, $p_{T,\text{holm16}}=0.161/0.251$, $p_{W,\text{clu,holm16}}=1.0$), significant only at secondary window level (`n=20`, $p_{W,\text{win,holm16}}=3.05\times 10^{-5}$), and n.s. against `RackBestFit (11.409h)` ($p_{W,\text{win,holm16}}=0.301$). In exploratory post-hoc **11 `Helios` low-CV windows**, `v11` engages the IFR gate (`61.98%`) to prevent `v7`'s Anti-Lindy spike (`Earth_229k: 28.218% -> 19.406%`, `+6.46%` overall Blk), whereas on 6 disjoint **`Alibaba GPUSpec33`** windows `LAVA/DVLA-BPack` beat `v7/v11` in BF Wait (`3.20-3.40h` vs `16.22h`). Across **982 valid disjoint windows over 4 public traces**, all strata—including `Alibaba GPU-v2026` (`40,522,321` jobs, CV `9.29`)—are globally DFR, with only **12 windows (1.22%)** exhibiting `CV < 0.9`.
> 4. **Kubernetes v1.34 Extender Prototype (v11.1)**: Implements official `[0, 10]` priority normalization, annotation/`runningPods` double-deduction prevention, `restartPolicy: Always` native sidecar summing, Pod-level `spec.resources`, case-sensitive memory suffixes (`400m` vs `400M`), strict same-name/different-UID separation, idempotent completion events, unified Pod-UID cohort KM, `NodeCacheCapable` TTL + `resourceVersion` checks, and passes **61 unit/math/HTTP tests (`-W error`, 0 warnings)** (full live `kube-scheduler`/Informer E2E left as future work).

---

## 1. Introduction

Modern GPU datacenters concurrently execute sub-GPU inference pods (`0.1-0.5 GPU`), multi-GPU training jobs, and gang-scheduled distributed workloads (`1-64 GPUs`). Effective scheduling requires preserving **discrete physical boundaries (1-, 4-, and 8-GPU slots)** as well as intra-rack locality for future large jobs while keeping per-decision overhead at $O(N)$.

Existing schedulers expose four structural blind spots:
1. **Time-blind spatial packing**: Kubernetes `MostAllocated(BestFit)`, `L2Norm`, and Microsoft VMM `DotProduct` ignore running job age, mixing transient minutes-long tasks with multi-day stragglers.
2. **Alibaba `FGD`'s dynamic dispersion trap**: Fragmentation Gradient Descent excels on static workloads, yet under dynamic arrivals its virtual-placement assumption breaks down, scattering small pods across idle 8-GPU nodes.
3. **Naive Lindy inversion**: Unbounded logarithmic age heuristics destroy `4-GPU` boundaries when job size correlates positively with runtime (`W1`), and suffer severe Anti-Lindy inversion when durations follow Increasing Failure Rate (IFR) distributions.
4. **Heterogeneous hoarding & cross-rack latency**: On mixed 1/2/4/8-GPU nodes, order statistics (`Max-of-N`) inflate observed maximum node age on 8-GPU hosts ("black-hole hoarding"), causing 342-574h Head-of-Line (HoL) stalls under `gpu_spec` constraints unless bounded scarcity penalties and topology-aware rack preservation are enforced.

We unify solutions to all four challenges without runtime predictors or live migration in **`Omni-Lindy` (v7.0)** and **`Omni-MRL` (v8.0 - v11.1)**.

---

## 2. Background: FGD's "Negative-Gradient Dispersion Trap" & Prior Work

### 2.1 Dynamic Physical GPU Cluster Model
Each node $i \in \{1, \dots, N\}$ exposes $G_i \in \{1, 2, 4, 8\}$ discrete GPU slots $\mathbf{s}_i \in [0, 1]^{G_i}$, residual CPU $c_i^{\text{free}}$ and RAM $r_i^{\text{free}}$, GPU model $m_i$, and rack $\text{rack}(i)$. An arriving job $(g_j, c_j, r_j, \text{spec}_j, k_j)$ must satisfy fractional-slot fit ($g_j < 1$), discrete empty-GPU count ($g_j \ge 1$), `gpu_spec`, and all-or-nothing multi-node Gang ($k_j > 1$) constraints. The scheduler only observes currently running ages $a_j(t) = t - t_j^{\text{start}}$ and completed lifetimes $D_{\text{comp}}$.

### 2.2 FGD Equations and 10-Seed Diagnosis
Evaluating `FGD` Case 1 + Case 2 dynamically (`baseline_results.json`, 10 seeds, Blocking GPU-Hour %):

| Workload (10-Seed Diagnosis) | `FirstFit` | `BestFit` | `DotProduct` | `L2Norm` | `FGD_Static` | `FGD_Adaptive` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `W1_PAI_Mixed` | 6.847% | **4.355%** | 4.426% | 4.420% | 8.815% | 8.527% |
| `W2_Regime_Shift` | 24.670% | 24.143% | 24.175% | 24.075% | 24.767% | **24.010%** |
| `W3_Heavy_Straggler` | 2.631% | 2.886% | 2.894% | **2.329%** | 6.159% | 6.416% |
| `W4_Memoryless_OOD` | 7.637% | 7.626% | 7.889% | **7.554%** | 14.134% | 13.807% |
| `W5_Skewed_Vec_OOD` (CPU/RAM skew) | 9.861% | **8.861%** | 30.834% | 9.015% | 29.657% | 37.833% |

Vectorized `FGD` incurs `310-384 µs` per decision (approx. 40x slower than `BestFit`'s `8 µs`) and collapses to `37.833%` blocking on `W5` and `8.527%` on `W1`.

### 2.3 Positioning vs. Recent Prior Work (2023-2026)

| Prior Work | Core Mechanism | Assumptions (ML / Migration) | Difference in `Omni-Lindy` / `Omni-MRL` |
| :--- | :--- | :--- | :--- |
| **`FGD`** (USENIX ATC '23) | Spatial fragmentation $\Delta F_n$ over workload $\mathcal{M}$ | Representative demand dist. | Suffers dynamic dispersion trap; time-blind |
| **`GFGD`** (2026.10) | Offline genetic search -> online $O(1)$ rules | Reliability guardrails | Evolves spatial weights only; no `Max-of-N` or hazard gate |
| **`LAVA` / `NILAS`** (MLSys '25) | 4-class recycling + continuous prediction + $\Delta T$ cost | **ML duration predictor** | `Omni-MRL` uses **Zero-ML** cohort KM MRL + inspection gate |
| **`DVLA`** (OSDI, Alibaba) | Hierarchical predictor + DAPP + live migration | **ML + live migration** | Distributed GPU training cannot cheaply live-migrate; Zero-ML/Zero-Migration |
| **`DeFragS`** | Timeline & queue rescheduling | Queue/timeline control | Orthogonal queue policy; compatible with `Omni` placement |

---

## 3. `Omni-Lindy` Algorithm Design & Theoretical Properties

### 3.1 Evolutionary Formula Discovery & `B-LindyPack`
Searching 840 candidate formulas revealed `EvoSTF_Dist`'s boundary inversion on `W1` (unbounded $\log(1+\text{max\_age})$ overwhelms pristine `4-GPU` bonus $1.302$ past 18.5h). `B-LindyPack` introduces bounded saturation $\frac{\text{age}}{\text{age}+\tau}$ masked by `(~breaks_4)` with 8 physical constants frozen from Phase 4 onward:

$$
\theta^* = [w_{\text{gpu}}, w_{\text{res}}, w_{\text{imb}}, w_{\text{pristine}}, w_{b4}, w_{\text{whole}}, w_{\text{lindy}}, \tau_{\text{hours}}] = [1.0,\; 0.822,\; 0.161,\; 1.302,\; 0.493,\; 0.344,\; 0.446,\; 7.15]
$$

### 3.2 Bidirectional Dual Gates (`Adaptive B-LindyPack` v5.0)
* **Decoupling gate $\alpha(\hat{\rho}_t)$**: Combines normalized running age-size log correlation $\rho_{\text{run}}$ and completed log correlation $\rho_{\text{comp}}$ via Empirical Bayes shrinkage ($n_{\text{prior}}=12, \rho_{\text{prior}}=0.20$) and EWMA ($\beta=0.35$) around $\rho_c=0.11, s_\rho=0.032$, interpolating smoothly between conservative `4-GPU` preservation $\phi_{\text{safe}}$ (`W1`: $\hat{\rho}=0.321 \Rightarrow \alpha=0.018$) and quarantine isolation $\phi_{\text{quar}}$ (`W3`: $\hat{\rho}=0.010 \Rightarrow \alpha=0.903$).
* **Inspection-Paradox CV gate $\gamma(\hat{c}_t)$**: Uses $c_{\text{insp}} = \sqrt{\max(0, 2\bar{a}_t/\bar{D}_t - 1)}$ and $c_{\text{obs}} = \max(c_{\text{dur}}, c_{\text{insp}})$ through sigmoid $\gamma(\hat{c}_t)$ ($c_0=0.65, s_c=0.08$), shrinking DFR Lindy weights (`0.078` on `IFR_Uniform`).

> **Property 1 (Practical Clock-Unit Invariance).** All ratios $u_i = a_i(t)/\hat{T}_{\text{ref}}$ and variance-normalized Pearson correlations (`1e-24 * dot(x,x)`) cancel clock scaling factors $\lambda$. Empirically verified 100% decision-identical across $\lambda \in [10^{-6}, 10^6]$ and non-binary scales (`1/3, 7.3, 3600`). $\blacksquare$

### 3.3 Heterogeneous & Topology Extensions (`Omni-Lindy` v7.0)
* Node-capacity normalization ($G_i$), pristine scaling $(G_i/G_{\max})w_{\text{pristine}}$, bounded `Live Spec-Scarcity` $P_{\text{spec}}(i) = 0.55\min(\log(1 + d(m_i)/\max(f(m_i), 0.02)), 1.5) \le 0.825 < 1.302$, and `Max-of-N` logarithmic attenuation $\psi_{\text{het}}(i) = 1 / (1 + 0.5\log_2(\max(1, G_i/G_{\min})))$.
* Multi-node `Rack-Preservation` penalty on pure racks for $k=1$ jobs and `Tightest-Fit Rack` selection for $k>1$ Gang requests.

> **Theorem 2 (Homogeneous Identity & Complexity).** On homogeneous single-node clusters ($G_{\max}=G_{\min}$, $\text{spec}=\varnothing$, $k=1$), all heterogeneous/topology terms vanish identically, matching v5.0 to `>= 12` decimal digits at amortized $O(N)$ per decision. $\blacksquare$

---

## 4. Experimental Setup & Real-Trace Audit

* **Alibaba PAI 2023 (`openb_node_list_gpu_node.csv`)**: **1,213 nodes, 6,212 physical GPUs** (**G2** 549x8=4,392; **T4** 387x2+17x4=842; **P100** 3x1+131x2=265; **V100M16** 19x1+28x4+8x8=195; **G3** 39x8=312; **V100M32** 9x4+21x8=204; **A10** 2x1=2 GPUs).
* **SenseTime Helios**: 3,362,981 raw rows across Earth/Saturn/Uranus/Venus; 20 single-node windows and 20 multi-node windows (16-64 GPU Gang, `cross_rack_penalty = 0.08`, cap `0.25`).
* Exact statistical testing (`stats_utils.py`): regularized incomplete-beta Student-$t$, Wilcoxon signed-rank, binomial sign test, Holm-Bonferroni, and hierarchical cluster bootstrap.

| Trace | Raw Rows | Valid GPU Jobs | $\text{Corr}(\log g, \log D)$ | Lifetime CV | Mean / Med / p95 (h) | Fractional GPU % | Multi-GPU/Node % |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: | :---: |
| `Alibaba_Default` | 8,152 | 7,063 | `0.0329` | `15.172` | 7.533 / 0.137 / 4.041 | 43.57% | 1.06% (1-8G) |
| `Alibaba_GPUSpec33` | 8,152 | 7,062 | `0.0333` | `15.171` | 7.534 / 0.137 / 4.042 | 43.57% | 1.05% (1-8G) |
| `Helios_Venus` (Single) | 246,708 | 112,555 | `0.0604` | `5.114` | 3.404 / 0.056 / 14.380 | 0.00% | 42.29% (2-8G) |
| `Helios_Earth` (Single) | 872,886 | 416,350 | `0.1707` | `8.302` | 1.305 / 0.065 / 3.887 | 0.00% | 8.52% (2-8G) |
| `Helios_Saturn` (Single) | 1,753,078 | 662,635 | `0.0974` | `7.285` | 1.281 / 0.034 / 2.811 | 0.00% | 28.01% (2-8G) |
| `Helios_Uranus` (Single) | 490,309 | 303,321 | `0.1542` | `6.532` | 2.133 / 0.072 / 7.240 | 0.00% | 28.10% (2-8G) |
| `Helios_Venus` (Multi <=64G) | 246,708 | 123,685 | `0.0883` | `4.881` | 3.651 / 0.057 / 16.288 | 0.00% | 47.48% (2-64G) |
| `Helios_Earth` (Multi <=64G) | 872,886 | 426,577 | `0.2146` | `7.813` | 1.421 / 0.065 / 4.133 | 0.00% | 10.71% (2-64G) |
| `Helios_Saturn` (Multi <=64G) | 1,753,078 | 697,002 | `0.1539` | `6.708` | 1.457 / 0.034 / 3.616 | 0.00% | 31.56% (2-64G) |
| `Helios_Uranus` (Multi <=64G) | 490,309 | 328,991 | `0.2417` | `5.783` | 2.544 / 0.078 / 10.744 | 0.00% | 33.71% (2-64G) |

---

## 5. Early Evaluation Highlights (v5.0 & v7.0)

* **Synthetic 30-Seed Pooled (`180` runs, `phase5_30seed_report.json`)**: `Adaptive-B-Lindy` beats strongest per-workload SOTA in Blocking (`122/176` wins, `+6.44%`, $p_t=3.02\times 10^{-5}$, $p_W=9.16\times 10^{-8}$) and Backfill Wait (`+4.33%`, $p_W=0.00694$), at `43.86 µs` decision latency on 128 nodes (`6.97x` faster than `FGD_Adaptive`).
* **Alibaba PAI 2023 (`phase7_report.json`)**: `Omni-Lindy` (v7.0) reduces `Alibaba_Default` Blocking from `0.743%` (v5.0) to `0.027%` (`+96.38%`, Stranded `4.51%`, though `L2Norm 0.022%` is comparable in Blk and faster in BF Wait) and `Alibaba_GPUSpec33` Blocking to `0.763%` / BF Wait `49.05h` (vs `DotProduct 330.70h`, `L2Norm 508.87h`).
* **Helios 16-64 GPU Multi-Node (`phase7_report.json`)**: `Omni-Lindy` (v7.0) achieves `18.740%` Blocking, `92.47%` single-rack locality (vs `RackBestFit 88.83%`, `L2Norm 33.59%`), and `11.354h` `Topo GW-JCT`.

---

## 6. Key Ablations Summary

1. **Boundary Inversion (`W1`)**: `EvoSTF_Dist` (`4.338%`) -> `Adaptive-B-Lindy` (`3.319%`, `+23.47%`, $p_t=5\times 10^{-6}$).
2. **IFR CV Gate (`Uniform(1.8, 2.2)`)**: vs `Adaptive-NoCVGate`, rescues `+2.82%` Blk ($p_W=0.0180$) and `+13.97%` BF Wait ($p_t=4.19\times 10^{-4}$).
3. **Heterogeneous Lockup**: Bounded `Live Spec-Scarcity` cuts `GPUSpec33` BF Wait from `151.14h` (`kr=0.01` `342h`) / `573.8h` max HoL wait down to `49.05h`.
4. **Rack Locality**: Single-rack locality climbs from `33.4-33.6%` to `92.5%`.

---

## 7. `Omni-MRL` Formulation & Mathematical Guarantees

For IFR workloads ($\gamma_{\text{IFR}}(\hat{c}_t) > 0$), `Omni-MRL` adds host extension cost computed from the youngest job's normalized age $u_i^{\min}$ via cohort KM MRL $\hat{R}(u_i^{\min})$:

$$
S_{\text{MRL}}(i) = S_{\text{omni}}(i)\Big|_{w'_{\text{pristine}}} + (1.45\,\gamma_{\text{IFR}})\,\min\!\left(\frac{\max(0,\; \hat{d}_{\text{new}}(g) - \hat{R}(u_i^{\min}))}{\max(\hat{d}_{\text{new}}(g), 10^{-6})},\; 1\right)(1 - \mathbb{I}_{\text{pristine}}(i))
$$

### Exact Properties of `EmpiricalSurvivalModel` (v11.1)
1. **Steady-state length-bias mitigation**: Same-entry cohort `recent_cohort` (`maxlen=1,024`) tracks completed and right-censored jobs jointly via Kaplan-Meier, mitigating 1.76-15x length bias (with explicit finite-sample `k_tail=5` caveat: e.g., `Exp(1)` `q50` ratio `≈0.89`, tail `r_cut` underestimated `≈14-33%`).
2. **$C^0$ boundary continuity (not $C^1$ in $u$)**: Left and right limits match at $r_{\text{cut}}$ at $u = u_{\text{cut}}$, while left slope ($-1.0$) and right slope differ.
3. **Smooth `tanh` saturation & guaranteed valid hazard rate ($\hat{R}'(u) \ge -1$)**: For $u > u_{\text{cut}}$, $\hat{R}_{\text{tail}}(u) = r_{\text{cut}} e^{-\lambda_{\text{IFR}}(u - u_{\text{cut}})} + 4 \tanh(\beta_{\text{DFR}}(u - u_{\text{cut}})/4)$, bounded above by $r_{\text{cut}} + 4.0$ and $C^\infty$ smooth within the tail. Crucially, v11.1 caps $\lambda_{\text{IFR}} = \min(\lambda_{\text{IFR,raw}},\, 1/r_{\text{cut}})$, mathematically guaranteeing $\hat{R}_{\text{tail}}'(u) \ge -1$ and non-negative hazard rate $h(u) = (1+\hat{R}'(u))/\hat{R}(u) \ge 0$ everywhere on $(u_{\text{cut}}, \infty)$. At $\text{CV}=1$, tail constancy and $C^1$ smoothness in $\text{CV}$ hold.
4. **Lower-bound-preserving unidentifiable quantile extrapolation**: Sets `quantile_identifiable=False` when $\hat{S}_{\text{KM}}(t_{\max}) > p_{\text{target}}$ and uses exponential tail extrapolation $\ge t_{\max}$.

---

## 8. Phase 11.1: Unseen Lockbox 30-Seed Factorial, Dual-Oracle Crossover & 40.5M-Job Audit (`phase11_report.json`)

*(Google `LAVA`/`NILAS` and Alibaba `DVLA` baselines are GPU-cluster `BPack`-adapted variants inspired by their VM-cloud counterparts.)* On completely unused **Lockbox 30-seeds (`[20011 + 103i]`)** with expanded Dev grid $w \in [0.05, 4.0]$, **all 5 baselines attain interior optima** (`NILAS-BPack=1.0`, `LAVA-BPack=0.2` [grid `0.05: 4.4853`, `0.10: 4.4445`, `0.20: 4.4086`, `0.40: 4.4470`], `DVLA-BPack=0.4`, `LAVA-ML-BPack=1.0`, `NILAS-ML-BPack=2.0` [`3.6850`, neighbors `3.7726`]).

### 8.1 Suite 1 — Lockbox 30-Seed Full 8-Baseline Factorial (`Holm-96`)

*(Cell = GPU-Hour Blocking / Backfill Wait)*

| Workload (Lockbox 30) | IFR Active | `FirstFit` | `L2Norm` | `Google-NILAS-L2` | `BPack-Spatial` | `Google-NILAS-BPack` | `Google-LAVA-BPack` | `Alibaba-DVLA-BPack` | `Omni-Lindy-v7` | **`Omni-MRL-v11`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`IFR_Uniform`** | **88.75%** | 6.943% / 0.163h | 7.251% / 0.250h | 6.365% / 0.137h | 7.282% / 0.255h | **6.251%** / 0.128h | 7.221% / 0.244h | 7.282% / 0.255h | 7.362% / 0.264h | 6.444% / **0.127h** |
| **`IFR_Weibull_k2`** | **85.05%** | 7.495% / **0.459h** | 7.575% / 0.612h | 7.372% / 0.540h | 7.510% / 0.590h | 7.288% / 0.498h | 7.447% / 0.569h | 7.510% / 0.590h | 7.595% / 0.581h | **7.132%** / 0.513h |
| **`W4_Memoryless_OOD`** | 0.00% | 6.602% / **0.422h** | 6.910% / 0.647h | 6.962% / 0.617h | 6.862% / 0.613h | 6.861% / 0.603h | 6.872% / 0.572h | 6.862% / 0.608h | **6.383%** / 0.493h | **6.383%** / 0.493h |
| **`W1_PAI_Mixed`** | 0.02% | 6.743% / 0.131h | 3.806% / 0.077h | 4.055% / 0.094h | 3.710% / 0.078h | 3.978% / 0.081h | 3.855% / 0.077h | 3.857% / 0.080h | **3.699%** / **0.074h** | **3.699%** / **0.074h** |
| **`W3_Heavy_Straggler`** | 0.00% | 1.319% / 0.033h | 1.057% / 0.042h | 1.093% / 0.033h | 0.965% / 0.034h | 0.973% / 0.037h | **0.894%** / 0.036h | 0.983% / 0.039h | 0.956% / **0.021h** | 0.956% / **0.021h** |
| **`W2_Regime_Shift`** | 0.28% | 24.681% / 0.835h | 24.629% / 0.870h | 25.130% / 0.896h | 24.690% / 0.866h | 24.864% / 0.875h | 24.584% / 0.872h | 24.844% / 0.857h | **24.419%** / **0.834h** | 24.450% / 0.835h |

### 8.2 Suite 1b — Pure MRL Mechanism Ablation (Lockbox 30-Seed)

| Workload | `BPack-SpatialOnly` | `Omni-Lindy-v7` | `Omni-MRL-NoPristineRelax` | **`Omni-MRL-v11`** | Pure MRL Gain (`v7 -> NoRelax`) | `pristine_relax` Effect (`NoRelax -> v11`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`IFR_Uniform`** | 7.282% / 0.255h | 7.362% / 0.264h | **6.336%** / 0.127h | 6.444% / **0.127h** | **Blk +13.94%** ($p_W=2.02\times 10^{-6}$, `29/1`) · **BF +52.00%** (`30/0`) | Blk `-1.71%` ($p_W=0.0341$) · BF `+0.04%` (`14/11/5`, $p_W=0.914$) |
| **`IFR_Weibull_k2`** | 7.510% / 0.590h | 7.595% / 0.581h | **7.039%** / 0.517h | 7.132% / **0.513h** | **Blk +7.32%** ($p_W=7.33\times 10^{-6}$, `27/3`) · **BF +11.02%** (`25/5`) | Blk `-1.32%` ($p_W=0.0728$) · BF `+0.78%` ($p_W=0.25$) |

### 8.3 Suite 2 — Bias-Corrected Dual-Oracle ML Crossover (Lockbox 30-Seed, `Holm-48`)

Zero-ML `Omni-MRL-v11`: `W3` `0.956% / 0.0214h`, `IFR_Uniform` `6.444% / 0.1267h`. Under zero-mean-bias lognormal noise ($\mathbb{E}[\exp(N(-\sigma^2/2, \sigma^2))] = 1$), **`NILAS-ML` ($w=2.0$) degrades strictly monotonically on `IFR_Uniform` across $\sigma = 0.0 - 2.0$** (`W3` exhibits minor non-monotone fluctuations):

| Error $\sigma$ | `W3` `LAVA-ML` | `W3` `NILAS-ML` | `IFR` `LAVA-ML` | **`IFR` `NILAS-ML`** (`w=2.0`) | Zero-ML `v11` vs `IFR NILAS-ML` (`bias_corrected`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `sigma=0.00` | 0.906% / 0.0247h | **0.687%** / **0.0148h** | 6.983% / 0.2013h | **6.241%** / **0.1203h** | Blk `-3.27%` ($p_{\text{holm48}}=3.48\times 10^{-3}$ **Oracle wins**) · BF `-5.29%` ($p_{\text{holm48}}=1.0$) |
| `sigma=0.25` | 0.928% / 0.0252h | 0.929% / 0.0166h | 6.817% / 0.1809h | **6.373%** / 0.1334h | Blk `-1.12%` ($p_{\text{holm48}}=1.0$) · BF `+5.06%` ($p_{\text{holm48}}=0.403$) |
| `sigma=0.50` | **0.862%** / 0.0241h | **0.853%** / 0.0259h | 6.909% / 0.1946h | 6.582% / 0.1674h | Blk `+2.10%` ($p_{\text{holm48}}=0.751$) · **BF `+24.31%` ($p_{\text{holm48}}=9.86\times 10^{-5}$ Zero-ML wins)** |
| `sigma=1.00` | 0.965% / 0.0301h | 1.043% / 0.0263h | 6.951% / 0.2171h | 7.060% / 0.2278h | **Blk `+8.72%` ($p_{\text{holm48}}=1.75\times 10^{-4}$)** · **BF `+44.39%` ($p_{\text{holm48}}=8.76\times 10^{-5}$)** |
| `sigma=1.50` | 0.900% / 0.0222h | 0.994% / 0.0266h | 7.116% / 0.2225h | 7.335% / 0.2633h | **Blk `+12.14%` ($p_{\text{holm48}}=8.76\times 10^{-5}$)** · **BF `+51.88%` ($p_{\text{holm48}}=8.76\times 10^{-5}$)** |
| `sigma=2.00` | 1.058% / 0.0306h | 1.102% / 0.0314h | 7.077% / 0.2133h | 7.521% / 0.2834h | **Blk `+14.32%` ($p_{\text{holm48}}=8.76\times 10^{-5}$)** · **BF `+55.29%` ($p_{\text{holm48}}=8.76\times 10^{-5}$)** |

### 8.4 Suite 3 — Disjoint Real-Trace Windows, Cluster Bootstrap & Low-CV Exploratory Analysis

In `Alibaba 2023`, raw CSV creation timestamps span **3,583.8h** (approx. 149 days), the 5%-trimmed active trace spans **757.0h**, and `GPUSpec33` disjoint slices (`[700:6700]`) span **705.0h** (approx. 29.4 days).

| Scheduler (`Helios_MN` 20 Windows) | Blk (%) | GW-JCT (h) | Single-Rack (%) | Primary Cluster ($n=4$) & Secondary Window ($n=20$) Summary vs `v11` |
| :--- | :---: | :---: | :---: | :--- |
| `FirstFit` | 18.952% | 12.142h | 36.88% | Cluster ($n=4$, **4/4**, $p_T=0.0101$, $p_{T,\text{holm16}}=0.161$, $p_{W,\text{clu,holm16}}=1.0$ n.s.); Window ($n=20$): CI `[0.514, 1.115]`, $p_{W,\text{win,holm16}}=3.05\times 10^{-5}$ |
| `L2Norm` | 18.849% | 12.063h | 33.59% | Cluster ($n=4$, **4/4**, $p_T=0.0167$, $p_{T,\text{holm16}}=0.251$, $p_{W,\text{clu,holm16}}=1.0$ n.s.); Window ($n=20$): CI `[0.435, 1.021]`, $p_{W,\text{win,holm16}}=3.05\times 10^{-5}$ |
| `RackBestFit` | 19.164% | 11.409h | 88.83% | Both Cluster ($n=4$, $p_{T,\text{holm16}}=1.0$) & Window ($n=20$, CI `[-0.011, 0.137]`, $p_{W,\text{win,holm16}}=0.301$) n.s. (no equivalence test) |
| `BPack-SpatialOnly` | 19.288% | 11.372h | 92.38% | — |
| `Google-NILAS-BPack` | 19.074% | 11.381h | 91.23% | — |
| `Google-LAVA-BPack` | 19.317% | 11.403h | 92.33% | Cluster ($n=4$, `4/4`, $p_T=0.0265$, $p_{T,\text{holm16}}=0.345$, $p_{W,\text{clu,holm16}}=1.0$) n.s. |
| `Alibaba-DVLA-BPack` | 19.099% | 11.395h | 92.07% | — |
| `Omni-Lindy-v7` | **18.740%** | **11.354h** | 92.47% | Numerically equivalent at 4-cluster level |
| **`Omni-MRL-v11`** | 18.743% | **11.354h** | **92.76%** | Outperforms `L2Norm`/`FirstFit` across all 4 clusters numerically (n.s. after cluster Holm-16) |

| Scheduler (Disjoint / Sensitivity) | `Helios` Real Low-CV 11 (IFR Active 61.98%) | `Alibaba GPUSpec33` Disjoint 6 | `Alibaba GPUSpec33` Scale Sensitivity (3/12 Unique) |
| :--- | :---: | :---: | :---: |
| `FirstFit` | 13.282% / 0.0599h | 1.378% / 20.72h | 1.546% / 286.74h |
| `L2Norm` | 12.641% / 0.0504h | 1.502% / 14.84h | 1.192% / 508.86h |
| `BPack-SpatialOnly` | 12.714% / 0.0503h | 1.158% / 18.30h | 0.963% / 799.32h |
| `Google-NILAS-BPack` | 12.641% / **0.0500h** | 1.236% / 5.62h | 0.825% / 829.95h |
| `Google-LAVA-BPack` | 13.435% / 0.0503h | **1.151%** / **3.20h** | **0.675%** / 212.19h |
| `Alibaba-DVLA-BPack` | **12.530%** / 0.0521h | 1.155% / 3.40h | 0.692% / 700.75h |
| `Omni-Lindy-v7` | 13.515% / 0.0503h | 1.327% / 16.22h | 0.763% / **49.05h** |
| **`Omni-MRL-v11`** | 12.641% / 0.0513h | 1.327% / 16.22h | 0.763% / **49.05h** |

On the exploratory post-hoc `Helios` low-CV 11 windows (`Earth[229000:230500]`), `Omni-MRL-v11` activates its IFR gate (`92.48%`) to contain `v7`'s Anti-Lindy spike from `28.218%` down to `19.406%` (`+6.46%` Blk across 11 windows, `2W/0L/9T`, $p_W=0.50$). On `Alibaba GPUSpec33` disjoint 6 windows, `Google-LAVA-BPack` (`1.151% / 3.20h`) and `Alibaba-DVLA-BPack` (`1.155% / 3.40h`) outperform `v7/v11` (`1.327% / 16.22h`).

### 8.5 Multi-Trace Stratified Hazard Audit (`empirical_regime_audit.json`)

To test whether stratifying by job type or GPU request size reveals widespread IFR behavior in production AI traces, we audited **4 public traces** including **`Alibaba GPU-v2026` (OSDI '26, 40,522,321 jobs)** and **`AcmeTrace Kalos` (19,907 jobs)**:

| Trace Dataset | Valid GPU Jobs | Global Lifetime CV | Stratified CV & `MRL(q75)/MRL(0)` (Regime) |
| :--- | :---: | :---: | :--- |
| **`Alibaba GPU-v2026`** | `40,522,321` | `9.29` (mean `5.15h`, p90 `4.71h`) | `offline_inference` (`32,194,936`, CV=`5.06`, MRL `2.62`), `online_inference` (`5,054,990`, CV=`4.49`, MRL `3.69`), `training` (`2,999,580`, CV=`3.35`, MRL `2.47`), `dev` (`31,708`, CV=`2.48`, MRL `3.17`) — **all strata DFR** |
| **`AcmeTrace Kalos`** | `19,907` (comp `10,762`) | `10.16` (MRL `3.50`) | All `1G`-`>8G` strata DFR (lowest `COMPLETED_8G`: `420` jobs, CV=`1.29`, MRL `1.27`) |
| **`SenseTime Helios` (4 clusters)** | `1,494,861` | `5.11-8.30` (MRL `3.71-3.85`) | `Earth` (`416,350`), `Saturn` (`662,635`), `Uranus` (`303,321`), `Venus` (`112,555`), min stratum `Earth 2_4G` CV=`3.34` (MRL `3.80`) — **all strata DFR** |
| **`Alibaba PAI 2023`** | `7,063` / `7,062` | `15.17` (MRL `3.88`) | Lowest stratum `GPUSpec33 8G` CV=`3.37` (MRL `3.68`) — **all strata DFR** |

All strata remain in the DFR regime (`CV 1.29-18.80`, `MRL(q75)/MRL(0) 1.27-3.93`). Across **982 valid non-overlapping windows** extracted via fixed a-priori rules (`block_size=1,500`, `AcmeTrace=1,000`, `min_conc=16.0` / `2.0`), only **12 windows (1.22%)** show `CV < 0.9` (`Helios_Earth` 9, `Helios_Uranus` 2, `AcmeTrace_Kalos` 1) and **9 windows (0.92%)** show multimodal size/type regime splits. Thus, global DFR/IFR dual gating matches empirical cluster workloads better than per-size hazard splitting.

---

## 9. Kubernetes v1.34 HTTP Scheduler Extender Prototype (v11.1)

Implemented in pure Python (`ThreadingHTTPServer`) + `NumPy` under `k8s_extender/` supporting `/filter`, `/prioritize`, `/gang_schedule`, `/events/pod_completed`, `/healthz`, and `/metrics` (JSON/Prometheus). Key v11.1 protocol fixes include: `[0, 10]` priority scaling, double-deduction prevention across `omni-lindy.io/running-pods` and `runningPods`, strict separation of same-name pods with distinct `uid`, Pod-level `spec.resources`, case-sensitive memory units (`400m` vs `400M`), `restartPolicy: Always` native sidecars, unified Pod-UID cohort KM tracking, fully idempotent `/events/pod_completed` (guarding `recent_completions`, cohort counts, and online EWMA), and `NodeCacheCapable` TTL/`resourceVersion` validation.

* **Live HTTP Latency Probe** (with active KM `mrl_mode == "ifr_nilas"`, 100 calls/scale on Windows loopback; **61/61 tests passing**, `-W error` zero warnings):
  * **64 nodes** (cohort `78`): Adapter **`0.921 ms`** (`921.0 µs`), HTTP mean **`5.527 ms`**, `p95 18.011 ms`.
  * **128 nodes** (cohort `121`): Adapter **`1.621 ms`** (`1,621.4 µs`), HTTP mean **`6.238 ms`**, `p95 21.908 ms` (final `reproduce_all.py` rerun prior to publish measured 64N `0.918 ms / 4.945 ms`, 128N `1.636 ms / 7.817 ms`, well within `kube-scheduler`'s `5s` budget).

---

## 10. Limitations & Future Work

1. **Trace DFR Dominance & Trade-offs**: Empirically, `982` audited windows yield only `12` (`1.22%`) low-CV windows. Quantitative trade-offs remain: `NILAS-BPack` beats `v11` on `IFR_Uniform` Blk (`6.251% < 6.444%`), `FirstFit` leads in BF Wait on `IFR_Weibull_k2` and `W4`, and `LAVA/DVLA-BPack` lead on disjoint `Alibaba GPUSpec33`.
2. **Extreme CPU Bottlenecks & No Preemption**: Adding preemption/checkpoint migration would further improve CPU-bound cases (`cpu050`).
3. **2-Tier Topology Model**: Models intra-rack vs. cross-rack penalties (`0.08`/rack, cap `0.25`) without Fat-Tree hop counts or NVLink/PCIe intra-node topology.
4. **Queue Ordering Orthogonality**: Placement heuristic only; orthogonal to SJF/LAS queue disciplines.
5. **Production K8s E2E Scope**: Prototype-validated via unit/live HTTP tests; a Pod Informer sidecar supplying `runningPods`, atomic multi-pod Gang reservations, multi-replica state persistence, MPS/MIG hardware isolation, and KWOK/live-cluster load testing remain future work.

---

## 11. Artifact & Reproducibility

* **61 unit/math/HTTP/trace-audit tests (`-W error`)**: `python -W error test_suite.py`
* **One-click verification (`16/16 SHA-256`, 10 JSONs, CSV, `251+` table/text cells)**: `python reproduce_all.py --mode quick`
* **Full pipeline rerun**: `python reproduce_all.py --mode full`
* **Extender CLI**: `python -m k8s_extender.server --host 0.0.0.0 --port 8888 --nodes-per-rack 8 --engine omni_mrl`
* **Public Traces**: Alibaba (`cluster-trace-gpu-v2023`, `cluster-trace-gpu-v2026`), SenseTime Helios (`S-Lab/HeliosData`), InternLM AcmeTrace (`trace_kalos.csv`).
