"""
11단계(Phase 11): Omni-MRL v11.1 완전 신규 Lockbox 30-시드, 전체 비교군 Holm 보정,
편향 분리 ML 오라클, 실트레이스 독립 시간 구간·계층 부트스트랩 및 순수 기여 Ablation 평가.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from adaptive_lindy import AdaptiveBLindyPackScheduler, BPackSpatialOnlyScheduler
from baselines import FirstFitScheduler, L2NormScheduler, RackBestFitScheduler
from bigtech_baselines import (
    AlibabaDVLAGPUScheduler,
    GoogleLAVAClassGPUScheduler,
    GoogleLAVAMLOracleScheduler,
    GoogleNILASGPUScheduler,
    GoogleNILASMLOracleScheduler,
    run_ml_lava_simulation,
)
from confirm_30seeds import generate_ifr_antilindy_ood
from omni_mrl import OmniMRLConfig, OmniMRLScheduler
from run_phase8_eval import generate_ifr_weibull_k2
from simulator import BaseScheduler, ClusterConfig, Job, run_queue_simulation, run_simulation
from stats_utils import cluster_aware_summary_stats, holm_bonferroni_adjust, paired_summary_stats
from trace_loader import (
    extract_helios_multinode_windows,
    load_alibaba_2023,
    load_helios_cluster_multinode,
    scale_heterogeneous_cluster,
    slice_active_trace,
)
from workloads import (
    generate_w1_pai_mixed,
    generate_w2_regime_shift,
    generate_w3_heavy_straggler,
    generate_w4_memoryless_ood,
)

BASE_DIR = Path(__file__).resolve().parent
ALIBABA_DIR = BASE_DIR / "data" / "alibaba_2023"
_HELIOS_NESTED = BASE_DIR / "data" / "helios" / "data"
HELIOS_DIR = _HELIOS_NESTED if (_HELIOS_NESTED / "Earth" / "cluster_log.csv").exists() else (BASE_DIR / "data" / "helios")

# Dev 10-시드 (가중치 선택·동결 전용) vs 완전 신규 Lockbox 30-시드 (사후 튜닝 절대 금지)
DEV_SEEDS_10: List[int] = [5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081]
LOCKBOX_SEEDS_30: List[int] = [20011 + 103 * i for i in range(30)]
DEV_WORKLOADS: List[str] = ["IFR_Uniform", "IFR_Weibull_k2", "W1_PAI_Mixed", "W3_Heavy_Straggler"]
SUITE1_WORKLOADS: List[str] = [
    "IFR_Uniform",
    "IFR_Weibull_k2",
    "W4_Memoryless_OOD",
    "W1_PAI_Mixed",
    "W3_Heavy_Straggler",
    "W2_Regime_Shift",
]

# 경계 잘림 방지를 위해 하한·상한을 모두 확장한 Dev 튜닝 그리드 (Omni-MRL 파라미터는 완전 동결)
DEV_GRIDS: Dict[str, List[float]] = {
    "Google-NILAS-BPack": [0.10, 0.25, 0.45, 0.65, 1.00, 1.50, 2.00, 3.00, 4.00],
    "Google-LAVA-BPack": [0.05, 0.10, 0.20, 0.40, 0.70, 1.00, 1.50, 2.00, 3.00],
    "Alibaba-DVLA-BPack": [0.20, 0.40, 0.70, 1.00, 1.50, 2.00, 3.00],
    "Google-LAVA-ML-BPack": [0.20, 0.40, 0.70, 1.00, 1.50, 2.00, 3.00],
    "Google-NILAS-ML-BPack": [0.10, 0.25, 0.45, 0.65, 1.00, 1.50, 2.00, 3.00, 4.00],
}

# Helios 실제 저/중-CV 혼합 GPU 비중첩 구간 (Earth 10개 + Uranus 1개)
HELIOS_LOW_CV_SLICES: List[Tuple[str, str, int, int]] = [
    ("Earth", "Earth_24k", 24000, 25500),
    ("Earth", "Earth_34k", 34500, 36000),
    ("Earth", "Earth_36k", 36000, 37500),
    ("Earth", "Earth_154k", 154000, 155500),
    ("Earth", "Earth_226k", 226500, 228000),
    ("Earth", "Earth_229k", 229000, 230500),
    ("Earth", "Earth_357k", 357500, 359000),
    ("Earth", "Earth_390k", 390000, 391500),
    ("Earth", "Earth_401k", 401000, 402500),
    ("Earth", "Earth_421k", 421500, 423000),
    ("Uranus", "Uranus_182k", 182000, 183500),
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _gen_workload(w_name: str, seed: int) -> List[Job]:
    if w_name == "IFR_Uniform":
        return generate_ifr_antilindy_ood(seed=seed)
    if w_name == "IFR_Weibull_k2":
        return generate_ifr_weibull_k2(seed=seed)
    if w_name == "W4_Memoryless_OOD":
        return generate_w4_memoryless_ood(seed=seed)
    if w_name == "W1_PAI_Mixed":
        return generate_w1_pai_mixed(seed=seed)
    if w_name == "W3_Heavy_Straggler":
        return generate_w3_heavy_straggler(seed=seed)
    if w_name == "W2_Regime_Shift":
        return generate_w2_regime_shift(seed=seed)
    raise ValueError(f"알 수 없는 워크로드: {w_name}")


def _build_scheduler(
    name: str,
    tuned_weights: Dict[str, float],
    topology_aware: bool = False,
    hetero_aware: bool = True,
) -> BaseScheduler:
    if name == "FirstFit":
        return FirstFitScheduler()
    if name == "L2Norm":
        return L2NormScheduler()
    if name == "RackBestFit":
        return RackBestFitScheduler()
    if name == "BPack-SpatialOnly":
        return BPackSpatialOnlyScheduler(
            name="BPack-SpatialOnly",
            hetero_aware=hetero_aware,
            topology_aware=topology_aware,
        )
    if name == "Google-NILAS-L2":
        return GoogleNILASGPUScheduler(
            name="Google-NILAS-L2",
            topology_aware=topology_aware,
            w_temporal=0.25,
            spatial_base="l2",
            hetero_aware=hetero_aware,
        )
    if name == "Google-NILAS-BPack":
        return GoogleNILASGPUScheduler(
            name="Google-NILAS-BPack",
            topology_aware=topology_aware,
            w_temporal=tuned_weights.get("Google-NILAS-BPack", 1.00),
            spatial_base="bpack",
            hetero_aware=hetero_aware,
        )
    if name == "Google-LAVA-BPack":
        return GoogleLAVAClassGPUScheduler(
            name="Google-LAVA-BPack",
            topology_aware=topology_aware,
            spatial_base="bpack",
            hetero_aware=hetero_aware,
            w_tier=tuned_weights.get("Google-LAVA-BPack", 0.20),
        )
    if name == "Alibaba-DVLA-BPack":
        return AlibabaDVLAGPUScheduler(
            name="Alibaba-DVLA-BPack",
            topology_aware=topology_aware,
            spatial_base="bpack",
            hetero_aware=hetero_aware,
            w_debt=tuned_weights.get("Alibaba-DVLA-BPack", 0.70),
        )
    if name == "Omni-Lindy-v7":
        return AdaptiveBLindyPackScheduler(
            name="Omni-Lindy-v7",
            hetero_aware=hetero_aware,
            topology_aware=topology_aware,
        )
    if name == "Omni-MRL-NoPristineRelax":
        return OmniMRLScheduler(
            mrl_cfg=OmniMRLConfig(pristine_ifr_relax=0.0),
            name="Omni-MRL-NoPristineRelax",
            record_diagnostics=True,
            hetero_aware=hetero_aware,
            topology_aware=topology_aware,
        )
    if name == "Omni-MRL-v11":
        return OmniMRLScheduler(
            name="Omni-MRL-v11",
            record_diagnostics=True,
            hetero_aware=hetero_aware,
            topology_aware=topology_aware,
        )
    raise ValueError(f"알 수 없는 스케줄러: {name}")


def _eval_dev_unit(task: Tuple[str, float, str, int]) -> Tuple[str, float, str, int, float]:
    family, weight, wn, seed = task
    cfg = ClusterConfig(num_nodes=64)
    jobs = _gen_workload(wn, seed)
    if family == "Google-NILAS-BPack":
        sch: BaseScheduler = GoogleNILASGPUScheduler(w_temporal=weight, spatial_base="bpack")
        mb = run_simulation(jobs, sch, config=cfg, workload_name=wn, seed=seed)
        return family, weight, wn, seed, mb.gpu_hour_blocking_rate
    if family == "Google-LAVA-BPack":
        sch = GoogleLAVAClassGPUScheduler(w_tier=weight, spatial_base="bpack")
        mb = run_simulation(jobs, sch, config=cfg, workload_name=wn, seed=seed)
        return family, weight, wn, seed, mb.gpu_hour_blocking_rate
    if family == "Alibaba-DVLA-BPack":
        sch = AlibabaDVLAGPUScheduler(w_debt=weight, spatial_base="bpack")
        mb = run_simulation(jobs, sch, config=cfg, workload_name=wn, seed=seed)
        return family, weight, wn, seed, mb.gpu_hour_blocking_rate

    pred_map = {id(j): float(j.duration) for j in jobs}
    if family == "Google-LAVA-ML-BPack":
        q25, q50, q75 = np.quantile([j.duration for j in jobs], [0.25, 0.50, 0.75])
        sch = GoogleLAVAMLOracleScheduler(
            pred_map=pred_map,
            lc_thresholds=(float(q25), float(q50), float(q75)),
            spatial_base="bpack",
            hetero_aware=True,
            w_tier=weight,
        )
    else:
        sch = GoogleNILASMLOracleScheduler(
            pred_map=pred_map,
            spatial_base="bpack",
            hetero_aware=True,
            w_temporal=weight,
        )
    res = run_ml_lava_simulation(jobs, sch, cfg)
    return family, weight, wn, seed, float(res["blocking"])


def _eval_suite1_unit(task: Tuple[str, int, str, Dict[str, float]]) -> Dict[str, Any]:
    wn, seed, sn, tuned = task
    cfg = ClusterConfig(num_nodes=64)
    jobs = _gen_workload(wn, seed)
    sch = _build_scheduler(sn, tuned, topology_aware=False, hetero_aware=True)
    mb = run_simulation(jobs, sch, config=cfg, workload_name=wn, seed=seed)
    diag = sch.get_diagnostics_summary() if hasattr(sch, "get_diagnostics_summary") else {}
    mbf = run_queue_simulation(
        jobs, sch, config=cfg, workload_name=wn, seed=seed,
        queue_policy="backfill", backfill_window=16, max_hol_bypasses=8,
    )
    return {
        "workload": wn,
        "seed": seed,
        "scheduler": sn,
        "blocking": mb.gpu_hour_blocking_rate,
        "bf_wait": mbf.mean_wait_time,
        "lat_us": mb.avg_decision_us,
        "diag": diag,
    }


def _eval_suite2_unit(task: Tuple[str, int, str, float, str, Dict[str, float]]) -> Dict[str, Any]:
    wn, seed, kind, sigma, bias_mode, tuned = task
    cfg = ClusterConfig(num_nodes=64)
    jobs = _gen_workload(wn, seed)
    if kind == "Omni-MRL-v11":
        sch: BaseScheduler = OmniMRLScheduler(name="Omni-MRL-v11", hetero_aware=True, topology_aware=False)
    elif kind == "BPack-SpatialOnly":
        sch = BPackSpatialOnlyScheduler(name="BPack-SpatialOnly", hetero_aware=True, topology_aware=False)
    else:
        rng = np.random.default_rng(seed * 97 + int(round(sigma * 100)))
        noise = rng.normal(0.0, sigma, size=len(jobs)) if sigma > 0.0 else np.zeros(len(jobs))
        # bias_corrected: E[exp(N(-sigma^2/2, sigma^2))] == 1.0 (평균 편향 없는 순수 예측 분산)
        shift = (-0.5 * sigma * sigma) if (bias_mode == "bias_corrected" and sigma > 0.0) else 0.0
        pred_map = {id(j): max(1e-6, float(j.duration) * math.exp(float(noise[i]) + shift)) for i, j in enumerate(jobs)}
        if kind == "LAVA-ML-BPack":
            q25, q50, q75 = np.quantile([j.duration for j in jobs], [0.25, 0.50, 0.75])
            sch = GoogleLAVAMLOracleScheduler(
                pred_map=pred_map,
                time_unit_hours=1.0,
                name=f"Google-LAVA-ML-BPack({bias_mode},{sigma:.2f})",
                lc_thresholds=(float(q25), float(q50), float(q75)),
                spatial_base="bpack",
                hetero_aware=True,
                topology_aware=False,
                w_tier=tuned.get("Google-LAVA-ML-BPack", 1.00),
            )
        else:
            sch = GoogleNILASMLOracleScheduler(
                pred_map=pred_map,
                time_unit_hours=1.0,
                name=f"Google-NILAS-ML-BPack({bias_mode},{sigma:.2f})",
                spatial_base="bpack",
                hetero_aware=True,
                topology_aware=False,
                w_temporal=tuned.get("Google-NILAS-ML-BPack", 2.00),
            )
    res = run_ml_lava_simulation(jobs, sch, cfg)
    return {
        "workload": wn,
        "seed": seed,
        "kind": kind,
        "sigma": sigma,
        "bias_mode": bias_mode,
        "blocking": res["blocking"],
        "bf_wait": res["backfill_wait"],
    }


def _eval_real_ali_unit(
    task: Tuple[str, int, ClusterConfig, List[Job], str, Dict[str, float]],
) -> Dict[str, Any]:
    sc, idx, cfg, jobs, sn, tuned = task
    sch = _build_scheduler(sn, tuned, topology_aware=True, hetero_aware=True)
    mb = run_simulation(jobs, sch, config=cfg, workload_name=sc, seed=idx)
    diag = sch.get_diagnostics_summary() if hasattr(sch, "get_diagnostics_summary") else {}
    mbf = run_queue_simulation(jobs, sch, config=cfg, workload_name=sc, seed=idx, queue_policy="backfill")
    return {
        "scenario": sc,
        "idx": idx,
        "scheduler": sn,
        "blk_gh": mb.gpu_hour_blocking_rate,
        "bf_wait": mbf.mean_wait_time,
        "gamma_ifr_active_frac": float(diag.get("gamma_ifr_active_frac", 0.0)),
    }


def _eval_real_hel_unit(
    task: Tuple[str, int, str, ClusterConfig, List[Job], str, Dict[str, float], float],
) -> Dict[str, Any]:
    sc, idx, cluster_tag, cfg, jobs, sn, tuned, cross_pen = task
    sch = _build_scheduler(sn, tuned, topology_aware=True, hetero_aware=True)
    mb = run_simulation(jobs, sch, config=cfg, workload_name=sc, seed=idx)
    diag = sch.get_diagnostics_summary() if hasattr(sch, "get_diagnostics_summary") else {}
    mbf = run_queue_simulation(
        jobs, sch, config=cfg, workload_name=sc, seed=idx, queue_policy="backfill", cross_rack_penalty=cross_pen
    )
    return {
        "scenario": sc,
        "idx": idx,
        "cluster": cluster_tag,
        "scheduler": sn,
        "blk_gh": mb.gpu_hour_blocking_rate,
        "blk_locality": mb.single_rack_locality_rate,
        "bft_wait": mbf.mean_wait_time,
        "bft_gw_jct": mbf.gpu_weighted_jct,
        "gamma_ifr_active_frac": float(diag.get("gamma_ifr_active_frac", 0.0)),
    }


def _compute_pareto_frontier(points: Dict[str, Tuple[float, float]]) -> List[str]:
    """(blocking_mean, bf_wait_mean) 쌍이 모두 낮을수록 좋을 때 파레토 비지배(non-dominated) 스케줄러 목록을 반환한다."""
    names = list(points.keys())
    frontier: List[str] = []
    for n in names:
        b_n, w_n = points[n]
        dominated = False
        for m in names:
            if m == n:
                continue
            b_m, w_m = points[m]
            if (b_m <= b_n + 1e-12 and w_m <= w_n + 1e-12) and (b_m < b_n - 1e-12 or w_m < w_n - 1e-12):
                dominated = True
                break
        if not dominated:
            frontier.append(n)
    return frontier


def main() -> None:
    t0 = time.perf_counter()
    n_workers = max(1, min(8, (mp.cpu_count() or 4) - 1))
    print(f"[Phase 11 v11.1 Lockbox & 독립 실트레이스 평가] CPU 워커 {n_workers}개 시작...", flush=True)

    # 0) Step 0: Dev 10-시드 확장 그리드 튜닝 (Omni-MRL 파라미터 완전 동결)
    dev_tasks = [
        (fam, w, wn, s)
        for fam, grid in DEV_GRIDS.items()
        for w in grid
        for wn in DEV_WORKLOADS
        for s in DEV_SEEDS_10
    ]
    print(f"  [0/4] Dev 10-시드 확장 그리드 스윕 ({len(dev_tasks)} 단위)...", flush=True)
    with mp.Pool(processes=n_workers) as pool:
        dev_rows = pool.map(_eval_dev_unit, dev_tasks)

    dev_grid_scores: Dict[str, Dict[str, float]] = {}
    tuned_weights: Dict[str, float] = {}
    interior_optimum: Dict[str, bool] = {}
    for fam, grid in DEV_GRIDS.items():
        w_scores: Dict[str, float] = {}
        best_w = grid[0]
        best_val = float("inf")
        for w in grid:
            vals = [r[4] for r in dev_rows if r[0] == fam and abs(r[1] - w) <= 1e-9]
            mval = float(np.mean(vals))
            w_scores[f"{w:.2f}"] = mval
            if mval < best_val:
                best_val = mval
                best_w = w
        tuned_weights[fam] = float(best_w)
        dev_grid_scores[fam] = w_scores
        interior_optimum[fam] = (best_w > grid[0] + 1e-9) and (best_w < grid[-1] - 1e-9)
    print(f"    -> Dev 동결 가중치: {tuned_weights} | 내부 최적 여부: {interior_optimum}", flush=True)

    # 1) Suite 1: Lockbox 30-시드 전체 비교군 팩토리얼 (FirstFit, NILAS-L2 포함 9종)
    s1_scheds = [
        "FirstFit",
        "L2Norm",
        "BPack-SpatialOnly",
        "Google-NILAS-L2",
        "Google-NILAS-BPack",
        "Google-LAVA-BPack",
        "Alibaba-DVLA-BPack",
        "Omni-Lindy-v7",
        "Omni-MRL-v11",
    ]
    comp_map_keys = {
        "FirstFit": "vs_firstfit",
        "L2Norm": "vs_l2norm",
        "BPack-SpatialOnly": "vs_bpack_spatial_only",
        "Google-NILAS-L2": "vs_google_nilas_l2",
        "Google-NILAS-BPack": "vs_google_nilas_bpack",
        "Google-LAVA-BPack": "vs_google_lava_bpack",
        "Alibaba-DVLA-BPack": "vs_alibaba_dvla_bpack",
        "Omni-Lindy-v7": "vs_omni_lindy_v7",
    }

    tasks_s1 = [(wn, s, sn, tuned_weights) for wn in SUITE1_WORKLOADS for s in LOCKBOX_SEEDS_30 for sn in s1_scheds]
    print(f"  [1/4] Suite 1 Lockbox 30-시드 전체 비교군 팩토리얼 ({len(tasks_s1)} 단위)...", flush=True)
    with mp.Pool(processes=n_workers) as pool:
        rows_s1 = pool.map(_eval_suite1_unit, tasks_s1)

    by_w_s1: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for r in rows_s1:
        by_w_s1.setdefault(r["workload"], {}).setdefault(r["scheduler"], {})[r["seed"]] = r

    suite1_out: Dict[str, Any] = {"workloads": {}, "holm_family_size": 0}
    holm_refs: List[dict] = []
    for wn in SUITE1_WORKLOADS:
        smap = by_w_s1[wn]
        w_res: Dict[str, Any] = {"schedulers": {}, "comparisons": {}, "raw_per_seed": {}}
        for metric in ("blocking", "bf_wait"):
            w_res["schedulers"][metric] = {
                sn: {
                    "mean": float(np.mean([smap[sn][s][metric] for s in LOCKBOX_SEEDS_30])),
                    "std": float(np.std([smap[sn][s][metric] for s in LOCKBOX_SEEDS_30], ddof=1)),
                }
                for sn in s1_scheds
            }
            w_res["raw_per_seed"][metric] = {
                sn: [float(smap[sn][s][metric]) for s in LOCKBOX_SEEDS_30]
                for sn in s1_scheds
            }
            v11_arr = np.array(w_res["raw_per_seed"][metric]["Omni-MRL-v11"], dtype=np.float64)
            comps = {}
            for ref_sn, ckey in comp_map_keys.items():
                ref_arr = np.array(w_res["raw_per_seed"][metric][ref_sn], dtype=np.float64)
                comps[ckey] = paired_summary_stats(ref_arr, v11_arr)
            w_res["comparisons"][metric] = comps
            holm_refs.extend(comps.values())

        pts = {
            sn: (w_res["schedulers"]["blocking"][sn]["mean"], w_res["schedulers"]["bf_wait"][sn]["mean"])
            for sn in s1_scheds
        }
        w_res["pareto_frontier"] = _compute_pareto_frontier(pts)
        diags = [smap["Omni-MRL-v11"][s]["diag"] for s in LOCKBOX_SEEDS_30 if smap["Omni-MRL-v11"][s]["diag"]]
        w_res["diagnostics"] = {
            "gamma_ifr_mean": float(np.mean([d.get("gamma_ifr_mean", 0.0) for d in diags])),
            "gamma_ifr_active_frac": float(np.mean([d.get("gamma_ifr_active_frac", 0.0) for d in diags])),
            "gamma_dfr_mean": float(np.mean([d.get("gamma_dfr_mean", 0.0) for d in diags])),
            "alpha_mean": float(np.mean([d.get("alpha_mean_step", 0.0) for d in diags])),
        }
        suite1_out["workloads"][wn] = w_res

    s1_adj = holm_bonferroni_adjust([c["p_value_wilcoxon"] for c in holm_refs])
    for c, ph in zip(holm_refs, s1_adj):
        c["p_value_wilcoxon_holm96"] = float(ph)
    suite1_out["holm_family_size"] = len(holm_refs)

    # 1b) Suite 1b: MRL 순수 메커니즘 분리 Ablation (IFR_Uniform, IFR_Weibull_k2)
    ab_tasks = [
        (wn, s, "Omni-MRL-NoPristineRelax", tuned_weights)
        for wn in ("IFR_Uniform", "IFR_Weibull_k2")
        for s in LOCKBOX_SEEDS_30
    ]
    print(f"  [1b/4] Suite 1b MRL 순수 메커니즘 분리 Ablation ({len(ab_tasks)} 단위)...", flush=True)
    with mp.Pool(processes=n_workers) as pool:
        rows_ab = pool.map(_eval_suite1_unit, ab_tasks)
    ab_map: Dict[str, Dict[int, dict]] = {}
    for r in rows_ab:
        ab_map.setdefault(r["workload"], {})[r["seed"]] = r

    ablation_out: Dict[str, Any] = {}
    for wn in ("IFR_Uniform", "IFR_Weibull_k2"):
        smap = by_w_s1[wn]
        ab_res: Dict[str, Any] = {}
        for metric in ("blocking", "bf_wait"):
            sp_arr = np.array([smap["BPack-SpatialOnly"][s][metric] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
            v7_arr = np.array([smap["Omni-Lindy-v7"][s][metric] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
            nop_arr = np.array([ab_map[wn][s][metric] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
            v11_arr = np.array([smap["Omni-MRL-v11"][s][metric] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
            ab_res[metric] = {
                "BPack_SpatialOnly_mean": float(np.mean(sp_arr)),
                "Omni_Lindy_v7_mean": float(np.mean(v7_arr)),
                "Omni_MRL_NoPristineRelax_mean": float(np.mean(nop_arr)),
                "Omni_MRL_v11_mean": float(np.mean(v11_arr)),
                "pure_mrl_term_vs_v7": paired_summary_stats(v7_arr, nop_arr),
                "pristine_relax_vs_nopristine": paired_summary_stats(nop_arr, v11_arr),
                "total_v11_vs_v7": paired_summary_stats(v7_arr, v11_arr),
                "raw_per_seed_nopristine": [float(x) for x in nop_arr],
            }
        ablation_out[wn] = ab_res

    # 2) Suite 2: Lockbox 30-시드 Dual-Oracle ML 교차점 (편향 분리 bias_corrected + uncentered)
    sigmas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    tasks_s2: List[Tuple[str, int, str, float, str, Dict[str, float]]] = []
    for wn in ("W3_Heavy_Straggler", "IFR_Uniform"):
        for s in LOCKBOX_SEEDS_30:
            tasks_s2.append((wn, s, "Omni-MRL-v11", -1.0, "none", tuned_weights))
            tasks_s2.append((wn, s, "BPack-SpatialOnly", -1.0, "none", tuned_weights))
            for bm in ("bias_corrected", "uncentered"):
                for sg in sigmas:
                    if bm == "uncentered" and sg == 0.0:
                        continue  # sigma=0.0은 bias_corrected와 수학적으로 동일하므로 재사용
                    tasks_s2.append((wn, s, "LAVA-ML-BPack", sg, bm, tuned_weights))
                    tasks_s2.append((wn, s, "NILAS-ML-BPack", sg, bm, tuned_weights))

    print(f"  [2/4] Suite 2 Lockbox 30-시드 편향분리 Dual-Oracle ML 교차점 ({len(tasks_s2)} 단위)...", flush=True)
    with mp.Pool(processes=n_workers) as pool:
        rows_s2 = pool.map(_eval_suite2_unit, tasks_s2)

    suite2_out: Dict[str, Any] = {}
    for bm in ("bias_corrected", "uncentered"):
        bm_out: Dict[str, Any] = {}
        bm_holm_refs: List[dict] = []
        for wn in ("W3_Heavy_Straggler", "IFR_Uniform"):
            rows_w = [r for r in rows_s2 if r["workload"] == wn]
            om_map = {r["seed"]: r for r in rows_w if r["kind"] == "Omni-MRL-v11"}
            sp_map = {r["seed"]: r for r in rows_w if r["kind"] == "BPack-SpatialOnly"}
            om_blk = np.array([om_map[s]["blocking"] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
            om_bf = np.array([om_map[s]["bf_wait"] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
            sp_blk = np.array([sp_map[s]["blocking"] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
            sp_bf = np.array([sp_map[s]["bf_wait"] for s in LOCKBOX_SEEDS_30], dtype=np.float64)

            lava_sweep: List[Dict[str, Any]] = []
            nilas_sweep: List[Dict[str, Any]] = []
            for sg in sigmas:
                eff_bm = "bias_corrected" if (bm == "uncentered" and sg == 0.0) else bm
                for ml_kind, target_list in (("LAVA-ML-BPack", lava_sweep), ("NILAS-ML-BPack", nilas_sweep)):
                    ml_map = {
                        r["seed"]: r
                        for r in rows_w
                        if r["kind"] == ml_kind and r["bias_mode"] == eff_bm and abs(r["sigma"] - sg) < 1e-6
                    }
                    ml_blk = np.array([ml_map[s]["blocking"] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
                    ml_bf = np.array([ml_map[s]["bf_wait"] for s in LOCKBOX_SEEDS_30], dtype=np.float64)
                    st_blk = paired_summary_stats(ml_blk, om_blk)
                    st_bf = paired_summary_stats(ml_bf, om_bf)
                    bm_holm_refs.extend([st_blk, st_bf])
                    target_list.append(
                        {
                            "sigma": sg,
                            "ml_blocking_mean": float(np.mean(ml_blk)),
                            "ml_bf_wait_mean": float(np.mean(ml_bf)),
                            "zeroml_vs_ml_blocking": st_blk,
                            "zeroml_vs_ml_bf_wait": st_bf,
                            "raw_per_seed_ml_blocking": [float(x) for x in ml_blk],
                            "raw_per_seed_ml_bf_wait": [float(x) for x in ml_bf],
                        }
                    )
            bm_out[wn] = {
                "omni_mrl_v11": {
                    "blocking_mean": float(np.mean(om_blk)),
                    "bf_wait_mean": float(np.mean(om_bf)),
                    "raw_per_seed_blocking": [float(x) for x in om_blk],
                    "raw_per_seed_bf_wait": [float(x) for x in om_bf],
                },
                "bpack_spatial_only": {
                    "blocking_mean": float(np.mean(sp_blk)),
                    "bf_wait_mean": float(np.mean(sp_bf)),
                },
                "lava_ml_bpack_sweep": lava_sweep,
                "nilas_ml_bpack_sweep": nilas_sweep,
            }
        adj = holm_bonferroni_adjust([c["p_value_wilcoxon"] for c in bm_holm_refs])
        for c, ph in zip(bm_holm_refs, adj):
            c["p_value_wilcoxon_holm48"] = float(ph)
        suite2_out[bm] = bm_out

    # 3) Suite 3: 실측 트레이스 독립 표본·계층 통계·실트레이스 IFR·민감도 분석
    node_csv = ALIBABA_DIR / "openb_node_list_gpu_node.csv"
    ali_cfg, ali_def_jobs, _ = load_alibaba_2023(node_csv, ALIBABA_DIR / "openb_pod_list_default.csv", use_gpu_spec=True)
    _, ali_spec_jobs, _ = load_alibaba_2023(node_csv, ALIBABA_DIR / "openb_pod_list_gpuspec33.csv", use_gpu_spec=True)
    ali_def_act, _ = slice_active_trace(ali_def_jobs, 0.05)
    ali_spec_act, _ = slice_active_trace(ali_spec_jobs, 0.05)

    # 3-A: Helios MultiNode (4 클러스터 × 5 윈도우, FirstFit·Google-LAVA-BPack 포함 9종)
    s3_hel_scheds = [
        "FirstFit",
        "L2Norm",
        "RackBestFit",
        "BPack-SpatialOnly",
        "Google-NILAS-BPack",
        "Google-LAVA-BPack",
        "Alibaba-DVLA-BPack",
        "Omni-Lindy-v7",
        "Omni-MRL-v11",
    ]
    hel_tasks: List[Tuple[str, int, str, ClusterConfig, List[Job], str, Dict[str, float], float]] = []
    hel_cache: Dict[str, List[Job]] = {}
    hel_idx = 0
    for cl in ("Venus", "Earth", "Saturn", "Uranus"):
        hj, _ = load_helios_cluster_multinode(HELIOS_DIR / cl / "cluster_log.csv", cl, max_nodes_per_job=8)
        hel_cache[cl] = hj
        wins = extract_helios_multinode_windows(
            hj, num_windows=5, jobs_per_window=3000, target_load=0.85, min_span_hours=2.0, nodes_per_rack=8
        )
        for wc, wj, _ in wins:
            for sn in s3_hel_scheds:
                hel_tasks.append(("Helios_MultiNode_Pooled_20", hel_idx, cl, wc, wj, sn, tuned_weights, 0.08))
            hel_idx += 1

    # 3-B: Helios 실제 저/중-CV 혼합 구간 11개 (탐색적 사례 분석; Google-LAVA-BPack 포함)
    s3_lowcv_scheds = [
        "FirstFit",
        "L2Norm",
        "BPack-SpatialOnly",
        "Google-NILAS-BPack",
        "Google-LAVA-BPack",
        "Alibaba-DVLA-BPack",
        "Omni-Lindy-v7",
        "Omni-MRL-v11",
    ]
    lowcv_tasks: List[Tuple[str, int, str, ClusterConfig, List[Job], str, Dict[str, float], float]] = []
    lowcv_meta: List[Dict[str, Any]] = []
    for l_idx, (cl, label, s_i, e_i) in enumerate(HELIOS_LOW_CV_SLICES):
        sub = hel_cache[cl][s_i:e_i]
        t0_sub = sub[0].arrival_time
        wj = [
            Job(k, j.arrival_time - t0_sub, j.duration, j.gpu, j.cpu, j.ram, j.gpu_spec, j.num_nodes)
            for k, j in enumerate(sub)
        ]
        span = wj[-1].arrival_time
        cgh = sum(j.gpu * max(0.0, min(j.arrival_time + j.duration, span) - j.arrival_time) for j in wj)
        conc = cgh / max(span, 1e-6)
        ds = np.array([j.duration for j in wj], dtype=np.float64)
        cv = float(np.std(ds) / np.mean(ds))
        nn = max(4, int(round(conc / (8.0 * 0.90))))
        w_cfg = ClusterConfig(num_nodes=nn, nodes_per_rack=8)
        lowcv_meta.append({"idx": l_idx, "label": label, "cluster": cl, "num_nodes": nn, "cv": cv, "conc_gpus": conc})
        for sn in s3_lowcv_scheds:
            lowcv_tasks.append(("Helios_Real_LowCV_11", l_idx, cl, w_cfg, wj, sn, tuned_weights, 0.08))

    # 3-C: Alibaba 비중첩 독립 시간 윈도우 (GPUSpec33 6구간 + Default 6구간, 원본 도착 시각 보존)
    s3_ali_scheds = [
        "FirstFit",
        "L2Norm",
        "BPack-SpatialOnly",
        "Google-NILAS-BPack",
        "Google-LAVA-BPack",
        "Alibaba-DVLA-BPack",
        "Omni-Lindy-v7",
        "Omni-MRL-v11",
    ]
    ali_win_tasks: List[Tuple[str, int, ClusterConfig, List[Job], str, Dict[str, float]]] = []
    tot_full_gpus = sum(ali_cfg.node_gpus) if ali_cfg.node_gpus is not None else 6212
    for sc_name, job_src in (("Alibaba_Default_Disjoint_6", ali_def_jobs), ("Alibaba_GPUSpec33_Disjoint_6", ali_spec_jobs)):
        for i in range(6):
            s_pos = 700 + 1000 * i
            sub = job_src[s_pos : s_pos + 1000]
            t0_s = sub[0].arrival_time
            wj = [
                Job(k, j.arrival_time - t0_s, j.duration, j.gpu, j.cpu, j.ram, j.gpu_spec, j.num_nodes)
                for k, j in enumerate(sub)
            ]
            span = wj[-1].arrival_time
            cgh = sum(j.gpu * max(0.0, min(j.arrival_time + j.duration, span) - j.arrival_time) for j in wj)
            conc = cgh / max(span, 1e-6)
            kr = max(0.002, (conc / 0.88) / tot_full_gpus)
            c_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=701 + i, jobs=wj)
            for sn in s3_ali_scheds:
                ali_win_tasks.append((sc_name, i, c_sub, wj, sn, tuned_weights))

    # 3-D: 기존 Alibaba 클러스터 축소 민감도 스윕 (FirstFit 포함 8종 — 민감도 분석 전용 명시)
    ali_sweep_tasks: List[Tuple[str, int, ClusterConfig, List[Job], str, Dict[str, float]]] = []
    ali_sweep_configs: Dict[str, List[Tuple[Tuple[str, int], ...]]] = {
        "Alibaba_Default_Sensitivity": [],
        "Alibaba_GPUSpec33_Sensitivity": [],
    }
    for idx, (kr, s) in enumerate((k, sd) for k in [0.0025, 0.0030, 0.0035, 0.0045] for sd in [101, 202, 303]):
        c_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_def_act)
        sig = tuple(sorted((str(m), int(g)) for m, g in zip(c_sub.node_models or (), c_sub.node_gpus or ())))
        ali_sweep_configs["Alibaba_Default_Sensitivity"].append(sig)
        for sn in s3_ali_scheds:
            ali_sweep_tasks.append(("Alibaba_Default_Sensitivity", idx, c_sub, ali_def_act, sn, tuned_weights))
    for idx, (kr, s) in enumerate((k, sd) for k in [0.0050, 0.0065, 0.0080, 0.0100] for sd in [101, 202, 303]):
        c_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_spec_act)
        sig = tuple(sorted((str(m), int(g)) for m, g in zip(c_sub.node_models or (), c_sub.node_gpus or ())))
        ali_sweep_configs["Alibaba_GPUSpec33_Sensitivity"].append(sig)
        for sn in s3_ali_scheds:
            ali_sweep_tasks.append(("Alibaba_GPUSpec33_Sensitivity", idx, c_sub, ali_spec_act, sn, tuned_weights))

    total_s3 = len(hel_tasks) + len(lowcv_tasks) + len(ali_win_tasks) + len(ali_sweep_tasks)
    print(f"  [3/4] Suite 3 실트레이스 독립 구간·계층 통계·민감도 분석 ({total_s3} 단위)...", flush=True)
    with mp.Pool(processes=n_workers) as pool:
        res_hel = pool.map(_eval_real_hel_unit, hel_tasks)
        res_lowcv = pool.map(_eval_real_hel_unit, lowcv_tasks)
        res_ali_win = pool.map(_eval_real_ali_unit, ali_win_tasks)
        res_ali_swp = pool.map(_eval_real_ali_unit, ali_sweep_tasks)

    # Helios MultiNode 20 계층 + 윈도우 요약
    by_hel: Dict[str, List[dict]] = {sn: [] for sn in s3_hel_scheds}
    for r in res_hel:
        by_hel[r["scheduler"]].append(r)
    for sn in s3_hel_scheds:
        by_hel[sn].sort(key=lambda x: x["idx"])
    hel_stats = {
        sn: {
            "blk_gh_mean": float(np.mean([x["blk_gh"] for x in by_hel[sn]])),
            "blk_locality_mean": float(np.mean([x["blk_locality"] for x in by_hel[sn]])),
            "bft_wait_mean": float(np.mean([x["bft_wait"] for x in by_hel[sn]])),
            "bft_gw_jct_mean": float(np.mean([x["bft_gw_jct"] for x in by_hel[sn]])),
            "gamma_ifr_active_frac": float(np.mean([x["gamma_ifr_active_frac"] for x in by_hel[sn]])),
        }
        for sn in s3_hel_scheds
    }
    clusters_order = ["Venus", "Earth", "Saturn", "Uranus"]
    comps_hel: Dict[str, Any] = {}
    hel_win_refs: List[dict] = []
    hel_clu_refs: List[dict] = []
    for ref_sn in [sn for sn in s3_hel_scheds if sn != "Omni-MRL-v11"]:
        v11_gw_grp = [np.array([x["bft_gw_jct"] for x in by_hel["Omni-MRL-v11"] if x["cluster"] == cl]) for cl in clusters_order]
        ref_gw_grp = [np.array([x["bft_gw_jct"] for x in by_hel[ref_sn] if x["cluster"] == cl]) for cl in clusters_order]
        v11_hb_grp = [np.array([x["blk_gh"] for x in by_hel["Omni-MRL-v11"] if x["cluster"] == cl]) for cl in clusters_order]
        ref_hb_grp = [np.array([x["blk_gh"] for x in by_hel[ref_sn] if x["cluster"] == cl]) for cl in clusters_order]
        cgw = cluster_aware_summary_stats(ref_gw_grp, v11_gw_grp)
        chb = cluster_aware_summary_stats(ref_hb_grp, v11_hb_grp)
        comps_hel[f"gw_jct_vs_{ref_sn}"] = cgw
        comps_hel[f"blk_vs_{ref_sn}"] = chb
        hel_win_refs.extend([cgw["window_level_with_hier_ci"], chb["window_level_with_hier_ci"]])  # type: ignore[arg-type]
        hel_clu_refs.extend([cgw["cluster_level_stats"], chb["cluster_level_stats"]])  # type: ignore[arg-type]

    win_adj = holm_bonferroni_adjust([c["p_value_wilcoxon"] for c in hel_win_refs])
    clu_w_adj = holm_bonferroni_adjust([c["p_value_wilcoxon"] for c in hel_clu_refs])
    clu_t_adj = holm_bonferroni_adjust([c["p_value_t"] for c in hel_clu_refs])
    for c, ph in zip(hel_win_refs, win_adj):
        c["window_p_value_wilcoxon_holm"] = float(ph)
    for c, ph_w, ph_t in zip(hel_clu_refs, clu_w_adj, clu_t_adj):
        c["cluster_p_value_wilcoxon_holm"] = float(ph_w)
        c["cluster_p_value_t_holm"] = float(ph_t)

    # Helios Real Low-CV 11 요약
    by_lowcv: Dict[str, List[dict]] = {sn: [] for sn in s3_lowcv_scheds}
    for r in res_lowcv:
        by_lowcv[r["scheduler"]].append(r)
    for sn in s3_lowcv_scheds:
        by_lowcv[sn].sort(key=lambda x: x["idx"])
    lowcv_stats = {
        sn: {
            "blk_gh_mean": float(np.mean([x["blk_gh"] for x in by_lowcv[sn]])),
            "bft_wait_mean": float(np.mean([x["bft_wait"] for x in by_lowcv[sn]])),
            "gamma_ifr_active_frac": float(np.mean([x["gamma_ifr_active_frac"] for x in by_lowcv[sn]])),
        }
        for sn in s3_lowcv_scheds
    }
    lowcv_v11_vs_v7 = {
        "blk": paired_summary_stats(
            np.array([x["blk_gh"] for x in by_lowcv["Omni-Lindy-v7"]]),
            np.array([x["blk_gh"] for x in by_lowcv["Omni-MRL-v11"]]),
        ),
        "bf_wait": paired_summary_stats(
            np.array([x["bft_wait"] for x in by_lowcv["Omni-Lindy-v7"]]),
            np.array([x["bft_wait"] for x in by_lowcv["Omni-MRL-v11"]]),
        ),
    }

    # Alibaba Disjoint 6 요약
    ali_disjoint_out: Dict[str, Any] = {}
    for sc in ("Alibaba_Default_Disjoint_6", "Alibaba_GPUSpec33_Disjoint_6"):
        by_sn: Dict[str, List[dict]] = {sn: [] for sn in s3_ali_scheds}
        for r in res_ali_win:
            if r["scenario"] == sc:
                by_sn[r["scheduler"]].append(r)
        for sn in s3_ali_scheds:
            by_sn[sn].sort(key=lambda x: x["idx"])
        ali_disjoint_out[sc] = {
            "schedulers": {
                sn: {
                    "blk_gh_mean": float(np.mean([x["blk_gh"] for x in by_sn[sn]])),
                    "bf_wait_mean": float(np.mean([x["bf_wait"] for x in by_sn[sn]])),
                    "gamma_ifr_active_frac": float(np.mean([x["gamma_ifr_active_frac"] for x in by_sn[sn]])),
                }
                for sn in s3_ali_scheds
            },
            "raw_per_unit": {
                sn: [{"idx": x["idx"], "blk_gh": x["blk_gh"], "bf_wait": x["bf_wait"]} for x in by_sn[sn]]
                for sn in s3_ali_scheds
            },
        }

    # Alibaba Sensitivity Sweep 요약 (민감도 분석 전용 명시)
    ali_sens_out: Dict[str, Any] = {}
    for sc in ("Alibaba_Default_Sensitivity", "Alibaba_GPUSpec33_Sensitivity"):
        by_sn = {sn: [] for sn in s3_ali_scheds}
        for r in res_ali_swp:
            if r["scenario"] == sc:
                by_sn[r["scheduler"]].append(r)
        for sn in s3_ali_scheds:
            by_sn[sn].sort(key=lambda x: x["idx"])
        unique_cfgs = len(set(ali_sweep_configs[sc]))
        ali_sens_out[sc] = {
            "sensitivity_analysis_only": True,
            "total_runs": len(ali_sweep_configs[sc]),
            "unique_model_gpu_configs": unique_cfgs,
            "note": f"단일 트레이스 위 클러스터 축소율·시드 민감도 분석({len(ali_sweep_configs[sc])}회 중 고유 (model, gpu) 구성 {unique_cfgs}개; 독립 표본 통계 검정 불가).",
            "schedulers": {
                sn: {
                    "blk_gh_mean": float(np.mean([x["blk_gh"] for x in by_sn[sn]])),
                    "bf_wait_mean": float(np.mean([x["bf_wait"] for x in by_sn[sn]])),
                }
                for sn in s3_ali_scheds
            },
            "raw_per_unit": {
                sn: [{"idx": x["idx"], "blk_gh": x["blk_gh"], "bf_wait": x["bf_wait"]} for x in by_sn[sn]]
                for sn in s3_ali_scheds
            },
        }

    # 아티팩트 해시 기록 (코어 + 평가 스크립트 + K8s 어댑터 + Alibaba CSV 포함 16개 파일)
    hash_files = [
        BASE_DIR / "simulator.py",
        BASE_DIR / "adaptive_lindy.py",
        BASE_DIR / "omni_mrl.py",
        BASE_DIR / "bigtech_baselines.py",
        BASE_DIR / "baselines.py",
        BASE_DIR / "trace_loader.py",
        BASE_DIR / "stats_utils.py",
        BASE_DIR / "workloads.py",
        BASE_DIR / "confirm_30seeds.py",
        BASE_DIR / "run_phase8_eval.py",
        BASE_DIR / "run_phase11_eval.py",
        BASE_DIR / "k8s_extender" / "adapter.py",
        BASE_DIR / "k8s_extender" / "server.py",
        ALIBABA_DIR / "openb_node_list_gpu_node.csv",
        ALIBABA_DIR / "openb_pod_list_default.csv",
        ALIBABA_DIR / "openb_pod_list_gpuspec33.csv",
    ]
    hashes = {str(p.relative_to(BASE_DIR)).replace("\\", "/"): _sha256_file(p) for p in hash_files if p.exists()}

    report = {
        "protocol": {
            "version": "11.1",
            "dev_seeds_10": DEV_SEEDS_10,
            "lockbox_seeds_30": LOCKBOX_SEEDS_30,
            "omni_mrl_frozen": True,
            "cohort_km_length_bias_mitigated": True,
            "tuned_weights": tuned_weights,
            "interior_optimum": interior_optimum,
            "dev_grid_mean_blocking": dev_grid_scores,
            "artifact_sha256": hashes,
        },
        "suite_1_lockbox30_factorial": suite1_out,
        "suite_1b_ablation_lockbox30": ablation_out,
        "suite_2_ml_crossover_lockbox30": suite2_out,
        "suite_3_real_traces": {
            "helios_multinode_20": {
                "schedulers": hel_stats,
                "comparisons": comps_hel,
                "holm_family_size": len(hel_clu_refs),
                "cluster_level_n": len(clusters_order),
                "window_level_n": 20,
                "raw_per_unit": {
                    sn: [
                        {"idx": x["idx"], "cluster": x["cluster"], "blk_gh": x["blk_gh"], "bft_gw_jct": x["bft_gw_jct"], "blk_locality": x["blk_locality"]}
                        for x in by_hel[sn]
                    ]
                    for sn in s3_hel_scheds
                },
            },
            "helios_low_cv_real_11": {
                "windows_meta": lowcv_meta,
                "schedulers": lowcv_stats,
                "v11_vs_v7": lowcv_v11_vs_v7,
                "raw_per_unit": {
                    sn: [
                        {"idx": x["idx"], "cluster": x["cluster"], "blk_gh": x["blk_gh"], "bft_wait": x["bft_wait"], "gamma_ifr_active_frac": x["gamma_ifr_active_frac"]}
                        for x in by_lowcv[sn]
                    ]
                    for sn in s3_lowcv_scheds
                },
            },
            "alibaba_disjoint_windows": ali_disjoint_out,
            "alibaba_scale_sensitivity": ali_sens_out,
        },
        "elapsed_sec": time.perf_counter() - t0,
    }

    out_path = BASE_DIR / "phase11_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[완료] {out_path} 저장 완료 ({report['elapsed_sec']:.1f}초)", flush=True)


if __name__ == "__main__":
    main()
