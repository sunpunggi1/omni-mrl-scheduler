"""
10단계(Phase 10): v10.0 무편향 코호트 Kaplan-Meier, 연속 NILAS-ML 오라클,
전 비교군 Holm-Bonferroni 보정 및 101k↔Fresh 시드 재현성 감사 평가.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from adaptive_lindy import AdaptiveBLindyPackScheduler, BPackSpatialOnlyScheduler
from baselines import L2NormScheduler, RackBestFitScheduler
from bigtech_baselines import (
    AlibabaDVLAGPUScheduler,
    GoogleLAVAClassGPUScheduler,
    GoogleLAVAMLOracleScheduler,
    GoogleNILASGPUScheduler,
    GoogleNILASMLOracleScheduler,
    run_ml_lava_simulation,
)
from confirm_30seeds import UNSEEN_SEEDS_30, generate_ifr_antilindy_ood
from omni_mrl import OmniMRLScheduler
from run_phase8_eval import generate_ifr_weibull_k2
from simulator import BaseScheduler, ClusterConfig, Job, run_queue_simulation, run_simulation
from stats_utils import holm_bonferroni_adjust, paired_summary_stats
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
HELIOS_DIR = BASE_DIR / "data" / "helios"

DEV_SEEDS_10: List[int] = [5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081]
FRESH_TEST_SEEDS_30: List[int] = [10007 + 97 * i for i in range(30)]
DEV_WORKLOADS: List[str] = ["IFR_Uniform", "IFR_Weibull_k2", "W1_PAI_Mixed", "W3_Heavy_Straggler"]
SUITE1_WORKLOADS: List[str] = [
    "IFR_Uniform",
    "IFR_Weibull_k2",
    "W4_Memoryless_OOD",
    "W1_PAI_Mixed",
    "W3_Heavy_Straggler",
    "W2_Regime_Shift",
]
REPRO_WORKLOADS: List[str] = ["IFR_Uniform", "W4_Memoryless_OOD", "W1_PAI_Mixed", "W3_Heavy_Straggler"]

DEV_GRIDS: Dict[str, List[float]] = {
    "Google-NILAS-BPack": [0.10, 0.25, 0.45, 0.65, 1.00],
    "Google-LAVA-BPack": [0.20, 0.40, 0.70, 1.00, 1.50],
    "Alibaba-DVLA-BPack": [0.20, 0.40, 0.70, 1.00, 1.50],
    "Google-LAVA-ML-BPack": [0.20, 0.40, 0.70, 1.00, 1.50],
    "Google-NILAS-ML-BPack": [0.10, 0.25, 0.45, 0.65, 1.00],
}


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
            w_temporal=tuned_weights.get("Google-NILAS-BPack", 0.65),
            spatial_base="bpack",
            hetero_aware=hetero_aware,
        )
    if name == "Google-LAVA-BPack":
        return GoogleLAVAClassGPUScheduler(
            name="Google-LAVA-BPack",
            topology_aware=topology_aware,
            spatial_base="bpack",
            hetero_aware=hetero_aware,
            w_tier=tuned_weights.get("Google-LAVA-BPack", 1.00),
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
    if name == "Omni-MRL-v10":
        return OmniMRLScheduler(
            name="Omni-MRL-v10",
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

    # Oracle ML family (sigma = 0.0 on Dev seeds)
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


def _eval_suite1_unit(task: Tuple[str, int, str, Dict[str, float], str]) -> Dict[str, Any]:
    wn, seed, sn, tuned, split_tag = task
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
        "split": split_tag,
        "workload": wn,
        "seed": seed,
        "scheduler": sn,
        "blocking": mb.gpu_hour_blocking_rate,
        "bf_wait": mbf.mean_wait_time,
        "lat_us": mb.avg_decision_us,
        "diag": diag,
    }


def _eval_suite2_unit(task: Tuple[str, int, str, float, Dict[str, float]]) -> Dict[str, Any]:
    wn, seed, kind, sigma, tuned = task
    cfg = ClusterConfig(num_nodes=64)
    jobs = _gen_workload(wn, seed)
    if kind == "Omni-MRL-v10":
        sch: BaseScheduler = OmniMRLScheduler(name="Omni-MRL-v10", hetero_aware=True, topology_aware=False)
    elif kind == "BPack-SpatialOnly":
        sch = BPackSpatialOnlyScheduler(name="BPack-SpatialOnly", hetero_aware=True, topology_aware=False)
    else:
        rng = np.random.default_rng(seed * 97 + int(round(sigma * 100)))
        noise = rng.normal(0.0, sigma, size=len(jobs)) if sigma > 0.0 else np.zeros(len(jobs))
        pred_map = {id(j): max(1e-6, float(j.duration) * math.exp(float(noise[i]))) for i, j in enumerate(jobs)}
        if kind == "LAVA-ML-BPack":
            q25, q50, q75 = np.quantile([j.duration for j in jobs], [0.25, 0.50, 0.75])
            sch = GoogleLAVAMLOracleScheduler(
                pred_map=pred_map,
                time_unit_hours=1.0,
                name=f"Google-LAVA-ML-BPack({sigma:.2f})",
                lc_thresholds=(float(q25), float(q50), float(q75)),
                spatial_base="bpack",
                hetero_aware=True,
                topology_aware=False,
                w_tier=tuned.get("Google-LAVA-ML-BPack", 0.70),
            )
        else:
            sch = GoogleNILASMLOracleScheduler(
                pred_map=pred_map,
                time_unit_hours=1.0,
                name=f"Google-NILAS-ML-BPack({sigma:.2f})",
                spatial_base="bpack",
                hetero_aware=True,
                topology_aware=False,
                w_temporal=tuned.get("Google-NILAS-ML-BPack", 0.65),
            )
    res = run_ml_lava_simulation(jobs, sch, cfg)
    return {
        "workload": wn,
        "seed": seed,
        "kind": kind,
        "sigma": sigma,
        "blocking": res["blocking"],
        "bf_wait": res["backfill_wait"],
    }


def _eval_suite3_ali_unit(
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


def _eval_suite3_hel_unit(
    task: Tuple[str, int, ClusterConfig, List[Job], str, Dict[str, float]],
) -> Dict[str, Any]:
    sc, idx, cfg, jobs, sn, tuned = task
    sch = _build_scheduler(sn, tuned, topology_aware=True, hetero_aware=True)
    mb = run_simulation(jobs, sch, config=cfg, workload_name=sc, seed=idx)
    diag = sch.get_diagnostics_summary() if hasattr(sch, "get_diagnostics_summary") else {}
    mbf = run_queue_simulation(
        jobs, sch, config=cfg, workload_name=sc, seed=idx, queue_policy="backfill", cross_rack_penalty=0.08
    )
    return {
        "scenario": sc,
        "idx": idx,
        "scheduler": sn,
        "blk_gh": mb.gpu_hour_blocking_rate,
        "blk_locality": mb.single_rack_locality_rate,
        "bft_wait": mbf.mean_wait_time,
        "bft_gw_jct": mbf.gpu_weighted_jct,
        "gamma_ifr_active_frac": float(diag.get("gamma_ifr_active_frac", 0.0)),
    }


def main() -> None:
    t0 = time.perf_counter()
    n_workers = max(1, min(8, (mp.cpu_count() or 4) - 1))
    print(f"[Phase 10 v10.0 무편향 코호트 KM 평가] CPU 워커 {n_workers}개 시작...")

    # 0) Step 0: Dev 10-시드 동일 5점 튜닝 (Omni-MRL 완전 동결; ML 오라클 포함 5종 동등 예산)
    dev_tasks = [
        (fam, w, wn, s)
        for fam, grid in DEV_GRIDS.items()
        for w in grid
        for wn in DEV_WORKLOADS
        for s in DEV_SEEDS_10
    ]
    print(f"  [0/3] Dev 10-시드 5종 가중치 스윕 ({len(dev_tasks)} 단위)...")
    with mp.Pool(processes=n_workers) as pool:
        dev_rows = pool.map(_eval_dev_unit, dev_tasks)

    dev_grid_scores: Dict[str, Dict[str, float]] = {}
    tuned_weights: Dict[str, float] = {}
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
    print(f"    -> Dev 동결 가중치: {tuned_weights}")

    # 1) Suite 1: Fresh 30-시드 팩토리얼 (8종, vs_L2Norm 포함) + [101k] 재현성 대조
    s1_scheds = [
        "L2Norm",
        "BPack-SpatialOnly",
        "Google-NILAS-L2",
        "Google-NILAS-BPack",
        "Google-LAVA-BPack",
        "Alibaba-DVLA-BPack",
        "Omni-Lindy-v7",
        "Omni-MRL-v10",
    ]
    repro_scheds = ["L2Norm", "Google-NILAS-BPack", "Google-LAVA-BPack", "Omni-MRL-v10"]
    bpack_bt = ["Google-NILAS-BPack", "Google-LAVA-BPack", "Alibaba-DVLA-BPack"]

    tasks_s1 = (
        [(wn, s, sn, tuned_weights, "fresh30") for wn in SUITE1_WORKLOADS for s in FRESH_TEST_SEEDS_30 for sn in s1_scheds]
        + [(wn, s, sn, tuned_weights, "hist101k") for wn in REPRO_WORKLOADS for s in UNSEEN_SEEDS_30 for sn in repro_scheds]
    )
    print(f"  [1/3] Suite 1 Fresh 30-시드 + 101k 재현성 대조 ({len(tasks_s1)} 단위)...")
    with mp.Pool(processes=n_workers) as pool:
        rows_s1 = pool.map(_eval_suite1_unit, tasks_s1)

    by_w_fresh: Dict[str, Dict[str, Dict[int, dict]]] = {}
    by_w_101k: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for r in rows_s1:
        target = by_w_fresh if r["split"] == "fresh30" else by_w_101k
        target.setdefault(r["workload"], {}).setdefault(r["scheduler"], {})[r["seed"]] = r

    suite1_out: Dict[str, Any] = {"workloads": {}, "holm_family_size": 0}
    holm_refs: List[dict] = []
    for wn in SUITE1_WORKLOADS:
        smap = by_w_fresh[wn]
        w_res: Dict[str, Any] = {"schedulers": {}, "comparisons": {}}
        for metric in ("blocking", "bf_wait"):
            w_res["schedulers"][metric] = {
                sn: {
                    "mean": float(np.mean([smap[sn][s][metric] for s in FRESH_TEST_SEEDS_30])),
                    "std": float(np.std([smap[sn][s][metric] for s in FRESH_TEST_SEEDS_30], ddof=1)),
                }
                for sn in s1_scheds
            }
            best_bpack_bt = min(bpack_bt, key=lambda n: w_res["schedulers"][metric][n]["mean"])
            v10_arr = np.array([smap["Omni-MRL-v10"][s][metric] for s in FRESH_TEST_SEEDS_30], dtype=np.float64)
            comps = {
                "vs_l2norm": paired_summary_stats(
                    np.array([smap["L2Norm"][s][metric] for s in FRESH_TEST_SEEDS_30], dtype=np.float64),
                    v10_arr,
                ),
                "vs_bpack_spatial_only": paired_summary_stats(
                    np.array([smap["BPack-SpatialOnly"][s][metric] for s in FRESH_TEST_SEEDS_30], dtype=np.float64),
                    v10_arr,
                ),
                "vs_omni_lindy_v7": paired_summary_stats(
                    np.array([smap["Omni-Lindy-v7"][s][metric] for s in FRESH_TEST_SEEDS_30], dtype=np.float64),
                    v10_arr,
                ),
                "vs_google_nilas_bpack": paired_summary_stats(
                    np.array([smap["Google-NILAS-BPack"][s][metric] for s in FRESH_TEST_SEEDS_30], dtype=np.float64),
                    v10_arr,
                ),
                "vs_best_bpack_bigtech": paired_summary_stats(
                    np.array([smap[best_bpack_bt][s][metric] for s in FRESH_TEST_SEEDS_30], dtype=np.float64),
                    v10_arr,
                ),
            }
            w_res["comparisons"][metric] = {
                "best_bpack_bigtech_name": best_bpack_bt,
                **comps,
            }
            holm_refs.extend(comps.values())

        diags = [smap["Omni-MRL-v10"][s]["diag"] for s in FRESH_TEST_SEEDS_30 if smap["Omni-MRL-v10"][s]["diag"]]
        w_res["diagnostics"] = {
            "gamma_ifr_mean": float(np.mean([d.get("gamma_ifr_mean", 0.0) for d in diags])),
            "gamma_ifr_active_frac": float(np.mean([d.get("gamma_ifr_active_frac", 0.0) for d in diags])),
            "gamma_dfr_mean": float(np.mean([d.get("gamma_dfr_mean", 0.0) for d in diags])),
            "alpha_mean": float(np.mean([d.get("alpha_mean_step", 0.0) for d in diags])),
        }
        suite1_out["workloads"][wn] = w_res

    s1_adj = holm_bonferroni_adjust([c["p_value_wilcoxon"] for c in holm_refs])
    for c, ph in zip(holm_refs, s1_adj):
        c["p_value_wilcoxon_holm60"] = float(ph)
    suite1_out["holm_family_size"] = len(holm_refs)

    # 101k vs Fresh 30-시드 재현성 비교 표
    repro_out: Dict[str, Any] = {}
    for wn in REPRO_WORKLOADS:
        repro_out[wn] = {}
        for split_name, seeds_list, src_map in (
            ("hist101k_30", UNSEEN_SEEDS_30, by_w_101k[wn]),
            ("fresh_30", FRESH_TEST_SEEDS_30, by_w_fresh[wn]),
        ):
            v10_blk = np.array([src_map["Omni-MRL-v10"][s]["blocking"] for s in seeds_list], dtype=np.float64)
            v10_bf = np.array([src_map["Omni-MRL-v10"][s]["bf_wait"] for s in seeds_list], dtype=np.float64)
            l2_blk = np.array([src_map["L2Norm"][s]["blocking"] for s in seeds_list], dtype=np.float64)
            l2_bf = np.array([src_map["L2Norm"][s]["bf_wait"] for s in seeds_list], dtype=np.float64)
            ni_blk = np.array([src_map["Google-NILAS-BPack"][s]["blocking"] for s in seeds_list], dtype=np.float64)
            la_blk = np.array([src_map["Google-LAVA-BPack"][s]["blocking"] for s in seeds_list], dtype=np.float64)
            repro_out[wn][split_name] = {
                "Omni-MRL-v10_blk": float(np.mean(v10_blk)),
                "L2Norm_blk": float(np.mean(l2_blk)),
                "NILAS-BPack_blk": float(np.mean(ni_blk)),
                "LAVA-BPack_blk": float(np.mean(la_blk)),
                "blk_vs_l2norm": paired_summary_stats(l2_blk, v10_blk),
                "bf_vs_l2norm": paired_summary_stats(l2_bf, v10_bf),
            }

    # 2) Suite 2: Fresh 30-시드 Zero-ML vs Dev 튜닝 LAVA-ML-BPack & 연속 NILAS-ML-BPack
    sigmas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    tasks_s2 = (
        [(wn, s, "Omni-MRL-v10", -1.0, tuned_weights) for wn in ("W3_Heavy_Straggler", "IFR_Uniform") for s in FRESH_TEST_SEEDS_30]
        + [(wn, s, "BPack-SpatialOnly", -1.0, tuned_weights) for wn in ("W3_Heavy_Straggler", "IFR_Uniform") for s in FRESH_TEST_SEEDS_30]
        + [(wn, s, "LAVA-ML-BPack", sg, tuned_weights) for wn in ("W3_Heavy_Straggler", "IFR_Uniform") for s in FRESH_TEST_SEEDS_30 for sg in sigmas]
        + [(wn, s, "NILAS-ML-BPack", sg, tuned_weights) for wn in ("W3_Heavy_Straggler", "IFR_Uniform") for s in FRESH_TEST_SEEDS_30 for sg in sigmas]
    )
    print(f"  [2/3] Suite 2 Fresh 30-시드 Dual-Oracle ML 교차점 ({len(tasks_s2)} 단위)...")
    with mp.Pool(processes=n_workers) as pool:
        rows_s2 = pool.map(_eval_suite2_unit, tasks_s2)

    suite2_out: Dict[str, Any] = {}
    s2_holm_refs: List[dict] = []
    for wn in ("W3_Heavy_Straggler", "IFR_Uniform"):
        rows_w = [r for r in rows_s2 if r["workload"] == wn]
        om_map = {r["seed"]: r for r in rows_w if r["kind"] == "Omni-MRL-v10"}
        sp_map = {r["seed"]: r for r in rows_w if r["kind"] == "BPack-SpatialOnly"}
        om_blk = np.array([om_map[s]["blocking"] for s in FRESH_TEST_SEEDS_30], dtype=np.float64)
        om_bf = np.array([om_map[s]["bf_wait"] for s in FRESH_TEST_SEEDS_30], dtype=np.float64)
        sp_blk = np.array([sp_map[s]["blocking"] for s in FRESH_TEST_SEEDS_30], dtype=np.float64)
        sp_bf = np.array([sp_map[s]["bf_wait"] for s in FRESH_TEST_SEEDS_30], dtype=np.float64)

        lava_sweep: List[Dict[str, Any]] = []
        nilas_sweep: List[Dict[str, Any]] = []
        for sg in sigmas:
            for ml_kind, target_list in (("LAVA-ML-BPack", lava_sweep), ("NILAS-ML-BPack", nilas_sweep)):
                ml_map = {r["seed"]: r for r in rows_w if r["kind"] == ml_kind and abs(r["sigma"] - sg) < 1e-6}
                ml_blk = np.array([ml_map[s]["blocking"] for s in FRESH_TEST_SEEDS_30], dtype=np.float64)
                ml_bf = np.array([ml_map[s]["bf_wait"] for s in FRESH_TEST_SEEDS_30], dtype=np.float64)
                st_blk = paired_summary_stats(ml_blk, om_blk)
                st_bf = paired_summary_stats(ml_bf, om_bf)
                s2_holm_refs.extend([st_blk, st_bf])
                target_list.append(
                    {
                        "sigma": sg,
                        "ml_blocking_mean": float(np.mean(ml_blk)),
                        "ml_bf_wait_mean": float(np.mean(ml_bf)),
                        "zeroml_vs_ml_blocking": st_blk,
                        "zeroml_vs_ml_bf_wait": st_bf,
                    }
                )
        suite2_out[wn] = {
            "omni_mrl_v10": {"blocking_mean": float(np.mean(om_blk)), "bf_wait_mean": float(np.mean(om_bf))},
            "bpack_spatial_only": {"blocking_mean": float(np.mean(sp_blk)), "bf_wait_mean": float(np.mean(sp_bf))},
            "lava_ml_bpack_sweep": lava_sweep,
            "nilas_ml_bpack_sweep": nilas_sweep,
        }

    s2_adj = holm_bonferroni_adjust([c["p_value_wilcoxon"] for c in s2_holm_refs])
    for c, ph in zip(s2_holm_refs, s2_adj):
        c["p_value_wilcoxon_holm48"] = float(ph)

    # 3) Suite 3: 실트레이스 공정 팩토리얼 (전 비교군 누락 없이 Holm 보정)
    s3_ali_scheds = [
        "L2Norm",
        "BPack-SpatialOnly",
        "Google-NILAS-BPack",
        "Google-LAVA-BPack",
        "Alibaba-DVLA-BPack",
        "Omni-Lindy-v7",
        "Omni-MRL-v10",
    ]
    s3_hel_scheds = [
        "L2Norm",
        "RackBestFit",
        "BPack-SpatialOnly",
        "Google-NILAS-BPack",
        "Alibaba-DVLA-BPack",
        "Omni-Lindy-v7",
        "Omni-MRL-v10",
    ]

    node_csv = ALIBABA_DIR / "openb_node_list_gpu_node.csv"
    ali_cfg, ali_def_jobs, _ = load_alibaba_2023(node_csv, ALIBABA_DIR / "openb_pod_list_default.csv", use_gpu_spec=True)
    _, ali_spec_jobs, _ = load_alibaba_2023(node_csv, ALIBABA_DIR / "openb_pod_list_gpuspec33.csv", use_gpu_spec=True)
    ali_def_act, _ = slice_active_trace(ali_def_jobs, 0.05)
    ali_spec_act, _ = slice_active_trace(ali_spec_jobs, 0.05)

    ali_tasks: List[Tuple[str, int, ClusterConfig, List[Job], str, Dict[str, float]]] = []
    for idx, (kr, s) in enumerate((k, sd) for k in [0.0025, 0.0030, 0.0035, 0.0045] for sd in [101, 202, 303]):
        c_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_def_act)
        for sn in s3_ali_scheds:
            ali_tasks.append(("Alibaba_Default_Sweep", idx, c_sub, ali_def_act, sn, tuned_weights))
    for idx, (kr, s) in enumerate((k, sd) for k in [0.0050, 0.0065, 0.0080, 0.0100] for sd in [101, 202, 303]):
        c_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_spec_act)
        for sn in s3_ali_scheds:
            ali_tasks.append(("Alibaba_GPUSpec33_Sweep", idx, c_sub, ali_spec_act, sn, tuned_weights))

    hel_tasks: List[Tuple[str, int, ClusterConfig, List[Job], str, Dict[str, float]]] = []
    hel_idx = 0
    for cl in ("Venus", "Earth", "Saturn", "Uranus"):
        hj, _ = load_helios_cluster_multinode(HELIOS_DIR / cl / "cluster_log.csv", cl, max_nodes_per_job=8)
        wins = extract_helios_multinode_windows(
            hj, num_windows=5, jobs_per_window=3000, target_load=0.85, min_span_hours=2.0, nodes_per_rack=8
        )
        for wc, wj, _ in wins:
            for sn in s3_hel_scheds:
                hel_tasks.append(("Helios_MultiNode_Pooled_20", hel_idx, wc, wj, sn, tuned_weights))
            hel_idx += 1

    print(f"  [3/3] Suite 3 실트레이스 전 비교군 평가 (Alibaba {len(ali_tasks)} + Helios {len(hel_tasks)})...")
    with mp.Pool(processes=n_workers) as pool:
        res_ali = pool.map(_eval_suite3_ali_unit, ali_tasks)
        res_hel = pool.map(_eval_suite3_hel_unit, hel_tasks)

    ali_out: Dict[str, Any] = {}
    s3_holm_refs: List[dict] = []
    for sc in ("Alibaba_Default_Sweep", "Alibaba_GPUSpec33_Sweep"):
        by_sn: Dict[str, List[dict]] = {sn: [] for sn in s3_ali_scheds}
        for r in res_ali:
            if r["scenario"] == sc:
                by_sn[r["scheduler"]].append(r)
        for sn in s3_ali_scheds:
            by_sn[sn].sort(key=lambda x: x["idx"])
        sc_stats = {
            sn: {
                "blk_gh_mean": float(np.mean([x["blk_gh"] for x in by_sn[sn]])),
                "bf_wait_mean": float(np.mean([x["bf_wait"] for x in by_sn[sn]])),
                "gamma_ifr_active_frac": float(np.mean([x["gamma_ifr_active_frac"] for x in by_sn[sn]])),
            }
            for sn in s3_ali_scheds
        }
        v10_blk = np.array([x["blk_gh"] for x in by_sn["Omni-MRL-v10"]])
        v10_bf = np.array([x["bf_wait"] for x in by_sn["Omni-MRL-v10"]])
        comps_ali = {}
        for ref_sn in [sn for sn in s3_ali_scheds if sn != "Omni-MRL-v10"]:
            rb = paired_summary_stats(np.array([x["blk_gh"] for x in by_sn[ref_sn]]), v10_blk)
            rw = paired_summary_stats(np.array([x["bf_wait"] for x in by_sn[ref_sn]]), v10_bf)
            comps_ali[f"blk_vs_{ref_sn}"] = rb
            comps_ali[f"bf_wait_vs_{ref_sn}"] = rw
            s3_holm_refs.extend([rb, rw])
        ali_out[sc] = {
            "note": "12 runs share one fixed job trace across 4 cluster scale ratios x 3 sampling seeds (pseudo-replicated cluster subsampling).",
            "schedulers": sc_stats,
            "comparisons": comps_ali,
        }

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
    v10_gw = np.array([x["bft_gw_jct"] for x in by_hel["Omni-MRL-v10"]])
    v10_hb = np.array([x["blk_gh"] for x in by_hel["Omni-MRL-v10"]])
    comps_hel = {}
    for ref_sn in [sn for sn in s3_hel_scheds if sn != "Omni-MRL-v10"]:
        rgw = paired_summary_stats(np.array([x["bft_gw_jct"] for x in by_hel[ref_sn]]), v10_gw)
        rhb = paired_summary_stats(np.array([x["blk_gh"] for x in by_hel[ref_sn]]), v10_hb)
        comps_hel[f"gw_jct_vs_{ref_sn}"] = rgw
        comps_hel[f"blk_vs_{ref_sn}"] = rhb
        s3_holm_refs.extend([rgw, rhb])

    s3_adj = holm_bonferroni_adjust([c["p_value_wilcoxon"] for c in s3_holm_refs])
    for c, ph in zip(s3_holm_refs, s3_adj):
        c["p_value_wilcoxon_holm_all"] = float(ph)

    report = {
        "protocol": {
            "dev_seeds_10": DEV_SEEDS_10,
            "fresh_test_seeds_30": FRESH_TEST_SEEDS_30,
            "hist_101k_seeds_30": UNSEEN_SEEDS_30,
            "omni_mrl_frozen": True,
            "cohort_km_unbiased": True,
            "tuned_weights": tuned_weights,
            "dev_grid_mean_blocking": dev_grid_scores,
        },
        "suite_1_fresh30_factorial": suite1_out,
        "reproducibility_101k_vs_fresh": repro_out,
        "suite_2_ml_crossover_fresh30": suite2_out,
        "suite_3_real_traces_factorial": {
            "alibaba": ali_out,
            "helios_multinode": {"schedulers": hel_stats, "comparisons": comps_hel},
            "holm_family_size": len(s3_holm_refs),
        },
        "elapsed_sec": time.perf_counter() - t0,
    }

    out_path = BASE_DIR / "phase10_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[완료] {out_path} 저장 완료 ({report['elapsed_sec']:.1f}초)")


if __name__ == "__main__":
    main()
