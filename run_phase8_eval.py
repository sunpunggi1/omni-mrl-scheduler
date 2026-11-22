"""
8단계(Phase 8): 빅테크 최신 논문(`Google LAVA/NILAS` MLSys '25 · `Alibaba DVLA` OSDI · `FGD`)
정면 비교 및 통합 신알고리즘 `Omni-MRL` (v8.0) 대규모 실증 벤치마크.

- Suite A: 전체 위험률 스펙트럼 30-시드 벤치마크 (미지 시드 101..3030 × 6개 워크로드 × 7종 스케줄러)
    1) IFR_Uniform (CV≈0.06)
    2) IFR_Weibull_k2 (CV≈0.52, 고정 에폭 AI 학습)
    3) W4_Memoryless_OOD (CV≈1.0, 지수 분포)
    4) W1_PAI_Mixed (양의 상관 + Heavy-Tail)
    5) W3_Heavy_Straggler (무상관 + 극단 Heavy-Tail)
    6) W2_Regime_Shift (주야간 분포 급변)
- Suite B: 실제 공개 기업 트레이스 Zero-Shot 정면 비교
    1) Alibaba PAI 2023 Default Sweep (12회) & GPUSpec33 Sweep (12회)
    2) SenseTime Helios 16~64 GPU 멀티노드 20윈도우 (`cross_rack_penalty=0.08`)
- Suite C: Learning-Augmented ML 예측 오차 교차점 실험 (10시드 × `W3` & `IFR_Uniform`)
    - `Google-LAVA-ML(sigma)` (sigma in {0.0, 0.5, 1.0, 1.5, 2.0}) vs Zero-ML `Omni-MRL-v8`
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from adaptive_lindy import AdaptiveBLindyPackScheduler
from baselines import (
    DotProductScheduler,
    FGDScheduler,
    FirstFitScheduler,
    L2NormScheduler,
    RackBestFitScheduler,
)
from bigtech_baselines import (
    AlibabaDVLAGPUScheduler,
    GoogleLAVAClassGPUScheduler,
    GoogleLAVAMLOracleScheduler,
    GoogleNILASGPUScheduler,
    run_ml_lava_simulation,
)
from confirm_30seeds import UNSEEN_SEEDS_30, generate_ifr_antilindy_ood
from omni_mrl import OmniMRLConfig, OmniMRLScheduler
from simulator import BaseScheduler, ClusterConfig, Job, run_queue_simulation, run_simulation
from stats_utils import paired_summary_stats
from trace_loader import (
    build_empirical_fgd_distribution,
    extract_helios_multinode_windows,
    load_alibaba_2023,
    load_helios_cluster_multinode,
    scale_heterogeneous_cluster,
    slice_active_trace,
)
from workloads import (
    STANDARD_PROFILES,
    generate_w1_pai_mixed,
    generate_w2_regime_shift,
    generate_w3_heavy_straggler,
    generate_w4_memoryless_ood,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ALIBABA_DIR = DATA_DIR / "alibaba_2023"
HELIOS_DIR = DATA_DIR / "helios"

SUITE_A_SCHEDS = [
    "FirstFit",
    "L2Norm",
    "Google-NILAS",
    "Google-LAVA",
    "Alibaba-DVLA",
    "Omni-Lindy-v7",
    "Omni-MRL-v8",
]
SUITE_A_BASELINE_CANDS = ["FirstFit", "L2Norm", "Google-NILAS", "Google-LAVA", "Alibaba-DVLA"]

SUITE_B_ALI_SCHEDS = [
    "L2Norm",
    "DotProduct",
    "FGD_OracleStatic",
    "FGD_Adaptive",
    "Google-NILAS",
    "Google-LAVA",
    "Alibaba-DVLA",
    "Omni-Lindy-v7",
    "Omni-MRL-v8(Recyc0.25)",
    "Omni-MRL-v8",
]

SUITE_B_HEL_SCHEDS = [
    "L2Norm",
    "FGD_Adaptive",
    "RackBestFit",
    "Google-NILAS-Rack",
    "Alibaba-DVLA-Rack",
    "Omni-Lindy-v7",
    "Omni-MRL-v8",
]


def generate_ifr_weibull_k2(seed: int = 42, num_jobs: int = 4000, target_load: float = 0.88) -> List[Job]:
    """고정 에폭 AI 학습(초반 종료 드물고 평균 부근에 수렴)을 모사하는 Weibull(k=2.0, CV≈0.52) IFR 워크로드."""
    rng = np.random.default_rng(seed)
    probs = np.array([0.30, 0.24, 0.20, 0.11, 0.08, 0.07], dtype=np.float64)
    probs /= probs.sum()
    idx = rng.choice(len(STANDARD_PROFILES), size=num_jobs, p=probs)
    scale = 2.0 / math.gamma(1.5)
    durs = rng.weibull(2.0, size=num_jobs) * scale
    mgh = float(np.mean([STANDARD_PROFILES[idx[i]][0] * durs[i] for i in range(num_jobs)]))
    arr = np.cumsum(rng.exponential(mgh / (target_load * 512.0), size=num_jobs))
    return [Job(i, float(arr[i]), float(durs[i]), *STANDARD_PROFILES[idx[i]]) for i in range(num_jobs)]


def _generate_suite_a_jobs(w_name: str, seed: int) -> List[Job]:
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
    raise ValueError(f"알 수 없는 Suite A 워크로드: {w_name}")


def _make_scheduler(
    name: str,
    fgd_reqs: np.ndarray | None = None,
    fgd_probs: np.ndarray | None = None,
) -> BaseScheduler:
    if name == "FirstFit":
        return FirstFitScheduler()
    if name == "L2Norm":
        return L2NormScheduler()
    if name == "DotProduct":
        return DotProductScheduler()
    if name == "RackBestFit":
        return RackBestFitScheduler()
    if name == "FGD_OracleStatic":
        return FGDScheduler(adaptive=False, custom_reqs=fgd_reqs, custom_probs=fgd_probs, name="FGD_OracleStatic")
    if name == "FGD_Adaptive":
        return FGDScheduler(adaptive=True, custom_reqs=fgd_reqs, custom_probs=fgd_probs, name="FGD_Adaptive")
    if name == "Google-NILAS":
        return GoogleNILASGPUScheduler(name="Google-NILAS", topology_aware=False)
    if name == "Google-NILAS-Rack":
        return GoogleNILASGPUScheduler(name="Google-NILAS-Rack", topology_aware=True)
    if name == "Google-LAVA":
        return GoogleLAVAClassGPUScheduler(name="Google-LAVA", topology_aware=False)
    if name == "Alibaba-DVLA":
        return AlibabaDVLAGPUScheduler(name="Alibaba-DVLA", topology_aware=False)
    if name == "Alibaba-DVLA-Rack":
        return AlibabaDVLAGPUScheduler(name="Alibaba-DVLA-Rack", topology_aware=True)
    if name == "Omni-Lindy-v7":
        return AdaptiveBLindyPackScheduler(name="Omni-Lindy-v7", hetero_aware=True, topology_aware=True)
    if name == "Omni-MRL-v8(Recyc0.25)":
        return OmniMRLScheduler(
            mrl_cfg=OmniMRLConfig(recycling_base_weight=0.25),
            name="Omni-MRL-v8(Recyc0.25)",
            record_diagnostics=True,
            hetero_aware=True,
            topology_aware=True,
        )
    if name == "Omni-MRL-v8":
        return OmniMRLScheduler(name="Omni-MRL-v8", record_diagnostics=True, hetero_aware=True, topology_aware=True)
    raise ValueError(f"알 수 없는 스케줄러: {name}")


def _eval_suite_a_unit(task: Tuple[str, int, str]) -> Dict[str, Any]:
    w_name, seed, sched_name = task
    cfg = ClusterConfig(num_nodes=64, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)
    jobs = _generate_suite_a_jobs(w_name, seed)
    sch = _make_scheduler(sched_name)

    mb = run_simulation(jobs, sch, config=cfg, workload_name=w_name, seed=seed)
    diag = sch.get_diagnostics_summary() if hasattr(sch, "get_diagnostics_summary") else {}
    mf = run_queue_simulation(jobs, sch, config=cfg, workload_name=w_name, seed=seed, queue_policy="fifo")
    mbf = run_queue_simulation(
        jobs, sch, config=cfg, workload_name=w_name, seed=seed,
        queue_policy="backfill", backfill_window=16, max_hol_bypasses=8,
    )
    return {
        "workload": w_name,
        "seed": seed,
        "scheduler": sched_name,
        "blocking": mb.gpu_hour_blocking_rate,
        "fifo_wait": mf.mean_wait_time,
        "bf_wait": mbf.mean_wait_time,
        "bf_jct": mbf.mean_jct,
        "lat_us": mb.avg_decision_us,
        "diag": diag,
    }


def _eval_suite_b_ali_unit(
    task: Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str],
) -> Dict[str, Any]:
    sc, idx, cfg, jobs, reqs, probs, sched_name = task
    sch = _make_scheduler(sched_name, fgd_reqs=reqs, fgd_probs=probs)
    mb = run_simulation(jobs, sch, config=cfg, workload_name=sc, seed=idx)
    mf = run_queue_simulation(jobs, sch, config=cfg, workload_name=sc, seed=idx, queue_policy="fifo")
    mbf = run_queue_simulation(jobs, sch, config=cfg, workload_name=sc, seed=idx, queue_policy="backfill")
    return {
        "scenario": sc,
        "idx": idx,
        "scheduler": sched_name,
        "blk_gh": mb.gpu_hour_blocking_rate,
        "blk_stranded": mb.avg_stranded_gpu_ratio,
        "fifo_wait": mf.mean_wait_time,
        "bf_wait": mbf.mean_wait_time,
        "bf_jct": mbf.mean_jct,
        "lat_us": mb.avg_decision_us,
    }


def _eval_suite_b_hel_unit(
    task: Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str],
) -> Dict[str, Any]:
    sc, idx, cfg, jobs, reqs, probs, sched_name = task
    sch = _make_scheduler(sched_name, fgd_reqs=reqs, fgd_probs=probs)
    mb = run_simulation(jobs, sch, config=cfg, workload_name=sc, seed=idx)
    mbf_topo = run_queue_simulation(
        jobs, sch, config=cfg, workload_name=sc, seed=idx, queue_policy="backfill", cross_rack_penalty=0.08
    )
    return {
        "scenario": sc,
        "idx": idx,
        "scheduler": sched_name,
        "blk_gh": mb.gpu_hour_blocking_rate,
        "blk_locality": mb.single_rack_locality_rate,
        "bft_wait": mbf_topo.mean_wait_time,
        "bft_gw_jct": mbf_topo.gpu_weighted_jct,
        "lat_us": mb.avg_decision_us,
    }


def _eval_suite_c_unit(task: Tuple[str, int, float]) -> Dict[str, Any]:
    """Suite C: LAVA-ML(sigma) vs Zero-ML (`Omni-MRL-v8`, `Omni-Lindy-v7`, `L2Norm`) 교차점 단위 평가."""
    w_name, seed, sigma = task
    cfg = ClusterConfig(num_nodes=64)
    jobs = _generate_suite_a_jobs(w_name, seed)
    rng = np.random.default_rng(seed * 97 + int(round(sigma * 100)))
    noise = rng.normal(0.0, sigma, size=len(jobs)) if sigma > 0.0 else np.zeros(len(jobs))
    pred_map = {id(j): max(1e-6, float(j.duration) * math.exp(float(noise[i]))) for i, j in enumerate(jobs)}
    ml_sch = GoogleLAVAMLOracleScheduler(pred_map=pred_map, time_unit_hours=1.0, name=f"Google-LAVA-ML({sigma:.1f})")
    res = run_ml_lava_simulation(jobs, ml_sch, cfg)
    return {"workload": w_name, "seed": seed, "sigma": sigma, "blocking": res["blocking"], "bf_wait": res["backfill_wait"]}


def _summarize_suite_a(rows: List[dict]) -> Dict[str, Any]:
    workloads = ["IFR_Uniform", "IFR_Weibull_k2", "W4_Memoryless_OOD", "W1_PAI_Mixed", "W3_Heavy_Straggler", "W2_Regime_Shift"]
    by_w: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for r in rows:
        by_w.setdefault(r["workload"], {}).setdefault(r["scheduler"], {})[r["seed"]] = r

    out_workloads: Dict[str, Any] = {}
    pooled_pairs: Dict[str, Tuple[List[float], List[float]]] = {
        "blocking_vs_best_base": ([], []),
        "bf_wait_vs_best_base": ([], []),
        "blocking_vs_v7": ([], []),
        "bf_wait_vs_v7": ([], []),
    }

    for wn in workloads:
        smap = by_w[wn]
        w_out: Dict[str, Any] = {"schedulers": {}, "comparisons": {}}
        for metric in ("blocking", "fifo_wait", "bf_wait"):
            means = {sn: float(np.mean([smap[sn][s][metric] for s in UNSEEN_SEEDS_30])) for sn in SUITE_A_SCHEDS}
            stds = {sn: float(np.std([smap[sn][s][metric] for s in UNSEEN_SEEDS_30], ddof=1)) for sn in SUITE_A_SCHEDS}
            best_base = min(SUITE_A_BASELINE_CANDS, key=lambda n: means[n])
            best_bigtech = min(["Google-NILAS", "Google-LAVA", "Alibaba-DVLA"], key=lambda n: means[n])

            v8_arr = np.array([smap["Omni-MRL-v8"][s][metric] for s in UNSEEN_SEEDS_30], dtype=np.float64)
            v7_arr = np.array([smap["Omni-Lindy-v7"][s][metric] for s in UNSEEN_SEEDS_30], dtype=np.float64)
            base_arr = np.array([smap[best_base][s][metric] for s in UNSEEN_SEEDS_30], dtype=np.float64)
            bt_arr = np.array([smap[best_bigtech][s][metric] for s in UNSEEN_SEEDS_30], dtype=np.float64)

            if metric in ("blocking", "bf_wait"):
                k_base = f"{metric}_vs_best_base"
                pooled_pairs[k_base][0].extend(base_arr.tolist())
                pooled_pairs[k_base][1].extend(v8_arr.tolist())
                k_v7 = f"{metric}_vs_v7"
                pooled_pairs[k_v7][0].extend(v7_arr.tolist())
                pooled_pairs[k_v7][1].extend(v8_arr.tolist())

            w_out["schedulers"][metric] = {
                sn: {"mean": means[sn], "std": stds[sn]} for sn in SUITE_A_SCHEDS
            }
            w_out["comparisons"][metric] = {
                "best_baseline_name": best_base,
                "best_bigtech_name": best_bigtech,
                "vs_best_baseline": paired_summary_stats(base_arr, v8_arr),
                "vs_best_bigtech": paired_summary_stats(bt_arr, v8_arr),
                "vs_v7": paired_summary_stats(v7_arr, v8_arr),
            }

        # 진단 요약 (gamma_ifr, gamma_dfr, alpha)
        diags = [smap["Omni-MRL-v8"][s]["diag"] for s in UNSEEN_SEEDS_30 if smap["Omni-MRL-v8"][s]["diag"]]
        if diags:
            w_out["diagnostics"] = {
                "gamma_dfr_mean": float(np.mean([d["gamma_mean"] for d in diags])),
                "gamma_ifr_mean": float(np.mean([d.get("gamma_ifr", 0.0) for d in diags])),
                "alpha_mean": float(np.mean([d["alpha_mean"] for d in diags])),
                "lat_us_mean": float(np.mean([smap["Omni-MRL-v8"][s]["lat_us"] for s in UNSEEN_SEEDS_30])),
            }
        out_workloads[wn] = w_out

    pooled_summary = {
        k: paired_summary_stats(np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64))
        for k, (b, c) in pooled_pairs.items()
    }
    return {"workloads": out_workloads, "pooled_180": pooled_summary}


def _build_alibaba_tasks() -> List[Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str]]:
    """Alibaba Default & GPUSpec33 스윕 태스크 목록을 생성한다."""
    node_csv = ALIBABA_DIR / "openb_node_list_gpu_node.csv"
    ali_cfg, ali_def_jobs, _ = load_alibaba_2023(node_csv, ALIBABA_DIR / "openb_pod_list_default.csv", use_gpu_spec=True)
    _, ali_spec_jobs, _ = load_alibaba_2023(node_csv, ALIBABA_DIR / "openb_pod_list_gpuspec33.csv", use_gpu_spec=True)
    ali_def_act, _ = slice_active_trace(ali_def_jobs, 0.05)
    ali_spec_act, _ = slice_active_trace(ali_spec_jobs, 0.05)
    def_reqs, def_probs = build_empirical_fgd_distribution(ali_def_act)
    spec_reqs, spec_probs = build_empirical_fgd_distribution(ali_spec_act)

    ali_tasks: List[Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str]] = []
    for idx, (kr, s) in enumerate((k, sd) for k in [0.0025, 0.0030, 0.0035, 0.0045] for sd in [101, 202, 303]):
        c_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_def_act)
        for sn in SUITE_B_ALI_SCHEDS:
            ali_tasks.append(("Alibaba_Default_Sweep", idx, c_sub, ali_def_act, def_reqs, def_probs, sn))
    for idx, (kr, s) in enumerate((k, sd) for k in [0.0050, 0.0065, 0.0080, 0.0100] for sd in [101, 202, 303]):
        c_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_spec_act)
        for sn in SUITE_B_ALI_SCHEDS:
            ali_tasks.append(("Alibaba_GPUSpec33_Sweep", idx, c_sub, ali_spec_act, spec_reqs, spec_probs, sn))
    return ali_tasks


def _summarize_alibaba(res_b_ali: List[dict]) -> Dict[str, Any]:
    """Alibaba 스윕 실행 결과를 요약하고 통계 검정을 수행한다."""
    ali_summary: Dict[str, Any] = {}
    for sc in ("Alibaba_Default_Sweep", "Alibaba_GPUSpec33_Sweep"):
        rows_sc = [r for r in res_b_ali if r["scenario"] == sc]
        by_sn: Dict[str, List[dict]] = {sn: [] for sn in SUITE_B_ALI_SCHEDS}
        for r in rows_sc:
            by_sn[r["scheduler"]].append(r)
        for sn in SUITE_B_ALI_SCHEDS:
            by_sn[sn].sort(key=lambda x: x["idx"])
        sched_stats = {
            sn: {
                "blk_gh_mean": float(np.mean([x["blk_gh"] for x in by_sn[sn]])),
                "blk_stranded_mean": float(np.mean([x["blk_stranded"] for x in by_sn[sn]])),
                "fifo_wait_mean": float(np.mean([x["fifo_wait"] for x in by_sn[sn]])),
                "bf_wait_mean": float(np.mean([x["bf_wait"] for x in by_sn[sn]])),
                "lat_us_mean": float(np.mean([x["lat_us"] for x in by_sn[sn]])),
            }
            for sn in SUITE_B_ALI_SCHEDS
        }
        v8_blk = np.array([x["blk_gh"] for x in by_sn["Omni-MRL-v8"]])
        v7_blk = np.array([x["blk_gh"] for x in by_sn["Omni-Lindy-v7"]])
        recyc_blk = np.array([x["blk_gh"] for x in by_sn["Omni-MRL-v8(Recyc0.25)"]])
        v8_bf = np.array([x["bf_wait"] for x in by_sn["Omni-MRL-v8"]])
        v7_bf = np.array([x["bf_wait"] for x in by_sn["Omni-Lindy-v7"]])
        recyc_bf = np.array([x["bf_wait"] for x in by_sn["Omni-MRL-v8(Recyc0.25)"]])
        v8_fifo = np.array([x["fifo_wait"] for x in by_sn["Omni-MRL-v8"]])
        v7_fifo = np.array([x["fifo_wait"] for x in by_sn["Omni-Lindy-v7"]])
        fgd_adp_bf = np.array([x["bf_wait"] for x in by_sn["FGD_Adaptive"]])
        ali_summary[sc] = {
            "schedulers": sched_stats,
            "comparisons": {
                "blk_v8_vs_v7": paired_summary_stats(v7_blk, v8_blk),
                "blk_v8_vs_recyc025": paired_summary_stats(recyc_blk, v8_blk),
                "fifo_wait_v8_vs_v7": paired_summary_stats(v7_fifo, v8_fifo),
                "bf_wait_v8_vs_v7": paired_summary_stats(v7_bf, v8_bf),
                "bf_wait_v8_vs_recyc025": paired_summary_stats(recyc_bf, v8_bf),
                "bf_wait_v8_vs_fgd_adaptive": paired_summary_stats(fgd_adp_bf, v8_bf),
            },
        }
    return ali_summary


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 8: Omni-MRL (v8.0) vs 빅테크 대규모 평가")
    parser.add_argument(
        "--update-suite-b-alibaba",
        action="store_true",
        help="기존 phase8_report.json의 Suite A/Helios/Suite C는 유지하고 Suite B Alibaba만 재계산·갱신",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    n_workers = max(1, min(8, (mp.cpu_count() or 4) - 1))
    out_path = BASE_DIR / "phase8_report.json"

    if args.update_suite_b_alibaba:
        if not out_path.exists():
            raise FileNotFoundError(f"기존 리포트가 없어 부분 갱신할 수 없습니다: {out_path}")
        with out_path.open(encoding="utf-8") as f:
            report = json.load(f)
        ali_tasks = _build_alibaba_tasks()
        print(f"[Phase 8 부분 갱신] CPU 워커 {n_workers}개로 Suite B Alibaba ({len(ali_tasks)} 단위) 재실행...")
        with mp.Pool(processes=n_workers) as pool:
            res_b_ali = pool.map(_eval_suite_b_ali_unit, ali_tasks)
        ali_summary = _summarize_alibaba(res_b_ali)
        report["suite_b_real_traces"]["alibaba"] = ali_summary
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        for sc in ("Alibaba_Default_Sweep", "Alibaba_GPUSpec33_Sweep"):
            s8 = ali_summary[sc]["schedulers"]["Omni-MRL-v8"]
            sr = ali_summary[sc]["schedulers"]["Omni-MRL-v8(Recyc0.25)"]
            print(
                f"  [{sc}] Omni-MRL-v8: blk={s8['blk_gh_mean']:.4f}%, bf_w={s8['bf_wait_mean']:.2f}h | "
                f"Recyc0.25 Ablation: blk={sr['blk_gh_mean']:.4f}%, bf_w={sr['bf_wait_mean']:.2f}h"
            )
        print(f"[완료] {out_path} Suite B Alibaba 갱신 완료 ({time.perf_counter() - t0:.1f}초)")
        return

    print(f"[Phase 8 평가] CPU 워커 {n_workers}개로 Suite A / B / C 실행 시작...")

    # 1) Suite A 태스크 (6 워크로드 × 30 시드 × 7 스케줄러 = 1,260개)
    suite_a_workloads = [
        "IFR_Uniform",
        "IFR_Weibull_k2",
        "W4_Memoryless_OOD",
        "W1_PAI_Mixed",
        "W3_Heavy_Straggler",
        "W2_Regime_Shift",
    ]
    tasks_a = [(wn, s, sn) for wn in suite_a_workloads for s in UNSEEN_SEEDS_30 for sn in SUITE_A_SCHEDS]

    # 2) Suite B 태스크 (Alibaba Default & GPUSpec33 + Helios MultiNode 20윈도우)
    ali_tasks = _build_alibaba_tasks()

    hel_tasks: List[Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str]] = []
    hel_idx = 0
    for cl in ["Venus", "Earth", "Saturn", "Uranus"]:
        hj, _ = load_helios_cluster_multinode(HELIOS_DIR / cl / "cluster_log.csv", cl, max_nodes_per_job=8)
        wins = extract_helios_multinode_windows(
            hj, num_windows=5, jobs_per_window=3000, target_load=0.85, min_span_hours=2.0, nodes_per_rack=8
        )
        for wc, wj, _ in wins:
            wr, wp = build_empirical_fgd_distribution(wj)
            for sn in SUITE_B_HEL_SCHEDS:
                hel_tasks.append(("Helios_MultiNode_Pooled_20", hel_idx, wc, wj, wr, wp, sn))
            hel_idx += 1

    # 3) Suite C 태스크 (10 미지 시드 × 2 워크로드 × 5 sigma 수준 = 100개 ML-LAVA 실행; Zero-ML 비교군은 Suite A 결과 재사용)
    seeds_10 = UNSEEN_SEEDS_30[:10]
    sigmas = [0.0, 0.5, 1.0, 1.5, 2.0]
    tasks_c = [(wn, s, sg) for wn in ("W3_Heavy_Straggler", "IFR_Uniform") for s in seeds_10 for sg in sigmas]

    with mp.Pool(processes=n_workers) as pool:
        print(f"  - Suite A 실행 중 ({len(tasks_a)} 단위 시뮬레이션)...")
        res_a = pool.map(_eval_suite_a_unit, tasks_a)
        print(f"  - Suite B (Alibaba {len(ali_tasks)} + HeliosMN {len(hel_tasks)}) 실행 중...")
        res_b_ali = pool.map(_eval_suite_b_ali_unit, ali_tasks)
        res_b_hel = pool.map(_eval_suite_b_hel_unit, hel_tasks)
        print(f"  - Suite C ML 예측 오차 교차점 실행 중 ({len(tasks_c)} 단위)...")
        res_c = pool.map(_eval_suite_c_unit, tasks_c)

    suite_a_summary = _summarize_suite_a(res_a)
    ali_summary = _summarize_alibaba(res_b_ali)

    hel_by_sn: Dict[str, List[dict]] = {sn: [] for sn in SUITE_B_HEL_SCHEDS}
    for r in res_b_hel:
        hel_by_sn[r["scheduler"]].append(r)
    for sn in SUITE_B_HEL_SCHEDS:
        hel_by_sn[sn].sort(key=lambda x: x["idx"])
    hel_stats = {
        sn: {
            "blk_gh_mean": float(np.mean([x["blk_gh"] for x in hel_by_sn[sn]])),
            "blk_locality_mean": float(np.mean([x["blk_locality"] for x in hel_by_sn[sn]])),
            "bft_wait_mean": float(np.mean([x["bft_wait"] for x in hel_by_sn[sn]])),
            "bft_gw_jct_mean": float(np.mean([x["bft_gw_jct"] for x in hel_by_sn[sn]])),
            "lat_us_mean": float(np.mean([x["lat_us"] for x in hel_by_sn[sn]])),
        }
        for sn in SUITE_B_HEL_SCHEDS
    }
    v8_jct = np.array([x["bft_gw_jct"] for x in hel_by_sn["Omni-MRL-v8"]])
    rbf_jct = np.array([x["bft_gw_jct"] for x in hel_by_sn["RackBestFit"]])
    nilas_jct = np.array([x["bft_gw_jct"] for x in hel_by_sn["Google-NILAS-Rack"]])
    v7_jct = np.array([x["bft_gw_jct"] for x in hel_by_sn["Omni-Lindy-v7"]])
    hel_summary = {
        "schedulers": hel_stats,
        "comparisons": {
            "bft_gw_jct_v8_vs_rackbestfit": paired_summary_stats(rbf_jct, v8_jct),
            "bft_gw_jct_v8_vs_nilas_rack": paired_summary_stats(nilas_jct, v8_jct),
            "bft_gw_jct_v8_vs_v7": paired_summary_stats(v7_jct, v8_jct),
        },
    }

    # Suite C 요약 (ML-LAVA vs Zero-ML 10시드)
    suite_c_summary: Dict[str, Any] = {}
    seed10_set = set(seeds_10)
    for wn in ("W3_Heavy_Straggler", "IFR_Uniform"):
        ml_by_sig: Dict[str, Dict[str, float]] = {}
        for sg in sigmas:
            sub = [r for r in res_c if r["workload"] == wn and abs(r["sigma"] - sg) < 1e-9]
            ml_by_sig[f"sigma_{sg:.1f}"] = {
                "blocking_mean": float(np.mean([r["blocking"] for r in sub])),
                "bf_wait_mean": float(np.mean([r["bf_wait"] for r in sub])),
            }
        # 동일 10시드 Zero-ML 성적 추출
        zero_ml: Dict[str, Dict[str, float]] = {}
        for sn in ("L2Norm", "Google-LAVA", "Omni-Lindy-v7", "Omni-MRL-v8"):
            sub_z = [r for r in res_a if r["workload"] == wn and r["scheduler"] == sn and r["seed"] in seed10_set]
            zero_ml[sn] = {
                "blocking_mean": float(np.mean([r["blocking"] for r in sub_z])),
                "bf_wait_mean": float(np.mean([r["bf_wait"] for r in sub_z])),
            }
        suite_c_summary[wn] = {"lava_ml_sweep": ml_by_sig, "zero_ml_refs": zero_ml}

    elapsed = time.perf_counter() - t0
    report = {
        "suite_a_spectrum_30": suite_a_summary,
        "suite_b_real_traces": {"alibaba": ali_summary, "helios_multinode": hel_summary},
        "suite_c_ml_crossover_10": suite_c_summary,
        "elapsed_sec": elapsed,
    }
    out_path = BASE_DIR / "phase8_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[Phase 8 완료] {out_path} 저장 완료 (총 {elapsed:.1f}초)")
    for wn, wd in suite_a_summary["workloads"].items():
        b_comp = wd["comparisons"]["blocking"]
        w_comp = wd["comparisons"]["bf_wait"]
        print(
            f"  SuiteA [{wn:18s}] Blk v8={wd['schedulers']['blocking']['Omni-MRL-v8']['mean']:.3f}% "
            f"(vs {b_comp['best_baseline_name']} {b_comp['vs_best_baseline']['rel_improvement_pct']:+.2f}%) | "
            f"BFWait v8={wd['schedulers']['bf_wait']['Omni-MRL-v8']['mean']:.4f}h "
            f"(vs v7 {w_comp['vs_v7']['rel_improvement_pct']:+.2f}%)"
        )


if __name__ == "__main__":
    main()
