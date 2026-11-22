"""
5단계 대규모 통계 확증 벤치마크: 30개 미지 시드 × 6개 워크로드 × (Blocking + Pending Queue FIFO/Backfill)

엄격한 과학적 검증 원칙 (Zero Data Leakage):
- 진화/설계/진단에 사용된 시드(11..277)와 100% 격리된 30개 미지 시드(101..3030)만 평가
  (특히 시드 1111..3030의 20개는 이전 1~4단계에서도 단 한 번도 실행되지 않은 완전 신규 시드)
- 6개 전체 워크로드:
  * W1_PAI_Mixed       (표준 혼합, 양의 크기-수명 상관)
  * W2_Regime_Shift    (주야간 분포 급변)
  * W3_Heavy_Straggler (극단적 장단기 혼재, 크기-수명 무상관)
  * W4_Memoryless_OOD  (반례 OOD: 순수 지수분포)
  * W5_Skewed_Vec_OOD  (OOD: 극심한 CPU/RAM 비대칭)
  * W6_Scale1024_OOD   (OOD: 128노드 1,024-GPU 초대형 클러스터, 시드당 8,000작업)
- 3가지 시뮬레이션 모델 동시 확증:
  1) Blocking 모델 (GPU-Hr 거절률, Stranded GPU %, Active Node-Hours)
  2) Pending Queue Strict FIFO (Mean JCT, GPU-Weighted JCT, P95/P99 JCT, Mean Wait, Makespan)
  3) Pending Queue Starvation-Free Backfill (Mean JCT, GPU-Weighted JCT, Mean Wait, Makespan)
- 통계 검정: 정확 양측 Student-t, 비모수 Wilcoxon Signed-Rank, 정확 이항 부호 검정, 95% 신뢰구간
- 단일 스레드 전용 의사결정 지연시간(Latency, us) 별도 측정 (멀티코어 경합 노이즈 배제)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from adaptive_lindy import AdaptiveBLindyPackScheduler, AdaptiveLindyConfig, DEFAULT_BASE_THETA
from baselines import (
    BestFitScheduler,
    DotProductScheduler,
    FGDScheduler,
    FirstFitScheduler,
    L2NormScheduler,
)
from b_lindy_discovery import BoundedLindyPackScheduler
from evolve_scheduler import ParametricEvoScheduler
from simulator import (
    BaseScheduler,
    ClusterConfig,
    Job,
    run_queue_simulation,
    run_simulation,
)
from stats_utils import exact_sign_test_pvalue, paired_summary_stats
from verify_discovery import generate_w6_scale1024_ood
from workloads import STANDARD_PROFILES, WORKLOAD_SUITE

# 30개 미지 시드 (101..1010: 4단계 기존 미지 10시드 + 1111..3030: 완전 신규 20시드)
UNSEEN_SEEDS_30: List[int] = [101 * k for k in range(1, 31)]
BRAND_NEW_SEEDS_20: List[int] = [101 * k for k in range(11, 31)]

ALL_SCHEDULER_NAMES: List[str] = [
    "FirstFit",
    "BestFit",
    "DotProduct",
    "L2Norm",
    "FGD_Adaptive",
    "EvoSTF_Dist",
    "B-Pack_NoAge",
    "B-LindyPack",
    "Adaptive-B-Lindy",
]

BACKFILL_SCHEDULER_NAMES: List[str] = list(ALL_SCHEDULER_NAMES)

SOTA_BASELINE_NAMES: List[str] = [
    "FirstFit",
    "BestFit",
    "DotProduct",
    "L2Norm",
    "FGD_Adaptive",
]

WORKLOAD_ORDER: List[str] = [
    "W1_PAI_Mixed",
    "W2_Regime_Shift",
    "W3_Heavy_Straggler",
    "W4_Memoryless_OOD",
    "W5_Skewed_Vec_OOD",
    "W6_Scale1024_OOD",
]

_W_DIST_CACHE: np.ndarray = np.zeros(18, dtype=np.float64)


def _init_worker(w_dist: np.ndarray) -> None:
    global _W_DIST_CACHE
    _W_DIST_CACHE = w_dist


def _make_scheduler(name: str, w_dist: np.ndarray, record_diag: bool = False) -> BaseScheduler:
    if name == "FirstFit":
        return FirstFitScheduler()
    if name == "BestFit":
        return BestFitScheduler()
    if name == "DotProduct":
        return DotProductScheduler()
    if name == "L2Norm":
        return L2NormScheduler()
    if name == "FGD_Adaptive":
        return FGDScheduler(adaptive=True)
    if name == "EvoSTF_Dist":
        return ParametricEvoScheduler(w_dist, name="EvoSTF_Dist")
    if name == "B-Pack_NoAge":
        t0 = DEFAULT_BASE_THETA.copy()
        t0[6] = 0.0
        return BoundedLindyPackScheduler(t0, name="B-Pack_NoAge")
    if name == "B-LindyPack":
        return BoundedLindyPackScheduler(DEFAULT_BASE_THETA, name="B-LindyPack")
    if name == "Adaptive-B-Lindy":
        return AdaptiveBLindyPackScheduler(
            base_theta=DEFAULT_BASE_THETA,
            name="Adaptive-B-Lindy",
            record_diagnostics=record_diag,
        )
    raise ValueError(f"알 수 없는 스케줄러: {name}")


def _generate_jobs(w_name: str, seed: int):
    if w_name == "W6_Scale1024_OOD":
        return generate_w6_scale1024_ood(seed=seed)
    return WORKLOAD_SUITE[w_name](seed=seed)


def _worker_run_seed_scheduler(args: Tuple[str, int, str, bool]) -> Dict[str, Any]:
    """단일 (워크로드, 시드, 스케줄러) 조합에 대해 Blocking + FIFO + (선택) Backfill 시뮬레이션을 실행한다."""
    w_name, seed, sch_name, run_backfill = args
    cfg = (
        ClusterConfig(num_nodes=128, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)
        if w_name == "W6_Scale1024_OOD"
        else ClusterConfig(num_nodes=64, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)
    )
    jobs = _generate_jobs(w_name, seed)
    is_adapt = (sch_name == "Adaptive-B-Lindy")
    sch = _make_scheduler(sch_name, _W_DIST_CACHE, record_diag=is_adapt)

    # 1) Blocking 모델
    mb = run_simulation(jobs, sch, config=cfg, workload_name=w_name, seed=seed)
    diag_block = sch.get_diagnostics_summary() if is_adapt else {}

    # 2) Pending Queue Strict FIFO
    mq_fifo = run_queue_simulation(jobs, sch, config=cfg, workload_name=w_name, seed=seed, queue_policy="fifo")
    diag_fifo = sch.get_diagnostics_summary() if is_adapt else {}

    # 3) Pending Queue Starvation-Free Backfill (선택적)
    bf_dict = None
    if run_backfill:
        mq_bf = run_queue_simulation(
            jobs, sch, config=cfg, workload_name=w_name, seed=seed,
            queue_policy="backfill", backfill_window=16, max_hol_bypasses=8,
        )
        bf_dict = {
            "mean_jct": mq_bf.mean_jct,
            "gpu_weighted_jct": mq_bf.gpu_weighted_jct,
            "p95_jct": mq_bf.p95_jct,
            "p99_jct": mq_bf.p99_jct,
            "mean_wait_time": mq_bf.mean_wait_time,
            "mean_slowdown": mq_bf.mean_slowdown,
            "makespan": mq_bf.makespan,
            "avg_stranded_gpu_on_wait": mq_bf.avg_stranded_gpu_on_wait,
        }

    return {
        "workload": w_name,
        "seed": seed,
        "scheduler": sch_name,
        "blocking": {
            "gpu_hour_blocking_rate": mb.gpu_hour_blocking_rate,
            "job_blocking_rate": mb.job_blocking_rate,
            "avg_stranded_gpu_ratio": mb.avg_stranded_gpu_ratio,
            "active_node_hours": mb.active_node_hours,
            "avg_gpu_utilization": mb.avg_gpu_utilization,
        },
        "fifo": {
            "mean_jct": mq_fifo.mean_jct,
            "gpu_weighted_jct": mq_fifo.gpu_weighted_jct,
            "p95_jct": mq_fifo.p95_jct,
            "p99_jct": mq_fifo.p99_jct,
            "mean_wait_time": mq_fifo.mean_wait_time,
            "p95_wait_time": mq_fifo.p95_wait_time,
            "mean_slowdown": mq_fifo.mean_slowdown,
            "makespan": mq_fifo.makespan,
            "max_queue_len": mq_fifo.max_queue_len,
            "avg_queue_len": mq_fifo.avg_queue_len,
            "avg_stranded_gpu_on_wait": mq_fifo.avg_stranded_gpu_on_wait,
        },
        "backfill": bf_dict,
        "diag_block": diag_block,
        "diag_fifo": diag_fifo,
    }


def generate_ifr_antilindy_ood(seed: int = 42, num_jobs: int = 3000, target_load: float = 0.88) -> List[Job]:
    """
    [반린디(Anti-Lindy) IFR 스트레스 OOD]: 좁은 균등분포 Uniform(1.8, 2.2) 수명 워크로드.
    변동계수 CV ≈ 0.06 << 1 로 오래 돌수록 잔여 수명이 줄어드는 증가고장률(IFR) 환경이며,
    린디 효과가 정반대로 거짓일 때 CV 유효성 게이트(gamma)의 안전 보호 능력을 확증한다.
    """
    rng = np.random.default_rng(seed)
    probs = np.array([0.30, 0.24, 0.20, 0.11, 0.08, 0.07], dtype=np.float64)
    probs /= probs.sum()
    idx = rng.choice(len(STANDARD_PROFILES), size=num_jobs, p=probs)
    durs = rng.uniform(1.8, 2.2, size=num_jobs)
    mgh = float(np.mean([STANDARD_PROFILES[idx[i]][0] * durs[i] for i in range(num_jobs)]))
    arr = np.cumsum(rng.exponential(mgh / (target_load * 512.0), size=num_jobs))
    return [Job(i, float(arr[i]), float(durs[i]), *STANDARD_PROFILES[idx[i]]) for i in range(num_jobs)]


def _worker_run_ifr_seed(seed: int) -> Dict[str, Any]:
    """단일 시드의 IFR 반린디 OOD 비교 (CV 게이트 ON vs OFF 및 베이스라인)."""
    cfg = ClusterConfig(num_nodes=64, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)
    jobs = generate_ifr_antilindy_ood(seed=seed)
    no_cv_cfg = AdaptiveLindyConfig(cv_center=-10.0, cv_slope=0.08)
    scheds: List[BaseScheduler] = [
        FirstFitScheduler(),
        BestFitScheduler(),
        L2NormScheduler(),
        AdaptiveBLindyPackScheduler(adapt_cfg=no_cv_cfg, name="Adaptive-NoCVGate", record_diagnostics=True),
        AdaptiveBLindyPackScheduler(name="Adaptive-B-Lindy", record_diagnostics=True),
    ]
    out: Dict[str, Any] = {"seed": seed, "schedulers": {}}
    for sch in scheds:
        mb = run_simulation(jobs, sch, config=cfg, workload_name="IFR_AntiLindy_OOD", seed=seed)
        diag = sch.get_diagnostics_summary() if hasattr(sch, "get_diagnostics_summary") else {}
        mf = run_queue_simulation(jobs, sch, config=cfg, workload_name="IFR_AntiLindy_OOD", seed=seed, queue_policy="fifo")
        mbf = run_queue_simulation(
            jobs, sch, config=cfg, workload_name="IFR_AntiLindy_OOD", seed=seed,
            queue_policy="backfill", backfill_window=16, max_hol_bypasses=8,
        )
        out["schedulers"][sch.name] = {
            "blocking": mb.gpu_hour_blocking_rate,
            "fifo_wait": mf.mean_wait_time,
            "backfill_wait": mbf.mean_wait_time,
            "backfill_makespan": mbf.makespan,
            "diag": diag,
        }
    return out


def benchmark_single_thread_latency(w_dist: np.ndarray) -> Dict[str, Dict[str, float]]:
    """멀티프로세싱 CPU 경합이 전혀 없는 단일 스레드 환경에서 정확한 의사결정 지연시간(us)을 측정한다."""
    lat_report: Dict[str, Dict[str, float]] = {}
    configs = [
        ("W1_PAI_Mixed_64N", "W1_PAI_Mixed", ClusterConfig(num_nodes=64), [101, 202, 303]),
        ("W3_Heavy_Straggler_64N", "W3_Heavy_Straggler", ClusterConfig(num_nodes=64), [101, 202, 303]),
        ("W6_Scale1024_128N", "W6_Scale1024_OOD", ClusterConfig(num_nodes=128), [101, 202]),
    ]
    for label, w_name, cfg, seeds in configs:
        jobs_map = {s: _generate_jobs(w_name, s) for s in seeds}
        row: Dict[str, float] = {}
        for sch_name in ALL_SCHEDULER_NAMES:
            sch = _make_scheduler(sch_name, w_dist, record_diag=False)
            lats = [run_simulation(jobs_map[s], sch, config=cfg).avg_decision_us for s in seeds]
            row[sch_name] = float(np.mean(lats))
        lat_report[label] = row
    return lat_report


def _summarize_metric(
    records_by_sch: Dict[str, Dict[int, Dict[str, Any]]],
    seeds: List[int],
    mode: str,
    metric_key: str,
    sota_candidates: List[str],
    cand_name: str = "Adaptive-B-Lindy",
) -> Dict[str, Any]:
    """주어진 모드/지표에 대해 최강 SOTA 선정 및 쌍체 통계 요약을 산출한다."""
    sch_means: Dict[str, Dict[str, float]] = {}
    for sch_name, smap in records_by_sch.items():
        if smap[seeds[0]][mode] is None:
            continue
        vals = np.array([smap[s][mode][metric_key] for s in seeds], dtype=np.float64)
        sch_means[sch_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    avail_sota = [s for s in sota_candidates if s in sch_means]
    best_sota = min(avail_sota, key=lambda n: sch_means[n]["mean"])

    cand_arr = np.array([records_by_sch[cand_name][s][mode][metric_key] for s in seeds], dtype=np.float64)
    sota_arr = np.array([records_by_sch[best_sota][s][mode][metric_key] for s in seeds], dtype=np.float64)
    vs_sota = paired_summary_stats(sota_arr, cand_arr)

    comparisons: Dict[str, Any] = {"vs_best_sota": {"sota_name": best_sota, **vs_sota}}
    for ref_name in ("B-LindyPack", "EvoSTF_Dist", "B-Pack_NoAge", "FGD_Adaptive", "L2Norm"):
        if ref_name in sch_means:
            ref_arr = np.array([records_by_sch[ref_name][s][mode][metric_key] for s in seeds], dtype=np.float64)
            comparisons[f"vs_{ref_name}"] = paired_summary_stats(ref_arr, cand_arr)

    return {
        "best_sota": best_sota,
        "scheduler_stats": sch_means,
        "comparisons": comparisons,
    }


def main() -> None:
    t0 = time.perf_counter()
    root = Path(__file__).parent
    sol = json.loads((root / "evolved_solution.json").read_text(encoding="utf-8"))
    w_dist = np.array(sol["distilled_weights"], dtype=np.float64)

    # 작업 목록 생성 및 LPT(Longest Processing Time First) 정렬로 병렬 워커 꼬리 지연 최소화
    tasks: List[Tuple[str, int, str, bool]] = []
    for w_name in WORKLOAD_ORDER:
        for s in UNSEEN_SEEDS_30:
            for sch_name in ALL_SCHEDULER_NAMES:
                tasks.append((w_name, s, sch_name, sch_name in BACKFILL_SCHEDULER_NAMES))

    def _est_weight(t: Tuple[str, int, str, bool]) -> int:
        w_name, _, sch_name, run_bf = t
        cost = 4 if w_name == "W6_Scale1024_OOD" else 1
        if sch_name == "FGD_Adaptive":
            cost *= 12
        elif sch_name == "EvoSTF_Dist":
            cost *= 2
        if run_bf:
            cost = int(cost * 1.4)
        return -cost

    tasks.sort(key=_est_weight)

    n_workers = max(1, min(24, (os.cpu_count() or 8) - 2))
    print(
        f"[5단계 30-시드 대규모 확증 시작] {len(WORKLOAD_ORDER)} 워크로드 × {len(UNSEEN_SEEDS_30)} 미지 시드 × "
        f"{len(ALL_SCHEDULER_NAMES)} 스케줄러 = {len(tasks)}개 멀티모드 실행 ({n_workers} workers)",
        flush=True,
    )

    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(w_dist,)) as pool:
        raw_results = pool.map(_worker_run_seed_scheduler, tasks)

    sim_elapsed = time.perf_counter() - t0
    print(f"[시뮬레이션 완료] 총 {len(raw_results)}개 멀티모드 실행 완료 ({sim_elapsed:.1f}초)", flush=True)

    # 구조화: grouped[w_name][sch_name][seed] = record
    grouped: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {
        w: {sch: {} for sch in ALL_SCHEDULER_NAMES} for w in WORKLOAD_ORDER
    }
    for rec in raw_results:
        grouped[rec["workload"]][rec["scheduler"]][rec["seed"]] = rec

    # 단일 스레드 지연시간 벤치마크 (경합 없음)
    print("\n[단일 스레드 의사결정 지연시간 정밀 측정 중...]", flush=True)
    latency_single = benchmark_single_thread_latency(w_dist)

    # 워크로드별 통계 집계 및 출력
    report_workloads: Dict[str, Any] = {}
    pooled_block_wins = 0
    pooled_block_losses = 0
    pooled_fifo_wait_wins = 0
    pooled_fifo_wait_losses = 0
    pooled_fifo_jct_wins = 0
    pooled_fifo_jct_losses = 0
    pooled_fifo_mk_wins = 0
    pooled_fifo_mk_losses = 0
    pooled_bf_wait_wins = 0
    pooled_bf_wait_losses = 0
    pooled_bf_jct_wins = 0
    pooled_bf_jct_losses = 0
    pooled_bf_mk_wins = 0
    pooled_bf_mk_losses = 0
    pooled_rel_block_diffs: List[float] = []
    pooled_rel_fifo_wait_diffs: List[float] = []
    pooled_rel_fifo_jct_diffs: List[float] = []
    pooled_rel_fifo_mk_diffs: List[float] = []
    pooled_rel_bf_wait_diffs: List[float] = []
    pooled_rel_bf_jct_diffs: List[float] = []
    pooled_rel_bf_mk_diffs: List[float] = []

    print("\n" + "=" * 108)
    print("  [표 1] Blocking 모델 30개 미지 시드 확증 결과 (GPU-Hr Blocking % — 낮을수록 우수)")
    print("=" * 108)
    print(
        f"{'Workload':<20} | {'Best SOTA':<12} | {'SOTA(%)':>8} | {'EvoSTF':>7} | "
        f"{'B-Lindy':>7} | {'Adapt-BL':>8} | {'Impr':>7} | {'Wins':>6} | {'p(t)':>8} | {'p(W)':>8}"
    )
    print("-" * 108)

    for w_name in WORKLOAD_ORDER:
        rec_map = grouped[w_name]

        blk_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "blocking", "gpu_hour_blocking_rate", SOTA_BASELINE_NAMES)
        blk_20 = _summarize_metric(rec_map, BRAND_NEW_SEEDS_20, "blocking", "gpu_hour_blocking_rate", SOTA_BASELINE_NAMES)
        str_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "blocking", "avg_stranded_gpu_ratio", SOTA_BASELINE_NAMES)

        fifo_jct_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "fifo", "mean_jct", SOTA_BASELINE_NAMES)
        fifo_gwjct_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "fifo", "gpu_weighted_jct", SOTA_BASELINE_NAMES)
        fifo_p95_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "fifo", "p95_jct", SOTA_BASELINE_NAMES)
        fifo_wait_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "fifo", "mean_wait_time", SOTA_BASELINE_NAMES)
        fifo_mk_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "fifo", "makespan", SOTA_BASELINE_NAMES)
        fifo_str_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "fifo", "avg_stranded_gpu_on_wait", SOTA_BASELINE_NAMES)
        fifo_wait_20 = _summarize_metric(rec_map, BRAND_NEW_SEEDS_20, "fifo", "mean_wait_time", SOTA_BASELINE_NAMES)

        bf_sota_cands = [s for s in SOTA_BASELINE_NAMES if s in BACKFILL_SCHEDULER_NAMES]
        bf_jct_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "backfill", "mean_jct", bf_sota_cands)
        bf_gwjct_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "backfill", "gpu_weighted_jct", bf_sota_cands)
        bf_wait_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "backfill", "mean_wait_time", bf_sota_cands)
        bf_mk_30 = _summarize_metric(rec_map, UNSEEN_SEEDS_30, "backfill", "makespan", bf_sota_cands)

        # 온라인 진단(rho, alpha, gamma) 집계
        rhos_blk = [rec_map["Adaptive-B-Lindy"][s]["diag_block"]["rho_mean"] for s in UNSEEN_SEEDS_30]
        alphas_blk = [rec_map["Adaptive-B-Lindy"][s]["diag_block"]["alpha_mean"] for s in UNSEEN_SEEDS_30]
        gammas_blk = [rec_map["Adaptive-B-Lindy"][s]["diag_block"]["gamma_mean"] for s in UNSEEN_SEEDS_30]
        rhos_fifo = [rec_map["Adaptive-B-Lindy"][s]["diag_fifo"]["rho_mean"] for s in UNSEEN_SEEDS_30]
        alphas_fifo = [rec_map["Adaptive-B-Lindy"][s]["diag_fifo"]["alpha_mean"] for s in UNSEEN_SEEDS_30]
        gammas_fifo = [rec_map["Adaptive-B-Lindy"][s]["diag_fifo"]["gamma_mean"] for s in UNSEEN_SEEDS_30]

        vs_b = blk_30["comparisons"]["vs_best_sota"]
        pooled_block_wins += vs_b["wins"]
        pooled_block_losses += vs_b["losses"]

        vs_fw = fifo_wait_30["comparisons"]["vs_best_sota"]
        pooled_fifo_wait_wins += vs_fw["wins"]
        pooled_fifo_wait_losses += vs_fw["losses"]

        vs_fj = fifo_jct_30["comparisons"]["vs_best_sota"]
        pooled_fifo_jct_wins += vs_fj["wins"]
        pooled_fifo_jct_losses += vs_fj["losses"]

        vs_fm = fifo_mk_30["comparisons"]["vs_best_sota"]
        pooled_fifo_mk_wins += vs_fm["wins"]
        pooled_fifo_mk_losses += vs_fm["losses"]

        vs_bw = bf_wait_30["comparisons"]["vs_best_sota"]
        pooled_bf_wait_wins += vs_bw["wins"]
        pooled_bf_wait_losses += vs_bw["losses"]

        vs_bj = bf_jct_30["comparisons"]["vs_best_sota"]
        pooled_bf_jct_wins += vs_bj["wins"]
        pooled_bf_jct_losses += vs_bj["losses"]

        vs_bm = bf_mk_30["comparisons"]["vs_best_sota"]
        pooled_bf_mk_wins += vs_bm["wins"]
        pooled_bf_mk_losses += vs_bm["losses"]

        # 정규화 상대 개선률 풀링 (각 시드별 (sota - adapt) / mean(sota))
        b_sota_name = blk_30["best_sota"]
        b_sota_mean = blk_30["scheduler_stats"][b_sota_name]["mean"]
        for s in UNSEEN_SEEDS_30:
            sb = rec_map[b_sota_name][s]["blocking"]["gpu_hour_blocking_rate"]
            ab = rec_map["Adaptive-B-Lindy"][s]["blocking"]["gpu_hour_blocking_rate"]
            pooled_rel_block_diffs.append((sb - ab) / max(b_sota_mean, 1e-9) * 100.0)

        fw_sota_name = fifo_wait_30["best_sota"]
        fw_sota_mean = fifo_wait_30["scheduler_stats"][fw_sota_name]["mean"]
        fj_sota_name = fifo_jct_30["best_sota"]
        fj_sota_mean = fifo_jct_30["scheduler_stats"][fj_sota_name]["mean"]
        fm_sota_name = fifo_mk_30["best_sota"]
        fm_sota_mean = fifo_mk_30["scheduler_stats"][fm_sota_name]["mean"]
        for s in UNSEEN_SEEDS_30:
            sf = rec_map[fw_sota_name][s]["fifo"]["mean_wait_time"]
            af = rec_map["Adaptive-B-Lindy"][s]["fifo"]["mean_wait_time"]
            pooled_rel_fifo_wait_diffs.append((sf - af) / max(fw_sota_mean, 1e-9) * 100.0)

            sj = rec_map[fj_sota_name][s]["fifo"]["mean_jct"]
            aj = rec_map["Adaptive-B-Lindy"][s]["fifo"]["mean_jct"]
            pooled_rel_fifo_jct_diffs.append((sj - aj) / max(fj_sota_mean, 1e-9) * 100.0)

            sm = rec_map[fm_sota_name][s]["fifo"]["makespan"]
            am = rec_map["Adaptive-B-Lindy"][s]["fifo"]["makespan"]
            pooled_rel_fifo_mk_diffs.append((sm - am) / max(fm_sota_mean, 1e-9) * 100.0)

        bw_sota_name = bf_wait_30["best_sota"]
        bw_sota_mean = bf_wait_30["scheduler_stats"][bw_sota_name]["mean"]
        bj_sota_name = bf_jct_30["best_sota"]
        bj_sota_mean = bf_jct_30["scheduler_stats"][bj_sota_name]["mean"]
        bm_sota_name = bf_mk_30["best_sota"]
        bm_sota_mean = bf_mk_30["scheduler_stats"][bm_sota_name]["mean"]
        for s in UNSEEN_SEEDS_30:
            sbw = rec_map[bw_sota_name][s]["backfill"]["mean_wait_time"]
            abw = rec_map["Adaptive-B-Lindy"][s]["backfill"]["mean_wait_time"]
            pooled_rel_bf_wait_diffs.append((sbw - abw) / max(bw_sota_mean, 1e-9) * 100.0)

            sbj = rec_map[bj_sota_name][s]["backfill"]["mean_jct"]
            abj = rec_map["Adaptive-B-Lindy"][s]["backfill"]["mean_jct"]
            pooled_rel_bf_jct_diffs.append((sbj - abj) / max(bj_sota_mean, 1e-9) * 100.0)

            sbm = rec_map[bm_sota_name][s]["backfill"]["makespan"]
            abm = rec_map["Adaptive-B-Lindy"][s]["backfill"]["makespan"]
            pooled_rel_bf_mk_diffs.append((sbm - abm) / max(bm_sota_mean, 1e-9) * 100.0)

        print(
            f"{w_name:<20} | {b_sota_name:<12} | {vs_b['base_mean']:8.3f} | "
            f"{blk_30['scheduler_stats']['EvoSTF_Dist']['mean']:7.3f} | "
            f"{blk_30['scheduler_stats']['B-LindyPack']['mean']:7.3f} | "
            f"{vs_b['cand_mean']:8.3f} | {vs_b['rel_improvement_pct']:+6.2f}% | "
            f"{vs_b['wins']:2d}/30 | {vs_b['p_value_t']:8.5f} | {vs_b['p_value_wilcoxon']:8.5f}"
        )

        report_workloads[w_name] = {
            "blocking_30": blk_30,
            "blocking_new20": blk_20,
            "stranded_30": str_30,
            "fifo_jct_30": fifo_jct_30,
            "fifo_gpu_weighted_jct_30": fifo_gwjct_30,
            "fifo_p95_jct_30": fifo_p95_30,
            "fifo_wait_30": fifo_wait_30,
            "fifo_wait_new20": fifo_wait_20,
            "fifo_makespan_30": fifo_mk_30,
            "fifo_stranded_30": fifo_str_30,
            "backfill_jct_30": bf_jct_30,
            "backfill_gpu_weighted_jct_30": bf_gwjct_30,
            "backfill_wait_30": bf_wait_30,
            "backfill_makespan_30": bf_mk_30,
            "online_diagnostics": {
                "blocking_rho_mean": float(np.mean(rhos_blk)),
                "blocking_rho_std": float(np.std(rhos_blk, ddof=1)),
                "blocking_alpha_mean": float(np.mean(alphas_blk)),
                "blocking_alpha_std": float(np.std(alphas_blk, ddof=1)),
                "blocking_gamma_mean": float(np.mean(gammas_blk)),
                "fifo_rho_mean": float(np.mean(rhos_fifo)),
                "fifo_alpha_mean": float(np.mean(alphas_fifo)),
                "fifo_gamma_mean": float(np.mean(gammas_fifo)),
            },
        }

    print("\n" + "=" * 116)
    print("  [표 2] Pending Queue (Strict FIFO) 30개 미지 시드 확증 결과 — Mean Wait Time / JCT / Makespan")
    print("=" * 116)
    print(
        f"{'Workload':<20} | {'Best SOTA':<11} | {'SOTA Wait':>9} | {'BL Wait':>8} | {'AdBL Wait':>9} | "
        f"{'Wait Impr':>9} | {'Wins':>6} | {'p(t)':>8} | {'p(W)':>8} | {'JCT(S->A)':>14} | {'Mkspn(S->A)':>13}"
    )
    print("-" * 116)
    for w_name in WORKLOAD_ORDER:
        fw = report_workloads[w_name]["fifo_wait_30"]
        fj = report_workloads[w_name]["fifo_jct_30"]
        fm = report_workloads[w_name]["fifo_makespan_30"]
        vs_fw = fw["comparisons"]["vs_best_sota"]
        sota_name = fw["best_sota"]
        bl_w = fw["scheduler_stats"]["B-LindyPack"]["mean"]
        sj = fj["scheduler_stats"][sota_name]["mean"]
        aj = fj["scheduler_stats"]["Adaptive-B-Lindy"]["mean"]
        sm = fm["scheduler_stats"][sota_name]["mean"]
        am = fm["scheduler_stats"]["Adaptive-B-Lindy"]["mean"]
        print(
            f"{w_name:<20} | {sota_name:<11} | {vs_fw['base_mean']:9.4f} | {bl_w:8.4f} | {vs_fw['cand_mean']:9.4f} | "
            f"{vs_fw['rel_improvement_pct']:+8.2f}% | {vs_fw['wins']:2d}/30 | {vs_fw['p_value_t']:8.5f} | "
            f"{vs_fw['p_value_wilcoxon']:8.5f} | {sj:6.3f}->{aj:6.3f} | {sm:5.2f}->{am:5.2f}"
        )

    print("\n" + "=" * 124)
    print("  [표 3] Pending Queue (Starvation-Free Backfill) 30개 미지 시드 확증 결과 — Mean Wait / GW-JCT / Makespan")
    print("=" * 124)
    print(
        f"{'Workload':<20} | {'Best SOTA':<11} | {'SOTA Wait':>9} | {'BL Wait':>8} | {'AdBL Wait':>9} | "
        f"{'Wait Impr':>9} | {'Wins':>6} | {'p(t)':>8} | {'p(W)':>8} | {'GW-JCT(S->A)':>14} | {'Mkspn(S->A)':>13}"
    )
    print("-" * 124)
    for w_name in WORKLOAD_ORDER:
        bw = report_workloads[w_name]["backfill_wait_30"]
        bg = report_workloads[w_name]["backfill_gpu_weighted_jct_30"]
        bm = report_workloads[w_name]["backfill_makespan_30"]
        vs_bw = bw["comparisons"]["vs_best_sota"]
        sota_name = bw["best_sota"]
        bl_w = bw["scheduler_stats"]["B-LindyPack"]["mean"]
        sg = bg["scheduler_stats"][sota_name]["mean"]
        ag = bg["scheduler_stats"]["Adaptive-B-Lindy"]["mean"]
        sm = bm["scheduler_stats"][sota_name]["mean"]
        am = bm["scheduler_stats"]["Adaptive-B-Lindy"]["mean"]
        print(
            f"{w_name:<20} | {sota_name:<11} | {vs_bw['base_mean']:9.4f} | {bl_w:8.4f} | {vs_bw['cand_mean']:9.4f} | "
            f"{vs_bw['rel_improvement_pct']:+8.2f}% | {vs_bw['wins']:2d}/30 | {vs_bw['p_value_t']:8.5f} | "
            f"{vs_bw['p_value_wilcoxon']:8.5f} | {sg:6.3f}->{ag:6.3f} | {sm:5.2f}->{am:5.2f}"
        )

    print("\n" + "=" * 98)
    print("  [표 4] 온라인 상관(Corr) + CV 유효성 게이트(gamma) 자동 적응 궤적 및 단일 스레드 지연시간")
    print("=" * 98)
    for w_name in WORKLOAD_ORDER:
        od = report_workloads[w_name]["online_diagnostics"]
        print(
            f"  {w_name:<20} | Blocking: rho={od['blocking_rho_mean']:+.3f} +/- {od['blocking_rho_std']:.3f}, "
            f"alpha={od['blocking_alpha_mean']:.3f}, gamma={od['blocking_gamma_mean']:.3f} | "
            f"FIFO: rho={od['fifo_rho_mean']:+.3f}, alpha={od['fifo_alpha_mean']:.3f}"
        )

    print("\n  단일 스레드 평균 의사결정 지연시간 (us):")
    for label, row in latency_single.items():
        fgd_l = row["FGD_Adaptive"]
        ad_l = row["Adaptive-B-Lindy"]
        bl_l = row["B-LindyPack"]
        print(
            f"  {label:<24} | L2Norm={row['L2Norm']:5.1f} | B-Lindy={bl_l:5.1f} | "
            f"Adaptive-BL={ad_l:5.1f} | FGD={fgd_l:6.1f} (vs FGD {fgd_l/max(ad_l,0.1):.1f}x 빠름)"
        )

    # 전체 180-시드 풀링(Pooled Grand Statistical Summary)
    def _pack_pooled(wins: int, losses: int, rel_diffs: List[float]) -> Dict[str, Any]:
        arr = np.asarray(rel_diffs, dtype=np.float64)
        st = paired_summary_stats(arr, np.zeros(len(arr), dtype=np.float64))
        return {
            "wins": wins,
            "losses": losses,
            "sign_p_value": exact_sign_test_pvalue(wins, losses),
            "mean_rel_improvement_pct": st["diff_mean"],
            "t_stat": st["t_stat"],
            "p_value_t": st["p_value_t"],
            "p_value_wilcoxon": st["p_value_wilcoxon"],
        }

    pooled_summary = {
        "total_unseen_runs": len(WORKLOAD_ORDER) * len(UNSEEN_SEEDS_30),
        "blocking_vs_best_sota": _pack_pooled(pooled_block_wins, pooled_block_losses, pooled_rel_block_diffs),
        "fifo_wait_vs_best_sota": _pack_pooled(pooled_fifo_wait_wins, pooled_fifo_wait_losses, pooled_rel_fifo_wait_diffs),
        "fifo_jct_vs_best_sota": _pack_pooled(pooled_fifo_jct_wins, pooled_fifo_jct_losses, pooled_rel_fifo_jct_diffs),
        "fifo_makespan_vs_best_sota": _pack_pooled(pooled_fifo_mk_wins, pooled_fifo_mk_losses, pooled_rel_fifo_mk_diffs),
        "backfill_wait_vs_best_sota": _pack_pooled(pooled_bf_wait_wins, pooled_bf_wait_losses, pooled_rel_bf_wait_diffs),
        "backfill_jct_vs_best_sota": _pack_pooled(pooled_bf_jct_wins, pooled_bf_jct_losses, pooled_rel_bf_jct_diffs),
        "backfill_makespan_vs_best_sota": _pack_pooled(pooled_bf_mk_wins, pooled_bf_mk_losses, pooled_rel_bf_mk_diffs),
    }

    print("\n" + "=" * 98)
    print("  [전체 180회 미지 실행 통합 통계 검정 (6 워크로드 × 30 미지 시드)]")
    print("=" * 98)
    for label, key in [
        ("Blocking (vs 부문별 최강 SOTA)     ", "blocking_vs_best_sota"),
        ("FIFO Wait (vs 부문별 최강 SOTA)    ", "fifo_wait_vs_best_sota"),
        ("FIFO JCT (vs 부문별 최강 SOTA)     ", "fifo_jct_vs_best_sota"),
        ("FIFO Makespan (vs 부문별 최강 SOTA)", "fifo_makespan_vs_best_sota"),
        ("Backfill Wait (vs 부문별 최강 SOTA)", "backfill_wait_vs_best_sota"),
        ("Backfill JCT (vs 부문별 최강 SOTA) ", "backfill_jct_vs_best_sota"),
        ("Backfill Mksp (vs 부문별 최강 SOTA)", "backfill_makespan_vs_best_sota"),
    ]:
        p = pooled_summary[key]
        print(
            f"  {label}: {p['wins']}/{p['wins']+p['losses']}승 "
            f"(sign p={p['sign_p_value']:.2e}), 평균 상대 개선 {p['mean_rel_improvement_pct']:+.2f}%, "
            f"t={p['t_stat']:.2f}, p(t)={p['p_value_t']:.2e}, p(W)={p['p_value_wilcoxon']:.2e}"
        )

    # 별도 스트레스 검증: IFR 반린디 OOD (Uniform(1.8, 2.2), 30개 미지 시드) — CV 게이트 ON vs OFF
    print("\n[반린디(IFR) 스트레스 OOD 30-시드 안전성 검증 중...]", flush=True)
    with mp.Pool(processes=n_workers) as pool:
        ifr_raw = pool.map(_worker_run_ifr_seed, UNSEEN_SEEDS_30)
    ifr_by_sch: Dict[str, Dict[str, List[float]]] = {}
    ifr_gammas: List[float] = []
    for r in ifr_raw:
        for s_name, vals in r["schedulers"].items():
            if s_name not in ifr_by_sch:
                ifr_by_sch[s_name] = {"blocking": [], "fifo_wait": [], "backfill_wait": [], "backfill_makespan": []}
            for mk in ("blocking", "fifo_wait", "backfill_wait", "backfill_makespan"):
                ifr_by_sch[s_name][mk].append(vals[mk])
        ifr_gammas.append(r["schedulers"]["Adaptive-B-Lindy"]["diag"]["gamma_mean"])

    ifr_summary: Dict[str, Any] = {
        "gamma_mean": float(np.mean(ifr_gammas)),
        "scheduler_means": {
            s_name: {mk: float(np.mean(arr)) for mk, arr in m_map.items()}
            for s_name, m_map in ifr_by_sch.items()
        },
        "cv_gate_rescue_vs_nocv": {
            mk: paired_summary_stats(
                np.asarray(ifr_by_sch["Adaptive-NoCVGate"][mk]),
                np.asarray(ifr_by_sch["Adaptive-B-Lindy"][mk]),
            )
            for mk in ("blocking", "fifo_wait", "backfill_wait", "backfill_makespan")
        },
    }
    print(f"  IFR Anti-Lindy OOD (30 unseen seeds, gamma={ifr_summary['gamma_mean']:.3f}):")
    for s_name, sm in ifr_summary["scheduler_means"].items():
        print(
            f"    {s_name:<18} | Block={sm['blocking']:6.3f}% | FIFO_W={sm['fifo_wait']:7.4f} | "
            f"BF_W={sm['backfill_wait']:7.4f} | BF_Mk={sm['backfill_makespan']:6.2f}"
        )
    r_blk = ifr_summary["cv_gate_rescue_vs_nocv"]["blocking"]
    r_fw = ifr_summary["cv_gate_rescue_vs_nocv"]["fifo_wait"]
    print(
        f"  >> CV 게이트 구출 효과 (NoCVGate -> Adaptive-B-Lindy): "
        f"Block {r_blk['rel_improvement_pct']:+.2f}% (p_t={r_blk['p_value_t']:.2e}, {r_blk['wins']}/30승), "
        f"FIFO Wait {r_fw['rel_improvement_pct']:+.2f}% (p_t={r_fw['p_value_t']:.2e}, {r_fw['wins']}/30승)"
    )

    out_path = root / "phase5_30seed_report.json"
    payload = {
        "unseen_seeds_30": UNSEEN_SEEDS_30,
        "brand_new_seeds_20": BRAND_NEW_SEEDS_20,
        "base_theta": DEFAULT_BASE_THETA.tolist(),
        "adaptive_config": AdaptiveBLindyPackScheduler().adapt_cfg.__dict__,
        "workloads": report_workloads,
        "pooled_180_summary": pooled_summary,
        "ifr_antilindy_ood_30": ifr_summary,
        "single_thread_latency_us": latency_single,
        "elapsed_sec": time.perf_counter() - t0,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[5단계 30-시드 확증 리포트 저장 완료] {out_path} (총 {time.perf_counter() - t0:.1f}초)")


if __name__ == "__main__":
    main()
