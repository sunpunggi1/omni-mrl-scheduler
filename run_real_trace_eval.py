"""
6단계(B안): 실제 공개 기업 GPU 클러스터 트레이스 Zero-Shot 실증 벤치마크
(Alibaba cluster-trace-gpu-v2023 + SenseTime HeliosData 4개 클러스터)

원칙:
- 4~5단계에서 확정된 B-LindyPack 및 Adaptive-B-Lindy 파라미터를 단 1개도 재튜닝하지 않고
  그대로(Zero-Shot) 실제 기업 프로덕션 로그에 투입한다.
- Alibaba FGD (USENIX ATC '23) 원논문의 오라클 분포 세팅(FGD_OracleStatic) 및 FGD_Adaptive와
  동일 조건에서 직접 비교한다.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from adaptive_lindy import AdaptiveBLindyPackScheduler, DEFAULT_BASE_THETA
from b_lindy_discovery import BoundedLindyPackScheduler
from baselines import (
    BestFitScheduler,
    DotProductScheduler,
    FGDScheduler,
    FirstFitScheduler,
    L2NormScheduler,
)
from simulator import (
    BaseScheduler,
    ClusterConfig,
    Job,
    run_queue_simulation,
    run_simulation,
)
from stats_utils import paired_summary_stats
from trace_loader import (
    build_empirical_fgd_distribution,
    extract_helios_windows,
    load_alibaba_2023,
    load_helios_cluster,
    scale_heterogeneous_cluster,
    slice_active_trace,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ALIBABA_DIR = DATA_DIR / "alibaba_2023"
HELIOS_DIR = DATA_DIR / "helios"

SCHED_ORDER = [
    "FirstFit",
    "BestFit",
    "DotProduct",
    "L2Norm",
    "FGD_OracleStatic",
    "FGD_Adaptive",
    "B-LindyPack",
    "Adaptive-B-Lindy",
    "Adaptive-B-Lindy-Het",
]
SOTA_NAMES = ["FirstFit", "BestFit", "DotProduct", "L2Norm", "FGD_OracleStatic", "FGD_Adaptive"]


def _make_scheduler(name: str, fgd_reqs: np.ndarray, fgd_probs: np.ndarray) -> BaseScheduler:
    if name == "FirstFit":
        return FirstFitScheduler()
    if name == "BestFit":
        return BestFitScheduler()
    if name == "DotProduct":
        return DotProductScheduler()
    if name == "L2Norm":
        return L2NormScheduler()
    if name == "FGD_OracleStatic":
        return FGDScheduler(adaptive=False, custom_reqs=fgd_reqs, custom_probs=fgd_probs, name="FGD_OracleStatic")
    if name == "FGD_Adaptive":
        return FGDScheduler(adaptive=True, custom_reqs=fgd_reqs, custom_probs=fgd_probs, name="FGD_Adaptive")
    if name == "B-LindyPack":
        return BoundedLindyPackScheduler(DEFAULT_BASE_THETA, name="B-LindyPack")
    if name == "Adaptive-B-Lindy":
        return AdaptiveBLindyPackScheduler(record_diagnostics=True, name="Adaptive-B-Lindy", hetero_aware=False)
    if name == "Adaptive-B-Lindy-Het":
        return AdaptiveBLindyPackScheduler(record_diagnostics=True, name="Adaptive-B-Lindy-Het", hetero_aware=True)
    raise ValueError(f"알 수 없는 스케줄러: {name}")


def _eval_unit(
    task: Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str],
) -> Dict[str, object]:
    """단일 (시나리오, 인덱스, 설정, 작업, FGD 분포, 스케줄러명) 작업 3모드 평가."""
    scenario_id, item_idx, cfg, jobs, fgd_reqs, fgd_probs, sched_name = task
    sch = _make_scheduler(sched_name, fgd_reqs, fgd_probs)

    m_blk = run_simulation(jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx)
    diag_blk = sch.get_diagnostics_summary() if isinstance(sch, AdaptiveBLindyPackScheduler) else {}

    m_fifo = run_queue_simulation(
        jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx, queue_policy="fifo"
    )
    diag_fifo = sch.get_diagnostics_summary() if isinstance(sch, AdaptiveBLindyPackScheduler) else {}

    m_bf = run_queue_simulation(
        jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx, queue_policy="backfill"
    )

    return {
        "scenario": scenario_id,
        "item_idx": item_idx,
        "scheduler": sched_name,
        "blk_gh": m_blk.gpu_hour_blocking_rate,
        "blk_job": m_blk.job_blocking_rate,
        "blk_stranded": m_blk.avg_stranded_gpu_ratio,
        "blk_active_hr": m_blk.active_node_hours,
        "blk_lat_us": m_blk.avg_decision_us,
        "fifo_wait": m_fifo.mean_wait_time,
        "fifo_jct": m_fifo.mean_jct,
        "fifo_gw_jct": m_fifo.gpu_weighted_jct,
        "fifo_p95_jct": m_fifo.p95_jct,
        "fifo_makespan": m_fifo.makespan,
        "bf_wait": m_bf.mean_wait_time,
        "bf_jct": m_bf.mean_jct,
        "bf_gw_jct": m_bf.gpu_weighted_jct,
        "bf_p95_jct": m_bf.p95_jct,
        "bf_makespan": m_bf.makespan,
        "diag_blk": diag_blk,
        "diag_fifo": diag_fifo,
    }


def _summarize_group(
    rows_by_sched: Dict[str, List[dict]],
) -> Dict[str, object]:
    """동일 시나리오 내 복수 시드/구간 결과를 집계하고 쌍체 통계 검정을 수행한다."""
    summary_sched: Dict[str, Dict[str, float]] = {}
    for s_name in SCHED_ORDER:
        rows = sorted(rows_by_sched[s_name], key=lambda r: int(r["item_idx"]))
        summary_sched[s_name] = {
            "n": len(rows),
            "blk_gh_mean": float(np.mean([r["blk_gh"] for r in rows])),
            "blk_gh_std": float(np.std([r["blk_gh"] for r in rows])),
            "blk_stranded_mean": float(np.mean([r["blk_stranded"] for r in rows])),
            "blk_lat_us_mean": float(np.mean([r["blk_lat_us"] for r in rows])),
            "fifo_wait_mean": float(np.mean([r["fifo_wait"] for r in rows])),
            "fifo_jct_mean": float(np.mean([r["fifo_jct"] for r in rows])),
            "fifo_gw_jct_mean": float(np.mean([r["fifo_gw_jct"] for r in rows])),
            "fifo_makespan_mean": float(np.mean([r["fifo_makespan"] for r in rows])),
            "bf_wait_mean": float(np.mean([r["bf_wait"] for r in rows])),
            "bf_jct_mean": float(np.mean([r["bf_jct"] for r in rows])),
            "bf_gw_jct_mean": float(np.mean([r["bf_gw_jct"] for r in rows])),
            "bf_makespan_mean": float(np.mean([r["bf_makespan"] for r in rows])),
        }

    def _paired(metric_key: str, base_name: str, cand_name: str = "Adaptive-B-Lindy-Het") -> Dict[str, float]:
        b_rows = sorted(rows_by_sched[base_name], key=lambda r: int(r["item_idx"]))
        c_rows = sorted(rows_by_sched[cand_name], key=lambda r: int(r["item_idx"]))
        b_arr = np.array([r[metric_key] for r in b_rows], dtype=np.float64)
        c_arr = np.array([r[metric_key] for r in c_rows], dtype=np.float64)
        return paired_summary_stats(b_arr, c_arr)

    best_blk_sota = min(SOTA_NAMES, key=lambda n: summary_sched[n]["blk_gh_mean"])
    best_fifo_sota = min(SOTA_NAMES, key=lambda n: summary_sched[n]["fifo_wait_mean"])
    best_bf_sota = min(SOTA_NAMES, key=lambda n: summary_sched[n]["bf_wait_mean"])

    adapt_rows = rows_by_sched["Adaptive-B-Lindy-Het"]
    diag_blk_list = [r["diag_blk"] for r in adapt_rows if r.get("diag_blk")]
    diag_fifo_list = [r["diag_fifo"] for r in adapt_rows if r.get("diag_fifo")]

    def _mean_diag(dlist: List[dict], key: str) -> float:
        vals = [d[key] for d in dlist if key in d]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "schedulers": summary_sched,
        "best_sota": {
            "blocking": best_blk_sota,
            "fifo_wait": best_fifo_sota,
            "backfill_wait": best_bf_sota,
        },
        "comparisons": {
            "blocking_vs_best_sota": _paired("blk_gh", best_blk_sota),
            "blocking_vs_l2norm": _paired("blk_gh", "L2Norm"),
            "blocking_vs_fgd_oracle": _paired("blk_gh", "FGD_OracleStatic"),
            "blocking_vs_fgd_adaptive": _paired("blk_gh", "FGD_Adaptive"),
            "blocking_vs_blindy": _paired("blk_gh", "B-LindyPack"),
            "blocking_vs_v5_adapt": _paired("blk_gh", "Adaptive-B-Lindy"),
            "fifo_wait_vs_best_sota": _paired("fifo_wait", best_fifo_sota),
            "fifo_wait_vs_l2norm": _paired("fifo_wait", "L2Norm"),
            "fifo_wait_vs_fgd_oracle": _paired("fifo_wait", "FGD_OracleStatic"),
            "bf_wait_vs_best_sota": _paired("bf_wait", best_bf_sota),
            "bf_wait_vs_l2norm": _paired("bf_wait", "L2Norm"),
            "bf_wait_vs_fgd_oracle": _paired("bf_wait", "FGD_OracleStatic"),
            "fifo_makespan_vs_best_sota": _paired("fifo_makespan", best_fifo_sota),
            "bf_makespan_vs_best_sota": _paired("bf_makespan", best_bf_sota),
        },
        "diagnostics": {
            "blk_rho_mean": _mean_diag(diag_blk_list, "rho_mean"),
            "blk_alpha_mean": _mean_diag(diag_blk_list, "alpha_mean"),
            "blk_gamma_mean": _mean_diag(diag_blk_list, "gamma_mean"),
            "fifo_rho_mean": _mean_diag(diag_fifo_list, "rho_mean"),
            "fifo_alpha_mean": _mean_diag(diag_fifo_list, "alpha_mean"),
            "fifo_gamma_mean": _mean_diag(diag_fifo_list, "gamma_mean"),
        },
    }


def _print_group_summary(title: str, grp: Dict[str, object]) -> None:
    print(f"\n==================== {title} ====================")
    print(
        f"{'Scheduler':<17} | {'Blk(%)':>8} | {'Stranded(%)':>11} | "
        f"{'FIFO Wait':>10} | {'BF Wait':>10} | {'BF JCT':>9} | {'Lat(us)':>8}"
    )
    print("-" * 88)
    scheds = grp["schedulers"]
    for s in SCHED_ORDER:
        m = scheds[s]
        print(
            f"{s:<17} | {m['blk_gh_mean']:8.3f} | {m['blk_stranded_mean']:11.2f} | "
            f"{m['fifo_wait_mean']:10.4f} | {m['bf_wait_mean']:10.4f} | "
            f"{m['bf_jct_mean']:9.3f} | {m['blk_lat_us_mean']:8.1f}"
        )
    bs = grp["best_sota"]
    cmp_b = grp["comparisons"]["blocking_vs_best_sota"]
    cmp_f = grp["comparisons"]["fifo_wait_vs_best_sota"]
    cmp_bf = grp["comparisons"]["bf_wait_vs_best_sota"]
    cmp_fgd = grp["comparisons"]["blocking_vs_fgd_oracle"]
    cmp_v5 = grp["comparisons"]["blocking_vs_v5_adapt"]
    d = grp["diagnostics"]
    print(
        f"  >> Blk vs {bs['blocking']}: {cmp_b['rel_improvement_pct']:+.2f}% "
        f"(W/L={cmp_b['wins']}/{cmp_b['losses']}, p_t={cmp_b['p_value_t']:.4g}, p_W={cmp_b['p_value_wilcoxon']:.4g})"
        f" | vs FGD_Oracle: {cmp_fgd['rel_improvement_pct']:+.2f}% | vs v5: {cmp_v5['rel_improvement_pct']:+.2f}%"
    )
    print(
        f"  >> FIFO Wait vs {bs['fifo_wait']}: {cmp_f['rel_improvement_pct']:+.2f}% "
        f"(p_W={cmp_f['p_value_wilcoxon']:.4g}) | "
        f"BF Wait vs {bs['backfill_wait']}: {cmp_bf['rel_improvement_pct']:+.2f}% "
        f"(p_W={cmp_bf['p_value_wilcoxon']:.4g})"
    )
    print(
        f"  >> 게이트 실측: rho={d['blk_rho_mean']:+.3f}, alpha={d['blk_alpha_mean']:.3f}, gamma={d['blk_gamma_mean']:.3f}"
    )


def main() -> None:
    t_start = time.perf_counter()
    node_csv = ALIBABA_DIR / "openb_node_list_gpu_node.csv"
    pod_default_csv = ALIBABA_DIR / "openb_pod_list_default.csv"
    pod_spec33_csv = ALIBABA_DIR / "openb_pod_list_gpuspec33.csv"

    print("[1/4] 실제 공개 기업 트레이스 로드 및 물리 감사 중...")
    ali_cfg, ali_def_jobs, ali_def_audit = load_alibaba_2023(node_csv, pod_default_csv, use_gpu_spec=True)
    _, ali_spec_jobs, ali_spec_audit = load_alibaba_2023(node_csv, pod_spec33_csv, use_gpu_spec=True)

    # Alibaba 활성 집중 구간 추출 (초반 5% 2,800시간 공백 제외, 95% 파드 포함)
    ali_def_active, ali_def_act_conc = slice_active_trace(ali_def_jobs, start_quantile=0.05)
    ali_spec_active, ali_spec_act_conc = slice_active_trace(ali_spec_jobs, start_quantile=0.05)
    print(
        f"  - Alibaba Default 전체({len(ali_def_jobs)}건, avg={ali_def_audit.mean_concurrent_gpus:.1f} GPU) "
        f"-> 활성 95% 구간({len(ali_def_active)}건, avg={ali_def_act_conc:.1f} GPU)"
    )
    print(
        f"  - Alibaba GPUSpec33 전체({len(ali_spec_jobs)}건) "
        f"-> 활성 95% 구간({len(ali_spec_active)}건, avg={ali_spec_act_conc:.1f} GPU)"
    )

    # (A1) 원본 1:1 Native Replay (전체 1,213 이기종 노드, 6,212 GPU)
    print("\n[2/4] Alibaba 2023 원본 1:1 Native Replay (1,213 이기종 노드, 6,212 GPU) 실행 중...")
    def_reqs, def_probs = build_empirical_fgd_distribution(ali_def_jobs)
    native_res: Dict[str, dict] = {}
    for s_name in SCHED_ORDER:
        sch = _make_scheduler(s_name, def_reqs, def_probs)
        m_b = run_simulation(ali_def_jobs, sch, config=ali_cfg, workload_name="Alibaba_Native_1to1", seed=0)
        m_q = run_queue_simulation(
            ali_def_jobs, sch, config=ali_cfg, workload_name="Alibaba_Native_1to1", seed=0, queue_policy="fifo"
        )
        native_res[s_name] = {
            "blk_gh": m_b.gpu_hour_blocking_rate,
            "fifo_wait": m_q.mean_wait_time,
            "fifo_jct": m_q.mean_jct,
            "lat_us": m_b.avg_decision_us,
        }
    print("  - Native 1:1 결과 (Adaptive-B-Lindy):", native_res["Adaptive-B-Lindy"])

    # 고부하 스트레스 태스크 구성 (Alibaba Scaled 10시드 × 2트레이스 + Helios 20구간)
    tasks: List[Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str]] = []
    audits: Dict[str, dict] = {
        "Alibaba_Default": ali_def_audit.to_dict(),
        "Alibaba_GPUSpec33": ali_spec_audit.to_dict(),
    }

    # (A2) Alibaba 부하 곡선 스윕 (Load Curve Sweep: 4개 비율 × 3개 서브샘플 시드 = 각 12회)
    # - Default_Sweep: keep_ratio in [0.0025, 0.0030, 0.0035, 0.0045]
    # - GPUSpec33_Sweep: keep_ratio in [0.0050, 0.0065, 0.0080, 0.0100]
    ali_seeds = [101, 202, 303]
    def_ratios = [0.0025, 0.0030, 0.0035, 0.0045]
    spec_ratios = [0.0050, 0.0065, 0.0080, 0.0100]
    spec_reqs, spec_probs = build_empirical_fgd_distribution(ali_spec_active)
    def_act_reqs, def_act_probs = build_empirical_fgd_distribution(ali_def_active)

    def_idx = 0
    for kr in def_ratios:
        for s in ali_seeds:
            cfg_sub_def = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_def_active)
            for s_name in SCHED_ORDER:
                tasks.append((
                    "Alibaba_Default_Sweep", def_idx, cfg_sub_def, ali_def_active, def_act_reqs, def_act_probs, s_name
                ))
            def_idx += 1

    spec_idx = 0
    for kr in spec_ratios:
        for s in ali_seeds:
            cfg_sub_spec = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_spec_active)
            for s_name in SCHED_ORDER:
                tasks.append((
                    "Alibaba_GPUSpec33_Sweep", spec_idx, cfg_sub_spec, ali_spec_active, spec_reqs, spec_probs, s_name
                ))
            spec_idx += 1

    # (B) SenseTime Helios 4개 클러스터 (각 5개 비중첩 실제 피크 구간 = 20개 독립 실제 구간)
    helios_clusters = ["Venus", "Earth", "Saturn", "Uranus"]
    helios_meta: Dict[str, list] = {}
    for cl in helios_clusters:
        h_jobs, h_audit = load_helios_cluster(HELIOS_DIR / cl / "cluster_log.csv", cl)
        audits[f"Helios_{cl}"] = h_audit.to_dict()
        wins = extract_helios_windows(h_jobs, num_windows=5, jobs_per_window=3000, target_load=0.85, min_span_hours=2.0)
        helios_meta[cl] = [
            {"window_idx": w_i, "num_nodes": w_cfg.num_nodes, "mean_conc_gpu": round(mc, 2)}
            for w_i, (w_cfg, _, mc) in enumerate(wins)
        ]
        print(f"  - Helios {cl}: valid={h_audit.valid_gpu_jobs}, corr={h_audit.corr_log_gpu_dur:+.3f}, "
              f"cv={h_audit.dur_cv:.2f}, windows={[(w[0].num_nodes, round(w[2],1)) for w in wins]}")
        for w_i, (w_cfg, w_jobs, _) in enumerate(wins):
            w_reqs, w_probs = build_empirical_fgd_distribution(w_jobs)
            for s_name in SCHED_ORDER:
                tasks.append((
                    f"Helios_{cl}", w_i, w_cfg, w_jobs, w_reqs, w_probs, s_name
                ))

    n_workers = max(1, min(8, (mp.cpu_count() or 4) - 1))
    print(f"\n[3/4] 총 {len(tasks)}개 실제 트레이스 시뮬레이션 단위(×3모드={len(tasks)*3}회) 병렬 실행 중 ({n_workers} workers)...")
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_eval_unit, tasks)

    # 시나리오별 그룹화
    grouped: Dict[str, Dict[str, List[dict]]] = {}
    for r in results:
        sc = str(r["scenario"])
        sn = str(r["scheduler"])
        grouped.setdefault(sc, {k: [] for k in SCHED_ORDER})[sn].append(r)

    # Helios 4개 클러스터 20개 구간 통합 그룹 생성 (고유 item_idx 부여)
    helios_pooled: Dict[str, List[dict]] = {k: [] for k in SCHED_ORDER}
    for cl_idx, cl in enumerate(helios_clusters):
        sc = f"Helios_{cl}"
        for sn in SCHED_ORDER:
            for r in grouped[sc][sn]:
                r_copy = dict(r)
                r_copy["item_idx"] = cl_idx * 10 + int(r["item_idx"])
                helios_pooled[sn].append(r_copy)

    print("\n[4/4] 실제 기업 트레이스 실증 요약")
    report_groups: Dict[str, object] = {}
    for sc in ["Alibaba_Default_Sweep", "Alibaba_GPUSpec33_Sweep"] + [f"Helios_{c}" for c in helios_clusters]:
        grp = _summarize_group(grouped[sc])
        report_groups[sc] = grp
        _print_group_summary(sc, grp)

    helios_pooled_grp = _summarize_group(helios_pooled)
    report_groups["Helios_Pooled_20"] = helios_pooled_grp
    _print_group_summary("Helios_Pooled_20 (4 Clusters x 5 Windows = 20 Real Production Windows)", helios_pooled_grp)

    elapsed = time.perf_counter() - t_start
    payload = {
        "audits": audits,
        "alibaba_native_1to1": native_res,
        "helios_windows_meta": helios_meta,
        "scenarios": report_groups,
        "elapsed_sec": elapsed,
    }
    out_path = BASE_DIR / "real_trace_report.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[완료] 소요 시간: {elapsed:.1f}초 | 저장: {out_path}")


if __name__ == "__main__":
    main()
