"""
7단계(D안): 통합 스케줄러 Omni-Lindy (v7.0) 실제 기업 트레이스 실증 벤치마크

Part A — Alibaba PAI 2023 이기종 돌파 검증:
  - Alibaba_Default_Sweep (4비율 x 3시드 = 12회) & Alibaba_GPUSpec33_Sweep (4비율 x 3시드 = 12회)
  - 9종 스케줄러 x 3모드 (Blocking, Strict FIFO, Starvation-Free Backfill)
Part B — SenseTime Helios 풀 클러스터(단일 + 멀티노드 16~64 GPU) 20개 실제 구간:
  - Venus, Earth, Saturn, Uranus x 5 비중첩 피크 구간 = 20개 실제 프로덕션 구간
  - 9종 스케줄러(현업 RackBestFit 포함) x 4모드:
    1) Blocking (전체/멀티노드 거절률 %, 단일 랙 집적률 %)
    2) Strict FIFO (Wait, GW-JCT, Locality %)
    3) Starvation-Free Backfill (Wait, GW-JCT, Locality %)
    4) Topology-Coupled Backfill (cross_rack_penalty=0.08 실전 Wait, GW-JCT)
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
    RackBestFitScheduler,
)
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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ALIBABA_DIR = DATA_DIR / "alibaba_2023"
HELIOS_DIR = DATA_DIR / "helios"

ALI_SCHED_ORDER = [
    "FirstFit",
    "BestFit",
    "DotProduct",
    "L2Norm",
    "FGD_OracleStatic",
    "FGD_Adaptive",
    "B-LindyPack",
    "Adaptive-B-Lindy",
    "Omni-Lindy",
]
ALI_SOTA_NAMES = ["FirstFit", "BestFit", "DotProduct", "L2Norm", "FGD_OracleStatic", "FGD_Adaptive"]

HEL_SCHED_ORDER = [
    "FirstFit",
    "BestFit",
    "DotProduct",
    "L2Norm",
    "RackBestFit",
    "FGD_OracleStatic",
    "FGD_Adaptive",
    "B-LindyPack",
    "Omni-Lindy",
]
HEL_SOTA_NAMES = ["FirstFit", "BestFit", "DotProduct", "L2Norm", "RackBestFit", "FGD_OracleStatic", "FGD_Adaptive"]


def _make_scheduler(name: str, fgd_reqs: np.ndarray, fgd_probs: np.ndarray) -> BaseScheduler:
    if name == "FirstFit":
        return FirstFitScheduler()
    if name == "BestFit":
        return BestFitScheduler()
    if name == "DotProduct":
        return DotProductScheduler()
    if name == "L2Norm":
        return L2NormScheduler()
    if name == "RackBestFit":
        return RackBestFitScheduler()
    if name == "FGD_OracleStatic":
        return FGDScheduler(adaptive=False, custom_reqs=fgd_reqs, custom_probs=fgd_probs, name="FGD_OracleStatic")
    if name == "FGD_Adaptive":
        return FGDScheduler(adaptive=True, custom_reqs=fgd_reqs, custom_probs=fgd_probs, name="FGD_Adaptive")
    if name == "B-LindyPack":
        return BoundedLindyPackScheduler(DEFAULT_BASE_THETA, name="B-LindyPack")
    if name == "Adaptive-B-Lindy":
        return AdaptiveBLindyPackScheduler(record_diagnostics=True, name="Adaptive-B-Lindy", hetero_aware=False, topology_aware=False)
    if name == "Omni-Lindy":
        return AdaptiveBLindyPackScheduler(record_diagnostics=True, name="Omni-Lindy", hetero_aware=True, topology_aware=True)
    raise ValueError(f"알 수 없는 스케줄러: {name}")


def _eval_ali_unit(
    task: Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str],
) -> Dict[str, object]:
    """Alibaba 이기종 스윕 단위 작업 (3모드 평가)."""
    scenario_id, item_idx, cfg, jobs, fgd_reqs, fgd_probs, sched_name = task
    sch = _make_scheduler(sched_name, fgd_reqs, fgd_probs)

    m_blk = run_simulation(jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx)
    diag_blk = sch.get_diagnostics_summary() if isinstance(sch, AdaptiveBLindyPackScheduler) else {}
    m_fifo = run_queue_simulation(jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx, queue_policy="fifo")
    m_bf = run_queue_simulation(jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx, queue_policy="backfill")

    return {
        "scenario": scenario_id,
        "item_idx": item_idx,
        "scheduler": sched_name,
        "blk_gh": m_blk.gpu_hour_blocking_rate,
        "blk_stranded": m_blk.avg_stranded_gpu_ratio,
        "blk_lat_us": m_blk.avg_decision_us,
        "fifo_wait": m_fifo.mean_wait_time,
        "fifo_jct": m_fifo.mean_jct,
        "fifo_gw_jct": m_fifo.gpu_weighted_jct,
        "bf_wait": m_bf.mean_wait_time,
        "bf_jct": m_bf.mean_jct,
        "bf_gw_jct": m_bf.gpu_weighted_jct,
        "diag_blk": diag_blk,
    }


def _eval_hel_unit(
    task: Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str],
) -> Dict[str, object]:
    """Helios 멀티노드 단위 작업 (4모드 평가)."""
    scenario_id, item_idx, cfg, jobs, fgd_reqs, fgd_probs, sched_name = task
    sch = _make_scheduler(sched_name, fgd_reqs, fgd_probs)

    m_blk = run_simulation(jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx)
    diag_blk = sch.get_diagnostics_summary() if isinstance(sch, AdaptiveBLindyPackScheduler) else {}
    m_fifo = run_queue_simulation(jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx, queue_policy="fifo")
    m_bf = run_queue_simulation(jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx, queue_policy="backfill")
    m_bf_topo = run_queue_simulation(
        jobs, sch, config=cfg, workload_name=scenario_id, seed=item_idx, queue_policy="backfill", cross_rack_penalty=0.08
    )

    return {
        "scenario": scenario_id,
        "item_idx": item_idx,
        "scheduler": sched_name,
        "blk_gh": m_blk.gpu_hour_blocking_rate,
        "blk_multi_gh": m_blk.multinode_blocking_rate,
        "blk_locality": m_blk.single_rack_locality_rate,
        "blk_racks": m_blk.avg_racks_spanned,
        "blk_stranded": m_blk.avg_stranded_gpu_ratio,
        "blk_lat_us": m_blk.avg_decision_us,
        "fifo_wait": m_fifo.mean_wait_time,
        "fifo_gw_jct": m_fifo.gpu_weighted_jct,
        "fifo_locality": m_fifo.single_rack_locality_rate,
        "bf_wait": m_bf.mean_wait_time,
        "bf_gw_jct": m_bf.gpu_weighted_jct,
        "bf_locality": m_bf.single_rack_locality_rate,
        "bft_wait": m_bf_topo.mean_wait_time,
        "bft_gw_jct": m_bf_topo.gpu_weighted_jct,
        "bft_locality": m_bf_topo.single_rack_locality_rate,
        "diag_blk": diag_blk,
    }


def _summarize_ali(rows_by_sched: Dict[str, List[dict]]) -> Dict[str, object]:
    summary_sched: Dict[str, Dict[str, float]] = {}
    for s_name in ALI_SCHED_ORDER:
        rows = sorted(rows_by_sched[s_name], key=lambda r: int(r["item_idx"]))
        summary_sched[s_name] = {
            "n": len(rows),
            "blk_gh_mean": float(np.mean([r["blk_gh"] for r in rows])),
            "blk_stranded_mean": float(np.mean([r["blk_stranded"] for r in rows])),
            "blk_lat_us_mean": float(np.mean([r["blk_lat_us"] for r in rows])),
            "fifo_wait_mean": float(np.mean([r["fifo_wait"] for r in rows])),
            "bf_wait_mean": float(np.mean([r["bf_wait"] for r in rows])),
            "bf_jct_mean": float(np.mean([r["bf_jct"] for r in rows])),
        }

    def _paired(metric_key: str, base_name: str, cand_name: str = "Omni-Lindy") -> Dict[str, float]:
        b_rows = sorted(rows_by_sched[base_name], key=lambda r: int(r["item_idx"]))
        c_rows = sorted(rows_by_sched[cand_name], key=lambda r: int(r["item_idx"]))
        return paired_summary_stats(
            np.array([r[metric_key] for r in b_rows], dtype=np.float64),
            np.array([r[metric_key] for r in c_rows], dtype=np.float64),
        )

    best_blk = min(ALI_SOTA_NAMES, key=lambda n: summary_sched[n]["blk_gh_mean"])
    best_fifo = min(ALI_SOTA_NAMES, key=lambda n: summary_sched[n]["fifo_wait_mean"])
    best_bf = min(ALI_SOTA_NAMES, key=lambda n: summary_sched[n]["bf_wait_mean"])

    return {
        "schedulers": summary_sched,
        "best_sota": {"blocking": best_blk, "fifo_wait": best_fifo, "backfill_wait": best_bf},
        "comparisons": {
            "blocking_vs_best_sota": _paired("blk_gh", best_blk),
            "blocking_vs_l2norm": _paired("blk_gh", "L2Norm"),
            "blocking_vs_fgd_oracle": _paired("blk_gh", "FGD_OracleStatic"),
            "blocking_vs_v5": _paired("blk_gh", "Adaptive-B-Lindy"),
            "fifo_wait_vs_best_sota": _paired("fifo_wait", best_fifo),
            "bf_wait_vs_best_sota": _paired("bf_wait", best_bf),
            "bf_wait_vs_v5": _paired("bf_wait", "Adaptive-B-Lindy"),
        },
    }


def _summarize_hel(rows_by_sched: Dict[str, List[dict]]) -> Dict[str, object]:
    summary_sched: Dict[str, Dict[str, float]] = {}
    for s_name in HEL_SCHED_ORDER:
        rows = sorted(rows_by_sched[s_name], key=lambda r: int(r["item_idx"]))
        summary_sched[s_name] = {
            "n": len(rows),
            "blk_gh_mean": float(np.mean([r["blk_gh"] for r in rows])),
            "blk_multi_gh_mean": float(np.mean([r["blk_multi_gh"] for r in rows])),
            "blk_locality_mean": float(np.mean([r["blk_locality"] for r in rows])),
            "blk_racks_mean": float(np.mean([r["blk_racks"] for r in rows])),
            "blk_lat_us_mean": float(np.mean([r["blk_lat_us"] for r in rows])),
            "fifo_wait_mean": float(np.mean([r["fifo_wait"] for r in rows])),
            "fifo_gw_jct_mean": float(np.mean([r["fifo_gw_jct"] for r in rows])),
            "fifo_locality_mean": float(np.mean([r["fifo_locality"] for r in rows])),
            "bf_wait_mean": float(np.mean([r["bf_wait"] for r in rows])),
            "bf_gw_jct_mean": float(np.mean([r["bf_gw_jct"] for r in rows])),
            "bf_locality_mean": float(np.mean([r["bf_locality"] for r in rows])),
            "bft_wait_mean": float(np.mean([r["bft_wait"] for r in rows])),
            "bft_gw_jct_mean": float(np.mean([r["bft_gw_jct"] for r in rows])),
        }

    def _paired(metric_key: str, base_name: str, cand_name: str = "Omni-Lindy") -> Dict[str, float]:
        b_rows = sorted(rows_by_sched[base_name], key=lambda r: int(r["item_idx"]))
        c_rows = sorted(rows_by_sched[cand_name], key=lambda r: int(r["item_idx"]))
        return paired_summary_stats(
            np.array([r[metric_key] for r in b_rows], dtype=np.float64),
            np.array([r[metric_key] for r in c_rows], dtype=np.float64),
        )

    best_blk = min(HEL_SOTA_NAMES, key=lambda n: summary_sched[n]["blk_gh_mean"])
    best_fifo = min(HEL_SOTA_NAMES, key=lambda n: summary_sched[n]["fifo_wait_mean"])
    best_bf = min(HEL_SOTA_NAMES, key=lambda n: summary_sched[n]["bf_wait_mean"])
    best_bft = min(HEL_SOTA_NAMES, key=lambda n: summary_sched[n]["bft_wait_mean"])
    best_bft_gw = min(HEL_SOTA_NAMES, key=lambda n: summary_sched[n]["bft_gw_jct_mean"])

    return {
        "schedulers": summary_sched,
        "best_sota": {
            "blocking": best_blk,
            "fifo_wait": best_fifo,
            "bf_wait": best_bf,
            "bft_wait": best_bft,
            "bft_gw_jct": best_bft_gw,
        },
        "comparisons": {
            "blocking_vs_best_sota": _paired("blk_gh", best_blk),
            "blocking_vs_rackbestfit": _paired("blk_gh", "RackBestFit"),
            "blocking_vs_fgd_oracle": _paired("blk_gh", "FGD_OracleStatic"),
            "blocking_vs_fgd_adaptive": _paired("blk_gh", "FGD_Adaptive"),
            "multi_blocking_vs_rackbestfit": _paired("blk_multi_gh", "RackBestFit"),
            "fifo_wait_vs_best_sota": _paired("fifo_wait", best_fifo),
            "fifo_wait_vs_rackbestfit": _paired("fifo_wait", "RackBestFit"),
            "bf_wait_vs_best_sota": _paired("bf_wait", best_bf),
            "bft_wait_vs_best_sota": _paired("bft_wait", best_bft),
            "bft_wait_vs_rackbestfit": _paired("bft_wait", "RackBestFit"),
            "bft_gw_jct_vs_best_sota": _paired("bft_gw_jct", best_bft_gw),
        },
    }


def main() -> None:
    t_start = time.perf_counter()
    node_csv = ALIBABA_DIR / "openb_node_list_gpu_node.csv"
    pod_default_csv = ALIBABA_DIR / "openb_pod_list_default.csv"
    pod_spec33_csv = ALIBABA_DIR / "openb_pod_list_gpuspec33.csv"

    print("[1/4] Part A — Alibaba PAI 2023 로드 중...")
    ali_cfg, ali_def_jobs, _ = load_alibaba_2023(node_csv, pod_default_csv, use_gpu_spec=True)
    _, ali_spec_jobs, _ = load_alibaba_2023(node_csv, pod_spec33_csv, use_gpu_spec=True)
    ali_def_active, _ = slice_active_trace(ali_def_jobs, start_quantile=0.05)
    ali_spec_active, _ = slice_active_trace(ali_spec_jobs, start_quantile=0.05)

    def_reqs, def_probs = build_empirical_fgd_distribution(ali_def_active)
    spec_reqs, spec_probs = build_empirical_fgd_distribution(ali_spec_active)

    ali_tasks: List[Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str]] = []
    ali_seeds = [101, 202, 303]
    for idx, (kr, s) in enumerate((k, sd) for k in [0.0025, 0.0030, 0.0035, 0.0045] for sd in ali_seeds):
        cfg_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_def_active)
        for sn in ALI_SCHED_ORDER:
            ali_tasks.append(("Alibaba_Default_Sweep", idx, cfg_sub, ali_def_active, def_reqs, def_probs, sn))

    for idx, (kr, s) in enumerate((k, sd) for k in [0.0050, 0.0065, 0.0080, 0.0100] for sd in ali_seeds):
        cfg_sub = scale_heterogeneous_cluster(ali_cfg, keep_ratio=kr, seed=s, jobs=ali_spec_active)
        for sn in ALI_SCHED_ORDER:
            ali_tasks.append(("Alibaba_GPUSpec33_Sweep", idx, cfg_sub, ali_spec_active, spec_reqs, spec_probs, sn))

    print("[2/4] Part B — SenseTime Helios 멀티노드(1~8노드, 16~64 GPU) 로드 및 20개 실제 구간 추출 중...")
    helios_clusters = ["Venus", "Earth", "Saturn", "Uranus"]
    hel_tasks: List[Tuple[str, int, ClusterConfig, List[Job], np.ndarray, np.ndarray, str]] = []
    hel_audits: Dict[str, dict] = {}
    hel_meta: Dict[str, list] = {}

    for cl in helios_clusters:
        h_jobs, h_audit = load_helios_cluster_multinode(HELIOS_DIR / cl / "cluster_log.csv", cl, max_nodes_per_job=8)
        hel_audits[cl] = h_audit.to_dict()
        wins = extract_helios_multinode_windows(
            h_jobs, num_windows=5, jobs_per_window=3000, target_load=0.85, min_span_hours=2.0, nodes_per_rack=8
        )
        hel_meta[cl] = [
            {
                "window_idx": wi,
                "num_nodes": wc.num_nodes,
                "num_racks": wc.num_nodes // 8,
                "clipped_conc_gpu": round(mc, 1),
                "multi_job_pct": round(100.0 * sum(1 for j in wj if j.num_nodes > 1) / len(wj), 1),
            }
            for wi, (wc, wj, mc) in enumerate(wins)
        ]
        print(
            f"  - Helios {cl} (MultiNode): valid={h_audit.valid_gpu_jobs}, over>8N={h_audit.excluded_multinode}, "
            f"corr={h_audit.corr_log_gpu_dur:+.3f}, windows={[(w[0].num_nodes, round(w[2], 1)) for w in wins]}"
        )
        for wi, (w_cfg, w_jobs, _) in enumerate(wins):
            w_reqs, w_probs = build_empirical_fgd_distribution(w_jobs)
            for sn in HEL_SCHED_ORDER:
                hel_tasks.append((f"HeliosMN_{cl}", wi, w_cfg, w_jobs, w_reqs, w_probs, sn))

    n_workers = max(1, min(8, (mp.cpu_count() or 4) - 1))
    print(f"\n[3/4] 병렬 실행 중 ({n_workers} workers): Part A {len(ali_tasks)}x3 + Part B {len(hel_tasks)}x4...")
    with mp.Pool(processes=n_workers) as pool:
        ali_res = pool.map(_eval_ali_unit, ali_tasks)
        hel_res = pool.map(_eval_hel_unit, hel_tasks)

    ali_grouped: Dict[str, Dict[str, List[dict]]] = {}
    for r in ali_res:
        ali_grouped.setdefault(str(r["scenario"]), {k: [] for k in ALI_SCHED_ORDER})[str(r["scheduler"])].append(r)

    hel_grouped: Dict[str, Dict[str, List[dict]]] = {}
    hel_pooled: Dict[str, List[dict]] = {k: [] for k in HEL_SCHED_ORDER}
    for r in hel_res:
        sc = str(r["scenario"])
        sn = str(r["scheduler"])
        hel_grouped.setdefault(sc, {k: [] for k in HEL_SCHED_ORDER})[sn].append(r)
    for cl_idx, cl in enumerate(helios_clusters):
        sc = f"HeliosMN_{cl}"
        for sn in HEL_SCHED_ORDER:
            for r in hel_grouped[sc][sn]:
                rc = dict(r)
                rc["item_idx"] = cl_idx * 10 + int(r["item_idx"])
                hel_pooled[sn].append(rc)

    print("\n[4/4] 7단계 최종 실증 성적표")
    ali_summaries = {sc: _summarize_ali(ali_grouped[sc]) for sc in ("Alibaba_Default_Sweep", "Alibaba_GPUSpec33_Sweep")}
    for sc, grp in ali_summaries.items():
        print(f"\n==================== {sc} ====================")
        print(f"{'Scheduler':<18} | {'Blk(%)':>7} | {'Stranded':>8} | {'FIFO Wait':>10} | {'BF Wait':>10} | {'Lat(us)':>8}")
        print("-" * 74)
        for sn in ALI_SCHED_ORDER:
            m = grp["schedulers"][sn]
            print(
                f"{sn:<18} | {m['blk_gh_mean']:7.3f} | {m['blk_stranded_mean']:8.2f} | "
                f"{m['fifo_wait_mean']:10.3f} | {m['bf_wait_mean']:10.3f} | {m['blk_lat_us_mean']:8.1f}"
            )
        cb = grp["comparisons"]["blocking_vs_best_sota"]
        cbf = grp["comparisons"]["bf_wait_vs_best_sota"]
        print(
            f"  >> Blk vs {grp['best_sota']['blocking']}: {cb['rel_improvement_pct']:+.2f}% "
            f"(W/L={cb['wins']}/{cb['losses']}, p_t={cb['p_value_t']:.4g}, p_W={cb['p_value_wilcoxon']:.4g})"
            f" | BF Wait vs {grp['best_sota']['backfill_wait']}: {cbf['rel_improvement_pct']:+.2f}% (p_W={cbf['p_value_wilcoxon']:.4g})"
        )

    hel_summaries = {f"HeliosMN_{cl}": _summarize_hel(hel_grouped[f"HeliosMN_{cl}"]) for cl in helios_clusters}
    hel_summaries["HeliosMN_Pooled_20"] = _summarize_hel(hel_pooled)
    grp = hel_summaries["HeliosMN_Pooled_20"]
    print("\n==================== Helios_MultiNode_Pooled_20 (16~64 GPU Gang + Rack Topology) ====================")
    print(
        f"{'Scheduler':<18} | {'Blk(%)':>7} | {'MN-Blk%':>7} | {'Loc(%)':>6} | "
        f"{'FIFO W':>8} | {'BF W':>8} | {'Topo-BF W':>9} | {'Topo GW-JCT':>11} | {'Lat(us)':>7}"
    )
    print("-" * 102)
    for sn in HEL_SCHED_ORDER:
        m = grp["schedulers"][sn]
        print(
            f"{sn:<18} | {m['blk_gh_mean']:7.3f} | {m['blk_multi_gh_mean']:7.3f} | {m['blk_locality_mean']:6.1f} | "
            f"{m['fifo_wait_mean']:8.3f} | {m['bf_wait_mean']:8.3f} | {m['bft_wait_mean']:9.3f} | "
            f"{m['bft_gw_jct_mean']:11.3f} | {m['blk_lat_us_mean']:7.1f}"
        )
    cb = grp["comparisons"]["blocking_vs_best_sota"]
    cr = grp["comparisons"]["blocking_vs_rackbestfit"]
    ct = grp["comparisons"]["bft_wait_vs_best_sota"]
    cgw = grp["comparisons"]["bft_gw_jct_vs_best_sota"]
    print(
        f"  >> Blk vs {grp['best_sota']['blocking']}: {cb['rel_improvement_pct']:+.2f}% "
        f"(W/L={cb['wins']}/{cb['losses']}, p_t={cb['p_value_t']:.4g}, p_W={cb['p_value_wilcoxon']:.4g}) "
        f"| vs RackBestFit: {cr['rel_improvement_pct']:+.2f}% (p_W={cr['p_value_wilcoxon']:.4g})"
    )
    print(
        f"  >> Topo-BF Wait vs {grp['best_sota']['bft_wait']}: {ct['rel_improvement_pct']:+.2f}% "
        f"(p_t={ct['p_value_t']:.4g}, p_W={ct['p_value_wilcoxon']:.4g}) "
        f"| Topo GW-JCT vs {grp['best_sota']['bft_gw_jct']}: {cgw['rel_improvement_pct']:+.2f}% (p_W={cgw['p_value_wilcoxon']:.4g})"
    )

    elapsed = time.perf_counter() - t_start
    payload = {
        "part_a_alibaba": ali_summaries,
        "part_b_helios_audits": hel_audits,
        "part_b_helios_meta": hel_meta,
        "part_b_helios": hel_summaries,
        "elapsed_sec": elapsed,
    }
    out_path = BASE_DIR / "phase7_report.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[완료] 소요 시간: {elapsed:.1f}초 | 저장: {out_path}")


if __name__ == "__main__":
    main()
