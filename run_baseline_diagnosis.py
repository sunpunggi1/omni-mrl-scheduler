"""
1단계: 학계 SOTA 베이스라인 6종 진단 벤치마크 실행 스크립트

5개 워크로드 × 5개 랜덤 시드에서 각 베이스라인의
- GPU-시간 거절률 (Blocking %)
- 순수 단편화 지표 (Stranded GPU %)
- 누적 활성 노드-시간 (전력·운영비)
- 의사결정 지연시간 (μs)
을 평균 ± 표준편차로 정량 측정한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from baselines import get_all_baselines
from simulator import ClusterConfig, SimulationMetrics, run_simulation
from workloads import WORKLOAD_SUITE


def main() -> None:
    seeds = [101, 202, 303, 404, 505]
    config = ClusterConfig(num_nodes=64, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)

    # 워크로드별 시드 데이터 사전 생성 (모든 스케줄러가 100% 동일한 도착/수명 시퀀스로 공정 대결)
    datasets = {
        w_name: {s: gen_fn(seed=s) for s in seeds}
        for w_name, gen_fn in WORKLOAD_SUITE.items()
    }

    schedulers = get_all_baselines()
    results: Dict[str, Dict[str, List[SimulationMetrics]]] = {
        w_name: {sch.name: [] for sch in schedulers}
        for w_name in WORKLOAD_SUITE
    }

    for w_name, seed_map in datasets.items():
        print(f"\n==================== {w_name} ====================")
        print(
            f"{'Scheduler':<14} | {'GPU-Hr Block(%)':>16} | {'Job Block(%)':>13} | "
            f"{'Stranded GPU(%)':>15} | {'Active Node-Hr':>14} | {'Latency(us)':>11}"
        )
        print("-" * 98)

        for sch in schedulers:
            metrics_list: List[SimulationMetrics] = []
            for s, jobs in seed_map.items():
                m = run_simulation(jobs, sch, config=config, workload_name=w_name, seed=s)
                metrics_list.append(m)
            results[w_name][sch.name] = metrics_list

            gh_b = np.array([m.gpu_hour_blocking_rate for m in metrics_list])
            jb_b = np.array([m.job_blocking_rate for m in metrics_list])
            st_g = np.array([m.avg_stranded_gpu_ratio for m in metrics_list])
            an_h = np.array([m.active_node_hours for m in metrics_list])
            lat = np.array([m.avg_decision_us for m in metrics_list])

            print(
                f"{sch.name:<14} | {gh_b.mean():6.2f} +/- {gh_b.std():5.2f} | "
                f"{jb_b.mean():5.2f} +/- {jb_b.std():4.2f} | "
                f"{st_g.mean():6.2f} +/- {st_g.std():5.2f} | "
                f"{an_h.mean():14.1f} | {lat.mean():11.1f}"
            )

    # JSON 저장 (2단계 진화 연구 비교 기준선)
    out_path = Path(__file__).parent / "baseline_results.json"
    summary = {}
    for w_name, sch_map in results.items():
        summary[w_name] = {}
        for sch_name, m_list in sch_map.items():
            summary[w_name][sch_name] = {
                "gpu_hour_block_mean": float(np.mean([m.gpu_hour_blocking_rate for m in m_list])),
                "gpu_hour_block_std": float(np.std([m.gpu_hour_blocking_rate for m in m_list])),
                "job_block_mean": float(np.mean([m.job_blocking_rate for m in m_list])),
                "stranded_gpu_mean": float(np.mean([m.avg_stranded_gpu_ratio for m in m_list])),
                "active_node_hours_mean": float(np.mean([m.active_node_hours for m in m_list])),
                "latency_us_mean": float(np.mean([m.avg_decision_us for m in m_list])),
            }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[저장 완료] {out_path}")


if __name__ == "__main__":
    main()
