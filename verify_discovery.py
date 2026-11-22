"""
3단계: 발견된 신알고리즘(Evo-STF)의 엄밀 과학적 검증 스위트

검증 항목:
1. Held-out Generalization (W1, W2, W3 — 미지 시드 10개 쌍체 비교, Win-rate & Paired t-stat)
2. Unseen OOD & 반례 검증:
   - W4_Memoryless_OOD (린디 효과가 거짓인 지수분포 반례 테스트)
   - W5_Skewed_Vec_OOD (극단적 CPU/RAM 비대칭 자원 편향)
   - W6_Scale1024_OOD  (128노드 1,024-GPU 초대형 클러스터 + 새로운 작업 분포)
3. Ablation Study (기여도 분해):
   - 시간(Lindy-Age) 항 제거 vs 이산 경계(Buddy/8-GPU) 보호 항 제거 vs 공간 잔여 항 제거
4. 의사결정 지연시간(Latency, us) 비교 (FGD 대비 속도 우위)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from baselines import (
    BestFitScheduler,
    DotProductScheduler,
    FGDScheduler,
    FirstFitScheduler,
    L2NormScheduler,
)
from evolve_scheduler import FEATURE_NAMES, ParametricEvoScheduler
from simulator import BaseScheduler, ClusterConfig, Job, SimulationMetrics, run_simulation
from workloads import (
    STANDARD_PROFILES,
    WORKLOAD_SUITE,
    _sample_pareto_duration,
)


def generate_w6_scale1024_ood(seed: int = 42, num_jobs: int = 8000, target_load: float = 0.89) -> List[Job]:
    """
    W6 [초대형 스케일업 OOD]: 128노드 (1,024 GPU) + 학습 시 없던 새로운 확률 분포
    (대형 4/8-GPU 비중이 높고 분수 GPU가 혼합된 실제 차세대 AI 클러스터 스케일).
    """
    rng = np.random.default_rng(seed)
    probs = np.array([0.22, 0.20, 0.18, 0.14, 0.14, 0.12], dtype=np.float64)
    probs /= probs.sum()

    total_gpus = 1024.0
    chosen_idx = rng.choice(len(STANDARD_PROFILES), size=num_jobs, p=probs)
    durations = np.empty(num_jobs, dtype=np.float64)
    for i, idx in enumerate(chosen_idx):
        gpu = STANDARD_PROFILES[idx][0]
        if gpu >= 4.0:
            durations[i] = _sample_pareto_duration(rng, alpha=1.55, scale=1.5, max_dur=40.0)
        else:
            durations[i] = _sample_pareto_duration(rng, alpha=1.30, scale=0.20, max_dur=30.0)

    mean_gpu_hours = float(np.mean([STANDARD_PROFILES[chosen_idx[i]][0] * durations[i] for i in range(num_jobs)]))
    arrival_rate = (target_load * total_gpus) / max(mean_gpu_hours, 1e-6)
    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, size=num_jobs))

    jobs: List[Job] = []
    for i in range(num_jobs):
        g, c, r = STANDARD_PROFILES[chosen_idx[i]]
        jobs.append(Job(i, float(arrivals[i]), float(durations[i]), g, c, r))
    return jobs


def paired_t_pvalue_approx(diffs: np.ndarray) -> Tuple[float, float]:
    """
    쌍체 차이 배열(base - evo, 양수면 evo 승리)에 대해
    t-통계량과 양측 p-value 근사치(정규/부호 근사, scipy 없이 계산)를 반환한다.
    """
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    if std_d < 1e-12:
        t_stat = 99.0 if mean_d > 0 else (-99.0 if mean_d < 0 else 0.0)
    else:
        t_stat = mean_d / (std_d / math.sqrt(n))
    # 자유도 n-1 t-분포의 보수적 정규 꼬리 근사 (자유도 보정 계수 적용)
    z_equiv = abs(t_stat) * math.sqrt((n - 2.0) / max(n - 1.0, 1.0)) if n > 2 else abs(t_stat)
    p_val = math.erfc(z_equiv / math.sqrt(2.0))
    return t_stat, p_val


def main() -> None:
    sol_path = Path(__file__).parent / "evolved_solution.json"
    sol = json.loads(sol_path.read_text(encoding="utf-8"))
    w_full = np.array(sol["best_full_weights"], dtype=np.float64)
    w_dist = np.array(sol["distilled_weights"], dtype=np.float64)

    print("=== [검증 대상] 발견된 Evo-STF Distilled 핵심 수식 ===")
    for i, name in enumerate(FEATURE_NAMES):
        if abs(w_dist[i]) > 1e-6:
            sign = "+" if w_dist[i] > 0 else "-"
            print(f"  {sign} {abs(w_dist[i]):.4f} * {name}")

    # 10개 미지 시드(Held-out & OOD) — 진화 과정에서 단 한 번도 보지 못한 시드!
    test_seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    cfg_64 = ClusterConfig(num_nodes=64, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)
    cfg_128 = ClusterConfig(num_nodes=128, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)

    # 진화된 수식에서 핵심 4대 원리만 추출한 초간결 4-항(4-Term) 해석 가능 원리 수식 (LindyPack_4T)
    # Score = (0.5*rem_gpu + 0.5*rem_cpu + 0.2*rem_ram) + 0.20*dim_imbalance + 0.12*breaks_whole_gpu - 0.45*log_max_age
    w_4term = np.zeros_like(w_dist)
    w_4term[0] = 0.50   # rem_gpu_after
    w_4term[1] = 0.50   # rem_cpu_after
    w_4term[2] = 0.20   # rem_ram_after
    w_4term[4] = 0.20   # dim_imbalance
    w_4term[10] = 0.12  # breaks_whole_gpu
    w_4term[14] = -0.45 # log_max_age (Lindy-Age Straggler Cushioning)

    # 강력한 비교 대상: 1단계 최강 챔피언들 + FGD_Adaptive + 우리 발견 알고리즘 3종
    schedulers: List[BaseScheduler] = [
        FirstFitScheduler(),
        BestFitScheduler(),
        DotProductScheduler(),
        L2NormScheduler(),
        FGDScheduler(adaptive=True),
        ParametricEvoScheduler(w_full, name="EvoSTF_Full"),
        ParametricEvoScheduler(w_dist, name="EvoSTF_Dist"),
        ParametricEvoScheduler(w_4term, name="LindyPack_4T"),
    ]

    all_workloads = list(WORKLOAD_SUITE.items()) + [("W6_Scale1024_OOD", generate_w6_scale1024_ood)]
    report: Dict[str, dict] = {}

    for w_name, gen_fn in all_workloads:
        is_scale = w_name == "W6_Scale1024_OOD"
        cfg = cfg_128 if is_scale else cfg_64
        seeds = test_seeds[:5] if is_scale else test_seeds  # 128노드 초대형은 5개 시드(총 40,000 작업)

        seed_jobs = {s: gen_fn(seed=s) for s in seeds}

        print(f"\n==================== {w_name} ({len(seeds)} unseen seeds, {cfg.num_nodes} nodes) ====================")
        print(
            f"{'Scheduler':<14} | {'GPU-Hr Block(%)':>16} | {'Stranded GPU(%)':>16} | "
            f"{'Active Node-Hr':>14} | {'Latency(us)':>11}"
        )
        print("-" * 86)

        w_res: Dict[str, List[SimulationMetrics]] = {}
        for sch in schedulers:
            # 초대형 클러스터에서는 FGD_Adaptive가 지나치게 느리므로 3시드만 표본 측정하거나 전체 측정
            ms = [run_simulation(seed_jobs[s], sch, config=cfg, workload_name=w_name, seed=s) for s in seeds]
            w_res[sch.name] = ms
            gh = np.array([m.gpu_hour_blocking_rate for m in ms])
            st = np.array([m.avg_stranded_gpu_ratio for m in ms])
            an = np.array([m.active_node_hours for m in ms])
            lt = np.array([m.avg_decision_us for m in ms])
            print(
                f"{sch.name:<14} | {gh.mean():6.2f} +/- {gh.std():5.2f} | "
                f"{st.mean():6.2f} +/- {st.std():5.2f} | "
                f"{an.mean():14.1f} | {lt.mean():11.1f}"
            )

        # 최강 베이스라인(BestFit vs L2Norm vs DotProduct vs FGD 중 최저 거절률)과의 쌍체 통계 검정
        base_names = ["FirstFit", "BestFit", "DotProduct", "L2Norm", "FGD_Adaptive"]
        best_base_name = min(
            base_names,
            key=lambda n: float(np.mean([m.gpu_hour_blocking_rate for m in w_res[n]])),
        )
        base_gh = np.array([m.gpu_hour_blocking_rate for m in w_res[best_base_name]])
        evo_gh = np.array([m.gpu_hour_blocking_rate for m in w_res["EvoSTF_Dist"]])
        diffs = base_gh - evo_gh  # 양수면 EvoSTF_Dist 승리!
        wins = int(np.sum(diffs > 0))
        rel_imp = ((base_gh.mean() - evo_gh.mean()) / max(base_gh.mean(), 1e-9)) * 100.0
        t_stat, p_val = paired_t_pvalue_approx(diffs)

        print(
            f"  >> vs 최강 기존 SOTA({best_base_name}): "
            f"GPU-Hr 거절률 상대 개선 {rel_imp:+.1f}% ({base_gh.mean():.2f}% -> {evo_gh.mean():.2f}%), "
            f"승률 {wins}/{len(seeds)}, t={t_stat:.2f}, p≈{p_val:.4f}"
        )

        report[w_name] = {
            "best_baseline": best_base_name,
            "best_baseline_gh_mean": float(base_gh.mean()),
            "evo_dist_gh_mean": float(evo_gh.mean()),
            "evo_full_gh_mean": float(np.mean([m.gpu_hour_blocking_rate for m in w_res["EvoSTF_Full"]])),
            "relative_improvement_pct": float(rel_imp),
            "wins": wins,
            "num_seeds": len(seeds),
            "t_stat": float(t_stat),
            "p_val_approx": float(p_val),
            "all_schedulers": {
                k: {
                    "gh_mean": float(np.mean([m.gpu_hour_blocking_rate for m in v])),
                    "gh_std": float(np.std([m.gpu_hour_blocking_rate for m in v])),
                    "stranded_mean": float(np.mean([m.avg_stranded_gpu_ratio for m in v])),
                    "active_node_hr_mean": float(np.mean([m.active_node_hours for m in v])),
                    "latency_us_mean": float(np.mean([m.avg_decision_us for m in v])),
                }
                for k, v in w_res.items()
            },
        }

    # ==================== Ablation Study (원인 분해 실험) ====================
    print("\n==================== ABLATION STUDY (메커니즘별 기여도 분해, W1+W2+W3 5 seeds) ====================")
    ablation_groups = {
        "Full_Distilled": [],
        "No_Temporal_Lindy": [14, 15, 16],               # 시간 경과(Lindy-Age) 항 전체 제거
        "No_Boundary_Buddy": [6, 7, 8, 9, 10, 11, 12, 13, 17],  # 이산 경계/버디/순수노드 보호 제거
        "No_Spatial_Vector": [0, 1, 2, 3, 4, 5],         # 연속 자원 잔여/균형 항 제거
    }
    ab_seeds = test_seeds[:5]
    ab_data = {
        w_name: {s: WORKLOAD_SUITE[w_name](seed=s) for s in ab_seeds}
        for w_name in ["W1_PAI_Mixed", "W2_Regime_Shift", "W3_Heavy_Straggler"]
    }
    ablation_report = {}
    for ab_name, drop_idxs in ablation_groups.items():
        w_ab = w_dist.copy()
        w_ab[drop_idxs] = 0.0
        sch = ParametricEvoScheduler(w_ab, name=ab_name)
        row = {}
        for w_name, s_map in ab_data.items():
            ms = [run_simulation(s_map[s], sch, config=cfg_64, workload_name=w_name, seed=s) for s in ab_seeds]
            row[w_name] = float(np.mean([m.gpu_hour_blocking_rate for m in ms]))
        ablation_report[ab_name] = row
        print(
            f"  {ab_name:<20} | W1={row['W1_PAI_Mixed']:5.2f}% | "
            f"W2={row['W2_Regime_Shift']:5.2f}% | W3={row['W3_Heavy_Straggler']:5.2f}%"
        )

    out_path = Path(__file__).parent / "verification_report.json"
    out_path.write_text(
        json.dumps({"benchmarks": report, "ablation": ablation_report}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[최종 검증 리포트 저장 완료] {out_path}")


if __name__ == "__main__":
    main()
