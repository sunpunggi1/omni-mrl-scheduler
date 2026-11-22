"""
2단계: AlphaEvolve / FunSearch 스타일 다중 전략 진화 탐색 엔진 (Evo-STF Discovery v2)

핵심 개선:
- 이진 버디 정렬(Buddy Alignment: 8-GPU 노드에서 배치 후 빈 GPU가 {0, 2, 4}로 떨어지는 정렬성) 추가
- 소형/중형 잡별 순수 8-GPU 노드 개봉 페널티 분리
- Lindy-Age 기반 장기 점유 노드 재활용 vs 젊은 노드 보호 피처 정교화
- 매 5세대 체크포인트 저장 + 안전한 pool.apply 기반 오컴의 면도날 수식 증류

엄격한 데이터 분리 (Zero Data Leakage):
- 진화 학습(Train): W1, W2, W3의 seed=[11, 22] 만 사용
- 보유 평가(Held-out Test): W1, W2, W3의 seed=[101, 202, 303, 404, 505]
- 미지 분포 및 반례(OOD Test): W4_Memoryless_OOD, W5_Skewed_Vec_OOD, W6_Scale1024_OOD
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from baselines import BestFitScheduler, L2NormScheduler
from simulator import BaseScheduler, ClusterConfig, ClusterState, Job, run_simulation
from workloads import (
    generate_w1_pai_mixed,
    generate_w2_regime_shift,
    generate_w3_heavy_straggler,
)

# 18개 물리·이산경계·시공간·적응형 기본 피처 (모두 Non-clairvoyant & O(1) 벡터화)
FEATURE_NAMES: List[str] = [
    "rem_gpu_after",          # 0: 배치 후 정규화 잔여 GPU [0, 1]
    "rem_cpu_after",          # 1: 배치 후 정규화 잔여 CPU [0, 1]
    "rem_ram_after",          # 2: 배치 후 정규화 잔여 RAM [0, 1]
    "rem_min_dim",            # 3: 배치 후 가장 꽉 찬 병목 자원 잔여량 min(g, c, r)
    "dim_imbalance",          # 4: 배치 후 다차원 불균형 max(g, c, r) - min(g, c, r)
    "l2_residual",            # 5: 배치 후 잔여 벡터 L2 제곱합
    "small_breaks_pristine",  # 6: 소형 잡(<=1 GPU)이 완전 빈 8-GPU 노드를 개봉
    "med_breaks_pristine",    # 7: 중형 잡(2~4 GPU)이 완전 빈 8-GPU 노드를 개봉
    "breaks_4gpu",            # 8: 배치로 인해 빈 GPU가 >=4에서 <4로 깨짐
    "breaks_2gpu",            # 9: 배치로 인해 빈 GPU가 >=2에서 <2로 깨짐
    "breaks_whole_gpu",       # 10: 분수 GPU 잡이 완전한 1.0 물리 GPU를 새로 쪼갬
    "buddy_aligned_after",    # 11: 배치 후 빈 GPU가 이진 경계 {0, 2, 4}에 딱 정렬됨 (Buddy Alignment)
    "exact_fill_bonus",       # 12: 배치 후 빈 GPU 또는 타깃 분수 슬롯이 정확히 0으로 떨어짐
    "partial_waste_after",    # 13: 배치 후 노드 내 자투리(0 < slot < 1) GPU 총합 / 8
    "log_max_age",            # 14: log(1 + 최장 실행 작업 경과 시간) — 린디 효과 핵심
    "log_mean_age",           # 15: log(1 + 평균 실행 작업 경과 시간)
    "small_x_log_max_age",    # 16: 소형 잡(<=1 GPU) * log(1 + max_age) — 장기 노드 재활용 상호작용
    "regime_8gpu_protect",    # 17: 온라인 감지 대형 잡 비율(p_large) * 소형 잡의 빈 8-GPU 개봉
]
NUM_FEATURES = len(FEATURE_NAMES)


def extract_node_features(
    state: ClusterState,
    feasible_nodes: np.ndarray,
    gpu: float,
    cpu: float,
    ram: float,
) -> np.ndarray:
    """후보 노드 집합에 대해 18개 물리·시공간·적응형 피처 행렬 (K, 18)을 O(1) 벡터 연산으로 계산한다."""
    eps = 1e-9
    cfg = state.config
    k = len(feasible_nodes)

    slots = state.gpu_slots[feasible_nodes]            # (K, 8)
    cpu_free = state.cpu_free[feasible_nodes]          # (K,)
    ram_free = state.ram_free[feasible_nodes]          # (K,)

    total_gpu_before = np.sum(slots, axis=1)           # (K,)
    empty_before = np.sum(slots >= 1.0 - eps, axis=1)  # (K,)

    # 배치 후 잔여 자원 계산
    rem_g = np.maximum(0.0, total_gpu_before - gpu) / cfg.gpus_per_node
    rem_c = np.maximum(0.0, cpu_free - cpu) / cfg.cpu_per_node
    rem_r = np.maximum(0.0, ram_free - ram) / cfg.ram_per_node

    rem_min = np.minimum(rem_g, np.minimum(rem_c, rem_r))
    rem_max = np.maximum(rem_g, np.maximum(rem_c, rem_r))
    dim_imb = rem_max - rem_min
    l2_res = rem_g * rem_g + rem_c * rem_c + rem_r * rem_r

    # 순수 빈 8-GPU 노드 여부
    is_pristine = (state.node_job_count[feasible_nodes] == 0).astype(np.float64)
    is_small = 1.0 if gpu <= 1.0 + eps else 0.0
    is_med = 1.0 if (1.0 + eps < gpu < cfg.gpus_per_node - eps) else 0.0

    is_frac = gpu < 1.0 - eps
    if is_frac:
        has_partial_fit = np.any((slots < 1.0 - eps) & (slots + eps >= gpu), axis=1)
        breaks_whole = (~has_partial_fit).astype(np.float64)
        empty_after = empty_before - breaks_whole

        masked = np.where(slots + eps >= gpu, slots, 10.0)
        best_slot_val = np.min(masked, axis=1)
        rem_slot = np.maximum(0.0, best_slot_val - gpu)
        exact_fill = (rem_slot <= 1e-6).astype(np.float64)

        partial_before = np.sum(np.where((slots > eps) & (slots < 1.0 - eps), slots, 0.0), axis=1)
        partial_after = np.where(
            has_partial_fit,
            np.maximum(0.0, partial_before - gpu),
            partial_before + rem_slot,
        ) / cfg.gpus_per_node
    else:
        needed = int(round(gpu))
        breaks_whole = np.zeros(k, dtype=np.float64)
        empty_after = empty_before - needed
        exact_fill = (empty_after == 0).astype(np.float64)
        partial_after = (
            np.sum(np.where((slots > eps) & (slots < 1.0 - eps), slots, 0.0), axis=1)
            / cfg.gpus_per_node
        )

    breaks_4 = ((empty_before >= 4) & (empty_after < 4)).astype(np.float64)
    breaks_2 = ((empty_before >= 2) & (empty_after < 2)).astype(np.float64)
    # 이진 버디 정렬: 배치 후 빈 물리 GPU가 정확히 0, 2, 4 중 하나인 경우
    buddy_aligned = (
        (empty_after == 0) | (empty_after == 2) | (empty_after == 4)
    ).astype(np.float64)

    # 시간적 계층화 피처 (Non-clairvoyant Lindy-Age)
    max_age, mean_age, _ = state.node_age_stats(feasible_nodes)
    log_max_age = np.log1p(np.maximum(0.0, max_age))
    log_mean_age = np.log1p(np.maximum(0.0, mean_age))

    # 온라인 분포 적응 피처
    recent = state.recent_requests
    if len(recent) >= 10:
        tail = recent[-60:]
        p_large = sum(1 for r in tail if r[0] >= 4.0) / len(tail)
    else:
        p_large = 0.14

    feats = np.empty((k, NUM_FEATURES), dtype=np.float64)
    feats[:, 0] = rem_g
    feats[:, 1] = rem_c
    feats[:, 2] = rem_r
    feats[:, 3] = rem_min
    feats[:, 4] = dim_imb
    feats[:, 5] = l2_res
    feats[:, 6] = is_small * is_pristine
    feats[:, 7] = is_med * is_pristine
    feats[:, 8] = breaks_4
    feats[:, 9] = breaks_2
    feats[:, 10] = breaks_whole
    feats[:, 11] = buddy_aligned
    feats[:, 12] = exact_fill
    feats[:, 13] = partial_after
    feats[:, 14] = log_max_age
    feats[:, 15] = log_mean_age
    feats[:, 16] = is_small * log_max_age
    feats[:, 17] = p_large * is_small * is_pristine
    return feats


class ParametricEvoScheduler(BaseScheduler):
    """진화된 가중치 벡터로 노드를 평가하는 초고속 O(1) 스케줄러."""

    def __init__(self, weights: np.ndarray, name: str = "EvoSTF"):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.name = name
        self.active_idx = np.flatnonzero(np.abs(self.weights) > 1e-6)
        self.active_w = self.weights[self.active_idx]

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        feats = extract_node_features(state, feasible_nodes, gpu, cpu, ram)
        if len(self.active_idx) == 0:
            return int(feasible_nodes[0])
        scores = feats[:, self.active_idx] @ self.active_w
        return int(feasible_nodes[np.argmin(scores)])


_WORKER_DATASETS: Dict[str, List[Tuple[int, List[Job]]]] = {}
_WORKER_BASELINES: Dict[str, Dict[str, float]] = {}
_WORKER_CONFIG = ClusterConfig(num_nodes=64, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)


def _init_worker(datasets, baselines):
    global _WORKER_DATASETS, _WORKER_BASELINES
    _WORKER_DATASETS = datasets
    _WORKER_BASELINES = baselines


def _evaluate_candidate(args: Tuple[np.ndarray, float]) -> Tuple[float, Dict[str, float]]:
    """단일 후보 가중치 벡터를 학습 워크로드(W1, W2, W3 × seed 11, 22)에서 평가한다."""
    weights, l1_penalty = args
    sch = ParametricEvoScheduler(weights)

    total_loss = 0.0
    breakdown: Dict[str, float] = {}

    for w_name, seed_jobs in _WORKER_DATASETS.items():
        base = _WORKER_BASELINES[w_name]
        gh_blocks = []
        strandeds = []
        active_hrs = []
        for s, jobs in seed_jobs:
            m = run_simulation(jobs, sch, config=_WORKER_CONFIG, workload_name=w_name, seed=s)
            gh_blocks.append(m.gpu_hour_blocking_rate)
            strandeds.append(m.avg_stranded_gpu_ratio)
            active_hrs.append(m.active_node_hours)

        mean_gh = float(np.mean(gh_blocks))
        mean_st = float(np.mean(strandeds))
        mean_an = float(np.mean(active_hrs))

        rel_gh = mean_gh / max(base["gh"], 0.5)
        rel_st = mean_st / max(base["st"], 1.0)
        rel_an = mean_an / max(base["an"], 100.0)

        w_loss = 0.65 * rel_gh + 0.25 * rel_st + 0.10 * rel_an
        total_loss += w_loss
        breakdown[w_name] = mean_gh

    avg_loss = total_loss / len(_WORKER_DATASETS)
    nnz = int(np.sum(np.abs(weights) > 1e-3))
    reg_loss = avg_loss + l1_penalty * (nnz / NUM_FEATURES)
    return float(reg_loss), breakdown


def build_train_datasets() -> Tuple[Dict[str, List[Tuple[int, List[Job]]]], Dict[str, Dict[str, float]]]:
    """진화 학습 전용 데이터셋(seed 11, 22)과 기준 베이스라인을 준비한다."""
    train_seeds = [11, 22]
    generators = {
        "W1_PAI_Mixed": generate_w1_pai_mixed,
        "W2_Regime_Shift": generate_w2_regime_shift,
        "W3_Heavy_Straggler": generate_w3_heavy_straggler,
    }
    datasets = {
        w_name: [(s, gen(seed=s)) for s in train_seeds]
        for w_name, gen in generators.items()
    }

    baselines: Dict[str, Dict[str, float]] = {}
    ref_schedulers = [BestFitScheduler(), L2NormScheduler()]
    for w_name, seed_jobs in datasets.items():
        best_gh = 1e9
        best_st = 1e9
        best_an = 1e9
        for sch in ref_schedulers:
            ms = [run_simulation(jobs, sch, config=_WORKER_CONFIG, workload_name=w_name, seed=s) for s, jobs in seed_jobs]
            best_gh = min(best_gh, float(np.mean([m.gpu_hour_blocking_rate for m in ms])))
            best_st = min(best_st, float(np.mean([m.avg_stranded_gpu_ratio for m in ms])))
            best_an = min(best_an, float(np.mean([m.active_node_hours for m in ms])))
        baselines[w_name] = {"gh": best_gh, "st": best_st, "an": best_an}
    return datasets, baselines


def _save_checkpoint(out_path: Path, payload: dict) -> None:
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    t_start = time.perf_counter()
    out_path = Path(__file__).parent / "evolved_solution.json"

    print("[1/3] 진화 학습 데이터셋(Train seeds: 11, 22) 및 기준선 준비 중...")
    datasets, baselines = build_train_datasets()
    for w_name, b in baselines.items():
        print(f"  - Train 기준선 {w_name}: GPU-Hr Block={b['gh']:.2f}%, Stranded={b['st']:.2f}%, ActiveNodeHr={b['an']:.1f}")

    rng = np.random.default_rng(2026)
    pop_size = 24
    generations = 35
    l1_penalty = 0.006

    # 초기 개체군: 도메인 앵커 5개 + 희소 무작위 19개
    pop = np.zeros((pop_size, NUM_FEATURES), dtype=np.float64)
    pop[0, [0, 1, 2]] = [1.0, 0.05, 0.05]                                    # BestFit
    pop[1, 5] = 1.0                                                          # L2Norm
    pop[2, [0, 4, 6, 8, 10, 11, 14]] = [0.6, 0.3, 1.0, 0.5, 0.7, -0.4, -0.4] # Buddy + Lindy 재활용
    pop[3, [0, 4, 6, 8, 10, 11, 14]] = [0.6, 0.3, 1.0, 0.5, 0.7, -0.4, +0.4] # Buddy + Lindy 회피
    pop[4, [0, 3, 6, 7, 10, 12, 16]] = [0.5, 0.4, 1.0, 0.6, 0.8, -0.5, -0.5] # Bottleneck + 소형 장기노드 집중

    for i in range(5, pop_size):
        mask = rng.random(NUM_FEATURES) < 0.55
        vals = rng.uniform(-1.0, 1.0, size=NUM_FEATURES)
        pop[i] = np.where(mask, vals, 0.0)
        if np.all(pop[i] == 0.0):
            pop[i, 0] = 1.0

    n_workers = max(1, min(8, (mp.cpu_count() or 4) - 1))
    print(f"[2/3] 멀티코어({n_workers} workers) 진화 탐색 시작: {generations}세대 x {pop_size}개체 = {generations * pop_size} 후보 평가")

    history = []
    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(datasets, baselines)) as pool:
        eval_res = pool.map(_evaluate_candidate, [(ind, l1_penalty) for ind in pop])
        fitness = np.array([r[0] for r in eval_res], dtype=np.float64)
        breakdowns = [r[1] for r in eval_res]

        best_idx = int(np.argmin(fitness))
        print(
            f"  Gen  0 | Best Loss={fitness[best_idx]:.4f} | "
            f"W1={breakdowns[best_idx]['W1_PAI_Mixed']:.2f}% "
            f"W2={breakdowns[best_idx]['W2_Regime_Shift']:.2f}% "
            f"W3={breakdowns[best_idx]['W3_Heavy_Straggler']:.2f}% | "
            f"nnz={int(np.sum(np.abs(pop[best_idx]) > 1e-3))}",
            flush=True,
        )

        for gen in range(1, generations + 1):
            trials = np.empty_like(pop)
            best_ind = pop[int(np.argmin(fitness))]

            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[rng.choice(idxs, size=3, replace=False)]
                F = rng.uniform(0.35, 0.85)
                CR = rng.uniform(0.5, 0.9)

                if rng.random() < 0.65:
                    mutant = pop[i] + F * (best_ind - pop[i]) + F * (a - b)
                else:
                    mutant = a + F * (b - c)

                cross = rng.random(NUM_FEATURES) < CR
                cross[rng.integers(NUM_FEATURES)] = True
                trial = np.where(cross, mutant, pop[i])

                if rng.random() < 0.25:
                    trial[rng.integers(NUM_FEATURES)] = 0.0
                if rng.random() < 0.20:
                    trial[rng.integers(NUM_FEATURES)] += rng.normal(0.0, 0.4)

                max_abs = float(np.max(np.abs(trial)))
                if max_abs > 1e-6:
                    trial = trial / max_abs
                trial = np.where(np.abs(trial) < 0.03, 0.0, trial)
                trials[i] = trial

            trial_res = pool.map(_evaluate_candidate, [(ind, l1_penalty) for ind in trials])
            for i, (t_fit, t_bd) in enumerate(trial_res):
                if t_fit <= fitness[i]:
                    pop[i] = trials[i]
                    fitness[i] = t_fit
                    breakdowns[i] = t_bd

            best_idx = int(np.argmin(fitness))
            history.append({
                "gen": gen,
                "best_loss": float(fitness[best_idx]),
                "breakdown": breakdowns[best_idx],
                "nnz": int(np.sum(np.abs(pop[best_idx]) > 1e-3)),
            })
            if gen % 5 == 0 or gen == generations:
                print(
                    f"  Gen {gen:2d} | Best Loss={fitness[best_idx]:.4f} | "
                    f"W1={breakdowns[best_idx]['W1_PAI_Mixed']:.2f}% "
                    f"W2={breakdowns[best_idx]['W2_Regime_Shift']:.2f}% "
                    f"W3={breakdowns[best_idx]['W3_Heavy_Straggler']:.2f}% | "
                    f"nnz={int(np.sum(np.abs(pop[best_idx]) > 1e-3))}",
                    flush=True,
                )
                _save_checkpoint(out_path, {
                    "status": f"gen_{gen}",
                    "feature_names": FEATURE_NAMES,
                    "best_full_weights": pop[best_idx].tolist(),
                    "train_baselines": baselines,
                    "history": history,
                })

        # [3/3] 오컴의 면도날 증류 (pool.apply / pool.map 으로 안전 실행)
        print("\n[3/3] 오컴의 면도날 증류(Backward Feature Pruning)로 핵심 최소 수식 추출 중...", flush=True)
        best_full = pop[int(np.argmin(fitness))].copy()
        best_full_raw, _ = pool.apply(_evaluate_candidate, ((best_full, 0.0),))
        print(f"  - 증류 전 Full 모델 Raw Loss={best_full_raw:.4f} (nnz={int(np.sum(np.abs(best_full)>1e-6))})", flush=True)

        pruned = best_full.copy()
        current_raw = best_full_raw
        while True:
            active = np.flatnonzero(np.abs(pruned) > 1e-6)
            if len(active) <= 3:
                break
            candidates = []
            for idx in active:
                cand = pruned.copy()
                cand[idx] = 0.0
                candidates.append((idx, cand))
            res = pool.map(_evaluate_candidate, [(c, 0.0) for _, c in candidates])
            best_drop_i = int(np.argmin([r[0] for r in res]))
            drop_fit = res[best_drop_i][0]
            # 성능 저하 0.4% 이내라면 항 제거
            if drop_fit <= current_raw * 1.004:
                dropped_feat = FEATURE_NAMES[candidates[best_drop_i][0]]
                pruned = candidates[best_drop_i][1]
                current_raw = drop_fit
                print(
                    f"  - 항 제거: {dropped_feat:<22} -> Raw Loss={current_raw:.4f} "
                    f"(남은 항 {int(np.sum(np.abs(pruned)>1e-6))}개)",
                    flush=True,
                )
            else:
                break

        max_abs = float(np.max(np.abs(pruned)))
        if max_abs > 1e-6:
            pruned = np.round(pruned / max_abs, 4)

    elapsed = time.perf_counter() - t_start
    print(f"\n[진화 완료] 총 소요 시간: {elapsed:.1f}초")

    print("\n=== 발견된 최종 핵심 수식 (Evo-STF Distilled, 낮을수록 좋은 노드) ===")
    for i, name in enumerate(FEATURE_NAMES):
        if abs(pruned[i]) > 1e-6:
            sign = "+" if pruned[i] > 0 else "-"
            print(f"  {sign} {abs(pruned[i]):.4f} * {name}")

    payload = {
        "status": "completed",
        "feature_names": FEATURE_NAMES,
        "best_full_weights": best_full.tolist(),
        "best_full_raw_loss": float(best_full_raw),
        "distilled_weights": pruned.tolist(),
        "distilled_raw_loss": float(current_raw),
        "train_baselines": baselines,
        "history": history,
        "elapsed_sec": elapsed,
    }
    _save_checkpoint(out_path, payload)
    print(f"[저장 완료] {out_path}")


if __name__ == "__main__":
    main()
