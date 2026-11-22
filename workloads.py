"""
현실형 AI 클러스터 워크로드 생성기 (Alibaba PAI / Microsoft Philly 특성 반영)

5가지 평가 시나리오(학습/진화용 + 미지 분포 OOD + 반례 테스트)를 제공한다:
1. W1_PAI_Mixed       : 분수 GPU(0.25/0.5) 추론 + 1/2/4/8-GPU 학습 혼합 + Heavy-Tail(Pareto) 수명
2. W2_Regime_Shift    : 주야간 분포 급변 (소형 추론 폭주 -> 4/8-GPU 대형 학습 폭주)
3. W3_Heavy_Straggler : 극단적 수명 편차 (초단기 디버깅/추론 vs 초장기 학습 혼재)
4. W4_Memoryless_OOD  : [반례 테스트] 지수분포 수명 (린디 효과 = 0 인 메모리리스 환경)
5. W5_Skewed_Vec_OOD  : [OOD 테스트] CPU/RAM 비대칭 편향이 극심한 데이터 전처리·서빙 혼합
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from simulator import Job


# 작업 프로파일: (gpu, cpu, ram)
STANDARD_PROFILES: List[Tuple[float, float, float]] = [
    (0.25, 4.0, 24.0),    # 경량 추론 / 임베딩
    (0.50, 6.0, 48.0),    # 중형 추론 / 파인튜닝
    (1.0,  8.0, 64.0),    # 단일 GPU 학습
    (2.0, 16.0, 128.0),   # 2-GPU 분산 학습
    (4.0, 32.0, 256.0),   # 4-GPU 분산 학습
    (8.0, 60.0, 480.0),   # 8-GPU 풀노드 대형 학습
]

SKEWED_PROFILES: List[Tuple[float, float, float]] = [
    (0.25, 16.0, 160.0),  # 초고메모리 RAG/데이터 로더 + 소량 GPU
    (0.50, 24.0, 96.0),   # CPU 전처리 병목 + 반장 GPU
    (1.0,   4.0, 192.0),  # KV-Cache 메모리 과다 LLM 서빙
    (2.0,  28.0, 128.0),  # CPU 편중 분산 학습
    (4.0,  32.0, 320.0),  # RAM 편중 4-GPU 학습
    (8.0,  60.0, 480.0),  # 8-GPU 풀노드 학습
]


def _sample_pareto_duration(rng: np.random.Generator, alpha: float, scale: float, max_dur: float) -> float:
    """절단 파레토(Heavy-tail) 수명 샘플링 (단위: 시간)."""
    raw = (rng.pareto(alpha) + 1.0) * scale
    return float(min(raw, max_dur))


def generate_w1_pai_mixed(seed: int = 42, num_jobs: int = 4000, target_load: float = 0.88) -> List[Job]:
    """
    W1: Alibaba PAI 논문 스타일 혼합 워크로드.
    분수 GPU(0.25, 0.5)와 멀티 GPU(1, 2, 4, 8)가 공존하며,
    작업 수명은 강한 Heavy-tail(Pareto) 분포를 따른다.
    """
    rng = np.random.default_rng(seed)
    probs = np.array([0.32, 0.24, 0.20, 0.10, 0.08, 0.06], dtype=np.float64)
    probs /= probs.sum()

    # 각 프로파일별 기대 GPU-시간을 맞추어 target_load(클러스터 총 512 GPU 기준)에 해당하는 도착률 계산
    total_gpus = 512.0
    durations = np.empty(num_jobs, dtype=np.float64)
    chosen_idx = rng.choice(len(STANDARD_PROFILES), size=num_jobs, p=probs)

    for i, idx in enumerate(chosen_idx):
        gpu = STANDARD_PROFILES[idx][0]
        # 대형 학습 잡일수록 평균적으로 더 오래 돌지만, 소형 잡에도 장기 서빙(straggler)이 존재
        if gpu >= 4.0:
            durations[i] = _sample_pareto_duration(rng, alpha=1.5, scale=1.2, max_dur=36.0)
        else:
            durations[i] = _sample_pareto_duration(rng, alpha=1.35, scale=0.25, max_dur=24.0)

    mean_gpu_hours = float(np.mean([STANDARD_PROFILES[idx][0] * durations[i] for i, idx in enumerate(chosen_idx)]))
    arrival_rate = (target_load * total_gpus) / max(mean_gpu_hours, 1e-6)
    inter_arrivals = rng.exponential(1.0 / arrival_rate, size=num_jobs)
    arrivals = np.cumsum(inter_arrivals)

    jobs: List[Job] = []
    for i in range(num_jobs):
        g, c, r = STANDARD_PROFILES[chosen_idx[i]]
        # CPU/RAM에 약간의 현실적 변동성(±15%) 부여
        c_var = min(64.0, round(c * rng.uniform(0.85, 1.15), 2))
        r_var = min(512.0, round(r * rng.uniform(0.85, 1.15), 2))
        jobs.append(Job(i, float(arrivals[i]), float(durations[i]), g, c_var, r_var))
    return jobs


def generate_w2_regime_shift(seed: int = 42, num_jobs: int = 4000, target_load: float = 0.86) -> List[Job]:
    """
    W2: 주야간 워크로드 분포 급변 시나리오.
    - 1구간(낮): 0.25/0.5/1-GPU 소형 추론·실험 잡 위주
    - 2구간(밤): 4/8-GPU 대형 학습 잡 급증
    - 3구간(다시 낮): 소형 잡 복귀
    - 4구간(다시 밤): 8-GPU 폭주
    """
    rng = np.random.default_rng(seed)
    day_probs = np.array([0.42, 0.30, 0.18, 0.06, 0.03, 0.01])
    night_probs = np.array([0.10, 0.12, 0.18, 0.16, 0.22, 0.22])

    total_gpus = 512.0
    chosen_idx = np.empty(num_jobs, dtype=np.int32)
    durations = np.empty(num_jobs, dtype=np.float64)

    phase_len = num_jobs // 4
    for i in range(num_jobs):
        phase = (i // phase_len) % 2
        probs = day_probs if phase == 0 else night_probs
        idx = int(rng.choice(len(STANDARD_PROFILES), p=probs))
        chosen_idx[i] = idx
        gpu = STANDARD_PROFILES[idx][0]
        if gpu >= 4.0:
            durations[i] = _sample_pareto_duration(rng, alpha=1.6, scale=1.0, max_dur=28.0)
        else:
            durations[i] = _sample_pareto_duration(rng, alpha=1.4, scale=0.30, max_dur=20.0)

    mean_gpu_hours = float(np.mean([STANDARD_PROFILES[chosen_idx[i]][0] * durations[i] for i in range(num_jobs)]))
    arrival_rate = (target_load * total_gpus) / max(mean_gpu_hours, 1e-6)
    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, size=num_jobs))

    jobs: List[Job] = []
    for i in range(num_jobs):
        g, c, r = STANDARD_PROFILES[chosen_idx[i]]
        jobs.append(Job(i, float(arrivals[i]), float(durations[i]), g, c, r))
    return jobs


def generate_w3_heavy_straggler(seed: int = 42, num_jobs: int = 4000, target_load: float = 0.88) -> List[Job]:
    """
    W3: 극단적 수명 편차(Temporal Straggler) 워크로드.
    짧은 디버깅/배치 작업(수 분)과 초장기 서빙/학습 작업(수십 시간)이
    동일한 자원 규격으로 뒤섞여 들어오는 최악의 시공간 단편화 시나리오.
    """
    rng = np.random.default_rng(seed)
    probs = np.array([0.30, 0.25, 0.20, 0.10, 0.08, 0.07])
    probs /= probs.sum()

    total_gpus = 512.0
    chosen_idx = rng.choice(len(STANDARD_PROFILES), size=num_jobs, p=probs)
    durations = np.empty(num_jobs, dtype=np.float64)

    for i in range(num_jobs):
        # 85%는 아주 짧은 잡(0.05~0.4시간), 15%는 초장기 잡(8~48시간) — 극단적 바이모달+헤비테일
        if rng.random() < 0.85:
            durations[i] = float(rng.uniform(0.05, 0.40))
        else:
            durations[i] = _sample_pareto_duration(rng, alpha=1.6, scale=8.0, max_dur=48.0)

    mean_gpu_hours = float(np.mean([STANDARD_PROFILES[chosen_idx[i]][0] * durations[i] for i in range(num_jobs)]))
    arrival_rate = (target_load * total_gpus) / max(mean_gpu_hours, 1e-6)
    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, size=num_jobs))

    jobs: List[Job] = []
    for i in range(num_jobs):
        g, c, r = STANDARD_PROFILES[chosen_idx[i]]
        jobs.append(Job(i, float(arrivals[i]), float(durations[i]), g, c, r))
    return jobs


def generate_w4_memoryless_ood(seed: int = 42, num_jobs: int = 4000, target_load: float = 0.88) -> List[Job]:
    """
    W4 [반례 검증용 OOD]: 순수 지수분포(Exponential) 수명 워크로드.
    지수분포는 메모리리스(Memoryless) 성질을 가지므로 '오래 돌수록 앞으로도 오래 돈다(린디 효과)'가
    수학적으로 완전히 거짓이다. 시간 경과 휴리스틱이 과적합되었는지 판별하는 반례 테스트.
    """
    rng = np.random.default_rng(seed)
    probs = np.array([0.30, 0.24, 0.20, 0.11, 0.08, 0.07])
    probs /= probs.sum()

    total_gpus = 512.0
    chosen_idx = rng.choice(len(STANDARD_PROFILES), size=num_jobs, p=probs)
    durations = rng.exponential(scale=2.0, size=num_jobs)

    mean_gpu_hours = float(np.mean([STANDARD_PROFILES[chosen_idx[i]][0] * durations[i] for i in range(num_jobs)]))
    arrival_rate = (target_load * total_gpus) / max(mean_gpu_hours, 1e-6)
    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, size=num_jobs))

    jobs: List[Job] = []
    for i in range(num_jobs):
        g, c, r = STANDARD_PROFILES[chosen_idx[i]]
        jobs.append(Job(i, float(arrivals[i]), float(durations[i]), g, c, r))
    return jobs


def generate_w5_skewed_vec_ood(seed: int = 42, num_jobs: int = 4000, target_load: float = 0.85) -> List[Job]:
    """
    W5 [OOD 테스트]: CPU/RAM 비대칭 편향이 극심한 다차원 벡터 워크로드.
    GPU는 남는데 RAM이나 CPU가 먼저 고갈되어 노드가 막히는 다차원 자원 고립 테스트.
    """
    rng = np.random.default_rng(seed)
    probs = np.array([0.32, 0.24, 0.20, 0.10, 0.08, 0.06])
    probs /= probs.sum()

    chosen_idx = rng.choice(len(SKEWED_PROFILES), size=num_jobs, p=probs)
    durations = np.array(
        [_sample_pareto_duration(rng, alpha=1.45, scale=0.6, max_dur=30.0) for _ in range(num_jobs)],
        dtype=np.float64,
    )

    # GPU/CPU/RAM 중 가장 먼저 병목이 되는 차원의 부하가 target_load(0.85)가 되도록 보정
    total_caps = np.array([512.0, 64.0 * 64.0, 64.0 * 512.0], dtype=np.float64)
    mean_demands = np.mean(
        [np.array(SKEWED_PROFILES[chosen_idx[i]]) * durations[i] for i in range(num_jobs)],
        axis=0,
    )
    max_norm_demand = float(np.max(mean_demands / total_caps))
    arrival_rate = target_load / max(max_norm_demand, 1e-6)
    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, size=num_jobs))

    jobs: List[Job] = []
    for i in range(num_jobs):
        g, c, r = SKEWED_PROFILES[chosen_idx[i]]
        jobs.append(Job(i, float(arrivals[i]), float(durations[i]), g, c, r))
    return jobs


WORKLOAD_SUITE = {
    "W1_PAI_Mixed": generate_w1_pai_mixed,
    "W2_Regime_Shift": generate_w2_regime_shift,
    "W3_Heavy_Straggler": generate_w3_heavy_straggler,
    "W4_Memoryless_OOD": generate_w4_memoryless_ood,
    "W5_Skewed_Vec_OOD": generate_w5_skewed_vec_ood,
}
