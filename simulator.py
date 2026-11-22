"""
동적 이기종 다차원 GPU 클러스터 시뮬레이터 (Non-Clairvoyant Event-Driven)

실제 AI 데이터센터(Alibaba PAI, Microsoft Philly 등)의 물리 제약을 반영한다:
- 각 노드는 8개의 개별 물리 GPU 슬롯(각 0.0~1.0)과 연속 자원(CPU, RAM)을 보유
- 분수 GPU 요청(g < 1.0)은 단일 물리 GPU 슬롯 내부에 수용되어야 함
- 정수/멀티 GPU 요청(g in {1, 2, 4, 8})은 완전히 비어 있는(1.0) 물리 GPU가 g개 이상 필요
- 스케줄러는 도착한 작업의 미래 수명을 절대 알 수 없음(Non-clairvoyant 원칙 엄수)
- 대신 현재 노드에서 실행 중인 작업들의 경과 시간(running age)은 실시간 관찰 가능
"""

from __future__ import annotations

import bisect
from collections import deque
import heapq
import math
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class OnlineTemporalStats:
    """
    스케줄러가 실시간(Non-Clairvoyant)으로 관찰 가능한 시공간 통계 요약.
    미래 수명 정보는 절대 포함하지 않으며, 오직 현재 실행 중인 작업의 경과 시간(age)과
    이미 완료된 과거 작업의 실측 수명(realized duration)만을 기반으로 추정한다.
    """
    corr_gpu_age: float             # 현재 실행 중 작업의 선형 피어슨 상관 Corr(gpu, age)
    corr_log_gpu_age: float         # 현재 실행 중 작업의 로그 상관 Corr(log(1+gpu), log(1+age))
    corr_completed_gpu_dur: float   # 최근 완료 작업의 실측 상관 Corr(log(1+gpu), log(1+dur))
    mean_age: float                 # 현재 실행 중 작업들의 평균 경과 시간
    mean_dur: float                 # 최근 완료된 과거 작업들의 평균 실측 수명 (시간 척도 정규화 기준)
    age_cv: float                   # 현재 실행 중 작업 경과 시간의 변동계수 (std / mean)
    dur_cv: float                   # 최근 완료 작업 수명의 변동계수 (헤비테일/린디 유효성 지표)
    n_running: int                  # 현재 실행 중인 작업 수
    n_completed: int                # 최근 완료 버퍼 내 작업 수


@dataclass(slots=True)
class Job:
    """도착하는 개별 클러스터 작업 정의."""
    job_id: int
    arrival_time: float
    duration: float  # 스케줄러에게 절대 노출되지 않는 실제 수명
    gpu: float       # 요청 GPU 총합 (예: 0.25, 0.5, 1~8, 멀티노드 시 16~64)
    cpu: float       # 요청 CPU 코어 총합
    ram: float       # 요청 RAM (GB) 총합
    gpu_spec: Optional[str] = None  # 허용 GPU 기종 제약 (예: "V100M32|T4", None이면 무제한)
    num_nodes: int = 1              # 요구 노드 수 (1이면 단일 노드, >1이면 Gang Allocation)


@dataclass
class ClusterConfig:
    """클러스터 하드웨어 규격 설정 (동종 클러스터 기본값 및 이기종·랙 토폴로지 지원)."""
    num_nodes: int = 64
    gpus_per_node: int = 8
    cpu_per_node: float = 64.0
    ram_per_node: float = 512.0
    node_gpus: Optional[Tuple[int, ...]] = None
    node_cpus: Optional[Tuple[float, ...]] = None
    node_rams: Optional[Tuple[float, ...]] = None
    node_models: Optional[Tuple[str, ...]] = None
    nodes_per_rack: int = 8


@dataclass
class SimulationMetrics:
    """단일 시뮬레이션 실행(즉시 거절 Blocking 모드)의 정량 평가 지표."""
    scheduler_name: str
    workload_name: str
    seed: int
    total_jobs: int
    rejected_jobs: int
    job_blocking_rate: float           # 거절된 작업 비율 (%)
    gpu_hour_blocking_rate: float      # 거절된 작업의 GPU-시간 비율 (%) — 핵심 지표 1
    avg_stranded_gpu_ratio: float      # 거절 시점 클러스터 유휴 GPU 비율 (%) — 순수 단편화 지표 2
    active_node_hours: float           # 누적 활성 노드-시간 (전력·운영비 지표 3)
    avg_gpu_utilization: float         # 시간 가중 평균 GPU 사용률 (%)
    avg_decision_us: float             # 스케줄링 1회당 평균 소요 시간 (마이크로초, 핵심 지표 4)
    multinode_blocking_rate: float = 0.0     # 멀티노드 작업(num_nodes>1) GPU-시간 거절률 (%)
    single_rack_locality_rate: float = 100.0 # 배치된 멀티노드 작업의 단일 랙 집적률 (%)
    avg_racks_spanned: float = 1.0           # 배치된 멀티노드 작업의 평균 스팬 랙 수


@dataclass
class QueueSimulationMetrics:
    """대기 큐(Pending Queue) 기반 시뮬레이션의 정량 평가 지표."""
    scheduler_name: str
    workload_name: str
    seed: int
    queue_policy: str                  # "fifo" 또는 "backfill"
    total_jobs: int
    completed_jobs: int
    mean_jct: float                    # 평균 작업 완료 시간 (JCT = 완료시각 - 도착시각)
    gpu_weighted_jct: float            # GPU-가중 평균 JCT (대형 작업 정체 민감도)
    p95_jct: float                     # 95백분위 꼬리 JCT
    p99_jct: float                     # 99백분위 꼬리 JCT
    mean_wait_time: float              # 평균 큐 대기 시간
    p95_wait_time: float               # 95백분위 꼬리 대기 시간
    mean_slowdown: float               # 평균 정규화 감속비 (JCT / duration)
    makespan: float                    # 전체 워크로드 완료 시간 (마지막 종료 - 최초 도착)
    max_queue_len: int                 # 최대 대기 큐 길이
    avg_queue_len: float               # 시간 가중 평균 대기 큐 길이
    avg_stranded_gpu_on_wait: float    # 큐 대기 발생 시점의 평균 클러스터 유휴 GPU 비율 (%)
    active_node_hours: float           # 누적 활성 노드-시간
    avg_gpu_utilization: float         # 시간 가중 평균 GPU 사용률 (%)
    avg_decision_us: float             # 스케줄링 1회당 평균 소요 시간 (마이크로초)
    single_rack_locality_rate: float = 100.0 # 멀티노드 작업의 단일 랙 집적률 (%)
    avg_racks_spanned: float = 1.0           # 멀티노드 작업의 평균 스팬 랙 수


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """척도 불변(Scale-Invariant) 안전한 피어슨 상관계수 계산 (분산이 0이면 0.0 반환)."""
    if len(x) < 3:
        return 0.0
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    nx = float(np.dot(dx, dx))
    ny = float(np.dot(dy, dy))
    sx = float(np.dot(x, x))
    sy = float(np.dot(y, y))
    # 부동소수점 반올림 오차(상수 배열의 1e-16 잔차)는 상대 임계값으로 차단하고,
    # 양변이 동일 척도 제곱(c^2)으로 비례하므로 시간/자원 단위 변환에 완전히 불변이다.
    if nx <= 0.0 or ny <= 0.0 or nx <= 1e-24 * max(sx, 0.0) or ny <= 1e-24 * max(sy, 0.0):
        return 0.0
    denom = math.sqrt(nx * ny)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float(np.clip(float(np.dot(dx, dy)) / denom, -1.0, 1.0))


class ClusterState:
    """스케줄러에게 공개되는 실시간 클러스터 상태 (미래 수명 정보 완전 배제)."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        n = config.num_nodes

        if config.node_gpus is not None:
            if (
                len(config.node_gpus) != n
                or (config.node_cpus is not None and len(config.node_cpus) != n)
                or (config.node_rams is not None and len(config.node_rams) != n)
                or (config.node_models is not None and len(config.node_models) != n)
            ):
                raise ValueError("이기종 ClusterConfig의 노드별 배열 길이가 num_nodes와 일치해야 합니다.")

            node_g = np.asarray(config.node_gpus, dtype=np.int32)
            g_max = int(np.max(node_g)) if n > 0 else config.gpus_per_node
            self.gpu_slots = np.zeros((n, max(g_max, 1)), dtype=np.float64)
            for i in range(n):
                if node_g[i] > 0:
                    self.gpu_slots[i, : int(node_g[i])] = 1.0
            self.node_gpu_cap = node_g.astype(np.float64)
            self.node_cpu_cap = (
                np.asarray(config.node_cpus, dtype=np.float64)
                if config.node_cpus is not None
                else np.full(n, config.cpu_per_node, dtype=np.float64)
            )
            self.node_ram_cap = (
                np.asarray(config.node_rams, dtype=np.float64)
                if config.node_rams is not None
                else np.full(n, config.ram_per_node, dtype=np.float64)
            )
        else:
            g = config.gpus_per_node
            # 노드별 물리 GPU 슬롯 잔여량 (N, G), 초기값 1.0
            self.gpu_slots = np.ones((n, g), dtype=np.float64)
            self.node_gpu_cap = np.full(n, float(g), dtype=np.float64)
            self.node_cpu_cap = np.full(n, config.cpu_per_node, dtype=np.float64)
            self.node_ram_cap = np.full(n, config.ram_per_node, dtype=np.float64)

        self.total_gpu_capacity: float = float(np.sum(self.node_gpu_cap))
        self.max_gpu_cap: float = float(np.max(self.node_gpu_cap)) if n > 0 else float(config.gpus_per_node)
        self.min_gpu_cap: float = float(np.min(self.node_gpu_cap)) if n > 0 else float(config.gpus_per_node)
        self.is_homogeneous_gpus: bool = (self.max_gpu_cap - self.min_gpu_cap) <= 1e-9
        self.node_models: Optional[Tuple[str, ...]] = config.node_models
        self._model_mask_cache: Dict[str, np.ndarray] = {}
        self.model_gpu_share: Dict[str, float] = {}
        if self.node_models is not None and self.total_gpu_capacity > 0.0:
            for i, m in enumerate(self.node_models):
                self.model_gpu_share[m] = self.model_gpu_share.get(m, 0.0) + float(self.node_gpu_cap[i])
            for m in list(self.model_gpu_share.keys()):
                self.model_gpu_share[m] /= self.total_gpu_capacity

        # 현재 의사결정 대상 작업의 gpu_spec 및 최근 도착 spec / 노드 수 히스토리 (Non-clairvoyant)
        self.current_gpu_spec: Optional[str] = None
        self.recent_specs: Deque[Optional[str]] = deque(maxlen=200)
        self.recent_node_counts: Deque[int] = deque(maxlen=200)

        # 랙 토폴로지 매핑 (각 노드의 소속 랙 ID)
        self.nodes_per_rack: int = max(1, int(config.nodes_per_rack))
        self.node_rack_ids: np.ndarray = np.arange(n, dtype=np.int32) // self.nodes_per_rack
        self.num_racks: int = int(self.node_rack_ids[-1]) + 1 if n > 0 else 0

        # 노드별 연속 자원 잔여량 (N,)
        self.cpu_free = self.node_cpu_cap.copy()
        self.ram_free = self.node_ram_cap.copy()

        # 노드별 현재 실행 중인 작업 수 및 실제 시작 시각 목록 (오름차순 유지)
        self.node_job_starts: List[List[float]] = [[] for _ in range(n)]
        self.node_oldest_start = np.full(n, np.inf, dtype=np.float64)
        self.node_newest_start = np.full(n, np.inf, dtype=np.float64)
        self.node_job_count = np.zeros(n, dtype=np.int32)

        # 현재 실행 중인 작업 레지스트리: id(allocations) -> (node_idx, actual_start_time, gpu)
        # 동일 Job 인스턴스가 동시에 중복 투입되어도 충돌 없이 개별 할당을 추적한다.
        self.active_jobs: Dict[int, Tuple[int, float, float]] = {}
        # 외부 호출자가 allocations 리스트를 복사해 전달하는 경우를 대비한 보조 인덱스
        self._active_by_job: Dict[Tuple[int, int], Deque[int]] = {}
        # 최근 완료된 과거 작업 버퍼 (100% 인과적·비예측): (gpu, realized_duration)
        self.recent_completions: Deque[Tuple[float, float]] = deque(maxlen=256)
        # 정상상태 길이편향을 완화하는 Kaplan-Meier 추정용 동일 진입 코호트 버퍼: [start_time, end_time_or_None, gpu]
        self.recent_cohort: Deque[List[object]] = deque(maxlen=1024)
        self._cohort_by_alloc: Dict[int, List[object]] = {}

        # 온라인 통계 갱신용 결정적 이벤트 카운터 및 캐시 (시간 단위 불변, O(1) 상각 비용)
        self._ops_counter: int = 0
        self._stats_last_ops: int = -10**9
        self._cached_temporal_stats: OnlineTemporalStats = OnlineTemporalStats(
            corr_gpu_age=0.0,
            corr_log_gpu_age=0.0,
            corr_completed_gpu_dur=0.0,
            mean_age=0.0,
            mean_dur=0.0,
            age_cv=1.0,
            dur_cv=1.0,
            n_running=0,
            n_completed=0,
        )

        # 현재 시뮬레이션 시각
        self.now: float = 0.0

        # 최근 도착한 작업 요청 히스토리 (온라인 적응형 알고리즘용, 미래 수명 없음)
        self.recent_requests: List[Tuple[float, float, float]] = []

    @property
    def ops_version(self) -> int:
        """할당/해제 이벤트가 발생할 때마다 1씩 증가하는 공개 단조 상태 버전."""
        return self._ops_counter

    def get_online_temporal_stats(self, refresh_interval: int = 16) -> OnlineTemporalStats:
        """
        현재 실행 중인 작업과 최근 완료된 작업으로부터 시공간 상관 통계
        (Corr(gpu, age), 평균 경과시간, 변동계수 등)를 O(1) 상각 비용으로 반환한다.
        실측 시간 척도가 아직 관측되지 않은 콜드 스타트 구간에서는 하드코딩된 1.0 대신 0.0을 반환하여
        완전한 시간 단위 불변성(Clock-Unit Invariance)을 보장한다.
        """
        n_run = len(self.active_jobs)
        need_scale_init = (self._cached_temporal_stats.mean_age <= 0.0 and n_run >= 1)
        if (
            refresh_interval > 0
            and not need_scale_init
            and self._ops_counter - self._stats_last_ops < refresh_interval
        ):
            return self._cached_temporal_stats

        self._stats_last_ops = self._ops_counter
        now = self.now
        n_comp = len(self.recent_completions)

        corr_lin = 0.0
        corr_log = 0.0
        mean_age = 0.0
        age_cv = 1.0

        if n_run >= 1:
            # active_jobs에서 (gpu, running_age) 벡터를 단일 NumPy 변환으로 고속 추출
            arr = np.asarray(list(self.active_jobs.values()), dtype=np.float64)
            ages = np.maximum(0.0, now - arr[:, 1])
            m_age = float(np.mean(ages))
            if m_age > 0.0:
                mean_age = m_age
                if n_run >= 6:
                    gs = arr[:, 2]
                    age_cv = float(np.std(ages) / m_age)
                    # 시간 척도에 완전히 불변한 정규화 로그 경과시간으로 상관 계산
                    corr_lin = _safe_pearson(gs, ages)
                    corr_log = _safe_pearson(np.log1p(gs), np.log1p(ages / m_age))

        corr_comp = 0.0
        mean_dur = mean_age
        dur_cv = 1.0
        if n_comp >= 12:
            carr = np.asarray(self.recent_completions, dtype=np.float64)
            c_gs = carr[:, 0]
            c_ds = carr[:, 1]
            m_dur = float(np.mean(c_ds))
            if m_dur > 0.0:
                mean_dur = m_dur
                dur_cv = float(np.std(c_ds) / m_dur)
                # 평균 수명으로 정규화하여 시간 단위(초/분/시간)에 완전히 불변한 로그 상관 계산
                corr_comp = _safe_pearson(np.log1p(c_gs), np.log1p(c_ds / m_dur))

        self._cached_temporal_stats = OnlineTemporalStats(
            corr_gpu_age=corr_lin,
            corr_log_gpu_age=corr_log,
            corr_completed_gpu_dur=corr_comp,
            mean_age=mean_age,
            mean_dur=mean_dur,
            age_cv=age_cv,
            dur_cv=dur_cv,
            n_running=n_run,
            n_completed=n_comp,
        )
        return self._cached_temporal_stats

    def get_feasible_nodes(
        self,
        gpu: float,
        cpu: float,
        ram: float,
        gpu_spec: Optional[str] = None,
    ) -> np.ndarray:
        """주어진 자원 요청 및 GPU 기종 제약을 수용 가능한 모든 노드 인덱스를 반환한다."""
        eps = 1e-9
        res_ok = (self.cpu_free + eps >= cpu) & (self.ram_free + eps >= ram)
        if gpu < 1.0 - eps:
            # 분수 GPU: 단일 물리 GPU 슬롯에 들어갈 수 있어야 함
            gpu_ok = np.any(self.gpu_slots + eps >= gpu, axis=1)
        else:
            # 정수/멀티 GPU: 완전히 빈(1.0) GPU 슬롯이 int(round(gpu))개 이상 필요
            needed = int(round(gpu))
            empty_gpus = np.sum(self.gpu_slots >= 1.0 - eps, axis=1)
            gpu_ok = empty_gpus >= needed

        mask = res_ok & gpu_ok
        if gpu_spec and self.node_models is not None:
            model_mask = self._model_mask_cache.get(gpu_spec)
            if model_mask is None:
                allowed = {tok.strip() for tok in gpu_spec.split("|") if tok.strip()}
                model_mask = np.array([m in allowed for m in self.node_models], dtype=bool)
                self._model_mask_cache[gpu_spec] = model_mask
            mask &= model_mask
        return np.flatnonzero(mask)

    def node_free_gpu_total(self) -> np.ndarray:
        """노드별 총 잔여 GPU 합계 (N,)."""
        return np.sum(self.gpu_slots, axis=1)

    def node_empty_gpu_count(self) -> np.ndarray:
        """노드별 완전히 비어 있는 물리 GPU 개수 (N,)."""
        return np.sum(self.gpu_slots >= 1.0 - 1e-9, axis=1)

    def node_age_stats(self, node_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        지정 노드들에서 현재 실행 중인 작업들의 경과 시간(age) 통계를 반환한다.
        반환값: (max_age, mean_age, min_age) — 빈 노드는 모두 0.0
        """
        k = len(node_indices)
        max_age = np.zeros(k, dtype=np.float64)
        mean_age = np.zeros(k, dtype=np.float64)
        min_age = np.zeros(k, dtype=np.float64)
        now = self.now
        starts_list = self.node_job_starts
        for i, idx in enumerate(node_indices):
            starts = starts_list[idx]
            if starts:
                # starts는 도착 순이므로 starts[0]이 가장 오래된 작업, starts[-1]이 가장 최근 작업
                s_min = starts[0]
                s_max = starts[-1]
                max_age[i] = now - s_min
                min_age[i] = now - s_max
                mean_age[i] = now - (sum(starts) / len(starts))
        return max_age, mean_age, min_age

    def allocate(self, node_idx: int, job: Job) -> List[Tuple[int, float]]:
        """노드에 작업을 배치하고 할당된 (gpu_slot_idx, allocated_amount) 목록을 반환한다."""
        eps = 1e-9
        slots = self.gpu_slots[node_idx]
        allocations: List[Tuple[int, float]] = []

        if job.gpu < 1.0 - eps:
            # 노드 내부 GPU 슬롯 선택: 이미 부분 사용 중이면서 들어가는 가장 타이트한 슬롯(Best-Fit) 우선
            feasible_slots = np.flatnonzero(slots + eps >= job.gpu)
            # 남은 용량이 가장 작은 슬롯 선택 (완전 빈 1.0 GPU를 최대한 보존)
            best_slot = int(feasible_slots[np.argmin(slots[feasible_slots])])
            slots[best_slot] = max(0.0, slots[best_slot] - job.gpu)
            allocations.append((best_slot, job.gpu))
        else:
            needed = int(round(job.gpu))
            empty_slots = np.flatnonzero(slots >= 1.0 - eps)
            chosen = empty_slots[:needed]
            for s in chosen:
                slots[s] = 0.0
                allocations.append((int(s), 1.0))

        self.cpu_free[node_idx] = max(0.0, self.cpu_free[node_idx] - job.cpu)
        self.ram_free[node_idx] = max(0.0, self.ram_free[node_idx] - job.ram)

        # 실제 실행 시작 시각 기록: Blocking 모드에서는 now == arrival_time,
        # Pending Queue 모드에서는 now >= arrival_time (큐 대기 시간을 실행 경과시간으로 오인 방지)
        start_tol = 1e-12 * abs(job.arrival_time)
        start_time = self.now if self.now >= job.arrival_time - start_tol else job.arrival_time
        starts = self.node_job_starts[node_idx]
        if not starts or start_time >= starts[-1]:
            starts.append(start_time)
        else:
            bisect.insort(starts, start_time)
        self.node_oldest_start[node_idx] = starts[0]
        self.node_newest_start[node_idx] = starts[-1]
        self.node_job_count[node_idx] += 1

        alloc_key = id(allocations)
        job_key = (job.job_id, id(job))
        self.active_jobs[alloc_key] = (node_idx, start_time, job.gpu)
        cohort_entry: List[object] = [start_time, None, job.gpu]
        self.recent_cohort.append(cohort_entry)
        self._cohort_by_alloc[alloc_key] = cohort_entry
        q_keys = self._active_by_job.get(job_key)
        if q_keys is None:
            self._active_by_job[job_key] = deque([alloc_key])
        else:
            q_keys.append(alloc_key)
        self._ops_counter += 1
        return allocations

    def release(
        self,
        node_idx: int,
        job: Job,
        allocations: List[Tuple[int, float]],
    ) -> None:
        """종료된 작업의 자원을 노드에 반환한다."""
        slots = self.gpu_slots[node_idx]
        for slot_idx, amount in allocations:
            slots[slot_idx] = min(1.0, slots[slot_idx] + amount)
        self.cpu_free[node_idx] = min(float(self.node_cpu_cap[node_idx]), self.cpu_free[node_idx] + job.cpu)
        self.ram_free[node_idx] = min(float(self.node_ram_cap[node_idx]), self.ram_free[node_idx] + job.ram)

        alloc_key = id(allocations)
        resolved_key = alloc_key
        job_key = (job.job_id, id(job))
        entry = self.active_jobs.pop(alloc_key, None)
        q_keys = self._active_by_job.get(job_key)
        if entry is not None:
            if q_keys is not None:
                try:
                    q_keys.remove(alloc_key)
                except ValueError:
                    pass
                if not q_keys:
                    del self._active_by_job[job_key]
        elif q_keys:
            # 외부에서 allocations 리스트 복사본을 전달한 경우 동일 노드의 할당을 안전하게 회수
            matched_key = None
            for k in q_keys:
                cand = self.active_jobs.get(k)
                if cand is not None and cand[0] == node_idx:
                    matched_key = k
                    break
            if matched_key is None:
                matched_key = q_keys[0]
            q_keys.remove(matched_key)
            if not q_keys:
                del self._active_by_job[job_key]
            resolved_key = matched_key
            entry = self.active_jobs.pop(matched_key, None)

        cohort_entry = self._cohort_by_alloc.pop(resolved_key, None)
        if cohort_entry is not None:
            cohort_entry[1] = self.now

        start_time = entry[1] if entry is not None else job.arrival_time

        # 시작 시각 제거 및 최장 경과 시작 시각 O(1) 갱신
        starts = self.node_job_starts[node_idx]
        starts.remove(start_time)
        self.node_oldest_start[node_idx] = starts[0] if starts else np.inf
        self.node_newest_start[node_idx] = starts[-1] if starts else np.inf
        self.node_job_count[node_idx] -= 1

        # 완료된 작업의 과거 실측 수명 기록 (100% 인과적·Non-Clairvoyant)
        realized_dur = max(0.0, self.now - start_time)
        self.recent_completions.append((job.gpu, realized_dur))
        self._ops_counter += 1

    def _allocate_node_resources(self, node_idx: int, gpu_req: float, cpu_req: float, ram_req: float, start_time: float) -> List[Tuple[int, float]]:
        """단일 노드에 gpu_req/cpu_req/ram_req를 할당하고 슬롯 할당 내역을 반환한다."""
        eps = 1e-9
        slots = self.gpu_slots[node_idx]
        allocations: List[Tuple[int, float]] = []

        if gpu_req < 1.0 - eps:
            feasible_slots = np.flatnonzero(slots + eps >= gpu_req)
            best_slot = int(feasible_slots[np.argmin(slots[feasible_slots])])
            slots[best_slot] = max(0.0, slots[best_slot] - gpu_req)
            allocations.append((best_slot, gpu_req))
        else:
            needed = int(round(gpu_req))
            empty_slots = np.flatnonzero(slots >= 1.0 - eps)
            chosen = empty_slots[:needed]
            for s in chosen:
                slots[s] = 0.0
                allocations.append((int(s), 1.0))

        self.cpu_free[node_idx] = max(0.0, self.cpu_free[node_idx] - cpu_req)
        self.ram_free[node_idx] = max(0.0, self.ram_free[node_idx] - ram_req)

        starts = self.node_job_starts[node_idx]
        if not starts or start_time >= starts[-1]:
            starts.append(start_time)
        else:
            bisect.insort(starts, start_time)
        self.node_oldest_start[node_idx] = starts[0]
        self.node_newest_start[node_idx] = starts[-1]
        self.node_job_count[node_idx] += 1
        return allocations

    def allocate_gang(
        self,
        node_indices: List[int],
        job: Job,
    ) -> List[Tuple[int, List[Tuple[int, float]]]]:
        """멀티노드 Gang 작업을 서로 다른 node_indices에 원자적으로 동시 배치한다."""
        if len(node_indices) != job.num_nodes or len(set(node_indices)) != job.num_nodes:
            raise ValueError(f"Gang 할당 노드 목록이 유효하지 않습니다: req={job.num_nodes}, got={node_indices}")

        gpu_pn = job.gpu / job.num_nodes
        cpu_pn = job.cpu / job.num_nodes
        ram_pn = job.ram / job.num_nodes

        start_tol = 1e-12 * abs(job.arrival_time)
        start_time = self.now if self.now >= job.arrival_time - start_tol else job.arrival_time

        gang_allocs: List[Tuple[int, List[Tuple[int, float]]]] = []
        for node_idx in node_indices:
            ni = int(node_idx)
            node_alloc = self._allocate_node_resources(ni, gpu_pn, cpu_pn, ram_pn, start_time)
            gang_allocs.append((ni, node_alloc))

        alloc_key = id(gang_allocs)
        job_key = (job.job_id, id(job))
        self.active_jobs[alloc_key] = (int(node_indices[0]), start_time, job.gpu)
        cohort_entry: List[object] = [start_time, None, job.gpu]
        self.recent_cohort.append(cohort_entry)
        self._cohort_by_alloc[alloc_key] = cohort_entry
        q_keys = self._active_by_job.get(job_key)
        if q_keys is None:
            self._active_by_job[job_key] = deque([alloc_key])
        else:
            q_keys.append(alloc_key)
        self._ops_counter += 1
        return gang_allocs

    def release_gang(
        self,
        gang_allocs: List[Tuple[int, List[Tuple[int, float]]]],
        job: Job,
    ) -> None:
        """종료된 멀티노드 Gang 작업의 모든 노드 자원을 원자적으로 반환한다."""
        cpu_pn = job.cpu / job.num_nodes
        ram_pn = job.ram / job.num_nodes

        alloc_key = id(gang_allocs)
        resolved_key = alloc_key
        job_key = (job.job_id, id(job))
        entry = self.active_jobs.pop(alloc_key, None)
        q_keys = self._active_by_job.get(job_key)
        if entry is not None:
            if q_keys is not None:
                try:
                    q_keys.remove(alloc_key)
                except ValueError:
                    pass
                if not q_keys:
                    del self._active_by_job[job_key]
        elif q_keys:
            matched_key = q_keys.popleft()
            if not q_keys:
                del self._active_by_job[job_key]
            resolved_key = matched_key
            entry = self.active_jobs.pop(matched_key, None)

        cohort_entry = self._cohort_by_alloc.pop(resolved_key, None)
        if cohort_entry is not None:
            cohort_entry[1] = self.now

        start_time = entry[1] if entry is not None else job.arrival_time

        for node_idx, node_alloc in gang_allocs:
            slots = self.gpu_slots[node_idx]
            for slot_idx, amount in node_alloc:
                slots[slot_idx] = min(1.0, slots[slot_idx] + amount)
            self.cpu_free[node_idx] = min(float(self.node_cpu_cap[node_idx]), self.cpu_free[node_idx] + cpu_pn)
            self.ram_free[node_idx] = min(float(self.node_ram_cap[node_idx]), self.ram_free[node_idx] + ram_pn)
            starts = self.node_job_starts[node_idx]
            starts.remove(start_time)
            self.node_oldest_start[node_idx] = starts[0] if starts else np.inf
            self.node_newest_start[node_idx] = starts[-1] if starts else np.inf
            self.node_job_count[node_idx] -= 1

        realized_dur = max(0.0, self.now - start_time)
        self.recent_completions.append((job.gpu, realized_dur))
        self._ops_counter += 1


class BaseScheduler:
    """모든 스케줄러의 공통 인터페이스."""

    name: str = "Base"

    def reset(self, config: ClusterConfig) -> None:
        """새 시뮬레이션 시작 전 내부 상태를 초기화한다."""
        pass

    def score_nodes(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
    ) -> Optional[np.ndarray]:
        """후보 노드별 점수(낮을수록 우수)를 반환한다. 미구현 시 None 반환."""
        return None

    def select_node(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
    ) -> int:
        """배치 가능한 노드 목록 중 최적 노드 1개의 인덱스를 반환한다."""
        raise NotImplementedError

    def select_nodes_gang(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        num_nodes: int,
        gpu_per_node: float,
        cpu_per_node: float,
        ram_per_node: float,
    ) -> List[int]:
        """멀티노드 Gang 작업에 대해 서로 다른 num_nodes개 노드 인덱스를 반환한다."""
        scores = self.score_nodes(state, feasible_nodes, gpu_per_node, cpu_per_node, ram_per_node)
        if scores is not None:
            order = np.argsort(scores, kind="mergesort")
            return [int(feasible_nodes[i]) for i in order[:num_nodes]]

        chosen: List[int] = []
        remaining = feasible_nodes
        for _ in range(num_nodes):
            picked = int(self.select_node(state, remaining, gpu_per_node, cpu_per_node, ram_per_node))
            chosen.append(picked)
            remaining = remaining[remaining != picked]
        return chosen


def run_simulation(
    jobs: List[Job],
    scheduler: BaseScheduler,
    config: Optional[ClusterConfig] = None,
    workload_name: str = "unnamed",
    seed: int = 0,
    recent_window: int = 200,
    cross_rack_penalty: float = 0.0,
) -> SimulationMetrics:
    """
    주어진 작업 시퀀스에 대해 이벤트 기반 동적 클러스터 시뮬레이션을 실행한다.
    """
    if config is None:
        config = ClusterConfig()

    state = ClusterState(config)
    scheduler.reset(config)

    # 도착 시각 순 정렬 보장 (인과성 유지, run_queue_simulation과 동일 규격)
    ordered_jobs = sorted(jobs, key=lambda x: (x.arrival_time, x.job_id))

    # 종료 이벤트 최소 힙: (departure_time, dep_seq, node_idx(-1이면 Gang), job, allocations)
    departure_heap: List[Tuple[float, int, int, Job, object]] = []

    total_gpu_capacity = state.total_gpu_capacity
    prev_time = ordered_jobs[0].arrival_time if ordered_jobs else 0.0

    active_node_time_acc = 0.0
    used_gpu_time_acc = 0.0

    rejected_jobs = 0
    total_gpu_hours = 0.0
    rejected_gpu_hours = 0.0
    multi_total_gpu_hours = 0.0
    multi_rejected_gpu_hours = 0.0
    multi_placed = 0
    multi_single_rack = 0
    multi_racks_sum = 0
    stranded_ratios: List[float] = []

    decision_time_acc = 0.0
    decision_count = 0
    dep_seq = 0

    def advance_time(target_time: float) -> None:
        nonlocal prev_time, active_node_time_acc, used_gpu_time_acc
        dt = target_time - prev_time
        if dt > 0.0:
            active_nodes = int(np.sum(state.node_job_count > 0))
            used_gpus = total_gpu_capacity - float(np.sum(state.gpu_slots))
            active_node_time_acc += active_nodes * dt
            used_gpu_time_acc += used_gpus * dt
            prev_time = target_time

    for job in ordered_jobs:
        # 1) 현재 작업 도착 시각 이전에 끝나는 모든 작업의 자원 해제
        while departure_heap and departure_heap[0][0] <= job.arrival_time:
            dep_time, _, node_idx, dep_job, allocs = heapq.heappop(departure_heap)
            advance_time(dep_time)
            state.now = dep_time
            if node_idx < 0:
                state.release_gang(allocs, dep_job)  # type: ignore[arg-type]
            else:
                state.release(node_idx, dep_job, allocs)  # type: ignore[arg-type]

        # 2) 현재 작업 도착 시각으로 시간 전진
        advance_time(job.arrival_time)
        state.now = job.arrival_time

        # 최근 요청 기록 (온라인 적응형 알고리즘이 참고 가능, 수명 제외)
        n_req_nodes = max(1, int(job.num_nodes))
        gpu_pn = job.gpu / n_req_nodes
        cpu_pn = job.cpu / n_req_nodes
        ram_pn = job.ram / n_req_nodes

        state.recent_requests.append((gpu_pn, cpu_pn, ram_pn))
        if len(state.recent_requests) > recent_window:
            del state.recent_requests[0]
        state.recent_specs.append(job.gpu_spec)
        state.recent_node_counts.append(n_req_nodes)

        job_gpu_hours = job.gpu * job.duration
        total_gpu_hours += job_gpu_hours
        if n_req_nodes > 1:
            multi_total_gpu_hours += job_gpu_hours

        # 3) 배치 가능한 노드 필터링
        feasible = state.get_feasible_nodes(gpu_pn, cpu_pn, ram_pn, job.gpu_spec)
        if len(feasible) < n_req_nodes:
            rejected_jobs += 1
            rejected_gpu_hours += job_gpu_hours
            if n_req_nodes > 1:
                multi_rejected_gpu_hours += job_gpu_hours
            free_gpus = float(np.sum(state.gpu_slots))
            stranded_ratios.append(free_gpus / total_gpu_capacity)
            continue

        # 4) 스케줄러 의사결정 및 지연시간 측정
        if n_req_nodes == 1:
            if len(feasible) == 1:
                chosen_node = int(feasible[0])
            else:
                t0 = time.perf_counter()
                state.current_gpu_spec = job.gpu_spec
                try:
                    chosen_node = int(scheduler.select_node(state, feasible, gpu_pn, cpu_pn, ram_pn))
                finally:
                    state.current_gpu_spec = None
                decision_time_acc += (time.perf_counter() - t0)
                decision_count += 1

            allocs = state.allocate(chosen_node, job)
            dep_seq += 1
            heapq.heappush(
                departure_heap,
                (job.arrival_time + job.duration, dep_seq, chosen_node, job, allocs),
            )
        else:
            if len(feasible) == n_req_nodes:
                chosen_nodes = [int(x) for x in feasible]
            else:
                t0 = time.perf_counter()
                state.current_gpu_spec = job.gpu_spec
                try:
                    chosen_nodes = scheduler.select_nodes_gang(
                        state, feasible, n_req_nodes, gpu_pn, cpu_pn, ram_pn
                    )
                finally:
                    state.current_gpu_spec = None
                decision_time_acc += (time.perf_counter() - t0)
                decision_count += 1

            racks_used = len({int(state.node_rack_ids[ni]) for ni in chosen_nodes})
            multi_placed += 1
            if racks_used == 1:
                multi_single_rack += 1
            multi_racks_sum += racks_used

            eff_dur = job.duration
            if cross_rack_penalty > 0.0 and racks_used > 1:
                eff_dur = job.duration * (1.0 + min(0.25, cross_rack_penalty * (racks_used - 1)))

            gang_allocs = state.allocate_gang(chosen_nodes, job)
            dep_seq += 1
            heapq.heappush(
                departure_heap,
                (job.arrival_time + eff_dur, dep_seq, -1, job, gang_allocs),
            )

    # 5) 남아 있는 모든 실행 중 작업 종료까지 시간 정산
    while departure_heap:
        dep_time, _, node_idx, dep_job, allocs = heapq.heappop(departure_heap)
        advance_time(dep_time)
        state.now = dep_time
        if node_idx < 0:
            state.release_gang(allocs, dep_job)  # type: ignore[arg-type]
        else:
            state.release(node_idx, dep_job, allocs)  # type: ignore[arg-type]

    total_span = max(prev_time - (jobs[0].arrival_time if jobs else 0.0), 1e-9)
    avg_util = (used_gpu_time_acc / (total_gpu_capacity * total_span)) * 100.0
    avg_stranded = (float(np.mean(stranded_ratios)) * 100.0) if stranded_ratios else 0.0
    avg_dec_us = (decision_time_acc / max(decision_count, 1)) * 1e6
    multi_blk_rate = (multi_rejected_gpu_hours / max(multi_total_gpu_hours, 1e-9)) * 100.0 if multi_total_gpu_hours > 0.0 else 0.0
    locality_rate = (multi_single_rack / multi_placed) * 100.0 if multi_placed > 0 else 100.0
    avg_racks = (multi_racks_sum / multi_placed) if multi_placed > 0 else 1.0

    return SimulationMetrics(
        scheduler_name=scheduler.name,
        workload_name=workload_name,
        seed=seed,
        total_jobs=len(jobs),
        rejected_jobs=rejected_jobs,
        job_blocking_rate=(rejected_jobs / max(len(jobs), 1)) * 100.0,
        gpu_hour_blocking_rate=(rejected_gpu_hours / max(total_gpu_hours, 1e-9)) * 100.0,
        avg_stranded_gpu_ratio=avg_stranded,
        active_node_hours=active_node_time_acc,
        avg_gpu_utilization=avg_util,
        avg_decision_us=avg_dec_us,
        multinode_blocking_rate=multi_blk_rate,
        single_rack_locality_rate=locality_rate,
        avg_racks_spanned=avg_racks,
    )


def run_queue_simulation(
    jobs: List[Job],
    scheduler: BaseScheduler,
    config: Optional[ClusterConfig] = None,
    workload_name: str = "unnamed",
    seed: int = 0,
    queue_policy: str = "fifo",
    backfill_window: int = 16,
    max_hol_bypasses: int = 8,
    recent_window: int = 200,
    cross_rack_penalty: float = 0.0,
) -> QueueSimulationMetrics:
    """
    현실형 대기 큐(Pending Queue) 기반 동적 클러스터 시뮬레이션을 실행하여
    작업 완료 시간(JCT), 꼬리 지연(P95/P99 JCT), 대기 시간, Makespan을 측정한다.

    지원 큐 정책(queue_policy):
      - "fifo"     : 엄격한 FIFO(선입선출). 헤드 작업이 배치 불가하면 자원 반환까지 대기 (기아 0%).
      - "backfill" : 기아 방지 유계 바이패스 백필(Starvation-Free Bounded-Bypass Backfill).
                     헤드 작업이 대기할 때 뒤쪽 최대 backfill_window개 작업 중 배치 가능한 작업을
                     최대 max_hol_bypasses회까지 선행 배치하며, 한도 도달 시 헤드 예약 보호.
    """
    if queue_policy not in ("fifo", "backfill"):
        raise ValueError(f"지원하지 않는 queue_policy입니다: {queue_policy!r} ('fifo' 또는 'backfill'만 허용)")
    if backfill_window < 0:
        raise ValueError(f"backfill_window는 0 이상이어야 합니다: {backfill_window}")
    if max_hol_bypasses < 0:
        raise ValueError(f"max_hol_bypasses는 0 이상이어야 합니다: {max_hol_bypasses}")

    if config is None:
        config = ClusterConfig()

    if config.num_nodes <= 0:
        raise ValueError(f"유효하지 않은 ClusterConfig 설정입니다: {config}")

    if config.node_gpus is not None:
        n = config.num_nodes
        if (
            len(config.node_gpus) != n
            or (config.node_cpus is not None and len(config.node_cpus) != n)
            or (config.node_rams is not None and len(config.node_rams) != n)
            or (config.node_models is not None and len(config.node_models) != n)
        ):
            raise ValueError(f"이기종 ClusterConfig 배열 길이가 num_nodes와 불일치합니다: {config}")
        for idx in range(n):
            ng = config.node_gpus[idx]
            nc = config.cpu_per_node if config.node_cpus is None else config.node_cpus[idx]
            nr = config.ram_per_node if config.node_rams is None else config.node_rams[idx]
            if (
                ng <= 0
                or not math.isfinite(nc)
                or nc <= 0.0
                or not math.isfinite(nr)
                or nr <= 0.0
            ):
                raise ValueError(f"유효하지 않은 이기종 노드 용량이 있습니다(node={idx}): gpu={ng}, cpu={nc}, ram={nr}")
    elif (
        config.gpus_per_node <= 0
        or not math.isfinite(config.cpu_per_node)
        or config.cpu_per_node <= 0.0
        or not math.isfinite(config.ram_per_node)
        or config.ram_per_node <= 0.0
    ):
        raise ValueError(f"유효하지 않은 ClusterConfig 설정입니다: {config}")

    if not jobs:
        return QueueSimulationMetrics(
            scheduler_name=scheduler.name,
            workload_name=workload_name,
            seed=seed,
            queue_policy=queue_policy,
            total_jobs=0,
            completed_jobs=0,
            mean_jct=0.0,
            gpu_weighted_jct=0.0,
            p95_jct=0.0,
            p99_jct=0.0,
            mean_wait_time=0.0,
            p95_wait_time=0.0,
            mean_slowdown=1.0,
            makespan=0.0,
            max_queue_len=0,
            avg_queue_len=0.0,
            avg_stranded_gpu_on_wait=0.0,
            active_node_hours=0.0,
            avg_gpu_utilization=0.0,
            avg_decision_us=0.0,
        )

    # 데드락·무한루프 사전 차단: NaN/Inf, 음수 자원/수명, 비정수 멀티 GPU, 단일 빈 노드 용량 초과 검증
    eps = 1e-9
    n_nodes = config.num_nodes
    node_g_caps = (
        tuple(config.node_gpus)
        if config.node_gpus is not None
        else (config.gpus_per_node,) * n_nodes
    )
    node_c_caps = (
        tuple(config.node_cpus)
        if config.node_cpus is not None
        else (config.cpu_per_node,) * n_nodes
    )
    node_r_caps = (
        tuple(config.node_rams)
        if config.node_rams is not None
        else (config.ram_per_node,) * n_nodes
    )
    node_m_caps = config.node_models
    max_g_cap = max(node_g_caps)

    for j in jobs:
        if not (
            math.isfinite(j.arrival_time)
            and math.isfinite(j.duration)
            and math.isfinite(j.gpu)
            and math.isfinite(j.cpu)
            and math.isfinite(j.ram)
        ):
            raise ValueError(f"NaN 또는 Inf 값이 포함된 작업은 허용되지 않습니다: {j}")
        if j.gpu < -eps or j.cpu < -eps or j.ram < -eps or j.duration < -eps:
            raise ValueError(f"음수 자원 또는 음수 수명 작업은 허용되지 않습니다: {j}")
        if j.num_nodes < 1 or j.num_nodes > n_nodes:
            raise ValueError(f"유효하지 않은 num_nodes 요청입니다: {j}")
        g_pn = j.gpu / j.num_nodes
        c_pn = j.cpu / j.num_nodes
        r_pn = j.ram / j.num_nodes
        if g_pn >= 1.0 - eps and abs(g_pn - round(g_pn)) > eps:
            raise ValueError(f"1.0 이상의 노드당 GPU 요청은 정수여야 합니다: {j}")

        req_g_int = int(round(g_pn)) if g_pn >= 1.0 - eps else 0
        allowed_models = (
            {tok.strip() for tok in j.gpu_spec.split("|") if tok.strip()}
            if (j.gpu_spec and node_m_caps is not None)
            else None
        )
        hostable_cnt = 0
        for i in range(n_nodes):
            if allowed_models is not None and node_m_caps[i] not in allowed_models:
                continue
            if c_pn > node_c_caps[i] + eps or r_pn > node_r_caps[i] + eps:
                continue
            if req_g_int > 0:
                if req_g_int <= node_g_caps[i]:
                    hostable_cnt += 1
            else:
                if g_pn <= min(1.0, float(node_g_caps[i])) + eps and g_pn <= max_g_cap + eps:
                    hostable_cnt += 1
            if hostable_cnt >= j.num_nodes:
                break
        if hostable_cnt < j.num_nodes:
            raise ValueError(f"노드 용량, num_nodes 또는 gpu_spec 제약을 충족할 수 없는 작업이 있습니다: {j}")

    # 도착 시각 순 정렬 보장 (인과성 유지)
    ordered_jobs = sorted(jobs, key=lambda x: (x.arrival_time, x.job_id))
    n_jobs = len(ordered_jobs)

    state = ClusterState(config)
    scheduler.reset(config)

    departure_heap: List[Tuple[float, int, int, Job, object]] = []
    # (고유 티켓 번호, Job) 튜플로 대기 큐를 관리하여 중복 job_id/객체에도 HoL 기아 방지가 안전하게 동작
    pending: Deque[Tuple[int, Job]] = deque()

    total_gpu_capacity = state.total_gpu_capacity
    first_arrival = ordered_jobs[0].arrival_time
    prev_time = first_arrival
    state.now = first_arrival

    active_node_time_acc = 0.0
    used_gpu_time_acc = 0.0
    queue_len_time_acc = 0.0
    max_queue_len = 0

    jcts: List[float] = []
    waits: List[float] = []
    slowdowns: List[float] = []
    job_gpus: List[float] = []
    stranded_on_wait: List[float] = []

    multi_placed = 0
    multi_single_rack = 0
    multi_racks_sum = 0

    decision_time_acc = 0.0
    decision_count = 0
    dep_seq = 0

    hol_ticket: Optional[int] = None
    hol_bypasses = 0

    def advance_time(target_time: float) -> None:
        nonlocal prev_time, active_node_time_acc, used_gpu_time_acc, queue_len_time_acc
        dt = target_time - prev_time
        if dt > 0.0:
            active_nodes = int(np.sum(state.node_job_count > 0))
            used_gpus = total_gpu_capacity - float(np.sum(state.gpu_slots))
            active_node_time_acc += active_nodes * dt
            used_gpu_time_acc += used_gpus * dt
            queue_len_time_acc += len(pending) * dt
            prev_time = target_time

    def dispatch_job(job: Job, feasible: np.ndarray) -> None:
        nonlocal decision_time_acc, decision_count, dep_seq, multi_placed, multi_single_rack, multi_racks_sum
        n_req_nodes = max(1, int(job.num_nodes))
        dur = max(0.0, job.duration)
        wait_t = max(0.0, state.now - job.arrival_time)

        if n_req_nodes == 1:
            if len(feasible) == 1:
                chosen_node = int(feasible[0])
            else:
                t0 = time.perf_counter()
                state.current_gpu_spec = job.gpu_spec
                try:
                    chosen_node = int(scheduler.select_node(state, feasible, job.gpu, job.cpu, job.ram))
                finally:
                    state.current_gpu_spec = None
                decision_time_acc += (time.perf_counter() - t0)
                decision_count += 1

            allocs = state.allocate(chosen_node, job)
            eff_dur = dur
            dep_seq += 1
            heapq.heappush(
                departure_heap,
                (state.now + eff_dur, dep_seq, chosen_node, job, allocs),
            )
        else:
            g_pn = job.gpu / n_req_nodes
            c_pn = job.cpu / n_req_nodes
            r_pn = job.ram / n_req_nodes
            if len(feasible) == n_req_nodes:
                chosen_nodes = [int(x) for x in feasible]
            else:
                t0 = time.perf_counter()
                state.current_gpu_spec = job.gpu_spec
                try:
                    chosen_nodes = scheduler.select_nodes_gang(
                        state, feasible, n_req_nodes, g_pn, c_pn, r_pn
                    )
                finally:
                    state.current_gpu_spec = None
                decision_time_acc += (time.perf_counter() - t0)
                decision_count += 1

            racks_used = len({int(state.node_rack_ids[ni]) for ni in chosen_nodes})
            multi_placed += 1
            if racks_used == 1:
                multi_single_rack += 1
            multi_racks_sum += racks_used

            eff_dur = dur
            if cross_rack_penalty > 0.0 and racks_used > 1:
                eff_dur = dur * (1.0 + min(0.25, cross_rack_penalty * (racks_used - 1)))

            gang_allocs = state.allocate_gang(chosen_nodes, job)
            dep_seq += 1
            heapq.heappush(
                departure_heap,
                (state.now + eff_dur, dep_seq, -1, job, gang_allocs),
            )

        jct_t = wait_t + eff_dur
        waits.append(wait_t)
        jcts.append(jct_t)
        # 감속비(Slowdown) = 1 + wait/dur >= 1.0 보장 및 양수 수명에서 시간 단위 불변성 유지
        if dur <= 0.0:
            slowdowns.append(1.0 if wait_t <= 0.0 else (1.0 + wait_t / 1e-6))
        else:
            slowdowns.append(jct_t / dur)
        job_gpus.append(job.gpu)

    def try_dispatch_pending() -> None:
        nonlocal hol_ticket, hol_bypasses
        while pending:
            head_ticket, head = pending[0]
            if hol_ticket != head_ticket:
                hol_ticket = head_ticket
                hol_bypasses = 0

            h_nodes = max(1, int(head.num_nodes))
            feasible = state.get_feasible_nodes(head.gpu / h_nodes, head.cpu / h_nodes, head.ram / h_nodes, head.gpu_spec)
            if len(feasible) >= h_nodes:
                pending.popleft()
                dispatch_job(head, feasible)
                hol_ticket = None
                hol_bypasses = 0
                continue

            if queue_policy == "fifo" or hol_bypasses >= max_hol_bypasses:
                break

            # Starvation-Free Bounded-Bypass Backfill: 헤드 뒤쪽 유한 윈도우 탐색
            dispatched_bf = False
            scan_limit = min(len(pending), 1 + backfill_window)
            for q_idx in range(1, scan_limit):
                _, cand = pending[q_idx]
                c_nodes = max(1, int(cand.num_nodes))
                cand_feas = state.get_feasible_nodes(cand.gpu / c_nodes, cand.cpu / c_nodes, cand.ram / c_nodes, cand.gpu_spec)
                if len(cand_feas) >= c_nodes:
                    del pending[q_idx]
                    dispatch_job(cand, cand_feas)
                    hol_bypasses += 1
                    dispatched_bf = True
                    break
            if not dispatched_bf:
                break

        # 디스패치 시도 후에도 대기 큐가 남아 있으면 해당 시점의 클러스터 유휴 GPU 비율 기록
        if pending:
            free_gpus = float(np.sum(state.gpu_slots))
            stranded_on_wait.append(free_gpus / total_gpu_capacity)

    arr_idx = 0
    while arr_idx < n_jobs or departure_heap:
        next_arr = ordered_jobs[arr_idx].arrival_time if arr_idx < n_jobs else float("inf")
        next_dep = departure_heap[0][0] if departure_heap else float("inf")

        # 동일 시각(부동소수점 반올림 오차 1e-12 범위 포함)에서는 자원 해제(departure)를
        # 도착(arrival)보다 먼저 처리하여 즉시 재사용과 시간 단위 불변성을 보장
        dep_arr_tol = 1e-12 * abs(next_arr) if math.isfinite(next_arr) else 0.0
        if next_dep <= next_arr + dep_arr_tol:
            t_evt = max(state.now, min(next_dep, next_arr))
            evt_tol = 1e-12 * abs(t_evt)
            advance_time(t_evt)
            state.now = t_evt
            while departure_heap and departure_heap[0][0] <= t_evt + evt_tol:
                _, _, node_idx, dep_job, allocs = heapq.heappop(departure_heap)
                if node_idx < 0:
                    state.release_gang(allocs, dep_job)  # type: ignore[arg-type]
                else:
                    state.release(node_idx, dep_job, allocs)  # type: ignore[arg-type]
            try_dispatch_pending()
        else:
            t_evt = next_arr
            evt_tol = 1e-12 * abs(t_evt)
            advance_time(t_evt)
            state.now = t_evt
            while arr_idx < n_jobs and ordered_jobs[arr_idx].arrival_time <= t_evt + evt_tol:
                j = ordered_jobs[arr_idx]
                ticket = arr_idx
                arr_idx += 1
                n_rn = max(1, int(j.num_nodes))
                state.recent_requests.append((j.gpu / n_rn, j.cpu / n_rn, j.ram / n_rn))
                if len(state.recent_requests) > recent_window:
                    del state.recent_requests[0]
                state.recent_specs.append(j.gpu_spec)
                state.recent_node_counts.append(n_rn)
                pending.append((ticket, j))
            try_dispatch_pending()
            if len(pending) > max_queue_len:
                max_queue_len = len(pending)

    if pending:
        raise RuntimeError(f"시뮬레이션 종료 시 미배치 대기 작업이 {len(pending)}건 남아 있습니다 (데드락 오류).")

    makespan = max(prev_time - first_arrival, 0.0)
    avg_util = (used_gpu_time_acc / (total_gpu_capacity * makespan)) * 100.0 if makespan > 0.0 else 0.0
    avg_qlen = (queue_len_time_acc / makespan) if makespan > 0.0 else 0.0
    avg_stranded_wait = (float(np.mean(stranded_on_wait)) * 100.0) if stranded_on_wait else 0.0
    avg_dec_us = (decision_time_acc / max(decision_count, 1)) * 1e6

    jct_arr = np.asarray(jcts, dtype=np.float64)
    wait_arr = np.asarray(waits, dtype=np.float64)
    slow_arr = np.asarray(slowdowns, dtype=np.float64)
    gpu_arr = np.asarray(job_gpus, dtype=np.float64)
    gpu_sum = float(np.sum(gpu_arr))
    gpu_w_jct = float(np.dot(gpu_arr, jct_arr) / max(gpu_sum, 1e-9))

    locality_rate = (multi_single_rack / multi_placed) * 100.0 if multi_placed > 0 else 100.0
    avg_racks = (multi_racks_sum / multi_placed) if multi_placed > 0 else 1.0

    return QueueSimulationMetrics(
        scheduler_name=scheduler.name,
        workload_name=workload_name,
        seed=seed,
        queue_policy=queue_policy,
        total_jobs=n_jobs,
        completed_jobs=len(jcts),
        mean_jct=float(np.mean(jct_arr)),
        gpu_weighted_jct=gpu_w_jct,
        p95_jct=float(np.percentile(jct_arr, 95)),
        p99_jct=float(np.percentile(jct_arr, 99)),
        mean_wait_time=float(np.mean(wait_arr)),
        p95_wait_time=float(np.percentile(wait_arr, 95)),
        mean_slowdown=float(np.mean(slow_arr)),
        makespan=float(makespan),
        max_queue_len=max_queue_len,
        avg_queue_len=float(avg_qlen),
        avg_stranded_gpu_on_wait=avg_stranded_wait,
        active_node_hours=active_node_time_acc,
        avg_gpu_utilization=avg_util,
        avg_decision_us=avg_dec_us,
        single_rack_locality_rate=locality_rate,
        avg_racks_spanned=avg_racks,
    )
