"""
학계·산업계 SOTA 클러스터 스케줄링 베이스라인 6종 구현

1. FirstFit      : 배치 가능한 첫 번째 노드 선택
2. BestFit       : 배치 후 총 잔여 자원(GPU 중심)이 가장 적은 노드 선택 (K8s MostAllocated 계열)
3. DotProduct    : Microsoft VMM 기하 휴리스틱 (정규화 요청 벡터와 잔여 벡터의 내적 최대화)
4. L2Norm        : 배치 후 잔여 자원 벡터의 L2 노름 최소화 (다차원 균형 패킹)
5. FGD           : USENIX ATC '23 (Alibaba) Fragmentation Gradient Descent — 정적 기준 분포 사용
6. FGD-Adaptive  : 최근 슬라이딩 윈도우 분포를 실시간 반영한 강화판 FGD (초강력 비교 기준선)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from simulator import BaseScheduler, ClusterState
from workloads import STANDARD_PROFILES


class FirstFitScheduler(BaseScheduler):
    """배치 가능한 가장 앞 번호 노드를 선택한다."""
    name = "FirstFit"

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        return int(feasible_nodes[0])

    def select_nodes_gang(self, state, feasible_nodes, num_nodes, gpu_per_node, cpu_per_node, ram_per_node) -> List[int]:
        return [int(x) for x in feasible_nodes[:num_nodes]]


class BestFitScheduler(BaseScheduler):
    """GPU 잔여량(동률 시 CPU/RAM 잔여량)이 가장 적은 노드에 빈틈없이 채워 넣는다."""
    name = "BestFit"

    def score_nodes(self, state, feasible_nodes, gpu, cpu, ram) -> np.ndarray:
        g_cap = state.node_gpu_cap[feasible_nodes]
        c_cap = state.node_cpu_cap[feasible_nodes]
        r_cap = state.node_ram_cap[feasible_nodes]
        free_gpu = np.sum(state.gpu_slots[feasible_nodes], axis=1) / g_cap
        free_cpu = state.cpu_free[feasible_nodes] / c_cap
        free_ram = state.ram_free[feasible_nodes] / r_cap
        # GPU 팩킹 최우선 + 연속 자원 보조
        return free_gpu + 0.05 * free_cpu + 0.05 * free_ram

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        return int(feasible_nodes[np.argmin(self.score_nodes(state, feasible_nodes, gpu, cpu, ram))])


class DotProductScheduler(BaseScheduler):
    """
    Microsoft VMM 스타일 DotProduct 기하 휴리스틱.
    요청 벡터와 노드 잔여 자원 벡터의 내적이 최대인(방향이 가장 일치하는) 노드를 선택한다.
    """
    name = "DotProduct"

    def score_nodes(self, state, feasible_nodes, gpu, cpu, ram) -> np.ndarray:
        g_cap = state.node_gpu_cap[feasible_nodes]
        c_cap = state.node_cpu_cap[feasible_nodes]
        r_cap = state.node_ram_cap[feasible_nodes]
        req_g = gpu / g_cap
        req_c = cpu / c_cap
        req_r = ram / r_cap

        rem_g = np.sum(state.gpu_slots[feasible_nodes], axis=1) / g_cap
        rem_c = state.cpu_free[feasible_nodes] / c_cap
        rem_r = state.ram_free[feasible_nodes] / r_cap

        dot = req_g * rem_g + req_c * rem_c + req_r * rem_r
        norm_rem = np.sqrt(rem_g * rem_g + rem_c * rem_c + rem_r * rem_r) + 1e-9
        cos_sim = dot / norm_rem
        score = cos_sim - 0.15 * rem_g
        return -score

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        return int(feasible_nodes[np.argmin(self.score_nodes(state, feasible_nodes, gpu, cpu, ram))])


class L2NormScheduler(BaseScheduler):
    """배치 이후 남는 정규화 잔여 벡터의 L2 제곱합(불균형·여백)을 최소화하는 노드를 선택한다."""
    name = "L2Norm"

    def score_nodes(self, state, feasible_nodes, gpu, cpu, ram) -> np.ndarray:
        g_cap = state.node_gpu_cap[feasible_nodes]
        c_cap = state.node_cpu_cap[feasible_nodes]
        r_cap = state.node_ram_cap[feasible_nodes]
        rem_g = (np.sum(state.gpu_slots[feasible_nodes], axis=1) - gpu) / g_cap
        rem_c = (state.cpu_free[feasible_nodes] - cpu) / c_cap
        rem_r = (state.ram_free[feasible_nodes] - ram) / r_cap
        return rem_g * rem_g + 0.5 * rem_c * rem_c + 0.5 * rem_r * rem_r

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        return int(feasible_nodes[np.argmin(self.score_nodes(state, feasible_nodes, gpu, cpu, ram))])


def _compute_fgd_fragmentation(
    gpu_slots: np.ndarray,
    cpu_free: np.ndarray,
    ram_free: np.ndarray,
    typical_reqs: np.ndarray,
    typical_probs: np.ndarray,
) -> np.ndarray:
    """
    Alibaba FGD (USENIX ATC '23, Weng et al.) 원논문 노드 단편화 함수 F_n(M)의 정확한 벡터화 구현.

    각 대표 작업 m = (D^G_m, D^C_m, D^R_m)에 대해:
      - Case 1 (수용 불가): CPU/RAM 부족 또는 GPU 조건 불충족 시 노드의 모든 잔여 GPU가 단편화
        F_n(m) = sum_g R_{n,g}^G
      - Case 2 (수용 가능): 노드에 배치 가능하더라도 개별 GPU의 잔여량이 min(D^G_m, 1)보다 작으면
        해당 부분 GPU 자투리가 단편화로 집계됨
        F_n(m) = sum_g R_{n,g}^G * I(R_{n,g}^G < min(D^G_m, 1))
    """
    eps = 1e-9
    k = gpu_slots.shape[0]
    total_free_gpu = np.sum(gpu_slots, axis=1)             # (K,)
    empty_gpu_cnt = np.sum(gpu_slots >= 1.0 - eps, axis=1) # (K,)
    max_slot_free = np.max(gpu_slots, axis=1)              # (K,)

    frag = np.zeros(k, dtype=np.float64)
    for j in range(len(typical_probs)):
        p = typical_probs[j]
        if p <= 0.0:
            continue
        req_g, req_c, req_r = typical_reqs[j]
        res_ok = (cpu_free + eps >= req_c) & (ram_free + eps >= req_r)
        if req_g < 1.0 - eps:
            can_fit = res_ok & (max_slot_free + eps >= req_g)
            thresh = req_g
        else:
            can_fit = res_ok & (empty_gpu_cnt >= int(round(req_g)))
            thresh = 1.0

        # Case 2: 수용 가능 시 개별 슬롯 잔여량이 thresh 미만인 자투리 GPU 합계
        partial_frag = np.sum(np.where(gpu_slots < thresh - eps, gpu_slots, 0.0), axis=1)
        # Case 1 vs Case 2 결합
        frag += p * np.where(can_fit, partial_frag, total_free_gpu)
    return frag


class FGDScheduler(BaseScheduler):
    """
    Alibaba FGD (Fragmentation Gradient Descent, USENIX ATC '23).
    배치 전후의 노드 단편화 증가량 ΔF = F(after) - F(before)를 최소화하는 노드를 선택한다.
    """
    name = "FGD_Static"

    def __init__(
        self,
        adaptive: bool = False,
        custom_reqs: Optional[np.ndarray] = None,
        custom_probs: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ):
        self.adaptive = adaptive
        if name is not None:
            self.name = name
        elif adaptive:
            self.name = "FGD_Adaptive"
        # 정적 기준 분포 (사전 제공 분포 우선, 없으면 표준 프로파일)
        self.default_reqs = (
            np.asarray(custom_reqs, dtype=np.float64)
            if custom_reqs is not None
            else np.array(STANDARD_PROFILES, dtype=np.float64)
        )
        self.default_probs = (
            np.asarray(custom_probs, dtype=np.float64)
            if custom_probs is not None
            else np.array([0.32, 0.24, 0.20, 0.10, 0.08, 0.06], dtype=np.float64)
        )

    def _get_distribution(self, state: ClusterState) -> Tuple[np.ndarray, np.ndarray]:
        if not self.adaptive or len(state.recent_requests) < 20:
            return self.default_reqs, self.default_probs

        # 최근 슬라이딩 윈도우에서 요청 버킷별 빈도 집계 (실제 온라인 분포 추정)
        buckets = {}
        for g, c, r in state.recent_requests:
            # GPU 크기 기준 + CPU/RAM 평균 집계
            key = round(g, 2)
            if key not in buckets:
                buckets[key] = [0, 0.0, 0.0]
            buckets[key][0] += 1
            buckets[key][1] += c
            buckets[key][2] += r

        total = len(state.recent_requests)
        reqs = []
        probs = []
        for g_key, (cnt, c_sum, r_sum) in buckets.items():
            reqs.append((g_key, c_sum / cnt, r_sum / cnt))
            probs.append(cnt / total)
        return np.array(reqs, dtype=np.float64), np.array(probs, dtype=np.float64)

    def score_nodes(self, state, feasible_nodes, gpu, cpu, ram) -> np.ndarray:
        eps = 1e-9
        reqs, probs = self._get_distribution(state)

        slots_before = state.gpu_slots[feasible_nodes]
        cpu_before = state.cpu_free[feasible_nodes]
        ram_before = state.ram_free[feasible_nodes]

        frag_before = _compute_fgd_fragmentation(slots_before, cpu_before, ram_before, reqs, probs)

        # 가상 배치(What-if simulation) 상태 계산
        slots_after = slots_before.copy()
        if gpu < 1.0 - eps:
            # 각 노드에서 들어가는 가장 타이트한 슬롯에 가상 배치
            for i in range(len(feasible_nodes)):
                s = slots_after[i]
                feas = np.flatnonzero(s + eps >= gpu)
                best_s = feas[np.argmin(s[feas])]
                s[best_s] = max(0.0, s[best_s] - gpu)
        else:
            needed = int(round(gpu))
            for i in range(len(feasible_nodes)):
                s = slots_after[i]
                empty = np.flatnonzero(s >= 1.0 - eps)
                s[empty[:needed]] = 0.0

        cpu_after = np.maximum(0.0, cpu_before - cpu)
        ram_after = np.maximum(0.0, ram_before - ram)

        frag_after = _compute_fgd_fragmentation(slots_after, cpu_after, ram_after, reqs, probs)
        delta_frag = frag_after - frag_before

        # 동률 시 이미 사용 중인(잔여 GPU가 적은) 노드를 미세하게 선호
        tie_break = 1e-4 * np.sum(slots_after, axis=1)
        return delta_frag + tie_break

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        return int(feasible_nodes[np.argmin(self.score_nodes(state, feasible_nodes, gpu, cpu, ram))])


def select_rack_aware_gang_nodes(
    state: ClusterState,
    feasible_nodes: np.ndarray,
    node_scores: np.ndarray,
    num_nodes: int,
) -> List[int]:
    """
    노드별 점수(낮을수록 우수)를 기반으로 랙 토폴로지 집적(Single-Rack Locality)을
    최대화하면서 멀티노드 Gang 배치를 수행하는 공용 헬퍼.

    1) 가용 노드가 >= num_nodes인 랙 중 가용 노드 수가 가장 적은(Tightest-Fit) 랙
       (동률 시 상위 num_nodes개 노드 점수 합 최소) 내부에서 최상위 num_nodes개 선택
    2) 단일 랙 수용 불가 시 가용 노드가 많은 랙 순으로 묶어 스팬 랙 수 최소화
    """
    rack_ids = state.node_rack_ids[feasible_nodes]
    rack_groups: dict[int, List[Tuple[float, int]]] = {}
    for pos, r_id in enumerate(rack_ids):
        r = int(r_id)
        rack_groups.setdefault(r, []).append((float(node_scores[pos]), int(feasible_nodes[pos])))

    for r in rack_groups:
        rack_groups[r].sort(key=lambda x: (x[0], x[1]))

    # 1) 단일 랙 집적 가능 랙 탐색 (Tightest-Fit Rack)
    single_candidates = [
        (len(items), sum(x[0] for x in items[:num_nodes]), r, items)
        for r, items in rack_groups.items()
        if len(items) >= num_nodes
    ]
    if single_candidates:
        single_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        chosen_items = single_candidates[0][3][:num_nodes]
        return [ni for _, ni in chosen_items]

    # 2) 다중 랙 스팬 최소화 (가용 노드가 많은 랙 우선, 동률 시 평균 점수 낮은 순)
    multi_racks = sorted(
        rack_groups.items(),
        key=lambda kv: (-len(kv[1]), sum(x[0] for x in kv[1]) / len(kv[1]), kv[0]),
    )
    chosen: List[int] = []
    for _, items in multi_racks:
        need = num_nodes - len(chosen)
        if need <= 0:
            break
        chosen.extend(ni for _, ni in items[:need])
    return chosen


class RackBestFitScheduler(BaseScheduler):
    """
    현업 멀티노드 토폴로지 SOTA 베이스라인 (Slurm/Volcano Rack-Aware BestFit).
    - 단일 노드 배치: BestFit + 완전히 빈 랙 개봉 억제
    - 멀티노드 배치: 단일 랙 Tightest-Fit 집적 및 최소 랙 스팬
    """
    name = "RackBestFit"

    def _base_bestfit_scores(self, state: ClusterState, feasible_nodes: np.ndarray) -> np.ndarray:
        g_cap = state.node_gpu_cap[feasible_nodes]
        c_cap = state.node_cpu_cap[feasible_nodes]
        r_cap = state.node_ram_cap[feasible_nodes]
        free_gpu = np.sum(state.gpu_slots[feasible_nodes], axis=1) / g_cap
        free_cpu = state.cpu_free[feasible_nodes] / c_cap
        free_ram = state.ram_free[feasible_nodes] / r_cap
        return free_gpu + 0.05 * free_cpu + 0.05 * free_ram

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        scores = self._base_bestfit_scores(state, feasible_nodes)
        if state.num_racks > 1:
            rack_busy = np.bincount(
                state.node_rack_ids,
                weights=(state.node_job_count > 0).astype(np.float64),
                minlength=state.num_racks,
            )
            is_empty_rack = (rack_busy[state.node_rack_ids[feasible_nodes]] == 0).astype(np.float64)
            scores = scores + 0.05 * is_empty_rack
        return int(feasible_nodes[np.argmin(scores)])

    def select_nodes_gang(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        num_nodes: int,
        gpu_per_node: float,
        cpu_per_node: float,
        ram_per_node: float,
    ) -> List[int]:
        scores = self._base_bestfit_scores(state, feasible_nodes)
        return select_rack_aware_gang_nodes(state, feasible_nodes, scores, num_nodes)


def get_all_baselines() -> List[BaseScheduler]:
    """모든 비교 베이스라인 인스턴스 목록을 반환한다."""
    return [
        FirstFitScheduler(),
        BestFitScheduler(),
        DotProductScheduler(),
        L2NormScheduler(),
        FGDScheduler(adaptive=False),
        FGDScheduler(adaptive=True),
    ]
