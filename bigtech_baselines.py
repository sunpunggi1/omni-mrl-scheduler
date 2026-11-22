"""
빅테크 최신 논문(Google LAVA/NILAS MLSys '25, Alibaba DVLA OSDI)의 GPU 클러스터 정공법(Steel-Man) 구현체.

구성 (v11.0 코호트 KM & 연속/재예측 오라클 확장판):
1. EmpiricalSurvivalModel:
   - 동일 진입 코호트(`state.recent_cohort`)의 완료(비절단) + 미완료(우측 중도절단) 표본만으로
     Kaplan-Meier 생존곡선을 구축해 정상상태 길이편향(1.76~15배)을 억제한다.
     (유한 표본 및 `k_tail=5` 꼬리 구간에 의한 잔차 오차는 존재하며, 식별 불가 분위수는
      완료 표본 폴백 대신 관측 하한(`max_obs`) 이상의 지수 꼬리 외삽으로 구분 처리한다.)
   - `u_cut` 경계에서 좌·우극한 $C^0$ 연속(`lim_{u->u_cut-} = r_cut = lim_{u->u_cut+}`)을 보장한다.
     (`u` 방향 좌·우 미분은 일치하지 않는 $C^0$ 접합이며, `u > u_cut` 꼬리는 `tanh` 포화로
      무한번 미분 가능($C^inf$)하고 `CV=1.0` 주변에서 `CV` 방향 $C^1$ 평활성과 꼬리 상수성(`MRL(u)=r_cut`)을 만족한다.)
2. GoogleNILASGPUScheduler (`spatial_base` = "l2" | "bpack")
3. GoogleLAVAClassGPUScheduler (`spatial_base` = "l2" | "bpack")
4. AlibabaDVLAGPUScheduler (`spatial_base` = "l2" | "bpack")
5. GoogleLAVAMLOracleScheduler (`spatial_base` = "l2" | "bpack"): 분위수 4-클래스 ML 오라클
6. GoogleNILASMLOracleScheduler (`spatial_base` = "l2" | "bpack"): 이산화 손실 없는 연속 ML 오라클
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from adaptive_lindy import BPackSpatialOnlyScheduler
from baselines import select_rack_aware_gang_nodes
from simulator import BaseScheduler, ClusterConfig, ClusterState, Job, _safe_pearson


def _gpu_bucket(gpu: float) -> int:
    """작업 GPU 요청 크기를 최근 완료 평균 추정을 위한 5개 버킷으로 매핑한다."""
    if gpu < 1.0 - 1e-6:
        return 0
    if gpu < 1.5:
        return 1
    if gpu < 3.0:
        return 2
    if gpu < 6.0:
        return 3
    return 4


class EmpiricalSurvivalModel:
    """
    동일 진입 코호트(`recent_cohort`)로부터 무차원 Kaplan-Meier 생존곡선과
    경계 $C^0$ 연속 조건부 잔여수명(MRL)을 O(1) 상각 비용으로 유지·예측한다.
    """

    def __init__(self, refresh_interval: int = 32) -> None:
        self.refresh_interval = max(0, int(refresh_interval))
        self._last_ops: int = -10**9
        self.ref_scale: float = 0.0
        self.cv_eff: float = 1.0
        self.sorted_norm_durs: np.ndarray = np.empty(0, dtype=np.float64)
        self.suffix_sum: np.ndarray = np.empty(0, dtype=np.float64)
        self.bucket_norm_mean: Dict[int, float] = {}
        self.gpu_dur_corr: float = 0.0
        self.q25: float = 0.5
        self.q50: float = 1.0
        self.q75: float = 1.5
        self.upper_quartile: float = 1.5
        self.max_observed_norm: float = 1.0
        self.quantile_identifiable: Dict[str, bool] = {"q25": True, "q50": True, "q75": True}

        # Kaplan-Meier 적분 테이블 및 연속 꼬리 기준값
        self._km_t: np.ndarray = np.empty(0, dtype=np.float64)
        self._km_s: np.ndarray = np.empty(0, dtype=np.float64)
        self._km_area_to_cut: np.ndarray = np.empty(0, dtype=np.float64)
        self._km_u_cut: float = 0.0
        self._km_s_cut_left: float = 1.0
        self._km_r_cut: float = 1.0

    def reset(self) -> None:
        self._last_ops = -10**9
        self.ref_scale = 0.0
        self.cv_eff = 1.0
        self.sorted_norm_durs = np.empty(0, dtype=np.float64)
        self.suffix_sum = np.empty(0, dtype=np.float64)
        self.bucket_norm_mean = {}
        self.gpu_dur_corr = 0.0
        self.q25 = 0.5
        self.q50 = 1.0
        self.q75 = 1.5
        self.upper_quartile = 1.5
        self.max_observed_norm = 1.0
        self.quantile_identifiable = {"q25": True, "q50": True, "q75": True}
        self._km_t = np.empty(0, dtype=np.float64)
        self._km_s = np.empty(0, dtype=np.float64)
        self._km_area_to_cut = np.empty(0, dtype=np.float64)
        self._km_u_cut = 0.0
        self._km_s_cut_left = 1.0
        self._km_r_cut = 1.0

    def _build_kaplan_meier(
        self,
        norm_comp_sorted: np.ndarray,
        norm_cens: np.ndarray,
    ) -> None:
        """
        동일 진입 코호트의 비절단 완료 표본과 우측 중도절단 표본으로 Kaplan-Meier 생존·적분 테이블을 구축한다.
        `u_cut` 지점에서 좌극한 S(u_cut-)와 r_cut 적분 경계를 일치시켜 0-jump C^0 연속성을 보장한다.
        """
        m = len(norm_comp_sorted)
        k_tail = min(5, m)
        u_cut = max(float(norm_comp_sorted[m - k_tail]), 1e-9)

        events: List[Tuple[float, int]] = [(float(t), 1) for t in norm_comp_sorted if t > 0.0]
        max_obs = float(norm_comp_sorted[-1]) if m > 0 else 1.0
        for c in norm_cens:
            fc = float(c)
            if fc > 0.0 and math.isfinite(fc):
                events.append((fc, 0))
                if fc > max_obs:
                    max_obs = fc
        events.sort(key=lambda item: (item[0], -item[1]))

        n_risk = len(events)
        if n_risk == 0:
            self._km_t = np.array([0.0, u_cut], dtype=np.float64)
            self._km_s = np.array([1.0, 1.0], dtype=np.float64)
            self._km_area_to_cut = np.array([u_cut, 0.0], dtype=np.float64)
            self._km_u_cut = u_cut
            self._km_s_cut_left = 1.0
            self._km_r_cut = 1.0
            return

        km_t_list = [0.0]
        km_s_list = [1.0]
        s_curr = 1.0
        idx = 0
        total_events = len(events)
        while idx < total_events:
            t_cur = events[idx][0]
            d_cnt = 0
            c_cnt = 0
            while idx < total_events and abs(events[idx][0] - t_cur) <= 1e-12 * max(1.0, abs(t_cur)):
                if events[idx][1] == 1:
                    d_cnt += 1
                else:
                    c_cnt += 1
                idx += 1
            if d_cnt > 0 and n_risk > 0:
                s_curr *= max(0.0, 1.0 - d_cnt / float(n_risk))
                km_t_list.append(t_cur)
                km_s_list.append(s_curr)
            n_risk -= (d_cnt + c_cnt)

        km_t = np.asarray(km_t_list, dtype=np.float64)
        km_s = np.asarray(km_s_list, dtype=np.float64)
        self.max_observed_norm = max(1e-6, float(max_obs))

        def _km_quantile(surv_target: float) -> Tuple[float, bool]:
            hits = np.flatnonzero(km_s <= surv_target + 1e-12)
            if len(hits) > 0:
                return max(1e-6, float(km_t[int(hits[0])])), True
            # 우측 중도절단으로 생존곡선이 surv_target까지 내려가지 못한 경우:
            # 완료 표본 백분위수로 퇴행하지 않고 관측 하한(max_obs) 이상의 명시적 모형 의존 꼬리 외삽 사용
            s_end = float(km_s[-1])
            t_end = self.max_observed_norm
            if 1e-12 < s_end < 1.0 - 1e-12:
                extrap = t_end * (math.log(max(surv_target, 1e-9)) / math.log(s_end))
                return max(t_end, float(extrap)), False
            return t_end, False

        self.q25, id25 = _km_quantile(0.75)
        self.q50, id50 = _km_quantile(0.50)
        self.q75, id75 = _km_quantile(0.25)
        self.quantile_identifiable = {"q25": id25, "q50": id50, "q75": id75}
        self.upper_quartile = self.q75

        self._km_u_cut = u_cut
        # 엄격하게 u_cut 미만인 KM 점들만 내부 그리드로 취해 좌극한 S(u_cut-)를 보존한다.
        prefix_mask = km_t < (u_cut - 1e-12)
        if np.any(prefix_mask):
            p_t_head = km_t[prefix_mask]
            p_s_head = km_s[prefix_mask]
            s_cut_left = max(float(p_s_head[-1]), 1e-4)
        else:
            p_t_head = np.array([0.0], dtype=np.float64)
            p_s_head = np.array([1.0], dtype=np.float64)
            s_cut_left = 1.0
        self._km_s_cut_left = s_cut_left

        # u_cut 이후 구간의 실제 KM 조건부 잔여수명 r_cut (S(u_cut-) 기준으로 정확 정규화)
        cut_pos_right = int(np.searchsorted(km_t, u_cut, side="right")) - 1
        cut_pos_right = max(0, min(cut_pos_right, len(km_t) - 1))
        beyond_area = 0.0
        cur_t = u_cut
        cur_s = float(km_s[cut_pos_right])
        for j in range(cut_pos_right + 1, len(km_t)):
            if km_t[j] > cur_t:
                beyond_area += cur_s * (float(km_t[j]) - cur_t)
                cur_t = float(km_t[j])
                cur_s = float(km_s[j])
        if cur_s > 0.0 and max_obs > cur_t:
            beyond_area += cur_s * (max_obs - cur_t)

        raw_comp_tail = float(np.mean(norm_comp_sorted[m - k_tail:])) - u_cut
        self._km_r_cut = max(0.05, raw_comp_tail, beyond_area / s_cut_left)

        # [0, u_cut] 내부 적분 테이블 (좌극한과 r_cut이 u_cut에서 정확히 일치)
        p_t = np.append(p_t_head, u_cut)
        p_s = np.append(p_s_head, s_cut_left)
        area = np.zeros(len(p_t), dtype=np.float64)
        for i in range(len(p_t) - 2, -1, -1):
            area[i] = area[i + 1] + float(p_s[i]) * max(0.0, float(p_t[i + 1] - p_t[i]))
        self._km_t = p_t
        self._km_s = p_s
        self._km_area_to_cut = area

    def update(self, state: ClusterState, force: bool = False) -> Tuple[float, float]:
        """필요 시 동일 코호트 Kaplan-Meier 생존 분포를 갱신하고 `(ref_scale, cv_eff)`를 반환한다."""
        ops_ver = state.ops_version
        need_scale = (self.ref_scale <= 0.0 and len(state.active_jobs) >= 1)
        if not force and not need_scale and (ops_ver - self._last_ops < self.refresh_interval):
            return self.ref_scale, self.cv_eff

        stats = state.get_online_temporal_stats(refresh_interval=0)
        if not force and need_scale and stats.mean_age <= 0.0 and stats.n_completed < 12:
            if ops_ver - self._last_ops < self.refresh_interval:
                return self.ref_scale, self.cv_eff

        self._last_ops = ops_ver

        if stats.n_completed >= 12 and stats.mean_dur > 0.0:
            self.ref_scale = stats.mean_dur
        elif stats.mean_age > 0.0:
            self.ref_scale = stats.mean_age
        else:
            self.ref_scale = 0.0

        n_comp = len(state.recent_completions)
        if n_comp >= 12 and self.ref_scale > 0.0:
            carr = np.asarray(state.recent_completions, dtype=np.float64)
            gs = carr[:, 0]
            ds = carr[:, 1]
            raw_norm = ds / self.ref_scale
            norm_ds = np.sort(raw_norm)
            self.sorted_norm_durs = norm_ds
            self.suffix_sum = np.cumsum(norm_ds[::-1], dtype=np.float64)[::-1]

            # 동일 진입 코호트(`state.recent_cohort`)에 완료 표본이 충분하면 코호트 내 비절단/중도절단 분리 사용.
            # 코호트 표본 부족(<12건) 시에는 완료 버퍼만 사용하므로 우측 중도절단 비율이 높을 때 단기 편향이 생길 수 있다.
            cohort_comp: List[float] = []
            cohort_cens: List[float] = []
            for st_t, end_t, _ in state.recent_cohort:
                s_f = float(st_t)  # type: ignore[arg-type]
                if end_t is not None:
                    cohort_comp.append(max(0.0, float(end_t) - s_f) / self.ref_scale)  # type: ignore[arg-type]
                else:
                    cohort_cens.append(max(0.0, state.now - s_f) / self.ref_scale)

            if len(cohort_comp) >= 12:
                km_comp_sorted = np.sort(np.asarray(cohort_comp, dtype=np.float64))
                km_cens = np.asarray(cohort_cens, dtype=np.float64)
                self._build_kaplan_meier(km_comp_sorted, km_cens)
            else:
                self._build_kaplan_meier(norm_ds, np.empty(0, dtype=np.float64))

            cv_obs = stats.dur_cv
            if stats.mean_dur > 0.0 and stats.mean_age > 0.0:
                cv_insp = math.sqrt(max(0.0, 2.0 * (stats.mean_age / stats.mean_dur) - 1.0))
                cv_obs = max(cv_obs, cv_insp)
            self.cv_eff = cv_obs

            self.gpu_dur_corr = abs(_safe_pearson(np.log1p(gs), np.log1p(raw_norm)))
            new_bmean: Dict[int, float] = {}
            buckets = np.array([_gpu_bucket(float(g)) for g in gs], dtype=np.int32)
            for b in range(5):
                mask = (buckets == b)
                cnt = int(np.sum(mask))
                if cnt > 0:
                    b_raw = float(np.mean(raw_norm[mask]))
                    new_bmean[b] = (cnt * b_raw + 8.0 * 1.0) / (cnt + 8.0)
                else:
                    new_bmean[b] = 1.0
            self.bucket_norm_mean = new_bmean
        else:
            self.sorted_norm_durs = np.empty(0, dtype=np.float64)
            self.suffix_sum = np.empty(0, dtype=np.float64)
            self.bucket_norm_mean = {}
            self.gpu_dur_corr = 0.0
            self._km_t = np.empty(0, dtype=np.float64)

        return self.ref_scale, self.cv_eff

    def predict_new_duration(self, gpu: float, require_corr: bool = True) -> float:
        """도착 작업의 GPU 크기에 기반한 기대 무차원 수명을 반환한다 (무상관 시 중앙값 q50 사용)."""
        if not self.bucket_norm_mean:
            return 1.0
        if require_corr and self.gpu_dur_corr < 0.12:
            return float(self.q50) if self.q50 > 0.0 else 1.0
        return float(self.bucket_norm_mean.get(_gpu_bucket(gpu), 1.0))

    def predict_mrl(self, norm_ages: np.ndarray) -> np.ndarray:
        """
        무차원 경과시간 벡터 `norm_ages`에 대해 Kaplan-Meier 기반 조건부 잔여수명 E[X - u | X > u]를 근사한다.
        - `u = u_cut`에서 좌·우극한 값 일치 ($C^0$ 연속, 단 `u` 방향 좌·우 미분은 일치하지 않음)
        - `u > u_cut` 꼬리는 `tanh` 포화를 사용해 `u` 방향 무한번 미분 가능($C^inf$) 평활성을 유지
        - `CV = 1.0`에서 꼬리 구간(`u >= u_cut`)은 상수 `r_cut` 및 `CV` 방향 $C^1$ 평활성 만족
        """
        u = np.asarray(norm_ages, dtype=np.float64)
        if len(self._km_t) < 2 or len(self.sorted_norm_durs) < 12:
            return np.ones_like(u, dtype=np.float64)

        u_cut = self._km_u_cut
        s_cut_left = self._km_s_cut_left
        r_cut = self._km_r_cut

        d_pos = max(0.0, self.cv_eff - 1.0)
        d_neg = max(0.0, 1.0 - self.cv_eff)
        slope_dfr = 0.65 * (d_pos * d_pos) / (d_pos * d_pos + 0.25)
        decay_ifr_raw = 2.0 * (d_neg * d_neg) / (d_neg * d_neg + 0.04)
        # 유효한 생존분포의 MRL 항등식 R'(u) = h(u)R(u) - 1 >= -1 (비음수 위험률 h(u) >= 0)을 만족하도록
        # 꼬리 감쇠율 상한을 1 / r_cut으로 제한한다. (CV=1.0 근방 C^1 평활성 보존)
        decay_ifr = min(decay_ifr_raw, 1.0 / max(r_cut, 1e-9))

        out = np.empty_like(u, dtype=np.float64)
        interior = u < u_cut
        if np.any(interior):
            ui = np.maximum(0.0, u[interior])
            pos = np.searchsorted(self._km_t, ui, side="right") - 1
            pos = np.clip(pos, 0, len(self._km_t) - 2)
            s_u = np.maximum(self._km_s[pos], 1e-4)
            area_u = self._km_s[pos] * (self._km_t[pos + 1] - ui) + self._km_area_to_cut[pos + 1]
            out[interior] = np.maximum(0.0, (area_u + s_cut_left * r_cut) / s_u)

        tail = ~interior
        if np.any(tail):
            excess = np.maximum(0.0, u[tail] - u_cut)
            # tanh 포화로 꼬리 내부 매끄러움을 유지하며 DFR 포화 기여는 최대 r_cut + 4.0으로 유계화
            dfr_tail = 4.0 * np.tanh((slope_dfr * excess) / 4.0) if slope_dfr > 0.0 else 0.0
            out[tail] = r_cut * np.exp(-decay_ifr * excess) + dfr_tail

        return out


def _spatial_l2_base(
    state: ClusterState,
    feasible_nodes: np.ndarray,
    gpu: float,
    cpu: float,
    ram: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """L2Norm 기본 공간 비용과 유휴 여부를 반환한다 (0-GPU 노드 나눗셈 안전)."""
    g_cap = state.node_gpu_cap[feasible_nodes]
    safe_g_cap = np.maximum(g_cap, 1.0)
    c_cap = np.maximum(state.node_cpu_cap[feasible_nodes], 1e-9)
    r_cap = np.maximum(state.node_ram_cap[feasible_nodes], 1e-9)
    rem_g = np.where(g_cap > 0.0, (np.sum(state.gpu_slots[feasible_nodes], axis=1) - gpu) / safe_g_cap, 0.0)
    rem_c = (state.cpu_free[feasible_nodes] - cpu) / c_cap
    rem_r = (state.ram_free[feasible_nodes] - ram) / r_cap
    l2 = rem_g * rem_g + 0.5 * rem_c * rem_c + 0.5 * rem_r * rem_r
    is_pristine = state.node_job_count[feasible_nodes] == 0
    return l2, is_pristine


class _BigTechSpatialMixin(BaseScheduler):
    """빅테크 비교군이 `l2` 또는 `bpack` 공간 기반을 동일 인터페이스로 전환하기 위한 공통 믹스인."""

    def _init_spatial(self, spatial_base: str, hetero_aware: bool, topology_aware: bool) -> None:
        if spatial_base not in ("l2", "bpack"):
            raise ValueError(f"spatial_base는 'l2' 또는 'bpack'여야 합니다: {spatial_base!r}")
        self.spatial_base = spatial_base
        self.hetero_aware = hetero_aware
        self.topology_aware = topology_aware
        self._bpack: Optional[BPackSpatialOnlyScheduler] = (
            BPackSpatialOnlyScheduler(
                hetero_aware=hetero_aware,
                topology_aware=topology_aware,
            )
            if spatial_base == "bpack"
            else None
        )

    def _reset_spatial(self, config: ClusterConfig) -> None:
        if self._bpack is not None:
            self._bpack.reset(config)

    def _base_spatial_score(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
        for_gang: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._bpack is not None:
            score, is_pristine, _, _ = self._bpack._compute_spatial_base(
                state, feasible_nodes, gpu, cpu, ram, for_gang=for_gang
            )
            return score, is_pristine
        l2, is_pristine = _spatial_l2_base(state, feasible_nodes, gpu, cpu, ram)
        g_cap = state.node_gpu_cap[feasible_nodes]
        safe_max = max(state.max_gpu_cap, 1.0)
        score = l2 + 0.35 * (g_cap / safe_max) * (is_pristine & (gpu < g_cap - 1e-9))
        return score, is_pristine


class GoogleNILASGPUScheduler(_BigTechSpatialMixin):
    """Google MLSys '25 `NILAS` (Non-Invasive Lifetime-Aware Scoring) GPU 클러스터 이식 구현."""

    def __init__(
        self,
        name: str = "Google-NILAS",
        topology_aware: bool = False,
        w_temporal: float = 0.25,
        spatial_base: str = "l2",
        hetero_aware: bool = True,
    ) -> None:
        self.name = name
        self.w_temporal = float(w_temporal)
        self._init_spatial(spatial_base, hetero_aware=hetero_aware, topology_aware=topology_aware)
        self.survival = EmpiricalSurvivalModel(refresh_interval=32)

    def reset(self, config: ClusterConfig) -> None:
        self._reset_spatial(config)
        self.survival.reset()

    def score_nodes(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
        for_gang: bool = False,
    ) -> np.ndarray:
        ref_scale, _ = self.survival.update(state)
        score, is_pristine = self._base_spatial_score(state, feasible_nodes, gpu, cpu, ram, for_gang=for_gang)
        if ref_scale <= 0.0:
            return score

        u_max = np.where(is_pristine, 0.0, np.maximum(0.0, state.now - state.node_oldest_start[feasible_nodes]) / ref_scale)
        u_min = np.where(is_pristine, 0.0, np.maximum(0.0, state.now - state.node_newest_start[feasible_nodes]) / ref_scale)
        e_max = self.survival.predict_mrl(u_max)
        e_min = self.survival.predict_mrl(u_min)
        e_host = np.where(is_pristine, 0.0, np.maximum(e_max, e_min))

        d_new = max(1e-6, self.survival.predict_new_duration(gpu, require_corr=True))
        empty_ext = 0.35 if self.spatial_base == "bpack" else 1.0
        ext_cost = np.where(
            is_pristine,
            empty_ext,
            np.minimum(np.maximum(0.0, d_new - e_host) / d_new, 2.0),
        )
        sync_gap = np.where(
            is_pristine,
            0.0,
            0.15 * np.minimum(np.maximum(0.0, e_host - d_new) / d_new, 2.0),
        )
        return score + self.w_temporal * (ext_cost + sync_gap)

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        scores = self.score_nodes(state, feasible_nodes, gpu, cpu, ram, for_gang=False)
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
        scores = self.score_nodes(state, feasible_nodes, gpu_per_node, cpu_per_node, ram_per_node, for_gang=True)
        if not self.topology_aware:
            order = np.argsort(scores, kind="mergesort")
            return [int(feasible_nodes[i]) for i in order[:num_nodes]]
        return select_rack_aware_gang_nodes(state, feasible_nodes, scores, num_nodes)


class GoogleLAVAClassGPUScheduler(_BigTechSpatialMixin):
    """Google MLSys '25 `LAVA` 4-클래스 Recycling 엔진의 무예측 코호트 KM-MRL 기반 GPU 이식 구현."""

    def __init__(
        self,
        name: str = "Google-LAVA",
        topology_aware: bool = False,
        spatial_base: str = "l2",
        hetero_aware: bool = True,
        w_tier: float = 0.70,
    ) -> None:
        self.name = name
        self.w_tier = float(w_tier)
        self._init_spatial(spatial_base, hetero_aware=hetero_aware, topology_aware=topology_aware)
        self.survival = EmpiricalSurvivalModel(refresh_interval=32)

    def reset(self, config: ClusterConfig) -> None:
        self._reset_spatial(config)
        self.survival.reset()

    def _to_lc(self, norm_dur: np.ndarray) -> np.ndarray:
        """경험적 분위수 (Q25, Q50, Q75) 기반으로 4개 수명 클래스(0..3)에 매핑한다."""
        q25 = max(1e-6, self.survival.q25)
        q50 = max(q25 + 1e-6, self.survival.q50)
        q75 = max(q50 + 1e-6, self.survival.q75)
        lc = np.zeros_like(norm_dur, dtype=np.int32)
        lc = np.where(norm_dur >= q25, 1, lc)
        lc = np.where(norm_dur >= q50, 2, lc)
        lc = np.where(norm_dur >= q75, 3, lc)
        return lc

    def score_nodes(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
        for_gang: bool = False,
    ) -> np.ndarray:
        ref_scale, _ = self.survival.update(state)
        base_score, is_pristine = self._base_spatial_score(state, feasible_nodes, gpu, cpu, ram, for_gang=for_gang)
        if ref_scale <= 0.0 or len(self.survival.sorted_norm_durs) < 12:
            if self.spatial_base == "bpack":
                return base_score
            return np.where(is_pristine, 1.0, 0.0) + 0.25 * base_score

        u_max = np.where(is_pristine, 0.0, np.maximum(0.0, state.now - state.node_oldest_start[feasible_nodes]) / ref_scale)
        u_min = np.where(is_pristine, 0.0, np.maximum(0.0, state.now - state.node_newest_start[feasible_nodes]) / ref_scale)
        e_host = np.maximum(self.survival.predict_mrl(u_max), self.survival.predict_mrl(u_min))

        host_lc = self._to_lc(e_host)
        d_new = self.survival.predict_new_duration(gpu, require_corr=True)
        job_lc = int(self._to_lc(np.array([d_new]))[0])

        diff = host_lc - job_lc
        non_empty = ~is_pristine
        tier_cost = np.full(len(feasible_nodes), 2.5, dtype=np.float64)
        tier_cost = np.where(non_empty & (diff > 0), 0.1 * diff, tier_cost)
        tier_cost = np.where(non_empty & (diff == 0), 0.8, tier_cost)
        tier_cost = np.where(non_empty & (diff < 0), 1.6 + 0.1 * (-diff), tier_cost)

        if self.spatial_base == "bpack":
            rel_tier = np.where(non_empty, tier_cost - 0.8, 0.35)
            return base_score + self.w_tier * rel_tier
        return self.w_tier * tier_cost + 0.25 * base_score

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        scores = self.score_nodes(state, feasible_nodes, gpu, cpu, ram, for_gang=False)
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
        scores = self.score_nodes(state, feasible_nodes, gpu_per_node, cpu_per_node, ram_per_node, for_gang=True)
        if not self.topology_aware:
            order = np.argsort(scores, kind="mergesort")
            return [int(feasible_nodes[i]) for i in order[:num_nodes]]
        return select_rack_aware_gang_nodes(state, feasible_nodes, scores, num_nodes)


class AlibabaDVLAGPUScheduler(_BigTechSpatialMixin):
    """Alibaba OSDI `DVLA` 온라인 배치 부채 회피 정책 `DAPP` (라이브 마이그레이션 제외 GPU 이식 구현)."""

    def __init__(
        self,
        name: str = "Alibaba-DVLA",
        topology_aware: bool = False,
        spatial_base: str = "l2",
        hetero_aware: bool = True,
        w_debt: float = 0.70,
    ) -> None:
        self.name = name
        self.w_debt = float(w_debt)
        self._init_spatial(spatial_base, hetero_aware=hetero_aware, topology_aware=topology_aware)
        self.survival = EmpiricalSurvivalModel(refresh_interval=32)

    def reset(self, config: ClusterConfig) -> None:
        self._reset_spatial(config)
        self.survival.reset()

    def score_nodes(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
        for_gang: bool = False,
    ) -> np.ndarray:
        ref_scale, _ = self.survival.update(state)
        base_score, is_pristine = self._base_spatial_score(state, feasible_nodes, gpu, cpu, ram, for_gang=for_gang)
        if ref_scale <= 0.0 or len(self.survival.sorted_norm_durs) < 12:
            if self.spatial_base == "bpack":
                return base_score
            return base_score + 0.3 * is_pristine

        u_max = np.where(is_pristine, 0.0, np.maximum(0.0, state.now - state.node_oldest_start[feasible_nodes]) / ref_scale)
        e_host = self.survival.predict_mrl(u_max)
        debt_thresh = max(1e-6, self.survival.q75)
        is_debt_host = (~is_pristine) & (e_host >= debt_thresh)
        is_churn_host = (~is_pristine) & (e_host < debt_thresh)

        d_new = self.survival.predict_new_duration(gpu, require_corr=True)
        is_long_job = (d_new >= debt_thresh) or (self.survival.gpu_dur_corr < 0.12 and gpu >= 2.0 - 1e-9)

        if is_long_job:
            debt_penalty = np.where(is_debt_host, 0.0, np.where(is_pristine, 0.8, 1.4))
        else:
            debt_penalty = np.where(is_churn_host, 0.0, np.where(is_debt_host, 0.35, 0.9))

        if self.spatial_base == "bpack":
            rel_debt = np.where(is_pristine, 0.25, debt_penalty)
            return base_score + self.w_debt * rel_debt
        return self.w_debt * debt_penalty + 0.5 * base_score

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        scores = self.score_nodes(state, feasible_nodes, gpu, cpu, ram, for_gang=False)
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
        scores = self.score_nodes(state, feasible_nodes, gpu_per_node, cpu_per_node, ram_per_node, for_gang=True)
        if not self.topology_aware:
            order = np.argsort(scores, kind="mergesort")
            return [int(feasible_nodes[i]) for i in order[:num_nodes]]
        return select_rack_aware_gang_nodes(state, feasible_nodes, scores, num_nodes)


class GoogleLAVAMLOracleScheduler(_BigTechSpatialMixin):
    """
    [Suite 2 전용 — 분위수 4-클래스 ML 오라클 비교군]
    워크로드 사분위수(Q25, Q50, Q75) 4-클래스와 순수 코호트 KM 재예측 폴백을 사용한다.
    """

    def __init__(
        self,
        pred_map: Dict[int, float],
        time_unit_hours: float = 1.0,
        name: str = "Google-LAVA-ML",
        lc_thresholds: Optional[Tuple[float, float, float]] = None,
        spatial_base: str = "l2",
        hetero_aware: bool = False,
        topology_aware: bool = False,
        w_tier: float = 0.70,
    ) -> None:
        self.pred_map = pred_map
        self.time_unit_hours = max(1e-12, float(time_unit_hours))
        self.name = name
        self.w_tier = float(w_tier)
        self._init_spatial(spatial_base, hetero_aware=hetero_aware, topology_aware=topology_aware)
        if lc_thresholds is not None:
            self.lc_thresholds = (float(lc_thresholds[0]), float(lc_thresholds[1]), float(lc_thresholds[2]))
        elif pred_map:
            vals = np.asarray(list(pred_map.values()), dtype=np.float64) / self.time_unit_hours
            q25, q50, q75 = np.quantile(vals, [0.25, 0.50, 0.75])
            self.lc_thresholds = (float(q25), float(q50), float(q75))
        else:
            self.lc_thresholds = (0.5, 1.0, 2.0)

        self._node_active_preds: Dict[int, Dict[int, Tuple[float, float]]] = {}
        self._current_pred_dur: float = self.time_unit_hours
        self._survival = EmpiricalSurvivalModel(refresh_interval=32)

    def reset(self, config: ClusterConfig) -> None:
        self._reset_spatial(config)
        self._node_active_preds.clear()
        self._current_pred_dur = self.time_unit_hours
        self._survival.reset()

    def set_current_job(self, job: Job) -> None:
        self._current_pred_dur = float(self.pred_map.get(id(job), self.time_unit_hours))

    def on_allocated(self, node_idx: int, job: Job, now: float) -> None:
        pred_dur = float(self.pred_map.get(id(job), self.time_unit_hours))
        self._node_active_preds.setdefault(int(node_idx), {})[id(job)] = (float(now), float(now + pred_dur))

    def on_released(self, node_idx: int, job: Job) -> None:
        bucket = self._node_active_preds.get(int(node_idx))
        if bucket is not None:
            bucket.pop(id(job), None)
            if not bucket:
                del self._node_active_preds[int(node_idx)]

    def _hours_to_lc(self, hours: np.ndarray) -> np.ndarray:
        q25, q50, q75 = self.lc_thresholds
        lc = np.zeros_like(hours, dtype=np.int32)
        lc = np.where(hours >= q25, 1, lc)
        lc = np.where(hours >= q50, 2, lc)
        lc = np.where(hours >= q75, 3, lc)
        return lc

    def _score_with_pred(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
        job_pred_dur: float,
        for_gang: bool = False,
    ) -> np.ndarray:
        self._survival.update(state)
        base_score, is_pristine = self._base_spatial_score(state, feasible_nodes, gpu, cpu, ram, for_gang=for_gang)

        now = state.now
        ref = max(self._survival.ref_scale, 1e-6)
        rem_hours = np.zeros(len(feasible_nodes), dtype=np.float64)
        for pos, ni in enumerate(feasible_nodes):
            n_idx = int(ni)
            if is_pristine[pos]:
                rem_hours[pos] = 0.0
                continue
            preds = self._node_active_preds.get(n_idx)
            if preds:
                # 개별 실행 작업별로 미만료 예측 잔여수명 또는 경과시간(uptime) 조건부 KM 재추정 최댓값 계산
                best_rem = 0.0
                expired_ages: List[float] = []
                for st_t, ex_t in preds.values():
                    if ex_t > now:
                        best_rem = max(best_rem, (ex_t - now) / self.time_unit_hours)
                    else:
                        expired_ages.append(max(0.0, now - st_t) / ref)
                if expired_ages:
                    mrls = self._survival.predict_mrl(np.asarray(expired_ages, dtype=np.float64))
                    km_rem = float(np.max(mrls)) * ref / self.time_unit_hours
                    best_rem = max(best_rem, km_rem)
                rem_hours[pos] = best_rem
            else:
                age = max(0.0, now - float(state.node_oldest_start[n_idx]))
                mrl_norm = float(self._survival.predict_mrl(np.array([age / ref]))[0])
                rem_hours[pos] = max(0.0, (mrl_norm * ref) / self.time_unit_hours)

        job_lc = int(self._hours_to_lc(np.array([job_pred_dur / self.time_unit_hours]))[0])
        host_lc = self._hours_to_lc(rem_hours)
        diff = host_lc - job_lc

        non_empty = ~is_pristine
        tier_cost = np.full(len(feasible_nodes), 2.5, dtype=np.float64)
        tier_cost = np.where(non_empty & (diff > 0), 0.1 * diff, tier_cost)
        tier_cost = np.where(non_empty & (diff == 0), 0.8, tier_cost)
        tier_cost = np.where(non_empty & (diff < 0), 1.6 + 0.1 * (-diff), tier_cost)

        if self.spatial_base == "bpack":
            rel_tier = np.where(non_empty, tier_cost - 0.8, 0.35)
            return base_score + self.w_tier * rel_tier
        return self.w_tier * tier_cost + 0.25 * base_score

    def score_nodes(self, state, feasible_nodes, gpu, cpu, ram) -> np.ndarray:
        return self._score_with_pred(state, feasible_nodes, gpu, cpu, ram, self._current_pred_dur, for_gang=False)

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        scores = self.score_nodes(state, feasible_nodes, gpu, cpu, ram)
        return int(feasible_nodes[np.argmin(scores)])


class GoogleNILASMLOracleScheduler(_BigTechSpatialMixin):
    """
    [Suite 2 신규 연속 ML 오라클 상한 — `Google-NILAS-ML`]

    4-클래스 이산화(quantization loss) 없이 연속 ML 예측 잔여수명 `e_host`와
    도착 작업 예측 수명 `d_new`로 직접 호스트 연장 비용 `Delta T`를 계산하는 강력한 ML 오라클.
    오예측 초과 시에는 린디 강제 없이 순수 코호트 KM `predict_mrl`만 사용한다.
    """

    def __init__(
        self,
        pred_map: Dict[int, float],
        time_unit_hours: float = 1.0,
        name: str = "Google-NILAS-ML",
        spatial_base: str = "bpack",
        hetero_aware: bool = True,
        topology_aware: bool = False,
        w_temporal: float = 0.65,
    ) -> None:
        self.pred_map = pred_map
        self.time_unit_hours = max(1e-12, float(time_unit_hours))
        self.name = name
        self.w_temporal = float(w_temporal)
        self._init_spatial(spatial_base, hetero_aware=hetero_aware, topology_aware=topology_aware)
        self._node_active_preds: Dict[int, Dict[int, Tuple[float, float]]] = {}
        self._current_pred_dur: float = self.time_unit_hours
        self._survival = EmpiricalSurvivalModel(refresh_interval=32)

    def reset(self, config: ClusterConfig) -> None:
        self._reset_spatial(config)
        self._node_active_preds.clear()
        self._current_pred_dur = self.time_unit_hours
        self._survival.reset()

    def set_current_job(self, job: Job) -> None:
        self._current_pred_dur = float(self.pred_map.get(id(job), self.time_unit_hours))

    def on_allocated(self, node_idx: int, job: Job, now: float) -> None:
        pred_dur = float(self.pred_map.get(id(job), self.time_unit_hours))
        self._node_active_preds.setdefault(int(node_idx), {})[id(job)] = (float(now), float(now + pred_dur))

    def on_released(self, node_idx: int, job: Job) -> None:
        bucket = self._node_active_preds.get(int(node_idx))
        if bucket is not None:
            bucket.pop(id(job), None)
            if not bucket:
                del self._node_active_preds[int(node_idx)]

    def score_nodes(self, state, feasible_nodes, gpu, cpu, ram) -> np.ndarray:
        self._survival.update(state)
        base_score, is_pristine = self._base_spatial_score(state, feasible_nodes, gpu, cpu, ram, for_gang=False)

        now = state.now
        ref = max(self._survival.ref_scale, 1e-6)
        rem_hours = np.zeros(len(feasible_nodes), dtype=np.float64)
        for pos, ni in enumerate(feasible_nodes):
            n_idx = int(ni)
            if is_pristine[pos]:
                rem_hours[pos] = 0.0
                continue
            preds = self._node_active_preds.get(n_idx)
            if preds:
                # 미만료 예측과 만료 후 uptime 조건부 KM MRL을 개별 작업 단위로 결합
                best_rem = 0.0
                expired_ages: List[float] = []
                for st_t, ex_t in preds.values():
                    if ex_t > now:
                        best_rem = max(best_rem, (ex_t - now) / self.time_unit_hours)
                    else:
                        expired_ages.append(max(0.0, now - st_t) / ref)
                if expired_ages:
                    mrls = self._survival.predict_mrl(np.asarray(expired_ages, dtype=np.float64))
                    km_rem = float(np.max(mrls)) * ref / self.time_unit_hours
                    best_rem = max(best_rem, km_rem)
                rem_hours[pos] = best_rem
            else:
                age = max(0.0, now - float(state.node_oldest_start[n_idx]))
                mrl_norm = float(self._survival.predict_mrl(np.array([age / ref]))[0])
                rem_hours[pos] = max(0.0, (mrl_norm * ref) / self.time_unit_hours)

        d_new = max(1e-6, self._current_pred_dur / self.time_unit_hours)
        empty_ext = 0.35 if self.spatial_base == "bpack" else 1.0
        ext_cost = np.where(
            is_pristine,
            empty_ext,
            np.minimum(np.maximum(0.0, d_new - rem_hours) / d_new, 2.0),
        )
        sync_gap = np.where(
            is_pristine,
            0.0,
            0.15 * np.minimum(np.maximum(0.0, rem_hours - d_new) / d_new, 2.0),
        )
        return base_score + self.w_temporal * (ext_cost + sync_gap)

    def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
        scores = self.score_nodes(state, feasible_nodes, gpu, cpu, ram)
        return int(feasible_nodes[np.argmin(scores)])


def run_ml_lava_simulation(
    jobs: List[Job],
    scheduler: BaseScheduler,
    config: ClusterConfig,
    recent_window: int = 200,
) -> Dict[str, float]:
    """
    [Suite 2 통합 하네스] `GoogleLAVAMLOracleScheduler`, `GoogleNILASMLOracleScheduler`,
    일반 Non-Clairvoyant 스케줄러(`Omni-MRL`, `BPack-SpatialOnly`)를 **완전 동일 루프**로 실행한다.
    """
    from collections import deque

    ordered = sorted(jobs, key=lambda x: (x.arrival_time, x.job_id))
    has_oracle_hooks = isinstance(scheduler, (GoogleLAVAMLOracleScheduler, GoogleNILASMLOracleScheduler))

    # 1) Blocking 모드
    state = ClusterState(config)
    scheduler.reset(config)
    dep_heap: List[Tuple[float, int, int, Job, object]] = []
    dep_seq = 0
    total_gh = 0.0
    rej_gh = 0.0
    if ordered:
        state.now = ordered[0].arrival_time

    for j in ordered:
        while dep_heap and dep_heap[0][0] <= j.arrival_time:
            dt_t, _, n_idx, dj, allocs = heapq.heappop(dep_heap)
            state.now = dt_t
            state.release(n_idx, dj, allocs)  # type: ignore[arg-type]
            if has_oracle_hooks:
                scheduler.on_released(n_idx, dj)  # type: ignore[attr-defined]

        state.now = j.arrival_time
        state.recent_requests.append((j.gpu, j.cpu, j.ram))
        if len(state.recent_requests) > recent_window:
            del state.recent_requests[0]
        state.recent_specs.append(j.gpu_spec)
        state.recent_node_counts.append(max(1, int(j.num_nodes)))

        total_gh += j.gpu * j.duration
        feas = state.get_feasible_nodes(j.gpu, j.cpu, j.ram, j.gpu_spec)
        if len(feas) == 0:
            rej_gh += j.gpu * j.duration
            continue
        if has_oracle_hooks:
            scheduler.set_current_job(j)  # type: ignore[attr-defined]
        state.current_gpu_spec = j.gpu_spec
        try:
            chosen = int(feas[0]) if len(feas) == 1 else int(scheduler.select_node(state, feas, j.gpu, j.cpu, j.ram))
        finally:
            state.current_gpu_spec = None
        allocs = state.allocate(chosen, j)
        if has_oracle_hooks:
            scheduler.on_allocated(chosen, j, state.now)  # type: ignore[attr-defined]
        dep_seq += 1
        heapq.heappush(dep_heap, (j.arrival_time + j.duration, dep_seq, chosen, j, allocs))

    blk_rate = (100.0 * rej_gh / total_gh) if total_gh > 0.0 else 0.0

    # 2) Backfill Pending Queue 모드 (backfill_window=16, max_hol_bypasses=8)
    state_q = ClusterState(config)
    scheduler.reset(config)
    dep_heap_q: List[Tuple[float, int, int, Job, object]] = []
    pending: deque = deque()
    dep_seq_q = 0
    arr_idx = 0
    n_jobs = len(ordered)
    waits: List[float] = []
    hol_ticket: Optional[int] = None
    hol_bypasses = 0
    if ordered:
        state_q.now = ordered[0].arrival_time

    def dispatch_q(jb: Job, feas: np.ndarray) -> None:
        nonlocal dep_seq_q
        waits.append(max(0.0, state_q.now - jb.arrival_time))
        if has_oracle_hooks:
            scheduler.set_current_job(jb)  # type: ignore[attr-defined]
        state_q.current_gpu_spec = jb.gpu_spec
        try:
            chosen = int(feas[0]) if len(feas) == 1 else int(scheduler.select_node(state_q, feas, jb.gpu, jb.cpu, jb.ram))
        finally:
            state_q.current_gpu_spec = None
        allocs = state_q.allocate(chosen, jb)
        if has_oracle_hooks:
            scheduler.on_allocated(chosen, jb, state_q.now)  # type: ignore[attr-defined]
        dep_seq_q += 1
        heapq.heappush(dep_heap_q, (state_q.now + max(0.0, jb.duration), dep_seq_q, chosen, jb, allocs))

    def try_dispatch() -> None:
        nonlocal hol_ticket, hol_bypasses
        while pending:
            tkt, hd = pending[0]
            if hol_ticket != tkt:
                hol_ticket = tkt
                hol_bypasses = 0
            feas = state_q.get_feasible_nodes(hd.gpu, hd.cpu, hd.ram, hd.gpu_spec)
            if len(feas) >= 1:
                pending.popleft()
                dispatch_q(hd, feas)
                hol_ticket = None
                hol_bypasses = 0
                continue
            if hol_bypasses >= 8:
                break
            moved = False
            for idx in range(1, min(len(pending), 17)):
                _, cand = pending[idx]
                cf = state_q.get_feasible_nodes(cand.gpu, cand.cpu, cand.ram, cand.gpu_spec)
                if len(cf) >= 1:
                    del pending[idx]
                    dispatch_q(cand, cf)
                    hol_bypasses += 1
                    moved = True
                    break
            if not moved:
                break

    while arr_idx < n_jobs or dep_heap_q:
        na = ordered[arr_idx].arrival_time if arr_idx < n_jobs else float("inf")
        nd = dep_heap_q[0][0] if dep_heap_q else float("inf")
        tol = 1e-12 * abs(na) if math.isfinite(na) else 0.0
        if nd <= na + tol:
            te = max(state_q.now, min(na, nd))
            etol = 1e-12 * abs(te)
            state_q.now = te
            while dep_heap_q and dep_heap_q[0][0] <= te + etol:
                _, _, ni, dj, allocs = heapq.heappop(dep_heap_q)
                state_q.release(ni, dj, allocs)  # type: ignore[arg-type]
                if has_oracle_hooks:
                    scheduler.on_released(ni, dj)  # type: ignore[attr-defined]
            try_dispatch()
        else:
            te = na
            etol = 1e-12 * abs(te)
            state_q.now = te
            while arr_idx < n_jobs and ordered[arr_idx].arrival_time <= te + etol:
                aj = ordered[arr_idx]
                state_q.recent_requests.append((aj.gpu, aj.cpu, aj.ram))
                if len(state_q.recent_requests) > recent_window:
                    del state_q.recent_requests[0]
                state_q.recent_specs.append(aj.gpu_spec)
                state_q.recent_node_counts.append(max(1, int(aj.num_nodes)))
                pending.append((arr_idx, aj))
                arr_idx += 1
            try_dispatch()

    return {
        "blocking": blk_rate,
        "backfill_wait": float(np.mean(waits)) if waits else 0.0,
    }
