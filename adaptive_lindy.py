"""
5단계 핵심 알고리즘: Adaptive B-LindyPack (온라인 Corr(gpu, age) 적응형 시공간 스케줄러)

기존 4단계 B-LindyPack과 무제한 EvoSTF_Dist 사이에 존재했던 파레토 트레이드오프
(W1 표준혼합 +8.4% 방어 vs W3 장단기 혼재 +18.1% 돌파)를 원리적으로 해소한다.

핵심 수학적 원리:
1. 온라인 크기-경과시간 상관관계 추정 (Non-Clairvoyant, O(1) 상각, 시간 단위 불변):
   현재 실행 중인 작업의 (GPU 크기, 정규화 경과시간) 로그 상관계수 rho_run과
   최근 완료된 과거 작업의 실측 상관계수 rho_comp를 Empirical Bayes 수축 및 EWMA로 안정화하여
   실시간 상관계수 rho_eff를 추정한다.
2. 연속 비연동 게이트(Decoupling Gate) alpha(rho_eff) in (0, 1):
   alpha = sigmoid((rho_center - rho_eff) / rho_slope)
   - 양(+)의 상관이 강할 때(W1: 대형 4-GPU 잡이 장기 실행): alpha -> 0
     => 경계 보호형 유계 린디-에이지 활성화 (4-GPU 경계 역전 원천 차단)
   - 상관이 0 또는 음(-)일 때(W3/W4/W5: 소형 잡도 장기 스트래글러 발생): alpha -> 1
     => 척도 불변 로그 린디-에이지 활성화 (오래된 스트래글러 노드에 신규 잡을 집중 격리하여
        젊은 노드가 빠르게 순수 8-GPU 노드로 재생되도록 유도)
3. 완전한 시간 단위 불변성(Clock-Unit Invariance):
   모든 경과시간을 온라인 평균 실측 수명(mean_dur)으로 무차원 정규화(max_age / mean_dur)하여
   워크로드 타임스탬프가 초/분/시간 어느 단위로 주어져도 100% 동일한 배치 결정을 내린다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from baselines import select_rack_aware_gang_nodes
from simulator import BaseScheduler, ClusterConfig, ClusterState

# 4단계에서 발견된 B-LindyPack 핵심 물리 상수 (고정 — 공간 과적합 방지)
# [w_gpu, w_res, w_imb, w_pristine, w_b4, w_whole, w_lindy, tau_hours]
DEFAULT_BASE_THETA = np.array(
    [1.0, 0.822, 0.161, 1.302, 0.493, 0.344, 0.446, 7.15],
    dtype=np.float64,
)


@dataclass(frozen=True)
class AdaptiveLindyConfig:
    """Adaptive B-LindyPack의 온라인 상관·헤비테일 적응 파라미터."""

    rho_center: float = 0.11          # 4-GPU 경계 역전 전이 임계 상관계수 (rho_c)
    rho_slope: float = 0.032          # 시그모이드 전이 폭 (s_rho)
    tau_safe_norm: float = 6.0        # 보수 모드(alpha->0) 무차원 포화 척도 (tau / mean_dur)
    tau_log_norm: float = 0.38        # 격리 모드(alpha->1) 무차원 로그 척도 (age / (tau_log * mean_dur))
    lindy_log_scale: float = 1.12     # 격리 모드 로그 린디 상대 강도 (kappa_log)
    max_log_potential: float = 4.0    # 극단 이상치 폭주 방지용 소프트 상한
    corr_source: str = "blend"        # "log_run", "lin_run", "blend"
    comp_blend_weight: float = 0.35   # 완료 버퍼 실측 상관 블렌딩 비중 (corr_source=="blend"일 때)
    prior_rho: float = 0.20           # 콜드 스타트 Empirical Bayes 보수 사전값 (초기 W1 안전 보호)
    prior_n: float = 12.0             # 콜드 스타트 사전 표본 수 (정상 상태 편향 최소화)
    ewma_beta: float = 0.35           # 온라인 통계 지수가중이동평균(EWMA) 반응 계수
    refresh_interval: int = 32        # 온라인 통계 갱신 주기 (할당/해제 이벤트 횟수)
    cv_center: float = 0.65           # 헤비테일/린디 유효성 게이트 임계 변동계수 (IFR < 0.65 차단)
    cv_slope: float = 0.08            # 변동계수 게이트 시그모이드 전이 폭
    prior_cv: float = 1.0             # 콜드 스타트 중립 변동계수 사전값 (지수분포 기준 CV=1.0)

    def __post_init__(self) -> None:
        if self.corr_source not in ("blend", "log_run", "lin_run"):
            raise ValueError(
                f"지원하지 않는 corr_source입니다: {self.corr_source!r} "
                "('blend', 'log_run', 'lin_run' 중 하나여야 합니다)"
            )
        if self.rho_slope <= 0.0 or self.cv_slope <= 0.0:
            raise ValueError("rho_slope와 cv_slope는 양수여야 합니다.")
        if self.tau_safe_norm <= 0.0 or self.tau_log_norm <= 0.0:
            raise ValueError("tau_safe_norm과 tau_log_norm은 양수여야 합니다.")
        if not (0.0 <= self.ewma_beta <= 1.0):
            raise ValueError(f"ewma_beta는 [0, 1] 범위여야 합니다: {self.ewma_beta}")
        if self.refresh_interval < 0:
            raise ValueError(f"refresh_interval은 0 이상이어야 합니다: {self.refresh_interval}")


class AdaptiveBLindyPackScheduler(BaseScheduler):
    """
    온라인 Corr(gpu, age) + 변동계수(CV) 이중 게이트 기반 적응형 시공간 GPU 스케줄러: Adaptive B-LindyPack.

    - 100% Non-Clairvoyant (미래 작업 수명 절대 미사용)
    - O(1) 벡터화 노드 점수 계산 (Python 루프 제거, FGD 대비 ~7~10배 고속)
    - 엄격한 시간 단위 불변성(Clock-Unit Invariant: 실측 시간 척도 관측 전까지 하드코딩 상수 배제)
    - IFR/반린디(Anti-Lindy) 안전 보호: 실측 완료 수명 CV < 0.65인 저분산/결정적 환경에서 린디 항 자동 감쇠
    """

    def __init__(
        self,
        base_theta: Optional[np.ndarray] = None,
        adapt_cfg: Optional[AdaptiveLindyConfig] = None,
        name: str = "Adaptive-B-Lindy",
        record_diagnostics: bool = False,
        hetero_aware: bool = False,
        topology_aware: bool = False,
    ):
        self.base_theta = np.asarray(
            DEFAULT_BASE_THETA if base_theta is None else base_theta,
            dtype=np.float64,
        )
        self.adapt_cfg = AdaptiveLindyConfig() if adapt_cfg is None else adapt_cfg
        self.name = name
        self.record_diagnostics = record_diagnostics
        self.hetero_aware = hetero_aware
        self.topology_aware = topology_aware

        # 실행 중 온라인 상태 (reset에서 초기화)
        self._smoothed_rho: float = self.adapt_cfg.prior_rho
        self._smoothed_cv: float = self.adapt_cfg.prior_cv
        self._last_stats_ops: int = -10**9
        self._cached_alpha: float = 0.0
        self._cached_gamma: float = 1.0
        self._cached_ref_scale: float = 0.0
        self._cached_spec_demand: Optional[Dict[str, float]] = None
        self._last_spec_len: int = -1
        self._last_spec_ops: int = -10**9
        self._cached_model_ids: Optional[Tuple[List[str], np.ndarray]] = None

        # 선택적 진단 로그 (상관/게이트/CV 궤적 검증용)
        self.rho_history: List[float] = []
        self.alpha_history: List[float] = []
        self.gamma_history: List[float] = []

    def reset(self, config: ClusterConfig) -> None:
        self._smoothed_rho = self.adapt_cfg.prior_rho
        self._smoothed_cv = self.adapt_cfg.prior_cv
        self._last_stats_ops = -10**9
        self._cached_alpha = self._rho_to_alpha(self._smoothed_rho)
        self._cached_gamma = self._cv_to_gamma(self._smoothed_cv)
        self._cached_ref_scale = 0.0
        self._cached_spec_demand = None
        self._last_spec_len = -1
        self._last_spec_ops = -10**9
        self._cached_model_ids = None
        self.rho_history = []
        self.alpha_history = []
        self.gamma_history = []

    def _get_spec_pressure_vector(self, state: ClusterState, feasible_nodes: np.ndarray) -> Optional[np.ndarray]:
        """
        무제약(gpu_spec=None) 작업 배치 시 희소 GPU 기종의 실시간 가용 여력을 보호하기 위한
        온라인 Live Spec-Scarcity 압력 벡터(len(feasible_nodes),)를 O(1) 상각 비용으로 계산한다.
        """
        if state.node_models is None or not state.model_gpu_share:
            return None
        specs = state.recent_specs
        n_specs = len(specs)
        if n_specs == 0:
            return None

        ops_ver = state.ops_version
        if self._last_spec_len != n_specs or (ops_ver - self._last_spec_ops >= 16):
            demand: Dict[str, float] = {}
            has_explicit_spec = False
            for sp in specs:
                if not sp:
                    continue
                has_explicit_spec = True
                allowed = [tok.strip() for tok in sp.split("|") if tok.strip()]
                if not allowed:
                    continue
                share = 1.0 / len(allowed)
                for m in allowed:
                    demand[m] = demand.get(m, 0.0) + share

            if not has_explicit_spec:
                self._cached_spec_demand = None
            else:
                self._cached_spec_demand = {m: v / n_specs for m, v in demand.items()}
            self._last_spec_len = n_specs
            self._last_spec_ops = ops_ver

        dem_map = self._cached_spec_demand
        if not dem_map:
            return None

        if self._cached_model_ids is None or len(self._cached_model_ids[1]) != len(state.node_models):
            uniq_models = sorted(set(state.node_models))
            m2id = {m: idx for idx, m in enumerate(uniq_models)}
            id_arr = np.array([m2id[m] for m in state.node_models], dtype=np.int32)
            self._cached_model_ids = (uniq_models, id_arr)

        uniq_models, id_arr = self._cached_model_ids
        node_free = np.sum(state.gpu_slots, axis=1)
        tot_free = max(float(np.sum(node_free)), 1e-9)
        free_by_id = np.bincount(id_arr, weights=node_free, minlength=len(uniq_models))

        p_by_id = np.empty(len(uniq_models), dtype=np.float64)
        for idx, m in enumerate(uniq_models):
            free_share = float(free_by_id[idx]) / tot_free
            p_by_id[idx] = math.log1p(dem_map.get(m, 0.0) / max(free_share, 0.02))

        return p_by_id[id_arr[feasible_nodes]]

    def _rho_to_alpha(self, rho: float) -> float:
        """상관계수 추정치 rho를 연속 비연동 게이트 alpha in (0, 1)로 매핑한다."""
        z = (rho - self.adapt_cfg.rho_center) / max(self.adapt_cfg.rho_slope, 1e-4)
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(z))

    def _cv_to_gamma(self, cv: float) -> float:
        """완료 수명 변동계수(CV)를 헤비테일/린디 유효성 게이트 gamma in (0, 1)로 매핑한다."""
        z = (cv - self.adapt_cfg.cv_center) / max(self.adapt_cfg.cv_slope, 1e-4)
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _update_online_gate(self, state: ClusterState) -> Tuple[float, float, float]:
        """
        현재 클러스터 상태에서 온라인 Corr(gpu, age), 헤비테일 유효성 게이트(gamma),
        그리고 정규화 시간 척도를 O(1) 상각 비용으로 갱신한다.
        반환값: (alpha, gamma, ref_time_scale)
        """
        cfg = self.adapt_cfg
        ops_ver = state.ops_version
        # 콜드 스타트 구간에서 실측 시간 척도가 아직 0.0이고 실행 중 작업이 존재하면 즉시 척도를 획득
        need_scale_init = (self._cached_ref_scale <= 0.0 and len(state.active_jobs) >= 1)
        if not need_scale_init and (ops_ver - self._last_stats_ops < cfg.refresh_interval):
            return self._cached_alpha, self._cached_gamma, self._cached_ref_scale

        # 스케줄러 측에서 이미 주기를 제어하므로 중복 캐시 위상 지연을 막기 위해 refresh_interval=0 호출
        stats = state.get_online_temporal_stats(refresh_interval=0)

        # 동시 도착 버스트(t=0 등)로 아직 경과시간과 완료 표본이 모두 부족한 경우,
        # refresh_interval을 우회해 EWMA와 진단 이력을 반복 오염시키지 않도록 보수 캐시를 유지
        if need_scale_init and stats.mean_age <= 0.0 and stats.n_completed < 12:
            if ops_ver - self._last_stats_ops < cfg.refresh_interval:
                return self._cached_alpha, self._cached_gamma, self._cached_ref_scale

        self._last_stats_ops = ops_ver

        n_run = float(stats.n_running)
        n_comp = float(stats.n_completed)
        # 경과시간 평균·분산이 모두 양수일 때만 실행 중 작업의 상관계수를 유효 표본으로 반영
        has_run_corr = (n_run >= 6.0 and stats.mean_age > 0.0 and stats.age_cv > 1e-9)

        if cfg.corr_source == "lin_run":
            raw_rho = stats.corr_gpu_age
            eff_n = n_run if has_run_corr else 0.0
        elif cfg.corr_source == "log_run":
            raw_rho = stats.corr_log_gpu_age
            eff_n = n_run if has_run_corr else 0.0
        else:
            # 실행 중 작업의 로그 상관(즉각 반응)과 완료 작업 실측 상관(분산 안정화)의 가중 결합
            w_r = n_run if has_run_corr else 0.0
            w_c = cfg.comp_blend_weight * n_comp if n_comp >= 12.0 else 0.0
            eff_n = w_r + w_c
            if eff_n > 0.0:
                raw_rho = (w_r * stats.corr_log_gpu_age + w_c * stats.corr_completed_gpu_dur) / eff_n
            else:
                raw_rho = cfg.prior_rho

        # 콜드 스타트 Empirical Bayes 수축: 표본이 적을수록 보수 사전값(prior_rho) 유지
        shrunk_rho = (eff_n * raw_rho + cfg.prior_n * cfg.prior_rho) / (eff_n + cfg.prior_n)

        # EWMA 스무딩으로 단기 노이즈 억제 및 레짐 전환 시 부드러운 추종
        beta = cfg.ewma_beta
        self._smoothed_rho = (1.0 - beta) * self._smoothed_rho + beta * shrunk_rho
        self._cached_alpha = self._rho_to_alpha(self._smoothed_rho)

        # 실측 완료 수명 변동계수(CV) + 갱신이론 검사 역설(Inspection Paradox) 우측 중도절단 보정:
        # 정지 상태 갱신 과정에서 E[Age]/E[Dur] = (1 + CV^2)/2 이므로
        # 완료 버퍼에 장기 스트래글러가 아직 덜 들어온 초반 구간(우측 중도절단)에서도
        # CV_insp = sqrt(max(0, 2 * mean_age/mean_dur - 1))로 헤비테일을 즉각 포착하고,
        # 저분산 IFR(Uniform/결정적) 환경에서는 두 지표 모두 작아 gamma -> 0으로 안전 감쇠된다.
        if n_comp >= 12.0:
            cv_obs = stats.dur_cv
            if stats.mean_dur > 0.0 and stats.mean_age > 0.0:
                cv_insp = math.sqrt(max(0.0, 2.0 * (stats.mean_age / stats.mean_dur) - 1.0))
                cv_obs = max(cv_obs, cv_insp)
            shrunk_cv = (n_comp * cv_obs + cfg.prior_n * cfg.prior_cv) / (n_comp + cfg.prior_n)
            self._smoothed_cv = (1.0 - beta) * self._smoothed_cv + beta * shrunk_cv
        self._cached_gamma = self._cv_to_gamma(self._smoothed_cv)

        # 엄격한 시간 단위 불변 정규화 척도: 실측 완료 평균 수명 우선, 초기에는 실행 평균 경과시간 사용
        # 아직 양수 경과시간이 전혀 관측되지 않았으면 0.0 유지 (임의 상수 1.0 사용 금지)
        if stats.n_completed >= 12 and stats.mean_dur > 0.0:
            self._cached_ref_scale = stats.mean_dur
        elif stats.mean_age > 0.0:
            self._cached_ref_scale = stats.mean_age
        else:
            self._cached_ref_scale = 0.0

        if self.record_diagnostics:
            self.rho_history.append(self._smoothed_rho)
            self.alpha_history.append(self._cached_alpha)
            self.gamma_history.append(self._cached_gamma)

        return self._cached_alpha, self._cached_gamma, self._cached_ref_scale

    def _compute_spatial_base(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
        for_gang: bool = False,
        pristine_relax: float = 0.0,
        recycling_base_weight: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        모든 B-Pack 계열 및 공정 비교군(`spatial_base='bpack'`)이 공유하는 순수 공간·이기종·토폴로지 점수를 계산한다.
        반환값: `(score, is_pristine, breaks_4, g_cap, is_het_gpu_arr)` 형태 대신 `(score, is_pristine, breaks_4, g_cap)`
        """
        eps = 1e-9
        w_gpu, w_res, w_imb, w_pristine, w_b4, w_whole, _, _ = self.base_theta

        g_cap = state.node_gpu_cap[feasible_nodes]
        safe_g_cap = np.maximum(g_cap, 1.0)
        c_cap = np.maximum(state.node_cpu_cap[feasible_nodes], 1e-9)
        r_cap = np.maximum(state.node_ram_cap[feasible_nodes], 1e-9)
        is_het_gpu = self.hetero_aware and (not state.is_homogeneous_gpus)

        slots = state.gpu_slots[feasible_nodes]
        free_gpu_tot = np.sum(slots, axis=1)
        rem_g = np.where(g_cap > 0.0, (free_gpu_tot - gpu) / safe_g_cap, 0.0)
        rem_c = (state.cpu_free[feasible_nodes] - cpu) / c_cap
        rem_r = (state.ram_free[feasible_nodes] - ram) / r_cap

        # 1) 다차원 공간 패킹 및 자원 불균형 최소화
        rem_min = np.minimum(rem_g, np.minimum(rem_c, rem_r))
        rem_max = np.maximum(rem_g, np.maximum(rem_c, rem_r))
        score = (
            w_gpu * rem_g
            + w_res * (0.5 * rem_c + 0.5 * rem_r + 0.5 * rem_min)
            + w_imb * (rem_max - rem_min)
        )

        # 2) 이산 GPU 경계(풀노드 순수 노드 / 4-GPU 버디 / 1.0 물리 GPU) 보호
        empty_before = np.sum(slots >= 1.0 - eps, axis=1)
        is_pristine = (state.node_job_count[feasible_nodes] == 0)
        eff_pristine_w = w_pristine * (1.0 - pristine_relax)
        if is_het_gpu:
            pristine_scale = g_cap / max(state.max_gpu_cap, 1.0)
            score += eff_pristine_w * pristine_scale * (is_pristine & (gpu < g_cap - eps))
        else:
            score += eff_pristine_w * (is_pristine & (gpu < g_cap - eps))

        # 2-b) 무제약 파드 배치 시 희소 GPU 기종의 실시간 가용 여력 보호 (유계 상한으로 HoL 역전 차단)
        if self.hetero_aware and state.node_models is not None and not state.current_gpu_spec:
            spec_pressure = self._get_spec_pressure_vector(state, feasible_nodes)
            if spec_pressure is not None:
                if recycling_base_weight >= 1.0 - 1e-9:
                    score += 0.55 * np.minimum(spec_pressure, 1.5)
                else:
                    free_ratio = np.clip(free_gpu_tot / safe_g_cap, 0.0, 1.0)
                    recyc_scale = recycling_base_weight + (1.0 - recycling_base_weight) * free_ratio
                    score += 0.55 * recyc_scale * np.minimum(spec_pressure, 1.5)

        # 2-c) 멀티노드 토폴로지 인지: 멀티노드 수요 감지 시 단일 노드 잡의 순수 랙 개봉 억제 (Rack-Preservation)
        if self.topology_aware and (not for_gang) and state.num_racks > 1 and state.recent_node_counts:
            n_nc = len(state.recent_node_counts)
            p_multi = sum(1 for nc in state.recent_node_counts if nc > 1) / n_nc
            if p_multi > 0.0:
                p_mn_eff = min(1.0, 4.0 * p_multi)
                is_free_node = (state.node_job_count == 0).astype(np.float64)
                rack_tot = np.bincount(state.node_rack_ids, minlength=state.num_racks).astype(np.float64)
                rack_free = np.bincount(state.node_rack_ids, weights=is_free_node, minlength=state.num_racks)
                rack_free_ratio = rack_free / np.maximum(rack_tot, 1.0)
                node_racks = state.node_rack_ids[feasible_nodes]
                score += 1.2 * p_mn_eff * rack_free_ratio[node_racks] * is_pristine.astype(np.float64)

        if gpu < 1.0 - eps:
            has_partial_fit = np.any((slots < 1.0 - eps) & (slots + eps >= gpu), axis=1)
            breaks_whole = ~has_partial_fit
            empty_after = empty_before - breaks_whole
            score += w_whole * breaks_whole
        else:
            empty_after = empty_before - int(round(gpu))

        breaks_4 = (empty_before >= 4) & (empty_after < 4)
        score += w_b4 * breaks_4
        return score, is_pristine, breaks_4, g_cap

    def _compute_scores(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
        for_gang: bool = False,
    ) -> np.ndarray:
        eps = 1e-9
        acfg = self.adapt_cfg
        w_lindy = float(self.base_theta[6])

        alpha, gamma, ref_scale = self._update_online_gate(state)
        score, is_pristine, breaks_4, g_cap = self._compute_spatial_base(
            state, feasible_nodes, gpu, cpu, ram, for_gang=for_gang
        )
        is_het_gpu = self.hetero_aware and (not state.is_homogeneous_gpus)

        # 3) 온라인 Corr(gpu, age) + CV 이중 게이트 적응형 린디-에이지 포텐셜 (O(1) 벡터화, 시간 단위 불변)
        if w_lindy > 1e-6 and ref_scale > 0.0 and gamma > 1e-4:
            max_age = np.where(
                is_pristine,
                0.0,
                np.maximum(0.0, state.now - state.node_oldest_start[feasible_nodes]),
            )
            norm_age = max_age / ref_scale

            # 극단 영역 고속 분기(Fast-Path): 불필요한 로그/분수 중복 연산 제거
            if alpha <= 0.01:
                phi_adapt = (norm_age / (norm_age + acfg.tau_safe_norm)) * (~breaks_4)
            elif alpha >= 0.99:
                phi_adapt = acfg.lindy_log_scale * np.minimum(
                    np.log1p(norm_age / acfg.tau_log_norm),
                    acfg.max_log_potential,
                )
            else:
                # 보수 모드(alpha -> 0): 4-GPU 경계 보호형 유계 린디 (W1 4-GPU 경계 역전 원천 차단)
                phi_safe = (norm_age / (norm_age + acfg.tau_safe_norm)) * (~breaks_4)
                # 격리 모드(alpha -> 1): 장기 스트래글러 집중 격리용 척도 불변 로그 린디 (W3 돌파)
                phi_quarantine = acfg.lindy_log_scale * np.minimum(
                    np.log1p(norm_age / acfg.tau_log_norm),
                    acfg.max_log_potential,
                )
                phi_adapt = (1.0 - alpha) * phi_safe + alpha * phi_quarantine

            if is_het_gpu and gpu <= 1.0 + eps:
                # 이기종 클러스터에서 대형(8-GPU) 노드의 Max-of-N 린디 블랙홀 편향을 부드럽게 감쇠
                min_g = max(state.min_gpu_cap, 1.0)
                big_node_lindy_damp = 1.0 / (1.0 + 0.5 * np.log2(np.maximum(1.0, g_cap / min_g)))
                phi_adapt = phi_adapt * big_node_lindy_damp

            score -= (w_lindy * gamma) * phi_adapt * (~is_pristine)

        return score

    def score_nodes(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
    ) -> np.ndarray:
        return self._compute_scores(state, feasible_nodes, gpu, cpu, ram, for_gang=False)

    def select_node(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
    ) -> int:
        scores = self._compute_scores(state, feasible_nodes, gpu, cpu, ram, for_gang=False)
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
        scores = self._compute_scores(state, feasible_nodes, gpu_per_node, cpu_per_node, ram_per_node, for_gang=True)
        if not self.topology_aware:
            order = np.argsort(scores, kind="mergesort")
            return [int(feasible_nodes[i]) for i in order[:num_nodes]]
        return select_rack_aware_gang_nodes(state, feasible_nodes, scores, num_nodes)

    def get_diagnostics_summary(self) -> Dict[str, float]:
        """기록된 온라인 상관(rho), 게이트(alpha), 린디 유효성(gamma) 궤적 요약을 반환한다."""
        if not self.rho_history:
            return {
                "rho_mean": float(self._smoothed_rho),
                "alpha_mean": float(self._cached_alpha),
                "gamma_mean": float(self._cached_gamma),
            }
        rhos = np.asarray(self.rho_history, dtype=np.float64)
        alphas = np.asarray(self.alpha_history, dtype=np.float64)
        gammas = np.asarray(self.gamma_history, dtype=np.float64)
        return {
            "rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos)),
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
            "gamma_mean": float(np.mean(gammas)),
            "gamma_min": float(np.min(gammas)),
        }


class BPackSpatialOnlyScheduler(AdaptiveBLindyPackScheduler):
    """
    수명/해저드 항(`w_lindy=0`)을 완전히 끄고 순수 B-Pack 공간·경계·이기종·토폴로지 항만 사용하는 비교군.
    빅테크 시간 인지 기법(`NILAS`, `LAVA`, `DVLA`)과의 공정한 팩토리얼(Factorial) 대조 기준선 역할을 한다.
    """

    def __init__(
        self,
        base_theta: Optional[np.ndarray] = None,
        name: str = "BPack-SpatialOnly",
        hetero_aware: bool = True,
        topology_aware: bool = True,
    ) -> None:
        theta = np.array(DEFAULT_BASE_THETA if base_theta is None else base_theta, dtype=np.float64)
        theta[6] = 0.0
        super().__init__(
            base_theta=theta,
            name=name,
            record_diagnostics=False,
            hetero_aware=hetero_aware,
            topology_aware=topology_aware,
        )

    def _compute_scores(
        self,
        state: ClusterState,
        feasible_nodes: np.ndarray,
        gpu: float,
        cpu: float,
        ram: float,
        for_gang: bool = False,
    ) -> np.ndarray:
        score, _, _, _ = self._compute_spatial_base(
            state, feasible_nodes, gpu, cpu, ram, for_gang=for_gang
        )
        return score
