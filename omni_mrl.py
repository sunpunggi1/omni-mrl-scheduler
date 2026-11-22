"""
8~11단계 통합 스케줄러: `Omni-MRL` (v8.0 -> v11.0)

Google MLSys '25 `NILAS`/`LAVA`(경험적 조건부 잔여수명 MRL 기반 호스트 연장 비용 최소화) 및
Alibaba OSDI `DVLA` 통찰을 기존 `Omni-Lindy` (v7.0) 위에 통합 발전시킨
**양방향 위험률 적응형(Bidirectional Hazard-Adaptive) Non-Clairvoyant 휴리스틱 스케줄러**.

핵심 설계 및 정직한 범위:
1. 양방향 위험률 스펙트럼 단일 통합 (DFR + Memoryless + IFR):
   - 헤비테일/DFR (CV >> 0.7): `Omni-Lindy` 이중 게이트(상관 alpha + DFR CV gamma)로
     장기 스트래글러 노드에 집중 격리 (실전 범용 AI 트레이스 주력 성능 보존).
   - 저분산/고정에폭 IFR (CV < 0.72): `gamma_IFR` 게이트가 열리며 동일 진입 코호트
     Kaplan-Meier 조건부 잔여수명(MRL) 기반 호스트 점유 연장 비용(Delta T)을 작동시켜,
     곧 비워질 오래된 IFR 노드는 회수하고 젊은 IFR 노드에 신규 작업을 동기화 패킹한다.
2. 이기종·토폴로지 통합 보호 (`Spec-Scarcity` + `Rack-Preservation`):
   참고: `recycling_base_weight < 1.0`으로 희소 기종 노드의 가용률 비례 감쇠를 적용하는 변형은
   Alibaba `GPUSpec33`에서 Backfill 대기 악화(49.05h -> 373.74h)를 유발한 탐색적 Negative Result로
   기록되며, 기본값(`1.0`)은 균일한 Spec-Scarcity 보호를 유지한다.
3. 100% Non-Clairvoyant & 실증된 시간 단위 불변성(Clock-Unit Invariance):
   외부 ML 수명 예측기 없이 오직 동일 진입 코호트(`recent_cohort`)의 완료·중도절단 표본 기반
   무차원 Kaplan-Meier MRL과 실시간 `age`만 사용하며, 실측 시간 배율 변환에서도 결정 시퀀스가 불변이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from adaptive_lindy import AdaptiveBLindyPackScheduler, AdaptiveLindyConfig
from bigtech_baselines import EmpiricalSurvivalModel
from simulator import ClusterConfig, ClusterState


@dataclass(frozen=True)
class OmniMRLConfig:
    """`Omni-MRL` (v8.0)의 IFR 동기화 종료 및 Recycling 적응 파라미터."""

    cv_ifr_center: float = 0.65       # IFR 동기화 종료 게이트 중심 변동계수 (gamma_dfr 대칭점)
    cv_ifr_slope: float = 0.06        # gamma_IFR 시그모이드 전이 폭 (CV >= 0.83에서 정확히 0.0 차단)
    ifr_deadzone: float = 0.05        # DFR/메모리리스 환경의 미세 잔차(0.03 등) 완전 소거용 데드존
    w_nilas: float = 1.45             # IFR 환경에서 곧 비워질 노드 보호(호스트 연장 비용) 가중치
    pristine_ifr_relax: float = 0.35  # 순수 IFR 환경에서 빈 노드와 젊은 노드의 균형 완화 계수
    # 주의: 이기종 GPUSpec33 탐색 실험에서 부분 점유 희소 노드에 0.25 감쇠(Spec-Penalty Scaling)를
    # 적용하면 Backfill Wait가 49.05h -> 373.74h로 악화되는 Negative Result가 확인되었으므로,
    # 기본값은 균일 모델 보호(1.0)를 동결 유지한다. (0.25 미만/이하 실험은 ablation 전용)
    recycling_base_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.cv_ifr_slope <= 0.0:
            raise ValueError("cv_ifr_slope는 양수여야 합니다.")
        if self.w_nilas < 0.0:
            raise ValueError("w_nilas는 0 이상이어야 합니다.")
        if not (0.0 <= self.pristine_ifr_relax <= 1.0):
            raise ValueError("pristine_ifr_relax는 [0, 1] 범위여야 합니다.")
        if not (0.0 <= self.recycling_base_weight <= 1.0):
            raise ValueError("recycling_base_weight는 [0, 1] 범위여야 합니다.")


class OmniMRLScheduler(AdaptiveBLindyPackScheduler):
    """
    `Omni-Lindy` (v7.0) + `EmpiricalSurvivalModel` MRL 동기화 종료 + LAVA Recycling을 결합한
    차세대 무예측 통합 AI GPU 스케줄러 (`Omni-MRL` v8.0).
    """

    def __init__(
        self,
        base_theta: Optional[np.ndarray] = None,
        adapt_cfg: Optional[AdaptiveLindyConfig] = None,
        mrl_cfg: Optional[OmniMRLConfig] = None,
        name: str = "Omni-MRL-v8",
        record_diagnostics: bool = False,
        hetero_aware: bool = True,
        topology_aware: bool = True,
    ) -> None:
        super().__init__(
            base_theta=base_theta,
            adapt_cfg=adapt_cfg,
            name=name,
            record_diagnostics=record_diagnostics,
            hetero_aware=hetero_aware,
            topology_aware=topology_aware,
        )
        self.mrl_cfg = OmniMRLConfig() if mrl_cfg is None else mrl_cfg
        self.survival = EmpiricalSurvivalModel(
            refresh_interval=self.adapt_cfg.refresh_interval
        )
        self._cached_gamma_ifr: float = 0.0
        self._last_mrl_ops: int = -10**9
        self._diag_steps: int = 0
        self._gamma_ifr_sum: float = 0.0
        self._gamma_ifr_active_steps: int = 0
        self._gamma_dfr_sum: float = 0.0
        self._alpha_sum: float = 0.0

    def reset(self, config: ClusterConfig) -> None:
        super().reset(config)
        self.survival.reset()
        self._cached_gamma_ifr = 0.0
        self._last_mrl_ops = -10**9
        self._diag_steps = 0
        self._gamma_ifr_sum = 0.0
        self._gamma_ifr_active_steps = 0
        self._gamma_dfr_sum = 0.0
        self._alpha_sum = 0.0

    def _cv_to_gamma_ifr(self, cv: float) -> float:
        """변동계수(CV)가 낮을수록(IFR/고정에폭) 1에 가까워지는 게이트(CV >= 0.83 시 정확히 0.0)를 계산한다."""
        z = (self.mrl_cfg.cv_ifr_center - cv) / max(self.mrl_cfg.cv_ifr_slope, 1e-4)
        z = max(-30.0, min(30.0, z))
        raw = 1.0 / (1.0 + math.exp(-z))
        dz = self.mrl_cfg.ifr_deadzone
        if raw <= dz:
            return 0.0
        return (raw - dz) / max(1.0 - dz, 1e-9)

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
        mcfg = self.mrl_cfg
        w_gpu, w_res, w_imb, w_pristine, w_b4, w_whole, w_lindy, _ = self.base_theta

        alpha, gamma_dfr, ref_scale = self._update_online_gate(state)
        n_comp = len(state.recent_completions)

        # IFR 게이트 및 경험적 생존 모델 갱신 (완료 표본 12건 이상 관측 전에는 gamma_IFR = 0.0 유지)
        if n_comp >= 12 and ref_scale > 0.0:
            self._cached_gamma_ifr = self._cv_to_gamma_ifr(self._smoothed_cv)
            if self._cached_gamma_ifr > 1e-3 or (state.ops_version - self._last_mrl_ops >= acfg.refresh_interval):
                self.survival.update(state, force=(state.ops_version - self._last_mrl_ops >= acfg.refresh_interval))
                self._last_mrl_ops = state.ops_version
        else:
            self._cached_gamma_ifr = 0.0

        gamma_ifr = self._cached_gamma_ifr
        self._diag_steps += 1
        self._gamma_ifr_sum += gamma_ifr
        if gamma_ifr > 1e-3:
            self._gamma_ifr_active_steps += 1
        self._gamma_dfr_sum += gamma_dfr
        self._alpha_sum += alpha

        score, is_pristine, breaks_4, g_cap = self._compute_spatial_base(
            state,
            feasible_nodes,
            gpu,
            cpu,
            ram,
            for_gang=for_gang,
            pristine_relax=mcfg.pristine_ifr_relax * gamma_ifr,
            recycling_base_weight=mcfg.recycling_base_weight,
        )
        is_het_gpu = self.hetero_aware and (not state.is_homogeneous_gpus)

        # 3-a) DFR/헤비테일 린디-에이지 격리 포텐셜 (O(1) 벡터화, 시간 단위 불변)
        if w_lindy > 1e-6 and ref_scale > 0.0 and gamma_dfr > 1e-4:
            max_age = np.where(
                is_pristine,
                0.0,
                np.maximum(0.0, state.now - state.node_oldest_start[feasible_nodes]),
            )
            norm_age = max_age / ref_scale
            if alpha <= 0.01:
                phi_adapt = (norm_age / (norm_age + acfg.tau_safe_norm)) * (~breaks_4)
            elif alpha >= 0.99:
                phi_adapt = acfg.lindy_log_scale * np.minimum(
                    np.log1p(norm_age / acfg.tau_log_norm),
                    acfg.max_log_potential,
                )
            else:
                phi_safe = (norm_age / (norm_age + acfg.tau_safe_norm)) * (~breaks_4)
                phi_quarantine = acfg.lindy_log_scale * np.minimum(
                    np.log1p(norm_age / acfg.tau_log_norm),
                    acfg.max_log_potential,
                )
                phi_adapt = (1.0 - alpha) * phi_safe + alpha * phi_quarantine

            if is_het_gpu and gpu <= 1.0 + eps:
                min_g = max(state.min_gpu_cap, 1.0)
                big_node_lindy_damp = 1.0 / (1.0 + 0.5 * np.log2(np.maximum(1.0, g_cap / min_g)))
                phi_adapt = phi_adapt * big_node_lindy_damp

            score -= (w_lindy * gamma_dfr) * phi_adapt * (~is_pristine)

        # 3-b) IFR/고정에폭 NILAS 스타일 동기화 종료 항 (Kaplan-Meier MRL 기반 호스트 연장 비용)
        if mcfg.w_nilas > 1e-6 and gamma_ifr > 1e-3 and ref_scale > 0.0 and len(self.survival.sorted_norm_durs) >= 12:
            min_age = np.where(
                is_pristine,
                0.0,
                np.maximum(0.0, state.now - state.node_newest_start[feasible_nodes]),
            )
            u_min = min_age / ref_scale
            e_host_ifr = self.survival.predict_mrl(u_min)
            d_new = max(1e-6, self.survival.predict_new_duration(gpu, require_corr=True))
            delta_t = np.maximum(0.0, d_new - e_host_ifr) / d_new
            # 연장 비용이 작을수록(e_host_ifr >= d_new) 유리; 유휴 노드는 본 항에서 제외
            score += (mcfg.w_nilas * gamma_ifr) * np.minimum(delta_t, 1.0) * (~is_pristine)

        return score

    def get_diagnostics_summary(self) -> Dict[str, float]:
        """온라인 상관(rho), 게이트(alpha, gamma_dfr, gamma_ifr) 및 시간 평균 요약을 반환한다."""
        base = super().get_diagnostics_summary()
        steps = max(1, self._diag_steps)
        base["gamma_ifr"] = float(self._cached_gamma_ifr)
        base["gamma_ifr_mean"] = float(self._gamma_ifr_sum / steps) if self._diag_steps > 0 else float(self._cached_gamma_ifr)
        base["gamma_ifr_active_frac"] = float(self._gamma_ifr_active_steps / steps) if self._diag_steps > 0 else 0.0
        base["gamma_dfr_mean"] = float(self._gamma_dfr_sum / steps) if self._diag_steps > 0 else float(self._cached_gamma)
        base["alpha_mean_step"] = float(self._alpha_sum / steps) if self._diag_steps > 0 else float(self._cached_alpha)
        base["diag_steps"] = float(self._diag_steps)
        return base
