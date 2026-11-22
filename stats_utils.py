"""
엄밀 통계 검정 유틸리티 (순수 Python + NumPy, 외부 SciPy 비의존)

30-시드 쌍체 비교(Paired Comparison)에서 정규 근사에 의존하지 않고
정확한(Exact) p-value와 신뢰구간을 산출한다:
1. exact_student_t_pvalue  : 정규화 불완전 베타 함수(Lentz 연분수) 기반 정확 양측 Student-t p-value
2. wilcoxon_signed_rank    : 비모수 윌콕슨 부호순위 검정 (동률 보정 + 소표본 정확/대표본 연속성 보정)
3. exact_sign_test_pvalue  : 이항 분포 기반 정확 양측 부호 검정(Win-Rate 유의성)
4. paired_summary_stats    : 평균 개선율, 승률, 95% t-신뢰구간, t-검정 및 Wilcoxon p-value 일괄 요약
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def _betacf(a: float, b: float, x: float) -> float:
    """정규화 불완전 베타 함수 I_x(a, b) 계산을 위한 수정 Lentz 연분수 (Numerical Recipes 표준)."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # 짝수 항
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        # 홀수 항
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """정규화 불완전 베타 함수 I_x(a, b)를 1e-12 정밀도로 계산한다."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def exact_student_t_pvalue(t_stat: float, df: int) -> float:
    """자유도 df에 대한 양측(two-sided) Student-t 분포의 정확한 p-value를 반환한다."""
    if math.isnan(t_stat):
        raise ValueError("t_stat은 NaN일 수 없습니다.")
    if df <= 0:
        return 1.0
    if math.isinf(t_stat):
        return 0.0
    x = df / (df + t_stat * t_stat)
    return float(min(1.0, max(0.0, regularized_incomplete_beta(x, 0.5 * df, 0.5))))


def student_t_critical_95(df: int) -> float:
    """자유도 df에 대한 양측 95% 신뢰구간 임계값 t_{0.025, df}를 이분 탐색으로 정확히 구한다."""
    if df <= 0:
        return 1.96
    lo, hi = 1.95, 15.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if exact_student_t_pvalue(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_t_test_exact(diffs: np.ndarray) -> Tuple[float, float]:
    """
    쌍체 차이 배열(base - cand, 양수면 후보 알고리즘 승리)에 대해
    정확한 Student-t 통계량과 양측 p-value를 반환한다.
    """
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"paired_t_test_exact 입력은 1차원 배열이어야 합니다: shape={arr.shape}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("paired_t_test_exact 입력에 NaN 또는 Inf가 포함되어 있습니다.")
    n = len(arr)
    if n < 2:
        return 0.0, 1.0
    mean_d = float(np.mean(arr))
    std_d = float(np.std(arr, ddof=1))
    if std_d <= 1e-15:
        if abs(mean_d) <= 1e-15:
            return 0.0, 1.0
        return (float("inf") if mean_d > 0 else float("-inf")), 0.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    p_val = exact_student_t_pvalue(t_stat, n - 1)
    return float(t_stat), float(p_val)


def wilcoxon_signed_rank_test(diffs: np.ndarray) -> Tuple[float, float]:
    """
    비모수 윌콕슨 부호순위 검정(Wilcoxon Signed-Rank Test, 양측).
    정규성 가정이 깨지는 헤비테일 워크로드에서 강건한 통계 유의성 검증에 사용한다.
    반환값: (W_stat, two_sided_p_value)
    """
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"wilcoxon_signed_rank_test 입력은 1차원 배열이어야 합니다: shape={arr.shape}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("wilcoxon_signed_rank_test 입력에 NaN 또는 Inf가 포함되어 있습니다.")
    nz = arr[np.abs(arr) > 1e-15]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0

    abs_vals = np.abs(nz)
    order = np.argsort(abs_vals)
    sorted_abs = abs_vals[order]

    # 동률 평균 순위(average ranks) 계산 및 동률 보정항 누적
    ranks = np.empty(n, dtype=np.float64)
    tie_correction = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(sorted_abs[j + 1] - sorted_abs[i]) <= 1e-15:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        t_cnt = j - i + 1
        if t_cnt > 1:
            tie_correction += t_cnt * (t_cnt * t_cnt - 1)
        i = j + 1

    w_plus = float(np.sum(ranks[nz > 0]))
    w_minus = float(np.sum(ranks[nz < 0]))
    w_stat = min(w_plus, w_minus)

    # 동률이 없고 표본이 작으면(n <= 20) 정확 분포 동적계획법 계산
    if tie_correction == 0.0 and n <= 20:
        max_s = n * (n + 1) // 2
        dp = [0] * (max_s + 1)
        dp[0] = 1
        for k in range(1, n + 1):
            for s in range(max_s, k - 1, -1):
                dp[s] += dp[s - k]
        w_int = int(round(w_stat))
        tail = sum(dp[: w_int + 1])
        p_val = min(1.0, 2.0 * tail / (1 << n))
        return w_stat, float(p_val)

    # 대표본(n > 20) 또는 동률 존재 시 연속성·동률 보정 정규 근사
    mean_w = n * (n + 1) / 4.0
    var_w = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_correction) / 24.0
    if var_w <= 1e-15:
        return w_stat, 1.0
    z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
    z = max(0.0, z)
    p_val = math.erfc(z / math.sqrt(2.0))
    return w_stat, float(min(1.0, max(0.0, p_val)))


def exact_sign_test_pvalue(wins: int, losses: int) -> float:
    """동률을 제외한 (승, 패)에 대한 정확 양측 이항 부호 검정 p-value (대표본 오버플로 안전)."""
    if wins < 0 or losses < 0:
        raise ValueError(f"승/패 횟수는 0 이상이어야 합니다: wins={wins}, losses={losses}")
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    if n <= 2000:
        # 임의 정밀도 정수 나눗셈(tail / (1 << n))으로 부동소수점 2**n 오버플로 없이 정확 계산
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    else:
        # 이항 누적분포 항등식 P(Bin(n, 0.5) <= k) = I_{0.5}(n - k, k + 1)로 O(1) 정확 계산
        tail = regularized_incomplete_beta(0.5, float(n - k), float(k + 1))
    return float(min(1.0, max(0.0, 2.0 * tail)))


def holm_bonferroni_adjust(p_values: List[float] | np.ndarray) -> List[float]:
    """다중검정 오류(FWER) 억제를 위한 step-down Holm-Bonferroni 보정 p-value 리스트를 반환한다."""
    arr = np.asarray(p_values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"p_values는 1차원 시퀀스여야 합니다: shape={arr.shape}")
    m = len(arr)
    if m == 0:
        return []
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("모든 p-value는 [0, 1] 범위의 유한 실수여야 합니다.")

    order = np.argsort(arr, kind="mergesort")
    adj_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * float(arr[idx])
        running_max = min(1.0, max(running_max, scaled))
        adj_sorted[rank] = running_max

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adj_sorted
    return [float(v) for v in adjusted]


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """결정적 비모수 쌍체 Bootstrap 백분위 신뢰구간 `(low, high)`를 반환한다."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("diffs는 비어 있지 않은 1차원 배열이어야 합니다.")
    if not np.all(np.isfinite(d)):
        raise ValueError("diffs에 NaN 또는 Inf가 포함되어 있습니다.")
    n = len(d)
    if n == 1 or float(np.std(d)) <= 1e-15:
        val = float(np.mean(d))
        return val, val

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = np.mean(d[idx], axis=1)
    lo = float(np.quantile(boot_means, 0.5 * alpha))
    hi = float(np.quantile(boot_means, 1.0 - 0.5 * alpha))
    return lo, hi


def paired_summary_stats(base: np.ndarray, cand: np.ndarray) -> Dict[str, float]:
    """
    낮을수록 좋은 지표(거절률, JCT, Wait, Makespan 등)에 대해
    base 대비 cand의 쌍체 통계 요약 딕셔너리를 반환한다.
    """
    b = np.asarray(base, dtype=np.float64)
    c = np.asarray(cand, dtype=np.float64)
    if b.ndim != 1 or c.ndim != 1 or b.shape != c.shape:
        raise ValueError(f"base와 cand는 동일한 길이의 1차원 배열이어야 합니다: {b.shape} vs {c.shape}")
    if b.size == 0:
        raise ValueError("빈 배열에 대해서는 쌍체 통계 요약을 계산할 수 없습니다.")
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        raise ValueError("base 또는 cand에 NaN/Inf가 포함되어 있습니다.")
    diffs = b - c  # 양수면 cand가 우수
    n = len(diffs)

    mean_b = float(np.mean(b))
    mean_c = float(np.mean(c))
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / math.sqrt(n) if n > 1 else 0.0
    t_crit = student_t_critical_95(n - 1) if n > 1 else 0.0
    cohens_dz = (mean_diff / std_diff) if std_diff > 1e-15 else 0.0
    boot_lo, boot_hi = paired_bootstrap_ci(diffs)

    rel_imp_pct = ((mean_b - mean_c) / max(abs(mean_b), 1e-12)) * 100.0
    wins = int(np.sum(diffs > 1e-12))
    losses = int(np.sum(diffs < -1e-12))
    ties = n - wins - losses

    t_stat, p_t = paired_t_test_exact(diffs)
    w_stat, p_wilcoxon = wilcoxon_signed_rank_test(diffs)
    p_sign = exact_sign_test_pvalue(wins, losses)

    return {
        "n": n,
        "base_mean": mean_b,
        "base_std": float(np.std(b, ddof=1)) if n > 1 else 0.0,
        "cand_mean": mean_c,
        "cand_std": float(np.std(c, ddof=1)) if n > 1 else 0.0,
        "diff_mean": mean_diff,
        "diff_ci95_low": mean_diff - t_crit * se_diff,
        "diff_ci95_high": mean_diff + t_crit * se_diff,
        "boot_ci95_low": boot_lo,
        "boot_ci95_high": boot_hi,
        "cohens_dz": float(cohens_dz),
        "rel_improvement_pct": float(rel_imp_pct),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "t_stat": t_stat,
        "p_value_t": p_t,
        "wilcoxon_w": w_stat,
        "p_value_wilcoxon": p_wilcoxon,
        "p_value_sign": p_sign,
    }


def hierarchical_cluster_bootstrap_ci(
    cluster_diffs: List[np.ndarray],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    클러스터 계층 구조(예: Helios 4개 클러스터 × 클러스터별 다중 시간 윈도우)를 반영한
    2단계 비모수 계층 부트스트랩(Two-Stage Cluster Bootstrap) 95% 신뢰구간을 반환한다.
    1단계에서 클러스터를 복원 추출하고, 2단계에서 각 추출된 클러스터 내부 윈도우를 복원 추출한다.
    """
    if not cluster_diffs:
        raise ValueError("cluster_diffs는 최소 1개 이상의 클러스터 배열을 포함해야 합니다.")
    cleaned: List[np.ndarray] = []
    for arr in cluster_diffs:
        d = np.asarray(arr, dtype=np.float64)
        if d.ndim != 1 or d.size == 0:
            raise ValueError("각 클러스터 차이 배열은 비어 있지 않은 1차원 배열이어야 합니다.")
        if not np.all(np.isfinite(d)):
            raise ValueError("클러스터 차이 배열에 NaN 또는 Inf가 포함되어 있습니다.")
        cleaned.append(d)

    c_num = len(cleaned)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        c_idx = rng.integers(0, c_num, size=c_num)
        sampled_vals: List[np.ndarray] = []
        for ci in c_idx:
            grp = cleaned[int(ci)]
            w_idx = rng.integers(0, len(grp), size=len(grp))
            sampled_vals.append(grp[w_idx])
        boot_means[b] = float(np.mean(np.concatenate(sampled_vals)))

    lo = float(np.quantile(boot_means, 0.5 * alpha))
    hi = float(np.quantile(boot_means, 1.0 - 0.5 * alpha))
    return lo, hi


def cluster_aware_summary_stats(
    base_clusters: List[np.ndarray],
    cand_clusters: List[np.ndarray],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """
    클러스터 내 시간 윈도우 상관을 반영하기 위해:
    1) 클러스터 평균 단위(`n = len(base_clusters)`) 쌍체 통계 요약
    2) 전체 윈도우 풀링 평균 및 2단계 클러스터 계층 부트스트랩 95% 신뢰구간
    을 함께 산출해 반환한다. (각 클러스터 내 시간 블록 연속 재표본추출은 사용하지 않음)
    """
    if len(base_clusters) != len(cand_clusters) or not base_clusters:
        raise ValueError("base_clusters와 cand_clusters는 비어 있지 않고 길이가 같아야 합니다.")

    base_means = np.array([float(np.mean(np.asarray(g, dtype=np.float64))) for g in base_clusters], dtype=np.float64)
    cand_means = np.array([float(np.mean(np.asarray(g, dtype=np.float64))) for g in cand_clusters], dtype=np.float64)
    cluster_level = paired_summary_stats(base_means, cand_means)

    diff_groups = [
        np.asarray(b, dtype=np.float64) - np.asarray(c, dtype=np.float64)
        for b, c in zip(base_clusters, cand_clusters)
    ]
    hier_lo, hier_hi = hierarchical_cluster_bootstrap_ci(diff_groups, n_boot=n_boot, seed=seed)

    flat_base = np.concatenate([np.asarray(b, dtype=np.float64) for b in base_clusters])
    flat_cand = np.concatenate([np.asarray(c, dtype=np.float64) for c in cand_clusters])
    window_level = paired_summary_stats(flat_base, flat_cand)
    window_level["hier_boot_ci95_low"] = hier_lo
    window_level["hier_boot_ci95_high"] = hier_hi

    return {
        "n_clusters": len(base_clusters),
        "n_windows": int(len(flat_base)),
        "cluster_level_stats": cluster_level,
        "window_level_with_hier_ci": window_level,
    }
