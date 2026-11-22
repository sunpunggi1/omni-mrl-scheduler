"""
6단계(B안): 실제 공개 기업 GPU 클러스터 트레이스 파서 및 감사 모듈

지원 데이터셋:
1. Alibaba cluster-trace-gpu-v2023 (openb_node_list_gpu_node.csv + openb_pod_list_*.csv)
   - 이기종 GPU 기종(G2, T4, P100, V100M16, G3, V100M32, A10), 노드당 GPU 수(1/2/4/8) 및
     분수 GPU(gpu_milli), gpu_spec 제약 파싱
2. SenseTime HeliosData (Venus, Earth, Saturn, Uranus cluster_log.csv)
   - 336만여 개 실제 Slurm 작업 로그 파싱, 멀티노드(>8 GPU 또는 node_num>1)·CPU-only 제외 감사,
     비중첩 실제 피크 구간 추출
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator import ClusterConfig, Job, _safe_pearson


@dataclass
class TraceAudit:
    """실제 트레이스 파싱 과정의 필터링·물리 특성 감사 요약."""
    source_name: str
    raw_rows: int
    valid_gpu_jobs: int
    excluded_cpu_only: int
    excluded_bad_time: int
    excluded_multinode: int
    excluded_infeasible_node: int
    corr_log_gpu_dur: float
    dur_cv: float
    dur_mean_hours: float
    dur_median_hours: float
    dur_p95_hours: float
    fractional_ratio: float
    multi_gpu_ratio: float
    gpu_bucket_counts: Dict[str, int]
    time_span_hours: float
    mean_concurrent_gpus: float
    peak_concurrent_gpus: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _compute_concurrency_and_stats(
    jobs: List[Job],
) -> Tuple[float, float, float, float, float, float, float, float, float, float, Dict[str, int]]:
    """작업 시퀀스의 시간 가중 동시 GPU 수요 및 수명·상관 통계를 계산한다."""
    if not jobs:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}

    gs = np.array([j.gpu for j in jobs], dtype=np.float64)
    ds = np.array([j.duration for j in jobs], dtype=np.float64)

    mean_d = max(float(np.mean(ds)), 1e-12)
    dur_cv = float(np.std(ds) / mean_d)
    corr_log = _safe_pearson(np.log1p(gs), np.log1p(ds / mean_d))

    frac_ratio = float(np.mean(gs < 1.0 - 1e-9))
    multi_ratio = float(np.mean(gs > 1.0 + 1e-9))

    buckets: Counter[str] = Counter()
    for g in gs:
        if g < 1.0 - 1e-9:
            buckets["frac(<1)"] += 1
        else:
            buckets[str(int(round(g)))] += 1

    # Sweep-line으로 정확한 동시 실행 GPU 수요 적분
    events: List[Tuple[float, float]] = []
    for j in jobs:
        events.append((j.arrival_time, j.gpu))
        events.append((j.arrival_time + j.duration, -j.gpu))
    events.sort(key=lambda x: (x[0], x[1]))

    cur_gpu = 0.0
    peak_gpu = 0.0
    gpu_time_acc = 0.0
    t_start = events[0][0]
    prev_t = t_start
    for t, delta in events:
        if t > prev_t:
            gpu_time_acc += cur_gpu * (t - prev_t)
            prev_t = t
        cur_gpu = max(0.0, cur_gpu + delta)
        if cur_gpu > peak_gpu:
            peak_gpu = cur_gpu

    span = max(prev_t - t_start, 1e-9)
    mean_conc = gpu_time_acc / span
    return (
        corr_log,
        dur_cv,
        float(np.mean(ds)),
        float(np.median(ds)),
        float(np.percentile(ds, 95)),
        frac_ratio,
        multi_ratio,
        span,
        mean_conc,
        peak_gpu,
        dict(sorted(buckets.items())),
    )


def _node_can_host(
    job: Job,
    node_g: Tuple[int, ...],
    node_c: Tuple[float, ...],
    node_r: Tuple[float, ...],
    node_m: Optional[Tuple[str, ...]],
) -> bool:
    """해당 작업이 클러스터 내 최소 1개 노드에 단독 배치 가능한지 확인한다."""
    eps = 1e-9
    allowed = (
        {tok.strip() for tok in job.gpu_spec.split("|") if tok.strip()}
        if (job.gpu_spec and node_m is not None)
        else None
    )
    is_int_gpu = job.gpu >= 1.0 - eps
    req_g_int = int(round(job.gpu)) if is_int_gpu else 0
    for i in range(len(node_g)):
        if allowed is not None and node_m[i] not in allowed:
            continue
        if job.cpu > node_c[i] + eps or job.ram > node_r[i] + eps:
            continue
        if is_int_gpu:
            if req_g_int <= node_g[i]:
                return True
        else:
            if job.gpu <= min(1.0, float(node_g[i])) + eps:
                return True
    return False


def load_alibaba_2023(
    node_csv: Path,
    pod_csv: Path,
    use_gpu_spec: bool = True,
) -> Tuple[ClusterConfig, List[Job], TraceAudit]:
    """
    Alibaba cluster-trace-gpu-v2023의 노드 목록 및 파드 트레이스를 파싱한다.
    시각 단위는 초(sec) -> 시간(hour)으로 변환하며, arrival_time은 최소 시각 기준 0으로 이동한다.
    """
    node_gpus: List[int] = []
    node_cpus: List[float] = []
    node_rams: List[float] = []
    node_models: List[str] = []

    with Path(node_csv).open("rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = int(float(row["gpu"]))
            if g <= 0:
                continue
            cpu = float(row["cpu_milli"]) / 1000.0
            ram = float(row["memory_mib"]) / 1024.0
            model = row["model"].strip()
            node_gpus.append(g)
            node_cpus.append(cpu)
            node_rams.append(ram)
            node_models.append(model)

    ng_t = tuple(node_gpus)
    nc_t = tuple(node_cpus)
    nr_t = tuple(node_rams)
    nm_t = tuple(node_models)

    config = ClusterConfig(
        num_nodes=len(node_gpus),
        gpus_per_node=max(node_gpus),
        cpu_per_node=max(node_cpus),
        ram_per_node=max(node_rams),
        node_gpus=ng_t,
        node_cpus=nc_t,
        node_rams=nr_t,
        node_models=nm_t,
    )

    raw_rows = 0
    excluded_cpu_only = 0
    excluded_bad_time = 0
    excluded_multinode = 0
    excluded_infeasible = 0
    parsed: List[Tuple[float, float, float, float, float, Optional[str]]] = []

    with Path(pod_csv).open("rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows += 1
            num_gpu = int(float(row["num_gpu"] or 0))
            gpu_milli = float(row.get("gpu_milli") or 0.0)
            if num_gpu <= 0:
                excluded_cpu_only += 1
                continue

            if num_gpu == 1 and 0.0 < gpu_milli < 1000.0:
                gpu_req = gpu_milli / 1000.0
            else:
                gpu_req = float(num_gpu)

            c_str = (row.get("creation_time") or "").strip()
            d_str = (row.get("deletion_time") or "").strip()
            s_str = (row.get("scheduled_time") or "").strip()
            if not c_str or not d_str:
                excluded_bad_time += 1
                continue

            c_t = float(c_str)
            d_t = float(d_str)
            s_t = float(s_str) if s_str else c_t
            if d_t <= c_t:
                excluded_bad_time += 1
                continue

            # 실제 실행 시간 우선 (deletion - scheduled), 역전/결측 시 deletion - creation
            dur_sec = (d_t - s_t) if (c_t <= s_t < d_t) else (d_t - c_t)
            if dur_sec <= 0.0 or not math.isfinite(dur_sec):
                excluded_bad_time += 1
                continue

            cpu_req = float(row["cpu_milli"]) / 1000.0
            ram_req = float(row["memory_mib"]) / 1024.0
            raw_spec = (row.get("gpu_spec") or "").strip()
            spec = raw_spec if (use_gpu_spec and raw_spec) else None

            if gpu_req > config.gpus_per_node + 1e-9:
                excluded_multinode += 1
                continue

            parsed.append((c_t, dur_sec, gpu_req, cpu_req, ram_req, spec))

    if not parsed:
        raise RuntimeError(f"유효한 GPU 작업이 없습니다: {pod_csv}")

    min_c_t = min(item[0] for item in parsed)
    parsed.sort(key=lambda x: x[0])

    jobs: List[Job] = []
    for idx, (c_t, dur_sec, gpu_req, cpu_req, ram_req, spec) in enumerate(parsed):
        job = Job(
            job_id=idx,
            arrival_time=(c_t - min_c_t) / 3600.0,
            duration=dur_sec / 3600.0,
            gpu=gpu_req,
            cpu=cpu_req,
            ram=ram_req,
            gpu_spec=spec,
        )
        if not _node_can_host(job, ng_t, nc_t, nr_t, nm_t):
            excluded_infeasible += 1
            continue
        jobs.append(job)

    stats_tuple = _compute_concurrency_and_stats(jobs)
    audit = TraceAudit(
        source_name=f"Alibaba2023:{Path(pod_csv).stem}",
        raw_rows=raw_rows,
        valid_gpu_jobs=len(jobs),
        excluded_cpu_only=excluded_cpu_only,
        excluded_bad_time=excluded_bad_time,
        excluded_multinode=excluded_multinode,
        excluded_infeasible_node=excluded_infeasible,
        corr_log_gpu_dur=stats_tuple[0],
        dur_cv=stats_tuple[1],
        dur_mean_hours=stats_tuple[2],
        dur_median_hours=stats_tuple[3],
        dur_p95_hours=stats_tuple[4],
        fractional_ratio=stats_tuple[5],
        multi_gpu_ratio=stats_tuple[6],
        time_span_hours=stats_tuple[7],
        mean_concurrent_gpus=stats_tuple[8],
        peak_concurrent_gpus=stats_tuple[9],
        gpu_bucket_counts=stats_tuple[10],
    )
    return config, jobs, audit


def scale_heterogeneous_cluster(
    config: ClusterConfig,
    keep_ratio: float,
    seed: int = 42,
    jobs: Optional[List[Job]] = None,
) -> ClusterConfig:
    """
    이기종 클러스터의 기종·노드규격 비율을 계층적 비례 샘플링(Stratified Sampling)으로 유지하면서
    노드 수만 keep_ratio 비율로 축소한다. (원본 작업 도착·수명 시퀀스는 100% 그대로 보존)
    """
    if config.node_gpus is None or config.node_cpus is None or config.node_rams is None:
        n_keep = max(1, int(round(config.num_nodes * keep_ratio)))
        return ClusterConfig(
            num_nodes=n_keep,
            gpus_per_node=config.gpus_per_node,
            cpu_per_node=config.cpu_per_node,
            ram_per_node=config.ram_per_node,
        )

    rng = np.random.default_rng(seed)
    n = config.num_nodes
    models = config.node_models if config.node_models is not None else tuple(["GENERIC"] * n)

    # (모델, GPU수) 버킷별로 비례 샘플링하되, 작업 호환성을 위해 비어 있는 버킷이 없도록 최소 1개 보장
    groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i in range(n):
        groups[(models[i], config.node_gpus[i])].append(i)

    chosen: List[int] = []
    for _, idxs in sorted(groups.items()):
        cnt = max(1, int(round(len(idxs) * keep_ratio)))
        cnt = min(cnt, len(idxs))
        picked = rng.choice(idxs, size=cnt, replace=False)
        chosen.extend(int(x) for x in picked)

    chosen.sort()
    sub_g = tuple(config.node_gpus[i] for i in chosen)
    sub_c = tuple(config.node_cpus[i] for i in chosen)
    sub_r = tuple(config.node_rams[i] for i in chosen)
    sub_m = tuple(models[i] for i in chosen) if config.node_models is not None else None

    # 만약 특정 대형/특수 gpu_spec 작업이 축소 클러스터에 못 들어가면 원본에서 호환 최대 노드 1대 보강
    if jobs:
        chosen_set = set(chosen)
        for job in jobs:
            if not _node_can_host(job, sub_g, sub_c, sub_r, sub_m):
                for orig_i in range(n):
                    if orig_i in chosen_set:
                        continue
                    if _node_can_host(
                        job,
                        (config.node_gpus[orig_i],),
                        (config.node_cpus[orig_i],),
                        (config.node_rams[orig_i],),
                        (models[orig_i],) if sub_m is not None else None,
                    ):
                        chosen_set.add(orig_i)
                        break
        chosen = sorted(chosen_set)
        sub_g = tuple(config.node_gpus[i] for i in chosen)
        sub_c = tuple(config.node_cpus[i] for i in chosen)
        sub_r = tuple(config.node_rams[i] for i in chosen)
        sub_m = tuple(models[i] for i in chosen) if config.node_models is not None else None

    return ClusterConfig(
        num_nodes=len(chosen),
        gpus_per_node=max(sub_g),
        cpu_per_node=max(sub_c),
        ram_per_node=max(sub_r),
        node_gpus=sub_g,
        node_cpus=sub_c,
        node_rams=sub_r,
        node_models=sub_m,
    )


def _fast_parse_ts_sec(ts_str: str, epoch_cache: Dict[str, float]) -> float:
    """'YYYY-MM-DD HH:MM:SS' 타임스탬프를 날짜 프리픽스 캐싱으로 고속 유닉스 초로 변환한다."""
    date_part = ts_str[:10]
    day_base = epoch_cache.get(date_part)
    if day_base is None:
        day_base = datetime.strptime(date_part, "%Y-%m-%d").timestamp()
        epoch_cache[date_part] = day_base
    hh = int(ts_str[11:13])
    mm = int(ts_str[14:16])
    ss = int(ts_str[17:19])
    return day_base + hh * 3600.0 + mm * 60.0 + ss


def load_helios_cluster(
    log_csv: Path,
    cluster_name: str,
    max_rows: Optional[int] = None,
) -> Tuple[List[Job], TraceAudit]:
    """
    SenseTime HeliosData cluster_log.csv를 파싱하여 단일 노드 GPU 작업(1~8 GPU) 시퀀스와
    감사 요약을 반환한다. (RAM 컬럼 부재로 cpu*8 GB 범위 [16, 512] 보간 적용)
    """
    raw_rows = 0
    excluded_cpu_only = 0
    excluded_bad_time = 0
    excluded_multinode = 0
    parsed: List[Tuple[float, float, float, float, float]] = []
    date_cache: Dict[str, float] = {}

    with Path(log_csv).open("rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows is not None and raw_rows >= max_rows:
                break
            raw_rows += 1

            gpu_num = int(float(row["gpu_num"] or 0))
            if gpu_num <= 0:
                excluded_cpu_only += 1
                continue

            node_num = int(float(row.get("node_num") or 1))
            if node_num > 1 or gpu_num > 8:
                excluded_multinode += 1
                continue

            sub_str = (row.get("submit_time") or "").strip()
            if len(sub_str) < 19:
                excluded_bad_time += 1
                continue

            dur_str = (row.get("duration") or "").strip()
            if dur_str:
                dur_sec = float(dur_str)
            else:
                st_str = (row.get("start_time") or "").strip()
                ed_str = (row.get("end_time") or "").strip()
                if len(st_str) < 19 or len(ed_str) < 19:
                    excluded_bad_time += 1
                    continue
                dur_sec = _fast_parse_ts_sec(ed_str, date_cache) - _fast_parse_ts_sec(st_str, date_cache)

            if dur_sec <= 0.0 or not math.isfinite(dur_sec):
                excluded_bad_time += 1
                continue

            arr_sec = _fast_parse_ts_sec(sub_str, date_cache)
            cpu_raw = float(row.get("cpu_num") or 0.0)
            cpu_req = min(64.0, max(1.0, cpu_raw if cpu_raw > 0.0 else gpu_num * 4.0))
            ram_req = min(512.0, max(16.0, cpu_req * 8.0))

            parsed.append((arr_sec, dur_sec, float(gpu_num), cpu_req, ram_req))

    if not parsed:
        raise RuntimeError(f"Helios 트레이스에 유효한 단일 노드 GPU 작업이 없습니다: {log_csv}")

    parsed.sort(key=lambda x: x[0])
    min_arr = parsed[0][0]
    jobs = [
        Job(
            job_id=i,
            arrival_time=(arr - min_arr) / 3600.0,
            duration=dur / 3600.0,
            gpu=g,
            cpu=c,
            ram=r,
        )
        for i, (arr, dur, g, c, r) in enumerate(parsed)
    ]

    stats_tuple = _compute_concurrency_and_stats(jobs)
    audit = TraceAudit(
        source_name=f"Helios:{cluster_name}",
        raw_rows=raw_rows,
        valid_gpu_jobs=len(jobs),
        excluded_cpu_only=excluded_cpu_only,
        excluded_bad_time=excluded_bad_time,
        excluded_multinode=excluded_multinode,
        excluded_infeasible_node=0,
        corr_log_gpu_dur=stats_tuple[0],
        dur_cv=stats_tuple[1],
        dur_mean_hours=stats_tuple[2],
        dur_median_hours=stats_tuple[3],
        dur_p95_hours=stats_tuple[4],
        fractional_ratio=stats_tuple[5],
        multi_gpu_ratio=stats_tuple[6],
        time_span_hours=stats_tuple[7],
        mean_concurrent_gpus=stats_tuple[8],
        peak_concurrent_gpus=stats_tuple[9],
        gpu_bucket_counts=stats_tuple[10],
    )
    return jobs, audit


def extract_helios_windows(
    jobs: List[Job],
    num_windows: int = 5,
    jobs_per_window: int = 3000,
    target_load: float = 0.88,
    min_span_hours: float = 2.0,
) -> List[Tuple[ClusterConfig, List[Job], float]]:
    """
    긴 실제 프로덕션 로그에서 동시 GPU 부하 밀도가 높은 비중첩(non-overlapping) 연속 구간
    num_windows개를 추출한다. 각 구간의 원본 도착 간격과 실제 수명은 100% 보존하며,
    해당 구간의 실측 평균 동시 GPU 수요에 맞춰 8-GPU 노드 수만 현실적 경합 수준으로 맞춘다.

    반환값 리스트 원소: (ClusterConfig, window_jobs, mean_concurrent_gpus)
    """
    n = len(jobs)
    if n < jobs_per_window:
        raise ValueError(f"작업 수({n})가 윈도우 크기({jobs_per_window})보다 작습니다.")

    stride = jobs_per_window // 2
    candidates: List[Tuple[float, int, int, float]] = []  # (score, start_idx, end_idx, mean_conc)

    for start in range(0, n - jobs_per_window + 1, stride):
        end = start + jobs_per_window
        sub = jobs[start:end]
        span = sub[-1].arrival_time - sub[0].arrival_time
        # 0초 일괄 스크립트 스파이크 및 과도한 공백 구간 제외
        if span < min_span_hours:
            continue
        total_gpu_hrs = sum(j.gpu * j.duration for j in sub)
        mean_conc = total_gpu_hrs / span
        # 8-GPU 노드 최소 8대(64 GPU) 이상 감당 가능한 충분한 규모 구간 선호
        if mean_conc < 20.0:
            continue
        score = mean_conc
        candidates.append((score, start, end, mean_conc))

    # 점수가 높은 순으로 비중첩(시간·인덱스 모두 겹치지 않음) 구간 선택
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected: List[Tuple[int, int, float]] = []
    for _, s_idx, e_idx, mc in candidates:
        overlap = any(not (e_idx <= os_i or oe_i <= s_idx) for os_i, oe_i, _ in selected)
        if not overlap:
            selected.append((s_idx, e_idx, mc))
            if len(selected) >= num_windows:
                break

    selected.sort(key=lambda x: x[0])
    out: List[Tuple[ClusterConfig, List[Job], float]] = []
    for s_idx, e_idx, _ in selected:
        raw_sub = jobs[s_idx:e_idx]
        t0 = raw_sub[0].arrival_time
        win_jobs = [
            Job(
                job_id=k,
                arrival_time=j.arrival_time - t0,
                duration=j.duration,
                gpu=j.gpu,
                cpu=j.cpu,
                ram=j.ram,
                gpu_spec=j.gpu_spec,
            )
            for k, j in enumerate(raw_sub)
        ]
        # 도착 구간 [0, arr_span] 클리핑 평균 동시 GPU 수요 산출 (장기 꼬리에 의한 과소평가 방지)
        arr_span = max(win_jobs[-1].arrival_time, 1e-6)
        clipped_gpu_hrs = sum(
            j.gpu * max(0.0, min(j.arrival_time + j.duration, arr_span) - j.arrival_time)
            for j in win_jobs
        )
        clipped_mean_conc = clipped_gpu_hrs / arr_span
        num_nodes = max(8, min(128, int(round(clipped_mean_conc / (8.0 * target_load)))))
        cfg = ClusterConfig(num_nodes=num_nodes, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)
        out.append((cfg, win_jobs, clipped_mean_conc))
    return out


def slice_active_trace(
    jobs: List[Job],
    start_quantile: float = 0.05,
) -> Tuple[List[Job], float]:
    """
    장기간 트레이스 앞부분의 희소 워밍업 구간(예: 하위 5%)을 제외한 고밀도 활성 운영 구간을
    추출하여 시작 시각 0.0으로 정렬한다. 반환값: (sliced_jobs, mean_concurrent_gpus)
    """
    if not jobs:
        return [], 0.0
    start_idx = max(0, min(len(jobs) - 1, int(len(jobs) * start_quantile)))
    sub = jobs[start_idx:]
    t0 = sub[0].arrival_time
    sliced = [
        Job(
            job_id=i,
            arrival_time=j.arrival_time - t0,
            duration=j.duration,
            gpu=j.gpu,
            cpu=j.cpu,
            ram=j.ram,
            gpu_spec=j.gpu_spec,
        )
        for i, j in enumerate(sub)
    ]
    stats_tuple = _compute_concurrency_and_stats(sliced)
    return sliced, stats_tuple[8]


def build_empirical_fgd_distribution(jobs: List[Job]) -> Tuple[np.ndarray, np.ndarray]:
    """실제 트레이스 작업 목록으로부터 FGD_OracleStatic에 주입할 노드당 경험적 분포 (reqs, probs)를 산출한다."""
    if not jobs:
        return np.array([[1.0, 8.0, 64.0]], dtype=np.float64), np.array([1.0], dtype=np.float64)

    buckets: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
    for j in jobs:
        nn = max(1, int(j.num_nodes))
        key = round(j.gpu / nn, 2)
        buckets[key].append((j.cpu / nn, j.ram / nn))

    total = len(jobs)
    reqs = []
    probs = []
    for g_key in sorted(buckets.keys()):
        items = buckets[g_key]
        c_mean = sum(x[0] for x in items) / len(items)
        r_mean = sum(x[1] for x in items) / len(items)
        reqs.append((g_key, c_mean, r_mean))
        probs.append(len(items) / total)
    return np.array(reqs, dtype=np.float64), np.array(probs, dtype=np.float64)


def load_helios_cluster_multinode(
    log_csv: Path,
    cluster_name: str,
    max_nodes_per_job: int = 8,
    max_rows: Optional[int] = None,
) -> Tuple[List[Job], TraceAudit]:
    """
    SenseTime HeliosData cluster_log.csv에서 단일 노드(1~8 GPU)와
    멀티노드 분산 학습(2~8노드, 최대 64 GPU) 작업을 함께 파싱한다.
    """
    raw_rows = 0
    excluded_cpu_only = 0
    excluded_bad_time = 0
    excluded_multinode_over = 0
    excluded_infeasible = 0
    parsed: List[Tuple[float, float, float, float, float, int]] = []
    date_cache: Dict[str, float] = {}

    with Path(log_csv).open("rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows is not None and raw_rows >= max_rows:
                break
            raw_rows += 1

            gpu_num = int(float(row["gpu_num"] or 0))
            if gpu_num <= 0:
                excluded_cpu_only += 1
                continue

            node_num = max(1, int(float(row.get("node_num") or 1)))
            if node_num > max_nodes_per_job or gpu_num > max_nodes_per_job * 8:
                excluded_multinode_over += 1
                continue

            # 단일/멀티노드 모두 노드당 균등 정수 GPU (1..8) 분할 가능한 작업만 유효
            if node_num == 1:
                if gpu_num > 8:
                    excluded_multinode_over += 1
                    continue
                gpu_pn = gpu_num
            else:
                if gpu_num % node_num != 0 or not (1 <= (gpu_num // node_num) <= 8):
                    excluded_infeasible += 1
                    continue
                gpu_pn = gpu_num // node_num

            sub_str = (row.get("submit_time") or "").strip()
            if len(sub_str) < 19:
                excluded_bad_time += 1
                continue

            dur_str = (row.get("duration") or "").strip()
            if dur_str:
                dur_sec = float(dur_str)
            else:
                st_str = (row.get("start_time") or "").strip()
                ed_str = (row.get("end_time") or "").strip()
                if len(st_str) < 19 or len(ed_str) < 19:
                    excluded_bad_time += 1
                    continue
                dur_sec = _fast_parse_ts_sec(ed_str, date_cache) - _fast_parse_ts_sec(st_str, date_cache)

            if dur_sec <= 0.0 or not math.isfinite(dur_sec):
                excluded_bad_time += 1
                continue

            arr_sec = _fast_parse_ts_sec(sub_str, date_cache)
            cpu_raw = float(row.get("cpu_num") or 0.0)
            cpu_pn = min(64.0, max(1.0, (cpu_raw / node_num) if cpu_raw > 0.0 else gpu_pn * 4.0))
            ram_pn = min(512.0, max(16.0, cpu_pn * 8.0))

            parsed.append((arr_sec, dur_sec, float(gpu_num), cpu_pn * node_num, ram_pn * node_num, node_num))

    if not parsed:
        raise RuntimeError(f"Helios 멀티노드 트레이스에 유효한 GPU 작업이 없습니다: {log_csv}")

    parsed.sort(key=lambda x: x[0])
    min_arr = parsed[0][0]
    jobs = [
        Job(
            job_id=i,
            arrival_time=(arr - min_arr) / 3600.0,
            duration=dur / 3600.0,
            gpu=g,
            cpu=c,
            ram=r,
            num_nodes=nn,
        )
        for i, (arr, dur, g, c, r, nn) in enumerate(parsed)
    ]

    stats_tuple = _compute_concurrency_and_stats(jobs)
    audit = TraceAudit(
        source_name=f"HeliosMultiNode:{cluster_name}",
        raw_rows=raw_rows,
        valid_gpu_jobs=len(jobs),
        excluded_cpu_only=excluded_cpu_only,
        excluded_bad_time=excluded_bad_time,
        excluded_multinode=excluded_multinode_over,
        excluded_infeasible_node=excluded_infeasible,
        corr_log_gpu_dur=stats_tuple[0],
        dur_cv=stats_tuple[1],
        dur_mean_hours=stats_tuple[2],
        dur_median_hours=stats_tuple[3],
        dur_p95_hours=stats_tuple[4],
        fractional_ratio=stats_tuple[5],
        multi_gpu_ratio=stats_tuple[6],
        time_span_hours=stats_tuple[7],
        mean_concurrent_gpus=stats_tuple[8],
        peak_concurrent_gpus=stats_tuple[9],
        gpu_bucket_counts=stats_tuple[10],
    )
    return jobs, audit


def extract_helios_multinode_windows(
    jobs: List[Job],
    num_windows: int = 5,
    jobs_per_window: int = 3000,
    target_load: float = 0.85,
    min_span_hours: float = 2.0,
    nodes_per_rack: int = 8,
) -> List[Tuple[ClusterConfig, List[Job], float]]:
    """
    단일+멀티노드(1~8노드) 혼합 Helios 로그에서 도착 구간 클리핑 평균 수요가 높고
    실제 멀티노드 분산 작업(num_nodes > 1)이 포함된 비중첩 연속 구간을 추출한다.
    노드 수는 랙 단위(nodes_per_rack의 배수, 16~128노드)로 산정한다.
    """
    n = len(jobs)
    if n < jobs_per_window:
        raise ValueError(f"작업 수({n})가 윈도우 크기({jobs_per_window})보다 작습니다.")

    stride = jobs_per_window // 2
    candidates: List[Tuple[float, int, int]] = []
    for start in range(0, n - jobs_per_window + 1, stride):
        end = start + jobs_per_window
        sub = jobs[start:end]
        arr_span = sub[-1].arrival_time - sub[0].arrival_time
        if arr_span < min_span_hours:
            continue
        has_multi = any(j.num_nodes > 1 for j in sub)
        if not has_multi:
            continue
        t0 = sub[0].arrival_time
        clipped_gpu_hrs = sum(
            j.gpu * max(0.0, min(j.arrival_time + j.duration, t0 + arr_span) - j.arrival_time)
            for j in sub
        )
        clipped_conc = clipped_gpu_hrs / arr_span
        if clipped_conc < 80.0:
            continue
        candidates.append((clipped_conc, start, end))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected: List[Tuple[int, int]] = []
    for _, s_idx, e_idx in candidates:
        overlap = any(not (e_idx <= os_i or oe_i <= s_idx) for os_i, oe_i in selected)
        if not overlap:
            selected.append((s_idx, e_idx))
            if len(selected) >= num_windows:
                break

    selected.sort(key=lambda x: x[0])
    out: List[Tuple[ClusterConfig, List[Job], float]] = []
    for s_idx, e_idx in selected:
        raw_sub = jobs[s_idx:e_idx]
        t0 = raw_sub[0].arrival_time
        win_jobs = [
            Job(
                job_id=k,
                arrival_time=j.arrival_time - t0,
                duration=j.duration,
                gpu=j.gpu,
                cpu=j.cpu,
                ram=j.ram,
                gpu_spec=j.gpu_spec,
                num_nodes=j.num_nodes,
            )
            for k, j in enumerate(raw_sub)
        ]
        arr_span = max(win_jobs[-1].arrival_time, 1e-6)
        clipped_gpu_hrs = sum(
            j.gpu * max(0.0, min(j.arrival_time + j.duration, arr_span) - j.arrival_time)
            for j in win_jobs
        )
        clipped_conc = clipped_gpu_hrs / arr_span
        raw_nodes = int(round(clipped_conc / (8.0 * target_load)))
        # 랙 단위(nodes_per_rack 배수) 올림, 최소 2랙(16노드)~최대 16랙(128노드)
        racks = max(2, min(16, math.ceil(raw_nodes / nodes_per_rack)))
        num_nodes = racks * nodes_per_rack
        cfg = ClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=8,
            cpu_per_node=64.0,
            ram_per_node=512.0,
            nodes_per_rack=nodes_per_rack,
        )
        out.append((cfg, win_jobs, clipped_conc))
    return out
