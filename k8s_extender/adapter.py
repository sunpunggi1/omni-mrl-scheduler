"""
Kubernetes `kube-scheduler` Extender JSON 페이로드 ↔ `Omni-Lindy` / `Omni-MRL` 코어 변환 어댑터.

설계 원칙:
1. 무상태 스냅샷 + 상태유지 인과 버퍼/코호트 통합:
   매 요청마다 전달되는 K8s NodeList 스냅샷으로부터 `ClusterState`를 즉석 구성하되,
   완료 수명(`recent_completions`), 동일 진입 코호트(`recent_cohort`, Pod UID 기반 중도절단·완료 연계),
   최근 GPU 기종 수요(`recent_specs`), 멀티노드 요청 비율(`recent_node_counts`)을
   스레드 안전 링버퍼로 지속 유지해 시뮬레이터와 동일한 코호트 KM 생존 추정 경로를 사용한다.
2. K8s v1.34 Extender 공식 규약 준수:
   - `HostPriorityList` 점수는 공식 Extender 상한인 `[0, 10]` 정수로 정규화한다.
   - `initContainers` 중 `restartPolicy: Always` 네이티브 사이드카는 메인 컨테이너와 동시 실행으로 합산한다.
   - 노드 `running-pods` 어노테이션과 요청 `runningPods` 간 동일 파드의 자원 이중 차감을 방지한다.
   - `NodeNames` 캐시는 TTL 및 선택적 `resourceVersion` 신선도 검증을 강제한다.
3. 100% Non-Clairvoyant:
   실행 중인 파드의 `startTime`과 완료된 파드의 실측 수명만 사용하며 미래 수명을 참조하지 않는다.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import math
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from adaptive_lindy import AdaptiveBLindyPackScheduler, AdaptiveLindyConfig
from omni_mrl import OmniMRLConfig, OmniMRLScheduler
from simulator import ClusterConfig, ClusterState

# K8s 공식 resource.Quantity 메모리 접미사별 바이트 환산 계수 (대소문자 엄격 구분: m=milli, M=Mega)
_MEM_BINARY_MULTIPLIERS: Tuple[Tuple[str, float], ...] = (
    ("Ki", float(1024**1)),
    ("Mi", float(1024**2)),
    ("Gi", float(1024**3)),
    ("Ti", float(1024**4)),
    ("Pi", float(1024**5)),
    ("Ei", float(1024**6)),
)
_MEM_DECIMAL_MULTIPLIERS: Tuple[Tuple[str, float], ...] = (
    ("n", 1e-9),
    ("u", 1e-6),
    ("m", 1e-3),
    ("k", 1e3),
    ("K", 1e3),
    ("M", 1e6),
    ("G", 1e9),
    ("T", 1e12),
    ("P", 1e15),
    ("E", 1e18),
)
_GIB_BYTES = float(1024**3)


def _validate_finite_nonneg(val: float, name: str) -> float:
    """유한하고 0 이상인 실수인지 검증한다."""
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"{name} 값은 0 이상의 유한수여야 합니다: {val!r}")
    return float(val)


def parse_k8s_cpu(val: Any) -> float:
    """K8s CPU 문자열(예: '16', '500m', '2.5') 또는 숫자를 코어 수(float)로 파싱한다."""
    if val is None or isinstance(val, bool):
        raise ValueError(f"유효하지 않은 CPU 값입니다: {val!r}")
    if isinstance(val, (int, float)):
        return _validate_finite_nonneg(float(val), "CPU")
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError("빈 CPU 문자열입니다.")
        if s.endswith("m"):
            return _validate_finite_nonneg(float(s[:-1]) / 1000.0, "CPU")
        return _validate_finite_nonneg(float(s), "CPU")
    raise ValueError(f"지원하지 않는 CPU 형식입니다: {val!r}")


def parse_k8s_memory_gi(val: Any) -> float:
    """K8s 메모리 문자열(예: '512Gi', '400M', '400m'=0.4B) 또는 바이트 숫자를 GiB 단위로 파싱한다."""
    if val is None or isinstance(val, bool):
        raise ValueError(f"유효하지 않은 메모리 값입니다: {val!r}")
    if isinstance(val, (int, float)):
        return _validate_finite_nonneg(float(val) / _GIB_BYTES, "Memory")
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError("빈 메모리 문자열입니다.")
        for suffix, mult in _MEM_BINARY_MULTIPLIERS:
            if s.endswith(suffix):
                num = float(s[: -len(suffix)].strip())
                return _validate_finite_nonneg((num * mult) / _GIB_BYTES, "Memory")
        for suffix, mult in _MEM_DECIMAL_MULTIPLIERS:
            if s.endswith(suffix):
                num = float(s[: -len(suffix)].strip())
                return _validate_finite_nonneg((num * mult) / _GIB_BYTES, "Memory")
        return _validate_finite_nonneg(float(s) / _GIB_BYTES, "Memory")
    raise ValueError(f"지원하지 않는 메모리 형식입니다: {val!r}")


def parse_timestamp_unix(val: Any) -> float:
    """숫자(Unix 초) 또는 RFC3339/ISO8601 타임스탬프를 Unix 초(float)로 변환한다."""
    if isinstance(val, bool):
        raise ValueError(f"유효하지 않은 타임스탬프입니다: {val!r}")
    if isinstance(val, (int, float)):
        return _validate_finite_nonneg(float(val), "Timestamp")
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError("빈 타임스탬프 문자열입니다.")
        try:
            return _validate_finite_nonneg(float(s), "Timestamp")
        except ValueError:
            pass
        iso = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return _validate_finite_nonneg(dt.timestamp(), "Timestamp")
    raise ValueError(f"지원하지 않는 타임스탬프 형식입니다: {val!r}")


def _extract_container_res(res_dict: Dict[str, Any], key: str) -> Any:
    """컨테이너의 resources.requests를 우선 조회하고, 없으면 limits로 폴백한다."""
    reqs = res_dict.get("requests") or {}
    if key in reqs and reqs[key] is not None:
        return reqs[key]
    lims = res_dict.get("limits") or {}
    return lims.get(key)


def parse_pod_resources(pod: Dict[str, Any]) -> Tuple[float, float, float, Optional[str], int]:
    """
    K8s Pod JSON 객체로부터 스케줄링 요구량 튜플
    `(gpu, cpu, ram_gi, gpu_spec, gang_nodes)`을 추출한다.

    - 네이티브 사이드카(`initContainers` 중 `restartPolicy: Always`)는 이후 초기화·메인 컨테이너와
      동시 실행되므로 누적 합산하고, 일반 초기화 컨테이너(`restartPolicy != Always`)는 순차 최댓값을 적용한다.
    - 분수 GPU: `metadata.annotations["omni-lindy.io/gpu-milli"]` 우선 (예: "250" -> 0.25)
    - GPU 기종 제약: `metadata.annotations["omni-lindy.io/gpu-spec"]` 또는 `spec.nodeSelector`
    - 멀티노드 Gang: `metadata.annotations["omni-lindy.io/gang-nodes"]` (기본 1)
    """
    if not isinstance(pod, dict):
        raise ValueError("Pod 객체는 JSON object(dict)여야 합니다.")

    meta = pod.get("metadata") or {}
    spec = pod.get("spec") or {}
    annotations = meta.get("annotations") or {}
    node_selector = spec.get("nodeSelector") or {}

    containers = spec.get("containers") or []
    init_containers = spec.get("initContainers") or []

    cpu_sum = 0.0
    ram_sum = 0.0
    gpu_sum = 0.0
    for c in containers:
        res = (c or {}).get("resources") or {}
        c_val = _extract_container_res(res, "cpu")
        if c_val is not None:
            cpu_sum += parse_k8s_cpu(c_val)
        m_val = _extract_container_res(res, "memory")
        if m_val is not None:
            ram_sum += parse_k8s_memory_gi(m_val)
        g_val = _extract_container_res(res, "nvidia.com/gpu")
        if g_val is not None:
            gpu_sum += _validate_finite_nonneg(float(g_val), "GPU")

    # K8s 공식 PodRequests 규칙: restartPolicy=="Always" 네이티브 사이드카는 동시 누적, 일반 init는 순차 max
    sidecar_cpu = 0.0
    sidecar_ram = 0.0
    sidecar_gpu = 0.0
    cpu_init_peak = 0.0
    ram_init_peak = 0.0
    gpu_init_peak = 0.0
    for ic in init_containers:
        ic_obj = ic or {}
        res = ic_obj.get("resources") or {}
        c_ic = parse_k8s_cpu(_extract_container_res(res, "cpu")) if _extract_container_res(res, "cpu") is not None else 0.0
        m_ic = parse_k8s_memory_gi(_extract_container_res(res, "memory")) if _extract_container_res(res, "memory") is not None else 0.0
        g_raw = _extract_container_res(res, "nvidia.com/gpu")
        g_ic = _validate_finite_nonneg(float(g_raw), "initContainer.GPU") if g_raw is not None else 0.0

        is_sidecar = str(ic_obj.get("restartPolicy") or "").strip() == "Always"
        if is_sidecar:
            sidecar_cpu += c_ic
            sidecar_ram += m_ic
            sidecar_gpu += g_ic
            cpu_init_peak = max(cpu_init_peak, sidecar_cpu)
            ram_init_peak = max(ram_init_peak, sidecar_ram)
            gpu_init_peak = max(gpu_init_peak, sidecar_gpu)
        else:
            cpu_init_peak = max(cpu_init_peak, sidecar_cpu + c_ic)
            ram_init_peak = max(ram_init_peak, sidecar_ram + m_ic)
            gpu_init_peak = max(gpu_init_peak, sidecar_gpu + g_ic)

    # K8s v1.34 베타 기본 활성 Pod-level spec.resources.requests/limits 지원
    pod_res = spec.get("resources") or {}
    pod_cpu = parse_k8s_cpu(_extract_container_res(pod_res, "cpu")) if _extract_container_res(pod_res, "cpu") is not None else 0.0
    pod_ram = parse_k8s_memory_gi(_extract_container_res(pod_res, "memory")) if _extract_container_res(pod_res, "memory") is not None else 0.0
    pod_gpu_raw = _extract_container_res(pod_res, "nvidia.com/gpu")
    pod_gpu = _validate_finite_nonneg(float(pod_gpu_raw), "pod.spec.resources.GPU") if pod_gpu_raw is not None else 0.0

    overhead = spec.get("overhead") or {}
    overhead_cpu = parse_k8s_cpu(overhead["cpu"]) if overhead.get("cpu") is not None else 0.0
    overhead_ram = parse_k8s_memory_gi(overhead["memory"]) if overhead.get("memory") is not None else 0.0

    cpu = max(cpu_sum + sidecar_cpu, cpu_init_peak, pod_cpu) + overhead_cpu
    ram_gi = max(ram_sum + sidecar_ram, ram_init_peak, pod_ram) + overhead_ram

    if "omni-lindy.io/gpu-milli" in annotations and annotations["omni-lindy.io/gpu-milli"] is not None:
        gpu = _validate_finite_nonneg(float(annotations["omni-lindy.io/gpu-milli"]) / 1000.0, "gpu-milli")
    else:
        gpu = max(gpu_sum + sidecar_gpu, gpu_init_peak, pod_gpu)

    gpu_spec: Optional[str] = None
    raw_spec = annotations.get("omni-lindy.io/gpu-spec")
    if isinstance(raw_spec, str) and raw_spec.strip():
        tokens = [tok.strip() for tok in raw_spec.split("|") if tok.strip()]
        gpu_spec = "|".join(tokens) if tokens else None
    else:
        for sel_key in ("omni-lindy.io/gpu-model", "nvidia.com/gpu.product", "accelerator"):
            sel_val = node_selector.get(sel_key)
            if isinstance(sel_val, str) and sel_val.strip():
                gpu_spec = sel_val.strip()
                break

    raw_gang = annotations.get("omni-lindy.io/gang-nodes")
    if raw_gang is None:
        gang_nodes = 1
    else:
        try:
            gang_nodes = int(str(raw_gang).strip())
        except ValueError as exc:
            raise ValueError(f"gang-nodes 어노테이션은 정수여야 합니다: {raw_gang!r}") from exc
        if gang_nodes < 1:
            raise ValueError(f"gang-nodes는 1 이상이어야 합니다: {gang_nodes}")

    return gpu, cpu, ram_gi, gpu_spec, gang_nodes


def scores_to_k8s_priorities(
    node_names: List[str],
    raw_scores: np.ndarray,
    high_score: int = 10,
    low_score: int = 0,
) -> List[Dict[str, Any]]:
    """
    `Omni-Lindy` 원시 점수(낮을수록 우수)를 K8s v1.34 Extender 공식 `HostPriorityList`
    `[low_score, high_score]`(기본 `[0, 10]`, 높을수록 우수) 정수로 단조 역변환한다.
    """
    if len(node_names) != len(raw_scores):
        raise ValueError(
            f"node_names 길이({len(node_names)})와 raw_scores 길이({len(raw_scores)})가 일치하지 않습니다."
        )
    if not node_names:
        return []

    arr = np.asarray(raw_scores, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("원시 점수 배열에 NaN 또는 Inf가 포함되어 있습니다.")

    s_min = float(np.min(arr))
    s_max = float(np.max(arr))
    span = s_max - s_min

    priorities: List[Dict[str, Any]] = []
    if span <= 1e-12:
        for name in node_names:
            priorities.append({"host": name, "score": int(high_score)})
        return priorities

    scale = float(high_score - low_score)
    for name, val in zip(node_names, arr):
        norm = (s_max - float(val)) / span
        mapped = int(round(low_score + scale * norm))
        mapped = max(low_score, min(high_score, mapped))
        priorities.append({"host": name, "score": mapped})
    return priorities


class OmniLindyK8sAdapter:
    """
    `kube-scheduler` HTTP Extender 규격과 `Omni-MRL (v8.0)` / `Omni-Lindy (v7.0)` 코어를 연결하는
    스레드 안전 어댑터.
    """

    def __init__(
        self,
        nodes_per_rack: int = 8,
        adapt_cfg: Optional[AdaptiveLindyConfig] = None,
        engine: str = "omni_mrl",
        mrl_cfg: Optional[OmniMRLConfig] = None,
        node_cache_ttl_sec: float = 30.0,
    ) -> None:
        if engine not in ("omni_mrl", "omni_lindy"):
            raise ValueError(f"지원하지 않는 engine입니다: {engine!r} ('omni_mrl' 또는 'omni_lindy')")
        if not math.isfinite(node_cache_ttl_sec) or node_cache_ttl_sec <= 0.0:
            raise ValueError(f"node_cache_ttl_sec는 양의 유한수여야 합니다: {node_cache_ttl_sec!r}")
        self._lock = threading.RLock()
        self.nodes_per_rack = max(1, int(nodes_per_rack))
        self.engine = engine
        self.node_cache_ttl_sec = float(node_cache_ttl_sec)
        if engine == "omni_mrl":
            self._scheduler: AdaptiveBLindyPackScheduler = OmniMRLScheduler(
                adapt_cfg=adapt_cfg,
                mrl_cfg=mrl_cfg,
                name="Omni-MRL-K8s",
                hetero_aware=True,
                topology_aware=True,
            )
        else:
            self._scheduler = AdaptiveBLindyPackScheduler(
                adapt_cfg=adapt_cfg,
                name="Omni-Lindy-K8s",
                hetero_aware=True,
                topology_aware=True,
            )

        # 인과적 온라인 히스토리 링버퍼 및 동일 진입 코호트 (Non-Clairvoyant, UID-일관 추적)
        self.recent_completions: Deque[Tuple[float, float]] = deque(maxlen=256)
        self.recent_cohort: Deque[List[object]] = deque(maxlen=1024)
        self._cohort_keys_by_entry: Dict[int, set[str]] = {}
        self._cohort_uid_by_entry: Dict[int, Optional[str]] = {}
        self._cohort_by_uid: Dict[str, List[object]] = {}
        self._cohort_by_ns: Dict[str, List[List[object]]] = {}
        self._completed_uid_order: Deque[str] = deque(maxlen=4096)
        self._completed_uid_set: set[str] = set()
        self.recent_specs: Deque[Optional[str]] = deque(maxlen=200)
        self.recent_node_counts: Deque[int] = deque(maxlen=200)
        self._event_counter: int = 0
        # 노드 캐시: name -> (cached_at_unix, resource_version, node_dict)
        self._node_snapshot_cache: Dict[str, Tuple[float, Optional[str], Dict[str, Any]]] = {}

        # 관측 메트릭 카운터
        self.filter_calls: int = 0
        self.prioritize_calls: int = 0
        self.gang_calls: int = 0
        self.completion_events: int = 0
        self.prioritize_latency_us_sum: float = 0.0

    def _append_cohort_entry(
        self,
        uid_key: Optional[str],
        ns_key: Optional[str],
        entry: List[object],
    ) -> None:
        """코호트 링버퍼(`maxlen=1024`)에 항목을 추가하고 퇴거된 UID/(namespace,name) 인덱스를 정리한다."""
        if len(self.recent_cohort) == self.recent_cohort.maxlen:
            evicted_entry = self.recent_cohort[0]
            self._cohort_keys_by_entry.pop(id(evicted_entry), None)
            self._cohort_uid_by_entry.pop(id(evicted_entry), None)
            for k, v in list(self._cohort_by_uid.items()):
                if v is evicted_entry:
                    del self._cohort_by_uid[k]
            for nk, lst in list(self._cohort_by_ns.items()):
                filtered = [e for e in lst if e is not evicted_entry]
                if filtered:
                    self._cohort_by_ns[nk] = filtered
                else:
                    del self._cohort_by_ns[nk]
        self.recent_cohort.append(entry)
        self._bind_cohort_identity(entry, uid_key, ns_key)

    def _bind_cohort_identity(
        self,
        entry: List[object],
        uid_key: Optional[str],
        ns_key: Optional[str],
    ) -> None:
        """기존 코호트 항목에 UID 및 (namespace, name) 식별 키를 충돌 없이 연결한다."""
        ks = self._cohort_keys_by_entry.setdefault(id(entry), set())
        if uid_key is not None:
            self._cohort_uid_by_entry[id(entry)] = uid_key
            ks.add(uid_key)
            self._cohort_by_uid[uid_key] = entry
        else:
            self._cohort_uid_by_entry.setdefault(id(entry), None)
        if ns_key is not None:
            ks.add(ns_key)
            lst = self._cohort_by_ns.setdefault(ns_key, [])
            if not any(x is entry for x in lst):
                lst.append(entry)

    def _find_cohort_entry(
        self,
        uid_key: Optional[str],
        ns_key: Optional[str],
        st_unix_hint: Optional[float] = None,
        prefer_open: bool = False,
    ) -> Optional[List[object]]:
        """
        동명·상이 UID가 절대 병합되지 않도록 UID 우선 매칭 후,
        비충돌 `(namespace, name)` 항목만 탐색한다.
        """
        if uid_key is not None:
            exact = self._cohort_by_uid.get(uid_key)
            if exact is not None:
                return exact
        if ns_key is None:
            return None
        candidates = self._cohort_by_ns.get(ns_key) or []
        for entry in reversed(candidates):
            entry_uid = self._cohort_uid_by_entry.get(id(entry))
            if uid_key is not None and entry_uid is not None and entry_uid != uid_key:
                # 핵심 불변식: 서로 다른 UID를 가진 동명 파드는 절대 동일 코호트로 병합하지 않는다.
                continue
            if prefer_open and entry[1] is not None:
                continue
            if st_unix_hint is not None and entry[1] is not None and abs(float(entry[0]) - st_unix_hint) > 1.0:
                continue
            return entry
        return None

    @staticmethod
    def _split_pod_identity(
        uid: Any = None,
        namespace: Any = None,
        name: Any = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """`(uid_key, ns_key)` 튜플을 반환한다."""
        uid_key = f"uid:{uid.strip()}" if isinstance(uid, str) and uid.strip() else None
        ns_key: Optional[str] = None
        if isinstance(name, str) and name.strip():
            ns = str(namespace).strip() if isinstance(namespace, str) and namespace.strip() else "default"
            ns_key = f"ns:{ns}/{name.strip()}"
        return uid_key, ns_key

    @staticmethod
    def _pod_identity_keys(
        uid: Any = None,
        namespace: Any = None,
        name: Any = None,
    ) -> List[str]:
        """UID 및 (namespace, name) 기반 정규 식별 키 리스트를 반환한다."""
        uid_key, ns_key = OmniLindyK8sAdapter._split_pod_identity(uid, namespace, name)
        return [k for k in (uid_key, ns_key) if k is not None]

    def _extract_nodes(self, extender_args: Dict[str, Any], now_unix: float) -> List[Dict[str, Any]]:
        """ExtenderArgs의 `Nodes` 또는 신선도 검증된 `NodeNames` 캐시에서 K8s Node 목록을 추출한다."""
        if not isinstance(extender_args, dict):
            raise ValueError("요청 본문은 JSON object여야 합니다.")
        nodes_field = extender_args.get("Nodes")
        items: Optional[List[Dict[str, Any]]] = None
        if isinstance(nodes_field, dict):
            raw_items = nodes_field.get("items")
            if not isinstance(raw_items, list):
                raise ValueError("'Nodes.items'는 리스트여야 합니다.")
            items = raw_items
        elif isinstance(nodes_field, list):
            items = nodes_field
        elif nodes_field is not None:
            raise ValueError("'Nodes' 필드는 NodeList(dict) 또는 Node 배열이어야 합니다.")

        if items is not None:
            for node in items:
                if isinstance(node, dict):
                    meta = node.get("metadata") or {}
                    nm = meta.get("name")
                    if nm:
                        rv_raw = meta.get("resourceVersion")
                        rv = str(rv_raw).strip() if rv_raw is not None and str(rv_raw).strip() else None
                        self._node_snapshot_cache[str(nm)] = (now_unix, rv, node)
            return items

        node_names = extender_args.get("NodeNames")
        if isinstance(node_names, list):
            if not node_names:
                return []
            missing = [str(nm) for nm in node_names if str(nm) not in self._node_snapshot_cache]
            if missing:
                raise ValueError(
                    "NodeCacheCapable=true (NodeNames only) 요청에는 사전 노드 스냅샷 동기화가 필요합니다: "
                    f"누락 노드={missing[:5]}"
                )
            req_rvs = extender_args.get("NodeResourceVersions")
            if req_rvs is not None and not isinstance(req_rvs, dict):
                raise ValueError("'NodeResourceVersions' 필드는 JSON object(dict)여야 합니다.")

            resolved: List[Dict[str, Any]] = []
            for raw_nm in node_names:
                nm = str(raw_nm)
                cached_at, cached_rv, cached_node = self._node_snapshot_cache[nm]
                if (now_unix - cached_at) > self.node_cache_ttl_sec or (cached_at - now_unix) > self.node_cache_ttl_sec:
                    raise ValueError(
                        f"노드 캐시 TTL 만료 (node={nm}, age={now_unix - cached_at:.2f}s > ttl={self.node_cache_ttl_sec:.2f}s): "
                        "최신 Nodes 스냅샷을 재전송하세요."
                    )
                if req_rvs and nm in req_rvs:
                    want_rv = str(req_rvs[nm]).strip()
                    if cached_rv is None or cached_rv != want_rv:
                        raise ValueError(
                            f"노드 캐시 resourceVersion 불일치 (node={nm}, cached={cached_rv!r}, want={want_rv!r}): "
                            "최신 Nodes 스냅샷이 필요합니다."
                        )
                resolved.append(cached_node)
            return resolved

        raise ValueError("ExtenderArgs에 'Nodes' 또는 캐시된 'NodeNames' 필드가 필요합니다.")

    def _get_now_unix(self, extender_args: Dict[str, Any]) -> float:
        """결정적 테스트를 위한 선택적 `now` / `nowUnix` 타임스탬프 오버라이드를 지원한다."""
        for key in ("nowUnix", "now"):
            if key in extender_args and extender_args[key] is not None:
                return parse_timestamp_unix(extender_args[key])
        return time.time()

    def build_cluster_state(
        self,
        nodes_items: List[Dict[str, Any]],
        running_pods: Optional[List[Dict[str, Any]]] = None,
        now_unix: Optional[float] = None,
        gpu_spec: Optional[str] = None,
    ) -> Tuple[ClusterState, List[str]]:
        """K8s NodeList 스냅샷 및 실행 중 파드 정보로부터 `ClusterState`를 구성한다."""
        now_ts = time.time() if now_unix is None else float(now_unix)
        n = len(nodes_items)
        node_names: List[str] = []
        node_gpus: List[int] = []
        node_cpus: List[float] = []
        node_rams: List[float] = []
        raw_models: List[Optional[str]] = []
        raw_racks: List[Optional[str]] = []

        for idx, node in enumerate(nodes_items):
            if not isinstance(node, dict):
                raise ValueError(f"Nodes[{idx}] 항목은 JSON object여야 합니다.")
            meta = node.get("metadata") or {}
            labels = meta.get("labels") or {}
            status = node.get("status") or {}
            cap_src = status.get("allocatable") or status.get("capacity") or {}

            name = str(meta.get("name") or f"node-{idx}")
            node_names.append(name)

            # `nvidia.com/gpu`가 명시되지 않은 노드는 CPU 전용 노드(0 GPU)로 처리한다.
            # (단, status가 전혀 없고 테스트용 gpu-slots-free 어노테이션만 존재하는 경우에만 슬롯 길이 추론)
            if "nvidia.com/gpu" in cap_src:
                raw_g = cap_src["nvidia.com/gpu"]
                g_cap = max(0, int(round(_validate_finite_nonneg(float(raw_g), f"node[{name}].gpu"))))
            elif not cap_src and "omni-lindy.io/gpu-slots-free" in (meta.get("annotations") or {}):
                raw_slots = meta["annotations"]["omni-lindy.io/gpu-slots-free"]
                slots_tmp = json.loads(raw_slots) if isinstance(raw_slots, str) else raw_slots
                g_cap = len(slots_tmp) if isinstance(slots_tmp, list) else 0
            else:
                g_cap = 0
            c_cap = parse_k8s_cpu(cap_src.get("cpu", 64))
            r_cap = parse_k8s_memory_gi(cap_src.get("memory", "512Gi"))

            node_gpus.append(g_cap)
            node_cpus.append(c_cap)
            node_rams.append(r_cap)

            model = labels.get("omni-lindy.io/gpu-model") or labels.get("nvidia.com/gpu.product")
            raw_models.append(str(model).strip() if isinstance(model, str) and model.strip() else None)

            rack = labels.get("omni-lindy.io/rack") or labels.get("topology.kubernetes.io/zone")
            raw_racks.append(str(rack).strip() if isinstance(rack, str) and rack.strip() else None)

        has_models = any(m is not None for m in raw_models)
        node_models = tuple(m if m is not None else "GENERIC" for m in raw_models) if has_models else None

        cfg = ClusterConfig(
            num_nodes=n,
            gpus_per_node=max(1, max(node_gpus)) if node_gpus else 8,
            node_gpus=tuple(node_gpus),
            node_cpus=tuple(node_cpus),
            node_rams=tuple(node_rams),
            node_models=node_models,
            nodes_per_rack=self.nodes_per_rack,
        )
        state = ClusterState(cfg)
        state.now = now_ts
        state.current_gpu_spec = gpu_spec

        # 명시적 랙/존 레이블이 존재하면 연속 정수 랙 ID로 매핑
        if any(r is not None for r in raw_racks):
            rack_to_id: Dict[str, int] = {}
            rack_ids = np.zeros(n, dtype=np.int32)
            for idx, r in enumerate(raw_racks):
                r_key = r if r is not None else f"auto-rack-{idx // self.nodes_per_rack}"
                if r_key not in rack_to_id:
                    rack_to_id[r_key] = len(rack_to_id)
                rack_ids[idx] = rack_to_id[r_key]
            state.node_rack_ids = rack_ids
            state.num_racks = max(1, len(rack_to_id))

        name_to_idx = {nm: idx for idx, nm in enumerate(node_names)}
        # 자원별 독립 추적: CPU 어노테이션만 있어도 runningPods의 GPU 차감이 누락되지 않도록 분리
        explicit_gpu_nodes: set[int] = set()
        explicit_cpu_nodes: set[int] = set()
        explicit_ram_nodes: set[int] = set()

        # 노드 어노테이션 및 클러스터 runningPods 통합 파드 목록 (이중 차감 차단 & 동명·상이 UID 비병합)
        merged_pods: List[Dict[str, Any]] = []
        pods_by_uid: Dict[str, Dict[str, Any]] = {}
        pods_by_ns: Dict[str, List[Dict[str, Any]]] = {}
        anon_pods_by_node: Dict[int, List[Dict[str, Any]]] = {}

        def _find_merged_pod(uid_k: Optional[str], ns_k: Optional[str], node_idx_v: Optional[int] = None) -> Optional[Dict[str, Any]]:
            if uid_k is not None and uid_k in pods_by_uid:
                return pods_by_uid[uid_k]
            if ns_k is not None:
                for cand in pods_by_ns.get(ns_k, []):
                    c_uid = cand.get("uid_key")
                    if uid_k is not None and c_uid is not None and c_uid != uid_k:
                        continue
                    if node_idx_v is not None and int(cand["node_idx"]) != node_idx_v and uid_k is None:
                        continue
                    return cand
            return None

        def _index_merged_pod(rec: Dict[str, Any]) -> None:
            if rec.get("uid_key") is not None:
                pods_by_uid[str(rec["uid_key"])] = rec
            if rec.get("ns_key") is not None:
                pods_by_ns.setdefault(str(rec["ns_key"]), []).append(rec)

        # 1) 노드 어노테이션 기반 실시간 가용 슬롯·실행 파드 파싱
        for idx, node in enumerate(nodes_items):
            annotations = (node.get("metadata") or {}).get("annotations") or {}
            if "omni-lindy.io/gpu-slots-free" in annotations:
                explicit_gpu_nodes.add(idx)
                raw_slots = annotations["omni-lindy.io/gpu-slots-free"]
                slots_list = json.loads(raw_slots) if isinstance(raw_slots, str) else raw_slots
                if not isinstance(slots_list, list):
                    raise ValueError(f"노드 {node_names[idx]}의 gpu-slots-free는 JSON 배열이어야 합니다.")
                g_i = node_gpus[idx]
                state.gpu_slots[idx, :] = 0.0
                for s_idx, s_val in enumerate(slots_list[:g_i]):
                    state.gpu_slots[idx, s_idx] = min(1.0, max(0.0, float(s_val)))

            if "omni-lindy.io/cpu-free" in annotations:
                explicit_cpu_nodes.add(idx)
                state.cpu_free[idx] = min(state.node_cpu_cap[idx], parse_k8s_cpu(annotations["omni-lindy.io/cpu-free"]))

            if "omni-lindy.io/ram-free-gi" in annotations:
                explicit_ram_nodes.add(idx)
                state.ram_free[idx] = min(
                    state.node_ram_cap[idx],
                    _validate_finite_nonneg(float(annotations["omni-lindy.io/ram-free-gi"]), "ram-free-gi"),
                )

            if "omni-lindy.io/running-pods" in annotations:
                raw_rp = annotations["omni-lindy.io/running-pods"]
                rp_list = json.loads(raw_rp) if isinstance(raw_rp, str) else raw_rp
                if not isinstance(rp_list, list):
                    raise ValueError(f"노드 {node_names[idx]}의 running-pods는 JSON 배열이어야 합니다.")
                for r_idx, rp in enumerate(rp_list):
                    if not isinstance(rp, dict):
                        continue
                    p_gpu = _validate_finite_nonneg(float(rp.get("gpu", 1.0)), "running_pod.gpu")
                    raw_st = rp.get("startTime", now_ts)
                    st_unix = min(now_ts, parse_timestamp_unix(raw_st))
                    p_cpu = parse_k8s_cpu(rp["cpu"]) if rp.get("cpu") is not None else None
                    raw_rp_ram = rp.get("ram") if rp.get("ram") is not None else rp.get("memory")
                    p_ram = parse_k8s_memory_gi(raw_rp_ram) if raw_rp_ram is not None else None
                    uid_k, ns_k = self._split_pod_identity(rp.get("uid"), rp.get("namespace"), rp.get("name"))
                    if _find_merged_pod(uid_k, ns_k, node_idx_v=idx) is not None:
                        continue
                    rec = {
                        "node_idx": idx,
                        "st_unix": st_unix,
                        "gpu": p_gpu,
                        "cpu": p_cpu,
                        "ram": p_ram,
                        "uid_key": uid_k,
                        "ns_key": ns_k,
                        "pod_key": uid_k or ns_k or f"node:{node_names[idx]}:{st_unix:.6f}:{p_gpu:.6f}:{r_idx}",
                    }
                    merged_pods.append(rec)
                    _index_merged_pod(rec)
                    if uid_k is None and ns_k is None:
                        anon_pods_by_node.setdefault(idx, []).append(rec)

        # 2) 선택적 클러스터 레벨 `runningPods` 배열 병합 (동일 파드는 자원 보완만 수행하고 중복 차감 금지)
        if running_pods:
            for r_idx, pod in enumerate(running_pods):
                if not isinstance(pod, dict):
                    continue
                n_name = (pod.get("spec") or {}).get("nodeName")
                if not n_name or n_name not in name_to_idx:
                    continue
                idx = name_to_idx[n_name]
                p_gpu, p_cpu, p_ram, _, _ = parse_pod_resources(pod)
                raw_st = (pod.get("status") or {}).get("startTime")
                st_unix = min(now_ts, parse_timestamp_unix(raw_st)) if raw_st is not None else now_ts
                p_meta = pod.get("metadata") or {}
                uid_k, ns_k = self._split_pod_identity(p_meta.get("uid"), p_meta.get("namespace"), p_meta.get("name"))

                matched = _find_merged_pod(uid_k, ns_k, node_idx_v=idx)
                if matched is None and idx in anon_pods_by_node:
                    # 어노테이션 항목에 uid/name이 없는 레거시 형식인 경우 (node, startTime, gpu) 근접 일치로 중복 매칭
                    for cand in anon_pods_by_node[idx]:
                        if not cand.get("_matched_rp") and abs(cand["st_unix"] - st_unix) <= 1e-3 and abs(cand["gpu"] - p_gpu) <= 1e-6:
                            matched = cand
                            cand["_matched_rp"] = True
                            break

                if matched is not None:
                    # 동일 파드: 이미 등록·차감 목록에 있으므로 누락된 CPU/RAM 정보만 보완하고 중복 차감하지 않음
                    if matched["cpu"] is None:
                        matched["cpu"] = p_cpu
                    if matched["ram"] is None:
                        matched["ram"] = p_ram
                    if uid_k is not None and matched.get("uid_key") is None:
                        matched["uid_key"] = uid_k
                        matched["pod_key"] = uid_k
                        pods_by_uid[uid_k] = matched
                else:
                    rec = {
                        "node_idx": idx,
                        "st_unix": st_unix,
                        "gpu": p_gpu,
                        "cpu": p_cpu,
                        "ram": p_ram,
                        "uid_key": uid_k,
                        "ns_key": ns_k,
                        "pod_key": uid_k or ns_k or f"rp:{n_name}:{st_unix:.6f}:{p_gpu:.6f}:{r_idx}",
                    }
                    merged_pods.append(rec)
                    _index_merged_pod(rec)

        # 3) 중복 제거된 고유 파드 목록으로 상태 등록·자원 차감 및 동일 진입 코호트 동기화
        for rec_idx, rec in enumerate(merged_pods):
            idx = int(rec["node_idx"])
            st_unix = float(rec["st_unix"])
            p_gpu = float(rec["gpu"])
            p_cpu = rec["cpu"]
            p_ram = rec["ram"]
            uid_k = rec.get("uid_key")
            ns_k = rec.get("ns_key")

            self._register_running_pod(state, idx, st_unix, p_gpu, key_seed=(rec_idx + 1))
            if idx not in explicit_gpu_nodes:
                self._deduct_gpu_slots(state, idx, p_gpu)
            if idx not in explicit_cpu_nodes and p_cpu is not None:
                state.cpu_free[idx] = max(0.0, state.cpu_free[idx] - float(p_cpu))
            if idx not in explicit_ram_nodes and p_ram is not None:
                state.ram_free[idx] = max(0.0, state.ram_free[idx] - float(p_ram))

            # 완료되었거나 이미 등록된 동일 파드가 지연 스냅샷에 남아 있더라도 코호트에 중복 중도절단 추가를 방지
            # 단, 서로 다른 UID의 동명 파드는 _find_cohort_entry가 별도 코호트 항목으로 생성한다.
            existing_entry = self._find_cohort_entry(uid_k, ns_k, st_unix_hint=st_unix, prefer_open=False)
            if existing_entry is None:
                self._append_cohort_entry(uid_k, ns_k, [st_unix, None, p_gpu])
            else:
                self._bind_cohort_identity(existing_entry, uid_k, ns_k)

        # 인과적 히스토리·코호트 복사 및 상태 버전 동기화
        state.recent_completions = deque(self.recent_completions, maxlen=256)
        state.recent_cohort = deque((list(item) for item in self.recent_cohort), maxlen=1024)
        state.recent_specs = deque(self.recent_specs, maxlen=200)
        state.recent_node_counts = deque(self.recent_node_counts, maxlen=200)
        state._ops_counter = self._event_counter
        return state, node_names

    @staticmethod
    def _register_running_pod(
        state: ClusterState,
        node_idx: int,
        start_unix: float,
        gpu_req: float,
        key_seed: int,
    ) -> None:
        """노드에 실행 중인 파드의 시작 시각과 GPU 크기를 등록한다."""
        starts = state.node_job_starts[node_idx]
        starts.append(start_unix)
        starts.sort()
        state.node_oldest_start[node_idx] = starts[0]
        state.node_newest_start[node_idx] = starts[-1]
        state.node_job_count[node_idx] += 1
        state.active_jobs[key_seed] = (node_idx, start_unix, gpu_req)

    @staticmethod
    def _deduct_gpu_slots(state: ClusterState, node_idx: int, gpu_req: float) -> None:
        """실행 중인 파드의 GPU 사용량을 물리 GPU 슬롯에서 차감한다 (Best-Fit 방식)."""
        eps = 1e-9
        if gpu_req <= eps:
            return
        slots = state.gpu_slots[node_idx]
        if gpu_req < 1.0 - eps:
            feas = np.flatnonzero(slots + eps >= gpu_req)
            if len(feas) > 0:
                best_s = int(feas[np.argmin(slots[feas])])
                slots[best_s] = max(0.0, slots[best_s] - gpu_req)
        else:
            needed = int(round(gpu_req))
            empty = np.flatnonzero(slots >= 1.0 - eps)
            for s in empty[:needed]:
                slots[s] = 0.0

    def _explain_node_failure(
        self,
        state: ClusterState,
        node_idx: int,
        gpu: float,
        cpu: float,
        ram: float,
        gpu_spec: Optional[str],
    ) -> str:
        """필터링에서 탈락한 노드의 구체적인 물리 제약 사유를 생성한다."""
        eps = 1e-9
        reasons: List[str] = []
        if gpu_spec and state.node_models is not None:
            allowed = {tok.strip() for tok in gpu_spec.split("|") if tok.strip()}
            if state.node_models[node_idx] not in allowed:
                reasons.append(f"GPU spec mismatch (node={state.node_models[node_idx]}, want={gpu_spec})")
        slots = state.gpu_slots[node_idx]
        if gpu < 1.0 - eps:
            if not np.any(slots + eps >= gpu):
                reasons.append(f"insufficient fractional GPU slot (need={gpu:.3f})")
        elif gpu > eps:
            needed = int(round(gpu))
            avail = int(np.sum(slots >= 1.0 - eps))
            if avail < needed:
                reasons.append(f"insufficient whole GPUs (avail={avail}, need={needed})")
        if state.cpu_free[node_idx] + eps < cpu:
            reasons.append(f"insufficient CPU (free={state.cpu_free[node_idx]:.2f}, need={cpu:.2f})")
        if state.ram_free[node_idx] + eps < ram:
            reasons.append(f"insufficient memory GiB (free={state.ram_free[node_idx]:.2f}, need={ram:.2f})")
        return "; ".join(reasons) if reasons else "infeasible"

    def filter_nodes(self, extender_args: Dict[str, Any]) -> Dict[str, Any]:
        """`POST /filter`: Pod의 GPU 슬롯·CPU·RAM·GPU 기종 제약을 만족하는 노드만 필터링한다."""
        with self._lock:
            now_unix = self._get_now_unix(extender_args)
            nodes_items = self._extract_nodes(extender_args, now_unix=now_unix)
            pod = extender_args.get("Pod")
            if not isinstance(pod, dict):
                raise ValueError("ExtenderArgs에 'Pod' 객체가 필요합니다.")

            gpu, cpu, ram, gpu_spec, _ = parse_pod_resources(pod)
            running_pods = extender_args.get("runningPods")
            state, node_names = self.build_cluster_state(
                nodes_items, running_pods=running_pods, now_unix=now_unix, gpu_spec=gpu_spec
            )

            feasible_idx = set(int(i) for i in state.get_feasible_nodes(gpu, cpu, ram, gpu_spec=gpu_spec))
            feasible_items: List[Dict[str, Any]] = []
            feasible_names: List[str] = []
            failed_nodes: Dict[str, str] = {}

            for idx, name in enumerate(node_names):
                if idx in feasible_idx:
                    feasible_items.append(nodes_items[idx])
                    feasible_names.append(name)
                else:
                    failed_nodes[name] = self._explain_node_failure(state, idx, gpu, cpu, ram, gpu_spec)

            self.filter_calls += 1
            return {
                "Nodes": {"apiVersion": "v1", "kind": "NodeList", "items": feasible_items},
                "NodeNames": feasible_names,
                "FailedNodes": failed_nodes,
                "Error": "",
            }

    def prioritize_nodes(self, extender_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """`POST /prioritize`: `Omni-Lindy`/`Omni-MRL` 점수 계산 후 K8s v1.34 `HostPriorityList` (`0~10` 정수)를 반환한다."""
        t0 = time.perf_counter()
        with self._lock:
            now_unix = self._get_now_unix(extender_args)
            nodes_items = self._extract_nodes(extender_args, now_unix=now_unix)
            pod = extender_args.get("Pod")
            if not isinstance(pod, dict):
                raise ValueError("ExtenderArgs에 'Pod' 객체가 필요합니다.")

            gpu, cpu, ram, gpu_spec, gang_nodes = parse_pod_resources(pod)
            running_pods = extender_args.get("runningPods")

            # 현재 요청 메타데이터를 인과적 링버퍼에 기록
            self.recent_specs.append(gpu_spec)
            self.recent_node_counts.append(gang_nodes)
            self._event_counter += 1

            state, node_names = self.build_cluster_state(
                nodes_items, running_pods=running_pods, now_unix=now_unix, gpu_spec=gpu_spec
            )

            if not node_names:
                return []

            feasible = state.get_feasible_nodes(gpu, cpu, ram, gpu_spec=gpu_spec)
            # K8s 요청마다 필터링된 노드 부분집합·순서가 달라질 수 있으므로 모델 ID 캐시 초기화
            self._scheduler._cached_model_ids = None

            if len(feasible) == len(node_names):
                raw_scores = self._scheduler.score_nodes(state, feasible, gpu, cpu, ram)
                priorities = scores_to_k8s_priorities(node_names, raw_scores, high_score=10, low_score=0)
            elif len(feasible) > 0:
                raw_feas = self._scheduler.score_nodes(state, feasible, gpu, cpu, ram)
                feas_names = [node_names[int(i)] for i in feasible]
                feas_prio = scores_to_k8s_priorities(feas_names, raw_feas, high_score=10, low_score=1)
                prio_map = {item["host"]: item["score"] for item in feas_prio}
                priorities = [{"host": nm, "score": prio_map.get(nm, 0)} for nm in node_names]
            else:
                priorities = [{"host": nm, "score": 0} for nm in node_names]

            self.prioritize_calls += 1
            self.prioritize_latency_us_sum += (time.perf_counter() - t0) * 1e6
            return priorities

    def schedule_gang(self, extender_args: Dict[str, Any]) -> Dict[str, Any]:
        """`POST /gang_schedule`: 단일 랙 집적률(Rack Locality)을 극대화하는 멀티노드 Gang 배치를 계산한다."""
        with self._lock:
            now_unix = self._get_now_unix(extender_args)
            nodes_items = self._extract_nodes(extender_args, now_unix=now_unix)
            pod = extender_args.get("Pod")
            if not isinstance(pod, dict):
                raise ValueError("ExtenderArgs에 'Pod' 객체가 필요합니다.")

            gpu, cpu, ram, gpu_spec, gang_nodes = parse_pod_resources(pod)
            running_pods = extender_args.get("runningPods")

            self.recent_specs.append(gpu_spec)
            self.recent_node_counts.append(gang_nodes)
            self._event_counter += 1

            state, node_names = self.build_cluster_state(
                nodes_items, running_pods=running_pods, now_unix=now_unix, gpu_spec=gpu_spec
            )
            feasible = state.get_feasible_nodes(gpu, cpu, ram, gpu_spec=gpu_spec)
            self.gang_calls += 1

            if len(feasible) < gang_nodes:
                return {
                    "schedulable": False,
                    "selectedNodes": [],
                    "racksSpanned": 0,
                    "singleRackLocal": False,
                    "reason": f"가용 노드 부족 (feasible={len(feasible)}, required={gang_nodes})",
                }

            self._scheduler._cached_model_ids = None
            chosen_indices = self._scheduler.select_nodes_gang(
                state, feasible, gang_nodes, gpu, cpu, ram
            )
            chosen_names = [node_names[int(idx)] for idx in chosen_indices]
            racks = {int(state.node_rack_ids[int(idx)]) for idx in chosen_indices}

            return {
                "schedulable": True,
                "selectedNodes": chosen_names,
                "racksSpanned": len(racks),
                "singleRackLocal": len(racks) == 1,
                "reason": "ok",
            }

    def record_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        `POST /events/pod_completed`: 완료된 파드의 `(gpu, duration)` 및 선택적 UID/시각을 수신해
        동일 진입 코호트(`recent_cohort`), 온라인 `Corr(gpu, dur)`, CV 게이트 및 MRL 생존모델을 즉시 갱신한다.
        """
        if not isinstance(payload, dict):
            raise ValueError("완료 이벤트 페이로드는 JSON object여야 합니다.")

        if "completions" in payload:
            raw_list = payload["completions"]
            if not isinstance(raw_list, list):
                raise ValueError("'completions' 필드는 리스트여야 합니다.")
        elif "gpu" in payload and "duration" in payload:
            raw_list = [payload]
        else:
            raise ValueError("페이로드에 ('gpu', 'duration') 또는 'completions' 리스트가 필요합니다.")

        added = 0
        with self._lock:
            ref_now = self._get_now_unix(payload) if ("nowUnix" in payload or "now" in payload) else time.time()
            for item in raw_list:
                if not isinstance(item, dict):
                    raise ValueError("완료 이벤트 항목은 dict여야 합니다.")
                if "gpu" not in item or "duration" not in item:
                    raise ValueError("각 완료 항목에는 'gpu'와 'duration'이 필요합니다.")
                g_val = _validate_finite_nonneg(float(item["gpu"]), "completion.gpu")
                d_val = _validate_finite_nonneg(float(item["duration"]), "completion.duration")
                uid_k, ns_k = self._split_pod_identity(item.get("uid"), item.get("namespace"), item.get("name"))

                # 완료 이벤트 멱등성 보장: 이미 완료 처리된 UID 재전송 시 recent_completions 이중 집계를 차단
                if uid_k is not None and uid_k in self._completed_uid_set:
                    continue

                if item.get("startTime") is not None:
                    st_hint: Optional[float] = parse_timestamp_unix(item["startTime"])
                elif item.get("endTime") is not None:
                    st_hint = max(0.0, parse_timestamp_unix(item["endTime"]) - d_val)
                else:
                    st_hint = None

                matched_entry = self._find_cohort_entry(uid_k, ns_k, st_unix_hint=st_hint, prefer_open=True)
                if matched_entry is None and uid_k is not None:
                    matched_entry = self._cohort_by_uid.get(uid_k)

                if matched_entry is not None:
                    if matched_entry[1] is not None:
                        # 이미 완료 기록된 항목의 중복 완료 이벤트를 무시한다.
                        continue
                    matched_entry[1] = float(matched_entry[0]) + d_val  # type: ignore[arg-type]
                    self._bind_cohort_identity(matched_entry, uid_k, ns_k)
                else:
                    st_val = st_hint if st_hint is not None else max(0.0, ref_now - d_val)
                    self._append_cohort_entry(uid_k, ns_k, [st_val, st_val + d_val, g_val])

                if uid_k is not None:
                    if len(self._completed_uid_order) == self._completed_uid_order.maxlen:
                        old_uid = self._completed_uid_order.popleft()
                        self._completed_uid_set.discard(old_uid)
                    self._completed_uid_order.append(uid_k)
                    self._completed_uid_set.add(uid_k)

                self.recent_completions.append((g_val, d_val))
                added += 1

            if added > 0:
                self.completion_events += added
                self._event_counter += 1

                # 다음 `/prioritize`를 기다리지 않고 `/metrics`에서도 최신 게이트와 코호트 KM MRL을 즉시 갱신
                probe_state = ClusterState(ClusterConfig(num_nodes=1))
                probe_state.now = ref_now
                probe_state.recent_completions = deque(self.recent_completions, maxlen=256)
                probe_state.recent_cohort = deque((list(x) for x in self.recent_cohort), maxlen=1024)
                probe_state._ops_counter = self._event_counter
                self._scheduler._last_stats_ops = -10**9
                self._scheduler._update_online_gate(probe_state)
                if isinstance(self._scheduler, OmniMRLScheduler):
                    if len(self.recent_completions) >= 12 and self._scheduler._cached_ref_scale > 0.0:
                        self._scheduler._cached_gamma_ifr = self._scheduler._cv_to_gamma_ifr(self._scheduler._smoothed_cv)
                        self._scheduler.survival.update(probe_state, force=True)
                    else:
                        self._scheduler._cached_gamma_ifr = 0.0
                    gamma_ifr_val = float(self._scheduler._cached_gamma_ifr)
                else:
                    gamma_ifr_val = 0.0
            else:
                gamma_ifr_val = (
                    float(self._scheduler._cached_gamma_ifr)
                    if isinstance(self._scheduler, OmniMRLScheduler)
                    else 0.0
                )

            return {
                "status": "ok",
                "added": added,
                "completions_buffered": len(self.recent_completions),
                "cohort_size": len(self.recent_cohort),
                "smoothed_rho": float(self._scheduler._smoothed_rho),
                "alpha": float(self._scheduler._cached_alpha),
                "smoothed_cv": float(self._scheduler._smoothed_cv),
                "gamma": float(self._scheduler._cached_gamma),
                "gamma_ifr": gamma_ifr_val,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """현재 스케줄러 온라인 상태 및 HTTP 처리량 요약을 JSON 딕셔너리로 반환한다."""
        with self._lock:
            avg_lat_us = (
                self.prioritize_latency_us_sum / self.prioritize_calls
                if self.prioritize_calls > 0
                else 0.0
            )
            gamma_ifr_val = (
                float(self._scheduler._cached_gamma_ifr)
                if isinstance(self._scheduler, OmniMRLScheduler)
                else 0.0
            )
            cohort_comp = sum(1 for item in self.recent_cohort if item[1] is not None)
            cohort_cens = len(self.recent_cohort) - cohort_comp
            return {
                "scheduler": "Omni-MRL" if self.engine == "omni_mrl" else "Omni-Lindy",
                "engine": self.engine,
                "version": "11.1" if self.engine == "omni_mrl" else "7.0",
                "mrl_mode": "ifr_nilas" if gamma_ifr_val >= 0.5 else "dfr_lindy",
                "filter_calls": self.filter_calls,
                "prioritize_calls": self.prioritize_calls,
                "gang_calls": self.gang_calls,
                "completion_events": self.completion_events,
                "completions_buffered": len(self.recent_completions),
                "cohort_size": len(self.recent_cohort),
                "cohort_completed": cohort_comp,
                "cohort_censored": cohort_cens,
                "avg_prioritize_latency_us": avg_lat_us,
                "smoothed_rho": float(self._scheduler._smoothed_rho),
                "alpha_decoupling_gate": float(self._scheduler._cached_alpha),
                "smoothed_cv": float(self._scheduler._smoothed_cv),
                "gamma_lindy_validity": float(self._scheduler._cached_gamma),
                "gamma_ifr": gamma_ifr_val,
                "ref_time_scale": float(self._scheduler._cached_ref_scale),
            }

    def format_prometheus_metrics(self) -> str:
        """Prometheus text exposition 포맷(`0.0.4`) 문자열을 생성한다."""
        m = self.get_metrics()
        lines = [
            "# HELP omni_lindy_filter_calls_total Total /filter requests processed.",
            "# TYPE omni_lindy_filter_calls_total counter",
            f"omni_lindy_filter_calls_total {m['filter_calls']}",
            "# HELP omni_lindy_prioritize_calls_total Total /prioritize requests processed.",
            "# TYPE omni_lindy_prioritize_calls_total counter",
            f"omni_lindy_prioritize_calls_total {m['prioritize_calls']}",
            "# HELP omni_lindy_gang_calls_total Total /gang_schedule requests processed.",
            "# TYPE omni_lindy_gang_calls_total counter",
            f"omni_lindy_gang_calls_total {m['gang_calls']}",
            "# HELP omni_lindy_prioritize_avg_latency_microseconds Mean latency of /prioritize in microseconds.",
            "# TYPE omni_lindy_prioritize_avg_latency_microseconds gauge",
            f"omni_lindy_prioritize_avg_latency_microseconds {m['avg_prioritize_latency_us']:.3f}",
            "# HELP omni_lindy_smoothed_rho Online EWMA correlation between GPU size and age/duration.",
            "# TYPE omni_lindy_smoothed_rho gauge",
            f"omni_lindy_smoothed_rho {m['smoothed_rho']:.6f}",
            "# HELP omni_lindy_alpha_gate Online decoupling gate alpha in (0, 1).",
            "# TYPE omni_lindy_alpha_gate gauge",
            f"omni_lindy_alpha_gate {m['alpha_decoupling_gate']:.6f}",
            "# HELP omni_lindy_gamma_gate Online heavy-tail CV validity gate gamma in (0, 1).",
            "# TYPE omni_lindy_gamma_gate gauge",
            f"omni_lindy_gamma_gate {m['gamma_lindy_validity']:.6f}",
            "# HELP omni_mrl_gamma_ifr_gate Online IFR synchronized-exit MRL gate gamma_ifr in [0, 1].",
            "# TYPE omni_mrl_gamma_ifr_gate gauge",
            f"omni_mrl_gamma_ifr_gate {m['gamma_ifr']:.6f}",
            "# HELP omni_mrl_cohort_size Number of entries in the causal same-entry cohort buffer.",
            "# TYPE omni_mrl_cohort_size gauge",
            f"omni_mrl_cohort_size {m['cohort_size']}",
        ]
        return "\n".join(lines) + "\n"
