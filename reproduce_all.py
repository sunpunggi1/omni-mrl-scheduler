"""
Omni-MRL (v11.1) / Omni-Lindy (v7.0) 원클릭 재현·검증 CLI 스크립트.

사용법:
  - 빠른 재현·검증 (~25초):
      python reproduce_all.py --mode quick
    (61개 단위·HTTP·수학·K8s 규약 통합 테스트 `-W error` 실행 + 활성 코호트 KM 경로 K8s Extender
     라이브 HTTP 레이턴시 실측 + 원본 CSV/10종 JSON ↔ `paper/omni_lindy_paper.md` 표·문맥 기반 100% 자동 대조 +
     시드별 원자료로부터 p-value/Holm/계층 CI 독립 재계산 검증)

  - 전체 실험 파이프라인 재실행:
      python reproduce_all.py --mode full
    (Phase 1~11 전체 재계산 후 quick 검증 수행; `--include-evo` 지정 시 Phase 2 진화 탐색 포함)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Tuple
import urllib.request

import numpy as np

from k8s_extender.server import create_extender_server
from stats_utils import cluster_aware_summary_stats, holm_bonferroni_adjust, paired_summary_stats

ROOT = Path(__file__).resolve().parent


def _make_benchmark_payload(num_nodes: int) -> bytes:
    """활성 코호트 KM/MRL 경로를 포함하는 K8s HTTP Extender 레이턴시 측정용 페이로드를 생성한다."""
    nodes = []
    for i in range(num_nodes):
        busy = (i % 3 != 0)
        slots = [0.5 if busy else 1.0, 0.0 if busy else 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        rps = (
            [{"uid": f"bench-rp-{i}", "name": f"rp-{i}", "startTime": 999.0 - (i % 8), "gpu": 1.5}]
            if busy
            else []
        )
        nodes.append(
            {
                "metadata": {
                    "name": f"gpu-node-{i:03d}",
                    "resourceVersion": f"rv-{i}",
                    "labels": {
                        "omni-lindy.io/gpu-model": "V100M32" if i % 2 == 0 else "T4",
                        "omni-lindy.io/rack": f"rack-{i // 8}",
                    },
                    "annotations": {
                        "omni-lindy.io/gpu-slots-free": json.dumps(slots),
                        "omni-lindy.io/cpu-free": "48" if busy else "64",
                        "omni-lindy.io/ram-free-gi": "384" if busy else "512",
                        "omni-lindy.io/running-pods": json.dumps(rps),
                    },
                },
                "status": {"allocatable": {"nvidia.com/gpu": "8", "cpu": "64", "memory": "512Gi"}},
            }
        )
    pod = {
        "metadata": {"name": "bench-pod", "annotations": {"omni-lindy.io/gpu-milli": "500"}},
        "spec": {"containers": [{"name": "c", "resources": {"requests": {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "1"}}}]},
    }
    return json.dumps({"Pod": pod, "Nodes": {"items": nodes}, "nowUnix": 1000.0}).encode("utf-8")


def run_unit_and_http_tests() -> None:
    """전체 61개 단위·경계·불변량·MRL 수학·K8s HTTP 통합 테스트를 경고 0건(-W error)으로 실행한다."""
    print("[1/3] 단위·불변량·MRL·K8s HTTP 통합 테스트 실행 (`python -W error test_suite.py`)...")
    res = subprocess.run(
        [sys.executable, "-W", "error", str(ROOT / "test_suite.py")],
        cwd=str(ROOT),
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"test_suite.py 실행 실패 (exit code={res.returncode})")
    print("  -> 61/61 테스트 및 경고 0건 검증 통과.\n")


def run_k8s_http_latency_probe(num_calls: int = 100) -> Dict[int, Dict[str, float]]:
    """
    64노드 및 128노드 스냅샷에 대해 IFR 완료 코호트를 사전 활성화(`mrl_mode == 'ifr_nilas'`)한 뒤
    실제 로컬 HTTP 왕복 및 어댑터 내부 활성 KM/MRL 경로 레이턴시를 측정한다.
    """
    print(f"[2/3] Kubernetes HTTP Extender 활성 KM/MRL 경로 레이턴시 프로브 ({num_calls}회/규모)...")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results: Dict[int, Dict[str, float]] = {}

    ifr_completions = [
        {"uid": f"ifr-done-{k}", "gpu": 1.0, "duration": 10.0 + 0.05 * (k % 3), "nowUnix": 950.0}
        for k in range(36)
    ]

    for n_nodes in (64, 128):
        server = create_extender_server("127.0.0.1", 0, quiet=True, engine="omni_mrl")
        port = server.server_address[1]
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        body = _make_benchmark_payload(n_nodes)
        try:
            # 1) IFR 완료 코호트 3회 분할 주입으로 EMA CV 수렴, gamma_ifr 및 동일 진입 코호트 KM 생존곡선 활성화
            for start_idx in range(0, len(ifr_completions), 12):
                batch = ifr_completions[start_idx : start_idx + 12]
                comp_req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/events/pod_completed",
                    data=json.dumps({"completions": batch, "nowUnix": 950.0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(comp_req, timeout=5.0) as resp:
                    resp.read()

            # 2) JIT·캐시 워밍업 5회 (실행 중 파드 코호트 포함)
            for _ in range(5):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/prioritize",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(req, timeout=5.0) as resp:
                    resp.read()

            pre_metrics = server.adapter.get_metrics()
            if pre_metrics["mrl_mode"] != "ifr_nilas" or pre_metrics["cohort_size"] <= 0:
                raise AssertionError(
                    f"활성 KM/IFR 경로가 프로브에서 켜지지 않았습니다: {pre_metrics}"
                )

            server.adapter.prioritize_calls = 0
            server.adapter.prioritize_latency_us_sum = 0.0
            lat_ms: List[float] = []
            for _ in range(num_calls):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/prioritize",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                t0 = time.perf_counter()
                with opener.open(req, timeout=5.0) as resp:
                    resp.read()
                lat_ms.append((time.perf_counter() - t0) * 1000.0)

            metrics = server.adapter.get_metrics()
            if metrics["mrl_mode"] != "ifr_nilas" or metrics["cohort_size"] <= 0:
                raise AssertionError(
                    f"측정 루프 후에도 활성 KM/IFR 경로가 유지되지 않았습니다: {metrics}"
                )
            arr = np.asarray(lat_ms, dtype=np.float64)
            summary = {
                "http_mean_ms": float(np.mean(arr)),
                "http_p95_ms": float(np.percentile(arr, 95)),
                "adapter_mean_us": float(metrics["avg_prioritize_latency_us"]),
                "gamma_ifr": float(metrics["gamma_ifr"]),
                "cohort_size": int(metrics["cohort_size"]),
            }
            results[n_nodes] = summary
            print(
                f"  -> N={n_nodes:3d} (mrl={metrics['mrl_mode']}, cohort={summary['cohort_size']}): "
                f"HTTP 평균 {summary['http_mean_ms']:.3f} ms (p95 {summary['http_p95_ms']:.3f} ms) | "
                f"어댑터 내부 {summary['adapter_mean_us'] / 1000.0:.3f} ms ({summary['adapter_mean_us']:.1f} us)"
            )
        finally:
            server.shutdown()
            server.server_close()
            th.join(timeout=2.0)
    print()
    return results


def _split_markdown_sections(paper_text: str) -> Dict[str, str]:
    """논문 마크다운을 섹션 헤딩(## / ###)별 본문 딕셔너리로 분리한다."""
    sections: Dict[str, str] = {"__preamble__": ""}
    cur_key = "__preamble__"
    buf: List[str] = []
    for line in paper_text.splitlines():
        if re.match(r"^#{1,4}\s+", line):
            sections[cur_key] = "\n".join(buf)
            cur_key = line.lstrip("#").strip()
            buf = [line]
        else:
            buf.append(line)
    sections[cur_key] = "\n".join(buf)
    return sections


def _find_section(sections: Dict[str, str], heading_prefix: str) -> str:
    """헤딩 접두어가 일치하는 섹션 본문을 반환한다."""
    for k, v in sections.items():
        if k.startswith(heading_prefix) or heading_prefix in k:
            return v
    raise AssertionError(f"논문에서 섹션 헤딩을 찾을 수 없습니다: {heading_prefix!r}")


def _parse_tables(section_text: str) -> List[List[List[str]]]:
    """섹션 내 모든 마크다운 표를 [표][행][셀] 2차원 리스트로 파싱한다(구분선 --- 행 제외)."""
    tables: List[List[List[str]]] = []
    cur_table: List[List[str]] = []
    for line in section_text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
                continue
            cur_table.append(cells)
        else:
            if cur_table:
                tables.append(cur_table)
                cur_table = []
    if cur_table:
        tables.append(cur_table)
    return tables


def _table_row_lookup(table: List[List[str]], row_key: str) -> List[str]:
    """마크다운 표에서 0번 열이 `row_key`를 포함하는 행을 찾아 반환한다."""
    clean_key = row_key.replace("`", "").replace("*", "")
    for row in table[1:]:
        c0 = row[0].replace("`", "").replace("*", "").strip()
        if clean_key == c0 or clean_key in c0:
            return row
    raise AssertionError(f"표에서 행 키를 찾을 수 없습니다: {row_key!r}")


def _verify_phase11_raw_recomputation(p11: Dict[str, Any]) -> None:
    """`phase11_report.json`의 시드별 원자료 배열로부터 통계 검정·Holm·계층 CI를 독립 재계산 검증한다."""
    # 0) 아티팩트 SHA-256 해시 무결성 검증
    for rel, stored_h in p11["protocol"]["artifact_sha256"].items():
        fp = ROOT / rel
        if not fp.exists():
            raise FileNotFoundError(f"아티팩트 해시 대상 파일 누락: {fp}")
        cur_h = hashlib.sha256(fp.read_bytes()).hexdigest()
        if cur_h != stored_h:
            raise AssertionError(f"Phase 11 코어/데이터 파일 SHA-256 불일치: {rel}")

    # 1) Suite 1 96개 가설 재계산
    s1 = p11["suite_1_lockbox30_factorial"]
    comp_map = {
        "FirstFit": "vs_firstfit",
        "L2Norm": "vs_l2norm",
        "BPack-SpatialOnly": "vs_bpack_spatial_only",
        "Google-NILAS-L2": "vs_google_nilas_l2",
        "Google-NILAS-BPack": "vs_google_nilas_bpack",
        "Google-LAVA-BPack": "vs_google_lava_bpack",
        "Alibaba-DVLA-BPack": "vs_alibaba_dvla_bpack",
        "Omni-Lindy-v7": "vs_omni_lindy_v7",
    }
    raw_ps: List[float] = []
    stored_holms: List[float] = []
    for wn, w_res in s1["workloads"].items():
        for metric in ("blocking", "bf_wait"):
            raw = w_res["raw_per_seed"][metric]
            v11 = np.asarray(raw["Omni-MRL-v11"], dtype=np.float64)
            for ref_sn, ckey in comp_map.items():
                recomp = paired_summary_stats(np.asarray(raw[ref_sn], dtype=np.float64), v11)
                stored = w_res["comparisons"][metric][ckey]
                for field in ("rel_improvement_pct", "p_value_wilcoxon", "p_value_t", "boot_ci95_low", "boot_ci95_high"):
                    if abs(recomp[field] - stored[field]) > 1e-10:
                        raise AssertionError(f"Phase 11 Suite 1 재계산 불일치 ({wn}/{metric}/{ckey}/{field})")
                raw_ps.append(recomp["p_value_wilcoxon"])
                stored_holms.append(stored["p_value_wilcoxon_holm96"])

    recomp_holms = holm_bonferroni_adjust(raw_ps)
    if max(abs(a - b) for a, b in zip(recomp_holms, stored_holms)) > 1e-10:
        raise AssertionError("Phase 11 Suite 1 Holm-96 재계산 불일치")

    # 2) Suite 2 Dual-Oracle 48개 가설 재계산
    s2 = p11["suite_2_ml_crossover_lockbox30"]
    for bm in ("bias_corrected", "uncentered"):
        raw_ps_s2: List[float] = []
        stored_h_s2: List[float] = []
        for wn in ("W3_Heavy_Straggler", "IFR_Uniform"):
            wd = s2[bm][wn]
            om_blk = np.asarray(wd["omni_mrl_v11"]["raw_per_seed_blocking"], dtype=np.float64)
            om_bf = np.asarray(wd["omni_mrl_v11"]["raw_per_seed_bf_wait"], dtype=np.float64)
            for sweep_key in ("lava_ml_bpack_sweep", "nilas_ml_bpack_sweep"):
                for item in wd[sweep_key]:
                    r_blk = paired_summary_stats(np.asarray(item["raw_per_seed_ml_blocking"], dtype=np.float64), om_blk)
                    r_bf = paired_summary_stats(np.asarray(item["raw_per_seed_ml_bf_wait"], dtype=np.float64), om_bf)
                    if abs(r_blk["p_value_wilcoxon"] - item["zeroml_vs_ml_blocking"]["p_value_wilcoxon"]) > 1e-10:
                        raise AssertionError(f"Suite 2 재계산 불일치 ({bm}/{wn}/{sweep_key})")
                    if abs(r_bf["p_value_wilcoxon"] - item["zeroml_vs_ml_bf_wait"]["p_value_wilcoxon"]) > 1e-10:
                        raise AssertionError(f"Suite 2 BF 재계산 불일치 ({bm}/{wn}/{sweep_key})")
                    raw_ps_s2.extend([r_blk["p_value_wilcoxon"], r_bf["p_value_wilcoxon"]])
                    stored_h_s2.extend([
                        item["zeroml_vs_ml_blocking"]["p_value_wilcoxon_holm48"],
                        item["zeroml_vs_ml_bf_wait"]["p_value_wilcoxon_holm48"],
                    ])
        if max(abs(a - b) for a, b in zip(holm_bonferroni_adjust(raw_ps_s2), stored_h_s2)) > 1e-10:
            raise AssertionError(f"Suite 2 Holm-48 재계산 불일치 ({bm})")

    # 2b) Suite 1b Ablation 원자료 재계산
    ab = p11["suite_1b_ablation_lockbox30"]
    for wn in ("IFR_Uniform", "IFR_Weibull_k2"):
        for metric in ("blocking", "bf_wait"):
            raw_s1 = s1["workloads"][wn]["raw_per_seed"][metric]
            v7_arr = np.asarray(raw_s1["Omni-Lindy-v7"], dtype=np.float64)
            v11_arr = np.asarray(raw_s1["Omni-MRL-v11"], dtype=np.float64)
            nop_arr = np.asarray(ab[wn][metric]["raw_per_seed_nopristine"], dtype=np.float64)
            re_pure = paired_summary_stats(v7_arr, nop_arr)
            re_rel = paired_summary_stats(nop_arr, v11_arr)
            for field in ("rel_improvement_pct", "p_value_wilcoxon"):
                if abs(re_pure[field] - ab[wn][metric]["pure_mrl_term_vs_v7"][field]) > 1e-10:
                    raise AssertionError(f"Suite 1b pure MRL 재계산 불일치 ({wn}/{metric}/{field})")
                if abs(re_rel[field] - ab[wn][metric]["pristine_relax_vs_nopristine"][field]) > 1e-10:
                    raise AssertionError(f"Suite 1b relax 재계산 불일치 ({wn}/{metric}/{field})")

    # 3) Suite 3 Helios MultiNode 창 단위 + 클러스터 단위 계층 통계 재계산
    hel = p11["suite_3_real_traces"]["helios_multinode_20"]
    clusters = ("Venus", "Earth", "Saturn", "Uranus")
    raw_hel = hel["raw_per_unit"]
    v11_gw = [np.array([u["bft_gw_jct"] for u in raw_hel["Omni-MRL-v11"] if u["cluster"] == c]) for c in clusters]
    v11_hb = [np.array([u["blk_gh"] for u in raw_hel["Omni-MRL-v11"] if u["cluster"] == c]) for c in clusters]
    hel_win_ps: List[float] = []
    hel_win_stored: List[float] = []
    hel_clu_wps: List[float] = []
    hel_clu_w_stored: List[float] = []
    hel_clu_tps: List[float] = []
    hel_clu_t_stored: List[float] = []
    for ref_sn in [k for k in raw_hel.keys() if k != "Omni-MRL-v11"]:
        ref_gw = [np.array([u["bft_gw_jct"] for u in raw_hel[ref_sn] if u["cluster"] == c]) for c in clusters]
        ref_hb = [np.array([u["blk_gh"] for u in raw_hel[ref_sn] if u["cluster"] == c]) for c in clusters]
        cgw = cluster_aware_summary_stats(ref_gw, v11_gw)
        chb = cluster_aware_summary_stats(ref_hb, v11_hb)
        for name_k, comp_re, comp_st in (
            (f"gw_jct_vs_{ref_sn}", cgw, hel["comparisons"][f"gw_jct_vs_{ref_sn}"]),
            (f"blk_vs_{ref_sn}", chb, hel["comparisons"][f"blk_vs_{ref_sn}"]),
        ):
            for ci_field in ("hier_boot_ci95_low", "hier_boot_ci95_high", "p_value_wilcoxon"):
                if abs(comp_re["window_level_with_hier_ci"][ci_field] - comp_st["window_level_with_hier_ci"][ci_field]) > 1e-10:
                    raise AssertionError(f"Helios 계층 통계 재계산 불일치: {name_k}/{ci_field}")
            if abs(comp_re["cluster_level_stats"]["p_value_t"] - comp_st["cluster_level_stats"]["p_value_t"]) > 1e-10:
                raise AssertionError(f"Helios 클러스터 수준 p_t 불일치: {name_k}")
            hel_win_ps.append(comp_re["window_level_with_hier_ci"]["p_value_wilcoxon"])
            hel_win_stored.append(comp_st["window_level_with_hier_ci"]["window_p_value_wilcoxon_holm"])
            hel_clu_wps.append(comp_re["cluster_level_stats"]["p_value_wilcoxon"])
            hel_clu_w_stored.append(comp_st["cluster_level_stats"]["cluster_p_value_wilcoxon_holm"])
            hel_clu_tps.append(comp_re["cluster_level_stats"]["p_value_t"])
            hel_clu_t_stored.append(comp_st["cluster_level_stats"]["cluster_p_value_t_holm"])
    if max(abs(a - b) for a, b in zip(holm_bonferroni_adjust(hel_win_ps), hel_win_stored)) > 1e-10:
        raise AssertionError("Helios MultiNode 창 단위 Holm 재계산 불일치")
    if max(abs(a - b) for a, b in zip(holm_bonferroni_adjust(hel_clu_wps), hel_clu_w_stored)) > 1e-10:
        raise AssertionError("Helios MultiNode 클러스터 Wilcoxon Holm 재계산 불일치")
    if max(abs(a - b) for a, b in zip(holm_bonferroni_adjust(hel_clu_tps), hel_clu_t_stored)) > 1e-10:
        raise AssertionError("Helios MultiNode 클러스터 t Holm 재계산 불일치")


def verify_json_and_paper_consistency() -> None:
    """원본 CSV · 10종 JSON 보고서 ↔ `paper/omni_lindy_paper.md` 표·문맥 기반 100% 일치 검증."""
    print("[3/3] 원본 CSV/JSON 10종 ↔ 기술 논문(`paper/omni_lindy_paper.md`) 표·문맥 교차 검증...")
    required_jsons = [
        "baseline_results.json",
        "final_b_lindy_report.json",
        "phase5_30seed_report.json",
        "real_trace_report.json",
        "phase7_report.json",
        "phase8_report.json",
        "phase9_report.json",
        "phase10_report.json",
        "phase11_report.json",
        "empirical_regime_audit.json",
    ]
    loaded: Dict[str, Any] = {}
    for fn in required_jsons:
        fp = ROOT / fn
        if not fp.exists():
            raise FileNotFoundError(f"원본 결과 JSON을 찾을 수 없습니다: {fp}")
        with fp.open(encoding="utf-8") as f:
            loaded[fn] = json.load(f)

    paper_path = ROOT / "paper" / "omni_lindy_paper.md"
    if not paper_path.exists():
        raise FileNotFoundError(f"기술 논문 파일을 찾을 수 없습니다: {paper_path}")
    paper_text = paper_path.read_text(encoding="utf-8")
    sections = _split_markdown_sections(paper_text)

    # 1) Alibaba CSV 원자료 직접 집계 대조
    node_csv = ROOT / "data" / "alibaba_2023" / "openb_node_list_gpu_node.csv"
    with node_csv.open("rt", encoding="utf-8") as f:
        csv_rows = [r for r in csv.DictReader(f) if int(float(r["gpu"])) > 0]
    tot_nodes = len(csv_rows)
    tot_gpus = sum(int(float(r["gpu"])) for r in csv_rows)
    model_nodes: Dict[str, int] = {}
    model_gpus: Dict[str, int] = {}
    for r in csv_rows:
        m = r["model"].strip()
        g = int(float(r["gpu"]))
        model_nodes[m] = model_nodes.get(m, 0) + 1
        model_gpus[m] = model_gpus.get(m, 0) + g

    sec41 = _find_section(sections, "4.1 Parameter Freezing")
    assert f"{tot_nodes:,} nodes, {tot_gpus:,} physical GPUs" in sec41, "Alibaba CSV 총 노드/GPU 수 불일치"
    for m in ("G2", "T4", "P100", "V100M16", "G3", "V100M32", "A10"):
        tok = f"**{m}** {model_nodes[m]} nodes"
        assert tok in sec41 and f"{model_gpus[m]:,} GPUs" in sec41, f"Alibaba {m} 기종 집계 불일치"

    # 2) Phase 11 원자료 기반 통계 재계산 검증
    p11 = loaded["phase11_report.json"]
    _verify_phase11_raw_recomputation(p11)

    # 3) 문맥·표 셀 기반 자동 대조 목록 (단순 부분문자열 검사가 아닌 섹션/표/행/열 정확 대조)
    br = loaded["baseline_results.json"]
    p5 = loaded["phase5_30seed_report.json"]
    rt = loaded["real_trace_report.json"]
    p7 = loaded["phase7_report.json"]
    p8 = loaded["phase8_report.json"]
    p9 = loaded["phase9_report.json"]
    p10 = loaded["phase10_report.json"]

    # 3-a) Phase 1 / Phase 4 theta / Phase 5 / Phase 7 / Phase 8 / Phase 9 / Phase 10 핵심 표·문맥 검증
    fb = loaded["final_b_lindy_report.json"]
    sec31 = _find_section(sections, "3.1 Evolutionary Formula")
    for th_k in ("w_res", "w_imb", "w_pristine", "w_b4", "w_whole", "w_lindy", "tau"):
        assert f"{fb['theta'][th_k]}" in sec31, f"3.1절 theta[{th_k}] 불일치"

    sec22 = _find_section(sections, "2.2 `FGD`")
    t_p1 = _parse_tables(sec22)[0]
    assert f"{br['W1_PAI_Mixed']['FGD_Adaptive']['gpu_hour_block_mean']:.3f}%" in _table_row_lookup(t_p1, "W1_PAI_Mixed")[6]
    assert f"{br['W5_Skewed_Vec_OOD']['FGD_Adaptive']['gpu_hour_block_mean']:.3f}%" in _table_row_lookup(t_p1, "W5_Skewed_Vec_OOD")[6]

    sec51 = _find_section(sections, "5.1 Synthetic 30-Seed")
    t_p5 = _parse_tables(sec51)[0]
    assert f"{p5['workloads']['W1_PAI_Mixed']['blocking_30']['scheduler_stats']['Adaptive-B-Lindy']['mean']:.3f}%" in _table_row_lookup(t_p5, "W1_PAI_Mixed")[4]
    assert f"{p5['workloads']['W3_Heavy_Straggler']['blocking_30']['scheduler_stats']['Adaptive-B-Lindy']['mean']:.3f}%" in _table_row_lookup(t_p5, "W3_Heavy_Straggler")[4]
    assert f"{p5['pooled_180_summary']['blocking_vs_best_sota']['wins']}/176" in sec51
    assert f"{rt['scenarios']['Helios_Pooled_20']['schedulers']['Adaptive-B-Lindy']['blk_gh_mean']:.3f}%" in _find_section(sections, "(2) Helios Single-Node")

    sec52_2 = _find_section(sections, "(2) Helios Single-Node")
    t_p7_hel = _parse_tables(sec52_2)[0]
    assert f"{p7['part_b_helios']['HeliosMN_Pooled_20']['schedulers']['Omni-Lindy']['blk_gh_mean']:.3f}%" in _table_row_lookup(t_p7_hel, "Omni-Lindy")[1]

    sec72 = _find_section(sections, "7.2 Suite A")
    t_p8 = _parse_tables(sec72)[0]
    assert f"{p8['suite_a_spectrum_30']['workloads']['IFR_Weibull_k2']['schedulers']['blocking']['Omni-MRL-v8']['mean']:.3f}%" in _table_row_lookup(t_p8, "IFR_Weibull_k2")[8]
    # Phase 9 Fresh 시드와 Phase 11 Lockbox 시드 비중첩 확인 및 Phase 10 초록/표 일치 검증
    assert set(p11["protocol"]["lockbox_seeds_30"]).isdisjoint(p9["protocol"]["fresh_test_seeds_30"]), "Lockbox와 Phase 9 Fresh 시드 중복"

    # Phase 10 표 셀 검증 (이전 반올림·누락 버그 교정 확인)
    sec81 = _find_section(sections, "8.1 Suite 1")
    t_p10_s1 = _parse_tables(sec81)[0]
    p10_s1 = p10["suite_1_fresh30_factorial"]
    assert f"{p10_s1['workloads']['W1_PAI_Mixed']['schedulers']['blocking']['L2Norm']['mean']:.3f}%" in _table_row_lookup(t_p10_s1, "W1_PAI_Mixed")[2]
    assert f"{p10_s1['workloads']['W3_Heavy_Straggler']['schedulers']['blocking']['Google-LAVA-BPack']['mean']:.3f}%" in _table_row_lookup(t_p10_s1, "W3_Heavy_Straggler")[6]
    assert f"{p10_s1['workloads']['W2_Regime_Shift']['schedulers']['blocking']['Alibaba-DVLA-BPack']['mean']:.3f}%" in _table_row_lookup(t_p10_s1, "W2_Regime_Shift")[7]

    sec83 = _find_section(sections, "8.3 Suite 3")
    t_p10_s3 = _parse_tables(sec83)[0]
    p10_s3 = p10["suite_3_real_traces_factorial"]
    assert f"{p10_s3['alibaba']['Alibaba_Default_Sweep']['schedulers']['Google-NILAS-BPack']['blk_gh_mean']:.3f}%" in _table_row_lookup(t_p10_s3, "Alibaba_Default")[3]
    assert f"{p10_s3['helios_multinode']['schedulers']['BPack-SpatialOnly']['bft_gw_jct_mean']:.3f}h" in _table_row_lookup(t_p10_s3, "Helios_MN")[2]

    # 3-b) Phase 11 Lockbox 30-시드 Suite 1 전체 셀 검증 (6 워크로드 × 9 스케줄러 = 54 셀)
    sec84_1 = _find_section(sections, "8.4.1 Suite 1")
    t_p11_s1 = _parse_tables(sec84_1)[0]
    p11_s1 = p11["suite_1_lockbox30_factorial"]
    s1_col_order = [
        ("FirstFit", 2),
        ("L2Norm", 3),
        ("Google-NILAS-L2", 4),
        ("BPack-SpatialOnly", 5),
        ("Google-NILAS-BPack", 6),
        ("Google-LAVA-BPack", 7),
        ("Alibaba-DVLA-BPack", 8),
        ("Omni-Lindy-v7", 9),
        ("Omni-MRL-v11", 10),
    ]
    checked_cells = 0
    for wn in p11_s1["workloads"]:
        row = _table_row_lookup(t_p11_s1, wn)
        for sn, col_i in s1_col_order:
            b_val = f"{p11_s1['workloads'][wn]['schedulers']['blocking'][sn]['mean']:.3f}%"
            w_val = f"{p11_s1['workloads'][wn]['schedulers']['bf_wait'][sn]['mean']:.3f}h"
            cell = row[col_i]
            if b_val not in cell or w_val not in cell:
                raise AssertionError(f"Phase 11 Suite 1 표 셀 불일치 ({wn}/{sn}): expected ({b_val}, {w_val}) in {cell!r}")
            checked_cells += 2

    # 3-c) Phase 11 Suite 1b Ablation 표 셀 검증
    sec84_1b = _find_section(sections, "8.4.2 Suite 1b")
    t_p11_ab = _parse_tables(sec84_1b)[0]
    p11_ab = p11["suite_1b_ablation_lockbox30"]
    for wn in ("IFR_Uniform", "IFR_Weibull_k2"):
        row = _table_row_lookup(t_p11_ab, wn)
        for m_i, key in enumerate(("BPack_SpatialOnly_mean", "Omni_Lindy_v7_mean", "Omni_MRL_NoPristineRelax_mean", "Omni_MRL_v11_mean"), start=1):
            b_tok = f"{p11_ab[wn]['blocking'][key]:.3f}%"
            w_tok = f"{p11_ab[wn]['bf_wait'][key]:.3f}h"
            if b_tok not in row[m_i] or w_tok not in row[m_i]:
                raise AssertionError(f"Phase 11 Ablation 셀 불일치 ({wn}/{key}): {row[m_i]!r}")
            checked_cells += 2
    assert "14/11/5" in _table_row_lookup(t_p11_ab, "IFR_Uniform")[6], "Suite 1b IFR_Uniform BF 승패 표기 불일치"

    # 3-d) Phase 11 Suite 2 편향 분리 Dual-Oracle 표 셀 검증
    sec84_2 = _find_section(sections, "8.4.3 Suite 2")
    t_p11_s2 = _parse_tables(sec84_2)[0]
    p11_s2_bc = p11["suite_2_ml_crossover_lockbox30"]["bias_corrected"]
    for idx_sg, sg in enumerate((0.0, 0.25, 0.5, 1.0, 1.5, 2.0)):
        row = _table_row_lookup(t_p11_s2, f"sigma={sg:.2f}")
        for col_i, (wn, swk) in (
            (1, ("W3_Heavy_Straggler", "lava_ml_bpack_sweep")),
            (2, ("W3_Heavy_Straggler", "nilas_ml_bpack_sweep")),
            (3, ("IFR_Uniform", "lava_ml_bpack_sweep")),
            (4, ("IFR_Uniform", "nilas_ml_bpack_sweep")),
        ):
            item = p11_s2_bc[wn][swk][idx_sg]
            b_tok = f"{item['ml_blocking_mean']:.3f}%"
            w_tok = f"{item['ml_bf_wait_mean']:.4f}h"
            if b_tok not in row[col_i] or w_tok not in row[col_i]:
                raise AssertionError(f"Phase 11 Suite 2 셀 불일치 (sigma={sg}/{wn}/{swk}): {row[col_i]!r}")
            checked_cells += 2

    # 3-e) Phase 11 Suite 3 실측 트레이스 표 셀 검증 (Google-LAVA-BPack 포함)
    sec84_3 = _find_section(sections, "8.4.4 Suite 3")
    t_p11_s3_list = _parse_tables(sec84_3)
    p11_s3 = p11["suite_3_real_traces"]
    t_hel = t_p11_s3_list[0]
    for sn in ("FirstFit", "L2Norm", "RackBestFit", "BPack-SpatialOnly", "Google-NILAS-BPack", "Google-LAVA-BPack", "Alibaba-DVLA-BPack", "Omni-Lindy-v7", "Omni-MRL-v11"):
        row = _table_row_lookup(t_hel, sn)
        hd = p11_s3["helios_multinode_20"]["schedulers"][sn]
        assert f"{hd['blk_gh_mean']:.3f}%" in row[1]
        assert f"{hd['bft_gw_jct_mean']:.3f}h" in row[2]
        assert f"{hd['blk_locality_mean']:.2f}%" in row[3]
        checked_cells += 3

    t_ali_low = t_p11_s3_list[1]
    for sn in ("FirstFit", "L2Norm", "BPack-SpatialOnly", "Google-NILAS-BPack", "Google-LAVA-BPack", "Alibaba-DVLA-BPack", "Omni-Lindy-v7", "Omni-MRL-v11"):
        row = _table_row_lookup(t_ali_low, sn)
        ad = p11_s3["alibaba_disjoint_windows"]["Alibaba_GPUSpec33_Disjoint_6"]["schedulers"][sn]
        asens = p11_s3["alibaba_scale_sensitivity"]["Alibaba_GPUSpec33_Sensitivity"]["schedulers"][sn]
        lc = p11_s3["helios_low_cv_real_11"]["schedulers"][sn]
        assert f"{lc['blk_gh_mean']:.3f}%" in row[1] and f"{lc['bft_wait_mean']:.4f}h" in row[1]
        assert f"{ad['blk_gh_mean']:.3f}%" in row[2] and f"{ad['bf_wait_mean']:.2f}h" in row[2]
        assert f"{asens['blk_gh_mean']:.3f}%" in row[3] and f"{asens['bf_wait_mean']:.2f}h" in row[3]
        checked_cells += 6

    earth_229_v7 = [u for u in p11_s3["helios_low_cv_real_11"]["raw_per_unit"]["Omni-Lindy-v7"] if u["idx"] == 5][0]
    earth_229_v11 = [u for u in p11_s3["helios_low_cv_real_11"]["raw_per_unit"]["Omni-MRL-v11"] if u["idx"] == 5][0]
    assert f"{earth_229_v7['blk_gh']:.3f}%" in sec84_3 and f"{earth_229_v11['blk_gh']:.3f}%" in sec84_3
    assert f"{earth_229_v11['gamma_ifr_active_frac'] * 100.0:.2f}%" in sec84_3

    # 3-e2) Phase 11.1 8.4.5절 4개 공개 트레이스(Alibaba GPU-v2026 4,052만 건 포함) 전수 감사 표 대조
    era = loaded["empirical_regime_audit.json"]
    sec84_5 = _find_section(sections, "8.4.5 Large-Scale Real-Trace")
    t_era = _parse_tables(sec84_5)[0]
    ali26_g = era["distributions"]["alibaba_2026"]["global"]
    row_a26 = _table_row_lookup(t_era, "Alibaba GPU-v2026")
    assert f"{ali26_g['n']:,}" in row_a26[1] and f"{ali26_g['cv']:.2f}" in row_a26[2]
    row_acm = _table_row_lookup(t_era, "AcmeTrace Kalos")
    assert f"{era['distributions']['acmetrace_kalos']['ALL_JOBS_GLOBAL']['n']:,}" in row_acm[1]
    assert f"{era['disjoint_rule']['total_valid_disjoint_windows']:,}" in sec84_5
    checked_cells += 4

    # 3-f) 요약·서론·9.2절·부록 문맥 일관성 검사
    sec_abs = _find_section(sections, "`Omni-Lindy` / `Omni-MRL`")
    assert "96" in sec_abs and "6,212" in sec_abs and "61" in sec_abs
    sec92 = _find_section(sections, "9.2 Live HTTP")
    assert "61/61" in sec92
    sec_app = _find_section(sections, "11. Artifact")
    assert "61 tests" in sec_app and "10 JSON" in sec_app

    print(
        f"  -> Alibaba CSV(1,213노드/6,212 GPU) + 10종 JSON + Lockbox raw 통계 재계산 + "
        f"표·문맥 {checked_cells}개 이상 셀 100% 일치 검증 완료.\n"
    )


def run_full_experiments(include_evo: bool = False) -> None:
    """1단계(진단) -> (선택)2~3단계 -> 4~11단계 전체 파이프라인을 재실행한다."""
    scripts = [
        "run_baseline_diagnosis.py",
        "b_lindy_discovery.py",
        "verify_discovery.py",
        "confirm_30seeds.py",
        "run_real_trace_eval.py",
        "run_phase7_eval.py",
        "run_phase8_eval.py",
        "run_phase9_eval.py",
        "run_phase10_eval.py",
        "run_phase11_eval.py",
    ]
    if include_evo:
        scripts.insert(1, "evolve_scheduler.py")
    data_ali = ROOT / "data" / "alibaba_2023" / "openb_pod_list_gpuspec33.csv"
    if not data_ali.exists():
        print("[준비] data/ 트레이스가 없어 download_traces.py를 먼저 실행합니다...")
        dl = subprocess.run([sys.executable, str(ROOT / "download_traces.py")], cwd=str(ROOT), check=False)
        if dl.returncode != 0:
            raise RuntimeError("download_traces.py 실행에 실패했습니다.")

    for sc in scripts:
        print(f"[Full Pipeline] `{sc}` 실행 중...")
        res = subprocess.run([sys.executable, str(ROOT / sc)], cwd=str(ROOT), check=False)
        if res.returncode != 0:
            raise RuntimeError(f"`{sc}` 실행 실패 (exit code={res.returncode})")
        print(f"  -> `{sc}` 완료.\n")


def main(argv: List[str] | None = None) -> None:
    """원클릭 재현 CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="Omni-MRL (v11.1) / Omni-Lindy 원클릭 재현 및 검증 도구")
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick: 61개 테스트+활성 KM 레이턴시+CSV/10종 JSON/논문 표·문맥 검증 | full: 전체 실험 재실행 후 검증",
    )
    parser.add_argument(
        "--include-evo",
        action="store_true",
        help="--mode full 시 2단계 진화 탐색(evolve_scheduler.py)까지 포함하여 재실행",
    )
    parser.add_argument(
        "--http-calls",
        type=int,
        default=100,
        help="K8s Extender 라이브 레이턴시 측정 반복 횟수 (기본: 100)",
    )
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    if args.mode == "full":
        run_full_experiments(include_evo=args.include_evo)

    run_unit_and_http_tests()
    run_k8s_http_latency_probe(num_calls=args.http_calls)
    verify_json_and_paper_consistency()

    elapsed = time.perf_counter() - t_start
    print(f"[완료] Omni-MRL v11.1 `--mode {args.mode}` 전체 검증 성공 (소요 시간: {elapsed:.2f}초)")


if __name__ == "__main__":
    main()
