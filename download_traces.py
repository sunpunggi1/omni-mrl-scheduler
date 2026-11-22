"""
6단계(B안): 실제 공개 기업 GPU 클러스터 트레이스 다운로더 및 스키마 검사기

대상 출처:
1. Alibaba cluster-trace-gpu-v2023 (USENIX ATC '23 FGD 공식 트레이스)
   - 필수: openb_node_list_gpu_node.csv, openb_pod_list_default.csv
   - 추가: 20MB 이하 변형 파드 CSV 최대 3개
2. SenseTime HeliosData (SC '21 실제 AI 데이터센터 4개 클러스터 로그)
   - 대상: Venus, Earth, Saturn, Uranus cluster_log.csv (또는 .csv.gz)
"""

from __future__ import annotations

import gzip
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ALIBABA_DIR = DATA_DIR / "alibaba_2023"
HELIOS_DIR = DATA_DIR / "helios"

USER_AGENT = "evo-gpu-scheduler-research/1.0"
MAX_EXTRA_POD_BYTES = 20 * 1024 * 1024
MAX_HELIOS_BYTES = 85 * 1024 * 1024


def _fetch_json(url: str) -> object:
    """GitHub API JSON 응답을 조회한다."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github.v3+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, dest: Path) -> int:
    """URL에서 파일을 스트리밍 다운로드하여 저장하고 바이트 크기를 반환한다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fout:
        shutil.copyfileobj(resp, fout)
    return dest.stat().st_size


def _inspect_csv(path: Path, max_sample_rows: int = 5) -> Dict[str, object]:
    """CSV(또는 .csv.gz) 파일의 행 수, 헤더, 샘플 5행을 읽어 요약한다."""
    opener = gzip.open if path.suffix == ".gz" else open
    rows: List[str] = []
    line_count = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if len(rows) <= max_sample_rows:
                rows.append(line.rstrip("\r\n"))
    header = rows[0] if rows else ""
    samples = rows[1 : 1 + max_sample_rows]
    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "total_lines": line_count,
        "data_rows": max(0, line_count - 1),
        "header": header,
        "samples": samples,
    }


def download_alibaba_2023() -> List[Dict[str, object]]:
    """Alibaba cluster-trace-gpu-v2023 CSV 파일들을 조회·다운로드한다."""
    api_url = "https://api.github.com/repos/alibaba/clusterdata/contents/cluster-trace-gpu-v2023/csv"
    entries = _fetch_json(api_url)
    if not isinstance(entries, list):
        raise RuntimeError(f"Alibaba GitHub API 응답 형식 오류: {type(entries)}")

    by_name: Dict[str, dict] = {
        e["name"]: e for e in entries if isinstance(e, dict) and e.get("type") == "file"
    }

    required = ["openb_node_list_gpu_node.csv", "openb_pod_list_default.csv"]
    selected: List[str] = []
    for name in required:
        if name not in by_name:
            raise RuntimeError(f"필수 Alibaba 파일 누락: {name}")
        selected.append(name)

    # 추가 변형 파드 CSV 후보 우선 탐색 (20MB 이하 최대 3개)
    preferred_extras = [
        "openb_pod_list_multigpu50.csv",
        "openb_pod_list_gpuspec33.csv",
        "openb_pod_list_cpu050.csv",
        "openb_pod_list_multigpu20.csv",
        "openb_pod_list_gpuspec20.csv",
    ]
    extra_added = 0
    for name in preferred_extras:
        if name in by_name and by_name[name].get("size", 0) <= MAX_EXTRA_POD_BYTES:
            selected.append(name)
            extra_added += 1
            if extra_added >= 3:
                break

    if extra_added < 3:
        for name, meta in sorted(by_name.items()):
            if (
                name.startswith("openb_pod_list_")
                and name.endswith(".csv")
                and name not in selected
                and 0 < meta.get("size", 0) <= MAX_EXTRA_POD_BYTES
            ):
                selected.append(name)
                extra_added += 1
                if extra_added >= 3:
                    break

    summaries: List[Dict[str, object]] = []
    for name in selected:
        meta = by_name[name]
        dl_url = meta.get("download_url")
        if not dl_url:
            continue
        dest = ALIBABA_DIR / name
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"[Alibaba 다운로드 중] {name} ({meta.get('size', 0) / 1024:.1f} KB)...")
            _download_file(dl_url, dest)
        else:
            print(f"[Alibaba 캐시 사용] {name}")
        summaries.append(_inspect_csv(dest))
    return summaries


def _resolve_helios_contents(subpath: str) -> Tuple[str, List[dict]]:
    """HeliosData 저장소에서 main 또는 master 브랜치 경로를 자동 판별한다."""
    path_part = f"/{subpath}" if subpath else ""
    for branch in ("main", "master"):
        url = f"https://api.github.com/repos/S-Lab-System-Group/HeliosData/contents{path_part}?ref={branch}"
        try:
            res = _fetch_json(url)
            if isinstance(res, list):
                return branch, res
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
    raise RuntimeError(f"HeliosData 경로를 찾을 수 없습니다: {subpath or '/'}")


def download_helios() -> List[Dict[str, object]]:
    """SenseTime HeliosData의 data.zip(또는 디렉토리)에서 Venus/Earth/Saturn/Uranus 로그를 확보한다."""
    branch, root_entries = _resolve_helios_contents("")
    root_files = {e["name"].lower(): e for e in root_entries if e.get("type") == "file"}

    if "data.zip" in root_files and root_files["data.zip"].get("download_url"):
        zip_meta = root_files["data.zip"]
        zip_path = HELIOS_DIR / "helios_data.zip"
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            print(
                f"[Helios 다운로드 중] data.zip "
                f"({int(zip_meta.get('size', 0)) / (1024 * 1024):.2f} MB, branch={branch})..."
            )
            _download_file(zip_meta["download_url"], zip_path)
        else:
            print("[Helios 캐시 사용] helios_data.zip")

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            for m in members:
                target_name = m.replace("\\", "/").lstrip("/")
                # data/Venus/cluster_log.csv -> data/helios/Venus/cluster_log.csv
                rel_parts = [p for p in target_name.split("/") if p]
                if rel_parts and rel_parts[0].lower() == "data":
                    rel_parts = rel_parts[1:]
                if not rel_parts:
                    continue
                dest = HELIOS_DIR.joinpath(*rel_parts)
                if not dest.exists() or dest.stat().st_size == 0:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

        summaries: List[Dict[str, object]] = []
        for cl in ("Venus", "Earth", "Saturn", "Uranus"):
            cand = HELIOS_DIR / cl / "cluster_log.csv"
            if not cand.exists():
                matches = list(HELIOS_DIR.glob(f"**/{cl}*/cluster_log.csv"))
                if matches:
                    cand = matches[0]
            if cand.exists():
                summaries.append(_inspect_csv(cand))
            else:
                print(f"[Helios 경고] 압축 해제 후 {cl}/cluster_log.csv 미발견")
        return summaries

    # 폴백: 디렉토리 구조인 경우
    dir_map = {e["name"].lower(): e["name"] for e in root_entries if e.get("type") == "dir"}
    targets = ["Venus", "Earth", "Saturn", "Uranus"]
    file_items: List[Tuple[str, str, int, str]] = []
    for cl in targets:
        actual_dir = dir_map.get(cl.lower())
        if not actual_dir:
            continue
        _, cl_entries = _resolve_helios_contents(actual_dir)
        log_file: Optional[dict] = None
        for fname in ("cluster_log.csv", "cluster_log.csv.gz"):
            for e in cl_entries:
                if e.get("type") == "file" and e.get("name", "").lower() == fname:
                    log_file = e
                    break
            if log_file is not None:
                break
        if log_file and log_file.get("download_url"):
            file_items.append((cl, log_file["name"], int(log_file.get("size", 0)), log_file["download_url"]))

    eligible = [item for item in file_items if item[2] <= MAX_HELIOS_BYTES]
    summaries = []
    for cl, fname, _, dl_url in eligible:
        dest = HELIOS_DIR / f"{cl.lower()}_{fname}"
        if not dest.exists() or dest.stat().st_size == 0:
            _download_file(dl_url, dest)
        summaries.append(_inspect_csv(dest))
    return summaries


def main() -> None:
    print("=== [1/2] Alibaba cluster-trace-gpu-v2023 다운로드 및 스키마 확인 ===")
    ali_summaries = download_alibaba_2023()
    for s in ali_summaries:
        print(f"\n[Alibaba] {s['file']} | {s['size_mb']} MB | data_rows={s['data_rows']}")
        print(f"  HEADER : {s['header']}")
        for idx, row in enumerate(s["samples"], 1):
            print(f"  ROW {idx}  : {row}")

    print("\n=== [2/2] SenseTime HeliosData 다운로드 및 스키마 확인 ===")
    hel_summaries = download_helios()
    for s in hel_summaries:
        print(f"\n[Helios] {s['file']} | {s['size_mb']} MB | data_rows={s['data_rows']}")
        print(f"  HEADER : {s['header']}")
        for idx, row in enumerate(s["samples"], 1):
            print(f"  ROW {idx}  : {row}")


if __name__ == "__main__":
    main()
