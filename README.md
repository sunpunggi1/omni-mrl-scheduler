# `Omni-MRL` (v11.1) / `Omni-Lindy`: 시공간·해저드·이기종·토폴로지 통합 무예측 AI GPU 스케줄러

`Omni-MRL` (v11.1)은 **완전 무예측(Strict Non-Clairvoyant)** 조건 하에 Google `LAVA`/`NILAS`, Alibaba `DVLA`/`FGD` 원리를 **동일 진입 코호트 Kaplan-Meier**(정상상태 길이편향 완화) 및 $C^0$/비음수 위험률($\hat{R}'\ge -1$) 보장 매끄러운 `tanh` 꼬리 MRL 모델과 단일 `BPack` 공간 기반 위에 통합·검증한 차세대 AI GPU 클러스터 스케줄러이자 **Kubernetes v1.34 규약 정합 HTTP Scheduler Extender** 프로토타입입니다.

> * **기술 논문 전문**: [paper/omni_lindy_paper.md](./paper/omni_lindy_paper.md)
> * **라이선스**: [MIT License](./LICENSE)
> * **공개 트레이스 다운로드 (신규 클론 시)**: `python download_traces.py` (Alibaba PAI 2023 / SenseTime Helios 원본 트레이스 자동 내려받기)

---

## 핵심 연구 및 공정 팩토리얼 실증 요약 (Phase 11.1 Lockbox 30-시드, Holm-96 · 실트레이스 감사)

| 평가 축 | 공정 비교 기준 (코호트 KM · `BPack` 기반 · Dev 10-시드 동결) | `Omni-MRL` (v11.1) 실측 결과 | 통계적 유의성 / 정직한 한계 |
| :--- | :--- | :--- | :--- |
| **합성 IFR (`IFR_Uniform`, Lockbox 30)** | `FirstFit` / `L2Norm` / `NILAS-L2` / v7 / `NILAS-BPack` | Blk **6.444%**, BF Wait **0.127h** | vs v7 **+12.46% Blk · +52.02% BF** (`p_holm96<=2.6e-4`); 단 Blk는 `NILAS-BPack(6.251%)` 우위 (`-3.09%`, `p_holm=0.011`) |
| **마모 수명 (`IFR_Weibull_k2`, Lockbox 30)** | `FirstFit` / `L2Norm` / `NILAS-L2` / v7 / `NILAS-BPack` | Blk **7.132%** (1위), BF Wait **0.513h** | vs v7 **+6.10% Blk · +11.71% BF** (`p_holm96<=0.015`); BF Wait은 `FirstFit(0.459h)`이 더 빠름 |
| **메모리리스·DFR (`W4`, `W1`, `W3`, `W2`)** | `FirstFit` / `L2Norm` / `NILAS-BPack` / `LAVA-BPack` | `W4` Blk **6.383%** (vs 공간·빅테크 `+7.0~+8.3%`), `W1` **3.699%/0.074h**, `W3` BF **0.021h** | `W1` vs `L2Norm` 비유의(`p_holm=1`), `W3` Blk는 `LAVA-BPack(0.894%)` 최소, `W4` BF는 `FirstFit(0.422h)` 최소 |
| **순수 MRL Ablation (`Suite 1b`, Lockbox 30)** | `Omni-MRL-NoPristineRelax` vs `Omni-Lindy-v7` | `NoRelax`: `IFR_Uniform` **6.336%/0.127h**, `Weibull` **7.039%/0.517h** | 순수 MRL 항 단독 기여 **+13.94%/+7.32% Blk** (`p_W<=7.3e-6`), `pristine_relax=0.35`는 소폭 Blk 손실(`-1.3~-1.7%`) |
| **편향 분리 Dual-Oracle (`Suite 2`, Lockbox 30)** | 연속 `NILAS-ML-BPack` ($w=2.0$, $\mathbb{E}[\epsilon]=1$) | `IFR` **6.444% / 0.127h**, `W3` **0.956% / 0.021h** | `IFR` $\sigma=0$ 오라클 `6.241%` 승(`p=3.5e-3`) → **$\sigma\ge 0.5$ BF(+24.3%)**, **$\sigma\ge 1.0$ Blk(+8.7%~+14.3%)·BF(+44.4%~+55.3%) Zero-ML 단조 역전** |
| **실측 저-CV & 독립 구간 (`Suite 3`)** | `Helios` 저-CV 11창 · `Helios_MN 20` · `Alibaba` 비중첩 6창 | `Helios Low-CV`: **12.641%** (IFR **61.98%**, vs v7 `+6.46%`) · `Helios_MN`: **11.354h** (`4/4` 클러스터 승) | `Helios_MN` 클러스터(`n=4`) Holm-16 보정 비유의(`p>=0.161`), 창(`n=20`) 보조 유의; `Alibaba GPUSpec33` 6창은 `LAVA/DVLA` BF(`3.20~3.40h`) 우위 |
| **대규모 실트레이스 전수 감사 (4종 982창)** | `Alibaba GPU-v2026` (40,522,321건) · `AcmeTrace` · `Helios` · `Alibaba 2023` | 전 계층 DFR (`Alibaba v2026` CV **9.29**; `AcmeTrace` CV **10.16**) | 982개 비중복 창 중 `CV<0.9` **12개(1.22%)**, 다중모드 분화 **9개(0.92%)** — 현행 전역 DFR/IFR 게이트 정합성 확인 |
| **K8s HTTP Extender (v11.1 프로토타입)** | `[0, 10]` · `spec.resources` · `400m` 대소문자 · 멱등 완료 · 동명/상이UID 분리 | 128노드 활성 KM 포함 어댑터 **1.621 ms** / HTTP **6.238 ms** | **61개** 테스트 `-W error` 경고 0건 통과 (실물 `kube-scheduler`/Informer E2E·원자적 Gang·MIG/MPS는 향후 과제) |

---

## 빠른 시작 (Quick Start)

### 1. 의존성 설치 및 단위·HTTP 통합 테스트 실행
외부 웹 프레임워크나 Go 툴체인 없이 **Python 3.10+ 및 NumPy**만으로 동작합니다.

```bash
pip install -e .
python -W error test_suite.py
```

### 2. 원클릭 재현 및 JSON↔논문 교차 검증
```bash
# 빠른 검증 (< 25초): 61개 테스트 + 활성 KM K8s HTTP 레이턴시 + Alibaba CSV + 10종 JSON/Raw 통계·표·문맥 교차 검증
python reproduce_all.py --mode quick

# 전체 파이프라인 재실행 (트레이스 다운로드 + Phase 1~11 전체 재계산)
python reproduce_all.py --mode full
```

### 3. Kubernetes Scheduler HTTP Extender 실행 및 테스트
```bash
# 로컬 Extender 서버 기동 (기본 엔진 omni_mrl v11.1, 포트 8888)
python -m k8s_extender.server --host 127.0.0.1 --port 8888 --nodes-per-rack 8 --engine omni_mrl
```

```bash
# 헬스체크 및 온라인 메트릭 조회 (Linux/macOS curl 예시)
curl http://127.0.0.1:8888/healthz
curl http://127.0.0.1:8888/metrics?format=prometheus
```

```powershell
# PowerShell 헬스체크 예시 (Windows)
Invoke-RestMethod http://127.0.0.1:8888/healthz
```

### 4. Kubernetes 클러스터 배포 (`deploy/k8s/`)
* Extender Deployment/Service 배포: [extender-deployment.yaml](./deploy/k8s/extender-deployment.yaml)
* `kube-scheduler` Extender 연동 설정: [scheduler-config.yaml](./deploy/k8s/scheduler-config.yaml)
* 컨테이너 이미지 빌드: [Dockerfile](./Dockerfile)

---

## 저장소 구조

| 파일 / 디렉토리 | 역할 |
| :--- | :--- |
| [simulator.py](./simulator.py) | Non-Clairvoyant 동적 GPU 클러스터 이벤트 시뮬레이터 (분수·이기종·Gang·코호트 추적) |
| [omni_mrl.py](./omni_mrl.py) | 차세대 통합 스케줄러 `Omni-MRL` (`v8.0~v11.1`: 양방향 시간불변 해저드 + 코호트 KM MRL + 토폴로지·희소기종 방어) |
| [adaptive_lindy.py](./adaptive_lindy.py) | `Adaptive B-LindyPack` (v5.0) 및 통합 실클러스터 엔진 `Omni-Lindy` (v7.0) |
| [bigtech_baselines.py](./bigtech_baselines.py) | 동일 진입 코호트 KM, $C^0$/비음수 위험률 `tanh` 꼬리 MRL, 빅테크 비교 변형(`Google-NILAS`, `Google-LAVA`, `Alibaba-DVLA`) 및 연속 ML 오라클 |
| [baselines.py](./baselines.py) | 고전·단편화 SOTA 비교군 7종 (`FirstFit`, `BestFit`, `DotProduct`, `L2Norm`, `FGD`, `FGD-Adaptive`, `RackBestFit`) |
| [k8s_extender/](./k8s_extender/__init__.py) | `kube-scheduler` v1 HTTP Extender 프로토타입(`/filter`, `/prioritize`, `/gang_schedule`, `/events/pod_completed`, `/healthz`, `/metrics`) |
| [paper/omni_lindy_paper.md](./paper/omni_lindy_paper.md) | 1~11.1단계 전체 연구·정리·공정 팩토리얼·Ablation·Lockbox·실트레이스 전수 감사·시스템 구현 논문 |
| [run_phase11_eval.py](./run_phase11_eval.py) | Phase 11.1 Lockbox 30-시드, Ablation, 편향 분리 Dual-Oracle 및 실트레이스 독립 창 실행기 |
| [phase10_report.json](./phase10_report.json) | Phase 10 코호트 KM·Dual-Oracle·`101k↔Fresh` 재현성·전 비교군 Holm 보정 JSON |
| [phase11_report.json](./phase11_report.json) | Phase 11.1 Lockbox 30-시드 원자료·클러스터/창 Holm·Helios 저-CV 11창·아티팩트 16종 SHA-256 JSON |
| [empirical_regime_audit.json](./empirical_regime_audit.json) | `Alibaba GPU-v2026`(4,052만 건)·`AcmeTrace Kalos`·`Helios`·`Alibaba 2023` 전 계층 DFR 및 982창 감사 JSON |
| [reproduce_all.py](./reproduce_all.py) | 원클릭 빠른 검증(`--mode quick`: CSV·10종 JSON·Raw 통계·표·문맥 일치 검증) 및 전체 재실행(`--mode full`) CLI |
| [test_suite.py](./test_suite.py) | 경계·불변량·Non-Clairvoyant·시간불변·MRL 수학·코호트 KM·K8s v1.34 HTTP 통합 61개 테스트 |
| [trace_loader.py](./trace_loader.py) | Alibaba 2023(1,213 노드/6,212 GPU) & SenseTime Helios(단일/멀티노드/저-CV) 실측 로그 파서 |
| [stats_utils.py](./stats_utils.py) | SciPy 무의존 Student-t, Wilcoxon 부호순위, 이항 부호 검정, Holm-Bonferroni, 계층 클러스터 부트스트랩 라이브러리 |
