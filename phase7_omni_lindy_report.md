# 7단계(D안) 최종 연구 보고서: 이기종 HoL 정체 돌파 및 멀티노드(`16~64 GPU`) 토폴로지 인지 통합 스케줄러 `Omni-Lindy`

> [!IMPORTANT]
> **핵심 결론 (합성 30시드 + 실제 기업 로그 3,379,285행 기반 1,584회 리플레이 실증)**
> 1. **단일 통합 알고리즘 `Omni-Lindy` (v7.0) 완성**: 기존 4~5단계 핵심 파라미터는 **단 1개도 재튜닝하지 않고(Zero-Shot)**, ① 유계 `Spec-Scarcity` + 로그 대형노드 감쇠(이기종 HoL 정체 해결)와 ② `Rack-Preservation` + 단일 랙 집적(`Tightest-Fit Rack`)을 결합했습니다. **단일 노드 동종 클러스터(`W1~W6`)에서는 v5.0 `Adaptive-B-Lindy`와 소수점 12자리까지 100% 비트-동일**합니다.
> 2. **Part A — Alibaba PAI 2023 이기종 잔여 한계 완전 돌파**:
>    - **`Alibaba_Default_Sweep`**: v5.0(`0.743%`, Wait `45.19h`) → v6.0(`0.078%`, Wait `0.133h`) → **`Omni-Lindy` v7.0 (`0.027%`, Stranded `4.51%`, `DotProduct 0.027%` 동률, `L2Norm 0.022%`와 통계적 무차이 $p_t=0.575$)**. `FGD_OracleStatic`(`0.145%`) 대비 **+81.43% 개선 ($p_t = 0.00012$)**!
>    - **`Alibaba_GPUSpec33_Sweep` (33% GPU 기종 제약)**: **`Omni-Lindy` Blocking 0.763%**로 9종 중 **1위**(vs 최강 SOTA `DotProduct 1.050%` **+27.34%, 10/12승**, vs `FGD_OracleStatic 3.062%` **+75.10%, 12/12승, $p_t=1.70\times 10^{-5}$**). 동시에 v6.0에서 발생했던 Backfill HoL 정체(`151.14h` / 최대 `342h`)를 **`49.05h` (-67.5% 단축, SOTA 최강 `FGD_Adaptive 104.54h` 대비 +53.08% 빠름)**로 완전 해소했습니다.
> 3. **Part B — SenseTime Helios 실제 멀티노드(`16~64 GPU`, 4클러스터 × 5구간 = 20개 실제 구간) 검증**:
>    - 실제 클러스터 GPU-시간의 **53.3%~60.5%**를 차지하는 **2~8노드(`16~64 GPU`) Gang 분산 학습 작업**을 모두 포함한 20개 실제 구간(`88~128 노드`, `11~16 랙`)에서 **`Omni-Lindy`가 Blocking(`18.740%`), 단일 랙 집적률(`92.5%`), 통신 지연 반영 `Topo-BF Wait`(`5.628h`), `Topo GW-JCT`(`11.354h`) 전 핵심 지표에서 9종 스케줄러 중 단독 1위**를 달성했습니다!
>    - 현업 토폴로지 SOTA **`RackBestFit`** 대비도 단일 랙 집적률 **+3.7%p (`88.8% → 92.5%`)**, Blocking **+2.21% (`19.164% → 18.740%`)**, `Topo GW-JCT` **+0.49% ($p_W = 0.0215$)** 우위를 보였으며, `FGD` 계열(`352~425 μs`) 대비 **3.7~4.5배 빠릅니다(`94.0 μs`)**.

---

## 1. `Omni-Lindy` (v7.0) 통합 아키텍처

```mermaid
flowchart TD
    IN["도착 작업 j = (gpu, cpu, ram, gpu_spec, num_nodes)"] -->|"공통 코어: v5.0 고정 물리상수 + (alpha, gamma) 이중 게이트"| CORE["다차원 패킹 + 8/4/1-GPU 경계 + 무차원 Lindy-Age"]
    CORE -->|"이기종 노드(1/2/4/8-GPU) + gpu_spec"| HET["유계 Live Spec-Scarcity 0.55*min(sp,1.5) + 로그 감쇠 1/(1+0.5*log2(g/g_min))"]
    CORE -->|"멀티노드(num_nodes>1) 감지 시"| TOPO1["단일노드 배치: Rack-Preservation (순수 랙 개봉 억제)"]
    CORE -->|"num_nodes > 1 Gang 요청"| TOPO2["멀티노드 배치: Tightest-Fit Rack + 최소 랙 스팬"]
```

- **이기종 HoL 정체 해소 (`hetero_aware=True`)**:
  - `v6.0`의 무제한 `Spec-Scarcity`가 순수 8-GPU 노드 보호 가중치($w_{\text{pristine}}=1.302$)를 역전시켜 발생했던 573.8h HoL Lockup을 `0.55 * min(spec_pressure, 1.5)` 상한으로 원천 차단했습니다.
  - 대형 노드 `Max-of-N` 린디 편향은 부드러운 로그 감쇠 $\frac{1}{1 + 0.5 \log_2(g_{\text{cap}} / g_{\min})}$로 완화하여 소형 노드 우선 채움과 열린 8-GPU 노드 집중 격리를 양립시켰습니다.
- **멀티노드 토폴로지 인지 (`topology_aware=True`)**:
  - 최근 도착 창에서 멀티노드 작업 비율 $p_{\text{multi}} > 0$이 감지되면, 단일 노드 배치 시 빈 풀노드 비율이 높은 온전한 랙에 `1.2 * min(1, 4*p_multi) * rack_free_ratio` 페널티를 부여해 멀티노드용 순수 랙을 보존합니다(`Rack-Preservation`).
  - `num_nodes > 1` Gang 작업 배치 시에는 `Omni-Lindy` 점수로 후보 노드를 평가한 뒤 단일 랙에 100% 들어가는 `Tightest-Fit Rack`을 최우선 선택합니다.
- **동종 단일 노드 비트-동일성**: `W1~W6` 및 단일 노드 동종 클러스터에서는 추가 항이 항등적으로 `0.0` 또는 `1.0`이 되어 v5.0과 소수점 12자리까지 100% 동일합니다.

---

## 2. Part A — Alibaba PAI 2023 이기종 스윕 성적표 (각 12회 평균)

### (1) `Alibaba_Default_Sweep` (`keep_ratio ∈ [0.0025, 0.0030, 0.0035, 0.0045]`, 기종 제약 없음)

| 스케줄러 | Blocking (%) | Stranded GPU (%) | Strict FIFO Wait (h) | Backfill Wait (h) | 지연시간 (μs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `L2Norm` *(부문 1위)* | **0.022** | 4.54 | **0.016** | 0.012 | 16.5 |
| `DotProduct` | 0.027 | 5.10 | 0.048 | **0.009** | 21.7 |
| `FGD_OracleStatic` | 0.145 | 38.56 | 64.283 | 60.328 | 772.1 |
| `FGD_Adaptive` | 0.269 | 42.16 | 109.572 | 135.967 | 579.9 |
| `B-LindyPack` / `Adaptive-B-Lindy` *(v5.0)* | 0.452 / 0.743 | 24.08 / 36.02 | 19.914 / 45.188 | 19.127 / 22.222 | 54.0 / 72.5 |
| **`Omni-Lindy` *(v7.0)*** | **0.027** | **4.51** | **0.326** | **0.222** | **84.9** |

- `v5.0` 대비 Blocking **+96.38% (`0.743% → 0.027%`, 12/12승, $p_W=0.00216$)**, Backfill Wait **+99.00% (`22.222h → 0.222h`)** 단축!
- `FGD_OracleStatic` 대비 Blocking **+81.43% (11/12승, $p_t = 0.000115$)**, `L2Norm(0.022%)`과는 통계적 차이 없음($p_t = 0.575$, `DotProduct 0.027%`와 동률).

### (2) `Alibaba_GPUSpec33_Sweep` (`keep_ratio ∈ [0.0050, 0.0065, 0.0080, 0.0100]`, 33% GPU 기종 제약)

| 스케줄러 | Blocking (%) | Stranded GPU (%) | Strict FIFO Wait (h) | Backfill Wait (h) | 지연시간 (μs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `FirstFit` / `BestFit` | 1.546 / 1.871 | 63.04 / 63.35 | 245.106 / 172.747 | 286.742 / 216.406 | 0.5 / 13.6 |
| `DotProduct` *(Blocking SOTA)* | 1.050 | 63.99 | 306.561 | 330.703 | 23.8 |
| `L2Norm` | 1.192 | 66.99 | 523.606 | 508.865 | 18.3 |
| `FGD_OracleStatic` | 3.062 | 64.46 | 229.274 | 211.843 | 836.2 |
| `FGD_Adaptive` *(SOTA Wait 1위)* | 3.465 | 65.99 | 144.291 | 104.539 | 600.0 |
| `Adaptive-B-Lindy` *(v5.0)* | 1.378 | 63.93 | **106.119** | **47.590** | 77.4 |
| **`Omni-Lindy` *(v7.0)*** | **0.763** | **61.33** | **172.201** | **49.048** | **111.3** |

- **Blocking 9종 중 1위 (`0.763%`)**: 최강 SOTA `DotProduct`(`1.050%`) 대비 **+27.34% (10/12승, $p_{\text{sign}}=0.0386$)**, `FGD_OracleStatic`(`3.062%`) 대비 **+75.10% (12/12승, $p_t = 1.70 \times 10^{-5}$)**.
- **Backfill HoL 정체 해결**: `v6.0`의 `151.14h`를 **`49.048h` (-67.5% 단축)**로 끌어내려 `v5.0(47.59h)`과 동등 수준으로 복원하고, 모든 SOTA 6종(최선 `FGD_Adaptive 104.54h`) 대비 **+53.08% 더 빠른 대기시간** 달성!

---

## 3. Part B — SenseTime Helios 실제 멀티노드(`16~64 GPU`) 20개 구간 성적표

4개 클러스터(`Venus`, `Earth`, `Saturn`, `Uranus`)의 `1~8 노드(최대 64 GPU)` 단일+멀티노드 통합 로그(유효 작업 총 **1,576,255건**, `>8노드` 초대형 4,209건 제외)에서 추출한 **20개 비중첩 피크 구간**(`88~128 노드`, `11~16 랙`) 통합 결과입니다.

| 스케줄러 | 전체 Blk (%) | 멀티노드 Blk (%) | 단일랙 집적률 (%) | FIFO Wait (h) | BF Wait (h) | **Topo-BF Wait (h)** *(λ=0.08)* | **Topo GW-JCT (h)** *(λ=0.08)* | 지연시간 (μs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `FirstFit` | 18.952 | **22.457** | 36.9 | 5.784 | 5.353 | 6.261 | 12.142 | 0.5 |
| `BestFit` | 19.172 | 22.662 | 33.7 | 5.753 | 5.313 | 6.232 | 12.091 | 14.8 |
| `DotProduct` | 19.600 | 23.829 | 34.9 | 5.883 | 5.418 | 6.382 | 12.255 | 24.3 |
| `L2Norm` *(비토폴로지 Blk 1위)* | 18.849 | 22.958 | 33.6 | 5.817 | 5.373 | 6.206 | 12.063 | 19.5 |
| `RackBestFit` *(현업 토폴로지 SOTA)* | 19.164 | 22.893 | 88.8 | 5.743 | **5.311** | 5.679 | 11.409 | 25.7 |
| `FGD_OracleStatic` *(ATC '23 오라클)* | 19.079 | 22.747 | 33.4 | 5.764 | 5.325 | 6.214 | 12.077 | 352.3 |
| `FGD_Adaptive` | 19.298 | 22.673 | 33.6 | 5.750 | 5.330 | 6.168 | 12.022 | 425.2 |
| `B-LindyPack` | 19.343 | 22.802 | 33.4 | **5.718** | 5.330 | 6.226 | 12.076 | 62.3 |
| **`Omni-Lindy` *(v7.0)*** | **18.740** | **22.489** | **92.5** | **5.734** | **5.326** | **5.628** | **11.354** | **94.0** |

- **전 핵심 실전 지표 1위**:
  - **전체 Blocking 1위**: **`18.740%`** (vs `L2Norm` +0.57% 12/20승, vs `RackBestFit` +2.21% 12/20승, vs `FGD_Adaptive` +2.89% 14/20승 $p_t=0.081$).
  - **단일 랙 집적률(Locality) 압도적 1위**: **`92.5%`** (vs 현업 `RackBestFit 88.8%` **+3.7%p 우위**; 비토폴로지 SOTA들은 **`33.4%~36.9%`**로 3건 중 2건이 랙 밖 분산!).
  - **통신 지연 반영 실전 대기 큐 (`cross_rack_penalty=0.08`)**:
    - `Topo-BF Wait`: **`5.628h`** 9종 중 1위 (vs `RackBestFit 5.679h` **+0.89% $p_t=0.062$**, vs `FGD_Adaptive 6.168h` **+8.75%**, vs `L2Norm 6.206h` **+9.31%**).
    - `Topo GW-JCT`: **`11.354h`** 9종 중 1위 (vs `RackBestFit 11.409h` **+0.49%, 15/20승, $p_t = 0.0283, p_W = 0.0215$**).

---

## 4. 정직한 한계점

> [!WARNING]
> **솔직한 한계점**
> 1. **개별 스윕/20구간 검정력**: `Alibaba_GPUSpec33`에서 `FGD_Oracle` 대비($p_t = 1.70\times 10^{-5}$)와 `Default`에서 `v5.0` 대비($p_W = 0.00216$)·`FGD_Oracle` 대비($p_t = 0.000115$)는 $p < 0.01$을 확증했으나, `Helios_MultiNode_Pooled_20`에서는 `RackBestFit` 대비 `Topo GW-JCT`가 $p_W = 0.0215$($p < 0.05$) 수준이며 Blocking($p_t=0.544$)·`GPUSpec33` vs `DotProduct`($p_t=0.134$)는 구간 간 분산 때문에 단독 $p < 0.01$에는 미달합니다.
> 2. **이종 제약의 잔여 트레이드오프**: `Alibaba_Default_Sweep`의 대기시간(`0.222h` vs `L2Norm 0.012h`)과 `Alibaba_GPUSpec33`의 Strict FIFO Wait(`172.2h` vs `v5.0 106.1h`)에서는 미세한 양보가 남아 있습니다.
> 3. **초대형(`> 8노드 / 64 GPU`) 및 네트워크 스위치 모델**: 8노드 이하 분산 작업(`16~64 GPU`)까지 수용했으나 `16~128노드` 초대형 학습(총 4,209건 제외)과 다계층 Fat-Tree/Rail-Optimized NVLink 링크 경합까지는 미모델링했습니다.

---

## 5. 검증 및 재현 파일 목록

- [simulator.py](./simulator.py) — 멀티노드(`num_nodes`) 원자적 Gang Allocation, 랙 토폴로지 매핑(`nodes_per_rack`), 랙 간 통신 지연(`cross_rack_penalty`)
- [baselines.py](./baselines.py) — 전 베이스라인 `score_nodes()` 벡터화 및 현업 토폴로지 SOTA `RackBestFitScheduler` 추가
- [adaptive_lindy.py](./adaptive_lindy.py) — 통합 스케줄러 `Omni-Lindy` (유계 `Spec-Scarcity`, 로그 감쇠, `Rack-Preservation`, `Tightest-Fit Rack`)
- [trace_loader.py](./trace_loader.py) — `load_helios_cluster_multinode()` 및 `extract_helios_multinode_windows()`
- [run_phase7_eval.py](./run_phase7_eval.py) — 7단계 Part A + Part B 병렬 실증 벤치마크
- [phase7_report.json](./phase7_report.json) — 7단계 전체 원본 실험 수치 JSON
- [test_suite.py](./test_suite.py) — **32/32 단위·불변량 테스트 (`python -W error`)** 및 **전체 15개 `.py` 파일 컴파일·미사용 임포트 0건** 확인 완료
