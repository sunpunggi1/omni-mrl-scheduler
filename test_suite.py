"""
동적 GPU 클러스터 스케줄러(simulator, stats_utils, adaptive_lindy) 종합 단위·경계·불변량 테스트 스위트

검증 항목:
1. 시뮬레이터 경계 조건: 빈 작업 목록, 단일 작업, 수명 0 작업, 미정렬 도착 시각, 동시 도착/종료
2. 데드락 및 예외 방어: 노드 용량 초과 작업, 음수 자원/수명, 잘못된 큐 정책 ValueError 검증
3. 자원 보존 및 대기 큐 정합성: 종료 후 자원 100% 반환, JCT == Wait + Duration, 실제 시작시각 기반 경과시간 검증
4. 기아 방지 백필(Starvation-Free Backfill) 검증: 소형 잡 폭주 시에도 8-GPU 헤드 작업이 기아 없이 실행됨을 보장
5. Non-Clairvoyance(미래 수명 비노출) 반증 테스트: t <= T 구간 동일·미래 수명만 다른 두 트레이스에서 t <= T 배치 100% 일치
6. 시간 단위 불변성(Clock-Unit Invariance): 초/분/시간 스케일 변환 시 Adaptive B-LindyPack의 결정 100% 동일
7. 온라인 Corr(gpu, age) 방향성·콜드스타트 안전성·reset() 멱등성 검증
8. stats_utils.py 정확 Student-t / Wilcoxon / 이항 부호 검정 수학적 정밀도 검증
"""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from adaptive_lindy import AdaptiveBLindyPackScheduler, AdaptiveLindyConfig, DEFAULT_BASE_THETA
from baselines import FirstFitScheduler, L2NormScheduler
from b_lindy_discovery import BoundedLindyPackScheduler
from simulator import (
    ClusterConfig,
    ClusterState,
    Job,
    run_queue_simulation,
    run_simulation,
)
from stats_utils import (
    exact_sign_test_pvalue,
    exact_student_t_pvalue,
    paired_summary_stats,
    paired_t_test_exact,
    student_t_critical_95,
    wilcoxon_signed_rank_test,
)
from workloads import generate_w1_pai_mixed, generate_w3_heavy_straggler


class TestSimulatorEdgeCasesAndQueue(unittest.TestCase):
    """시뮬레이터(Blocking 및 Pending Queue) 경계값·보존법칙·정합성 테스트."""

    def test_empty_jobs_blocking_and_queue(self) -> None:
        cfg = ClusterConfig(num_nodes=4)
        sch = AdaptiveBLindyPackScheduler()
        mb = run_simulation([], sch, config=cfg)
        self.assertEqual(mb.total_jobs, 0)
        self.assertEqual(mb.rejected_jobs, 0)
        self.assertEqual(mb.gpu_hour_blocking_rate, 0.0)

        for pol in ("fifo", "backfill"):
            mq = run_queue_simulation([], sch, config=cfg, queue_policy=pol)
            self.assertEqual(mq.total_jobs, 0)
            self.assertEqual(mq.completed_jobs, 0)
            self.assertEqual(mq.mean_jct, 0.0)
            self.assertEqual(mq.makespan, 0.0)

    def test_single_job_and_zero_duration(self) -> None:
        cfg = ClusterConfig(num_nodes=2)
        sch = AdaptiveBLindyPackScheduler()

        j1 = [Job(0, arrival_time=3.0, duration=4.5, gpu=4.0, cpu=16.0, ram=128.0)]
        mq = run_queue_simulation(j1, sch, config=cfg, queue_policy="fifo")
        self.assertEqual(mq.completed_jobs, 1)
        self.assertAlmostEqual(mq.mean_wait_time, 0.0, places=9)
        self.assertAlmostEqual(mq.mean_jct, 4.5, places=9)
        self.assertAlmostEqual(mq.makespan, 4.5, places=9)
        self.assertAlmostEqual(mq.mean_slowdown, 1.0, places=9)

        # 수명이 0인 극단 작업에서도 ZeroDivisionError 없이 동작하며 slowdown >= 1.0 보장
        j0 = [Job(0, arrival_time=1.0, duration=0.0, gpu=0.25, cpu=4.0, ram=16.0)]
        mq0 = run_queue_simulation(j0, sch, config=cfg, queue_policy="fifo")
        self.assertEqual(mq0.completed_jobs, 1)
        self.assertAlmostEqual(mq0.mean_wait_time, 0.0, places=9)
        self.assertAlmostEqual(mq0.makespan, 0.0, places=9)
        self.assertAlmostEqual(mq0.mean_slowdown, 1.0, places=9)

    def test_invalid_inputs_raise_value_error(self) -> None:
        cfg = ClusterConfig(num_nodes=2, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)
        sch = FirstFitScheduler()

        # 단일 노드 용량 초과 (데드락 방지)
        with self.assertRaises(ValueError):
            run_queue_simulation([Job(0, 0.0, 1.0, 9.0, 8.0, 64.0)], sch, config=cfg)
        with self.assertRaises(ValueError):
            run_queue_simulation([Job(0, 0.0, 1.0, 1.0, 65.0, 64.0)], sch, config=cfg)
        with self.assertRaises(ValueError):
            run_queue_simulation([Job(0, 0.0, 1.0, 1.0, 8.0, 600.0)], sch, config=cfg)
        # 음수 수명/자원
        with self.assertRaises(ValueError):
            run_queue_simulation([Job(0, 0.0, -1.0, 1.0, 8.0, 64.0)], sch, config=cfg)
        # 잘못된 큐 정책 및 음수 백필 파라미터
        with self.assertRaises(ValueError):
            run_queue_simulation([Job(0, 0.0, 1.0, 1.0, 8.0, 64.0)], sch, config=cfg, queue_policy="invalid")
        with self.assertRaises(ValueError):
            run_queue_simulation([Job(0, 0.0, 1.0, 1.0, 8.0, 64.0)], sch, config=cfg, queue_policy="backfill", backfill_window=-1)
        with self.assertRaises(ValueError):
            run_queue_simulation([Job(0, 0.0, 1.0, 1.0, 8.0, 64.0)], sch, config=cfg, queue_policy="backfill", max_hol_bypasses=-1)

    def test_concurrent_duplicate_job_instance_no_crash(self) -> None:
        """동일한 Job 객체 인스턴스가 서로 다른 노드·시작시각에 동시 실행되어도 충돌 없이 해제된다."""
        cfg = ClusterConfig(num_nodes=2)
        j_dup = Job(10, arrival_time=0.0, duration=5.0, gpu=4.0, cpu=16.0, ram=128.0)
        jobs = [
            Job(1, 0.0, 2.0, 4.0, 16.0, 128.0),
            Job(2, 0.0, 1.0, 8.0, 32.0, 256.0),
            j_dup,
            j_dup,
        ]
        mq = run_queue_simulation(jobs, FirstFitScheduler(), config=cfg, queue_policy="fifo")
        self.assertEqual(mq.completed_jobs, 4)
        self.assertAlmostEqual(mq.makespan, 6.0, places=9)
        self.assertAlmostEqual(mq.mean_wait_time, 0.25, places=9)

        # Blocking 모드에서 동시 종료되는 동일 Job 인스턴스도 heap 비교 TypeError 없이 완주
        mb = run_simulation([j_dup, j_dup], FirstFitScheduler(), config=cfg)
        self.assertEqual(mb.total_jobs, 2)
        self.assertEqual(mb.rejected_jobs, 0)

    def test_duplicate_job_id_backfill_hol_tracking(self) -> None:
        """서로 다른 작업이 동일 job_id를 공유해도 Backfill HoL 바이패스 카운터가 독립 추적된다."""
        cfg = ClusterConfig(num_nodes=1)
        jobs = [
            Job(0, 0.0, 2.0, 1.0, 8.0, 64.0),
            Job(99, 0.1, 1.0, 8.0, 32.0, 256.0),
            Job(99, 0.2, 1.0, 8.0, 32.0, 256.0),
        ] + [Job(100 + k, 0.3 + 0.1 * k, 1.0, 1.0, 8.0, 64.0) for k in range(10)]
        mq = run_queue_simulation(jobs, FirstFitScheduler(), config=cfg, queue_policy="backfill", max_hol_bypasses=2)
        self.assertEqual(mq.completed_jobs, len(jobs))

    def test_public_ops_version_property(self) -> None:
        """ClusterState.ops_version 공개 프로퍼티가 할당/해제마다 1씩 단조 증가한다."""
        cfg = ClusterConfig(num_nodes=1)
        state = ClusterState(cfg)
        self.assertEqual(state.ops_version, 0)
        j = Job(0, 0.0, 2.0, 1.0, 8.0, 64.0)
        alloc = state.allocate(0, j)
        self.assertEqual(state.ops_version, 1)
        state.now = 2.0
        state.release(0, j, alloc)
        self.assertEqual(state.ops_version, 2)

    def test_same_timestamp_departure_before_arrival(self) -> None:
        """t=5.0에 기존 8-GPU 작업이 종료되고 동시에 새 8-GPU 작업이 도착하면 대기 시간은 0이어야 한다."""
        cfg = ClusterConfig(num_nodes=1)
        jobs = [
            Job(0, arrival_time=0.0, duration=5.0, gpu=8.0, cpu=32.0, ram=256.0),
            Job(1, arrival_time=5.0, duration=3.0, gpu=8.0, cpu=32.0, ram=256.0),
        ]
        mq = run_queue_simulation(jobs, FirstFitScheduler(), config=cfg, queue_policy="fifo")
        self.assertAlmostEqual(mq.mean_wait_time, 0.0, places=9)
        self.assertAlmostEqual(mq.makespan, 8.0, places=9)

    def test_unsorted_arrivals_and_jct_identity(self) -> None:
        """도착 시각이 뒤섞인 입력도 인과 순으로 정렬 처리되며 mean_jct == mean_wait + mean_dur 성립."""
        cfg = ClusterConfig(num_nodes=2)
        jobs = [
            Job(2, arrival_time=2.0, duration=3.0, gpu=8.0, cpu=32.0, ram=256.0),
            Job(0, arrival_time=0.0, duration=4.0, gpu=8.0, cpu=32.0, ram=256.0),
            Job(1, arrival_time=1.0, duration=5.0, gpu=8.0, cpu=32.0, ram=256.0),
        ]
        mq = run_queue_simulation(jobs, AdaptiveBLindyPackScheduler(), config=cfg, queue_policy="fifo")
        self.assertEqual(mq.completed_jobs, 3)
        # 노드 2개: Job0(0->4), Job1(1->6) 즉시 실행, Job2(arr=2)는 t=4에 시작(wait=2, end=7)
        self.assertAlmostEqual(mq.mean_wait_time, 2.0 / 3.0, places=9)
        self.assertAlmostEqual(mq.mean_jct, (4.0 + 5.0 + 5.0) / 3.0, places=9)
        self.assertAlmostEqual(mq.makespan, 7.0, places=9)

    def test_queue_running_age_uses_actual_start_time_not_arrival(self) -> None:
        """대기 큐에서 오래 기다린 뒤 방금 시작된 작업의 실행 경과시간(age)은 대기 시간이 아닌 실제 실행 시간이어야 한다."""
        cfg = ClusterConfig(num_nodes=1)
        state = ClusterState(cfg)
        j_old = Job(0, arrival_time=0.0, duration=10.0, gpu=8.0, cpu=32.0, ram=256.0)
        state.now = 0.0
        alloc0 = state.allocate(0, j_old)
        state.now = 10.0
        state.release(0, j_old, alloc0)

        # j_queued는 t=1.0에 도착했으나 t=10.0에 실제 실행 시작
        j_queued = Job(1, arrival_time=1.0, duration=5.0, gpu=4.0, cpu=16.0, ram=128.0)
        state.allocate(0, j_queued)
        state.now = 12.0
        max_age, mean_age, min_age = state.node_age_stats(np.array([0]))
        # 실제 실행 경과시간은 12.0 - 10.0 = 2.0 (arrival 기준 11.0이 아님!)
        self.assertAlmostEqual(max_age[0], 2.0, places=9)
        self.assertAlmostEqual(mean_age[0], 2.0, places=9)
        self.assertAlmostEqual(min_age[0], 2.0, places=9)
        self.assertAlmostEqual(state.now - state.node_oldest_start[0], 2.0, places=9)

    def test_starvation_free_backfill_prevents_hol_starvation(self) -> None:
        """소형 잡이 계속 도착해도 max_hol_bypasses 한도로 인해 8-GPU 헤드 작업이 기아에 빠지지 않는다."""
        cfg = ClusterConfig(num_nodes=1)
        jobs = [
            Job(0, 0.0, 2.0, 1.0, 8.0, 64.0),   # 노드 점유 시작
            Job(1, 0.1, 1.0, 8.0, 32.0, 256.0), # 8-GPU 헤드 작업 (t=0.1 도착, 대기 필요)
        ]
        # t=0.2부터 0.2 간격으로 소형 잡 30개가 연속 도착 (무제한 백필이면 Job 1이 마지막까지 기아 발생)
        for k in range(30):
            jobs.append(Job(2 + k, 0.2 + 0.2 * k, 1.5, 1.0, 8.0, 64.0))

        mq = run_queue_simulation(
            jobs, FirstFitScheduler(), config=cfg, queue_policy="backfill", backfill_window=16, max_hol_bypasses=3
        )
        self.assertEqual(mq.completed_jobs, len(jobs))
        # Job 1이 기아 없이 일찍 실행되었으므로 전체 makespan이 유한하고 모든 작업 완료
        self.assertLess(mq.p99_jct, 10.0)

    def test_blocking_bit_exact_regression(self) -> None:
        """기존 Blocking 시뮬레이션 핵심 회귀 수치가 simulator.py 확장 이후에도 소수점 9자리까지 일치."""
        cfg = ClusterConfig(num_nodes=64)
        jobs = generate_w1_pai_mixed(seed=11)
        m_l2 = run_simulation(jobs, L2NormScheduler(), config=cfg)
        m_bl = run_simulation(jobs, BoundedLindyPackScheduler(DEFAULT_BASE_THETA), config=cfg)
        self.assertAlmostEqual(m_l2.gpu_hour_blocking_rate, 3.0012644009, places=8)
        self.assertAlmostEqual(m_bl.gpu_hour_blocking_rate, 1.3934976393, places=8)

    def test_queue_rejects_nan_inf_nonint_gpu_and_invalid_config(self) -> None:
        """NaN/Inf 입력(무한루프 유발 포함), 비정수 멀티 GPU, 잘못된 ClusterConfig가 즉시 ValueError를 발생시킨다."""
        cfg = ClusterConfig(num_nodes=1)
        sch = FirstFitScheduler()
        bad_jobs = [
            Job(0, float("nan"), 1.0, 1.0, 8.0, 64.0),
            Job(0, float("inf"), 1.0, 1.0, 8.0, 64.0),
            Job(0, 0.0, float("nan"), 1.0, 8.0, 64.0),
            Job(0, 0.0, float("inf"), 1.0, 8.0, 64.0),
            Job(0, 0.0, 1.0, float("nan"), 8.0, 64.0),
            Job(0, 0.0, 1.0, float("inf"), 8.0, 64.0),
            Job(0, 0.0, 1.0, 1.0, float("nan"), 64.0),
            Job(0, 0.0, 1.0, 1.0, 8.0, float("nan")),
            Job(0, 0.0, 1.0, 1.5, 8.0, 64.0),
            Job(0, 0.0, 1.0, 8.4, 8.0, 64.0),
        ]
        for bj in bad_jobs:
            with self.assertRaises(ValueError):
                run_queue_simulation([bj], sch, config=cfg)

        for bad_cfg in [
            ClusterConfig(num_nodes=0),
            ClusterConfig(num_nodes=1, gpus_per_node=0),
            ClusterConfig(num_nodes=1, cpu_per_node=0.0),
            ClusterConfig(num_nodes=1, ram_per_node=-1.0),
        ]:
            with self.assertRaises(ValueError):
                run_queue_simulation([Job(0, 0.0, 1.0, 1.0, 8.0, 64.0)], sch, config=bad_cfg)

    def test_fp_near_simultaneous_departure_before_arrival_scale_invariant(self) -> None:
        """0.1 + 0.2 = 0.30000000000000004 같은 부동소수점 반올림 오차에서도 종료가 도착보다 먼저 처리된다."""
        cfg = ClusterConfig(num_nodes=1)
        j1 = [
            Job(0, arrival_time=0.1, duration=0.2, gpu=8.0, cpu=32.0, ram=256.0),
            Job(1, arrival_time=0.3, duration=0.5, gpu=8.0, cpu=32.0, ram=256.0),
        ]
        j10 = [Job(j.job_id, j.arrival_time * 10.0, j.duration * 10.0, j.gpu, j.cpu, j.ram) for j in j1]
        mq1 = run_queue_simulation(j1, FirstFitScheduler(), config=cfg, queue_policy="fifo")
        mq10 = run_queue_simulation(j10, FirstFitScheduler(), config=cfg, queue_policy="fifo")
        self.assertEqual(mq1.max_queue_len, 0)
        self.assertEqual(mq10.max_queue_len, 0)
        self.assertAlmostEqual(mq1.mean_wait_time, 0.0, places=12)
        self.assertAlmostEqual(mq1.avg_stranded_gpu_on_wait, 0.0, places=12)
        self.assertAlmostEqual(mq1.makespan, mq10.makespan / 10.0, places=12)


class TestAdaptiveBLindyPackProperties(unittest.TestCase):
    """Adaptive B-LindyPack의 수학적 불변량·비예측성·온라인 적응성 테스트."""

    def test_clock_unit_invariance_blocking_and_queue(self) -> None:
        """타임스탬프와 수명을 극단 척도(1e-6 ~ 1e6, 분/초)로 변환해도 콜드스타트 포함 배치 결정이 100% 동일해야 한다."""
        sch = AdaptiveBLindyPackScheduler()
        # 기존 콜드스타트 회귀 버그를 유발했던 소규모 클러스터(8/16노드, W1 seed=101) 포함 엄격 검증
        for n_nodes, gen_fn, seed in [
            (8, generate_w1_pai_mixed, 101),
            (16, generate_w1_pai_mixed, 101),
            (32, generate_w3_heavy_straggler, 77),
        ]:
            cfg = ClusterConfig(num_nodes=n_nodes)
            jobs_hr = gen_fn(seed=seed, num_jobs=350, target_load=0.88)
            mb_hr = run_simulation(jobs_hr, sch, config=cfg)
            mq_hr = run_queue_simulation(jobs_hr, sch, config=cfg, queue_policy="fifo")

            for scale in (1e-6, 1.0 / 3600.0, 60.0, 3600.0, 1e6):
                jobs_sc = [
                    Job(j.job_id, j.arrival_time * scale, j.duration * scale, j.gpu, j.cpu, j.ram)
                    for j in jobs_hr
                ]
                mb_sc = run_simulation(jobs_sc, sch, config=cfg)
                self.assertAlmostEqual(mb_hr.gpu_hour_blocking_rate, mb_sc.gpu_hour_blocking_rate, places=9)

                mq_sc = run_queue_simulation(jobs_sc, sch, config=cfg, queue_policy="fifo")
                self.assertAlmostEqual(mq_hr.mean_jct, mq_sc.mean_jct / scale, places=7)
                self.assertAlmostEqual(mq_hr.makespan, mq_sc.makespan / scale, places=7)
                self.assertAlmostEqual(mq_hr.mean_slowdown, mq_sc.mean_slowdown, places=7)

    def test_ifr_antilindy_cv_gate_protection(self) -> None:
        """증가고장률(IFR: Uniform(1.8, 2.2), CV≈0.06) 환경에서 CV 게이트가 자동으로 린디 항을 감쇠시킨다."""
        from confirm_30seeds import generate_ifr_antilindy_ood

        cfg = ClusterConfig(num_nodes=64)
        jobs = generate_ifr_antilindy_ood(seed=11, num_jobs=3000)
        sch_cv = AdaptiveBLindyPackScheduler(record_diagnostics=True)
        sch_nocv = AdaptiveBLindyPackScheduler(
            adapt_cfg=AdaptiveLindyConfig(cv_center=-10.0, cv_slope=0.08),
            record_diagnostics=True,
        )
        m_cv = run_simulation(jobs, sch_cv, config=cfg)
        m_nocv = run_simulation(jobs, sch_nocv, config=cfg)
        diag = sch_cv.get_diagnostics_summary()

        q_cv = run_queue_simulation(jobs, sch_cv, config=cfg, queue_policy="fifo")
        q_nocv = run_queue_simulation(jobs, sch_nocv, config=cfg, queue_policy="fifo")

        # IFR 환경에서 CV 유효성 게이트 gamma가 자동으로 낮게(<=0.15) 수렴하고 Blocking·FIFO 대기시간 악화를 방지
        self.assertLess(diag["gamma_mean"], 0.15)
        self.assertLess(m_cv.gpu_hour_blocking_rate, m_nocv.gpu_hour_blocking_rate)
        self.assertLess(q_cv.mean_wait_time, q_nocv.mean_wait_time)

    def test_strict_non_clairvoyance(self) -> None:
        """
        반증 테스트: 시각 T=5.0 이전 도착·종료 이력은 완전히 동일하고,
        T=5.0 이후에야 끝나는 작업들의 미래 수명만 크게 바꾼 두 트레이스를 실행했을 때
        T=5.0까지의 모든 노드 선택 결과가 100% 일치해야 한다 (미래 수명 유출 0%).
        """
        cfg = ClusterConfig(num_nodes=16)
        base_jobs = generate_w1_pai_mixed(seed=55, num_jobs=400)
        cutoff = base_jobs[150].arrival_time

        # cutoff 시점 이전에 아직 끝나지 않은 작업들의 미래 수명을 임의로 3배 변경
        mutated_jobs = []
        for j in base_jobs:
            if j.arrival_time + j.duration > cutoff:
                mutated_jobs.append(Job(j.job_id, j.arrival_time, j.duration * 3.0 + 10.0, j.gpu, j.cpu, j.ram))
            else:
                mutated_jobs.append(copy.copy(j))

        class RecordingAdaptive(AdaptiveBLindyPackScheduler):
            def __init__(self):
                super().__init__()
                self.choices = []

            def reset(self, config):
                super().reset(config)
                self.choices = []

            def select_node(self, state, feasible_nodes, gpu, cpu, ram):
                node = super().select_node(state, feasible_nodes, gpu, cpu, ram)
                if state.now <= cutoff - 1e-9:
                    self.choices.append((round(state.now, 9), node))
                return node

        s1 = RecordingAdaptive()
        run_simulation(base_jobs, s1, config=cfg)
        c1 = list(s1.choices)

        run_simulation(mutated_jobs, s1, config=cfg)
        c2 = list(s1.choices)

        self.assertGreater(len(c1), 50)
        self.assertEqual(c1, c2)

    def test_online_correlation_regime_separation(self) -> None:
        """W1(양의 상관)에서는 alpha가 낮고, W3(무상관 스트래글러)에서는 alpha가 높아야 한다."""
        cfg = ClusterConfig(num_nodes=64)
        s = AdaptiveBLindyPackScheduler(record_diagnostics=True)

        run_simulation(generate_w1_pai_mixed(seed=33, num_jobs=2000), s, config=cfg)
        d_w1 = s.get_diagnostics_summary()

        run_simulation(generate_w3_heavy_straggler(seed=33, num_jobs=2000), s, config=cfg)
        d_w3 = s.get_diagnostics_summary()

        self.assertGreater(d_w1["rho_mean"], 0.15)
        self.assertLess(d_w1["alpha_mean"], 0.15)
        self.assertLess(d_w3["rho_mean"], 0.06)
        self.assertGreater(d_w3["alpha_mean"], 0.70)

    def test_reset_idempotency(self) -> None:
        """동일 스케줄러 인스턴스를 반복 실행해도 내부 상태 누수 없이 100% 동일 결과를 반환한다."""
        cfg = ClusterConfig(num_nodes=32)
        jobs = generate_w1_pai_mixed(seed=44, num_jobs=800)
        sch = AdaptiveBLindyPackScheduler(record_diagnostics=True)
        m1 = run_simulation(jobs, sch, config=cfg)
        m2 = run_simulation(jobs, sch, config=cfg)
        self.assertEqual(m1.gpu_hour_blocking_rate, m2.gpu_hour_blocking_rate)
        self.assertEqual(m1.avg_stranded_gpu_ratio, m2.avg_stranded_gpu_ratio)

    def test_simultaneous_t0_batch_does_not_poison_correlation_and_config_validation(self) -> None:
        """동시 도착(t=0.0) 배치에서 미정의 상관(0.0)이 EWMA를 오염시키지 않으며, 잘못된 설정은 거부된다."""
        cfg = ClusterConfig(num_nodes=8)
        batch_t0 = [
            Job(i, arrival_time=0.0, duration=5.0, gpu=1.0 if i % 2 == 0 else 4.0, cpu=8.0, ram=64.0)
            for i in range(15)
        ]
        sch = AdaptiveBLindyPackScheduler(record_diagnostics=True)
        run_simulation(batch_t0, sch, config=cfg)
        # 경과시간이 0인 t=0 동시 배치 구간에서 prior_rho(0.20)가 0으로 끌려 내려가지 않아야 함
        self.assertEqual(len(sch.rho_history), 1)
        self.assertAlmostEqual(sch.rho_history[0], sch.adapt_cfg.prior_rho, places=9)

        with self.assertRaises(ValueError):
            AdaptiveLindyConfig(corr_source="invalid_mode")
        with self.assertRaises(ValueError):
            AdaptiveLindyConfig(ewma_beta=1.5)
        with self.assertRaises(ValueError):
            AdaptiveLindyConfig(refresh_interval=-1)


class TestStatsUtilsAccuracy(unittest.TestCase):
    """stats_utils.py 정확 통계 검정 수학적 정밀도 검증."""

    def test_exact_student_t_against_textbook_critical_values(self) -> None:
        self.assertAlmostEqual(exact_student_t_pvalue(2.2621571627, 9), 0.05, places=8)
        self.assertAlmostEqual(exact_student_t_pvalue(2.0452296421, 29), 0.05, places=8)
        self.assertAlmostEqual(exact_student_t_pvalue(2.7563859037, 29), 0.01, places=8)
        self.assertAlmostEqual(exact_student_t_pvalue(3.6594149547, 29), 0.001, places=7)
        self.assertAlmostEqual(student_t_critical_95(29), 2.0452296421, places=6)

    def test_wilcoxon_and_sign_test_edge_cases(self) -> None:
        # 모두 양수 n=10 -> 정확 p = 2 / 2^10 = 1/512
        w, p = wilcoxon_signed_rank_test(np.arange(1, 11, dtype=float))
        self.assertEqual(w, 0.0)
        self.assertAlmostEqual(p, 1.0 / 512.0, places=12)

        # 모두 0인 차이 -> p = 1.0
        _, p_zero = wilcoxon_signed_rank_test(np.zeros(10))
        self.assertEqual(p_zero, 1.0)
        t_z, pt_z = paired_t_test_exact(np.zeros(10))
        self.assertEqual(t_z, 0.0)
        self.assertEqual(pt_z, 1.0)

        # 부호 검정 정확값 및 대표본(n > 1024) 오버플로 안전성 검증
        self.assertAlmostEqual(exact_sign_test_pvalue(10, 0), 2.0 / 1024.0, places=12)
        self.assertEqual(exact_sign_test_pvalue(5, 5), 1.0)
        self.assertEqual(exact_sign_test_pvalue(0, 0), 1.0)
        p_large_1 = exact_sign_test_pvalue(600, 500)
        self.assertAlmostEqual(p_large_1, 0.002819544991, places=9)
        p_large_2 = exact_sign_test_pvalue(5100, 4900)
        self.assertAlmostEqual(p_large_2, 0.046585527705, places=8)
        self.assertAlmostEqual(student_t_critical_95(1), 12.706204736, places=5)

    def test_stats_invalid_inputs_rejected(self) -> None:
        """NaN/Inf, 빈 배열, 길이 불일치, 음수 승/패가 조용히 p=0.0을 반환하지 않고 ValueError를 발생시킨다."""
        with self.assertRaises(ValueError):
            exact_student_t_pvalue(float("nan"), 29)
        with self.assertRaises(ValueError):
            paired_t_test_exact(np.array([1.0, float("nan"), 3.0]))
        with self.assertRaises(ValueError):
            wilcoxon_signed_rank_test(np.array([1.0, float("nan"), 3.0]))
        with self.assertRaises(ValueError):
            paired_summary_stats(np.array([]), np.array([]))
        with self.assertRaises(ValueError):
            paired_summary_stats(np.array([1.0, 2.0]), np.array([1.0]))
        with self.assertRaises(ValueError):
            paired_summary_stats(np.array([1.0, float("nan"), 3.0]), np.array([0.5, 1.0, 2.0]))
        with self.assertRaises(ValueError):
            exact_sign_test_pvalue(-1, 5)

    def test_paired_summary_stats_structure(self) -> None:
        base = np.array([10.0, 11.0, 9.5, 10.5, 12.0])
        cand = np.array([9.0, 10.0, 8.5, 9.5, 11.0])
        s = paired_summary_stats(base, cand)
        self.assertEqual(s["wins"], 5)
        self.assertEqual(s["losses"], 0)
        self.assertAlmostEqual(s["diff_mean"], 1.0, places=9)
        self.assertEqual(s["p_value_t"], 0.0)

    def test_stats_rejects_multidimensional_arrays(self) -> None:
        """2차원 이상 배열이 paired_t_test_exact·wilcoxon_signed_rank_test에서 오계산되지 않고 거부된다."""
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            paired_t_test_exact(mat)
        with self.assertRaises(ValueError):
            wilcoxon_signed_rank_test(mat)


class TestConfirmPipelineSmoke(unittest.TestCase):
    """confirm_30seeds.py 워커 및 집계 파이프라인의 소규모 end-to-end 스모크 검증."""

    def test_worker_and_summarize_smoke(self) -> None:
        import json
        from pathlib import Path
        import confirm_30seeds as c30

        sol = json.loads((Path(__file__).parent / "evolved_solution.json").read_text(encoding="utf-8"))
        c30._init_worker(np.array(sol["distilled_weights"], dtype=np.float64))

        # 2개 시드 × 핵심 스케줄러로 _worker_run_seed_scheduler + _summarize_metric 검증
        seeds = [11, 22]
        sch_names = ["FirstFit", "L2Norm", "B-LindyPack", "Adaptive-B-Lindy"]
        recs = {s: {} for s in sch_names}
        for s in seeds:
            for sch_n in sch_names:
                rec = c30._worker_run_seed_scheduler(("W1_PAI_Mixed", s, sch_n, True))
                recs[sch_n][s] = rec
                self.assertIsNotNone(rec["backfill"])
        self.assertIn("gamma_mean", recs["Adaptive-B-Lindy"][11]["diag_block"])

        blk = c30._summarize_metric(recs, seeds, "blocking", "gpu_hour_blocking_rate", ["FirstFit", "L2Norm"])
        bfw = c30._summarize_metric(recs, seeds, "backfill", "mean_wait_time", ["FirstFit", "L2Norm"])
        self.assertIn("vs_best_sota", blk["comparisons"])
        self.assertIn("vs_best_sota", bfw["comparisons"])
        self.assertTrue(math.isfinite(bfw["comparisons"]["vs_best_sota"]["p_value_t"]))


class TestHeterogeneousAndRealTraces(unittest.TestCase):
    """6단계(B안): 이기종 클러스터, gpu_spec 제약, 실제 트레이스 로더 정합성 테스트."""

    def test_hetero_aware_bit_exact_on_homogeneous(self) -> None:
        """동종 8-GPU 클러스터에서 v5.0과 Adaptive-B-Lindy-Het 결과가 소수점 12자리까지 일치한다."""
        cfg = ClusterConfig(num_nodes=32)
        for gen_fn, s in ((generate_w1_pai_mixed, 101), (generate_w3_heavy_straggler, 202)):
            jobs = gen_fn(seed=s, num_jobs=1200)
            s_v5 = AdaptiveBLindyPackScheduler(hetero_aware=False)
            s_het = AdaptiveBLindyPackScheduler(hetero_aware=True)
            mb0 = run_simulation(jobs, s_v5, config=cfg)
            mb1 = run_simulation(jobs, s_het, config=cfg)
            self.assertAlmostEqual(mb0.gpu_hour_blocking_rate, mb1.gpu_hour_blocking_rate, places=12)
            self.assertAlmostEqual(mb0.avg_stranded_gpu_ratio, mb1.avg_stranded_gpu_ratio, places=12)

            mq0 = run_queue_simulation(jobs[:600], s_v5, config=cfg, queue_policy="fifo")
            mq1 = run_queue_simulation(jobs[:600], s_het, config=cfg, queue_policy="fifo")
            self.assertAlmostEqual(mq0.mean_wait_time, mq1.mean_wait_time, places=12)

    def test_heterogeneous_cluster_and_gpu_spec_invariants(self) -> None:
        """1/2/4/8-GPU 이기종 노드 및 gpu_spec 제약에서 위반 0건과 자원 100% 반환을 보장한다."""
        from baselines import BestFitScheduler, DotProductScheduler, FGDScheduler

        cfg = ClusterConfig(
            num_nodes=6,
            gpus_per_node=8,
            cpu_per_node=128.0,
            ram_per_node=512.0,
            node_gpus=(1, 2, 4, 8, 2, 8),
            node_cpus=(16.0, 32.0, 64.0, 128.0, 32.0, 128.0),
            node_rams=(64.0, 128.0, 256.0, 512.0, 128.0, 512.0),
            node_models=("T4", "T4", "P100", "G2", "V100M32", "V100M32"),
        )
        jobs = [
            Job(0, 0.0, 3.0, 0.5, 4.0, 16.0, gpu_spec="T4"),
            Job(1, 0.2, 4.0, 2.0, 16.0, 64.0, gpu_spec="V100M32"),
            Job(2, 0.5, 2.0, 4.0, 32.0, 128.0, gpu_spec="P100|G2"),
            Job(3, 0.8, 1.5, 0.25, 4.0, 16.0, gpu_spec=None),
            Job(4, 1.0, 2.5, 8.0, 64.0, 256.0, gpu_spec="G2|V100M32"),
            Job(5, 1.2, 1.0, 1.0, 8.0, 32.0, gpu_spec="T4"),
        ]
        schedulers = [
            FirstFitScheduler(),
            BestFitScheduler(),
            DotProductScheduler(),
            L2NormScheduler(),
            FGDScheduler(adaptive=False),
            FGDScheduler(adaptive=True),
            BoundedLindyPackScheduler(DEFAULT_BASE_THETA),
            AdaptiveBLindyPackScheduler(hetero_aware=False),
            AdaptiveBLindyPackScheduler(hetero_aware=True),
        ]
        for sch in schedulers:
            mb = run_simulation(jobs, sch, config=cfg)
            self.assertGreaterEqual(mb.gpu_hour_blocking_rate, 0.0)
            mq = run_queue_simulation(jobs, sch, config=cfg, queue_policy="backfill")
            self.assertEqual(mq.completed_jobs, len(jobs))
            self.assertAlmostEqual(mq.mean_jct, mq.mean_wait_time + sum(j.duration for j in jobs) / len(jobs), places=9)

        # 호환 모델이 없는 gpu_spec 요청은 Queue 모드에서 즉시 ValueError 차단
        bad_jobs = [Job(99, 0.0, 1.0, 1.0, 4.0, 16.0, gpu_spec="H100")]
        with self.assertRaises(ValueError):
            run_queue_simulation(bad_jobs, L2NormScheduler(), config=cfg)

        # ClusterState.get_feasible_nodes가 허용 기종만 반환하는지 직접 확인
        st = ClusterState(cfg)
        feas_v100 = st.get_feasible_nodes(4.0, 16.0, 64.0, gpu_spec="V100M32")
        self.assertEqual(feas_v100.tolist(), [5])

    def test_real_trace_loaders_and_audit(self) -> None:
        """Alibaba 2023 및 Helios 파서의 경계 필터링, 스케일링, FGD 분포 산출 정합성을 검증한다."""
        import tempfile
        from pathlib import Path
        from trace_loader import (
            build_empirical_fgd_distribution,
            extract_helios_windows,
            load_alibaba_2023,
            load_helios_cluster,
            scale_heterogeneous_cluster,
        )

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            node_csv = td / "nodes.csv"
            node_csv.write_text(
                "sn,cpu_milli,memory_mib,gpu,model\n"
                "n0,64000,262144,2,T4\n"
                "n1,128000,524288,8,G2\n"
                "n2,32000,131072,0,NONE\n",
                encoding="utf-8",
            )
            pod_csv = td / "pods.csv"
            pod_csv.write_text(
                "name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,pod_phase,creation_time,deletion_time,scheduled_time\n"
                "p_cpu,4000,8192,0,0,,BE,Running,100,200,100\n"
                "p_bad,4000,8192,1,1000,,LS,Running,200,150,200\n"
                "p_multi_over,8000,16384,16,1000,,LS,Running,100,500,100\n"
                "p_infeas_spec,4000,8192,1,1000,H100,LS,Running,100,500,100\n"
                "p_frac,4000,8192,1,500,T4,LS,Running,100,3700,100\n"
                "p_big,16000,32768,4,1000,G2,LS,Running,200,7400,200\n",
                encoding="utf-8",
            )
            cfg, jobs, audit = load_alibaba_2023(node_csv, pod_csv)
            self.assertEqual(cfg.num_nodes, 2)
            self.assertEqual(audit.raw_rows, 6)
            self.assertEqual(audit.excluded_cpu_only, 1)
            self.assertEqual(audit.excluded_bad_time, 1)
            self.assertEqual(audit.excluded_multinode, 1)
            self.assertEqual(audit.excluded_infeasible_node, 1)
            self.assertEqual(audit.valid_gpu_jobs, 2)
            self.assertAlmostEqual(jobs[0].gpu, 0.5, places=6)
            self.assertAlmostEqual(jobs[0].duration, 1.0, places=6)

            scaled = scale_heterogeneous_cluster(cfg, 0.5, seed=7, jobs=jobs)
            self.assertEqual(scaled.num_nodes, 2)  # T4와 G2 기종 호환성 모두 유지 확인
            reqs, probs = build_empirical_fgd_distribution(jobs)
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=9)
            self.assertEqual(len(reqs), 2)

            hel_csv = td / "helios.csv"
            lines = ["job_id,user,vc,gpu_num,cpu_num,node_num,state,submit_time,start_time,end_time,duration,queue\n"]
            for i in range(60):
                # 60개 유효 1~4 GPU 작업 (1시간 간격, 수명 4시간) + 1개 CPU-only + 1개 멀티노드
                lines.append(
                    f"{i},u1,vc1,{(i % 3) + 1},8,1,COMPLETED,2020-04-01 {i // 6:02d}:{(i % 6) * 10:02d}:00,"
                    f"2020-04-01 {i // 6:02d}:{(i % 6) * 10:02d}:00,2020-04-01 {(i // 6) + 4:02d}:00:00,14400,0\n"
                )
            lines.append("998,u1,vc1,0,4,1,COMPLETED,2020-04-01 00:00:00,2020-04-01 00:00:00,2020-04-01 01:00:00,3600,0\n")
            lines.append("999,u1,vc1,16,32,2,COMPLETED,2020-04-01 00:00:00,2020-04-01 00:00:00,2020-04-01 01:00:00,3600,0\n")
            hel_csv.write_text("".join(lines), encoding="utf-8")

            h_jobs, h_audit = load_helios_cluster(hel_csv, "TestCluster")
            self.assertEqual(h_audit.valid_gpu_jobs, 60)
            self.assertEqual(h_audit.excluded_cpu_only, 1)
            self.assertEqual(h_audit.excluded_multinode, 1)
            wins = extract_helios_windows(h_jobs, num_windows=2, jobs_per_window=20, target_load=0.85, min_span_hours=1.0)
            self.assertEqual(len(wins), 2)


class TestPhase7OmniLindyAndMultiNode(unittest.TestCase):
    """7단계(D안): Omni-Lindy 단일노드 동종 비트-동일성, 멀티노드 Gang 원자성·Locality, HoL 해소, 멀티노드 로더 검증."""

    def test_omni_lindy_bit_exact_on_single_node_homogeneous(self) -> None:
        """동종 단일 노드(W1, W3)에서 Omni-Lindy와 v5.0 결과가 소수점 12자리까지 일치한다."""
        cfg = ClusterConfig(num_nodes=32)
        for gen_fn, s in ((generate_w1_pai_mixed, 101), (generate_w3_heavy_straggler, 202)):
            jobs = gen_fn(seed=s, num_jobs=1200)
            s_v5 = AdaptiveBLindyPackScheduler(hetero_aware=False, topology_aware=False)
            s_omni = AdaptiveBLindyPackScheduler(hetero_aware=True, topology_aware=True)
            mb0 = run_simulation(jobs, s_v5, config=cfg)
            mb1 = run_simulation(jobs, s_omni, config=cfg)
            self.assertAlmostEqual(mb0.gpu_hour_blocking_rate, mb1.gpu_hour_blocking_rate, places=12)
            self.assertAlmostEqual(mb0.avg_stranded_gpu_ratio, mb1.avg_stranded_gpu_ratio, places=12)

            mq0 = run_queue_simulation(jobs[:600], s_v5, config=cfg, queue_policy="fifo")
            mq1 = run_queue_simulation(jobs[:600], s_omni, config=cfg, queue_policy="fifo")
            self.assertAlmostEqual(mq0.mean_wait_time, mq1.mean_wait_time, places=12)

    def test_multinode_gang_allocation_atomicity_and_locality(self) -> None:
        """2/4/8노드 Gang 할당의 All-or-Nothing 원자성, 자원 100% 반환, 단일 랙 Locality 및 통신 패널티를 검증한다."""
        from baselines import RackBestFitScheduler

        cfg = ClusterConfig(num_nodes=16, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0, nodes_per_rack=8)
        # 랙 0에 1노드 점유 1건 배치 후 8노드 Gang 작업 도착 -> 반드시 온전한 랙 1 내부에 100% 집적되어야 함
        jobs = [
            Job(0, arrival_time=0.0, duration=5.0, gpu=4.0, cpu=32.0, ram=256.0, num_nodes=1),
            Job(1, arrival_time=0.5, duration=4.0, gpu=64.0, cpu=512.0, ram=4096.0, num_nodes=8),
            Job(2, arrival_time=1.0, duration=2.0, gpu=16.0, cpu=128.0, ram=1024.0, num_nodes=2),
            # 랙 간 스팬 강제 (12노드 = 최소 2랙 스팬) -> cross_rack_penalty 반영 검증
            Job(3, arrival_time=10.0, duration=10.0, gpu=96.0, cpu=768.0, ram=6144.0, num_nodes=12),
        ]
        for sch in (RackBestFitScheduler(), AdaptiveBLindyPackScheduler(hetero_aware=True, topology_aware=True)):
            mb = run_simulation(jobs[:3], sch, config=cfg)
            self.assertEqual(mb.rejected_jobs, 0)
            self.assertAlmostEqual(mb.single_rack_locality_rate, 100.0, places=9)

            mq0 = run_queue_simulation(jobs[3:4], sch, config=cfg, queue_policy="fifo", cross_rack_penalty=0.0)
            mqp = run_queue_simulation(jobs[3:4], sch, config=cfg, queue_policy="fifo", cross_rack_penalty=0.08)
            self.assertAlmostEqual(mq0.mean_jct, 10.0, places=9)
            self.assertAlmostEqual(mqp.mean_jct, 10.8, places=9)

        # 클러스터 전체 가용 노드보다 큰 num_nodes 요청은 Queue 모드에서 즉시 ValueError 차단
        with self.assertRaises(ValueError):
            run_queue_simulation(
                [Job(99, 0.0, 1.0, 128.0, 1024.0, 8192.0, num_nodes=20)],
                RackBestFitScheduler(),
                config=cfg,
            )

    def test_alibaba_gpuspec33_no_hol_lockup(self) -> None:
        """Alibaba GPUSpec33 (kr=0.01)에서 Omni-Lindy가 342h HoL Lockup 없이 Backfill Wait < 50h를 유지한다."""
        from pathlib import Path
        from trace_loader import load_alibaba_2023, scale_heterogeneous_cluster, slice_active_trace

        ali_dir = Path(__file__).parent / "data" / "alibaba_2023"
        cfg, jobs, _ = load_alibaba_2023(ali_dir / "openb_node_list_gpu_node.csv", ali_dir / "openb_pod_list_gpuspec33.csv")
        act, _ = slice_active_trace(jobs, 0.05)
        sub = scale_heterogeneous_cluster(cfg, 0.01, seed=101, jobs=act)

        omni = AdaptiveBLindyPackScheduler(hetero_aware=True, topology_aware=True)
        mb = run_simulation(act, omni, config=sub)
        mq = run_queue_simulation(act, omni, config=sub, queue_policy="backfill")
        self.assertLess(mb.gpu_hour_blocking_rate, 0.5)
        self.assertLess(mq.mean_wait_time, 50.0)

    def test_helios_multinode_loader_and_windows(self) -> None:
        """load_helios_cluster_multinode 및 extract_helios_multinode_windows의 멀티노드 파싱·랙 정렬을 검증한다."""
        import tempfile
        from pathlib import Path
        from trace_loader import extract_helios_multinode_windows, load_helios_cluster_multinode

        with tempfile.TemporaryDirectory() as tmp:
            hel_csv = Path(tmp) / "helios_mn.csv"
            lines = ["job_id,user,vc,gpu_num,cpu_num,node_num,state,submit_time,start_time,end_time,duration,queue\n"]
            for i in range(60):
                is_mn = (i % 4 == 0)
                nn = 4 if is_mn else 1
                gn = 32 if is_mn else 4
                lines.append(
                    f"{i},u1,vc1,{gn},{8 * nn},{nn},COMPLETED,2020-04-01 {i // 6:02d}:{(i % 6) * 10:02d}:00,"
                    f"2020-04-01 {i // 6:02d}:{(i % 6) * 10:02d}:00,2020-04-01 {(i // 6) + 5:02d}:00:00,18000,0\n"
                )
            lines.append("901,u1,vc1,0,4,1,COMPLETED,2020-04-01 00:00:00,2020-04-01 00:00:00,2020-04-01 01:00:00,3600,0\n")
            lines.append("902,u1,vc1,128,256,16,COMPLETED,2020-04-01 00:00:00,2020-04-01 00:00:00,2020-04-01 01:00:00,3600,0\n")
            lines.append("903,u1,vc1,5,8,2,COMPLETED,2020-04-01 00:00:00,2020-04-01 00:00:00,2020-04-01 01:00:00,3600,0\n")
            hel_csv.write_text("".join(lines), encoding="utf-8")

            jobs, audit = load_helios_cluster_multinode(hel_csv, "TestMN", max_nodes_per_job=8)
            self.assertEqual(audit.valid_gpu_jobs, 60)
            self.assertEqual(audit.excluded_cpu_only, 1)
            self.assertEqual(audit.excluded_multinode, 1)
            self.assertEqual(audit.excluded_infeasible_node, 1)
            self.assertTrue(any(j.num_nodes == 4 for j in jobs))

            wins = extract_helios_multinode_windows(jobs, num_windows=2, jobs_per_window=20, target_load=0.85, min_span_hours=1.0, nodes_per_rack=8)
            self.assertEqual(len(wins), 2)
            for wc, _, _ in wins:
                self.assertEqual(wc.num_nodes % 8, 0)


class TestK8sExtenderPrototype(unittest.TestCase):
    """Kubernetes Scheduler HTTP Extender 어댑터 및 라이브 HTTP 서버 통합 테스트."""

    @staticmethod
    def _make_node(
        name: str,
        gpus: int = 8,
        cpu: str = "64",
        mem: str = "512Gi",
        model: str = "V100M32",
        rack: str = "rack-0",
        slots_free: list[float] | None = None,
        cpu_free: str | None = None,
        ram_free_gi: float | None = None,
        running_pods: list[dict] | None = None,
    ) -> dict:
        import json

        annotations: dict[str, str] = {}
        if slots_free is not None:
            annotations["omni-lindy.io/gpu-slots-free"] = json.dumps(slots_free)
        if cpu_free is not None:
            annotations["omni-lindy.io/cpu-free"] = cpu_free
        if ram_free_gi is not None:
            annotations["omni-lindy.io/ram-free-gi"] = str(ram_free_gi)
        if running_pods is not None:
            annotations["omni-lindy.io/running-pods"] = json.dumps(running_pods)
        return {
            "metadata": {
                "name": name,
                "labels": {
                    "omni-lindy.io/gpu-model": model,
                    "omni-lindy.io/rack": rack,
                },
                "annotations": annotations,
            },
            "status": {
                "allocatable": {
                    "nvidia.com/gpu": str(gpus),
                    "cpu": cpu,
                    "memory": mem,
                }
            },
        }

    @staticmethod
    def _make_pod(
        name: str,
        gpu: float = 1.0,
        cpu: str = "8",
        mem: str = "64Gi",
        gpu_spec: str | None = None,
        gang_nodes: int | None = None,
    ) -> dict:
        annotations: dict[str, str] = {}
        res_req: dict[str, str] = {"cpu": cpu, "memory": mem}
        if 0.0 < gpu < 1.0:
            annotations["omni-lindy.io/gpu-milli"] = str(int(round(gpu * 1000)))
        else:
            res_req["nvidia.com/gpu"] = str(int(round(gpu)))
        if gpu_spec is not None:
            annotations["omni-lindy.io/gpu-spec"] = gpu_spec
        if gang_nodes is not None:
            annotations["omni-lindy.io/gang-nodes"] = str(gang_nodes)
        return {
            "metadata": {"name": name, "namespace": "default", "annotations": annotations},
            "spec": {"containers": [{"name": "worker", "resources": {"requests": res_req}}]},
        }

    @staticmethod
    def _http_json(base_url: str, path: str, method: str = "GET", payload: dict | str | None = None):
        import json
        import urllib.error
        import urllib.request

        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
        try:
            with opener.open(req, timeout=5.0) as resp:
                raw = resp.read().decode("utf-8")
                ctype = resp.headers.get("Content-Type", "")
                parsed = json.loads(raw) if "application/json" in ctype else raw
                return resp.status, parsed
        except urllib.error.HTTPError as err:
            raw = err.read().decode("utf-8")
            ctype = err.headers.get("Content-Type", "") if err.headers else ""
            err.close()
            parsed = json.loads(raw) if (raw and "application/json" in ctype) else raw
            return err.code, parsed

    def test_k8s_resource_parsers(self) -> None:
        from k8s_extender.adapter import parse_k8s_cpu, parse_k8s_memory_gi, parse_pod_resources

        self.assertAlmostEqual(parse_k8s_cpu("500m"), 0.5, places=9)
        self.assertAlmostEqual(parse_k8s_cpu("16"), 16.0, places=9)
        self.assertAlmostEqual(parse_k8s_memory_gi("512Gi"), 512.0, places=9)
        self.assertAlmostEqual(parse_k8s_memory_gi("1024Mi"), 1.0, places=9)
        self.assertAlmostEqual(parse_k8s_memory_gi("1Ti"), 1024.0, places=9)

        pod = self._make_pod("frac-pod", gpu=0.25, cpu="1500m", mem="16Gi", gpu_spec="T4|V100", gang_nodes=2)
        g, c, r, spec, gn = parse_pod_resources(pod)
        self.assertAlmostEqual(g, 0.25, places=9)
        self.assertAlmostEqual(c, 1.5, places=9)
        self.assertAlmostEqual(r, 16.0, places=9)
        self.assertEqual(spec, "T4|V100")
        self.assertEqual(gn, 2)

        with self.assertRaises(ValueError):
            parse_k8s_cpu("-1")
        with self.assertRaises(ValueError):
            parse_pod_resources(self._make_pod("bad-gang", gpu=1.0, gang_nodes=0))

    def test_scores_to_k8s_priorities_monotonicity(self) -> None:
        from k8s_extender.adapter import scores_to_k8s_priorities

        prio = scores_to_k8s_priorities(["n0", "n1", "n2"], np.array([1.5, 0.5, 2.5]))
        score_map = {item["host"]: item["score"] for item in prio}
        # K8s v1.34 Extender 공식 규약 [0, 10]: 원시 점수가 가장 낮은 n1이 10, 가장 높은 n2가 0
        self.assertEqual(score_map["n1"], 10)
        self.assertEqual(score_map["n2"], 0)
        self.assertEqual(score_map["n0"], 5)

        # 모두 동률일 때는 전원 10
        ties = scores_to_k8s_priorities(["a", "b"], np.array([3.0, 3.0]))
        self.assertTrue(all(item["score"] == 10 for item in ties))
        self.assertEqual(scores_to_k8s_priorities([], np.array([])), [])

    def test_live_http_healthz_filter_and_prioritize(self) -> None:
        import threading
        from k8s_extender.server import create_extender_server

        server = create_extender_server(host="127.0.0.1", port=0, nodes_per_rack=4, quiet=True)
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            st, hz = self._http_json(base_url, "/healthz", method="GET")
            self.assertEqual(st, 200)
            self.assertEqual(hz["status"], "ok")
            self.assertEqual(hz["scheduler"], "Omni-Lindy")

            nodes = [
                # n0: 2 GPU 사용 중(초기 시작), 4-GPU 파드 수용 불가 (V100)
                self._make_node(
                    "n0", gpus=4, model="V100", rack="r0",
                    slots_free=[0.0, 1.0, 1.0, 1.0], cpu_free="32", ram_free_gi=256.0,
                    running_pods=[{"startTime": 900.0, "gpu": 1.0}],
                ),
                # n1: T4, 분수 0.5 슬롯 존재하지만 cpu_free=2 부족
                self._make_node(
                    "n1", gpus=2, model="T4", rack="r0",
                    slots_free=[0.5, 1.0], cpu_free="2", ram_free_gi=64.0,
                    running_pods=[{"startTime": 950.0, "gpu": 0.5}],
                ),
                # n2: V100, 부분 슬롯 0.5 보유 + 매우 오래된 스트래글러(start=100) 실행 중
                self._make_node(
                    "n2", gpus=8, model="V100", rack="r0",
                    slots_free=[0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], cpu_free="48", ram_free_gi=384.0,
                    running_pods=[{"startTime": 100.0, "gpu": 0.5}],
                ),
                # n3: V100, 완전 유휴 노드
                self._make_node(
                    "n3", gpus=8, model="V100", rack="r0",
                    slots_free=[1.0] * 8, cpu_free="64", ram_free_gi=512.0,
                    running_pods=[],
                ),
            ]
            pod = self._make_pod("infer-half", gpu=0.5, cpu="4", mem="32Gi", gpu_spec="V100")
            args = {"Pod": pod, "Nodes": {"items": nodes}, "nowUnix": 1000.0}

            st_f, res_f = self._http_json(base_url, "/filter", method="POST", payload=args)
            self.assertEqual(st_f, 200)
            self.assertEqual(res_f["NodeNames"], ["n0", "n2", "n3"])
            self.assertIn("n1", res_f["FailedNodes"])

            # 필터 통과 노드만 /prioritize 전송 후 직접 코어 스코어와 일치성 검증
            prio_args = {"Pod": pod, "Nodes": res_f["Nodes"], "nowUnix": 1000.0}
            st_p, res_p = self._http_json(base_url, "/prioritize", method="POST", payload=prio_args)
            self.assertEqual(st_p, 200)
            score_map = {item["host"]: item["score"] for item in res_p}
            self.assertEqual(set(score_map.keys()), {"n0", "n2", "n3"})
            best_http_host = max(score_map, key=lambda k: score_map[k])
            self.assertEqual(score_map[best_http_host], 10)
            self.assertTrue(all(0 <= v <= 10 for v in score_map.values()))

            # 동일 상태에서 Omni-Lindy 코어가 직접 고른 노드와 100% 일치해야 함
            core_state, feas_names = server.adapter.build_cluster_state(
                res_f["Nodes"]["items"], now_unix=1000.0, gpu_spec="V100"
            )
            core_sch = AdaptiveBLindyPackScheduler(hetero_aware=True, topology_aware=True)
            feas_idx = core_state.get_feasible_nodes(0.5, 4.0, 32.0, gpu_spec="V100")
            chosen_core_name = feas_names[core_sch.select_node(core_state, feas_idx, 0.5, 4.0, 32.0)]
            self.assertEqual(best_http_host, chosen_core_name)
        finally:
            server.shutdown()
            server.server_close()
            th.join(timeout=2.0)

    def test_live_http_completion_events_and_metrics(self) -> None:
        import threading
        from k8s_extender.server import create_extender_server

        server = create_extender_server(host="127.0.0.1", port=0, quiet=True)
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            # 저분산(IFR) 완료 40건을 순차 배치(16 + 8*3)로 주입해 EWMA가 수렴하며 gamma < 0.5로 자동 감쇠함을 확인
            completions = [{"gpu": 1.0, "duration": 10.0 + 0.1 * (i % 3)} for i in range(40)]
            batches = [completions[:16], completions[16:24], completions[24:32], completions[32:40]]
            for batch in batches:
                st_c, res_c = self._http_json(
                    base_url, "/events/pod_completed", method="POST", payload={"completions": batch}
                )
                self.assertEqual(st_c, 200)
            self.assertEqual(res_c["added"], 8)
            self.assertEqual(res_c["completions_buffered"], 40)

            st_m, res_m = self._http_json(base_url, "/metrics", method="GET")
            self.assertEqual(st_m, 200)
            self.assertEqual(res_m["completion_events"], 40)
            self.assertLess(res_m["gamma_lindy_validity"], 0.5)

            st_prom, res_prom = self._http_json(base_url, "/metrics?format=prometheus", method="GET")
            self.assertEqual(st_prom, 200)
            self.assertIn("omni_lindy_gamma_gate", res_prom)
        finally:
            server.shutdown()
            server.server_close()
            th.join(timeout=2.0)

    def test_live_http_gang_schedule_rack_locality(self) -> None:
        import threading
        from k8s_extender.server import create_extender_server

        server = create_extender_server(host="127.0.0.1", port=0, nodes_per_rack=4, quiet=True)
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            # rack-0: 4대 모두 8 GPU 비어있음 / rack-1: 2대만 8 GPU 비어있고 2대는 부족
            nodes = [
                self._make_node("r0-n0", gpus=8, rack="rack-0"),
                self._make_node("r0-n1", gpus=8, rack="rack-0"),
                self._make_node("r0-n2", gpus=8, rack="rack-0"),
                self._make_node("r0-n3", gpus=8, rack="rack-0"),
                self._make_node("r1-n0", gpus=8, rack="rack-1"),
                self._make_node("r1-n1", gpus=8, rack="rack-1"),
                self._make_node("r1-n2", gpus=8, rack="rack-1", slots_free=[0.0] * 8),
                self._make_node("r1-n3", gpus=8, rack="rack-1", slots_free=[0.0] * 8),
            ]
            gang_pod = self._make_pod("ddp-32gpu", gpu=8.0, cpu="32", mem="256Gi", gang_nodes=4)
            st_g, res_g = self._http_json(
                base_url,
                "/gang_schedule",
                method="POST",
                payload={"Pod": gang_pod, "Nodes": nodes, "nowUnix": 2000.0},
            )
            self.assertEqual(st_g, 200)
            self.assertTrue(res_g["schedulable"])
            self.assertEqual(res_g["racksSpanned"], 1)
            self.assertTrue(res_g["singleRackLocal"])
            self.assertEqual(set(res_g["selectedNodes"]), {"r0-n0", "r0-n1", "r0-n2", "r0-n3"})
        finally:
            server.shutdown()
            server.server_close()
            th.join(timeout=2.0)

    def test_live_http_malformed_requests_return_400(self) -> None:
        import threading
        from k8s_extender.server import create_extender_server

        server = create_extender_server(host="127.0.0.1", port=0, quiet=True)
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            st1, _ = self._http_json(base_url, "/filter", method="POST", payload="{invalid-json")
            self.assertEqual(st1, 400)

            st2, _ = self._http_json(base_url, "/prioritize", method="POST", payload={"Pod": {}})
            self.assertEqual(st2, 400)

            st3, _ = self._http_json(
                base_url,
                "/events/pod_completed",
                method="POST",
                payload={"gpu": -1.0, "duration": 5.0},
            )
            self.assertEqual(st3, 400)

            st4, _ = self._http_json(base_url, "/unknown_route", method="GET")
            self.assertEqual(st4, 404)
        finally:
            server.shutdown()
            server.server_close()
            th.join(timeout=2.0)


class TestPhase8OmniMRLAndBigTech(unittest.TestCase):
    """8단계 `Omni-MRL` (v8.0) 및 빅테크 비교군(`NILAS`, `LAVA`, `DVLA`) 불변량·방향성·통합 검증."""

    def test_omni_mrl_strict_non_clairvoyance(self) -> None:
        from omni_mrl import OmniMRLScheduler

        cfg = ClusterConfig(num_nodes=8)
        t_cut = 5.0
        base = generate_w3_heavy_straggler(seed=42, num_jobs=300)
        alt = copy.deepcopy(base)
        for j in alt:
            if j.arrival_time <= t_cut and (j.arrival_time + j.duration) > t_cut:
                j.duration *= 15.0
            elif j.arrival_time > t_cut:
                j.duration = 0.01

        trace_a = [j for j in base if j.arrival_time <= t_cut]
        trace_b = [j for j in alt if j.arrival_time <= t_cut]

        # 두 트레이스에서 t <= t_cut 구간의 모든 배치 결정이 100% 일치해야 함
        s1 = OmniMRLScheduler()
        s2 = OmniMRLScheduler()

        class _LogMRL(OmniMRLScheduler):
            def __init__(self) -> None:
                super().__init__()
                self.decisions: list[int] = []

            def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
                picked = super().select_node(state, feasible_nodes, gpu, cpu, ram)
                self.decisions.append(picked)
                return picked

        l1, l2 = _LogMRL(), _LogMRL()
        run_simulation(trace_a, l1, config=cfg)
        run_simulation(trace_b, l2, config=cfg)
        self.assertEqual(l1.decisions, l2.decisions)
        self.assertGreater(len(l1.decisions), 10)
        # 기본 인스턴스 리셋 멱등성 확인
        m1 = run_simulation(trace_a, s1, config=cfg).gpu_hour_blocking_rate
        m2 = run_simulation(trace_a, s2, config=cfg).gpu_hour_blocking_rate
        self.assertAlmostEqual(m1, m2, places=12)

    def test_omni_mrl_clock_unit_invariance(self) -> None:
        from omni_mrl import OmniMRLScheduler

        cfg = ClusterConfig(num_nodes=16)
        base_jobs = generate_w1_pai_mixed(seed=101, num_jobs=400)
        sch = OmniMRLScheduler()
        ref_blk = run_simulation(base_jobs, sch, config=cfg).gpu_hour_blocking_rate
        ref_q = run_queue_simulation(base_jobs, sch, config=cfg, queue_policy="backfill")

        for scale in (1e-5, 60.0, 3600.0, 1e5):
            scaled = [
                Job(j.job_id, j.arrival_time * scale, j.duration * scale, j.gpu, j.cpu, j.ram, j.gpu_spec, j.num_nodes)
                for j in base_jobs
            ]
            mb = run_simulation(scaled, sch, config=cfg)
            mq = run_queue_simulation(scaled, sch, config=cfg, queue_policy="backfill")
            self.assertAlmostEqual(mb.gpu_hour_blocking_rate, ref_blk, places=9)
            self.assertAlmostEqual(mq.mean_wait_time / scale, ref_q.mean_wait_time, places=8)
            self.assertAlmostEqual(mq.mean_jct / scale, ref_q.mean_jct, places=8)

    def test_omni_mrl_bidirectional_hazard_sign_reversal(self) -> None:
        """
        핵심 방향성 반증 테스트:
        - IFR(균일 수명 CV≈0.03) 환경에서는 곧 비워질 OLD 노드를 아끼고 YOUNG 노드에 동기화 패킹
        - DFR(헤비테일 CV>3.0) 환경에서는 장기 스트래글러 OLD 노드에 신규 작업을 격리
        """
        from omni_mrl import OmniMRLScheduler

        cfg = ClusterConfig(num_nodes=2, gpus_per_node=8)
        # 노드 0: OLD 작업 (start=0.0, age=9.5), 노드 1: YOUNG 작업 (start=9.0, age=0.5)
        def _make_state(completions: list[tuple[float, float]]) -> ClusterState:
            st = ClusterState(cfg)
            st.now = 0.0
            st.allocate(0, Job(1, 0.0, 20.0, 1.0, 8.0, 64.0))
            st.now = 9.0
            st.allocate(1, Job(2, 9.0, 20.0, 1.0, 8.0, 64.0))
            st.now = 10.0
            for g, d in completions:
                st.recent_completions.append((g, d))
            return st

        # 1) IFR 완료 버퍼 (모두 약 10시간 부근에서 종료 -> CV 매우 낮음)
        ifr_comps = [(1.0, 9.8 + 0.1 * (i % 5)) for i in range(60)]
        st_ifr = _make_state(ifr_comps)
        sch_ifr = OmniMRLScheduler()
        sch_ifr.reset(cfg)
        for _ in range(5):
            sch_ifr._last_stats_ops = -10**9
            sch_ifr._last_mrl_ops = -10**9
            st_ifr._ops_counter += 1
            chosen_ifr = sch_ifr.select_node(st_ifr, np.array([0, 1]), 1.0, 8.0, 64.0)
        # IFR 모드에서는 곧 끝날 노드 0(age=9.5) 대신 방금 시작된 노드 1(age=0.5) 선택
        self.assertGreater(sch_ifr._cached_gamma_ifr, 0.7)
        self.assertEqual(chosen_ifr, 1)

        # 2) DFR 헤비테일 완료 버퍼 (짧은 잡 다수 + 초장기 꼬리 -> CV >> 2.0)
        dfr_comps = [(1.0, 0.5 if i % 6 != 0 else 45.0) for i in range(60)]
        st_dfr = _make_state(dfr_comps)
        sch_dfr = OmniMRLScheduler()
        sch_dfr.reset(cfg)
        for _ in range(5):
            sch_dfr._last_stats_ops = -10**9
            sch_dfr._last_mrl_ops = -10**9
            st_dfr._ops_counter += 1
            chosen_dfr = sch_dfr.select_node(st_dfr, np.array([0, 1]), 1.0, 8.0, 64.0)
        # DFR 모드에서는 gamma_ifr == 0.0이며, 장기 스트래글러 노드 0(age=9.5)을 선택해 격리
        self.assertEqual(sch_dfr._cached_gamma_ifr, 0.0)
        self.assertEqual(chosen_dfr, 0)

    def test_bigtech_baselines_invariants(self) -> None:
        from bigtech_baselines import (
            AlibabaDVLAGPUScheduler,
            GoogleLAVAClassGPUScheduler,
            GoogleNILASGPUScheduler,
        )

        cfg = ClusterConfig(
            num_nodes=8,
            node_gpus=(8, 8, 8, 8, 4, 4, 2, 2),
            node_cpus=(64.0, 64.0, 64.0, 64.0, 32.0, 32.0, 16.0, 16.0),
            node_rams=(512.0, 512.0, 512.0, 512.0, 256.0, 256.0, 128.0, 128.0),
            node_models=("V100", "V100", "V100", "V100", "T4", "T4", "G2", "G2"),
            nodes_per_rack=4,
        )
        jobs = [
            Job(1, 0.0, 4.0, 1.0, 8.0, 64.0, gpu_spec="V100"),
            Job(2, 0.2, 1.0, 0.25, 2.0, 16.0),
            Job(3, 0.4, 3.0, 16.0, 128.0, 1024.0, gpu_spec="V100", num_nodes=2),
            Job(4, 0.6, 2.0, 0.5, 4.0, 32.0, gpu_spec="T4|G2"),
        ] + [Job(10 + i, 1.0 + 0.1 * i, 1.5, 1.0, 8.0, 32.0) for i in range(25)]

        for sch in (
            GoogleNILASGPUScheduler(topology_aware=True),
            GoogleLAVAClassGPUScheduler(topology_aware=True),
            AlibabaDVLAGPUScheduler(topology_aware=True),
        ):
            mb = run_simulation(jobs, sch, config=cfg)
            mq = run_queue_simulation(jobs, sch, config=cfg, queue_policy="backfill", cross_rack_penalty=0.08)
            self.assertEqual(mq.completed_jobs, len(jobs))
            self.assertGreaterEqual(mb.gpu_hour_blocking_rate, 0.0)
            self.assertLessEqual(mq.single_rack_locality_rate, 100.0 + 1e-9)

    def test_k8s_extender_omni_mrl_live_http(self) -> None:
        import threading
        from k8s_extender.server import create_extender_server

        server = create_extender_server(host="127.0.0.1", port=0, quiet=True, engine="omni_mrl")
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            st_h, hz = TestK8sExtenderPrototype._http_json(base_url, "/healthz", method="GET")
            self.assertEqual(st_h, 200)
            self.assertEqual(hz["engine"], "omni_mrl")
            self.assertEqual(hz["version"], "11.1")

            # IFR 저분산 완료 이벤트 배치 순차 주입 -> mrl_mode == "ifr_nilas" 활성화 확인
            completions = [{"gpu": 1.0, "duration": 10.0 + 0.05 * (i % 3)} for i in range(40)]
            for batch in (completions[:16], completions[16:24], completions[24:32], completions[32:40]):
                st_c, _ = TestK8sExtenderPrototype._http_json(
                    base_url, "/events/pod_completed", method="POST", payload={"completions": batch}
                )
                self.assertEqual(st_c, 200)

            st_m, res_m = TestK8sExtenderPrototype._http_json(base_url, "/metrics", method="GET")
            self.assertEqual(st_m, 200)
            self.assertEqual(res_m["mrl_mode"], "ifr_nilas")
            self.assertGreater(res_m["gamma_ifr"], 0.5)

            nodes = [
                TestK8sExtenderPrototype._make_node(
                    "old-ifr", gpus=8, slots_free=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    running_pods=[{"startTime": 90.5, "gpu": 1.0}],
                ),
                TestK8sExtenderPrototype._make_node(
                    "young-ifr", gpus=8, slots_free=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    running_pods=[{"startTime": 99.5, "gpu": 1.0}],
                ),
            ]
            pod = TestK8sExtenderPrototype._make_pod("p-1gpu", gpu=1.0, cpu="8", mem="64Gi")
            st_p, res_p = TestK8sExtenderPrototype._http_json(
                base_url, "/prioritize", method="POST",
                payload={"Pod": pod, "Nodes": {"items": nodes}, "nowUnix": 100.0},
            )
            self.assertEqual(st_p, 200)
            score_map = {item["host"]: item["score"] for item in res_p}
            self.assertEqual(score_map["young-ifr"], 10)
            self.assertEqual(score_map["old-ifr"], 0)
        finally:
            server.shutdown()
            server.server_close()
            th.join(timeout=2.0)

    def test_omni_mrl_ifr_non_clairvoyance_and_decision_clock_invariance(self) -> None:
        from confirm_30seeds import generate_ifr_antilindy_ood
        from omni_mrl import OmniMRLScheduler
        from run_phase8_eval import generate_ifr_weibull_k2

        class _DecisionMRL(OmniMRLScheduler):
            def __init__(self) -> None:
                super().__init__()
                self.decisions: list[int] = []

            def select_node(self, state, feasible_nodes, gpu, cpu, ram) -> int:
                picked = super().select_node(state, feasible_nodes, gpu, cpu, ram)
                self.decisions.append(picked)
                return picked

        # 1) IFR 경로(gamma_ifr > 0.5)가 실제 켜진 상태에서 반사실 Non-Clairvoyance 결정 100% 일치
        cfg = ClusterConfig(num_nodes=16)
        t_cut = 18.0
        for gen_fn in (generate_ifr_antilindy_ood, generate_ifr_weibull_k2):
            base = gen_fn(seed=202, num_jobs=900)
            alt = copy.deepcopy(base)
            for j in alt:
                if j.arrival_time <= t_cut and (j.arrival_time + j.duration) > t_cut:
                    j.duration = j.duration * 9.0 + 3.0
                elif j.arrival_time > t_cut:
                    j.duration = 0.01
            tr_a = [j for j in base if j.arrival_time <= t_cut]
            tr_b = [j for j in alt if j.arrival_time <= t_cut]
            s_a, s_b = _DecisionMRL(), _DecisionMRL()
            run_simulation(tr_a, s_a, config=cfg)
            run_simulation(tr_b, s_b, config=cfg)
            self.assertEqual(s_a.decisions, s_b.decisions)
            self.assertGreater(len(s_a.decisions), 50)
            self.assertGreater(s_a.get_diagnostics_summary()["gamma_ifr_active_frac"], 0.20)

        # 2) IFR_Uniform에서 비이진 시간 배율(1/3, 7.3, 3600) 변환 시 배치 결정 시퀀스 100% 일치
        ifr_jobs = generate_ifr_antilindy_ood(seed=303, num_jobs=1200)
        ref_s = _DecisionMRL()
        run_simulation(ifr_jobs, ref_s, config=cfg)
        self.assertGreater(ref_s.get_diagnostics_summary()["gamma_ifr_active_frac"], 0.4)
        for scale in (1.0 / 3.0, 7.3, 3600.0):
            scaled = [
                Job(j.job_id, j.arrival_time * scale, j.duration * scale, j.gpu, j.cpu, j.ram, j.gpu_spec, j.num_nodes)
                for j in ifr_jobs
            ]
            cur_s = _DecisionMRL()
            run_simulation(scaled, cur_s, config=cfg)
            self.assertEqual(cur_s.decisions, ref_s.decisions)

    def test_survival_model_km_continuity_and_censoring(self) -> None:
        from bigtech_baselines import EmpiricalSurvivalModel

        cfg = ClusterConfig(num_nodes=4)
        st = ClusterState(cfg)
        st.now = 100.0
        for i in range(1, 21):
            st.recent_completions.append((1.0, float(i)))

        m1 = EmpiricalSurvivalModel(refresh_interval=0)
        m1.update(st, force=True)
        m1.cv_eff = 0.999999
        v1 = float(m1.predict_mrl(np.array([20.0 / m1.ref_scale]))[0])

        m2 = EmpiricalSurvivalModel(refresh_interval=0)
        m2.update(st, force=True)
        m2.cv_eff = 1.000000
        v2 = float(m2.predict_mrl(np.array([20.0 / m2.ref_scale]))[0])
        self.assertLess(abs(v2 - v1) / max(v1, v2, 1e-9), 1e-4)

        # 코호트 우측 중도절단(동일 코호트 내 장기 미완료 작업 존재) 시 MRL이 상승함 확인
        st_cens = ClusterState(cfg)
        st_cens.now = 100.0
        for i in range(1, 21):
            st_cens.recent_completions.append((1.0, float(i)))
            st_cens.recent_cohort.append([20.0, 20.0 + float(i), 1.0])
        for k in range(10):
            st_cens.recent_cohort.append([60.0, None, 1.0])  # age = 40.0 > max completed(20.0)
        mc = EmpiricalSurvivalModel(refresh_interval=0)
        mc.update(st_cens, force=True)
        v_plain = float(m1.predict_mrl(np.array([0.8]))[0])
        v_cens = float(mc.predict_mrl(np.array([0.8]))[0])
        self.assertGreater(v_cens, v_plain)

    def test_bigtech_baselines_behavioral_fidelity_and_harness_parity(self) -> None:
        from bigtech_baselines import (
            AlibabaDVLAGPUScheduler,
            GoogleLAVAClassGPUScheduler,
            GoogleLAVAMLOracleScheduler,
            GoogleNILASGPUScheduler,
            run_ml_lava_simulation,
        )
        from confirm_30seeds import generate_ifr_antilindy_ood
        from adaptive_lindy import BPackSpatialOnlyScheduler
        from omni_mrl import OmniMRLScheduler

        cfg = ClusterConfig(num_nodes=16)
        jobs = generate_ifr_antilindy_ood(seed=404, num_jobs=500)

        # 1) 분위수 기반 LAVA-ML은 IFR_Uniform에서도 4클래스(LC0..LC3) 모두 활성화해야 함
        pred_map = {id(j): j.duration for j in jobs}
        ml_sch = GoogleLAVAMLOracleScheduler(pred_map=pred_map, spatial_base="bpack")
        lcs = ml_sch._hours_to_lc(np.array([j.duration for j in jobs]))
        for cls_id in range(4):
            self.assertGreater(int(np.sum(lcs == cls_id)), 50)

        # 2) run_ml_lava_simulation과 simulator.py가 동일 비오라클 스케줄러에서 소수점 12자리까지 일치
        for sch in (BPackSpatialOnlyScheduler(), OmniMRLScheduler()):
            ref_b = run_simulation(jobs, sch, config=cfg)
            ref_q = run_queue_simulation(jobs, sch, config=cfg, queue_policy="backfill")
            ml_res = run_ml_lava_simulation(jobs, sch, config=cfg)
            self.assertAlmostEqual(ml_res["blocking"], ref_b.gpu_hour_blocking_rate, places=12)
            self.assertAlmostEqual(ml_res["backfill_wait"], ref_q.mean_wait_time, places=12)

        # 3) spatial_base="bpack" 빅테크 3종 정상 실행 검증
        for sch in (
            GoogleNILASGPUScheduler(spatial_base="bpack"),
            GoogleLAVAClassGPUScheduler(spatial_base="bpack"),
            AlibabaDVLAGPUScheduler(spatial_base="bpack"),
        ):
            res = run_simulation(jobs, sch, config=cfg)
            self.assertGreaterEqual(res.gpu_hour_blocking_rate, 0.0)

    def test_k8s_extender_partial_annotation_and_cpu_only_node_fix(self) -> None:
        from k8s_extender.adapter import OmniLindyK8sAdapter, parse_pod_resources

        adp = OmniLindyK8sAdapter(engine="omni_mrl")

        # 1) CPU free annotation만 있어도 runningPods의 1-GPU 사용량이 정확히 차감되어야 함
        gpu_node = {
            "metadata": {"name": "n-1gpu", "annotations": {"omni-lindy.io/cpu-free": "48"}},
            "status": {"allocatable": {"nvidia.com/gpu": "1", "cpu": "64", "memory": "256Gi"}},
        }
        running_pod = {
            "spec": {"nodeName": "n-1gpu", "containers": [{"resources": {"limits": {"nvidia.com/gpu": "1"}}}]},
            "status": {"startTime": 50.0},
        }
        st, _ = adp.build_cluster_state([gpu_node], running_pods=[running_pod], now_unix=100.0)
        self.assertAlmostEqual(float(np.sum(st.gpu_slots[0])), 0.0, places=9)
        self.assertAlmostEqual(float(st.cpu_free[0]), 48.0, places=9)

        # 2) nvidia.com/gpu가 없는 CPU 전용 노드는 0-GPU 노드로 간주돼 1-GPU Pod filter를 탈락
        cpu_only_node = {
            "metadata": {"name": "cpu-node"},
            "status": {"allocatable": {"cpu": "64", "memory": "256Gi"}},
        }
        gpu_pod = TestK8sExtenderPrototype._make_pod("need-1g", gpu=1.0, cpu="4", mem="8Gi")
        f_res = adp.filter_nodes({"Pod": gpu_pod, "Nodes": {"items": [cpu_only_node]}, "nowUnix": 100.0})
        self.assertEqual(len(f_res["NodeNames"]), 0)
        self.assertIn("cpu-node", f_res["FailedNodes"])

        # CPU-only Pod(gpu=0)는 CPU-only 노드 /prioritize에서도 0 나눗셈 없이 정상 점수 반환
        cpu_pod = {
            "metadata": {"name": "cpu-pod"},
            "spec": {"containers": [{"resources": {"requests": {"cpu": "4", "memory": "8Gi"}}}]},
        }
        p_res = adp.prioritize_nodes({"Pod": cpu_pod, "Nodes": {"items": [cpu_only_node]}, "nowUnix": 100.0})
        self.assertEqual(p_res[0]["score"], 10)

        # 3) initContainers GPU + spec.overhead 합산 및 NodeNames 캐시 미스 시 ValueError
        ic_pod = {
            "metadata": {"name": "ic-pod"},
            "spec": {
                "initContainers": [{"resources": {"limits": {"nvidia.com/gpu": "2", "cpu": "2", "memory": "4Gi"}}}],
                "containers": [{"resources": {"requests": {"nvidia.com/gpu": "1", "cpu": "4", "memory": "8Gi"}}}],
                "overhead": {"cpu": "1", "memory": "2Gi"},
            },
        }
        pg, pc, pr, _, _ = parse_pod_resources(ic_pod)
        self.assertAlmostEqual(pg, 2.0, places=9)
        self.assertAlmostEqual(pc, 5.0, places=9)
        self.assertAlmostEqual(pr, 10.0, places=9)

        with self.assertRaises(ValueError):
            adp.prioritize_nodes({"Pod": cpu_pod, "NodeNames": ["unknown-cached-node"]})

    def test_holm_bonferroni_and_bootstrap_ci(self) -> None:
        from stats_utils import holm_bonferroni_adjust, paired_bootstrap_ci

        adj = holm_bonferroni_adjust([0.01, 0.04, 0.03, 0.005])
        expected = [0.03, 0.06, 0.06, 0.02]
        for a, e in zip(adj, expected):
            self.assertAlmostEqual(a, e, places=12)

        lo, hi = paired_bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), n_boot=500, seed=42)
        self.assertLessEqual(lo, 3.0)
        self.assertGreaterEqual(hi, 3.0)

    def test_cohort_km_unbiased_steady_state_and_u_cut_memoryless(self) -> None:
        from bigtech_baselines import EmpiricalSurvivalModel, GoogleNILASGPUScheduler

        cfg = ClusterConfig(num_nodes=16)

        # 1) 정상상태 Exp(1) 시뮬레이션에서 코호트 KM q50이 길이편향(+78%) 없이 참값 0.693 근처 유지
        rng = np.random.default_rng(777)
        exp_jobs = [Job(i, float(i) * 0.08, float(rng.exponential(1.0)), 1.0, 4.0, 32.0) for i in range(1200)]
        st = ClusterState(cfg)
        import heapq
        h: list[tuple[float, int, int, Job, object]] = []
        seq = 0
        q50_ratios: list[float] = []
        km = EmpiricalSurvivalModel(refresh_interval=0)
        for jb in exp_jobs:
            while h and h[0][0] <= jb.arrival_time:
                dt, _, ni, dj, al = heapq.heappop(h)
                st.now = dt
                st.release(ni, dj, al)
            st.now = jb.arrival_time
            feas = st.get_feasible_nodes(jb.gpu, jb.cpu, jb.ram)
            if len(feas) > 0:
                al = st.allocate(int(feas[0]), jb)
                seq += 1
                heapq.heappush(h, (jb.arrival_time + jb.duration, seq, int(feas[0]), jb, al))
            if len(st.recent_cohort) >= 300 and jb.job_id % 100 == 0:
                km.update(st, force=True)
                q50_abs = km.q50 * km.ref_scale
                q50_ratios.append(q50_abs / math.log(2.0))
        self.assertTrue(0.80 <= float(np.mean(q50_ratios)) <= 1.20)

        # 2) Lognormal(sigma=1.5) 코호트 KM q75가 참값(~2.75) 대비 15배 폭발 없이 정상 범위 유지
        rng_ln = np.random.default_rng(888)
        ln_durs = np.exp(rng_ln.normal(0.0, 1.5, size=1200))
        ln_jobs = [Job(i, float(i) * 0.25, float(ln_durs[i]), 1.0, 4.0, 32.0) for i in range(1200)]
        sch_ln = GoogleNILASGPUScheduler(spatial_base="bpack")
        run_simulation(ln_jobs, sch_ln, config=cfg)
        q75_abs = sch_ln.survival.q75 * max(sch_ln.survival.ref_scale, 1e-9)
        true_q75 = math.exp(1.5 * 0.67449)
        self.assertLess(q75_abs, 2.5 * true_q75)

        # 3) u_cut 좌·우극한 C^0 연속성 및 CV=1.0 무기억성/C^1 연속 검증
        km.cv_eff = 1.0
        u_c = km._km_u_cut
        v_left = float(km.predict_mrl(np.array([u_c - 1e-9]))[0])
        v_right = float(km.predict_mrl(np.array([u_c + 1e-9]))[0])
        self.assertLess(abs(v_left - v_right) / max(v_right, 1e-9), 1e-6)

        v_tail = km.predict_mrl(np.array([u_c, u_c + 0.5, u_c + 2.0, u_c + 5.0]))
        self.assertLess(float(np.max(np.abs(v_tail - km._km_r_cut))), 1e-10)

        km.cv_eff = 0.999999
        m_low = float(km.predict_mrl(np.array([u_c + 1.5]))[0])
        km.cv_eff = 1.000001
        m_high = float(km.predict_mrl(np.array([u_c + 1.5]))[0])
        self.assertLess(abs(m_high - m_low), 1e-8)

    def test_k8s_extender_running_pods_annotation_deducts_gpu_no_overcommit(self) -> None:
        from k8s_extender.adapter import OmniLindyK8sAdapter

        adp = OmniLindyK8sAdapter(engine="omni_mrl")
        node = {
            "metadata": {
                "name": "gpu-busy-4",
                "annotations": {
                    "omni-lindy.io/running-pods": '[{"startTime": 50.0, "gpu": 4.0}]',
                },
            },
            "status": {"allocatable": {"nvidia.com/gpu": "8", "cpu": "64", "memory": "512Gi"}},
        }
        pod_8g = TestK8sExtenderPrototype._make_pod("want-8g", gpu=8.0, cpu="8", mem="32Gi")
        pod_4g = TestK8sExtenderPrototype._make_pod("want-4g", gpu=4.0, cpu="8", mem="32Gi")

        f8 = adp.filter_nodes({"Pod": pod_8g, "Nodes": {"items": [node]}, "nowUnix": 100.0})
        self.assertEqual(len(f8["NodeNames"]), 0)
        self.assertIn("gpu-busy-4", f8["FailedNodes"])

        f4 = adp.filter_nodes({"Pod": pod_4g, "Nodes": {"items": [node]}, "nowUnix": 100.0})
        self.assertEqual(f4["NodeNames"], ["gpu-busy-4"])

    def test_nilas_ml_continuous_oracle_harness_and_monotonicity(self) -> None:
        from bigtech_baselines import GoogleNILASMLOracleScheduler, run_ml_lava_simulation
        from confirm_30seeds import generate_ifr_antilindy_ood

        cfg = ClusterConfig(num_nodes=16)
        jobs = generate_ifr_antilindy_ood(seed=515, num_jobs=400)
        pred_map = {id(j): j.duration for j in jobs}
        sch = GoogleNILASMLOracleScheduler(pred_map=pred_map, spatial_base="bpack")
        res = run_ml_lava_simulation(jobs, sch, cfg)
        self.assertGreaterEqual(res["blocking"], 0.0)
        self.assertGreaterEqual(res["backfill_wait"], 0.0)


class TestPhase11MathematicalAndK8sCompliance(unittest.TestCase):
    """11단계 MRL 수학 정합성, 식별 불가 분위수, 계층 부트스트랩, K8s v1.34 규약, 원자료 검증 테스트."""

    def test_mrl_c0_boundary_smooth_tail_cv1_and_unidentifiable_quantiles(self) -> None:
        from bigtech_baselines import EmpiricalSurvivalModel

        cfg = ClusterConfig(num_nodes=4)
        st = ClusterState(cfg)
        st.now = 100.0
        for i in range(1, 21):
            st.recent_completions.append((1.0, float(i)))

        km = EmpiricalSurvivalModel(refresh_interval=0)
        km.update(st, force=True)
        u_cut = km._km_u_cut
        r_cut = km._km_r_cut
        h = 1e-6

        # 1) u_cut 경계 C^0 연속성 및 u-방향 좌·우 미분 비일치(정확한 C^0 성질) 검증
        km.cv_eff = 1.0
        v_l = float(km.predict_mrl(np.array([u_cut - h]))[0])
        v_0 = float(km.predict_mrl(np.array([u_cut]))[0])
        v_r = float(km.predict_mrl(np.array([u_cut + h]))[0])
        self.assertLess(abs(v_l - r_cut), 1e-5)
        self.assertLess(abs(v_0 - r_cut), 1e-12)
        self.assertLess(abs(v_r - r_cut), 1e-12)

        left_deriv = (v_0 - v_l) / h
        right_deriv_cv1 = (v_r - v_0) / h
        self.assertAlmostEqual(left_deriv, -1.0, places=4)
        self.assertAlmostEqual(right_deriv_cv1, 0.0, places=6)
        # CV=1.0에서 꼬리(u >= u_cut)는 상수이지만, 내부(u < u_cut)는 경험 KM 적분이므로 MRL(0) > r_cut
        self.assertGreater(float(km.predict_mrl(np.array([0.0]))[0]), r_cut)

        # 2) CV=2.0 꼬리에서 tanh 포화가 꺾임 없이 매끄럽게(C^1 이상) 연결되는지 검증
        km.cv_eff = 2.0
        slope_dfr = 0.65 * (1.0 / (1.0 + 0.25))  # 0.52
        u_sat_old = u_cut + 4.0 / slope_dfr
        d_left = float((km.predict_mrl(np.array([u_sat_old]))[0] - km.predict_mrl(np.array([u_sat_old - h]))[0]) / h)
        d_right = float((km.predict_mrl(np.array([u_sat_old + h]))[0] - km.predict_mrl(np.array([u_sat_old]))[0]) / h)
        self.assertLess(abs(d_left - d_right), 1e-4)

        # 3) CV=1.0 주변에서 CV 방향 좌·우 1차 도함수 모두 0으로 연속 수렴(C^1 평활성) 검증
        u_probe = np.array([u_cut + 2.0])
        eps_cv = 1e-5
        km.cv_eff = 1.0
        m_zero = float(km.predict_mrl(u_probe)[0])
        km.cv_eff = 1.0 - eps_cv
        m_minus = float(km.predict_mrl(u_probe)[0])
        km.cv_eff = 1.0 + eps_cv
        m_plus = float(km.predict_mrl(u_probe)[0])
        self.assertLess(abs(m_zero - m_minus) / eps_cv, 5e-4)
        self.assertLess(abs(m_plus - m_zero) / eps_cv, 5e-4)

        # 4) 식별 불가 분위수 반례: 100개 동일 코호트 중 12개만 완료(1..12), 88개는 나이 100에서 실행 중
        st_cens = ClusterState(cfg)
        st_cens.now = 100.0
        for i in range(1, 13):
            st_cens.recent_completions.append((1.0, float(i)))
            st_cens.recent_cohort.append([0.0, float(i), 1.0])
        for _ in range(88):
            st_cens.recent_cohort.append([0.0, None, 1.0])
        km_c = EmpiricalSurvivalModel(refresh_interval=0)
        km_c.update(st_cens, force=True)
        self.assertTrue(km_c.quantile_identifiable["q25"] is False)
        self.assertTrue(km_c.quantile_identifiable["q50"] is False)
        self.assertTrue(km_c.quantile_identifiable["q75"] is False)
        # 완료 표본 백분위수(q50=6.5)로 퇴행하지 않고 관측 하한(100.0) 이상을 유지해야 함
        self.assertGreaterEqual(km_c.q50 * km_c.ref_scale, 100.0)
        self.assertGreaterEqual(km_c.q75 * km_c.ref_scale, km_c.q50 * km_c.ref_scale)

    def test_hierarchical_cluster_bootstrap_and_summary(self) -> None:
        from stats_utils import cluster_aware_summary_stats, hierarchical_cluster_bootstrap_ci

        # 클러스터 내 상관 존재 시(클러스터별 효과 편차 큼) 계층 CI가 단조롭게 잘 계산되고 비정상 입력 거부
        b_groups = [np.array([10.0, 10.2, 9.8]), np.array([20.0, 20.1, 19.9]), np.array([15.0, 15.2, 14.8]), np.array([12.0, 12.1, 11.9])]
        c_groups = [np.array([9.0, 9.2, 8.8]), np.array([18.5, 18.6, 18.4]), np.array([14.8, 14.9, 14.7]), np.array([11.5, 11.6, 11.4])]
        res = cluster_aware_summary_stats(b_groups, c_groups, n_boot=500, seed=7)
        self.assertEqual(res["n_clusters"], 4)
        self.assertEqual(res["n_windows"], 12)
        w_stats = res["window_level_with_hier_ci"]
        self.assertLess(w_stats["hier_boot_ci95_low"], w_stats["diff_mean"])
        self.assertGreater(w_stats["hier_boot_ci95_high"], w_stats["diff_mean"])

        with self.assertRaises(ValueError):
            hierarchical_cluster_bootstrap_ci([])
        with self.assertRaises(ValueError):
            hierarchical_cluster_bootstrap_ci([np.array([1.0, float("nan")])])

    def test_k8s_extender_no_double_deduction_native_sidecar_cohort_and_cache(self) -> None:
        from k8s_extender.adapter import OmniLindyK8sAdapter, parse_pod_resources

        # 1) 이중 차감 방지: 동일 파드를 노드 어노테이션과 runningPods 양쪽에 동시 투입
        adp = OmniLindyK8sAdapter(engine="omni_mrl", node_cache_ttl_sec=10.0)
        node = {
            "metadata": {
                "name": "n-dup",
                "resourceVersion": "rv-101",
                "annotations": {
                    "omni-lindy.io/running-pods": '[{"uid": "pod-a", "name": "a", "startTime": 50.0, "gpu": 4.0, "cpu": "16", "ram": "32Gi"}]',
                },
            },
            "status": {"allocatable": {"nvidia.com/gpu": "8", "cpu": "64", "memory": "512Gi"}},
        }
        rp_dup = {
            "metadata": {"uid": "pod-a", "name": "a", "namespace": "default"},
            "spec": {"nodeName": "n-dup", "containers": [{"resources": {"requests": {"nvidia.com/gpu": "4", "cpu": "16", "memory": "32Gi"}}}]},
            "status": {"startTime": 50.0},
        }
        rp_extra = {
            "metadata": {"uid": "pod-b", "name": "b", "namespace": "default"},
            "spec": {"nodeName": "n-dup", "containers": [{"resources": {"requests": {"nvidia.com/gpu": "1", "cpu": "4", "memory": "8Gi"}}}]},
            "status": {"startTime": 70.0},
        }
        st1, _ = adp.build_cluster_state([node], running_pods=[rp_dup], now_unix=100.0)
        # 8 GPU / 64 CPU / 512 Gi - 4 GPU / 16 CPU / 32 Gi = 4 / 48 / 480
        self.assertAlmostEqual(float(np.sum(st1.gpu_slots[0])), 4.0, places=9)
        self.assertAlmostEqual(float(st1.cpu_free[0]), 48.0, places=9)
        self.assertAlmostEqual(float(st1.ram_free[0]), 480.0, places=9)

        # 두 번째 비중복 파드(pod-b) 추가 시에는 3 / 44 / 472로 정상 추가 차감
        st2, _ = adp.build_cluster_state([node], running_pods=[rp_dup, rp_extra], now_unix=100.0)
        self.assertAlmostEqual(float(np.sum(st2.gpu_slots[0])), 3.0, places=9)
        self.assertAlmostEqual(float(st2.cpu_free[0]), 44.0, places=9)
        self.assertAlmostEqual(float(st2.ram_free[0]), 472.0, places=9)

        # 2) 네이티브 사이드카(restartPolicy: Always) 동시 실행 자원 합산 검증
        sidecar_pod = {
            "metadata": {"name": "sc-pod"},
            "spec": {
                "initContainers": [
                    {"name": "sidecar", "restartPolicy": "Always", "resources": {"requests": {"nvidia.com/gpu": "1", "cpu": "2", "memory": "4Gi"}}},
                ],
                "containers": [
                    {"name": "app", "resources": {"requests": {"nvidia.com/gpu": "1", "cpu": "4", "memory": "8Gi"}}},
                ],
            },
        }
        sg, sc, sr, _, _ = parse_pod_resources(sidecar_pod)
        self.assertAlmostEqual(sg, 2.0, places=9)
        self.assertAlmostEqual(sc, 6.0, places=9)
        self.assertAlmostEqual(sr, 12.0, places=9)

        # 3) Pod UID 기반 코호트 KM 통일 + NodeNames 캐시 TTL / resourceVersion 신선도 검증
        req_pod = TestK8sExtenderPrototype._make_pod("req", gpu=1.0, cpu="4", mem="8Gi")
        # 12건 IFR 완료 + 실행 중 파드 코호트 연동 확인
        for k in range(15):
            adp.record_completions({"uid": f"done-{k}", "gpu": 1.0, "duration": 10.0 + 0.05 * (k % 3), "nowUnix": 100.0})
        adp.record_completions({"uid": "pod-a", "gpu": 4.0, "duration": 10.2, "nowUnix": 100.0})
        # first call populates node cache at nowUnix=100.0
        prio = adp.prioritize_nodes({"Pod": req_pod, "Nodes": {"items": [node]}, "runningPods": [rp_extra], "nowUnix": 100.0})
        self.assertTrue(all(0 <= item["score"] <= 10 for item in prio))
        metrics = adp.get_metrics()
        self.assertGreater(metrics["cohort_size"], 15)
        self.assertGreater(metrics["cohort_completed"], 12)
        self.assertGreaterEqual(metrics["cohort_censored"], 1)

        # TTL 내 + RV 일치 시 NodeNames 캐시 호출 성공
        prio_cached = adp.prioritize_nodes(
            {"Pod": req_pod, "NodeNames": ["n-dup"], "NodeResourceVersions": {"n-dup": "rv-101"}, "nowUnix": 105.0}
        )
        self.assertEqual(len(prio_cached), 1)
        # RV 불일치 또는 TTL 만료 시 ValueError
        with self.assertRaises(ValueError):
            adp.prioritize_nodes(
                {"Pod": req_pod, "NodeNames": ["n-dup"], "NodeResourceVersions": {"n-dup": "rv-999"}, "nowUnix": 105.0}
            )
        with self.assertRaises(ValueError):
            adp.prioritize_nodes({"Pod": req_pod, "NodeNames": ["n-dup"], "nowUnix": 125.0})

    def test_alibaba_csv_ground_truth_node_and_gpu_totals(self) -> None:
        import csv
        from pathlib import Path

        node_csv = Path(__file__).parent / "data" / "alibaba_2023" / "openb_node_list_gpu_node.csv"
        if not node_csv.exists():
            self.skipTest("Alibaba node CSV 파일이 존재하지 않습니다.")
        with node_csv.open("rt", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        gpu_nodes = [r for r in rows if int(float(r["gpu"])) > 0]
        total_gpus = sum(int(float(r["gpu"])) for r in gpu_nodes)
        self.assertEqual(len(gpu_nodes), 1213)
        self.assertEqual(total_gpus, 6212)

    def test_k8s_extender_stale_snapshot_after_completion_and_concurrent_http(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        import json
        import threading
        import urllib.request
        from k8s_extender.adapter import OmniLindyK8sAdapter
        from k8s_extender.server import create_extender_server

        adp = OmniLindyK8sAdapter(engine="omni_mrl")
        node_dup_ann = {
            "metadata": {
                "name": "n-stale",
                "annotations": {
                    # 동일 어노테이션 내 중복 uid 항목도 1회만 차감되어야 함
                    "omni-lindy.io/running-pods": (
                        '[{"uid": "u-life-1", "namespace": "ml", "name": "job-1", "startTime": 50.0, "gpu": 2.0, "cpu": "8", "ram": "16Gi"},'
                        ' {"uid": "u-life-1", "namespace": "ml", "name": "job-1", "startTime": 50.0, "gpu": 2.0, "cpu": "8", "ram": "16Gi"}]'
                    ),
                },
            },
            "status": {"allocatable": {"nvidia.com/gpu": "8", "cpu": "64", "memory": "512Gi"}},
        }
        req_pod = TestK8sExtenderPrototype._make_pod("req-stale", gpu=1.0, cpu="4", mem="8Gi")

        # 1) 실행 중 파드 관측 -> cohort_size=1, cohort_censored=1
        st_init, _ = adp.build_cluster_state([node_dup_ann], now_unix=100.0)
        self.assertAlmostEqual(float(np.sum(st_init.gpu_slots[0])), 6.0, places=9)
        m1 = adp.get_metrics()
        self.assertEqual(m1["cohort_size"], 1)
        self.assertEqual(m1["cohort_censored"], 1)

        # 2) uid 없이 (namespace, name)만으로 완료 이벤트 수신 -> 기존 코호트 항목 종료 전환
        adp.record_completions({"namespace": "ml", "name": "job-1", "gpu": 2.0, "duration": 55.0, "nowUnix": 105.0})
        m2 = adp.get_metrics()
        self.assertEqual(m2["cohort_size"], 1)
        self.assertEqual(m2["cohort_completed"], 1)
        self.assertEqual(m2["cohort_censored"], 0)

        # 3) 지연 스냅샷(Informer 캐시 지연)에 동일 파드가 남아 재유입되어도 중복 코호트 생성 금지
        adp.prioritize_nodes({"Pod": req_pod, "Nodes": {"items": [node_dup_ann]}, "nowUnix": 106.0})
        m3 = adp.get_metrics()
        self.assertEqual(m3["cohort_size"], 1)
        self.assertEqual(m3["cohort_completed"], 1)
        self.assertEqual(m3["cohort_censored"], 0)

        # 4) 빈 NodeNames: [] 요청은 빈 결과 반환
        self.assertEqual(adp.prioritize_nodes({"Pod": req_pod, "NodeNames": [], "nowUnix": 106.0}), [])

        # 5) 멀티스레드 HTTP 동시 요청(/filter, /prioritize, /events/pod_completed, /metrics) 안전성 검증
        server = create_extender_server("127.0.0.1", 0, quiet=True, engine="omni_mrl")
        port = server.server_address[1]
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            payload_prio = json.dumps({"Pod": req_pod, "Nodes": {"items": [node_dup_ann]}, "nowUnix": 110.0}).encode("utf-8")

            def _worker(idx: int) -> int:
                if idx % 3 == 0:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/events/pod_completed",
                        data=json.dumps({"uid": f"conc-{idx}", "gpu": 1.0, "duration": 10.0, "nowUnix": 110.0}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                elif idx % 3 == 1:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/filter",
                        data=payload_prio,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                else:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/prioritize",
                        data=payload_prio,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                with opener.open(req, timeout=5.0) as resp:
                    return resp.status

            with ThreadPoolExecutor(max_workers=6) as ex:
                statuses = list(ex.map(_worker, range(24)))
            self.assertEqual(statuses, [200] * 24)
        finally:
            server.shutdown()
            server.server_close()
            th.join(timeout=2.0)

    def test_phase11_report_integrity_and_disjoint_lockbox_seeds(self) -> None:
        import hashlib
        import json
        from pathlib import Path
        from omni_mrl import OmniMRLConfig

        root = Path(__file__).parent
        p11_path = root / "phase11_report.json"
        if not p11_path.exists():
            self.skipTest("phase11_report.json 파일이 존재하지 않습니다.")

        p11 = json.loads(p11_path.read_text(encoding="utf-8"))
        proto = p11["protocol"]

        # 1) Lockbox 30-시드는 Dev/Fresh(Phase 9/10)/초기 [101..3030] 시드와 완전 비중첩이어야 함
        lockbox = set(proto["lockbox_seeds_30"])
        dev = set(proto["dev_seeds_10"])
        fresh_p9_10 = {10007 + 97 * i for i in range(30)}
        legacy_p5_8 = {101 * (i + 1) for i in range(30)}
        self.assertEqual(len(lockbox), 30)
        self.assertTrue(lockbox.isdisjoint(dev))
        self.assertTrue(lockbox.isdisjoint(fresh_p9_10))
        self.assertTrue(lockbox.isdisjoint(legacy_p5_8))

        # 2) 핵심 시뮬레이션/평가/K8s/데이터 파일 SHA-256 해시 무결성 및 Omni-MRL 파라미터 동결 검증
        for rel, stored_hash in proto["artifact_sha256"].items():
            fp = root / rel
            self.assertTrue(fp.exists(), f"해시 대상 파일 누락: {rel}")
            cur_hash = hashlib.sha256(fp.read_bytes()).hexdigest()
            self.assertEqual(cur_hash, stored_hash, f"Phase 11.1 이후 코어 파일 변조 감지: {rel}")

        self.assertGreaterEqual(len(proto["artifact_sha256"]), 16)
        hel_s = p11["suite_3_real_traces"]["helios_multinode_20"]["schedulers"]
        low_s = p11["suite_3_real_traces"]["helios_low_cv_real_11"]["schedulers"]
        self.assertIn("Google-LAVA-BPack", hel_s)
        self.assertIn("Google-LAVA-BPack", low_s)
        first_comp = next(iter(p11["suite_3_real_traces"]["helios_multinode_20"]["comparisons"].values()))
        self.assertIn("cluster_p_value_wilcoxon_holm", first_comp["cluster_level_stats"])
        self.assertIn("window_p_value_wilcoxon_holm", first_comp["window_level_with_hier_ci"])

        frozen_cfg = OmniMRLConfig()
        self.assertAlmostEqual(frozen_cfg.w_nilas, 1.45, places=12)
        self.assertAlmostEqual(frozen_cfg.pristine_ifr_relax, 0.35, places=12)
        self.assertAlmostEqual(frozen_cfg.recycling_base_weight, 1.0, places=12)

    def test_mrl_ifr_tail_valid_nonnegative_hazard_rate(self) -> None:
        """극단적인 정상상태 스냅샷에서도 MRL 꼬리 미분 R'(u) >= -1 (위험률 h(u) >= 0)을 만족한다."""
        from bigtech_baselines import EmpiricalSurvivalModel
        from simulator import ClusterConfig, ClusterState, Job

        cfg = ClusterConfig(num_nodes=128, gpus_per_node=8, cpu_per_node=64.0, ram_per_node=512.0)
        st = ClusterState(cfg)
        st.now = 100.0
        for i in range(12):
            j = Job(i, float(i), 1.0 + 0.001 * i, 1.0, 2.0, 8.0)
            st.now = float(i)
            alloc = st.allocate(0, j)
            st.now = float(i) + j.duration
            st.release(0, j, alloc)
        # 나이 100 작업 1개 + 극단적으로 젊은 작업 1,000개 혼재 (cv_eff << 1, 큰 r_cut 유발)
        st.now = 0.0
        j_old = Job(9000, 0.0, 200.0, 1.0, 1.0, 1.0)
        st.allocate(1, j_old)
        st.now = 99.9
        for k in range(1000):
            node_idx = 2 + (k // 8)
            jk = Job(10000 + k, 99.9, 1.0, 1.0, 1.0, 1.0)
            st.allocate(node_idx, jk)
        st.now = 100.0

        surv = EmpiricalSurvivalModel(refresh_interval=1)
        surv.update(st, force=True)
        u_cut = surv._km_u_cut
        us = np.linspace(u_cut + 1e-4, u_cut + 20.0, 400)
        du = 1e-5
        m_plus = surv.predict_mrl(us + du)
        m_minus = surv.predict_mrl(us - du)
        deriv = (m_plus - m_minus) / (2.0 * du)
        self.assertGreaterEqual(float(np.min(deriv)), -1.0 - 1e-4)

    def test_k8s_extender_duplicate_completion_events_idempotent(self) -> None:
        """동일 UID 완료 이벤트를 여러 번 전송해도 recent_completions·코호트·gamma_ifr이 중복 오염되지 않는다."""
        from k8s_extender.adapter import OmniLindyK8sAdapter

        adp = OmniLindyK8sAdapter(engine="omni_mrl")
        for k in range(15):
            r = adp.record_completions({"uid": f"uniq-{k}", "gpu": 1.0, "duration": 2.0, "nowUnix": 50.0 + k})
            self.assertEqual(r["added"], 1)
        m_before = adp.get_metrics()

        # 동일 UID 재전송 40회 -> 모두 멱등 무시되어 카운터와 게이트 불변
        for rep in range(40):
            r_dup = adp.record_completions({"uid": "uniq-0", "gpu": 1.0, "duration": 2.0, "nowUnix": 100.0 + rep})
            self.assertEqual(r_dup["added"], 0)
        m_after = adp.get_metrics()
        self.assertEqual(m_after["completions_buffered"], m_before["completions_buffered"])
        self.assertEqual(m_after["cohort_size"], m_before["cohort_size"])
        self.assertEqual(m_after["completion_events"], m_before["completion_events"])
        self.assertAlmostEqual(m_after["gamma_ifr"], m_before["gamma_ifr"], places=12)

    def test_k8s_extender_same_name_diff_uid_memory_case_and_pod_resources(self) -> None:
        """동명·상이 UID 비병합, 메모리 대소문자 구분(400m vs 400M), K8s v1.34 Pod-level 자원을 검증한다."""
        from k8s_extender.adapter import OmniLindyK8sAdapter, parse_k8s_memory_gi, parse_pod_resources

        # 1) 대소문자 구분 메모리 파서: 400m(0.4 bytes) vs 400M(400 MB) -> 1e9배 차이
        m_milli = parse_k8s_memory_gi("400m")
        m_mega = parse_k8s_memory_gi("400M")
        self.assertAlmostEqual(m_mega / m_milli, 1e9, places=3)

        # 2) Pod-level spec.resources 지원 (v1.34)
        pod_lvl = {
            "metadata": {"name": "pod-lvl"},
            "spec": {
                "resources": {"requests": {"cpu": "10", "memory": "20Gi", "nvidia.com/gpu": "2"}},
                "containers": [{"name": "c", "resources": {"requests": {"cpu": "1", "memory": "1Gi"}}}],
            },
        }
        g, c, r, _, _ = parse_pod_resources(pod_lvl)
        self.assertAlmostEqual(g, 2.0, places=9)
        self.assertAlmostEqual(c, 10.0, places=9)
        self.assertAlmostEqual(r, 20.0, places=9)

        # 3) 동명(default/same)이지만 다른 UID의 파드 완료·자원 분리 검증
        adp = OmniLindyK8sAdapter(engine="omni_mrl")
        adp.record_completions({"uid": "uid-old", "namespace": "default", "name": "same", "gpu": 1.0, "duration": 10.0, "startTime": 100.0})
        node_dup = {
            "metadata": {
                "name": "n0",
                "annotations": {"omni-lindy.io/running-pods": '[{"uid": "uid-new", "namespace": "default", "name": "same", "startTime": 120.0, "gpu": 4.0}]'},
            },
            "status": {"allocatable": {"nvidia.com/gpu": "8", "cpu": "64", "memory": "512Gi"}},
        }
        st, _ = adp.build_cluster_state([node_dup], now_unix=125.0)
        self.assertAlmostEqual(float(np.sum(st.gpu_slots[0])), 4.0, places=9)
        adp.record_completions({"uid": "uid-new", "namespace": "default", "name": "same", "gpu": 4.0, "duration": 20.0, "startTime": 120.0})
        m = adp.get_metrics()
        self.assertEqual(m["cohort_size"], 2)
        self.assertEqual(m["cohort_completed"], 2)

    def test_empirical_regime_audit_integrity(self) -> None:
        """Alibaba GPU-v2026(4,052만 건) 포함 4개 트레이스 감사 JSON 핵심 수치를 검증한다."""
        import json
        from pathlib import Path

        audit_path = Path(__file__).parent / "empirical_regime_audit.json"
        if not audit_path.exists():
            self.skipTest("empirical_regime_audit.json 파일이 존재하지 않습니다.")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["distributions"]["alibaba_2026"]["global"]["n"], 40522321)
        self.assertEqual(audit["disjoint_rule"]["total_valid_disjoint_windows"], 982)
        for row in audit["distributions"]["alibaba_2026"]["by_job_type"]:
            self.assertGreater(row["cv"], 1.5)
            self.assertGreater(row["mrl_q75_over_mrl0"], 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
