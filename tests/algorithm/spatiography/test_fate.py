"""命运（fate）诊断量与八类分类器测试（Primer §7.2）。

- 事件机制复用 ``Dynamics.propagate`` 的 scipy 语义（终端定位由积分器完成）；
- 决策树真值表（纯单元，无传播）；
- 与 station_keeping 事件语义不冲突（那侧是控制时刻表，非几何事件面）。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.spatiography import primer_cr3bp_system
from e2m2e.algorithm.spatiography.fate import (
    FATE_CLASSES,
    FateDiagnostics,
    FateThresholds,
    build_cr3bp_fate_events,
    classify_fate,
    extract_cr3bp_fate,
)

pytestmark = pytest.mark.theory


def _diag(**overrides) -> FateDiagnostics:
    base = dict(
        escaped=False,
        t_escape_days=None,
        earth_reentry=False,
        t_reentry_days=None,
        moon_impact=False,
        t_impact_days=None,
        min_geocentric_km=1.0e5,
        min_selenocentric_km=1.0e5,
        moon_hill_entries=0,
        span_days=100.0,
    )
    base.update(overrides)
    return FateDiagnostics(**base)


class TestClassifierTruthTable:
    """八类决策树（§7.2 两条硬规则 + 阈值带细分）。"""

    def test_terminal_events_ignore_megno(self):
        for ybar in (1.5, 2.0, 8.0):
            assert classify_fate(_diag(earth_reentry=True), ybar)[0] == "earth_reentry"
            assert classify_fate(_diag(moon_impact=True), ybar)[0] == "moon_impact"

    def test_reentry_has_priority_over_impact(self):
        both = _diag(earth_reentry=True, moon_impact=True)
        assert classify_fate(both, 2.0)[0] == "earth_reentry"  # 登记的优先序

    def test_bounded_split_by_variation(self):
        assert classify_fate(_diag(), 2.0)[0] == "stable_quasiperiodic"
        assert classify_fate(_diag(), 2.15)[0] == "stable_quasiperiodic"
        assert classify_fate(_diag(), 4.0)[0] == "sticky_resident"
        assert classify_fate(_diag(), 2.5)[0] == "bounded_unclassified"

    def test_escape_split_by_variation(self):
        assert classify_fate(_diag(escaped=True, t_escape_days=30.0), 2.05)[0] == "orderly_escape"
        assert classify_fate(_diag(escaped=True, t_escape_days=30.0), 6.0)[0] == "chaotic_escape"
        assert classify_fate(_diag(escaped=True, t_escape_days=30.0), 2.5)[0] == (
            "escape_unclassified"
        )

    def test_thresholds_are_free_parameters(self):
        tight = FateThresholds(ybar_ordered_band=0.05, ybar_chaotic_excess=0.5)
        # 缺省带下 2.9 落在 unclassified；收紧后落 chaotic（sticky_resident）。
        assert classify_fate(_diag(), 2.9)[0] == "bounded_unclassified"
        assert classify_fate(_diag(), 2.9, tight)[0] == "sticky_resident"
        # 缺省带下 2.1 仍 ordered；收紧后落入未分辨带。
        assert classify_fate(_diag(), 2.1)[0] == "stable_quasiperiodic"
        assert classify_fate(_diag(), 2.1, tight)[0] == "bounded_unclassified"

    def test_all_classes_reachable_and_registered(self):
        outcomes = {
            classify_fate(_diag(earth_reentry=True), 2.0)[0],
            classify_fate(_diag(moon_impact=True), 2.0)[0],
            classify_fate(_diag(), 2.0)[0],
            classify_fate(_diag(), 5.0)[0],
            classify_fate(_diag(), 2.4)[0],
            classify_fate(_diag(escaped=True), 2.0)[0],
            classify_fate(_diag(escaped=True), 5.0)[0],
            classify_fate(_diag(escaped=True), 2.4)[0],
        }
        assert outcomes == set(FATE_CLASSES)

    def test_nonfinite_ybar_falls_to_unclassified(self):
        assert classify_fate(_diag(), float("nan"))[0] == "bounded_unclassified"
        assert classify_fate(_diag(escaped=True), float("inf"))[0] == "escape_unclassified"


class TestCr3bpFateEvents:
    """事件面在 CR3BP 会合系的端到端（scipy 事件 backend）。"""

    @pytest.fixture(scope="class")
    def setup(self):
        system = primer_cr3bp_system()
        dynamics = CR3BP_Dynamics(system)
        events = build_cr3bp_fate_events(system)
        return system, dynamics, events

    def test_earth_reentry_trajectory_is_terminal(self, setup):
        system, dynamics, events = setup
        rp_nd = 5000.0 / system.characteristic_length
        state = [-(system.mu) + rp_nd, 0.0, 0.0, 0.0, 9.0, 0.0]
        result = dynamics.propagate(
            state,
            (0.0, 30.0),
            t_eval=np.linspace(0.0, 30.0, 100),
            events=events,
            backend="scipy",
        )
        diag = extract_cr3bp_fate(result, system)
        assert diag.earth_reentry is True
        assert diag.t_reentry_days is not None and 0.0 < diag.t_reentry_days < 2.0
        label, status, _, _ = classify_fate(diag, 2.0)
        assert label == "earth_reentry"
        assert status.value == "converged"

    def test_moon_impact_trajectory_is_terminal(self, setup):
        system, dynamics, events = setup
        # 近月缓慢下落初态：月心距 0.005（≈1917 km，月面外 180 km）向月漂移，
        # 引力拉入撞击（月面碰撞判据，非 Hill 判据）。
        state = [1.0 - system.mu - 0.005, 0.0, 0.0, -0.01, 0.0, 0.0]
        result = dynamics.propagate(
            state,
            (0.0, 10.0),
            t_eval=np.linspace(0.0, 10.0, 200),
            events=events,
            backend="scipy",
        )
        diag = extract_cr3bp_fate(result, system)
        assert diag.moon_impact is True
        assert classify_fate(diag, 3.0)[0] == "moon_impact"

    def test_regular_bounded_trajectory_survives(self, setup):
        system, dynamics, events = setup
        # 近圆轨迹（v 略高于圆速 → r 在 ~0.088–0.1 间振荡）。
        state = [-(system.mu) + 0.1, 0.0, 0.0, 0.0, 3.2, 0.0]
        result = dynamics.propagate(
            state,
            (0.0, 40.0),
            t_eval=np.linspace(0.0, 40.0, 80),
            events=events,
            backend="scipy",
        )
        diag = extract_cr3bp_fate(result, system)
        assert diag.escaped is False
        assert diag.earth_reentry is False
        assert diag.moon_impact is False
        assert (
            0.08 * system.characteristic_length
            < diag.min_geocentric_km
            < (0.11 * system.characteristic_length)
        )
        assert classify_fate(diag, 2.01)[0] == "stable_quasiperiodic"

    def test_moon_hill_entry_counted_nonterminal(self, setup):
        system, dynamics, events = setup
        # 远点 ≈ 0.92 a☾ 的地心椭圆（近点 0.3）：扫描起角使远点几何对准月
        # Hill 界——上行事件计两次（进/出各一次），不终止、不撞月。
        state = [
            0.09045545872776034,
            -0.2819077862357725,
            0.0,
            1.8250419991183988,
            0.6764115482347226,
            0.0,
        ]
        result = dynamics.propagate(
            state,
            (0.0, 12.0),
            t_eval=np.linspace(0.0, 12.0, 600),
            events=events,
            backend="scipy",
        )
        diag = extract_cr3bp_fate(result, system)
        assert diag.min_selenocentric_km < 61364.0
        assert diag.moon_hill_entries >= 1
        assert diag.moon_impact is False

    def test_event_semantics_do_not_conflict_with_sections_convention(self, setup):
        """terminal/direction 属性与 PoincareSection.event() 同款 scipy 语义。"""
        _, _, events = setup
        for event in events[:3]:
            assert event.terminal is True
        assert events[3].terminal is False
        assert all(hasattr(e, "direction") for e in events)
