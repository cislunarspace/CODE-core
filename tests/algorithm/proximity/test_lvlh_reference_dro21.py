"""LVLH 参考体在 2:1 DRO 上的角动量符号回归与论文对照（#834）。

角动量符号事实用周期内 25 点采样固定。记 h1 为质心会合角动量
(r × v_syn)_z，h2 为月心会合角动量 ((r − r_moon) × v_syn)_z，h3 为
月心惯性角动量 ((r − r_moon) × (v_syn + ω × r))_z。同一 2:1 DRO 上
h1 在 t/T 约 0.21 与 0.79 过零，h2 全程为负，h3 在 t/T 约 0.28 与
0.72 过零。

论文对照：杨驰航等，航空学报 2023 年 44 卷 5 期 326563 的式(12) 与
式(13)。e3(0) = (0, 0.5554, 0, 0.8316, 0, 0) 由论文的随动导数速度
变换独立复算，本文件实测残差约 2.0e-4，可进断言。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.proximity.relative_dynamics import RelativeDynamics, TargetOrbit
from e2m2e.data.templates.enums import LVLHReference
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration

#: 论文式(13) 的切向基础解初值，单位化后的 LVLH 分量。
_PAPER_TANGENT_INITIAL = np.array([0.0, 0.5554, 0.0, 0.8316, 0.0, 0.0])

#: 周期内采样点数（含两端）。
_SAMPLES = 25


def _angular_momenta_z(states: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回质心会合、月心会合与月心惯性角动量的 z 分量。"""
    r = states[:, :3]
    v = states[:, 3:]
    r_moon = np.array([1.0 - mu, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 1.0])
    h1 = np.cross(r, v)[:, 2]
    h2 = np.cross(r - r_moon, v)[:, 2]
    h3 = np.cross(r - r_moon, v + np.cross(omega, r))[:, 2]
    return h1, h2, h3


class TestAngularMomentumSigns:
    """2:1 DRO 周期内 25 点采样的角动量符号。"""

    def test_sign_pattern(self, dro21):
        """月心会合角动量恒负，双质心与惯性口径各变号两次。"""
        orbit, dynamics = dro21
        period = float(orbit.period)
        times = np.linspace(0.0, period, _SAMPLES)
        traj = dynamics.propagate(orbit.states[0], (0.0, period), t_eval=times)
        h1, h2, h3 = _angular_momenta_z(np.asarray(traj["states"], dtype=float), dynamics.system.mu)

        assert np.all(h2 < 0.0)

        def change_indices(values: np.ndarray) -> list[int]:
            return [
                index
                for index in range(1, values.shape[0])
                if values[index - 1] * values[index] < 0.0
            ]

        h1_changes = change_indices(h1)
        h3_changes = change_indices(h3)
        assert len(h1_changes) == 2
        assert len(h3_changes) == 2
        # 25 点等分一个周期，索引 i 对应 t/T = i/24。
        assert h1_changes == [6, 19]
        assert h3_changes == [7, 18]


class TestPaperTangent:
    """论文式(13) 的切向基础解初值对照。"""

    def test_tangent_initial_condition(self, dro21_raw, modes21):
        """月心 LVLH 的切向基础解初值与论文单位向量逐分量一致。"""
        orbit, dynamics = dro21_raw
        period = float(orbit.period)
        margin = 1e-3
        times = np.linspace(-margin, period + margin, 20001)
        # propagate 的 initial_state 对应 t_span[0]，先从 t=0 反推 margin 处
        # 的状态，否则 t=0 的状态会被当成 t=-margin 的状态，整条轨迹平移。
        state_at_margin = dynamics.propagate(orbit.states[0], (0.0, -margin))["states"][-1]
        traj = dynamics.propagate(state_at_margin, (-margin, period + margin), t_eval=times)
        target = TargetOrbit(
            Orbit(
                states=np.asarray(traj["states"], dtype=float),
                times=times,
                system=dynamics.system,
            )
        )
        relative = RelativeDynamics(target, dynamics)
        rho, rho_dot = relative.to_lvlh(modes21.tangent, 0.0, reference=LVLHReference.SECONDARY)
        normalized = np.concatenate([rho, rho_dot])
        normalized = normalized / np.linalg.norm(normalized)
        np.testing.assert_allclose(normalized, _PAPER_TANGENT_INITIAL, atol=1e-3)
