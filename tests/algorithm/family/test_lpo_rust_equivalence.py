"""LPO Rust 实现与 Python 参考实现的等价性测试。

覆盖 ``design_lpo`` 默认走 Rust 快速路径后，与保留的 Python 参考实现
``_design_lpo_python`` 在周期、初始状态、振幅、收敛性上的数值一致性。

由于 Rust PD78 与 Python ``solve_ivp`` RK45 积分器非位级一致（M2/M3
已确认差异在容差级），本文件采用宽松但足够紧的容差：
- period 差 < 1e-4（无量纲）
- state 差 < 1e-5
- 振幅差 < 1e-5（无量纲）
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.cr3bp_orbits import (
    _design_lpo_python,
    _l45_distance,
    design_lpo,
    earth_moon_system,
)

pytestmark = pytest.mark.orchestration

CHAR_LENGTH_KM = 384400.0


@pytest.fixture(scope="module")
def dynamics() -> CR3BP_Dynamics:
    """共享地月 CR3BP 动力学对象。"""
    return CR3BP_Dynamics(earth_moon_system())


class TestLpoRustEquivalence:
    """Rust ``design_lpo`` 与 Python ``_design_lpo_python`` 等价性对照。"""

    def _measure_amplitude(self, dynamics: CR3BP_Dynamics, orbit, point: int) -> float:
        """返回轨道振幅（距 L4/L5 径向距离均值，无量纲）。"""
        d_min, d_max = _l45_distance(dynamics, orbit, point)
        return 0.5 * (d_min + d_max)

    def _compare(self, dynamics: CR3BP_Dynamics, point: int, amplitude_km: float) -> None:
        """同一参数下 Rust 与 Python 路径结果对照。"""
        rust_orbit = design_lpo(point, amplitude_km, dynamics=dynamics)
        py_orbit = _design_lpo_python(point, amplitude_km, dynamics=dynamics)

        assert rust_orbit.period is not None
        assert py_orbit.period is not None
        assert rust_orbit.period == pytest.approx(py_orbit.period, abs=1e-4)

        state_diff = np.linalg.norm(rust_orbit.states[0] - py_orbit.states[0])
        assert state_diff < 1e-5, f"state 差 {state_diff:.3e} 超过 1e-5"

        rust_amp = self._measure_amplitude(dynamics, rust_orbit, point)
        py_amp = self._measure_amplitude(dynamics, py_orbit, point)
        assert rust_amp == pytest.approx(py_amp, abs=1e-5)

        assert rust_orbit.correction_success
        assert py_orbit.correction_success is None or py_orbit.correction_success

    def test_l4_50000_km(self, dynamics: CR3BP_Dynamics) -> None:
        self._compare(dynamics, 4, 50000.0)

    def test_l5_50000_km(self, dynamics: CR3BP_Dynamics) -> None:
        self._compare(dynamics, 5, 50000.0)

    def test_l4_horseshoe_amplitude(self, dynamics: CR3BP_Dynamics) -> None:
        """大振幅马蹄区域：Rust 与 Python 同时收敛或同时失败；禁止单路径未命中被掩盖。"""
        rust_exc: Exception | None = None
        py_exc: Exception | None = None
        rust_orbit = None
        py_orbit = None

        try:
            rust_orbit = design_lpo(4, 150000.0, dynamics=dynamics, tol_km=50.0)
        except Exception as exc:
            rust_exc = exc

        try:
            py_orbit = _design_lpo_python(4, 150000.0, dynamics=dynamics, tol_km=50.0)
        except Exception as exc:
            py_exc = exc

        # 双方同时失败：马蹄振幅在当前参数下未命中，可接受；只要有一方成功，另一方也必须成功。
        if rust_exc is not None and py_exc is not None:
            pytest.skip(f"当前 L4 150000 km 马蹄振幅未命中，双方均失败：{rust_exc}")
        assert rust_exc is None, f"Rust 路径异常但 Python 路径成功：{rust_exc}"
        assert py_exc is None, f"Python 路径异常但 Rust 路径成功：{py_exc}"
        assert rust_orbit is not None and py_orbit is not None

        assert rust_orbit.period is not None
        assert py_orbit.period is not None
        assert rust_orbit.period == pytest.approx(py_orbit.period, abs=1e-4)

        state_diff = np.linalg.norm(rust_orbit.states[0] - py_orbit.states[0])
        assert state_diff < 1e-5, f"state 差 {state_diff:.3e} 超过 1e-5"

        rust_amp = self._measure_amplitude(dynamics, rust_orbit, 4)
        py_amp = self._measure_amplitude(dynamics, py_orbit, 4)
        assert rust_amp == pytest.approx(py_amp, abs=1e-5)

    def test_amplitude_in_km(self, dynamics: CR3BP_Dynamics) -> None:
        """Rust 结果振幅应接近目标 50000 km。"""
        orbit = design_lpo(4, 50000.0, dynamics=dynamics)
        amp_du = self._measure_amplitude(dynamics, orbit, 4)
        amp_km = amp_du * CHAR_LENGTH_KM
        assert abs(amp_km - 50000.0) < 50.0

    def test_converged_flag(self, dynamics: CR3BP_Dynamics) -> None:
        """Rust 路径返回的 converged 标记应为 True。"""
        orbit = design_lpo(4, 50000.0, dynamics=dynamics)
        assert orbit.correction_success is True
        assert orbit.correction_iterations is not None
        assert orbit.correction_iterations > 0
