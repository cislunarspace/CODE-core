"""HJB 结构网格求解绑定的契约测试（issue #497）。

覆盖 solve_hjb_py 通用入口与 solve_planar_lowthrust_hjb_py 兼容包装：
返回结构、时间方向、值函数的物理单调性与耗散收敛、参数校验。
动力学正确性（向量场与 propagate_cr3bp_py 对拍、Lagrange 点、
轨道周期）在 Rust 侧 e2m2e-hjb-dynamics 的测试中验证，此处不重复。
"""

import numpy as np
import pytest

from e2m2e import integrators

pytestmark = pytest.mark.integrator

MU_EARTH_MOON = 0.01215


def _grid(minimum, maximum, n):
    shape = [n] * len(minimum)
    axes = [
        np.linspace(lo, hi, k, endpoint=False) + (hi - lo) / (2 * k)
        for lo, hi, k in zip(minimum, maximum, shape, strict=True)
    ]
    return shape, axes


def _terminal_ball(axes, center, radius):
    states = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return (np.linalg.norm(states - np.asarray(center), axis=-1) - radius).ravel().tolist()


def _solve_double_integrator(n):
    """双积分器到原点小球的 HJB 求解（geo-nrho dp_lowdim 的量级）。"""
    shape, axes = _grid([-1.0] * 4, [1.0] * 4, n)
    terminal = _terminal_ball(axes, np.zeros(4), 0.2)
    raw = integrators.solve_planar_lowthrust_hjb_py(
        terminal, [-1.0] * 4, [1.0] * 4, shape, 0.0, 0.2, (0.0, 0.0), 0.5, 0.1, 0.5, 1.0
    )
    return raw, shape, axes


class TestDoubleIntegratorSolve:
    def test_result_structure_and_time_direction(self):
        raw, shape, _ = _solve_double_integrator(9)
        times = np.asarray(raw["times"])
        values = np.asarray(raw["values"]).reshape((-1, *shape))
        assert times.ndim == 1 and times.size == values.shape[0]
        assert np.all(np.diff(times) >= 0)
        assert times[0] == pytest.approx(0.0) and times[-1] == pytest.approx(0.2)
        assert raw["steps"] >= times.size - 1
        assert len(raw["axes"]) == 4
        assert np.isfinite(values).all()

    def test_terminal_frame_equals_terminal_cost(self):
        raw, shape, axes = _solve_double_integrator(9)
        values = np.asarray(raw["values"]).reshape((-1, *shape))
        terminal = np.asarray(_terminal_ball(axes, np.zeros(4), 0.2)).reshape(shape)
        np.testing.assert_allclose(values[-1], terminal, rtol=0, atol=1e-15)

    def test_value_minimum_near_terminal_center(self):
        """原点已在终端集内且可零燃料停留，真值 V(origin, t0) = −0.2。

        LF 耗散使粗网格偏离真值，断言落在合理带内（9 节点实测约 −0.10）。
        """
        raw, shape, _ = _solve_double_integrator(9)
        values = np.asarray(raw["values"]).reshape((-1, *shape))
        center = (4, 4, 4, 4)
        assert -0.2 - 1e-6 <= values[0][center] <= -0.05

    def test_dissipation_error_halves_with_refinement(self):
        """网格加密一倍，中心处耗散误差应明显下降（9 节点 0.099，21 节点 0.048）。"""
        raw9, shape9, _ = _solve_double_integrator(9)
        raw21, shape21, _ = _solve_double_integrator(21)
        v9 = np.asarray(raw9["values"]).reshape((-1, *shape9))[0][(4, 4, 4, 4)]
        v21 = np.asarray(raw21["values"]).reshape((-1, *shape21))[0][(10, 10, 10, 10)]
        err9 = abs(v9 + 0.2)
        err21 = abs(v21 + 0.2)
        assert err21 < 0.6 * err9

    def test_generic_entry_matches_compat_wrapper(self):
        """通用入口与兼容包装走同一条求解路径，输出逐位一致。"""
        shape, axes = _grid([-1.0] * 4, [1.0] * 4, 9)
        terminal = _terminal_ball(axes, np.zeros(4), 0.2)
        raw_generic = integrators.solve_hjb_py(
            terminal,
            [-1.0] * 4,
            [1.0] * 4,
            shape,
            0.0,
            0.2,
            "planar_double_integrator",
            {"drift_x": 0.0, "drift_y": 0.0, "max_accel": 0.5, "fuel_weight": 0.1},
            0.5,
            1.0,
        )
        raw_wrapper, _, _ = _solve_double_integrator(9)
        assert np.array_equal(np.asarray(raw_generic["values"]), np.asarray(raw_wrapper["values"]))
        assert raw_generic["times"] == raw_wrapper["times"]


class TestCr3bpSolve:
    def _solve(self):
        shape, axes = _grid([0.5, -0.5, -1.0, -1.0], [1.5, 0.5, 1.0, 1.0], 9)
        terminal = _terminal_ball(axes, np.array([1.0, 0.0, 0.0, 0.0]), 0.2)
        raw = integrators.solve_hjb_py(
            terminal,
            [0.5, -0.5, -1.0, -1.0],
            [1.5, 0.5, 1.0, 1.0],
            shape,
            0.0,
            0.3,
            "cr3bp_synodic",
            {"mu": MU_EARTH_MOON, "max_accel": 0.5, "fuel_weight": 0.1},
            0.5,
            1.0,
        )
        return raw, shape

    def test_solve_produces_finite_values(self):
        raw, shape = self._solve()
        times = np.asarray(raw["times"])
        values = np.asarray(raw["values"]).reshape((-1, *shape))
        assert np.isfinite(values).all()
        assert np.all(np.diff(times) >= 0)
        assert raw["steps"] > 0

    def test_snapshot_count_bounded(self):
        """快照数有上界（64），不随步数线性增长。"""
        raw, _ = self._solve()
        assert len(raw["times"]) <= 64


class TestParameterValidation:
    def _call(self, dynamics, params, shape=None):
        shape = shape or [9] * 4
        n = int(np.prod(shape))
        return integrators.solve_hjb_py(
            [0.0] * n,
            [-1.0] * len(shape),
            [1.0] * len(shape),
            shape,
            0.0,
            0.2,
            dynamics,
            params,
            0.5,
            1.0,
        )

    def test_unknown_dynamics_rejected(self):
        with pytest.raises(ValueError, match="未知动力学标识"):
            self._call("unknown", {})

    def test_missing_param_rejected(self):
        with pytest.raises(ValueError, match="缺少参数"):
            self._call("cr3bp_synodic", {"mu": MU_EARTH_MOON, "max_accel": 0.5})

    def test_unexpected_param_rejected(self):
        with pytest.raises(ValueError, match="不接受参数"):
            self._call(
                "cr3bp_synodic",
                {"mu": MU_EARTH_MOON, "max_accel": 0.5, "fuel_weight": 0.1, "omega": 1.0},
            )

    def test_invalid_param_value_rejected(self):
        with pytest.raises(ValueError, match="max_accel"):
            self._call(
                "cr3bp_synodic", {"mu": MU_EARTH_MOON, "max_accel": -0.5, "fuel_weight": 0.1}
            )
        with pytest.raises(ValueError, match="mu"):
            self._call("cr3bp_synodic", {"mu": 1.5, "max_accel": 0.5, "fuel_weight": 0.1})

    def test_wrong_dimension_rejected(self):
        with pytest.raises(ValueError, match="网格维数必须为 4"):
            self._call(
                "cr3bp_synodic",
                {"mu": MU_EARTH_MOON, "max_accel": 0.5, "fuel_weight": 0.1},
                shape=[9] * 3,
            )
