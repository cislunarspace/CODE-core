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


class TestEphemerisPlanar:
    """ephemeris_planar 动力学（#498，ADR 0034 平面全星历）。

    参数校验与入口契约不依赖内核；求解正确性（粗细模型回归、零 cspice）
    需要 SPICE 内核与星历缓存，标 ``spice`` 在内核缺失时跳过。
    """

    MU_EARTH = 398600.435436
    MU_MOON = 4902.800066
    MU_SUN = 1.32712440018e11

    def _params(self, **overrides):
        params = {
            "mu_earth": self.MU_EARTH,
            "mu_moon": self.MU_MOON,
            "mu_sun": self.MU_SUN,
            "et0": 0.0,
            "thrust": 1.0,
            "isp": 300.0,
            "g0": 9.80665,
            "fuel_weight": 0.1,
            "mass_mode": 0,
            "fixed_mass": 1000.0,
        }
        params.update(overrides)
        return params

    def _call(self, params, dim=None, t0=0.0, tf=1.0):
        dim = dim or (5 if params.get("mass_mode") == 1 else 4)
        shape, _ = _grid([0.5] + [-1.0] * (dim - 1), [1.5] + [1.0] * (dim - 1), 3)
        terminal = [0.0] * (3**dim)
        return integrators.solve_hjb_py(
            terminal,
            [0.5] + [-1.0] * (dim - 1),
            [1.5] + [1.0] * (dim - 1),
            shape,
            t0,
            tf,
            "ephemeris_planar",
            params,
            0.5,
            tf / 4,
        )

    def test_missing_param_rejected(self):
        params = self._params()
        del params["mu_moon"]
        with pytest.raises(ValueError, match="mu_moon"):
            self._call(params)

    def test_unknown_param_rejected(self):
        with pytest.raises(ValueError, match="不接受参数"):
            self._call(self._params(extra=1.0))

    def test_bad_mass_mode_rejected(self):
        with pytest.raises(ValueError, match="mass_mode"):
            self._call(self._params(mass_mode=2))

    def test_4d_requires_fixed_mass(self):
        params = self._params()
        del params["fixed_mass"]
        with pytest.raises(ValueError, match="fixed_mass"):
            self._call(params)

    def test_5d_rejects_fixed_mass(self):
        with pytest.raises(ValueError, match="不接受参数"):
            self._call(self._params(mass_mode=1, fixed_mass=1000.0))

    def test_bad_thrust_rejected(self):
        with pytest.raises(ValueError, match="thrust"):
            self._call(self._params(thrust=-1.0))

    def test_dim_mismatch_rejected(self):
        """mass_mode 与网格维数不一致时报维数错误。"""
        with pytest.raises(ValueError, match="维数"):
            self._call(self._params(), dim=5)

    def test_requires_enabled_cache(self):
        """未启用星历缓存时入口直接拒绝（热循环硬失败提前）。"""
        integrators.disable_ephem_cache()
        try:
            with pytest.raises(ValueError, match="enable_ephem_cache"):
                self._call(self._params())
        finally:
            integrators.disable_ephem_cache()


@pytest.mark.spice
class TestEphemerisPlanarSolve:
    """ephemeris_planar 求解（#498 验收 c/d）：零 cspice 与粗细模型回归。"""

    MU_EARTH = 398600.435436
    MU_MOON = 4902.800066
    MU_SUN = 1.32712440018e11

    def _moon_scale(self, spice):
        """et0 处月地距 L 与会合角速度 ω，用于与 CR3BP 无量纲对齐。"""
        et0 = spice.utc_to_et("2025-06-21T11:00:06")
        state, _ = integrators.spice_spkezr("MOON", et0, "J2000", "NONE", "EARTH")
        r = np.asarray(state[:3])
        v = np.asarray(state[3:])
        length = float(np.linalg.norm(r))
        omega = float(np.linalg.norm(np.cross(r, v))) / length**2
        return et0, length, omega

    def _solve_ephemeris(self, spice, tf_nondim=0.5, n=9):
        et0, length, omega = self._moon_scale(spice)
        tf_s = tf_nondim / omega
        # 与 CR3BP 基准同物理量级：a_nd = T/(1000·m0)（km/s²）↔ 无量纲 0.5。
        thrust = 0.5 * omega**2 * 1000.0 * 1000.0
        params = {
            "mu_earth": self.MU_EARTH,
            "mu_moon": self.MU_MOON,
            "mu_sun": self.MU_SUN,
            "et0": et0,
            "thrust": thrust,
            "isp": 300.0,
            "g0": 9.80665,
            "fuel_weight": 0.1 * omega,
            "mass_mode": 0,
            "fixed_mass": 1000.0,
        }
        minimum = [0.5, -0.5, -omega, -omega]
        maximum = [1.5, 0.5, omega, omega]
        shape, axes = _grid(minimum, maximum, n)
        # 终端代价与 CR3BP 基准同一物理函数：速度维 u/ω 换算回无量纲
        # （球在混合量纲坐标下不自洽，半径 0.2 会吞掉整条速度轴）。
        axes_nondim = [a / s for a, s in zip(axes, [1.0, 1.0, omega, omega], strict=True)]
        terminal = _terminal_ball(axes_nondim, np.array([1.0, 0.0, 0.0, 0.0]), 0.2)
        ffi_before = integrators.ephem_ffi_call_count()
        raw = integrators.solve_hjb_py(
            terminal,
            minimum,
            maximum,
            shape,
            0.0,
            tf_s,
            "ephemeris_planar",
            params,
            0.5,
            tf_s / 100,
        )
        ffi_after = integrators.ephem_ffi_call_count()
        return raw, shape, axes, (ffi_before, ffi_after), omega

    def _solve_cr3bp_baseline(self, tf_nondim=0.5, n=9):
        minimum = [0.5, -0.5, -1.0, -1.0]
        maximum = [1.5, 0.5, 1.0, 1.0]
        shape, axes = _grid(minimum, maximum, n)
        terminal = _terminal_ball(axes, np.array([1.0, 0.0, 0.0, 0.0]), 0.2)
        return integrators.solve_hjb_py(
            terminal,
            minimum,
            maximum,
            shape,
            0.0,
            tf_nondim,
            "cr3bp_synodic",
            {"mu": MU_EARTH_MOON, "max_accel": 0.5, "fuel_weight": 0.1},
            0.5,
            tf_nondim / 100,
        ), shape

    def test_zero_cspice_and_cr3bp_regression(self, spice_kernel_path):
        from e2m2e.data.kernels.manager import SPICEManager
        from e2m2e.integrators import disable_ephem_cache, enable_ephem_cache

        spice = SPICEManager()
        spice.load_kernel(spice_kernel_path)
        try:
            et0, _, omega = self._moon_scale(spice)
            tf_s = 0.5 / omega
            enable_ephem_cache(
                [("MOON", "EARTH"), ("SUN", "EARTH")], [], et0, et0 + tf_s + 1.0, 3600.0
            )
            try:
                raw, shape, _, ffi, _ = self._solve_ephemeris(spice)
            finally:
                disable_ephem_cache()
        finally:
            spice.unload_kernel(spice_kernel_path)

        # 验收 d：求解全程零 cspice FFI（纯缓存查表）。
        assert ffi[1] == ffi[0]

        values = np.asarray(raw["values"]).reshape((-1, *shape))
        assert np.isfinite(values).all()
        times = np.asarray(raw["times"])
        # t=0 处的微小偏差是积分器终止判据（100·EPS·tf 相对余量）的正常行为。
        assert times[0] == pytest.approx(0.0, abs=1e-6 * tf_s)
        assert times[-1] == pytest.approx(tf_s)

        # 验收 c：粗细模型回归——星历版与 CR3BP 版值函数量级与等值面
        # 结构一致。剩余差异是物理性的（真实月距 370000 km vs 标称
        # 384400 km、瞬时 ω vs 标称 ω、太阳第三体），随窗长增长；实现
        # 错误由退化对拍（Rust）与力一致性（逐点）锤死，不靠此断言。
        raw_c, shape_c = self._solve_cr3bp_baseline()
        v_e = values[0].ravel()
        v_c = np.asarray(raw_c["values"]).reshape((-1, *shape_c))[0].ravel()
        # 剔除月球邻域节点：会合系把月球钉在 (1, 0)，网格必命中；两模型
        # 在天体中心均无物理意义（CR3BP 截断垃圾 / 星历版取物理极限）。
        x_ax, y_ax = np.meshgrid(raw["axes"][0], raw["axes"][1], indexing="ij")
        dist = np.broadcast_to(np.hypot(x_ax - 1.0, y_ax)[..., None, None], tuple(shape)).ravel()
        keep = dist > 0.1
        diff = np.abs(v_e[keep] - v_c[keep])
        scale = np.abs(v_c[keep]).max()
        assert diff.max() < 0.6 * scale, f"值函数最大偏差 {diff.max():.4f} 超过 0.6×{scale:.4f}"
        # 量级：正负端点相对差在 15% 内（真实星历与标称参数的物理差异水平）。
        assert v_e[keep].max() == pytest.approx(v_c[keep].max(), rel=0.15)
        assert v_e[keep].min() == pytest.approx(v_c[keep].min(), abs=0.1)
        corr = np.corrcoef(v_e[keep], v_c[keep])[0, 1]
        assert corr > 0.96, f"等值面结构相关性 {corr:.5f} 过低"
