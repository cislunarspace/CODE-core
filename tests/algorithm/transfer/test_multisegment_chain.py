"""multisegment 链组装（evaluate_chain / LegKernel）的机制测试（#726）。

oracle 口径（ADR 0055）：

- ② 不变量：一致传播态（出发/到达共 conic 弧）下匹配点原始残差 ≈ 0；
- ③ 退化：conic 内核 + shift 列时窗口平移项逐位为零（时间不变动力学）；
  退化单天体星历系下逐 leg 混档与全 conic 参考一致；
- ④ 机制：归一化等式雅可比对冲量列中心差分；窗口平移列对非自治解析
  stub（仿射系统 ẋ = [v; a·sin(ωt)]，闭式状态 + STM [[I, tI],[0, I]]，
  鸭子类型直插 LegKernel）的窗口平移中心差分。
"""

from __future__ import annotations

import numpy as np
import pytest
from kernel_helpers import requires_native_symbols, requires_spice

from e2m2e.algorithm.dynamics import EphemerisDynamics, EphemerisSystem
from e2m2e.algorithm.transfer import ChainLegRequest, LegKernel, evaluate_chain
from e2m2e.integrators import propagate_kepler_py

pytestmark = [pytest.mark.orchestration, requires_native_symbols("propagate_kepler_py")]

MU = 398600.435507  # 地球 GM（km³/s²），固定值便于复算

_LEN_SCALE = 7000.0
_VEL_SCALE = 8.0


def _leo_departure() -> np.ndarray:
    """7000 km 圆轨道出发态（与 conic 档 SF 测试同型）。"""
    return np.array([7000.0, 0.0, 0.0, 0.0, float(np.sqrt(MU / 7000.0)), 0.0])


def _request(kernel, *, dv, dt, anchor_f, anchor_b, n_var, t_start=None, **extra):
    """单 leg 请求的便捷构造（锚灵敏度恒零）。"""
    return ChainLegRequest(
        kernel=kernel,
        n_segments=2,
        imp_offset=0,
        dv=dv,
        dt=dt,
        t_start=t_start,
        anchor_f=anchor_f,
        anchor_b=anchor_b,
        anchor_sens_f=np.zeros((6, n_var)),
        anchor_sens_b=np.zeros((6, n_var)),
        **extra,
    )


def _normalized(raw: np.ndarray) -> np.ndarray:
    return np.concatenate([raw[:3] / _LEN_SCALE, raw[3:] / _VEL_SCALE])


def _run(request, *, with_sens):
    return evaluate_chain(
        [request],
        n_var=request.anchor_sens_f.shape[1],
        len_scale=_LEN_SCALE,
        vel_scale=_VEL_SCALE,
        with_sens=with_sens,
    )[0]


@pytest.fixture
def eph_earth_only(spice_manager):
    """单天体退化星历系（N 体公式退化为纯二体；与星历档 SF 测试同宿主）。"""
    return EphemerisSystem(bodies=["EARTH"], spice=spice_manager, origin="EARTH")


class _SineDynamics:
    """非自治仿射动力学 stub：``ẋ = [v; a·sin(ωt)]``（a 固定向量）。

    闭式解：``v(t) = v0 + a·(cos ωt0 − cos ωt)/ω``、
    ``r(t) = r0 + v0·h + a·[h·cos ωt0/ω + (sin ωt0 − sin ωt)/ω²]``；
    STM 恒为 ``[[I, hI],[0, I]]``（加速度与状态无关）。
    ``propagate``/``equations_of_motion`` 鸭子类型兼容 LegKernel 星历档协议。
    """

    def __init__(self, a_vec: np.ndarray, omega: float) -> None:
        self.a = np.asarray(a_vec, dtype=float)
        self.omega = float(omega)

    def equations_of_motion(self, t, x):
        return np.concatenate([np.asarray(x, dtype=float)[3:], self.a * np.sin(self.omega * t)])

    def propagate(self, x0, t_span, t_eval=None, with_stm=False):
        x0 = np.asarray(x0, dtype=float)
        t0, tf = float(t_span[0]), float(t_span[1])
        h = tf - t0
        w = self.omega
        position = (
            x0[:3]
            + x0[3:] * h
            + self.a * (h * np.cos(w * t0) / w + (np.sin(w * t0) - np.sin(w * tf)) / w**2)
        )
        velocity = x0[3:] + self.a * (np.cos(w * t0) - np.cos(w * tf)) / w
        out = {"states": [np.concatenate([position, velocity])]}
        if with_stm:
            out["stm"] = [np.block([[np.eye(3), h * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])]
        return out


class TestConicChain:
    def test_consistent_propagation_residual_zero(self):
        """② 一致传播态（零冲量、共 conic 弧）下 residual_raw ≈ 0。"""
        departure = _leo_departure()
        tof = 5400.0
        arrival = np.asarray(
            propagate_kepler_py(departure.tolist(), [tof], MU)["states"][0], dtype=float
        )
        leg = _run(
            _request(
                LegKernel(MU),
                dv=np.zeros((2, 3)),
                dt=tof / 2.0,
                anchor_f=departure,
                anchor_b=arrival,
                n_var=6,
            ),
            with_sens=False,
        )
        np.testing.assert_allclose(leg.residual_raw, np.zeros(6), atol=1e-9)
        assert leg.mid_pos.shape == (2, 3)
        assert leg.r_mid.shape == (2,)
        assert leg.eq_jac is None

    def test_eq_jacobian_matches_finite_difference(self):
        """④ 冲量列解析雅可比对中心差分（atol 1e-5）。"""
        departure = _leo_departure()
        tof = 5400.0
        arrival = np.asarray(
            propagate_kepler_py(departure.tolist(), [tof], MU)["states"][0], dtype=float
        ).copy()
        arrival[3:] += np.array([0.02, -0.01, 0.0])
        x_test = np.tile(np.array([0.005, -0.003, 0.0]), 2)

        def defect(x_flat):
            request = _request(
                LegKernel(MU),
                dv=np.asarray(x_flat, dtype=float).reshape(2, 3),
                dt=tof / 2.0,
                anchor_f=departure,
                anchor_b=arrival,
                n_var=6,
            )
            return _normalized(_run(request, with_sens=False).residual_raw)

        request = _request(
            LegKernel(MU),
            dv=x_test.reshape(2, 3),
            dt=tof / 2.0,
            anchor_f=departure,
            anchor_b=arrival,
            n_var=6,
        )
        leg = _run(request, with_sens=True)
        assert leg.eq_jac is not None
        h = 1e-6
        for i in range(6):
            x_plus = x_test.copy()
            x_plus[i] += h
            x_minus = x_test.copy()
            x_minus[i] -= h
            fd = (defect(x_plus) - defect(x_minus)) / (2.0 * h)
            np.testing.assert_allclose(fd, leg.eq_jac[:, i], atol=1e-5)

    def test_conic_shift_columns_are_bitwise_zero(self):
        """③ conic 内核 + shift 列：雅可比与不带 shift 列逐位相同（时间不变退化）。"""
        departure = _leo_departure()
        tof = 5400.0
        arrival = np.asarray(
            propagate_kepler_py(departure.tolist(), [tof], MU)["states"][0], dtype=float
        ).copy()
        arrival[3:] += np.array([0.02, -0.01, 0.0])
        dv = np.tile(np.array([0.005, -0.003, 0.0]), 2).reshape(2, 3)
        with_shift = _run(
            _request(
                LegKernel(MU),
                dv=dv,
                dt=tof / 2.0,
                anchor_f=departure,
                anchor_b=arrival,
                n_var=7,
                shift_cols_f=np.array([6]),
                shift_cols_b=np.array([6]),
            ),
            with_sens=True,
        )
        without_shift = _run(
            _request(
                LegKernel(MU),
                dv=dv,
                dt=tof / 2.0,
                anchor_f=departure,
                anchor_b=arrival,
                n_var=7,
            ),
            with_sens=True,
        )
        assert with_shift.eq_jac is not None and without_shift.eq_jac is not None
        assert np.array_equal(with_shift.eq_jac, without_shift.eq_jac)
        assert not with_shift.smid_shift.any()


class TestWindowShiftSensitivity:
    def test_shift_columns_match_finite_difference(self):
        """④ 窗口平移列对非自治解析 stub 的窗口平移中心差分（atol 1e-6）。

        雅可比第 6 列（shift 列）应等于"整个 leg 窗口统一平移 δ（锚态固定
        为数据）"时归一化匹配点残差对 δ 的导数——FD 直接平移 t_start 复算。
        """
        dyn = _SineDynamics(a_vec=np.array([2.0, -1.0, 0.5]), omega=0.05)
        kernel = LegKernel(MU, dyn=dyn)
        anchor_f = np.array([7000.0, 0.0, 0.0, 0.0, 8.0, 1.0])
        anchor_b = anchor_f + np.array([40000.0, 5000.0, 0.0, -5.0, 6.0, 0.5])
        dv = np.array([[0.01, -0.02, 0.0], [0.0, 0.01, -0.005]])
        t0, dt = 100.0, 50.0

        def residual(shift: float) -> np.ndarray:
            request = _request(
                kernel,
                dv=dv,
                dt=dt,
                anchor_f=anchor_f,
                anchor_b=anchor_b,
                n_var=7,
                t_start=t0 + shift,
                shift_cols_f=np.array([6]),
                shift_cols_b=np.array([6]),
            )
            return _normalized(_run(request, with_sens=False).residual_raw)

        request = _request(
            kernel,
            dv=dv,
            dt=dt,
            anchor_f=anchor_f,
            anchor_b=anchor_b,
            n_var=7,
            t_start=t0,
            shift_cols_f=np.array([6]),
            shift_cols_b=np.array([6]),
        )
        leg = _run(request, with_sens=True)
        assert leg.eq_jac is not None
        h = 1e-2
        fd = (residual(+h) - residual(-h)) / (2.0 * h)
        np.testing.assert_allclose(fd, leg.eq_jac[:, 6], atol=1e-6)
        # 段中点窗口平移灵敏度非零（非自治动力学下该链确实被激活）。
        assert leg.smid_shift.any()


@pytest.mark.spice
@requires_spice
@requires_native_symbols("propagate_kepler_py", "propagate_with_stm_py")
class TestMixedTierDegenerateParity:
    def test_mixed_conic_ephemeris_matches_all_conic(
        self, spice_manager, eph_earth_only, reference_et
    ):
        """③ 逐 leg 混档：leg0 conic + leg1 星历（退化单天体系）与全 conic 一致。

        退化单天体星历系的 N 体公式退化为纯二体（时间不变），混档链的残差
        与雅可比只能来自积分截断（量级远小于断言容差）。
        """
        eph_dyn = EphemerisDynamics(system=eph_earth_only)
        eph_dyn.max_step = float("inf")
        mu = float(spice_manager.get_gm("EARTH"))
        departure = np.array([7000.0, 0.0, 0.0, 0.0, float(np.sqrt(mu / 7000.0)), 0.0])
        tofs = (3000.0, 3600.0)
        node = np.asarray(
            propagate_kepler_py(departure.tolist(), [tofs[0]], mu)["states"][0], dtype=float
        )
        arrival = np.asarray(
            propagate_kepler_py(node.tolist(), [tofs[1]], mu)["states"][0], dtype=float
        )
        n_list = (2, 2)
        n_var = 12
        anchors = [(departure, node), (node, arrival)]

        def build(kernels):
            requests = []
            for leg, kernel in enumerate(kernels):
                requests.append(
                    ChainLegRequest(
                        kernel=kernel,
                        n_segments=n_list[leg],
                        imp_offset=3 * sum(n_list[:leg]),
                        dv=np.tile(np.array([0.001, -0.002, 0.0]), (n_list[leg], 1)),
                        dt=tofs[leg] / n_list[leg],
                        t_start=None if leg == 0 else reference_et,
                        anchor_f=anchors[leg][0],
                        anchor_b=anchors[leg][1],
                        anchor_sens_f=np.zeros((6, n_var)),
                        anchor_sens_b=np.zeros((6, n_var)),
                    )
                )
            return evaluate_chain(
                requests, n_var=n_var, len_scale=_LEN_SCALE, vel_scale=_VEL_SCALE, with_sens=True
            )

        mixed = build([LegKernel(mu), LegKernel(mu, dyn=eph_dyn)])
        conic = build([LegKernel(mu), LegKernel(mu)])
        for leg_m, leg_c in zip(mixed, conic, strict=True):
            np.testing.assert_allclose(leg_m.residual_raw, leg_c.residual_raw, atol=1e-6)
            assert leg_m.eq_jac is not None and leg_c.eq_jac is not None
            np.testing.assert_allclose(leg_m.eq_jac, leg_c.eq_jac, atol=1e-6)
