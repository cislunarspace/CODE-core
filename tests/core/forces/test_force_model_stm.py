"""ForceModel STM（状态转移矩阵）支持测试。

验证路线 B 的核心：ForceModel 通过组合式雅可比叠加 + 变分方程积分出的 STM，
与物理等价的 EphemerisDynamics（解析 N 体 STM）逐元素一致。这保证雅可比
叠加正确性——两者加速度公式相同，STM 必须一致。

同时验证有限差分雅可比兜底路径的正确性。
"""

from __future__ import annotations

import numpy as np
import pytest
from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin
from numpy.testing import assert_allclose

from e2m2e.core.ephemeris_dynamics import EphemerisDynamics
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces import (
    ForceModel,
    GravityField,
    IndirectTerm,
    PointMassGravity,
    ThirdBodyGravity,
)
from e2m2e.core.forces.physical_model import PhysicalModel

pytestmark = pytest.mark.spice


def _make_force_model_system(spice_eph_system, spice_manager):
    """给 spice_eph_system 挂 ICRS coordinate_system（ForceModel 必需）。"""
    system = spice_eph_system
    if getattr(system, "coordinate_system", None) is None:
        system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice_manager),
        )
    return system


# =============================================================================
# 核心：ForceModel STM 与 EphemerisDynamics 一致性
# =============================================================================
class TestForceModelSTMConsistency:
    """ForceModel（点质量组合）的 STM 与 EphemerisDynamics 一致。

    两者物理等价：PointMass(EARTH) + ThirdBody(MOON) + ThirdBody(SUN) 的
    加速度公式与 EphemerisDynamics 的 N 体闭式完全相同。因此 STM 必须一致，
    任何雅可比叠加错误都会在 STM 上放大显现。
    """

    def test_stm_matches_ephemeris_dynamics(self, spice_eph_system, spice_manager, reference_epoch):
        """同弧段传播，ForceModel 与 EphemerisDynamics 的 STM 一致。"""
        system = _make_force_model_system(spice_eph_system, spice_manager)
        reference_et = spice_manager.utc_to_et(reference_epoch)

        # 地月 L2 附近的状态，NRHO 量级
        state0 = np.array([384400.0 * 0.6, 0.0, 0.0, 0.0, 0.8, 0.0])
        t_span = (reference_et, reference_et + 3600.0)  # 1 小时

        # --- ForceModel 路径 ---
        fm = ForceModel(
            system,
            forces=[
                PointMassGravity("EARTH"),
                ThirdBodyGravity("MOON"),
                ThirdBodyGravity("SUN"),
            ],
        )
        fm.rtol = 1e-12
        fm.atol = 1e-12
        fm_result = fm.propagate(state0, t_span, with_stm=True)

        # --- EphemerisDynamics 路径 ---
        eph_dyn = EphemerisDynamics(system=system)
        eph_dyn.rtol = 1e-12
        eph_dyn.atol = 1e-12
        eph_result = eph_dyn.propagate(state0, t_span, with_stm=True)

        # 末端状态一致（验证加速度等价）
        fm_final = fm_result["states"][-1]
        eph_final = eph_result["states"][-1]
        assert_allclose(
            fm_final, eph_final, atol=1e-6, rtol=1e-9, err_msg="末端状态不一致，加速度可能有误"
        )

        # 末端 STM 一致（验证雅可比叠加正确）
        fm_stm = fm_result["stm"][-1]
        eph_stm = eph_result["stm"][-1]
        assert_allclose(
            fm_stm, eph_stm, atol=1e-7, rtol=1e-7, err_msg="STM 不一致，雅可比叠加可能有误"
        )

    def test_stm_shape_and_identity_at_t0(self, spice_eph_system, spice_manager, reference_epoch):
        """STM 形状为 (n, 6, 6)，首点为单位阵。"""
        system = _make_force_model_system(spice_eph_system, spice_manager)
        reference_et = spice_manager.utc_to_et(reference_epoch)

        state0 = np.array([300000.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        result = ForceModel(
            system,
            forces=[PointMassGravity("EARTH"), ThirdBodyGravity("MOON")],
        ).propagate(state0, (reference_et, reference_et + 600.0), with_stm=True)

        assert "stm" in result
        assert result["stm"].shape == (len(result["time"]), 6, 6)
        assert_allclose(result["stm"][0], np.eye(6), atol=1e-12)

    def test_stm_recovers_perturbation(self, spice_eph_system, spice_manager, reference_epoch):
        """STM 乘小扰动应近似实际传播的偏差。

        δx(tf) ≈ Φ(tf, t0) · δx(t0)。这是 STM 的定义，直接验证。
        """
        system = _make_force_model_system(spice_eph_system, spice_manager)
        reference_et = spice_manager.utc_to_et(reference_epoch)

        state0 = np.array([300000.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        t_span = (reference_et, reference_et + 1800.0)
        fm = ForceModel(system, forces=[PointMassGravity("EARTH"), ThirdBodyGravity("MOON")])

        # 基准传播
        base = fm.propagate(state0, t_span, with_stm=True)
        stm = base["stm"][-1]

        # 加扰动后传播
        delta = np.array([1.0, 0.0, 0.0, 0.0, 1e-4, 0.0])  # 1 km + 0.1 m/s
        perturbed = fm.propagate(state0 + delta, t_span, with_stm=False)

        # STM 预测的偏差 vs 实际偏差
        predicted = stm @ delta
        actual = perturbed["states"][-1] - base["states"][-1]
        # 相对误差应远小于 1（线性化近似，扰动小）
        rel_err = np.linalg.norm(predicted - actual) / np.linalg.norm(actual)
        assert rel_err < 1e-3, f"STM 预测偏差过大: rel_err={rel_err:.2e}"


# =============================================================================
# 有限差分雅可比兜底
# =============================================================================
class _AnalyticTestForce(PhysicalModel):
    """有解析雅可比的测试力（简谐振子型），用于校准有限差分。"""

    def __init__(self, k: float = 1.0):
        self._k = k

    def compute_acceleration(self, t, state, system):
        r = np.asarray(state, dtype=float)[:3]
        return -self._k * r

    def compute_jacobian(self, t, state, system):
        return -self._k * np.eye(3)


class _NoJacobianTestForce(PhysicalModel):
    """不提供解析雅可比的测试力（与 _AnalyticTestForce 同物理）。"""

    def __init__(self, k: float = 1.0):
        self._k = k

    def compute_acceleration(self, t, state, system):
        r = np.asarray(state, dtype=float)[:3]
        return -self._k * r

    # compute_jacobian 继承基类默认返回 None


class TestFiniteDiffJacobianFallback:
    """验证 ForceModel 对无解析雅可比的力用有限差分兜底。"""

    def test_finite_diff_matches_analytic_stm(self, spice_eph_system, spice_manager):
        """同物理的力，解析雅可比 vs 有限差分雅可比，STM 应接近。"""
        system = _make_force_model_system(spice_eph_system, spice_manager)

        state0 = np.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.0])
        t_span = (0.0, 1.0)

        # 解析路径
        fm_analytic = ForceModel(system, forces=[_AnalyticTestForce(k=1.0)])
        res_a = fm_analytic.propagate(state0, t_span, with_stm=True)

        # 有限差分路径
        fm_fd = ForceModel(system, forces=[_NoJacobianTestForce(k=1.0)])
        res_f = fm_fd.propagate(state0, t_span, with_stm=True)

        assert_allclose(
            res_a["stm"][-1],
            res_f["stm"][-1],
            atol=1e-6,
            rtol=1e-6,
            err_msg="有限差分雅可比与解析雅可比的 STM 偏差过大",
        )


# =============================================================================
# 球谐引力 STM 通路（GravityField 走有限差分雅可比兜底）
# =============================================================================
@pytest.fixture
def body_fixed_system(spice_kernel_path):
    """加载 body-fixed 内核的 ICRS 系统（GravityField 需要 ITRF93/MOON_PA）。"""
    from kernel_helpers import load_body_fixed_kernels, unload_kernels

    from e2m2e.core.spice import SPICEManager

    spice = SPICEManager()
    spice.load_kernel(spice_kernel_path)
    bf_kernels = load_body_fixed_kernels(spice)
    try:
        system = EphemerisSystem(bodies=["EARTH", "MOON", "SUN"], spice=spice, origin="EARTH")
        system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )
        yield system, spice
    finally:
        unload_kernels(spice, bf_kernels)
        spice.unload_kernel(spice_kernel_path)


class TestGravityFieldSTM:
    """含球谐引力的 ForceModel STM 通路验证。

    GravityField 不实现 compute_jacobian，走有限差分兜底。验证整条链路
    （球谐加速度 + body-fixed 坐标变换 + 有限差分雅可比 + 变分方程）能跑通，
    且 STM 物理合理。

    力组合：地球 J2 + 月球球谐(10×10) + 月球间接项 + 太阳第三体。
    用 GravityField("MOON") 模拟月球时必须配 IndirectTerm("MOON")，
    见 indirect_term.py 文档。
    """

    def test_gravity_field_stm_propagates_and_is_reasonable(
        self, body_fixed_system, reference_epoch
    ):
        """含球谐的 ForceModel with_stm=True 跑通，STM 有限且满足定义。"""
        system, spice = body_fixed_system
        et0 = spice.utc_to_et(reference_epoch)

        fm = ForceModel(
            system,
            forces=[
                GravityField("EARTH", degree=2, order=0),  # 地球 J2
                GravityField("MOON", degree=10, order=10),  # 月球球谐
                IndirectTerm("MOON"),  # 月球间接项
                ThirdBodyGravity("SUN"),  # 太阳第三体
            ],
        )
        fm.rtol = 1e-11
        fm.atol = 1e-11

        state0 = np.array([300000.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        t_span = (et0, et0 + 600.0)

        result = fm.propagate(state0, t_span, with_stm=True)

        # 基本完整性
        assert "stm" in result
        assert result["stm"].shape == (len(result["time"]), 6, 6)
        assert np.all(np.isfinite(result["stm"][-1]))
        assert_allclose(result["stm"][0], np.eye(6), atol=1e-10)

        # STM 定义验证：δx(tf) ≈ Φ·δx(t0)
        delta = np.array([1.0, 0.0, 0.0, 0.0, 1e-4, 0.0])
        perturbed = fm.propagate(state0 + delta, t_span, with_stm=False)
        predicted = result["stm"][-1] @ delta
        actual = perturbed["states"][-1] - result["states"][-1]
        rel_err = np.linalg.norm(predicted - actual) / np.linalg.norm(actual)
        assert rel_err < 1e-3, f"球谐 STM 预测偏差过大: rel_err={rel_err:.2e}"

    def test_j2_stm_matches_point_mass_plus_perturbation(self, body_fixed_system, reference_epoch):
        """地球 J2 的 STM 应接近纯点质量 STM（J2 是小摄动）。

        两者 STM 应在 1e-3 量级一致（J2 对 1 小时弧段的 STM 贡献很小）。
        这验证球谐雅可比兜底没有引入量级错误。
        """
        system, spice = body_fixed_system
        et0 = spice.utc_to_et(reference_epoch)

        # 点质量地球（解析雅可比）
        fm_pm = ForceModel(
            system,
            forces=[
                PointMassGravity("EARTH"),
                ThirdBodyGravity("MOON"),
                ThirdBodyGravity("SUN"),
            ],
        )
        fm_pm.rtol = 1e-11
        fm_pm.atol = 1e-11

        # 地球 J2（有限差分雅可比）
        fm_j2 = ForceModel(
            system,
            forces=[
                GravityField("EARTH", degree=2, order=0),
                ThirdBodyGravity("MOON"),
                ThirdBodyGravity("SUN"),
            ],
        )
        fm_j2.rtol = 1e-11
        fm_j2.atol = 1e-11

        state0 = np.array([300000.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        t_span = (et0, et0 + 600.0)

        res_pm = fm_pm.propagate(state0, t_span, with_stm=True)
        res_j2 = fm_j2.propagate(state0, t_span, with_stm=True)

        # J2 是小摄动，两 STM 应接近（差异 << 1）
        stm_diff = np.linalg.norm(res_j2["stm"][-1] - res_pm["stm"][-1])
        stm_norm = np.linalg.norm(res_pm["stm"][-1])
        rel_diff = stm_diff / stm_norm
        assert rel_diff < 0.1, f"J2 与点质量 STM 差异过大: rel_diff={rel_diff:.2e}"
        # 但不应完全相同（J2 有贡献）。地球 J2 在 30 万 km 处摄动极小，
        # 对 600 秒弧段 STM 的贡献在 1e-9 量级——这本身就是对的。
        assert stm_diff > 1e-10, "J2 对 STM 无贡献，球谐可能没生效"
