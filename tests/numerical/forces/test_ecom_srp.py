"""ECOM 光压模型的 Rust 编译传播验证。

ECOM 的 D-Y-B 几何、阴影和加速度计算均在 Rust 内执行，Python 侧只保存
配置并序列化为力元组。因此本模块以同一 SPICE 场景下的短弧传播结果验证
Rust 端到端契约；不再保留不能覆盖运行路径的 Python 参考公式。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.ecom_srp import EcomSolarRadiationPressure
from e2m2e.algorithm.forces.point_mass_gravity import PointMassGravity
from e2m2e.algorithm.forces.shadow import ConicalShadowModel
from e2m2e.algorithm.forces.srp import SolarRadiationPressure

pytestmark = [pytest.mark.force, pytest.mark.spice]

_EARTH_MU = 398600.4418


@pytest.fixture
def ecom_system(spice_eph_system):
    """给 compiled forces 提供 SPICE 已加载的地心传播系统。"""
    spice_eph_system.coordinate_system = object()
    return spice_eph_system


def _propagate(system, force, t_eval):
    model = ForceModel(system, [PointMassGravity("EARTH", mu=_EARTH_MU), force])
    state = np.array([42164.0, 0.0, 0.0, 0.0, 3.074666284127684, 0.0])
    return model.propagate(state, (float(t_eval[0]), float(t_eval[-1])), t_eval=t_eval)


class TestEcomConstruction:
    """构造与配置序列化。"""

    def test_dyb_length_must_be_9(self):
        with pytest.raises(ValueError, match="9 elements"):
            EcomSolarRadiationPressure(dyb=[0.01] * 8)

    def test_dyb_returns_copy(self):
        ecom = EcomSolarRadiationPressure(dyb=[0.01] * 9)
        returned = ecom.dyb
        returned[0] = 999.0
        assert ecom.dyb[0] == 0.01

    def test_to_rust_spec_preserves_dyb_and_shadow_bodies(self):
        dyb = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ecom = EcomSolarRadiationPressure(
            dyb=dyb,
            shadow=ConicalShadowModel(bodies=["EARTH", "MOON"]),
        )

        assert ecom.to_rust_spec() == ("ecom_srp", dyb, ["EARTH", "MOON"])
        assert (
            EcomSolarRadiationPressure.from_config(ecom.to_config()).to_config() == ecom.to_config()
        )


class TestEcomCompiledPropagation:
    """验证实际传入 Rust 的 ECOM 物理行为。"""

    def test_zero_dyb_matches_point_mass_trajectory(self, ecom_system):
        """零 DYB 时 Rust ECOM 不应向轨迹加入任何加速度。"""
        t_eval = np.array([0.0, 300.0, 900.0])
        ecom = EcomSolarRadiationPressure(dyb=[0.0] * 9)
        result = _propagate(ecom_system, ecom, t_eval)
        baseline = ForceModel(ecom_system, [PointMassGravity("EARTH", mu=_EARTH_MU)]).propagate(
            np.array([42164.0, 0.0, 0.0, 0.0, 3.074666284127684, 0.0]),
            (0.0, 900.0),
            t_eval=t_eval,
        )

        assert_allclose(result["time"], t_eval, atol=0.0)
        assert_allclose(result["states"], baseline["states"], atol=1e-10)

    def test_cannonball_degradation_matches_rust_srp(self, ecom_system):
        """仅 dyb[0] 非零时，Rust ECOM 应退化为 Rust cannonball SRP。"""
        t_eval = np.array([0.0, 300.0, 900.0])
        area, mass, cr = 10.0, 1000.0, 1.5
        ecom = EcomSolarRadiationPressure(dyb=[cr * area / mass] + [0.0] * 8)
        srp = SolarRadiationPressure(area=area, mass=mass, cr=cr)

        ecom_result = _propagate(ecom_system, ecom, t_eval)
        srp_result = _propagate(ecom_system, srp, t_eval)

        assert_allclose(ecom_result["time"], srp_result["time"], atol=0.0)
        assert_allclose(ecom_result["states"], srp_result["states"], atol=1e-10)

    def test_periodic_dyb_coefficient_changes_rust_trajectory(self, ecom_system):
        """D 方向周期项写入 Rust 后，短弧轨迹必须与 cannonball 配置不同。"""
        t_eval = np.array([0.0, 900.0, 1800.0])
        baseline = _propagate(
            ecom_system,
            EcomSolarRadiationPressure(dyb=[0.01] + [0.0] * 8),
            t_eval,
        )
        periodic = _propagate(
            ecom_system,
            EcomSolarRadiationPressure(dyb=[0.01, 0.5] + [0.0] * 7),
            t_eval,
        )

        assert np.linalg.norm(periodic["states"][-1] - baseline["states"][-1]) > 1e-7
