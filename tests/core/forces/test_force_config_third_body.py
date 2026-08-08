"""ThirdBodyGravity / PointMassGravity 配置序列化往返测试（issue #183）。

验证两种力模型的 to_config / from_config 往返，以及 mu=None 的原样保留。
完整 ForceModel 往返通过单点加速度一致性验证（需 SPICE 系统）。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
from e2m2e.algorithm.forces import (
    ForceModel,
    PointMassGravity,
    ThirdBodyGravity,
)
from e2m2e.algorithm.forces.force_config import build_force, serialize_force

# mu=1.327e11 km^3/s^2（约太阳引力参数量级），用于 mu 往返测试。
MU_SUN_LIKE = 1.327e11


# =============================================================================
# ThirdBodyGravity：单力序列化 / 构造 / 往返
# =============================================================================
class TestThirdBodyGravitySerialization:
    """ThirdBodyGravity 的 serialize_force / build_force 往返。"""

    def test_serialize_mu_none(self):
        """serialize_force(ThirdBodyGravity('MOON')) 返回 mu=None。"""
        force = ThirdBodyGravity("MOON")
        config = serialize_force(force)
        assert config == {"type": "ThirdBodyGravity", "params": {"body": "MOON", "mu": None}}

    def test_build_mu_none(self):
        """build_force 用 mu=None 构造等价实例。"""
        force = build_force("ThirdBodyGravity", {"body": "MOON", "mu": None})
        assert isinstance(force, ThirdBodyGravity)
        assert force.body == "MOON"
        assert force.mu is None

    def test_serialize_explicit_mu(self):
        """显式 mu 序列化后 mu 字段保留。"""
        force = ThirdBodyGravity("SUN", mu=MU_SUN_LIKE)
        config = serialize_force(force)
        assert config["params"]["mu"] == MU_SUN_LIKE
        assert config["params"]["body"] == "SUN"

    def test_round_trip_explicit_mu(self):
        """ThirdBodyGravity(mu=具体值) serialize → build 后等价。"""
        original = ThirdBodyGravity("SUN", mu=MU_SUN_LIKE)
        rebuilt = build_force(
            serialize_force(original)["type"],
            serialize_force(original)["params"],
        )
        assert rebuilt.body == original.body
        assert rebuilt.mu == original.mu


# =============================================================================
# PointMassGravity：单力序列化 / 构造 / 往返
# =============================================================================
class TestPointMassGravitySerialization:
    """PointMassGravity 的 serialize_force / build_force 往返。"""

    def test_serialize_mu_none(self):
        """serialize_force(PointMassGravity('EARTH')) 返回 mu=None。"""
        force = PointMassGravity("EARTH")
        config = serialize_force(force)
        assert config == {"type": "PointMassGravity", "params": {"body": "EARTH", "mu": None}}

    def test_build_mu_none(self):
        """build_force 用 mu=None 构造等价实例。"""
        force = build_force("PointMassGravity", {"body": "EARTH", "mu": None})
        assert isinstance(force, PointMassGravity)
        assert force.body == "EARTH"
        assert force.mu is None

    def test_serialize_explicit_mu(self):
        """显式 mu 序列化后 mu 字段保留。"""
        force = PointMassGravity("EARTH", mu=398600.4415)
        config = serialize_force(force)
        assert config["params"]["mu"] == 398600.4415
        assert config["params"]["body"] == "EARTH"

    def test_round_trip_explicit_mu(self):
        """PointMassGravity(mu=具体值) serialize → build 后等价。"""
        original = PointMassGravity("EARTH", mu=398600.4415)
        rebuilt = build_force(
            serialize_force(original)["type"],
            serialize_force(original)["params"],
        )
        assert rebuilt.body == original.body
        assert rebuilt.mu == original.mu


# =============================================================================
# 完整 ForceModel 往返：单点加速度一致性（需 SPICE 系统）
# =============================================================================
pytestmark = [pytest.mark.spice, pytest.mark.l3]


class TestForceModelRoundTripWithPointMassAndThirdBody:
    """ForceModel 含 PointMassGravity + ThirdBodyGravity ×2 的完整往返。

    用 SPICE 系统，验证 to_config → from_config 重建后的力组合在单点上
    与原组合加速度一致。
    """

    def test_full_round_trip_acceleration_consistency(
        self, spice_eph_system, spice_manager, reference_epoch
    ):
        """PointMass(EARTH) + ThirdBody(MOON) + ThirdBody(SUN) 往返后加速度一致。"""
        # ForceModel 需要带 coordinate_system 的 system。
        system = spice_eph_system
        if getattr(system, "coordinate_system", None) is None:
            system.coordinate_system = CoordinateSystem(
                axes=ICRSAxes(),
                origin=CelestialBodyOrigin(body="EARTH", spice=spice_manager),
            )
        reference_et = spice_manager.utc_to_et(reference_epoch)

        state = np.array([384400.0, 0.0, 0.0, 0.0, -0.5, 0.0])

        # 原始 ForceModel：mu=None，运行时从 system 获取。
        fm = ForceModel(
            system,
            forces=[
                PointMassGravity("EARTH"),
                ThirdBodyGravity("MOON"),
                ThirdBodyGravity("SUN"),
            ],
        )

        # to_config → from_config 重建
        config = ForceModel.to_config(fm)
        fm2 = ForceModel.from_config(config, system)

        # config 结构断言
        assert len(config["forces"]) == 3
        types = [e["type"] for e in config["forces"]]
        assert types == ["PointMassGravity", "ThirdBodyGravity", "ThirdBodyGravity"]
        # mu=None 往返：三个力的 mu 均为 None
        for entry in config["forces"]:
            assert entry["params"]["mu"] is None

        # 二次往返幂等：fm2 再 to_config 与原 config 字典相等
        assert ForceModel.to_config(fm2) == config

        # 物理一致性：单点加速度一致
        acc1 = fm._compute_total_acceleration(reference_et, state)
        acc2 = fm2._compute_total_acceleration(reference_et, state)
        assert_allclose(acc2, acc1, atol=1e-12)
