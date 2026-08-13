"""ECOM 光压模型的构造与序列化契约。

验证 DYB 系数校验、拷贝语义与 ``to_rust_spec`` 序列化；物理行为由
``physics/test_ecom_srp.py`` 的 Rust 编译传播验证。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.forces.ecom_srp import EcomSolarRadiationPressure
from e2m2e.algorithm.forces.shadow import ConicalShadowModel

pytestmark = pytest.mark.force


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
