"""ForceModel ``sens_params`` 参数解析契约测试。

敏感列数值正确性验证（跨积分器一致性、shadow-particle 有限差分对照）
属重度真实计算，已随端到端测试裁剪移除；此处仅保留参数解析的显式
报错路径——校验发生在传播开始前，无需 SPICE。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, PointMassGravity
from tests.numerical.forces.conftest import EARTH_MU, FakeSystem

pytestmark = pytest.mark.force


class TestSensParamsContract:
    """参数解析的显式报错路径（无需 SPICE）。"""

    def _fm(self) -> ForceModel:
        return ForceModel(FakeSystem(), forces=[PointMassGravity("EARTH", mu=EARTH_MU)])

    def test_sens_requires_with_stm(self):
        with pytest.raises(ValueError, match="with_stm"):
            self._fm().propagate(
                np.array([7000.0, 0, 0, 0, 7.5, 0]),
                (0.0, 60.0),
                sens_params=["srp_cr"],
            )

    def test_unknown_label_rejected(self):
        with pytest.raises(ValueError, match="未知敏感参数"):
            self._fm().propagate(
                np.array([7000.0, 0, 0, 0, 7.5, 0]),
                (0.0, 60.0),
                with_stm=True,
                sens_params=["mu"],
            )

    def test_missing_force_rejected(self):
        """只有点质量力时求 srp_cr：显式报错而非静默忽略。"""
        with pytest.raises(ValueError, match="SolarRadiationPressure"):
            self._fm().propagate(
                np.array([7000.0, 0, 0, 0, 7.5, 0]),
                (0.0, 60.0),
                with_stm=True,
                sens_params=["srp_cr"],
            )
