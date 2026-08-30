"""低能转移流水线测试。

测试策略：只保留 ``design_low_energy_transfer`` 入口校验契约。
族延拓 + 流形管传播 + 流水线端到端收敛测试已按维护决策移除
（全量预算超 ADR 0037 上限）。
"""

import pytest

from e2m2e.algorithm.transfer import OrbitTerminal, design_low_energy_transfer
from e2m2e.data.types.orbit import Orbit

pytestmark = pytest.mark.orchestration


class TestDesignLowEnergyTransfer:
    """低能转移流水线入口校验。"""

    def test_invalid_model_raises(self):
        """不支持的 model 在入口即报错，先于任何流形计算。"""
        dummy = Orbit(states=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], times=[0.0], system=None)
        with pytest.raises(ValueError, match="cr3bp"):
            design_low_energy_transfer(OrbitTerminal(dummy), dummy, model="bcr4bp")
