"""design_halo 的 L1 超界报错语义回归（issue #773）。

报错必须区分固定 z0 延拓安全上界（0.07×特征长度 ≈26908 km）与族折叠点
（|z0|≈0.085 ≈32674 km）：前者是本实现路径的能力边界，族越过折叠点后
转入近月 NRHO 段继续延伸，报错不得声称目标振幅的 L1 Halo 不存在，
并须指向 design_nrho / design_nrho_family。
"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.family import Cr3bpOrbitError, design_halo

pytestmark = pytest.mark.orchestration


def test_l1_beyond_limit_message_distinguishes_bound_from_fold() -> None:
    """超界报错同时给出安全上界与折叠点两个数值，并指向 NRHO 设计路径。"""
    with pytest.raises(Cr3bpOrbitError) as exc_info:
        design_halo(1, 40000.0)
    message = str(exc_info.value)
    assert "26908" in message
    assert "32674" in message
    assert "design_nrho" in message
    assert "design_nrho_family" in message
    assert "不存在" not in message
