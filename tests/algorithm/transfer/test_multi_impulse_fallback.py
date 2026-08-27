"""multi_impulse 初猜回退的失败语义测试。

``_default_x0`` 的双脉冲封闭初猜失败时，回退端点线性插值必须带标记
（logger.warning），不静默；编程错误（非预期的 Lambert/打靶/传播失败）
不被 ``except Exception`` 吞掉。
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from e2m2e.algorithm.transfer.multi_impulse import MultiImpulseTransfer
from e2m2e.algorithm.transfer.terminal import StateTerminal

pytestmark = pytest.mark.orchestration

R1 = 6778.0
R2 = 42164.0
MU_EARTH = 398600.435507
TOF_HOHMANN = float(np.pi * np.sqrt((R1 + R2) ** 3 / (8.0 * MU_EARTH)))


def _circular(r: float, angle: float = 0.0) -> np.ndarray:
    v = np.sqrt(MU_EARTH / r)
    return np.array(
        [r * np.cos(angle), r * np.sin(angle), 0.0, -v * np.sin(angle), v * np.cos(angle), 0.0]
    )


def _make_transfer() -> MultiImpulseTransfer:
    return MultiImpulseTransfer(
        StateTerminal(_circular(R1), 0.0),
        StateTerminal(_circular(R2, np.pi), TOF_HOHMANN),
        mu=MU_EARTH,
    )


class TestDefaultX0FallbackNotSilent:
    def test_closure_failure_falls_back_with_warning(self, caplog):
        """双脉冲封闭失败 → 回退线性插值初猜并记录警告（不静默退化）。"""
        transfer = _make_transfer()

        def _boom_close_velocities(y, closure):
            raise RuntimeError("打靶未收敛（弧 0）")

        transfer._close_velocities = _boom_close_velocities  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger="e2m2e.algorithm.transfer.multi_impulse"):
            x0 = transfer._default_x0(n_mid=2, closure="two_body")

        assert any("回退端点线性插值" in rec.message for rec in caplog.records), caplog.text
        # 线性插值初猜：中途节点在两端点之间等间隔
        r0, rf = transfer.term0.state[:3], transfer.term1.state[:3]
        fracs = np.array([1.0, 2.0]) / 3.0
        expected = np.concatenate([fracs * TOF_HOHMANN, (r0 + np.outer(fracs, rf - r0)).ravel()])
        np.testing.assert_allclose(x0, expected, atol=1e-12)

    def test_programming_error_not_swallowed(self):
        """编程错误（TypeError）不被 ``except Exception`` 吞掉（收窄）。"""
        transfer = _make_transfer()

        def _boom_close_velocities(y, closure):
            raise TypeError("内部编程错误：参数类型不符")

        transfer._close_velocities = _boom_close_velocities  # type: ignore[method-assign]
        with pytest.raises(TypeError, match="内部编程错误"):
            transfer._default_x0(n_mid=2, closure="two_body")
