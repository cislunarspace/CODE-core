"""中心流形约化后 Lissajous 轨道有界性验证（issue #255，Phase 1）。

从独立验证脚本 ``_cm_bounded_lissajous_check.py`` 转化而来的正式测试。

覆盖：

- 双曲耦合已消除（``cm_result.max_hyperbolic_coupling == 0``）；
- 6 周期传播后位置振幅保持有界（最大偏移 < 3× 初始偏移）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.algorithm.family.lissajous_initial_guess import compute_lissajous_initial_guess
from e2m2e.algorithm.normal_form.constants import JD0_J2000, LibrationPoint
from e2m2e.algorithm.normal_form.context import NormalFormContext
from e2m2e.algorithm.normal_form.pipeline import NormalFormPipeline
from e2m2e.algorithm.normal_form.propagation import propagate_parametric

MU_EM = 1.215058560962404e-2


@pytest.fixture
def earth_moon_system():
    """地月 CR3BP 系统（与 normal_form/conftest 一致）。"""
    system = CR3BP_System(mu=MU_EM, primary="Earth", secondary="Moon")
    system.set_characteristic_scales(distance=384405.0, period=27.32 * 86400.0)
    return system


@pytest.fixture
def l2_lissajous_orbit(earth_moon_system):
    """L2 Lissajous 轨道初猜（小振幅，与 fast pipeline 匹配）。"""
    state0, T = compute_lissajous_initial_guess(earth_moon_system, 2, 100.0, 300.0, 0.0, 0.0)
    return state0, T


@pytest.fixture
def l2_context(earth_moon_system):
    """L2 共线点上下文（order=5，与 fast pipeline 匹配）。

    ``force_cr3bp=True`` 显式声明纯 CR3BP 约化：本测试验证的就是 CR3BP
    Lissajous 的有界性，与星历无关；也使测试不受全局 SPICE 内核态（其他
    用例加载后未卸载）的影响——整条约化路径直接用 CR3BP 常量、不探 SPICE。
    """
    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L2,
        epoch=JD0_J2000,
        order=5,
        force_cr3bp=True,
    )


@pytest.fixture
def l2_pipeline(l2_context):
    """L2 法型化简流水线（fast 参数，数秒内完成）。"""
    return NormalFormPipeline(
        context=l2_context,
        center_max_order=5,
        center_steps=("invariant", "center"),
        dynamical_kwargs={
            "t_total": 4.0,
            "node_step": 0.8,
            "dense_step": 0.2,
            "max_iter": 3,
            "tolerance": 1e-6,
            "prefer": "fft",
        },
    )


@pytest.fixture
def nf_result(l2_pipeline, l2_lissajous_orbit):
    """法型化简结果（缓存供同一类内多个测试复用）。"""
    state0, _ = l2_lissajous_orbit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return l2_pipeline.reduce(state0)


@pytest.fixture
def propagation_result(l2_lissajous_orbit, nf_result, l2_context):
    """6 周期传播结果（缓存供同一类内多个测试复用）。"""
    state0, T = l2_lissajous_orbit
    t_span = [0, 6 * T]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return propagate_parametric(state0, t_span, nf_result, l2_context)


@pytest.mark.slow
class TestBoundedLissajous:
    """中心流形约化后 Lissajous 轨道有界性验证。"""

    def test_hyperbolic_coupling_eliminated(self, nf_result):
        """中心流形约化应消除双曲耦合。"""
        assert nf_result.cm_result.max_hyperbolic_coupling == 0

    def test_m0_flow_amplitude_bounded(self, l2_lissajous_orbit, propagation_result):
        """6 周期传播后位置振幅保持有界（比值 < 3×）。"""
        state0, _ = l2_lissajous_orbit
        _, rho_out, _ = propagation_result
        pos = np.linalg.norm(rho_out[:, :3], axis=1)
        ratio = pos.max() / pos[0]
        assert ratio < 3.0, f"Amplitude ratio {ratio:.1f} exceeds 3x bound"
