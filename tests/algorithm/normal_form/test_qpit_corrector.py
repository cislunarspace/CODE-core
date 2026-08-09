"""QPIT 正确器测试。"""

from __future__ import annotations

import warnings

import pytest

from e2m2e.algorithm.dynamics import CR3BP_System
from e2m2e.algorithm.family.lissajous_initial_guess import compute_lissajous_initial_guess
from e2m2e.algorithm.normal_form.constants import JD0_J2000, LibrationPoint
from e2m2e.algorithm.normal_form.context import NormalFormContext
from e2m2e.algorithm.normal_form.corrector import QPITCorrector, QPITCorrectorResult
from e2m2e.algorithm.normal_form.pipeline import NormalFormPipeline

pytestmark = pytest.mark.theory


MU_EM = 1.215058560962404e-2


@pytest.fixture
def earth_moon_system():
    """地月 CR3BP 系统。"""
    system = CR3BP_System(mu=MU_EM, primary="Earth", secondary="Moon")
    system.set_characteristic_scales(distance=384405.0, period=27.32 * 86400.0)
    return system


@pytest.fixture
def nf_result_and_context(earth_moon_system):
    """L2 small-amplitude Lissajous normal form result + context."""
    state0, _ = compute_lissajous_initial_guess(earth_moon_system, 2, 200.0, 600.0, 0.0, 0.0)
    context = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L2,
        epoch=JD0_J2000,
        order=5,
    )
    pipeline = NormalFormPipeline(
        context=context,
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipeline.reduce(state0)
    return result, context, state0


class TestQPITCorrector:
    """QPITCorrector 基本功能测试。"""

    def test_converges_small_amplitude(self, nf_result_and_context):
        """小振幅修正应收敛（CM 空间解析映射，一步收敛）。"""
        nf_result, context, state0 = nf_result_and_context
        corrector = QPITCorrector(nf_result=nf_result, context=context, max_iter=30)
        result = corrector.correct(
            target_amplitude_in=200.0,
            target_amplitude_out=600.0,
            seed_state=state0,
            periods=2,
        )
        assert isinstance(result, QPITCorrectorResult)
        assert result.converged
        assert result.iterations <= 3

    def test_output_shapes(self, nf_result_and_context):
        """输出维度正确。"""
        nf_result, context, state0 = nf_result_and_context
        corrector = QPITCorrector(nf_result=nf_result, context=context)
        result = corrector.correct(200.0, 600.0, seed_state=state0, periods=2)
        assert result.param.shape == (6,)

    def test_amplitude_matches_target(self, nf_result_and_context):
        """修正后 CM 振幅应精确匹配目标值。"""
        nf_result, context, state0 = nf_result_and_context
        corrector = QPITCorrector(nf_result=nf_result, context=context)
        result = corrector.correct(200.0, 600.0, seed_state=state0, periods=2)
        # CM 振幅是 sqrt(2I)*LU，解析精确
        assert abs(result.amplitude_in_actual - 200.0) < 0.01
        assert abs(result.amplitude_out_actual - 600.0) < 0.01

    def test_residual_history_converged(self, nf_result_and_context):
        """收敛后残差应接近零。"""
        nf_result, context, state0 = nf_result_and_context
        corrector = QPITCorrector(nf_result=nf_result, context=context)
        result = corrector.correct(200.0, 600.0, seed_state=state0, periods=2)
        assert result.residual_history[-1] < 1.0

    def test_param_contains_seed_phases(self, nf_result_and_context):
        """param 中的相位角应来自种子状态的正向映射。"""
        nf_result, context, state0 = nf_result_and_context
        corrector = QPITCorrector(nf_result=nf_result, context=context)
        result = corrector.correct(200.0, 600.0, seed_state=state0, periods=2)
        # q1, p1, theta2, theta3 应非零（来自 seed forward map）
        assert result.param[0] != 0.0 or result.param[1] != 0.0
        assert result.param[3] != 0.0
        assert result.param[5] != 0.0
