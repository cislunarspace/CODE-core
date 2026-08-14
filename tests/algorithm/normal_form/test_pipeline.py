"""法型化流水线的端到端 smoke 测试（issue #175，切片 6）。

覆盖：

- :class:`NormalFormPipeline` 可用 :class:`NormalFormContext` 构造；
- ``reduce(orbit)`` 把四个 reducer 串成完整路径，返回
  :class:`NormalFormResult`；
- ``result.catalog_transformer.rho_to_param(orbit, t)`` 返回 ``(6,)`` 数组
  （验收标准 #2）；
- 失败路径：某步异常时 ``success=False`` 且保留已完成子结果；
- 输入校验：非法 orbit 形状直接抛 :class:`ValueError`；
- lazy export 经包顶层可导入。

SPICE leapseconds 不可用时，fixture 显式声明 ``spice_optional=True`` 走纯
CR3BP 路径（ADR 0020 决策 4：显式选择，非隐式降级），smoke 测试在该路径下
仍能通过（与 ``test_dynamical_substitution.py`` 一致）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from e2m2e.algorithm.dynamics import LibrationPoint
from e2m2e.algorithm.normal_form import (
    NormalFormPipeline,
    NormalFormResult,
)
from e2m2e.algorithm.normal_form.center_manifold import CenterManifoldResult
from e2m2e.algorithm.normal_form.dynamical_substitution import (
    DynamicalSubstituteResult,
)
from e2m2e.algorithm.normal_form.quasi_floquet import QuasiFloquetResult
from e2m2e.data.templates import ConvergenceState

pytestmark = pytest.mark.theory


# ---------------------------------------------------------------------------
# 公共 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def l1_context(earth_moon_system):
    """L1 共线点上下文，显式纯 CR3BP（不探 SPICE）。

    本文件的流水线测试以纯 CR3BP 归一化模型为前提；显式 force_cr3bp
    使结果与同 worker 内其他模块留下的内核池状态无关（ADR 0020：
    显式选择，不靠环境巧合）。
    """
    from e2m2e.algorithm.normal_form import NormalFormContext
    from e2m2e.algorithm.normal_form.constants import JD0_J2000

    return NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=4,
        force_cr3bp=True,
    )


@pytest.fixture
def fast_pipeline(l1_context) -> NormalFormPipeline:
    """小窗口流水线，让端到端 reduce 在数秒内完成。

    动力学替代默认窗口 ``t_total=0.1·2^16`` 太慢；这里压到 ``4.0 TU``，
    与 ``test_dynamical_substitution.tiny_corrector`` 同量级。中心流形
    截断到 5 阶（足以走通两步 Lie 变换）。
    """
    return NormalFormPipeline(
        context=l1_context,
        # CR3BP 归一化模型（无 SPICE 内核池）下 QF 须用 constant 方法
        # （M(t) 常数矩阵；ADR 0020 决策 4 显式选择，不静默降）。
        quasi_floquet_method="constant",
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


# ---------------------------------------------------------------------------
# 构造与导入
# ---------------------------------------------------------------------------


def test_pipeline_is_constructible(l1_context):
    """``NormalFormPipeline`` 可用 context 构造，默认旋钮合理。"""
    pipeline = NormalFormPipeline(context=l1_context)
    assert pipeline.context is l1_context
    assert pipeline.quasi_floquet_method == "matrix"
    assert pipeline.center_max_order == 10
    assert pipeline.center_steps == ("invariant", "center")


def test_pipeline_importable_via_package_root():
    """切片 #175 验收：能从包根直接 import。"""
    from e2m2e.algorithm.normal_form import NormalFormPipeline as P

    assert P is NormalFormPipeline


# ---------------------------------------------------------------------------
# 端到端 smoke（验收标准 #1、#2）
# ---------------------------------------------------------------------------


def test_reduce_returns_normal_form_result(fast_pipeline):
    """``reduce(orbit)`` 返回 :class:`NormalFormResult` 且暴露四个子句柄。"""
    x0 = np.array([1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 屏蔽 SPICE 降级警告
        result = fast_pipeline.reduce(x0)

    assert isinstance(result, NormalFormResult)
    assert result.context is fast_pipeline.context
    assert result.order == int(fast_pipeline.context.order)
    # 四个子句柄全部填充
    assert isinstance(result.ds_result, DynamicalSubstituteResult)
    assert isinstance(result.qf_result, QuasiFloquetResult)
    assert isinstance(result.cm_result, CenterManifoldResult)
    assert result.catalog_transformer is not None
    # 通用诊断字段
    assert result.status is ConvergenceState.CONVERGED
    assert isinstance(result.message, str) and result.message
    assert np.isfinite(result.residual)


def test_catalog_transformer_rho_to_param(fast_pipeline):
    """验收标准 #2：``result.catalog_transformer.rho_to_param`` 返回 (6,)。"""
    x0 = np.array([1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fast_pipeline.reduce(x0)

    param = result.catalog_transformer.rho_to_param(x0, t=0.0)
    assert np.asarray(param).shape == (6,)
    assert np.all(np.isfinite(param))


def test_catalog_transformer_roundtrip(fast_pipeline):
    """端到端 rho → param → rho 在合成初值上数值自洽。"""
    x0 = np.array([1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fast_pipeline.reduce(x0)

    tr = result.catalog_transformer
    t = 1.0
    param = tr.rho_to_param(x0, t)
    back = tr.param_to_rho(param, t)
    # 高阶 Lie 级数段主导，整体放宽（与 test_catalog 端到端往返一致）
    np.testing.assert_allclose(back, x0, atol=1e-6)


def test_reduce_works_without_spice_kernels(earth_moon_system, monkeypatch):
    """显式 spice_optional 路径：SPICE 探测失败时回退纯 CR3BP 仍跑通。

    与共享 fixture 的显式 CR3BP 上下文不同，本测试针对回退分支本身，
    故构造未设 force_cr3bp 的上下文并以 monkeypatch 模拟内核缺失。
    """
    import e2m2e.algorithm.normal_form.dynamical_substitution as ds
    from e2m2e.algorithm.normal_form import NormalFormContext
    from e2m2e.algorithm.normal_form.constants import JD0_J2000

    def _broken(*args, **kwargs):
        raise RuntimeError("simulated SPICE missing")

    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form._ephemeris.eval_params",
        _broken,
        raising=False,
    )
    monkeypatch.setattr(ds, "_build_dynamics_rhs_spice", _broken)

    context = NormalFormContext(
        system=earth_moon_system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=4,
    )
    pipeline = NormalFormPipeline(
        context=context,
        quasi_floquet_method="constant",
        center_max_order=4,
        dynamical_kwargs={
            "t_total": 4.0,
            "node_step": 0.8,
            "dense_step": 0.2,
            "max_iter": 3,
            "tolerance": 1e-6,
            "prefer": "fft",
            "spice_optional": True,
        },
    )

    x0 = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipeline.reduce(x0)

    assert result.status is ConvergenceState.CONVERGED
    assert result.ds_result.spice_available is False
    assert result.catalog_transformer is not None


# ---------------------------------------------------------------------------
# 输入校验与失败路径
# ---------------------------------------------------------------------------


def test_reduce_rejects_bad_orbit_shape(fast_pipeline):
    """orbit 非 (6,) 时直接抛 ValueError（调用方错误）。"""
    with pytest.raises(ValueError, match="orbit"):
        fast_pipeline.reduce(np.zeros(5))


def test_reduce_accepts_orbit_like_object(fast_pipeline, l1_context):
    """带 ``.states`` 的 Orbit-like 对象取首帧作为初值。"""
    from e2m2e.data.types.orbit import Orbit

    x0 = np.array([1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4])
    orbit = Orbit(
        states=np.tile(x0, (3, 1)),
        times=np.array([0.0, 0.1, 0.2]),
        system=l1_context.system,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fast_pipeline.reduce(orbit)
    assert result.status is ConvergenceState.CONVERGED


def test_failure_records_completed_subresults(l1_context, monkeypatch):
    """某步异常时 success=False，保留已完成子结果，message 记录失败步骤。"""

    # 让 quasi-Floquet 步抛异常，DS 步应已完成
    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic QF failure")

    monkeypatch.setattr(
        "e2m2e.algorithm.normal_form.quasi_floquet.QuasiFloquetReducer.reduce",
        _boom,
    )

    pipeline = NormalFormPipeline(
        context=l1_context,
        center_max_order=4,
        dynamical_kwargs={
            "t_total": 2.0,
            "node_step": 0.8,
            "dense_step": 0.4,
            "max_iter": 2,
            "tolerance": 1e-6,
            "prefer": "fft",
        },
    )
    x0 = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipeline.reduce(x0)

    assert result.status is ConvergenceState.FAILED
    assert "quasi_floquet" in result.message
    # DS 步已完成，后续步骤为 None
    assert isinstance(result.ds_result, DynamicalSubstituteResult)
    assert result.qf_result is None
    assert result.cm_result is None
    assert result.catalog_transformer is None
    assert result.metadata["failed_step"] == "quasi_floquet"


# ---------------------------------------------------------------------------
# metadata 诊断
# ---------------------------------------------------------------------------


def test_metadata_records_pipeline_config(fast_pipeline):
    """成功结果 metadata 记录流水线配置与各步诊断量。"""
    x0 = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fast_pipeline.reduce(x0)

    assert result.metadata["quasi_floquet_method"] == "constant"
    assert result.metadata["center_max_order"] == 5
    assert result.metadata["center_steps"] == ("invariant", "center")
    assert "qf_symplectic_error" in result.metadata
    assert "cm_hyperbolic_coupling" in result.metadata
