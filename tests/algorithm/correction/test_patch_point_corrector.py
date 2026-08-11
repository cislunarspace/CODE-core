"""PatchPointCorrector 接缝类型与注册表测试。

验证 Protocol 运行时检查、EphemerisCorrectionResult 不可变性、
错误类型与分发行为。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from e2m2e.algorithm import ephemeris_correction
from e2m2e.algorithm.ephemeris_correction import (
    EphemerisCorrectionResult,
    PatchPointCorrector,
    UnsupportedCorrectorMethodError,
    correct_ephemeris_patch_points,
    homotopy,
    standard,
    two_level,
)
from e2m2e.data.templates import ConvergenceState, FailureCause

pytestmark = pytest.mark.orchestration


# ---------------------------------------------------------------------------
# 接缝类型测试
# ---------------------------------------------------------------------------


class TestPatchPointCorrectorProtocol:
    """PatchPointCorrector Protocol 的基本契约。"""

    def test_protocol_is_runtime_checkable(self):
        """Protocol 应支持 isinstance 检查。"""
        # PatchPointCorrector 用 @runtime_checkable 装饰
        assert isinstance(PatchPointCorrector, type)

    def test_protocol_has_correct_method(self):
        """Protocol 应定义 correct 方法。"""
        assert hasattr(PatchPointCorrector, "correct")


class TestEphemerisCorrectionResult:
    """EphemerisCorrectionResult 的不可变契约。"""

    def test_is_frozen_dataclass(self):
        """结果应为不可变 dataclass。"""
        result = EphemerisCorrectionResult(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="修正收敛",
            iterations=1,
            max_residual=1e-9,
            residual_history=[1e-9],
            t_patch=np.array([0.0]),
            state_patch=np.zeros((1, 6)),
        )
        with pytest.raises(AttributeError):
            result.status = ConvergenceState.FAILED  # type: ignore[misc]  # type: ignore[misc]

    def test_velocity_fields_default_to_none(self):
        """velocity_residual 和 velocity_residual_history 默认为 None。"""
        result = EphemerisCorrectionResult(
            status=ConvergenceState.CONVERGED,
            cause=FailureCause.NONE,
            message="修正收敛",
            iterations=1,
            max_residual=1e-9,
            residual_history=[1e-9],
            t_patch=np.array([0.0]),
            state_patch=np.zeros((1, 6)),
        )
        assert result.velocity_residual is None
        assert result.velocity_residual_history is None


class TestUnsupportedCorrectorMethodError:
    """UnsupportedCorrectorMethodError 的错误信息。"""

    def test_contains_method_name(self):
        err = UnsupportedCorrectorMethodError("foo", ["standard", "two_level"])
        assert "foo" in str(err)

    def test_contains_available_methods(self):
        err = UnsupportedCorrectorMethodError("foo", ["standard", "two_level"])
        assert "standard" in str(err)
        assert "two_level" in str(err)

    def test_is_value_error(self):
        err = UnsupportedCorrectorMethodError("foo", [])
        assert isinstance(err, ValueError)


# ---------------------------------------------------------------------------
# 注册表测试
# ---------------------------------------------------------------------------


class TestRegistry:
    """注册表应包含所有已知修正方法。"""

    def test_registry_contains_standard(self):
        assert "standard" in ephemeris_correction._REGISTRY

    def test_registry_contains_two_level(self):
        assert "two_level" in ephemeris_correction._REGISTRY

    def test_registry_contains_homotopy(self):
        assert "homotopy" in ephemeris_correction._REGISTRY

    def test_registry_values_are_callable(self):
        for name, factory in ephemeris_correction._REGISTRY.items():
            assert callable(factory), f"{name} factory is not callable"


# ---------------------------------------------------------------------------
# 分发测试（新增路径）
# ---------------------------------------------------------------------------


class TestDispatch:
    """correct_ephemeris_patch_points 的注册表分发行为。"""

    def test_unsupported_method_raises_descriptive_error(self):
        """未知方法应抛 UnsupportedCorrectorMethodError。"""
        with pytest.raises(UnsupportedCorrectorMethodError, match="foobar"):
            correct_ephemeris_patch_points(
                "foobar",
                dynamics="dynamics",
                t_patch=np.array([0.0]),
                state_patch=np.zeros((1, 6)),
                tolerance=1e-3,
                max_iter=5,
                verbose=False,
                n_workers=1,
                kernel_dir="k",
            )

    def test_unsupported_method_error_lists_available_methods(self):
        """错误信息应列出可用方法。"""
        with pytest.raises(UnsupportedCorrectorMethodError, match="standard"):
            correct_ephemeris_patch_points(
                "nonexistent",
                dynamics="dynamics",
                t_patch=np.array([0.0]),
                state_patch=np.zeros((1, 6)),
                tolerance=1e-3,
                max_iter=5,
                verbose=False,
                n_workers=1,
                kernel_dir="k",
            )

    def test_standard_returns_ephemeris_correction_result(self, monkeypatch):
        """standard 方法应返回 EphemerisCorrectionResult。"""
        t_patch = np.array([0.0, 1.0])
        state_patch = np.ones((2, 6))

        class FakeMS:
            def __init__(self, dynamics, n_workers, kernel_dir):
                pass

            def correct(self, **kwargs):
                return SimpleNamespace(
                    status=ConvergenceState.CONVERGED,
                    cause=FailureCause.NONE,
                    message="修正收敛",
                    outer_iterations=2,
                    max_residual=1e-9,
                    residual_history=[1e-6, 1e-9],
                    t_patch=t_patch + 0.1,
                    state_patch=state_patch + 0.01,
                )

        monkeypatch.setattr(standard, "MultipleShooting", FakeMS)
        result = correct_ephemeris_patch_points(
            "standard",
            dynamics="dynamics",
            t_patch=t_patch,
            state_patch=state_patch,
            tolerance=1e-8,
            max_iter=5,
            verbose=False,
            n_workers=1,
            kernel_dir="k",
        )

        assert isinstance(result, EphemerisCorrectionResult)
        assert result.status is ConvergenceState.CONVERGED
        assert result.iterations == 2
        assert result.velocity_residual is None

    def test_two_level_returns_ephemeris_correction_result(self, monkeypatch):
        """two_level 方法应返回 EphemerisCorrectionResult。"""
        t_patch = np.array([0.0, 1.0, 2.0])
        state_patch = np.ones((3, 6))

        class FakeTwoLevelMS:
            def __init__(self, dynamics):
                pass

            def correct(self, **kwargs):
                return SimpleNamespace(
                    status=ConvergenceState.FAILED,
                    cause=FailureCause.UNKNOWN,
                    message="修正失败",
                    outer_iterations=4,
                    final_position_residual=2.5,
                    final_velocity_residual=0.4,
                    residual_history=[(5.0, 1.0), (2.5, 0.4)],
                    t_patch=kwargs["t_patch"] + 0.1,
                    state_patch=kwargs["state_patch"] + 0.01,
                )

        monkeypatch.setattr(two_level, "TwoLevelMultipleShooting", FakeTwoLevelMS)
        result = correct_ephemeris_patch_points(
            "two_level",
            dynamics="dynamics",
            t_patch=t_patch,
            state_patch=state_patch,
            tolerance=1e-3,
            velocity_tolerance=1e-6,
            max_iter=6,
            verbose=False,
            n_workers=1,
            kernel_dir="k",
        )

        assert isinstance(result, EphemerisCorrectionResult)
        assert result.status is not ConvergenceState.CONVERGED
        assert result.velocity_residual == 0.4
        assert result.velocity_residual_history == [1.0, 0.4]

    def test_homotopy_returns_ephemeris_correction_result(self, monkeypatch):
        """homotopy 方法应返回 EphemerisCorrectionResult。"""
        t_patch = np.array([0.0, 1.0])
        state_patch = np.ones((2, 6))

        def fake_homotopy(dynamics, t_patch, state_patch, **kwargs):
            return EphemerisCorrectionResult(
                status=ConvergenceState.CONVERGED,
                cause=FailureCause.NONE,
                message="修正收敛",
                iterations=1,
                max_residual=1e-9,
                residual_history=[1e-9],
                t_patch=t_patch,
                state_patch=state_patch,
            )

        monkeypatch.setattr(homotopy, "correct_with_homotopy", fake_homotopy)
        result = correct_ephemeris_patch_points(
            "homotopy",
            dynamics="dynamics",
            t_patch=t_patch,
            state_patch=state_patch,
            tolerance=1e-8,
            max_iter=5,
            verbose=False,
            n_workers=1,
            kernel_dir="k",
            base_bodies=["EARTH", "MOON"],
        )

        assert isinstance(result, EphemerisCorrectionResult)
        assert result.status is ConvergenceState.CONVERGED

    def test_standard_corrector_satisfies_protocol(self, monkeypatch):
        """_StandardPatchPointCorrector 应满足 PatchPointCorrector 协议。"""

        class FakeMS:
            def __init__(self, dynamics, n_workers, kernel_dir):
                pass

            def correct(self, **kwargs):
                return SimpleNamespace(
                    status=ConvergenceState.CONVERGED,
                    cause=FailureCause.NONE,
                    message="修正收敛",
                    outer_iterations=1,
                    max_residual=1e-9,
                    residual_history=[1e-9],
                    t_patch=np.array([0.0]),
                    state_patch=np.zeros((1, 6)),
                )

        monkeypatch.setattr(standard, "MultipleShooting", FakeMS)
        corrector = ephemeris_correction._StandardPatchPointCorrector(
            "dynamics", n_workers=1, kernel_dir="k"
        )
        assert isinstance(corrector, PatchPointCorrector)

    def test_two_level_corrector_satisfies_protocol(self, monkeypatch):
        """_TwoLevelPatchPointCorrector 应满足 PatchPointCorrector 协议。"""

        class FakeTwoLevelMS:
            def __init__(self, dynamics):
                pass

            def correct(self, **kwargs):
                return SimpleNamespace(
                    status=ConvergenceState.CONVERGED,
                    cause=FailureCause.NONE,
                    message="修正收敛",
                    outer_iterations=1,
                    final_position_residual=1e-9,
                    final_velocity_residual=1e-12,
                    residual_history=[(1e-9, 1e-12)],
                    t_patch=np.array([0.0]),
                    state_patch=np.zeros((1, 6)),
                )

        monkeypatch.setattr(two_level, "TwoLevelMultipleShooting", FakeTwoLevelMS)
        corrector = ephemeris_correction._TwoLevelPatchPointCorrector("dynamics")
        assert isinstance(corrector, PatchPointCorrector)

    def test_homotopy_corrector_satisfies_protocol(self):
        """_HomotopyPatchPointCorrector 应满足 PatchPointCorrector 协议。"""
        corrector = ephemeris_correction._HomotopyPatchPointCorrector(
            "dynamics", base_bodies=["EARTH"]
        )
        assert isinstance(corrector, PatchPointCorrector)


# ---------------------------------------------------------------------------
# 回归：新增 UnsupportedCorrectorMethodError 替代 ValueError
# ---------------------------------------------------------------------------


class TestErrorHandlingRegression:
    """确认新错误类型是 ValueError 的子类，保持向后兼容。"""

    def test_unsupported_method_is_value_error_subclass(self):
        """UnsupportedCorrectorMethodError 继承 ValueError。"""
        with pytest.raises(ValueError):
            correct_ephemeris_patch_points(
                "nonexistent",
                dynamics="dynamics",
                t_patch=np.array([0.0]),
                state_patch=np.zeros((1, 6)),
                tolerance=1e-3,
                max_iter=5,
                verbose=False,
                n_workers=1,
                kernel_dir="k",
            )
