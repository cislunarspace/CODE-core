"""StabilityAnalysis 未覆盖分支测试。

使用模拟单值矩阵覆盖分类、分岔、稳定性指数、
字符串表示与静态辅助方法。
"""

import numpy as np
import pytest

from e2m2e.algorithms.stability import BifurcationType, StabilityAnalysis, StabilityType
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit


def _make_orbit(n=10, period=None, system=None):
    states = np.random.RandomState(42).randn(n, 6)
    times = np.linspace(0, period or 1.0, n)
    orbit = Orbit(states, times)
    orbit.period = period
    if system is not None:
        orbit.system = system
    return orbit


def _inject_monodromy(analysis, matrix):
    analysis.monodromy_matrix = matrix
    analysis.has_monodromy = True


def _inject_eigenvalues(analysis, eigenvalues):
    analysis.eigenvalues = np.array(eigenvalues, dtype=complex)
    analysis.eigenvectors = np.eye(len(eigenvalues), dtype=complex)
    analysis.eigenvalue_magnitudes = np.abs(analysis.eigenvalues)
    analysis.eigenvalue_arguments = np.angle(analysis.eigenvalues)
    analysis.sorted_eigenvalues = analysis.eigenvalues[np.argsort(-analysis.eigenvalue_magnitudes)]
    analysis.floquet_multipliers = analysis.eigenvalues.copy()
    analysis.floquet_exponents = None
    analysis.has_eigenvalues = True
    analysis._pair_eigenvalues()


class TestInitWithSystemAttribute:
    def test_auto_creates_dynamics_from_system(self):
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        orbit = _make_orbit(system=system)
        analysis = StabilityAnalysis(orbit)
        assert isinstance(analysis.dynamics, CR3BP_Dynamics)


class TestComputeMonodromyNoPeriod:
    def test_raises_when_period_unknown(self):
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        orbit = _make_orbit(system=system)
        orbit.period = None
        analysis = StabilityAnalysis(orbit)
        with pytest.raises(ValueError, match="轨道周期未知"):
            analysis.compute_monodromy()


class TestClassifyOrbitStable:
    def test_all_on_unit_circle_is_stable(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [1.0 + 0j, 1.0 + 0j, np.exp(1j), np.exp(-1j), np.exp(0.5j), np.exp(-0.5j)]
        _inject_eigenvalues(analysis, eigs)
        result = analysis.classify_orbit()
        assert analysis.stability_type == StabilityType.STABLE
        assert analysis.is_stable
        assert "stability_type" in result


class TestClassifyOrbitUnstable:
    def test_unstable_eigenvalue(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [2.0, 0.5, 1.0 + 0j, 1.0 + 0j, np.exp(0.3j), np.exp(-0.3j)]
        _inject_eigenvalues(analysis, eigs)
        analysis.classify_orbit()
        assert analysis.is_unstable

    def test_hyperbolic_real_unstable(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [2.0, 0.5, 1.0, 1.0, np.exp(0.2j), np.exp(-0.2j)]
        _inject_eigenvalues(analysis, eigs)
        analysis.classify_orbit()
        assert analysis.stability_type == StabilityType.HYPERBOLIC

    def test_complex_unstable_not_hyperbolic(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [
            1.5 * np.exp(0.5j),
            (1.0 / 1.5) * np.exp(-0.5j),
            1.0,
            1.0,
            np.exp(0.2j),
            np.exp(-0.2j),
        ]
        _inject_eigenvalues(analysis, eigs)
        analysis.classify_orbit()
        assert analysis.is_unstable

    def test_marginally_stable(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [0.5, 0.5, 1.0 + 0j, 1.0 + 0j, np.exp(0.3j), np.exp(-0.3j)]
        _inject_eigenvalues(analysis, eigs)
        analysis.classify_orbit()
        assert analysis.stability_type == StabilityType.MARGINALLY_STABLE
        assert analysis.is_critical


class TestAnalyzeBifurcation:
    def test_saddle_node_eigenvalue_at_one(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [2.0, 0.5, 1.0, 1.0, np.exp(0.3j), np.exp(-0.3j)]
        _inject_eigenvalues(analysis, eigs)
        analysis.analyze_bifurcation()
        assert analysis.bifurcation_type == BifurcationType.SADDLE_NODE
        assert analysis.bifurcation_detected

    def test_period_doubling_eigenvalue_at_minus_one(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [2.0, 0.5, -1.0 + 0j, -1.0 + 0j, np.exp(0.3j), np.exp(-0.3j)]
        _inject_eigenvalues(analysis, eigs)
        analysis.analyze_bifurcation()
        assert analysis.bifurcation_type == BifurcationType.PERIOD_DOUBLING
        assert analysis.bifurcation_detected

    def test_torus_bifurcation_complex_on_unit_circle(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [2.0, 0.5, np.exp(0.5j), np.exp(-0.5j), np.exp(0.7j), np.exp(-0.7j)]
        _inject_eigenvalues(analysis, eigs)
        analysis.analyze_bifurcation()
        assert analysis.bifurcation_type == BifurcationType.TORUS

    def test_no_bifurcation(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [2.0, 0.5, 0.3, 3.0, 1.5, 0.7]
        _inject_eigenvalues(analysis, eigs)
        analysis.analyze_bifurcation()
        assert analysis.bifurcation_type == BifurcationType.NONE
        assert not analysis.bifurcation_detected


class TestComputeStabilityIndex:
    def test_computes_nu_indices(self):
        orbit = _make_orbit(period=2.0)
        analysis = StabilityAnalysis(orbit)
        eigs = [2.0, 0.5, np.exp(0.3j), np.exp(-0.3j), np.exp(0.7j), np.exp(-0.7j)]
        _inject_eigenvalues(analysis, eigs)
        result = analysis.compute_stability_index()
        assert result["nu1"] is not None
        assert result["broucke"] is not None


class TestFullAnalysis:
    def test_full_analysis_calls_all_steps(self):
        orbit = _make_orbit(period=2.0)
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        orbit.system = system
        analysis = StabilityAnalysis(orbit)

        identity = np.eye(6)
        analysis.compute_monodromy = lambda: (
            _inject_monodromy(analysis, identity),
            _inject_eigenvalues(
                analysis,
                [1.0, 1.0, np.exp(0.3j), np.exp(-0.3j), np.exp(0.7j), np.exp(-0.7j)],
            ),
        )[0]

        result = analysis.full_analysis()
        assert "monodromy_matrix" in result
        assert "classification" in result
        assert "bifurcation" in result


class TestStringRepr:
    def test_str_before_analysis(self):
        orbit = _make_orbit()
        s = str(StabilityAnalysis(orbit))
        assert "未分析" in s

    def test_repr(self):
        orbit = _make_orbit()
        r = repr(StabilityAnalysis(orbit))
        assert "StabilityAnalysis" in r
        assert "complete=" in r


class TestDetectBifurcationInFamily:
    def test_returns_empty_for_no_bifurcation(self):
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)
        orbit = _make_orbit(n=5, period=2.0, system=system)

        results = StabilityAnalysis.detect_bifurcation_in_family([orbit], dynamics, tolerance=1e-8)
        assert isinstance(results, list)

    def test_handles_exception_gracefully(self):
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)
        orbit = _make_orbit(n=5, period=None, system=system)

        results = StabilityAnalysis.detect_bifurcation_in_family([orbit], dynamics, tolerance=1e-8)
        assert isinstance(results, list)


class TestFindNearestBifurcation:
    def test_returns_none_when_no_bifurcation(self):
        system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
        dynamics = CR3BP_Dynamics(system)
        orbit = _make_orbit(n=5, period=2.0, system=system)

        result = StabilityAnalysis.find_nearest_bifurcation([orbit], dynamics, tolerance=1e-8)
        assert result is None
