"""data/types/orbit.py：Orbit/OrbitFamily 数据容器测试。"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from e2m2e.data.types.orbit import Orbit, OrbitFamily


def _sample_orbit(period: float = 1.0) -> Orbit:
    t = np.linspace(0, 1, 50)
    x = 0.8 + 0.1 * np.cos(2 * np.pi * t)
    y = 0.1 * np.sin(2 * np.pi * t)
    states = np.column_stack(
        [x, y, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)]
    )
    orbit = Orbit(states=states, times=t)
    orbit.period = period
    return orbit


class TestOrbit:
    def test_requires_six_components(self):
        with pytest.raises(ValueError, match="必须包含6个分量"):
            Orbit(states=np.zeros((5, 5)), times=np.zeros(5))

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="长度必须与状态序列长度一致"):
            Orbit(states=np.zeros((5, 6)), times=np.zeros(4))

    def test_compute_basic_properties(self):
        orbit = _sample_orbit()
        assert orbit.amplitudes["x"] > 0
        assert "x_max" in orbit.extrema
        assert orbit.center is not None and orbit.center.shape == (3,)

    def test_save_load_roundtrip(self, tmp_path):
        orbit = _sample_orbit(period=2.0)
        orbit.family_type = "halo"
        path = tmp_path / "orbit.json"
        orbit.save_to_file(path)
        loaded = Orbit.load_from_file(path)
        np.testing.assert_allclose(loaded.states, orbit.states, atol=1e-12)
        assert loaded.period == 2.0
        assert loaded.family_type == "halo"

    def test_state0(self):
        orbit = _sample_orbit()
        np.testing.assert_allclose(orbit.state0, orbit.states[0])


class TestOrbitFamily:
    def test_accepts_single_orbit(self):
        family = OrbitFamily(orbits=_sample_orbit())
        assert len(family) == 1

    def test_rejects_non_orbit(self):
        with pytest.raises(TypeError, match="Orbit instances"):
            OrbitFamily(orbits=[1, 2, 3])

    def test_periods_and_states(self):
        family = OrbitFamily(family_type="halo")
        for i in range(3):
            family.add_orbit(_sample_orbit(period=1.0 + 0.5 * i))
        np.testing.assert_allclose(family.get_periods(), [1.0, 1.5, 2.0])
        assert family.get_states().shape == (3, 6)

    def test_jacobi_without_cr3bp_system_returns_empty(self):
        family = OrbitFamily()
        assert family.get_jacobi_constants().shape == (0,)

    def test_save_load_family(self, tmp_path):
        family = OrbitFamily(family_type="dro")
        family.add_orbit(_sample_orbit(period=1.0))
        path: Path = tmp_path / "family.json"
        family.save_to_file(path)
        loaded = OrbitFamily.load_from_file(path)
        assert loaded.family_type == "dro"
        assert len(loaded.orbits) == 1


def test_save_and_load_with_tempfile_named():
    """旧测试风格：NamedTemporaryFile 也能工作（shim 后行为一致）。"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filename = f.name
    try:
        orbit = _sample_orbit()
        orbit.save_to_file(filename)
        loaded = Orbit.load_from_file(filename)
        np.testing.assert_allclose(loaded.states, orbit.states, atol=1e-12)
    finally:
        Path(filename).unlink(missing_ok=True)
