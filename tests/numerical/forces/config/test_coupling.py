"""耦合项（coupling=1）解析退化测试（ADR 0013）。"""

from __future__ import annotations

import pytest

from e2m2e.algorithm.forces.force_mapping import perturbation_to_force_config

pytestmark = pytest.mark.force


class TestCouplingMapping:
    """coupling 开关到力模型配置的映射。"""

    def test_coupling_on_enables_solid_tide(self):
        """coupling=1 强制 tide_mode='solid'。"""
        cfg = perturbation_to_force_config(
            {"coupling": 1, "earth_nonspherical": 1},
            earth_degree=4,
        )
        gf = next(
            f
            for f in cfg["forces"]
            if f["type"] == "GravityField" and f["params"]["body"] == "EARTH"
        )
        assert gf["params"]["tide_mode"] == "solid"

    def test_coupling_off_no_solid_tide(self):
        """coupling=0 + tide=0 → tide_mode='none'。"""
        cfg = perturbation_to_force_config(
            {"coupling": 0, "tide": 0, "earth_nonspherical": 1},
            earth_degree=4,
        )
        gf = next(
            f
            for f in cfg["forces"]
            if f["type"] == "GravityField" and f["params"]["body"] == "EARTH"
        )
        assert gf["params"]["tide_mode"] == "none"

    def test_coupling_forces_tide_even_when_tide_off(self):
        """coupling=1 + tide=0 → tide_mode='solid'（耦合项独立于 tide 开关）。"""
        cfg = perturbation_to_force_config(
            {"coupling": 1, "tide": 0, "earth_nonspherical": 1},
            earth_degree=4,
        )
        gf = next(
            f
            for f in cfg["forces"]
            if f["type"] == "GravityField" and f["params"]["body"] == "EARTH"
        )
        assert gf["params"]["tide_mode"] == "solid"

    def test_default_perturbation_no_error(self):
        """默认配置（coupling=1, tide=1）不再抛 NotImplementedError。"""
        from e2m2e.data.templates.perturbations import DEFAULT_PERTURBATION

        # 应正常返回，不抛异常
        cfg = perturbation_to_force_config(DEFAULT_PERTURBATION, earth_degree=4)
        assert "forces" in cfg
        assert len(cfg["forces"]) > 0

    def test_coupling_requires_earth_nonspherical(self):
        """coupling=1 + earth_nonspherical=0 应抛 ValueError。"""
        with pytest.raises(ValueError, match="earth_nonspherical"):
            perturbation_to_force_config({"coupling": 1, "earth_nonspherical": 0})
