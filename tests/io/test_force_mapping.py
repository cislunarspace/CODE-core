"""DFH 摄动开关 → e2m2e 力模型配置映射测试。"""

import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.force_config import dump_force_config, load_force_config
from e2m2e.io import PLANET_BODIES, dfh_perturbation_to_force_config

_ALL_OFF = {
    "sun_body": 0,
    "planets": 0,
    "earth_nonspherical": 0,
    "moon_nonspherical": 0,
    "solar_radiation": 0,
    "atmosphere": 0,
    "relativity": 0,
    "tide": 0,
    "coupling": 0,
}


def _on(**overrides):
    return {**_ALL_OFF, **overrides}


def _types(config):
    return [f["type"] for f in config["forces"]]


def _entries_by_type(config, type_name):
    return [f for f in config["forces"] if f["type"] == type_name]


class _FakeSystem:
    """最小 System 桩（力构造不需 system，仅 ForceModel 需要）。"""

    coordinate_system = object()


class TestBaseModel:
    def test_all_off_is_earth_point_mass_plus_moon_third_body(self):
        cfg = dfh_perturbation_to_force_config(_ALL_OFF)
        assert _types(cfg) == ["PointMassGravity", "ThirdBodyGravity"]
        earth, moon = cfg["forces"]
        assert earth["params"] == {"body": "EARTH", "mu": None}
        assert moon["params"] == {"body": "MOON", "mu": None}
        # 地心传播下月球质点不能用 PointMassGravity（状态不以月心为原点）；
        # ThirdBodyGravity 自带间接项，不应再出现 IndirectTerm
        assert "IndirectTerm" not in _types(cfg)
        assert [f["name"] for f in cfg["forces"]] == ["PointMassGravity", "ThirdBodyGravity"]

    def test_default_perturbation_raises(self):
        # DEFAULT_PERTURBATION 含 coupling=1（未实现）
        with pytest.raises(NotImplementedError, match="#253"):
            dfh_perturbation_to_force_config()


class TestSingleSwitches:
    def test_sun_body(self):
        cfg = dfh_perturbation_to_force_config(_on(sun_body=1))
        tb = _entries_by_type(cfg, "ThirdBodyGravity")
        bodies = [f["params"]["body"] for f in tb]
        assert bodies == ["MOON", "SUN"]  # MOON 是基础模型的月球质点

    def test_planets(self):
        cfg = dfh_perturbation_to_force_config(_on(planets=1))
        tb = _entries_by_type(cfg, "ThirdBodyGravity")
        bodies = [f["params"]["body"] for f in tb]
        assert bodies == ["MOON", *PLANET_BODIES]

    def test_earth_nonspherical_replaces_point_mass(self):
        cfg = dfh_perturbation_to_force_config(_on(earth_nonspherical=1), earth_degree=4)
        assert "PointMassGravity" not in [
            f["type"] for f in cfg["forces"] if f["params"]["body"] == "EARTH"
        ]
        (gf,) = _entries_by_type(cfg, "GravityField")
        assert gf["params"]["body"] == "EARTH"
        assert gf["params"]["degree"] == 4
        assert gf["params"]["order"] == 4
        assert gf["params"]["input_frame"] == "ITRF93"
        assert gf["params"]["tide_mode"] == "none"

    def test_moon_nonspherical_keeps_indirect_term(self):
        cfg = dfh_perturbation_to_force_config(_on(moon_nonspherical=1), moon_degree=6)
        (gf,) = _entries_by_type(cfg, "GravityField")
        assert gf["params"]["body"] == "MOON"
        assert gf["params"]["degree"] == 6
        assert gf["params"]["input_frame"] == "MOON_PA"
        # 月球中心项由球谐承担，间接项保留；不得再出现月球的质点/第三体
        assert _entries_by_type(cfg, "IndirectTerm")[0]["params"]["body"] == "MOON"
        assert not [
            f
            for f in cfg["forces"]
            if f["type"] in ("PointMassGravity", "ThirdBodyGravity")
            and f["params"]["body"] == "MOON"
        ]

    def test_solar_radiation_cannonball_uses_dyb0(self):
        dyb = [0.02] + [0.0] * 8
        cfg = dfh_perturbation_to_force_config(_on(solar_radiation=1), dyb=dyb)
        (srp,) = _entries_by_type(cfg, "SolarRadiationPressure")
        # 等效面质比已含 Cr：cr=1、area/mass=dyb[0]、无阴影模型
        assert srp["params"] == {"area": 0.02, "mass": 1.0, "cr": 1.0, "shadow": None}

    def test_area_to_mass_override(self):
        dyb = [0.02] + [0.0] * 8
        cfg = dfh_perturbation_to_force_config(_on(solar_radiation=1), dyb=dyb, area_to_mass=0.03)
        (srp,) = _entries_by_type(cfg, "SolarRadiationPressure")
        assert srp["params"]["area"] == 0.03

    def test_atmosphere(self):
        cfg = dfh_perturbation_to_force_config(_on(atmosphere=1), area_to_mass=0.01)
        (drag,) = _entries_by_type(cfg, "DragModel")
        assert drag["params"]["body"] == "EARTH"
        assert drag["params"]["cd"] == 2.2
        assert drag["params"]["area"] == 0.01
        assert drag["params"]["atmosphere"] == {
            "type": "ExponentialAtmosphere",
            "params": {"f107": 150.0, "ap": 15.0},
        }

    def test_relativity_schwarzschild_only(self):
        cfg = dfh_perturbation_to_force_config(_on(relativity=1))
        (rel,) = _entries_by_type(cfg, "RelativisticCorrection")
        assert rel["params"]["central_body"] == "EARTH"
        assert rel["params"]["enable_schwarzschild"] is True
        assert rel["params"]["enable_lense_thirring"] is False
        assert rel["params"]["enable_de_sitter"] is False

    def test_tide_requires_earth_nonspherical(self):
        with pytest.raises(ValueError, match="earth_nonspherical"):
            dfh_perturbation_to_force_config(_on(tide=1))

    def test_tide_attaches_to_earth_gravity_field(self):
        cfg = dfh_perturbation_to_force_config(_on(earth_nonspherical=1, tide=1))
        (gf,) = [f for f in _entries_by_type(cfg, "GravityField") if f["params"]["body"] == "EARTH"]
        assert gf["params"]["tide_mode"] == "solid"


class TestUnsupported:
    def test_ecom_srp_produces_ecom_config(self):
        """solar_radiation=2 应产出 EcomSolarRadiationPressure 配置。"""
        dyb = [0.02] + [0.0] * 8
        cfg = dfh_perturbation_to_force_config(_on(solar_radiation=2), dyb=dyb)
        (ecom,) = _entries_by_type(cfg, "EcomSolarRadiationPressure")
        assert ecom["params"]["dyb"] == dyb
        assert ecom["params"]["shadow"] is None

    def test_ecom_srp_default_dyb(self):
        """dyb=None 时使用 DEFAULT_DYB。"""
        cfg = dfh_perturbation_to_force_config(_on(solar_radiation=2))
        (ecom,) = _entries_by_type(cfg, "EcomSolarRadiationPressure")
        assert ecom["params"]["dyb"] == [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_coupling_not_implemented(self):
        with pytest.raises(NotImplementedError, match="#253"):
            dfh_perturbation_to_force_config(_on(coupling=1))

    def test_invalid_switch_value(self):
        with pytest.raises(ValueError, match="solar_radiation"):
            dfh_perturbation_to_force_config(_on(solar_radiation=3))
        with pytest.raises(ValueError, match="sun_body"):
            dfh_perturbation_to_force_config(_on(sun_body=2))

    def test_unknown_switch_key(self):
        with pytest.raises(ValueError, match="未知摄动开关"):
            dfh_perturbation_to_force_config({"bogus": 1})

    def test_bad_dyb_length(self):
        with pytest.raises(ValueError, match="dyb"):
            dfh_perturbation_to_force_config(_on(solar_radiation=1), dyb=[0.01] * 8)


class TestBuildAndRoundTrip:
    FULL = _on(
        sun_body=1,
        planets=1,
        earth_nonspherical=1,
        moon_nonspherical=1,
        solar_radiation=1,
        atmosphere=1,
        relativity=1,
        tide=1,
    )

    def test_build_force_model_from_config(self):
        cfg = dfh_perturbation_to_force_config(self.FULL, earth_degree=4, moon_degree=4)
        fm = ForceModel.from_config(cfg, _FakeSystem())
        types = [type(f).__name__ for f in fm.forces]
        assert types == _types(cfg)
        # 阶次截断生效
        gf_earth = fm.get_force("GravityField")
        assert gf_earth.body == "EARTH"
        assert gf_earth.degree == 4
        assert gf_earth.order == 4
        assert gf_earth.coefficients["C"].shape == (5, 5)
        # 固体潮挂载
        assert gf_earth.tide_mode == "solid"

    def test_config_round_trip_dict_equal(self):
        cfg = dfh_perturbation_to_force_config(self.FULL)
        fm = ForceModel.from_config(cfg, _FakeSystem())
        assert fm.to_config() == cfg

    def test_json_dump_load_round_trip(self, tmp_path):
        cfg = dfh_perturbation_to_force_config(self.FULL)
        system = _FakeSystem()
        fm = ForceModel.from_config(cfg, system)
        path = tmp_path / "force_config.json"
        dump_force_config(fm, path)
        fm2 = load_force_config(path, system)
        assert fm2.to_config() == cfg
