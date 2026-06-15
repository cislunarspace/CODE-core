"""ForceModel 配置驱动测试：from_config / to_config / JSON IO。"""

import numpy as np
import pytest

from e2m2e.core.atmosphere import ExponentialAtmosphere
from e2m2e.core.forces import (
    ConicalShadowModel,
    DragModel,
    FiniteBurn,
    ForceModel,
    GravityField,
    SolarRadiationPressure,
)
from e2m2e.core.forces.force_config import (
    NotSerializableError,
    dump_force_config,
    load_force_config,
)


class _FakeSystem:
    """仅用于配置测试的最小 System 桩（力构造不需 system，仅 ForceModel 需要）。"""

    coordinate_system = object()


def test_gravity_field_round_trip():
    """单个 GravityField 的 to_config → from_config → to_config 字典相等。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(GravityField("EARTH", degree=2, order=0), name="j2")

    config = ForceModel.to_config(fm)
    fm2 = ForceModel.from_config(config, system)

    # 形状：信封 + 单条 entry
    assert config["version"] == 1
    assert len(config["forces"]) == 1
    entry = config["forces"][0]
    assert entry["name"] == "j2"
    assert entry["type"] == "GravityField"
    assert entry["enabled"] is True
    # round-trip 契约：config 字典相等
    assert ForceModel.to_config(fm2) == config


def test_srp_with_shadow_round_trip():
    """SolarRadiationPressure 含嵌套 ConicalShadowModel 的 round-trip。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    shadow = ConicalShadowModel(bodies=["EARTH", "MOON"])
    fm.add_force(
        SolarRadiationPressure(area=5.0, mass=1000.0, cr=1.5, shadow=shadow),
        name="srp",
    )

    config = ForceModel.to_config(fm)
    fm2 = ForceModel.from_config(config, system)

    entry = config["forces"][0]
    assert entry["type"] == "SolarRadiationPressure"
    assert entry["params"]["shadow"] == {
        "type": "ConicalShadowModel",
        "params": {"bodies": ["EARTH", "MOON"], "radii": None},
    }
    assert ForceModel.to_config(fm2) == config


def test_srp_without_shadow_round_trip():
    """SRP 无阴影模型时 shadow 序列化为 null，round-trip 一致。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(
        SolarRadiationPressure(area=5.0, mass=1000.0, cr=1.5, shadow=None),
        name="srp",
    )

    config = ForceModel.to_config(fm)
    fm2 = ForceModel.from_config(config, system)

    assert config["forces"][0]["params"]["shadow"] is None
    assert ForceModel.to_config(fm2) == config


def test_finite_burn_constant_fixed_round_trip():
    """FiniteBurn 恒推力 + 固定方向的 from_config → to_config round-trip。"""
    system = _FakeSystem()
    config_dict = {
        "version": 1,
        "forces": [
            {
                "name": "engine",
                "type": "FiniteBurn",
                "enabled": True,
                "params": {
                    "mass": 1000.0,
                    "thrust_profile": {"kind": "constant", "thrust": 10.0},
                    "direction": {"kind": "fixed", "vector": [1.0, 0.0, 0.0]},
                },
            }
        ],
    }

    fm = ForceModel.from_config(config_dict, system)

    assert ForceModel.to_config(fm) == config_dict


def test_finite_burn_pulse_round_trip():
    """FiniteBurn 脉冲推力（on/off 区间）的 round-trip。"""
    system = _FakeSystem()
    config_dict = {
        "version": 1,
        "forces": [
            {
                "name": "engine",
                "type": "FiniteBurn",
                "enabled": True,
                "params": {
                    "mass": 1000.0,
                    "thrust_profile": {
                        "kind": "pulse",
                        "t_start": 100.0,
                        "t_end": 200.0,
                        "thrust": 10.0,
                    },
                    "direction": {"kind": "fixed", "vector": [0.0, 1.0, 0.0]},
                },
            }
        ],
    }

    fm = ForceModel.from_config(config_dict, system)

    assert ForceModel.to_config(fm) == config_dict


def test_finite_burn_built_force_computes_correct_acceleration():
    """DSL 构造的 FiniteBurn 物理行为正确（恒推力 10N / 1000kg → 1e-5 km/s²）。"""
    system = _FakeSystem()
    config_dict = {
        "version": 1,
        "forces": [
            {
                "name": "engine",
                "type": "FiniteBurn",
                "enabled": True,
                "params": {
                    "mass": 1000.0,
                    "thrust_profile": {"kind": "constant", "thrust": 10.0},
                    "direction": {"kind": "fixed", "vector": [1.0, 0.0, 0.0]},
                },
            }
        ],
    }

    fm = ForceModel.from_config(config_dict, system)
    engine = fm.get_force("engine")

    acc = engine.compute_acceleration(0.0, np.zeros(6), system)
    np.testing.assert_allclose(acc, [1e-5, 0.0, 0.0])


def test_finite_burn_unknown_callable_not_serializable():
    """用户手写 lambda 构造的 FiniteBurn 无法 to_config，抛 NotSerializableError。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(
        FiniteBurn(
            thrust_profile=lambda t: 5.0,
            direction=[1.0, 0.0, 0.0],
            mass=1000.0,
        ),
        name="engine",
    )

    with pytest.raises(NotSerializableError):
        ForceModel.to_config(fm)


def test_unknown_force_type_raises():
    """from_config 遇未知 type 抛 ValueError，列出已知类型。"""
    system = _FakeSystem()
    config = {
        "version": 1,
        "forces": [{"name": "x", "type": "BogusForce", "enabled": True, "params": {}}],
    }
    with pytest.raises(ValueError, match="unknown force type"):
        ForceModel.from_config(config, system)


def test_unsupported_version_raises():
    """from_config 遇不支持 version 抛 ValueError。"""
    system = _FakeSystem()
    config = {"version": 99, "forces": []}
    with pytest.raises(ValueError, match="version"):
        ForceModel.from_config(config, system)


def test_disabled_force_persisted_in_config():
    """disable 后 to_config 输出 enabled:false，from_config 重建仍为 disabled。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
    fm.disable("j2")

    config = ForceModel.to_config(fm)
    assert config["forces"][0]["enabled"] is False

    fm2 = ForceModel.from_config(config, system)
    assert fm2.list_forces()[0].enabled is False


def test_full_leo_config_round_trip():
    """完整 LEO 配置（J2 + 阻力 + 光压）round-trip。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
    fm.add_force(
        DragModel(atmosphere=ExponentialAtmosphere(), area=5.0, mass=1000.0),
        name="drag",
    )
    fm.add_force(SolarRadiationPressure(area=5.0, mass=1000.0), name="srp")

    config = ForceModel.to_config(fm)
    fm2 = ForceModel.from_config(config, system)

    assert len(config["forces"]) == 3
    assert [e["name"] for e in config["forces"]] == ["j2", "drag", "srp"]
    assert ForceModel.to_config(fm2) == config


def test_json_io_round_trip(tmp_path):
    """dump_force_config 写 JSON，load_force_config 读回，配置一致。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    fm.add_force(GravityField("EARTH", degree=2, order=0), name="j2")
    fm.add_force(
        DragModel(atmosphere=ExponentialAtmosphere(), area=5.0, mass=1000.0),
        name="drag",
    )

    path = tmp_path / "forces.json"
    dump_force_config(fm, path)
    fm2 = load_force_config(path, system)

    assert ForceModel.to_config(fm2) == ForceModel.to_config(fm)



def test_drag_model_round_trip():
    """DragModel 含嵌套 atmosphere 的 round-trip。"""
    system = _FakeSystem()
    fm = ForceModel(system)
    atm = ExponentialAtmosphere(f107=120.0, ap=10.0)
    fm.add_force(
        DragModel(atmosphere=atm, area=5.0, mass=1000.0, cd=2.2, body="EARTH"),
        name="drag",
    )

    config = ForceModel.to_config(fm)
    fm2 = ForceModel.from_config(config, system)

    entry = config["forces"][0]
    assert entry["type"] == "DragModel"
    # atmosphere 递归 {type, params}
    assert entry["params"]["atmosphere"] == {
        "type": "ExponentialAtmosphere",
        "params": {"f107": 120.0, "ap": 10.0},
    }
    assert ForceModel.to_config(fm2) == config
