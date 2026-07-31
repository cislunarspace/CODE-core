"""ForceModel 配置驱动：力模型 ↔ dict 序列化与 JSON IO。

设计见 ADR 0004。容器级编排（信封、version、entry 拼装）在
``ForceModel.to_config`` / ``from_config``；本模块只负责"单条力"的
类型分发与 JSON 文件读写。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from e2m2e.core.atmosphere import ExponentialAtmosphere

from .drag import DragModel
from .exceptions import NotSerializableError
from .gravity_field import GravityField
from .indirect_term import IndirectTerm
from .physical_model import PhysicalModel
from .point_mass_gravity import PointMassGravity
from .relativistic_correction import RelativisticCorrection
from .shadow import ConicalShadowModel
from .srp import SolarRadiationPressure
from .third_body_gravity import ThirdBodyGravity
from .thrust import FiniteBurn

# --- 嵌套依赖：大气模型 ---


def _serialize_atmosphere(atm: ExponentialAtmosphere) -> dict[str, Any]:
    """把大气模型序列化为 ``{type, params}`` 字典。"""
    if not isinstance(atm, ExponentialAtmosphere):
        raise NotSerializableError(f"atmosphere type {type(atm).__name__} has no config serializer")
    return {"type": "ExponentialAtmosphere", "params": {"f107": atm.f107, "ap": atm.ap}}


def _build_atmosphere(config: dict[str, Any]) -> ExponentialAtmosphere:
    """按配置字典构造大气模型。"""
    if config["type"] != "ExponentialAtmosphere":
        raise ValueError(f"unknown atmosphere type {config['type']!r}")
    return ExponentialAtmosphere(**config.get("params", {}))


# --- 嵌套依赖：阴影模型 ---


def _serialize_shadow(shadow: ConicalShadowModel) -> dict[str, Any]:
    """把阴影模型序列化为 ``{type, params}`` 字典。"""
    if not isinstance(shadow, ConicalShadowModel):
        raise NotSerializableError(f"shadow type {type(shadow).__name__} has no config serializer")
    return {
        "type": "ConicalShadowModel",
        "params": {"bodies": list(shadow.bodies), "radii": shadow.radii},
    }


def _build_shadow(config: dict[str, Any]) -> ConicalShadowModel:
    """按配置字典构造阴影模型。"""
    if config["type"] != "ConicalShadowModel":
        raise ValueError(f"unknown shadow type {config['type']!r}")
    return ConicalShadowModel(**config.get("params", {}))


# --- 单力模型序列化器：type(实例) -> params dict ---


def _serialize_gravity_field(force: GravityField) -> dict[str, Any]:
    """把球谐重力场力模型序列化为参数字典。"""
    return {
        "body": force.body,
        "degree": force.degree,
        "order": force.order,
        "input_frame": force.input_frame,
        "gravity_file": str(force.gravity_file) if force.gravity_file is not None else None,
        "tide_mode": force.tide_mode,
        "tide_convention": force.tide_convention,
    }


def _serialize_drag_model(force: DragModel) -> dict[str, Any]:
    """把大气阻力力模型序列化为参数字典。"""
    return {
        "body": force.body,
        "cd": force.cd,
        "area": force.area,
        "mass": force.mass,
        "atmosphere": _serialize_atmosphere(force.atmosphere),
    }


def _serialize_srp(force: SolarRadiationPressure) -> dict[str, Any]:
    """把太阳光压力模型序列化为参数字典。"""
    return {
        "area": force.area,
        "mass": force.mass,
        "cr": force.cr,
        "shadow": _serialize_shadow(force.shadow) if force.shadow is not None else None,
    }


# --- FiniteBurn DSL：callable 参数的封闭声明式表达 ---
#
# thrust_profile / direction 可能是任意 Python callable，无法通用序列化。
# DSL 只支持几种固定写法；from_config 构造的闭包打 _e2m2e_config_kind 标记，
# to_config 据此反向。用户手写 lambda 无标记 → NotSerializableError。


def _build_thrust_profile(config: dict[str, Any]) -> Callable[[float], float]:
    """按配置字典构造推力剖面可调用对象。"""
    kind = config["kind"]
    if kind == "constant":
        thrust = float(config["thrust"])

        def constant_profile(t: float, _thrust: float = thrust) -> float:
            return _thrust

        cast(Any, constant_profile)._e2m2e_config_kind = (
            "constant",
            {"kind": "constant", "thrust": thrust},
        )
        return constant_profile
    if kind == "pulse":
        t_start = float(config["t_start"])
        t_end = float(config["t_end"])
        thrust = float(config["thrust"])

        def pulse_profile(
            t: float,
            _t_start: float = t_start,
            _t_end: float = t_end,
            _thrust: float = thrust,
        ) -> float:
            return _thrust if _t_start <= t <= _t_end else 0.0

        cast(Any, pulse_profile)._e2m2e_config_kind = (
            "pulse",
            {"kind": "pulse", "t_start": t_start, "t_end": t_end, "thrust": thrust},
        )
        return pulse_profile
    raise ValueError(f"unknown thrust_profile kind {kind!r}")


def _serialize_thrust_profile(profile: Callable[[float], float]) -> dict[str, Any]:
    """把推力剖面可调用对象序列化为配置字典。"""
    kind_info = getattr(profile, "_e2m2e_config_kind", None)
    if kind_info is None:
        raise NotSerializableError(
            "FiniteBurn.thrust_profile is not config-serializable "
            "(only DSL constant/pulse profiles built via from_config round-trip)"
        )
    _kind, recorded_config = kind_info
    return recorded_config


def _build_direction(config: dict[str, Any]) -> Any:
    """按配置字典构造推力方向。"""
    kind = config["kind"]
    if kind == "fixed":
        return [float(v) for v in config["vector"]]
    raise ValueError(f"unknown direction kind {kind!r}")


def _serialize_direction(direction: Any) -> dict[str, Any]:
    """把推力方向序列化为配置字典。"""
    # 固定向量：按类型识别（FiniteBurn 把固定向量存成 ndarray 或 list）。
    if callable(direction):
        kind_info = getattr(direction, "_e2m2e_config_kind", None)
        if kind_info is None:
            raise NotSerializableError(
                "FiniteBurn.direction callable is not config-serializable "
                "(only DSL fixed direction round-trips)"
            )
        _kind, recorded_config = kind_info
        return recorded_config
    arr = np.asarray(direction, dtype=float)
    return {"kind": "fixed", "vector": arr.tolist()}


def _serialize_finite_burn(force: FiniteBurn) -> dict[str, Any]:
    """把连续推力力模型序列化为参数字典。"""
    params: dict[str, Any] = {
        "mass": force.mass,
        "thrust_profile": _serialize_thrust_profile(force.thrust_profile),
        "direction": _serialize_direction(force.direction),
    }
    if force.direction_frame is not None:
        params["direction_frame"] = force.direction_frame
    return params


def _serialize_relativistic_correction(force: RelativisticCorrection) -> dict[str, Any]:
    """把相对论修正力模型序列化为参数字典。"""
    return {
        "central_body": force.central_body,
        "primary_body": force.primary_body,
        "enable_schwarzschild": force.enable_schwarzschild,
        "enable_lense_thirring": force.enable_lense_thirring,
        "enable_de_sitter": force.enable_de_sitter,
        "angular_momentum_vector": (
            force.angular_momentum_vector.tolist()
            if force.angular_momentum_vector is not None
            else None
        ),
        "body_radius": force.body_radius,
        "c": force.c,
        "gamma": force.gamma,
    }


def _serialize_point_mass_gravity(force: PointMassGravity) -> dict[str, Any]:
    """把点质量引力力模型序列化为参数字典。"""
    return {"body": force.body, "mu": force.mu}


def _serialize_third_body_gravity(force: ThirdBodyGravity) -> dict[str, Any]:
    """把第三体引力力模型序列化为参数字典。"""
    return {"body": force.body, "mu": force.mu}


def _serialize_indirect_term(force: IndirectTerm) -> dict[str, Any]:
    """把第三体间接项力模型序列化为参数字典。"""
    return {"body": force.body, "mu": force.mu}


_SERIALIZERS: dict[type, Any] = {
    GravityField: _serialize_gravity_field,
    DragModel: _serialize_drag_model,
    SolarRadiationPressure: _serialize_srp,
    FiniteBurn: _serialize_finite_burn,
    RelativisticCorrection: _serialize_relativistic_correction,
    PointMassGravity: _serialize_point_mass_gravity,
    ThirdBodyGravity: _serialize_third_body_gravity,
    IndirectTerm: _serialize_indirect_term,
}


def serialize_force(force: PhysicalModel) -> dict[str, Any]:
    """把单条力序列化为 ``{type, params}``；未知类型抛 ``NotSerializableError``。"""
    serializer = _SERIALIZERS.get(type(force))
    if serializer is None:
        raise NotSerializableError(f"force type {type(force).__name__} has no config serializer")
    return {"type": type(force).__name__, "params": serializer(force)}


# --- 单力模型构造器：type 名 -> PhysicalModel ---


def _build_gravity_field(params: dict[str, Any]) -> GravityField:
    """从参数字典构造球谐重力场力模型。"""
    return GravityField(**params)


def _build_drag_model(params: dict[str, Any]) -> DragModel:
    """从参数字典构造大气阻力力模型。"""
    built = dict(params)
    built["atmosphere"] = _build_atmosphere(built["atmosphere"])
    return DragModel(**built)


def _build_srp(params: dict[str, Any]) -> SolarRadiationPressure:
    """从参数字典构造太阳光压力模型。"""
    built = dict(params)
    if built.get("shadow") is not None:
        built["shadow"] = _build_shadow(built["shadow"])
    return SolarRadiationPressure(**built)


def _build_finite_burn(params: dict[str, Any]) -> FiniteBurn:
    """从参数字典构造连续推力力模型。"""
    built = dict(params)
    built["thrust_profile"] = _build_thrust_profile(built["thrust_profile"])
    built["direction"] = _build_direction(built["direction"])
    return FiniteBurn(**built)


def _build_relativistic_correction(params: dict[str, Any]) -> RelativisticCorrection:
    """从参数字典构造相对论修正力模型。"""
    return RelativisticCorrection(**params)


def _build_point_mass_gravity(params: dict[str, Any]) -> PointMassGravity:
    """从参数字典构造点质量引力力模型。"""
    return PointMassGravity(**params)


def _build_third_body_gravity(params: dict[str, Any]) -> ThirdBodyGravity:
    """从参数字典构造第三体引力力模型。"""
    return ThirdBodyGravity(**params)


def _build_indirect_term(params: dict[str, Any]) -> IndirectTerm:
    """从参数字典构造第三体间接项力模型。"""
    return IndirectTerm(**params)


_BUILDERS: dict[str, Any] = {
    "GravityField": _build_gravity_field,
    "DragModel": _build_drag_model,
    "SolarRadiationPressure": _build_srp,
    "FiniteBurn": _build_finite_burn,
    "RelativisticCorrection": _build_relativistic_correction,
    "PointMassGravity": _build_point_mass_gravity,
    "ThirdBodyGravity": _build_third_body_gravity,
    "IndirectTerm": _build_indirect_term,
}


def build_force(type_name: str, params: dict[str, Any]) -> PhysicalModel:
    """按 type 名与 params 构造单条力；未知 type 抛 ``ValueError``。"""
    builder = _BUILDERS.get(type_name)
    if builder is None:
        raise ValueError(f"unknown force type {type_name!r}; known types: {sorted(_BUILDERS)}")
    return builder(params)


# --- JSON 文件 IO ---


def dump_force_config(fm: Any, path: str | Path) -> None:
    """把 ``ForceModel.to_config()`` 的结果写入 JSON 文件。"""
    config = fm.to_config()
    Path(path).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_force_config(path: str | Path, system: Any) -> Any:
    """从 JSON 文件读取配置并构建 ``ForceModel``。"""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    from .force_model import ForceModel

    return ForceModel.from_config(config, system)
