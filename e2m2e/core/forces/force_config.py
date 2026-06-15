"""ForceModel 配置驱动：力模型 ↔ dict 序列化与 JSON IO。

设计见 ADR 0004。容器级编排（信封、version、entry 拼装）在
``ForceModel.to_config`` / ``from_config``；本模块只负责"单条力"的
类型分发与 JSON 文件读写。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from e2m2e.core.atmosphere import AtmosphereModel, ExponentialAtmosphere

from .drag import DragModel
from .gravity_field import GravityField
from .physical_model import PhysicalModel
from .shadow import ConicalShadowModel, ShadowModel
from .srp import SolarRadiationPressure
from .thrust import FiniteBurn


class NotSerializableError(TypeError):
    """力模型无法序列化为配置（如含任意 Python callable）时抛出。"""

    pass


# --- 嵌套依赖：大气模型 ---

def _serialize_exponential_atmosphere(atm: ExponentialAtmosphere) -> dict[str, Any]:
    return {"f107": atm.f107, "ap": atm.ap}


_ATMOS_SERIALIZERS: dict[type, Any] = {
    ExponentialAtmosphere: _serialize_exponential_atmosphere,
}


def _serialize_atmosphere(atm: AtmosphereModel) -> dict[str, Any]:
    serializer = _ATMOS_SERIALIZERS.get(type(atm))
    if serializer is None:
        raise NotSerializableError(
            f"atmosphere type {type(atm).__name__} has no config serializer"
        )
    return {"type": type(atm).__name__, "params": serializer(atm)}


def _build_atmosphere(config: dict[str, Any]) -> AtmosphereModel:
    builder = _ATMOS_BUILDERS.get(config["type"])
    if builder is None:
        raise ValueError(
            f"unknown atmosphere type {config['type']!r}; "
            f"known types: {sorted(_ATMOS_BUILDERS)}"
        )
    return builder(config.get("params", {}))


def _build_exponential_atmosphere(params: dict[str, Any]) -> ExponentialAtmosphere:
    return ExponentialAtmosphere(**params)


_ATMOS_BUILDERS: dict[str, Any] = {
    "ExponentialAtmosphere": _build_exponential_atmosphere,
}


# --- 嵌套依赖：阴影模型 ---

def _serialize_conical_shadow(shadow: ConicalShadowModel) -> dict[str, Any]:
    return {
        "bodies": list(shadow.bodies),
        "radii": shadow.radii,
    }


_SHADOW_SERIALIZERS: dict[type, Any] = {
    ConicalShadowModel: _serialize_conical_shadow,
}


def _serialize_shadow(shadow: ShadowModel) -> dict[str, Any]:
    serializer = _SHADOW_SERIALIZERS.get(type(shadow))
    if serializer is None:
        raise NotSerializableError(
            f"shadow type {type(shadow).__name__} has no config serializer"
        )
    return {"type": type(shadow).__name__, "params": serializer(shadow)}


def _build_conical_shadow(params: dict[str, Any]) -> ConicalShadowModel:
    return ConicalShadowModel(**params)


_SHADOW_BUILDERS: dict[str, Any] = {
    "ConicalShadowModel": _build_conical_shadow,
}


def _build_shadow(config: dict[str, Any]) -> ShadowModel:
    builder = _SHADOW_BUILDERS.get(config["type"])
    if builder is None:
        raise ValueError(
            f"unknown shadow type {config['type']!r}; "
            f"known types: {sorted(_SHADOW_BUILDERS)}"
        )
    return builder(config.get("params", {}))


# --- 单力模型序列化器：type(实例) -> params dict ---

def _serialize_gravity_field(force: GravityField) -> dict[str, Any]:
    return {
        "body": force.body,
        "degree": force.degree,
        "order": force.order,
        "input_frame": force.input_frame,
        "gravity_file": str(force.gravity_file) if force.gravity_file is not None else None,
    }


def _serialize_drag_model(force: DragModel) -> dict[str, Any]:
    return {
        "body": force.body,
        "cd": force.cd,
        "area": force.area,
        "mass": force.mass,
        "atmosphere": _serialize_atmosphere(force.atmosphere),
    }


def _serialize_srp(force: SolarRadiationPressure) -> dict[str, Any]:
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
    kind = config["kind"]
    if kind == "constant":
        thrust = float(config["thrust"])

        def profile(t: float, _thrust: float = thrust) -> float:
            return _thrust

        profile._e2m2e_config_kind = ("constant", {"kind": "constant", "thrust": thrust})
        return profile
    if kind == "pulse":
        t_start = float(config["t_start"])
        t_end = float(config["t_end"])
        thrust = float(config["thrust"])

        def profile(
            t: float,
            _t_start: float = t_start,
            _t_end: float = t_end,
            _thrust: float = thrust,
        ) -> float:
            return _thrust if _t_start <= t <= _t_end else 0.0

        profile._e2m2e_config_kind = (
            "pulse",
            {"kind": "pulse", "t_start": t_start, "t_end": t_end, "thrust": thrust},
        )
        return profile
    raise ValueError(f"unknown thrust_profile kind {kind!r}")


def _serialize_thrust_profile(profile: Callable[[float], float]) -> dict[str, Any]:
    kind_info = getattr(profile, "_e2m2e_config_kind", None)
    if kind_info is None:
        raise NotSerializableError(
            "FiniteBurn.thrust_profile is not config-serializable "
            "(only DSL constant/pulse profiles built via from_config round-trip)"
        )
    _kind, recorded_config = kind_info
    return recorded_config


def _build_direction(config: dict[str, Any]) -> Any:
    kind = config["kind"]
    if kind == "fixed":
        return [float(v) for v in config["vector"]]
    raise ValueError(f"unknown direction kind {kind!r}")


def _serialize_direction(direction: Any) -> dict[str, Any]:
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
    return {
        "mass": force.mass,
        "thrust_profile": _serialize_thrust_profile(force.thrust_profile),
        "direction": _serialize_direction(force.direction),
    }


_SERIALIZERS: dict[type, Any] = {
    GravityField: _serialize_gravity_field,
    DragModel: _serialize_drag_model,
    SolarRadiationPressure: _serialize_srp,
    FiniteBurn: _serialize_finite_burn,
}


def serialize_force(force: PhysicalModel) -> dict[str, Any]:
    """把单条力序列化为 ``{type, params}``；未知类型抛 ``NotSerializableError``。"""
    serializer = _SERIALIZERS.get(type(force))
    if serializer is None:
        raise NotSerializableError(
            f"force type {type(force).__name__} has no config serializer"
        )
    return {"type": type(force).__name__, "params": serializer(force)}


# --- 单力模型构造器：type 名 -> PhysicalModel ---

def _build_gravity_field(params: dict[str, Any]) -> GravityField:
    return GravityField(**params)


def _build_drag_model(params: dict[str, Any]) -> DragModel:
    built = dict(params)
    built["atmosphere"] = _build_atmosphere(built["atmosphere"])
    return DragModel(**built)


def _build_srp(params: dict[str, Any]) -> SolarRadiationPressure:
    built = dict(params)
    if built.get("shadow") is not None:
        built["shadow"] = _build_shadow(built["shadow"])
    return SolarRadiationPressure(**built)


def _build_finite_burn(params: dict[str, Any]) -> FiniteBurn:
    built = dict(params)
    built["thrust_profile"] = _build_thrust_profile(built["thrust_profile"])
    built["direction"] = _build_direction(built["direction"])
    return FiniteBurn(**built)


_BUILDERS: dict[str, Any] = {
    "GravityField": _build_gravity_field,
    "DragModel": _build_drag_model,
    "SolarRadiationPressure": _build_srp,
    "FiniteBurn": _build_finite_burn,
}


def build_force(type_name: str, params: dict[str, Any]) -> PhysicalModel:
    """按 type 名与 params 构造单条力；未知 type 抛 ``ValueError``。"""
    builder = _BUILDERS.get(type_name)
    if builder is None:
        raise ValueError(
            f"unknown force type {type_name!r}; known types: {sorted(_BUILDERS)}"
        )
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
