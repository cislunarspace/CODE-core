"""摄动开关 → e2m2e 力模型配置映射表。

inputs-dac.txt（第 9~17 行 + 阶次/DYB 行）的力模型是"地球+月球质点
引力常开 + 一组摄动开关"；e2m2e 侧是 ``ForceModel`` 聚合若干
``PhysicalModel``（ADR 0004 配置驱动）。本模块给出两者的逐项对应，产出
``ForceModel.from_config`` 可直接消费的配置字典，保证"同款力模型"两侧
可复现。

对应关系（地心 GCRS 传播）如下：

- 基础模型（常开）：地球质点 ``PointMassGravity(EARTH)``（状态以地心为
  原点，中心项模型适用）；月球质点 ``ThirdBodyGravity(MOON)``——地心
  传播下航天器状态不以月心为原点，``PointMassGravity(MOON)`` 会把月心
  引力错算成朝向地心，必须用带星历直接项的 ``ThirdBodyGravity``（自带
  间接项，无需另补）。
- ``earth_nonspherical=1``：地球中心项换成 ``GravityField(EARTH,
  degree=order=earth_degree)``（球谐含 degree=0 中心项，不重复计质点）。
- ``moon_nonspherical=1``：月球中心项换成 ``GravityField(MOON,
  degree=order=moon_degree)`` + ``IndirectTerm(MOON)``（球谐只算直接
  引力，地心加速系需单补间接项）。
- ``sun_body=1``：``ThirdBodyGravity(SUN)``。
- ``planets=1``：七大行星（水星~海王星）各一个 ``ThirdBodyGravity``。
- ``solar_radiation=1``（炮弹模型）：``SolarRadiationPressure``，
  ``area=等效面质比, mass=1, cr=1``——输入侧"等效面质比"（dyb[0]）已把
  Cr 折进去，故 cr 取 1；无阴影模型（阴影行为未确认）。
- ``solar_radiation=2``（ECOM）：未实现，``NotImplementedError``（#253）。
- ``atmosphere=1``：``DragModel``（ExponentialAtmosphere 默认 f107/ap，
  cd=2.2，面积同取等效面质比；输入侧 Cd/大气模型参数不可见）。
- ``relativity=1``：``RelativisticCorrection(EARTH)``，仅 Schwarzschild
  主项（修正项构成未确认，待 P0 对齐实验核实）。
- ``tide=1``：地球固体潮，挂在地球 ``GravityField`` 的
  ``tide_mode="solid"`` 上——因此要求 ``earth_nonspherical=1``，
  否则抛 ``ValueError``。月球引力场不带潮（开关写明"地球的潮汐"）。
- ``coupling=1``（地球非球形×大天体耦合项）：强制启用固体潮
  ``tide_mode="solid"``（与 ``tide=1`` 共用 IERS TN32 固体潮公式）。

``output_step`` 不是力模型参数，不进配置；它是传播输出网格，由调用方
（阶段 3 传播对齐）用于构造 ``t_eval``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...data.templates.perturbations import DEFAULT_DYB, DEFAULT_PERTURBATION

__all__ = ["PLANET_BODIES", "perturbation_to_force_config"]

#: "大行星的第三体引力"对应的摄动天体（地球除外，月球有独立开关）
PLANET_BODIES: tuple[str, ...] = (
    "MERCURY",
    "VENUS",
    "MARS",
    "JUPITER",
    "SATURN",
    "URANUS",
    "NEPTUNE",
)

#: 各天体 GravityField 的默认固连系（与 gravity_field._DEFAULT_FRAME_BY_BODY 一致）
_DEFAULT_FRAME_BY_BODY = {"EARTH": "ITRF93", "MOON": "MOON_PA"}


def _resolve_switches(perturbation: dict[str, int] | None) -> dict[str, int]:
    """合并默认值并校验开关取值。"""
    vals = dict(DEFAULT_PERTURBATION)
    if perturbation:
        unknown = set(perturbation) - set(DEFAULT_PERTURBATION)
        if unknown:
            raise ValueError(f"未知摄动开关字段: {sorted(unknown)}")
        vals.update(perturbation)
    for key, v in vals.items():
        allowed = (0, 1, 2) if key == "solar_radiation" else (0, 1)
        if v not in allowed:
            raise ValueError(f"摄动开关 {key} 取值必须为 {allowed}，当前 {v!r}")
    return vals


def _unique_names(types: list[str]) -> list[str]:
    """按 ``ForceModel._auto_name`` 惯例生成条目名：类名 + ``_2``/``_3`` 消歧。"""
    names: list[str] = []
    for t in types:
        n = 1
        while True:
            candidate = t if n == 1 else f"{t}_{n}"
            if candidate not in names:
                names.append(candidate)
                break
            n += 1
    return names


def perturbation_to_force_config(
    perturbation: dict[str, int] | None = None,
    *,
    earth_degree: int = 10,
    moon_degree: int = 10,
    dyb: Sequence[float] | None = None,
    area_to_mass: float | None = None,
) -> dict[str, Any]:
    """把摄动开关映射为 e2m2e ``ForceModel`` 配置字典。

    Args:
        perturbation: 摄动开关字典（键与取值同 ``inputs_dac`` 的
            ``DEFAULT_PERTURBATION``）；缺省项取默认值。
        earth_degree: 地球非球形引力位阶次数（degree=order）。
        moon_degree: 月球非球形引力位阶次数（degree=order）。
        dyb: DYB 面质比系数 9 分量；``dyb[0]`` 为等效面质比（m²/kg），
            炮弹光压与大气阻力共用；其余分量在炮弹档忽略。
        area_to_mass: 显式等效面质比（m²/kg），给出时覆盖 ``dyb[0]``。

    Returns:
        ``{"version": 1, "forces": [...]}`` 配置字典，可直接交给
        ``ForceModel.from_config(config, system)`` 或
        ``dump_force_config`` 链路。

    Raises:
        NotImplementedError: ``solar_radiation=2``（ECOM，#253）。
        ValueError: 开关取值非法；``tide=1`` 或 ``coupling=1`` 而
            ``earth_nonspherical=0``；``dyb`` 非 9 分量。
    """
    sw = _resolve_switches(perturbation)

    # coupling=1 强制启用固体潮（无论 tide 开关值）。
    # 公式来源：IERS TN32 eqn 1 p.59 / Vallado 2022 Eq.8-48
    # 物理含义：大天体引力使地球弹性形变 → 修正球谐系数 C/S → 额外加速度
    if sw["coupling"] == 1 and sw["earth_nonspherical"] == 0:
        raise ValueError("coupling=1 需要 earth_nonspherical=1：耦合项挂在地球 GravityField 上")
    effective_tide = sw["tide"] == 1 or sw["coupling"] == 1
    if sw["tide"] == 1 and sw["earth_nonspherical"] == 0:
        raise ValueError("tide=1 需要 earth_nonspherical=1：e2m2e 的固体潮挂在地球 GravityField 上")

    if dyb is not None and len(dyb) != 9:
        raise ValueError(f"dyb 必须为 9 个分量，当前 {len(dyb)} 个")
    a2m = float(area_to_mass) if area_to_mass is not None else float((dyb or DEFAULT_DYB)[0])

    forces: list[dict[str, Any]] = []

    def add(type_name: str, params: dict[str, Any]) -> None:
        forces.append({"type": type_name, "params": params})

    # --- 基础模型：地球 + 月球中心项（质点或球谐），月球间接项常开 ---
    if sw["earth_nonspherical"] == 1:
        add(
            "GravityField",
            {
                "body": "EARTH",
                "degree": int(earth_degree),
                "order": int(earth_degree),
                "input_frame": _DEFAULT_FRAME_BY_BODY["EARTH"],
                "gravity_file": None,
                "tide_mode": "solid" if effective_tide else "none",
                "tide_convention": "tide_free",
            },
        )
    else:
        add("PointMassGravity", {"body": "EARTH", "mu": None})

    if sw["moon_nonspherical"] == 1:
        add(
            "GravityField",
            {
                "body": "MOON",
                "degree": int(moon_degree),
                "order": int(moon_degree),
                "input_frame": _DEFAULT_FRAME_BY_BODY["MOON"],
                "gravity_file": None,
                "tide_mode": "none",
                "tide_convention": "tide_free",
            },
        )
        # GravityField 只算球谐直接引力，地心加速系需单独补月球间接项
        add("IndirectTerm", {"body": "MOON", "mu": None})
    else:
        # 月球质点：PointMassGravity 假设状态以 body 为原点，不能用于
        # 地心传播下的月球；ThirdBodyGravity 自带直接项（用星历月位）
        # 与间接项，闭式正确
        add("ThirdBodyGravity", {"body": "MOON", "mu": None})

    # --- 第三体摄动 ---
    if sw["sun_body"] == 1:
        add("ThirdBodyGravity", {"body": "SUN", "mu": None})
    if sw["planets"] == 1:
        for body in PLANET_BODIES:
            add("ThirdBodyGravity", {"body": body, "mu": None})

    # --- 非引力摄动 ---
    if sw["solar_radiation"] == 1:
        add(
            "SolarRadiationPressure",
            {"area": a2m, "mass": 1.0, "cr": 1.0, "shadow": None},
        )
    if sw["solar_radiation"] == 2:
        dyb_full = list(dyb) if dyb is not None else list(DEFAULT_DYB)
        add(
            "EcomSolarRadiationPressure",
            {"dyb": dyb_full, "shadow": None},
        )
    if sw["atmosphere"] == 1:
        add(
            "DragModel",
            {
                "body": "EARTH",
                "cd": 2.2,
                "area": a2m,
                "mass": 1.0,
                "atmosphere": {
                    "type": "ExponentialAtmosphere",
                    "params": {"f107": 150.0, "ap": 15.0},
                },
            },
        )
    if sw["relativity"] == 1:
        add(
            "RelativisticCorrection",
            {
                "central_body": "EARTH",
                "primary_body": "SUN",
                "enable_schwarzschild": True,
                "enable_lense_thirring": False,
                "enable_de_sitter": False,
                "angular_momentum_vector": None,
                "body_radius": None,
                "c": 299792.458,
                "gamma": 1.0,
            },
        )

    names = _unique_names([f["type"] for f in forces])
    return {
        "version": 1,
        "forces": [
            {"name": name, "type": f["type"], "enabled": True, "params": f["params"]}
            for name, f in zip(names, forces, strict=True)
        ],
    }
