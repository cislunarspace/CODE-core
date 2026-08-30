"""轨道库记录格式：schema 常量、段数组键约定、记录校验与段序列化。

一条 catalog 记录 = JSON 元数据 + 同名 NPZ 数组段（ADR 0031 决策 1）。
轨道数据双段并存：``cr3bp`` 段（Orbit 原生，无量纲 states/times）与
``eph`` 段（EphemerisTable：GCRS、会合系、times_et），各自可空。族记录
的成员数组在 ``cr3bp/members/`` 下按序号存放。站保等结果附带的小数组
（机动序列、统计表）在 ``result`` 段。

记录文件是事实来源；本模块只定义格式，存储引擎见 ``store.py``。
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ...exceptions import E2M2EError
from ..types.trajectory import EphemerisTable

__all__ = [
    "SCHEMA_VERSION",
    "CatalogError",
    "RecordNotFoundError",
    "CatalogFilter",
    "CatalogRecord",
    "new_record_id",
    "validate_record_id",
    "validate_meta",
    "member_array_key",
    "geometric_amplitude_km",
    "point_interval",
    "member_count",
    "cr3bp_segment_arrays",
    "ephemeris_segment_arrays",
    "transfer_segment_arrays",
    "numeric_or_none",
    "ephemeris_from_arrays",
]


class CatalogError(E2M2EError):
    """轨道库数据层错误（记录损坏、版本不兼容、成员缺失等）。"""


class RecordNotFoundError(CatalogError, KeyError):
    """按 record_id（或族成员序号）查找的记录不存在。"""


#: record_id 合法形态：字母数字开头，只含字母数字、点、下划线、连字符，
#: 不含路径分隔符与 ``..``（record_id 直接拼文件路径，必须防路径穿越）。
_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_record_id(record_id: str) -> str:
    """校验 record_id 是安全的文件名；非法形态按记录不存在处理。"""
    if not _RECORD_ID_RE.match(record_id) or ".." in record_id:
        raise RecordNotFoundError(f"记录不存在：{record_id!r}")
    return record_id


#: 记录 schema 版本号，自 1 起（ADR 0031 决策 1）；不兼容读取其他版本。
SCHEMA_VERSION = 1

#: NPZ 段前缀
CR3BP_PREFIX = "cr3bp"
EPHEMERIS_PREFIX = "eph"
RESULT_PREFIX = "result"
TRANSFER_PREFIX = "transfer"

_META_REQUIRED_KEYS = (
    "schema_version",
    "record_id",
    "created_at",
    "source_tool",
    "source_record_id",
    "classification",
    "status",
    "cause",
    "message",
    "scalars",
    "request",
    "members",
    "arrays",
    "tags",
    "note",
)

_CLASSIFICATION_KEYS = (
    "orbit_family",
    "libration_point",
    "jacobi",
    "amplitude",
    "has_cr3bp",
    "has_ephemeris",
)


@dataclass(frozen=True)
class CatalogFilter:
    """多维查询过滤条件；各维度可独立组合（逻辑与）。

    ``jacobi_min/max`` 与 ``amplitude_min/max`` 与记录的 [min, max] 区间
    做相交匹配；记录对应维度为 ``None``（无该段数据）时不匹配区间过滤。
    ``tags`` 命中任一即匹配。transfer 维度（#574）：``transfer_type``
    等值；``delta_v``/``tli_epoch`` 区间——仅数值历元（JD_TDB）可作
    区间过滤，UTC 字符串历元不入索引列。
    """

    orbit_family: str | None = None
    libration_point: int | None = None
    jacobi_min: float | None = None
    jacobi_max: float | None = None
    amplitude_min_km: float | None = None
    amplitude_max_km: float | None = None
    has_cr3bp: bool | None = None
    has_ephemeris: bool | None = None
    status: str | None = None
    tags: tuple[str, ...] | None = None
    transfer_type: str | None = None
    delta_v_min_km_s: float | None = None
    delta_v_max_km_s: float | None = None
    tli_epoch_min: float | None = None
    tli_epoch_max: float | None = None


@dataclass
class CatalogRecord:
    """一条完整记录：JSON 元数据全文 + NPZ 数组段（键含 ``/`` 段前缀）。"""

    meta: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


def new_record_id() -> str:
    """生成按时间可排序且唯一的 record_id（文件名即 record_id）。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def validate_meta(meta: dict[str, Any]) -> None:
    """校验记录元数据的必备键与 schema 版本；不合格抛 :class:`CatalogError`。"""
    missing = [key for key in _META_REQUIRED_KEYS if key not in meta]
    if missing:
        raise CatalogError(f"记录元数据缺少必备键：{missing}")
    version = meta["schema_version"]
    if version != SCHEMA_VERSION:
        raise CatalogError(
            f"不支持的记录 schema 版本：{version!r}（当前支持 {SCHEMA_VERSION}）；"
            "旧产物不兼容读取，请删除后重算"
        )
    classification = meta["classification"]
    missing = [key for key in _CLASSIFICATION_KEYS if key not in classification]
    if missing:
        raise CatalogError(f"记录分类字段缺少必备键：{missing}")


def member_array_key(index: int, name: str) -> str:
    """族成员数组键（``cr3bp/members/0003/states``）。"""
    return f"{CR3BP_PREFIX}/members/{index:04d}/{name}"


def geometric_amplitude_km(states: np.ndarray, char_length_km: float | None) -> float | None:
    """主振幅（km）：位置三分量半极差最大值 × 特征长度。"""
    if char_length_km is None or states.size == 0:
        return None
    half_ranges = (states[:, :3].max(axis=0) - states[:, :3].min(axis=0)) / 2.0
    return float(half_ranges.max() * char_length_km)


def point_interval(value: float | None) -> list[float] | None:
    """单值包络：非 None 值退化为 [v, v]。"""
    return None if value is None else [value, value]


def member_count(meta: dict[str, Any]) -> int:
    """记录的成员数：族记录为成员数，单轨道记录为 1，纯星历记录为 0。"""
    members = meta["members"]
    if members:
        return len(members)
    return 1 if meta["classification"]["has_cr3bp"] else 0


def cr3bp_segment_arrays(states: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """CR3BP 段数组（Orbit 原生：无量纲 states/times）。"""
    return {
        f"{CR3BP_PREFIX}/states": np.asarray(states, dtype=float),
        f"{CR3BP_PREFIX}/times": np.asarray(times, dtype=float),
    }


def transfer_segment_arrays(trajectory: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """转移轨迹段数组（#574）。

    ``states`` 为 ADR 0040 契约下的轨迹数据：HMN/LGA/WSB 为会合系物理
    km/km/s (n, 6)，low_thrust 暂为力模型状态 (M, 7)（state_frame 标量
    注明数据系）；``times`` 为 TLI 起算秒 (n,)，与 states 逐行对齐。
    """
    return {
        f"{TRANSFER_PREFIX}/states": np.asarray(trajectory, dtype=float),
        f"{TRANSFER_PREFIX}/times": np.asarray(times, dtype=float),
    }


def numeric_or_none(value: Any) -> float | None:
    """数值安全转换：int/float（非 bool）且有限 → float，否则 None。

    索引列与摘要共用：``tli_epoch`` 可为 UTC 字符串（不落数值列）、
    ``delta_v`` 零结果为 inf（不入索引），均归一为 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def ephemeris_segment_arrays(ephemeris: EphemerisTable) -> dict[str, np.ndarray]:
    """星历段数组（EphemerisTable 全字段；``raw_text`` 不落盘）。"""
    arrays: dict[str, np.ndarray] = {
        f"{EPHEMERIS_PREFIX}/year": np.asarray(ephemeris.year),
        f"{EPHEMERIS_PREFIX}/month": np.asarray(ephemeris.month),
        f"{EPHEMERIS_PREFIX}/day": np.asarray(ephemeris.day),
        f"{EPHEMERIS_PREFIX}/hour": np.asarray(ephemeris.hour),
        f"{EPHEMERIS_PREFIX}/minute": np.asarray(ephemeris.minute),
        f"{EPHEMERIS_PREFIX}/second": np.asarray(ephemeris.second, dtype=float),
        f"{EPHEMERIS_PREFIX}/position_km": np.asarray(ephemeris.position_km, dtype=float),
        f"{EPHEMERIS_PREFIX}/velocity_mps": np.asarray(ephemeris.velocity_mps, dtype=float),
        f"{EPHEMERIS_PREFIX}/synodic_position": np.asarray(ephemeris.synodic_position, dtype=float),
    }
    if ephemeris.times_jd_tdb is not None:
        arrays[f"{EPHEMERIS_PREFIX}/times_jd_tdb"] = np.asarray(ephemeris.times_jd_tdb, dtype=float)
    return arrays


def ephemeris_from_arrays(arrays: dict[str, np.ndarray]) -> EphemerisTable:
    """从记录数组段重建 :class:`EphemerisTable`；缺星历段抛 :class:`CatalogError`。"""
    prefix = f"{EPHEMERIS_PREFIX}/"
    try:
        return EphemerisTable(
            year=np.asarray(arrays[f"{prefix}year"], dtype=int),
            month=np.asarray(arrays[f"{prefix}month"], dtype=int),
            day=np.asarray(arrays[f"{prefix}day"], dtype=int),
            hour=np.asarray(arrays[f"{prefix}hour"], dtype=int),
            minute=np.asarray(arrays[f"{prefix}minute"], dtype=int),
            second=np.asarray(arrays[f"{prefix}second"], dtype=float),
            position_km=np.asarray(arrays[f"{prefix}position_km"], dtype=float),
            velocity_mps=np.asarray(arrays[f"{prefix}velocity_mps"], dtype=float),
            synodic_position=np.asarray(arrays[f"{prefix}synodic_position"], dtype=float),
            times_jd_tdb=(
                np.asarray(arrays[f"{prefix}times_jd_tdb"], dtype=float)
                if f"{prefix}times_jd_tdb" in arrays
                else None
            ),
        )
    except KeyError as exc:
        raise CatalogError(f"记录缺少星历段字段：{exc}") from exc


def meta_to_json(meta: dict[str, Any]) -> str:
    """记录元数据序列化为 JSON 文本（中文不转义）。"""
    return json.dumps(meta, ensure_ascii=False, indent=2)


def meta_from_json(text: str) -> dict[str, Any]:
    """从 JSON 文本解析并校验记录元数据。"""
    meta = json.loads(text)
    validate_meta(meta)
    return meta
