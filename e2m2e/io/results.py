"""DFH RESULTS_HMN/LGA/WSB.TXT 转移设计结果文件解析。

三种文件均为带文字注释的自由格式：

- RESULTS_HMN.TXT（直接转移）：单段，含 TLI/NOI 点历元、状态、根数与
  delta-V；
- RESULTS_LGA.TXT / RESULTS_WSB.TXT（月球/太阳引力辅助间接转移）：共享
  多段布局，每段以 ``Info of successful designed orbit, N`` 开头，末尾有
  ``Orbit taken as an ephmerides example is N``（原文如此，ephmerides 是
  DFH 输出中的拼写）。

语义照 MATLAB ``parse_results_hmn.m`` / ``parse_results_multi.m`` 移植：
按标题字符串定位、其后扫描纯数值行（允许跳过少量标签行）。提取不出的
字段保持 NaN；原始文本完整保留在 ``raw_text``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "HmnResult",
    "MultiOrbitResult",
    "OrbitSegment",
    "parse_results_hmn",
    "parse_results_multi",
    "read_results_hmn",
    "read_results_lga",
    "read_results_wsb",
]

_FLOAT_HEAD_RE = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")
_SEG_START_RE = re.compile(r"^Info of successful designed orbit,\s*(\d+)")
_EXAMPLE_RE = re.compile(r"Orbit taken as an ephmerides example is\s*(\d+)")


def _nan6() -> np.ndarray:
    return np.full(6, np.nan)


def _sscanf_f(line: str) -> list[float]:
    """模拟 MATLAB ``sscanf(line, '%f')``：顺序解析行首开始的数值词元，
    遇到第一个非数值词元即停止。"""
    vals: list[float] = []
    for tok in line.strip().split():
        if _FLOAT_HEAD_RE.match(tok) is None:
            break
        vals.append(float(tok))
    return vals


def _grab_numeric_rows(lines: list[str], start: int, nrows: int) -> list[list[float]]:
    """从 ``start`` 起收集接下来 ``nrows`` 个纯数值行；非数值行（如
    ``a(km), e, ...`` 标签行）跳过，连续跳过不超过 3 行。"""
    rows: list[list[float]] = []
    skip = 0
    i = start
    while i < len(lines) and len(rows) < nrows:
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        vals = _sscanf_f(ln)
        if not vals or not all(np.isfinite(vals)):
            skip += 1
            if skip > 3:
                break
            i += 1
            continue
        rows.append(vals)
        skip = 0
        i += 1
    return rows


def _next_scalar(lines: list[str], k: int) -> float:
    """标题行 ``k`` 之后 5 行内第一个数值行的首个实数。"""
    for i in range(k + 1, min(k + 6, len(lines))):
        ln = lines[i].strip()
        if not ln:
            continue
        vals = _sscanf_f(ln)
        if vals:
            return vals[0]
    return np.nan


def _next_vec6(lines: list[str], k: int) -> np.ndarray:
    """标题行 ``k`` 之后两个纯数值行，每行 3 个实数，拼成 6 维向量。"""
    v = _nan6()
    rows = _grab_numeric_rows(lines, k + 1, 2)
    if len(rows) >= 1:
        v[0:3] = rows[0][:3]
    if len(rows) >= 2:
        v[3:6] = rows[1][:3]
    return v


# =====================================================================
# RESULTS_HMN.TXT（直接转移）
# =====================================================================


@dataclass
class HmnResult:
    """RESULTS_HMN.TXT 解析结果。

    状态向量为 ``[px, py, pz, vx, vy, vz]``（km, m/s）；根数为
    ``[a, e, i, Omicron, omega, M]``（a 单位 km，角度单位 deg）。
    提取不出的字段为 NaN。
    """

    tli_epoch_mjd: float = np.nan
    noi_epoch_mjd: float = np.nan
    tof_day: float = np.nan
    tli_state: np.ndarray = field(default_factory=_nan6)
    tli_elements: np.ndarray = field(default_factory=_nan6)
    noi_state: np.ndarray = field(default_factory=_nan6)
    noi_elements: np.ndarray = field(default_factory=_nan6)
    nominal_state: np.ndarray = field(default_factory=_nan6)
    delta_v_noi: float = np.nan
    raw_text: str = field(default="", repr=False)


def parse_results_hmn(raw: str) -> HmnResult:
    """解析 RESULTS_HMN.TXT 文本。"""
    lines = [ln.strip() for ln in raw.splitlines()]
    r = HmnResult(raw_text=raw)
    for k, ln in enumerate(lines):
        low = ln.lower()
        if "tli point epoch" in low and "mjd" in low:
            r.tli_epoch_mjd = _next_scalar(lines, k)
        elif "noi point epoch" in low and "mjd" in low:
            r.noi_epoch_mjd = _next_scalar(lines, k)
        elif "tof is" in low and "days" in low:
            r.tof_day = _next_scalar(lines, k)
        elif "position (km) and speed (m/s) at the tli point" in low:
            r.tli_state = _next_vec6(lines, k)
        elif "position (km) and speed (m/s) at the noi point" in low:
            r.noi_state = _next_vec6(lines, k)
        elif "position (km) and speed (m/s) at the nominal orbit" in low:
            r.nominal_state = _next_vec6(lines, k)
        elif ln.startswith("Keplerian elements at the TLI point"):
            r.tli_elements = _next_vec6(lines, k)
        elif ln.startswith("Keplerian elements at the NOI point"):
            r.noi_elements = _next_vec6(lines, k)
        elif "delta-v cost at the noi point" in low:
            r.delta_v_noi = _next_scalar(lines, k)
    return r


def read_results_hmn(path: str | Path) -> HmnResult:
    """从文件读入 RESULTS_HMN.TXT。"""
    return parse_results_hmn(Path(path).read_text(encoding="utf-8"))


# =====================================================================
# RESULTS_LGA.TXT / RESULTS_WSB.TXT（共享多段布局）
# =====================================================================


@dataclass
class OrbitSegment:
    """LGA/WSB 文件中一条成功设计轨道的字段。

    状态向量 ``[px, py, pz, vx, vy, vz]``（km, m/s）；根数
    ``[a, e, i, Omicron, omega, M]``（a 单位 km，角度 deg）；历元为 UTC
    六分量 ``[年, 月, 日, 时, 分, 秒]``。提取不出的字段为 NaN。
    """

    index: int = 0
    tli_state: np.ndarray = field(default_factory=_nan6)
    tli_elements: np.ndarray = field(default_factory=_nan6)
    patch_state_before: np.ndarray = field(default_factory=_nan6)
    patch_state_after: np.ndarray = field(default_factory=_nan6)
    patch_elements_before: np.ndarray = field(default_factory=_nan6)
    patch_elements_after: np.ndarray = field(default_factory=_nan6)
    target_state_before: np.ndarray = field(default_factory=_nan6)
    target_state_after: np.ndarray = field(default_factory=_nan6)
    target_elements_before: np.ndarray = field(default_factory=_nan6)
    target_elements_after: np.ndarray = field(default_factory=_nan6)
    delta_v_patch: float = np.nan
    delta_v_target: float = np.nan
    delta_v_total: float = np.nan
    tli_epoch_utc: np.ndarray = field(default_factory=_nan6)
    patch_epoch_utc: np.ndarray = field(default_factory=_nan6)
    target_epoch_utc: np.ndarray = field(default_factory=_nan6)
    tof_tli_patch_day: float = np.nan
    tof_patch_target_day: float = np.nan
    tof_total_day: float = np.nan


@dataclass
class MultiOrbitResult:
    """RESULTS_LGA/WSB.TXT 解析结果。

    Attributes:
        orbits: 各成功设计轨道段
        summary: 第一段之前的前导描述文本（原样保留）
        example_index: 用作星历示例的轨道索引；无此行时为 NaN
    """

    orbits: list[OrbitSegment] = field(default_factory=list)
    summary: str = ""
    example_index: float = np.nan
    raw_text: str = field(default="", repr=False)

    @property
    def num_orbits(self) -> int:
        return len(self.orbits)


def _read_row3(lines: list[str], idx: int) -> list[float] | None:
    if idx < 0 or idx >= len(lines):
        return None
    vals = _sscanf_f(lines[idx])
    return vals[:3] if vals else None


def _find_labeled_value(lines: list[str], start: int, label: str) -> list[float] | None:
    """从 ``start`` 起 5 行内找到含 ``label`` 的行，返回其后一行的 3 个实数。"""
    label = label.lower()
    for i in range(start, min(start + 6, len(lines))):
        if label in lines[i].lower():
            return _read_row3(lines, i + 1)
    return None


def _read_epoch_next(lines: list[str], k: int) -> np.ndarray:
    """UTC 历元跨两行：第一行 Y M D h m，第二行 s.s。"""
    e = _nan6()
    if k + 2 >= len(lines):
        return e
    n1 = _sscanf_f(lines[k + 1])
    n2 = _sscanf_f(lines[k + 2])
    if len(n1) >= 5:
        e[0:5] = n1[:5]
    if n2:
        e[5] = n2[0]
    return e


def _parse_one_segment(blk: list[str], idx: int) -> OrbitSegment:
    orb = OrbitSegment(index=idx)
    for k, ln in enumerate(blk):
        low = ln.lower()
        if "position (km) and speed (m/s) at the tli point" in low:
            orb.tli_state = _next_vec6(blk, k)
        elif "position (km) at the patch point" in low:
            pos = _read_row3(blk, k + 1)
            vb = _find_labeled_value(blk, k + 1, "speed (m/s) at the patch point before maneuver")
            va = _find_labeled_value(blk, k + 1, "speed (m/s) at the patch point after maneuver")
            if pos:
                orb.patch_state_before[0:3] = pos
                orb.patch_state_after[0:3] = pos
            if vb:
                orb.patch_state_before[3:6] = vb
            if va:
                orb.patch_state_after[3:6] = va
        elif "position (km) at the target point" in low:
            pos = _read_row3(blk, k + 1)
            vb = _find_labeled_value(blk, k + 1, "speed (m/s) at the target point before maneuver")
            va = _find_labeled_value(blk, k + 1, "speed (m/s) at the target point after maneuver")
            if pos:
                orb.target_state_before[0:3] = pos
                orb.target_state_after[0:3] = pos
            if vb:
                orb.target_state_before[3:6] = vb
            if va:
                orb.target_state_after[3:6] = va
        elif ln.startswith("Keplerian elements at the TLI point"):
            orb.tli_elements = _next_vec6(blk, k)
        elif ln.startswith("Keplerian elements at patch point before maneuver"):
            orb.patch_elements_before = _next_vec6(blk, k)
        elif ln.startswith("Keplerian elements at patch point after maneuver"):
            orb.patch_elements_after = _next_vec6(blk, k)
        elif ln.startswith("Keplerian element at target point before maneuver") or ln.startswith(
            "Keplerian elements at target point before maneuver"
        ):
            orb.target_elements_before = _next_vec6(blk, k)
        elif ln.startswith("Keplerian element at target point after maneuver") or ln.startswith(
            "Keplerian elements at target point after maneuver"
        ):
            orb.target_elements_after = _next_vec6(blk, k)
        elif "delta-v cost at the patch point" in low:
            orb.delta_v_patch = _next_scalar(blk, k)
        elif "delta-v cost at the target point" in low:
            orb.delta_v_target = _next_scalar(blk, k)
        elif "total delta-v cost" in low:
            orb.delta_v_total = _next_scalar(blk, k)
        elif "epoch at the tli point is (utc)" in low:
            orb.tli_epoch_utc = _read_epoch_next(blk, k)
        elif "epoch at the patch point is (utc)" in low:
            orb.patch_epoch_utc = _read_epoch_next(blk, k)
        elif "epoch at the target point is (utc)" in low:
            orb.target_epoch_utc = _read_epoch_next(blk, k)
        elif "tof from tli point to patch point" in low:
            orb.tof_tli_patch_day = _next_scalar(blk, k)
        elif "tof from patch point to target point" in low:
            orb.tof_patch_target_day = _next_scalar(blk, k)
        elif "total tof" in low:
            orb.tof_total_day = _next_scalar(blk, k)
    return orb


def parse_results_multi(raw: str) -> MultiOrbitResult:
    """解析 RESULTS_LGA.TXT 或 RESULTS_WSB.TXT 文本。"""
    lines = [ln.strip() for ln in raw.splitlines()]

    seg_starts = [i for i, ln in enumerate(lines) if _SEG_START_RE.match(ln)]

    example_index = np.nan
    for ln in reversed(lines):
        m = _EXAMPLE_RE.search(ln)
        if m:
            example_index = float(int(m.group(1)))
            break

    summary = "\n".join(lines[: seg_starts[0]]) if seg_starts else ""

    orbits: list[OrbitSegment] = []
    for s, k0 in enumerate(seg_starts):
        if s + 1 < len(seg_starts):
            k1 = seg_starts[s + 1]
        else:
            trailer = next(
                (i for i in range(k0, len(lines)) if _EXAMPLE_RE.search(lines[i])),
                len(lines),
            )
            k1 = trailer
        orbits.append(_parse_one_segment(lines[k0:k1], s + 1))

    return MultiOrbitResult(
        orbits=orbits, summary=summary, example_index=example_index, raw_text=raw
    )


def read_results_lga(path: str | Path) -> MultiOrbitResult:
    """从文件读入 RESULTS_LGA.TXT。"""
    return parse_results_multi(Path(path).read_text(encoding="utf-8"))


def read_results_wsb(path: str | Path) -> MultiOrbitResult:
    """从文件读入 RESULTS_WSB.TXT。"""
    return parse_results_multi(Path(path).read_text(encoding="utf-8"))
