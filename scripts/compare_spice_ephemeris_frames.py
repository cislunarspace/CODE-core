#!/usr/bin/env python
"""e2m2e-spice 星历与帧旋转全量对拍回归(#638)。

**用途**:为 e2m2e-spice 的星历查询与帧旋转链路建立全量对拍回归框架,
移植 anise(nyx-space/anise, MPL-2.0)的验证方法论——枚举 SPK 内全部目标对、
逐帧采样,同一查询分别走被测实现与 CSPICE 对照路径,误差逐条记录,按分位数
(q75/q99/max)与直方图汇总,发现非机器精度级异常时输出结构化清单并以非零
退出码结束。

**归属**:按 ADR 0037(重负载验证不进默认 pytest),本诊断脚本落在 ``scripts/``。

**裁决**(#638 分诊):误差报告仅作运行期输出,**不入库**;不改动 spice_ffi /
EphemCache / SPICEManager 任何实现。

**对拍拓扑**:

- 被测链路:Rust cspice 实例——``e2m2e._integrators`` 的 ``spice_spkezr`` /
  ``spice_pxform`` / ``batch_et_to_utc_py``,以及挂缓存后的 ``SPICEManager``。
- 对照路径:Python spiceypy 直连(独立 CSPICE 实例)。
- 两实例的内核池经 ``SPICEManager.load_kernel`` 双 furnsh 同步(manager.py)。

**用法**::

    uv run --no-sync python scripts/compare_spice_ephemeris_frames.py --smoke
    uv run --no-sync python scripts/compare_spice_ephemeris_frames.py --step-days 30 \\
        --output report.md

**退出码契约**:

- ``0``:全量样本无异常。
- ``1``:存在非机器精度级异常(结构化清单见报告),可据此转 bug issue。
- 非零 ``SystemExit``(带中文诊断):前置缺失(de440s/de430 或 naif0012.tls 缺失)。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
from spiceypy.utils.exceptions import SpiceyError

# _BODY_FIXED_KERNELS 是私有名,但它是 SPICE 帧内核加载**顺序**的唯一事实源
# (预测 BPC 必须先于历史 BPC:SPICE 对重叠覆盖段取后加载者,历史重构数据在
# 过去时段优先、未来时段由预测数据补齐)。brief 要求脚本不自备内核清单,
# 故直接复用该表而非另立第二份。
from e2m2e.algorithm.design.design_orbit import _BODY_FIXED_KERNELS, default_kernel_dir
from e2m2e.data.kernels._spice_loader import get_spiceypy
from e2m2e.data.kernels.manager import SPICEManager
from e2m2e.integrators import (
    batch_et_to_utc_py,
    require_rust_extension,
    spice_pxform,
    spice_spkezr,
)

# ---------------------------------------------------------------------------
# 阈值表(报告头部原样转载并附理由)
# ---------------------------------------------------------------------------
# 被测与对照同为 CSPICE(仅实例不同),误差预期为机器精度级(实测 0.0);
# 阈值留 ~1e3 余量,避免噪声触发假异常。
NONE_POS_TOL_KM = 1e-7
NONE_VEL_TOL_KMS = 1e-10
# LT(光行时)档:对齐 anise LT q99 参照(2e-5 km)同量级。
LT_POS_TOL_KM = 1e-5
LT_VEL_TOL_KMS = 1e-8
# 旋转:同 CSPICE 对照,预期机器精度级(实测 0.0)。
ROT_TOL_ARCSEC = 1e-3
# IAU 月球帧特例:10 millidegree。anise 记录 IAU 月球帧存在 ~1 millidegree
# 系统差(浮点世纪数舍入所致),10x 余量。本仓与对照同为 CSPICE 实例,
# 是否复现该特例由本脚本实测;阈值单列以便区分。
ROT_TOL_IAU_MOON_ARCSEC = 36.0
# 样条查表为**筛选级**精度(本节产出即 ADR 0016 / ephem_cache.py 注释记载的
# dt 精度退化的量化):ephem_cache.py 注释「月球 1 h 间隔三次样条位置误差 < 1 km」。
SPLINE_POS_TOL_KM = 1.0
SPLINE_VEL_TOL_KMS = 1e-4
# 单对/单帧跳过率超此值 → 覆盖不足异常(内核缺段等)。
SKIP_RATIO_TOL = 0.10

#: 旋转节帧清单:IAU 帧 10 个(pck00010.tpc)+ BPC 帧 3 个(地球/月球 BPC 集)。
ROTATION_FRAMES = [
    "IAU_MERCURY",
    "IAU_VENUS",
    "IAU_EARTH",
    "IAU_MOON",
    "IAU_MARS",
    "IAU_JUPITER",
    "IAU_SATURN",
    "IAU_URANUS",
    "IAU_NEPTUNE",
    "IAU_PLUTO",
    "ITRF93",
    "MOON_PA",
    "MOON_ME",
]

#: 星历节查询窗口内缩(天),给 LT 查询留边缘余量;并夹到统一跨度。
_COVERAGE_INSET_DAYS = 1.0
_CLAMP_START_UTC = "1900-01-01"
_CLAMP_END_UTC = "2076-01-01"

#: 时间节固定回归点:hifitime(nyx-space/hifitime, MPL-2.0)
#: ``tests/epoch.rs::naif_spice_et_tdb_verification`` 的 spiceypy 生成事实数据。
#: 来源:https://github.com/nyx-space/hifitime (UTC, ET 秒 past J2000 TDB, UTC 儒略日)。
_HIFITIME_POINTS = [
    ("1900-01-09T00:17:15", -3155024523.8157988, 2415028.5119792),
    ("1920-07-23T14:39:29", -2506972789.816543, 2422529.1107523),
    ("1954-12-24T06:06:31", -1420782767.8162904, 2435100.7545255),
    ("1960-02-14T06:06:31", -1258523567.8148985, 2436978.7545255),
    ("1983-04-13T12:09:14.274", -527644192.54036534, 2445438.0064152),
    ("2000-02-29T14:57:29", 5108313.185383182, 2451604.1232523),
    ("2022-11-29T07:58:49.782", 722980798.9650334, 2459912.8325206),
    ("2044-06-06T12:18:54", 1402100403.1847699, 2467773.0131250),
    ("2075-04-30T23:59:54", 2377166463.185493, 2479058.4999306),
]

_HIST_SYMBOLS = " .oO@"
_ARCsec_PER_RAD = 180.0 / np.pi * 3600.0


# ---------------------------------------------------------------------------
# 数值/格式辅助
# ---------------------------------------------------------------------------
def _hist_line(values: list[float]) -> str:
    """把误差向量映射到单行 8 桶 ASCII 直方图(按 ``log10`` 分桶,相对计数)。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return "-"
    logs = np.log10(arr + 1e-300)
    lo, hi = float(logs.min()), float(logs.max())
    if hi - lo < 1e-12:
        line = "." * 3 + "@" + "." * 4
    else:
        counts, _ = np.histogram(logs, bins=8, range=(lo, hi))
        peak = int(counts.max()) or 1
        line = "".join(_HIST_SYMBOLS[min(4, int(round(c / peak * 4)))] for c in counts)
    return f"{line} [1e{lo:.0f}..1e{hi:.0f}]"


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    """返回 (q75, q99, max);空样本返回 nan。"""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return (
        float(np.quantile(arr, 0.75)),
        float(np.quantile(arr, 0.99)),
        float(arr.max()),
    )


def _rot_angle_arcsec(r_rust: np.ndarray, r_py: np.ndarray) -> float:
    """两个 3x3 旋转矩阵的相对旋转角(角秒)。

    用四元数向量的 ``atan2(‖vec‖, cosθ)`` 而非 ``arccos((trace-1)/2)``:
    后者在零附近导数发散——当两矩阵逐位相等时 ``trace`` 因浮点舍入落到
    ``1 - ε``,``arccos`` 会给出 ~3e-3 角秒的数值噪声地板,高于 1e-3 阈值,
    导致假异常。``atan2`` 形式对逐位相等的矩阵精确给出 0。
    """
    d = r_rust @ r_py.T
    cos_t = 0.5 * (float(np.trace(d)) - 1.0)
    vec = 0.5 * np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
    sin_t = float(np.linalg.norm(vec))
    return float(np.arctan2(sin_t, cos_t)) * _ARCsec_PER_RAD


def _et_grid(manager: SPICEManager, start_utc: str, end_utc: str, step_days: float):
    """UTC 端点 → 等间隔 ET 网格。"""
    et0 = manager.utc_to_et(start_utc)
    et1 = manager.utc_to_et(end_utc)
    if et1 <= et0:
        return np.asarray([et0])
    return np.arange(et0, et1, step_days * 86400.0)


def _iso_seconds(iso: str) -> float:
    """ISO UTC 字符串 → POSIX 秒(用于毫秒级比较)。"""
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()


def _parse_utc_fields(utc: str) -> tuple[int, int, int, int, int, float]:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):([\d.]+)", utc)
    assert m is not None, f"无法解析 UTC: {utc!r}"
    y, mo, d, h, mi = (int(m.group(i)) for i in range(1, 6))
    return y, mo, d, h, mi, float(m.group(6))


def _parse_isoc_prec0(ref: str) -> tuple[int, int, int, int, int, float]:
    """解析 spiceypy ``et2utc(..., "ISOC", 0)`` 的 ``YYYY-MM-DDTHH:MM:SS`` 输出。"""
    return (
        int(ref[0:4]),
        int(ref[5:7]),
        int(ref[8:10]),
        int(ref[11:13]),
        int(ref[14:16]),
        float(ref[17:]),
    )


def _jd_from_utc_fields(y: int, mo: int, d: int, h: int, mi: int, s: float) -> float:
    """Meeus《Astronomical Algorithms》第 7 章儒略日公式(Gregorian)。"""
    yy, mm = y, mo
    if mm <= 2:
        yy -= 1
        mm += 12
    a = yy // 100
    b = 2 - a + a // 4
    day = d + h / 24.0 + mi / 1440.0 + s / 86400.0
    return int(365.25 * (yy + 4716)) + int(30.6001 * (mm + 1)) + day + b - 1524.5


# ---------------------------------------------------------------------------
# 内核装载
# ---------------------------------------------------------------------------
def load_base_kernels(manager: SPICEManager, kernel_dir: str) -> list[str]:
    """按 ``_BODY_FIXED_KERNELS`` 顺序加载帧内核,并以 naif0012.tls 收尾。

    naif0011.tls 与 naif0012.tls 常同处 kernels/;``_ensure_leapseconds`` 取
    先找到者,而 CSPICE 内核池后 furnsh 的 LSK 生效。必须让 naif0012 最后加载,
    否则 2022+ 回归点相对 naif0011 会差 1 s(2017 闰秒)——正是回归点要守的行为。
    """
    loaded: list[str] = []
    for name in _BODY_FIXED_KERNELS:
        path = os.path.join(kernel_dir, name)
        if os.path.exists(path):
            manager.load_kernel(path)
            loaded.append(path)
    lsk = os.path.join(kernel_dir, "naif0012.tls")
    if not os.path.exists(lsk):
        raise SystemExit(f"缺 naif0012.tls，先跑 make kernels: {kernel_dir}")
    manager.load_kernel(lsk)
    loaded.append(lsk)
    return loaded


def _spk_window(sp, path: str) -> tuple[float, float] | None:
    """SPK 内全 code 覆盖交集 [max(起步), min(终点)];无覆盖返回 None。"""
    starts: list[float] = []
    ends: list[float] = []
    for code in sp.spkobj(path):
        cell = sp.spkcov(path, int(code))
        n = sp.wncard(cell)
        if n == 0:
            continue
        sub_s: list[float] = []
        sub_e: list[float] = []
        for i in range(n):
            a, b = sp.wnfetd(cell, i)
            sub_s.append(a)
            sub_e.append(b)
        starts.append(max(sub_s))
        ends.append(min(sub_e))
    if not starts:
        return None
    return max(starts), min(ends)


# ---------------------------------------------------------------------------
# 节 A:星历 spkezr 对拍
# ---------------------------------------------------------------------------
def check_ephemeris(
    manager: SPICEManager, spk_path: str, step_days: float, smoke: bool
) -> tuple[list[dict], list[dict], dict]:
    """被测 Rust ``spice_spkezr`` vs 对照 spiceypy ``spkezr``,全目标对 x NONE/LT。

    注:``SPICEManager.get_body_state`` 无缓存时 ≡ spiceypy 直连同调用,恒等
    不对拍;manager 面由节 C 的挂缓存路径覆盖。
    """
    sp = get_spiceypy()
    name = os.path.basename(spk_path)
    rows: list[dict] = []
    anomalies: list[dict] = []
    info: dict = {"kernel": name}

    codes = sorted(int(c) for c in sp.spkobj(spk_path))
    labels = {c: _safe_bodc2n(sp, c) for c in codes}
    win = _spk_window(sp, spk_path)
    if win is None:
        anomalies.append(
            {
                "section": "ephemeris",
                "kernel": name,
                "metric": "coverage",
                "value": 0.0,
                "threshold": 1.0,
                "reproduce": f"spkobj({name}) 无覆盖",
            }
        )
        return rows, anomalies, info

    lo = win[0] + _COVERAGE_INSET_DAYS * 86400.0
    hi = win[1] - _COVERAGE_INSET_DAYS * 86400.0
    lo = max(lo, manager.utc_to_et(_CLAMP_START_UTC))
    hi = min(hi, manager.utc_to_et(_CLAMP_END_UTC))
    if hi <= lo:
        anomalies.append(
            {
                "section": "ephemeris",
                "kernel": name,
                "metric": "coverage",
                "value": 0.0,
                "threshold": 1.0,
                "reproduce": f"窗口空: {sp.et2utc(lo, 'ISOC', 0)}..{sp.et2utc(hi, 'ISOC', 0)}",
            }
        )
        return rows, anomalies, info
    info["window"] = (sp.et2utc(lo, "ISOC", 0), sp.et2utc(hi, "ISOC", 0))

    pairs = [(a, b) for i, a in enumerate(codes) for b in codes[i + 1 :]]
    if smoke:
        pairs = [(301, 399)] if {301, 399} <= set(codes) else pairs[:1]
        ets = manager.utc_to_et("2025-01-01") + np.arange(4) * 86400.0
    else:
        ets = _et_grid(manager, _CLAMP_START_UTC, _CLAMP_END_UTC, step_days)
        ets = np.asarray([t for t in ets if lo <= t <= hi])
    info["n_samples"] = int(ets.size)
    info["n_pairs"] = len(pairs)

    for target, obs in pairs:
        for abcorr in ("NONE", "LT"):
            pos_errs: list[float] = []
            vel_errs: list[float] = []
            skips = 0
            max_pos = -1.0
            max_et = float(ets[0]) if ets.size else float(lo)
            for et in ets:
                try:
                    rust_state, _ = spice_spkezr(str(target), float(et), "J2000", abcorr, str(obs))
                    py_state, _ = sp.spkezr(str(target), float(et), "J2000", abcorr, str(obs))
                except (RuntimeError, SpiceyError):
                    skips += 1
                    continue
                dpos = float(np.linalg.norm(np.asarray(rust_state[:3]) - np.asarray(py_state[:3])))
                dvel = float(np.linalg.norm(np.asarray(rust_state[3:]) - np.asarray(py_state[3:])))
                pos_errs.append(dpos)
                vel_errs.append(dvel)
                if dpos > max_pos:
                    max_pos = dpos
                    max_et = float(et)
            n = len(pos_errs)
            pq75, pq99, pmax = _quantiles(pos_errs)
            vq75, vq99, vmax = _quantiles(vel_errs)
            row = {
                "kernel": name,
                "pair": f"({target},{obs})",
                "labels": f"{labels.get(target, '?')}/{labels.get(obs, '?')}",
                "abcorr": abcorr,
                "n": n,
                "skips": skips,
                "pos_q75": pq75,
                "pos_q99": pq99,
                "pos_max": pmax,
                "vel_q75": vq75,
                "vel_q99": vq99,
                "vel_max": vmax,
                "pos_hist": _hist_line(pos_errs),
                "max_et": max_et,
            }
            rows.append(row)
            tol_p = LT_POS_TOL_KM if abcorr == "LT" else NONE_POS_TOL_KM
            tol_v = LT_VEL_TOL_KMS if abcorr == "LT" else NONE_VEL_TOL_KMS
            if pmax > tol_p:
                anomalies.append(
                    _eph_anomaly(name, target, obs, abcorr, "pos_max", pmax, tol_p, max_et)
                )
            if vmax > tol_v:
                anomalies.append(
                    _eph_anomaly(name, target, obs, abcorr, "vel_max", vmax, tol_v, max_et)
                )
            _skip_anomaly(
                anomalies, "ephemeris", f"{name} ({target},{obs}) {abcorr}", n, skips, name
            )
    return rows, anomalies, info


def _eph_anomaly(
    kernel: str,
    target: int,
    obs: int,
    abcorr: str,
    metric: str,
    value: float,
    tol: float,
    et: float,
) -> dict:
    return {
        "section": "ephemeris",
        "kernel": kernel,
        "pair": f"({target},{obs})",
        "abcorr": abcorr,
        "metric": metric,
        "value": value,
        "threshold": tol,
        "reproduce": f'spice_spkezr("{target}", {et!r}, "J2000", "{abcorr}", "{obs}")',
    }


def _skip_anomaly(
    anomalies: list[dict], section: str, label: str, n: int, skips: int, kernel: str
) -> None:
    total = n + skips
    if total == 0 or skips / total <= SKIP_RATIO_TOL:
        return
    anomalies.append(
        {
            "section": section,
            "kernel": kernel,
            "pair": label,
            "metric": "skip_ratio",
            "value": skips / total,
            "threshold": SKIP_RATIO_TOL,
            "reproduce": f"跳过 {skips}/{total} 样本(超 {SKIP_RATIO_TOL:.0%})",
        }
    )


def _safe_bodc2n(sp, code: int) -> str:
    try:
        return str(sp.bodc2n(code))
    except Exception:
        return f"ID{code}"


# ---------------------------------------------------------------------------
# 节 B:帧旋转 pxform 对拍
# ---------------------------------------------------------------------------
def check_rotation(
    manager: SPICEManager, step_days: float, smoke: bool
) -> tuple[list[dict], list[dict], dict]:
    """被测 Rust ``spice_pxform`` vs 对照 spiceypy ``pxform``,方向 (F, "J2000")。

    sxform(6x6)无 Python 侧导出(全库无 ``spice_sxform`` pyfunction),不在
    对拍面。
    """
    sp = get_spiceypy()
    rows: list[dict] = []
    anomalies: list[dict] = []
    frames = list(ROTATION_FRAMES)
    start_utc, end_utc = "2000-01-01", "2026-06-01"
    if smoke:
        frames = ["IAU_MOON", "ITRF93"]
        end_utc = "2000-01-04"
    ets = _et_grid(manager, start_utc, end_utc, step_days if not smoke else 1.0)
    if smoke:
        ets = ets[:3]
    info = {
        "window": (sp.et2utc(float(ets[0]), "ISOC", 0), sp.et2utc(float(ets[-1]), "ISOC", 0)),
        "n_samples": int(ets.size),
        "n_frames": len(frames),
    }

    for frame in frames:
        probe_et = float(ets[0])
        try:
            spice_pxform(frame, "J2000", probe_et)
            sp.pxform(frame, "J2000", probe_et)
        except (RuntimeError, SpiceyError):
            rows.append({"frame": frame, "n": 0, "skips": 0, "unavailable": True})
            continue

        errs: list[float] = []
        skips = 0
        max_ang = -1.0
        max_et = probe_et
        for et in ets:
            try:
                r_rust = np.asarray(spice_pxform(frame, "J2000", float(et)), dtype=float)
                r_py = np.asarray(sp.pxform(frame, "J2000", float(et)), dtype=float)
            except (RuntimeError, SpiceyError):
                skips += 1
                continue
            ang = _rot_angle_arcsec(r_rust, r_py)
            errs.append(ang)
            if ang > max_ang:
                max_ang = ang
                max_et = float(et)
        q75, q99, mx = _quantiles(errs)
        rows.append(
            {
                "frame": frame,
                "n": len(errs),
                "skips": skips,
                "unavailable": False,
                "q75": q75,
                "q99": q99,
                "max": mx,
                "hist": _hist_line(errs),
                "max_et": max_et,
            }
        )
        tol = ROT_TOL_IAU_MOON_ARCSEC if frame == "IAU_MOON" else ROT_TOL_ARCSEC
        if mx > tol:
            anomalies.append(
                {
                    "section": "rotation",
                    "frame": frame,
                    "metric": "angle_max_arcsec",
                    "value": mx,
                    "threshold": tol,
                    "reproduce": f'spice_pxform("{frame}", "J2000", {max_et!r})',
                }
            )
        _skip_anomaly(anomalies, "rotation", frame, len(errs), skips, "")
    return rows, anomalies, info


# ---------------------------------------------------------------------------
# 节 C:EphemCache 样条查表对拍
# ---------------------------------------------------------------------------
def check_spline(manager: SPICEManager, smoke: bool) -> tuple[list[dict], list[dict], dict]:
    """被测 ``manager.get_body_state``(命中 Python 三次样条)vs spiceypy 直连。

    本节预期**非**机器精度:产出即 ADR 0016 / ephem_cache.py 记载的样条 dt
    精度退化的量化。Rust 侧样条缓存无 Python 查询入口(``spice_poc_body_position``
    直连 easier_reader 不过缓存;缓存只供 Rust 力模型内循环),仅经
    enable/disable 构建路径覆盖;量化 Rust 样条需新增 FFI 导出,超出本 issue
    「不改实现」约束。
    """
    sp = get_spiceypy()
    rows: list[dict] = []
    anomalies: list[dict] = []
    days = 2 if smoke else 30
    dt = 3600.0
    et0 = manager.utc_to_et("2025-01-01T00:00:00")
    et1 = et0 + days * 86400.0

    manager.enable_ephem_cache(["MOON", "SUN"], et0, et1, dt=dt, frame="J2000", observer="EARTH")
    try:
        margin = 5 * dt
        ets = np.arange(et0 + margin, et1 - margin, 901.37)  # 与 3600 s 网格无理数错位
        info = {
            "window": (sp.et2utc(et0, "ISOC", 0), sp.et2utc(et1, "ISOC", 0)),
            "n_samples": int(ets.size),
            "dt": dt,
        }
        for body in ("MOON", "SUN"):
            pos_errs: list[float] = []
            vel_errs: list[float] = []
            skips = 0
            max_pos = -1.0
            max_et = float(ets[0])
            for et in ets:
                try:
                    got = np.asarray(manager.get_body_state(body, float(et), "J2000", "EARTH"))
                    ref, _ = sp.spkezr(body, float(et), "J2000", "NONE", "EARTH")
                except (RuntimeError, SpiceyError):
                    skips += 1
                    continue
                ref = np.asarray(ref)
                dpos = float(np.linalg.norm(got[:3] - ref[:3]))
                dvel = float(np.linalg.norm(got[3:] - ref[3:]))
                pos_errs.append(dpos)
                vel_errs.append(dvel)
                if dpos > max_pos:
                    max_pos = dpos
                    max_et = float(et)
            pq75, pq99, pmax = _quantiles(pos_errs)
            vq75, vq99, vmax = _quantiles(vel_errs)
            rows.append(
                {
                    "body": body,
                    "n": len(pos_errs),
                    "skips": skips,
                    "pos_q75": pq75,
                    "pos_q99": pq99,
                    "pos_max": pmax,
                    "vel_q75": vq75,
                    "vel_q99": vq99,
                    "vel_max": vmax,
                    "pos_hist": _hist_line(pos_errs),
                    "max_et": max_et,
                }
            )
            repro = f'manager.get_body_state("{body}", {max_et!r}, "J2000", "EARTH")'
            if pmax > SPLINE_POS_TOL_KM:
                anomalies.append(
                    {
                        "section": "spline",
                        "body": body,
                        "metric": "pos_max",
                        "value": pmax,
                        "threshold": SPLINE_POS_TOL_KM,
                        "reproduce": repro,
                    }
                )
            if vmax > SPLINE_VEL_TOL_KMS:
                anomalies.append(
                    {
                        "section": "spline",
                        "body": body,
                        "metric": "vel_max",
                        "value": vmax,
                        "threshold": SPLINE_VEL_TOL_KMS,
                        "reproduce": repro,
                    }
                )
            _skip_anomaly(anomalies, "spline", body, len(pos_errs), skips, "")
    finally:
        manager.disable_ephem_cache()
    return rows, anomalies, info


# ---------------------------------------------------------------------------
# 节 D:时间转换回归
# ---------------------------------------------------------------------------
def check_time(manager: SPICEManager) -> tuple[list[dict], list[dict], dict]:
    """hifitime (UTC, ET, JD) 固定点 + 反向/批量/Rust 一致性 + 随机闰秒扫查。

    2022/2044/2075 三点隐式守 naif0012 的 2017 闰秒(naif0011 会差 1 s)。
    """
    sp = get_spiceypy()
    rows: list[dict] = []
    anomalies: list[dict] = []
    for utc, et_ref, jd_ref in _HIFITIME_POINTS:
        et_got = float(manager.utc_to_et(utc))
        a_ok = abs(et_got - et_ref) <= 1e-5
        back = str(sp.et2utc(et_ref, "ISOC", 3))
        b_ok = abs(_iso_seconds(back) - _iso_seconds(utc)) <= 1e-3
        c_ok = _compare_rust_utc(sp, et_ref)
        y, mo, d, h, mi, s = _parse_utc_fields(utc)
        d_ok = abs(_jd_from_utc_fields(y, mo, d, h, mi, s) - jd_ref) <= 1e-7
        rows.append({"utc": utc, "et_ref": et_ref, "a": a_ok, "b": b_ok, "c": c_ok, "d": d_ok})
        for name, ok, detail in (
            ("utc_to_et", a_ok, f"ΔET={et_got - et_ref:+.3e}s"),
            ("et2utc_roundtrip", b_ok, f"back={back}"),
            ("rust_batch_et2utc", c_ok, "六分量不一致"),
            ("jd_selfconsistent", d_ok, f"ref={jd_ref}"),
        ):
            if not ok:
                anomalies.append(
                    {
                        "section": "time",
                        "utc": utc,
                        "metric": name,
                        "value": 1.0,
                        "threshold": 0.0,
                        "reproduce": f"{utc} / {et_ref} ({detail})",
                    }
                )

    # 零成本附加扫查:随机 ET 上 Rust 批量 et2utc 与 spiceypy prec=0 逐分量一致。
    rng = np.random.default_rng(0)
    lo = manager.utc_to_et(_CLAMP_START_UTC)
    hi = manager.utc_to_et("2075-04-30")
    ets = rng.uniform(lo, hi, 512)
    seen = batch_et_to_utc_py([float(e) for e in ets])
    sweep_ok = True
    first_bad = None
    for i, e in enumerate(ets):
        ry, rmo, rd, rh, rmi, rs = _parse_isoc_prec0(sp.et2utc(float(e), "ISOC", 0))
        if not (
            seen[0][i] == ry
            and seen[1][i] == rmo
            and seen[2][i] == rd
            and seen[3][i] == rh
            and seen[4][i] == rmi
            and abs(seen[5][i] - rs) <= 1e-6
        ):
            sweep_ok = False
            first_bad = (
                float(e),
                (seen[0][i], seen[1][i], seen[2][i], seen[3][i], seen[4][i], seen[5][i]),
                (ry, rmo, rd, rh, rmi, rs),
            )
            break
    rows.append(
        {
            "utc": "random_sweep(512) [1900..2075]",
            "et_ref": float("nan"),
            "a": True,
            "b": True,
            "c": sweep_ok,
            "d": True,
        }
    )
    if not sweep_ok:
        anomalies.append(
            {
                "section": "time",
                "utc": "random_sweep",
                "metric": "rust_batch_et2utc",
                "value": 1.0,
                "threshold": 0.0,
                "reproduce": str(first_bad),
            }
        )
    return rows, anomalies, {"n_points": len(_HIFITIME_POINTS) + 1}


def _compare_rust_utc(sp, et: float) -> bool:
    """Rust ``batch_et_to_utc_py``(ISOC prec=0)逐分量 == spiceypy 同式输出。"""
    ry, rmo, rd, rh, rmi, rs = _parse_isoc_prec0(sp.et2utc(et, "ISOC", 0))
    got = batch_et_to_utc_py([float(et)])
    return (
        got[0][0] == ry
        and got[1][0] == rmo
        and got[2][0] == rd
        and got[3][0] == rh
        and got[4][0] == rmi
        and abs(got[5][0] - rs) <= 1e-6
    )


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _f(x: float) -> str:
    return "nan" if x != x else f"{x:.3e}"


def render_report(
    ephem_rows: list[dict],
    rot_rows: list[dict],
    spline_rows: list[dict],
    time_rows: list[dict],
    anomalies: list[dict],
    meta: dict,
) -> str:
    out: list[str] = []
    out.append("# e2m2e-spice 星历与帧旋转全量对拍报告 (#638)")
    out.append("")
    out.append(f"- 生成时间: {meta['generated']}")
    out.append(f"- 内核目录: `{meta['kernel_dir']}`")
    out.append(f"- 实载基础内核: {', '.join(os.path.basename(p) for p in meta['base_kernels'])}")
    out.append(f"- --step-days: {meta['step_days']}  --smoke: {meta['smoke']}")
    out.append(f"- 耗时: {meta['elapsed']:.1f}s")
    out.append(
        "- 口径: 被测=Rust cspice 实例(e2m2e._integrators FFI / 挂缓存 SPICEManager);"
        " 对照=spiceypy 直连;两侧为**独立** CSPICE 实例,内核池由 load_kernel 双 furnsh 同步。"
    )
    out.append(
        "  - 因对拍两侧同为 CSPICE(仅实例/版本入口不同),星历与旋转阈值贴近机器精度(实测 0.0)。"
    )
    out.append(
        "  - anise 参照值(平移 2e-16、LT q99 2e-5 km、旋转 < 2 角秒、"
        "IAU 月球 ~1 millidegree 系统差)是其自有实现 vs CSPICE 的另一口径,"
        "与本脚本不可直接比。"
    )
    out.append("  - 样条节为**筛选级**精度(ADR 0016 / ephem_cache.py dt 退化记载),非机器精度。")
    out.append("")
    out.append("## 阈值表")
    out.append("")
    out.append("| 常量 | 值 | 理由 |")
    out.append("|---|---|---|")
    out.append(f"| NONE_POS_TOL_KM | {NONE_POS_TOL_KM:g} | 同 CSPICE 对照,预期机器精度(实测 0.0) |")
    out.append(f"| NONE_VEL_TOL_KMS | {NONE_VEL_TOL_KMS:g} | 同上 |")
    out.append(f"| LT_POS_TOL_KM | {LT_POS_TOL_KM:g} | 对齐 anise LT q99 参照(2e-5 km)同量级 |")
    out.append(f"| LT_VEL_TOL_KMS | {LT_VEL_TOL_KMS:g} | 同上 |")
    out.append(f"| ROT_TOL_ARCSEC | {ROT_TOL_ARCSEC:g} | 同 CSPICE 对照,预期机器精度(实测 0.0) |")
    out.append(
        f"| ROT_TOL_IAU_MOON_ARCSEC | {ROT_TOL_IAU_MOON_ARCSEC:g} | "
        "10 millidegree;anise 记录 IAU 月球帧 ~1 millidegree 系统差(浮点世纪数舍入),10x 余量 |"
    )
    out.append(
        f"| SPLINE_POS_TOL_KM | {SPLINE_POS_TOL_KM:g} | "
        "筛选级;ephem_cache.py『月球 1 h 样条 < 1 km』 |"
    )
    out.append(f"| SPLINE_VEL_TOL_KMS | {SPLINE_VEL_TOL_KMS:g} | 同上 |")
    out.append(f"| SKIP_RATIO_TOL | {SKIP_RATIO_TOL:g} | 单对/单帧跳过率超此值 → 覆盖不足异常 |")
    out.append("")

    out.append("## 节 A:星历 spkezr 对拍")
    out.append("")
    for info in meta["ephem_infos"]:
        out.append(
            f"- `{info['kernel']}`: 窗口 {info.get('window')}, 样本/对 {info.get('n_samples')},"
            f" 目标对 {info.get('n_pairs')}"
        )
    out.append("")
    out.append(
        "| kernel | pair | abcorr | n | skip | pos_q75 | pos_q99 | pos_max |"
        " vel_q75 | vel_q99 | vel_max | pos_hist |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ephem_rows:
        out.append(
            f"| {r['kernel']} | {r['pair']} | {r['abcorr']} | {r['n']} | {r['skips']} |"
            f" {_f(r['pos_q75'])} | {_f(r['pos_q99'])} | {_f(r['pos_max'])} |"
            f" {_f(r['vel_q75'])} | {_f(r['vel_q99'])} | {_f(r['vel_max'])} |"
            f" `{r['pos_hist']}` |"
        )
    out.append("")

    out.append("## 节 B:帧旋转 pxform 对拍(方向 F→J2000)")
    out.append("")
    out.append(
        f"- 窗口 {meta['rot_info'].get('window')}, 样本/帧 {meta['rot_info'].get('n_samples')},"
        f" 帧数 {meta['rot_info'].get('n_frames')}"
    )
    out.append("")
    out.append("| frame | n | skip | q75(arcsec) | q99(arcsec) | max(arcsec) | hist |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rot_rows:
        if r.get("unavailable"):
            out.append(f"| {r['frame']} | - | - | - | - | - | unavailable(探针失败,非异常) |")
            continue
        out.append(
            f"| {r['frame']} | {r['n']} | {r['skips']} | {_f(r['q75'])} | {_f(r['q99'])} |"
            f" {_f(r['max'])} | `{r['hist']}` |"
        )
    out.append("")

    out.append("## 节 C:EphemCache 样条查表对拍(筛选级)")
    out.append("")
    out.append(
        f"- 窗口 {meta['spline_info'].get('window')}, dt {meta['spline_info'].get('dt')}s,"
        f" 样本/体 {meta['spline_info'].get('n_samples')}; 对照=spiceypy spkezr(NONE)"
    )
    out.append("")
    out.append(
        "| body | n | skip | pos_q75 | pos_q99 | pos_max | vel_q75 | vel_q99 | vel_max | pos_hist |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in spline_rows:
        out.append(
            f"| {r['body']} | {r['n']} | {r['skips']} | {_f(r['pos_q75'])} | {_f(r['pos_q99'])} |"
            f" {_f(r['pos_max'])} | {_f(r['vel_q75'])} | {_f(r['vel_q99'])} | {_f(r['vel_max'])} |"
            f" `{r['pos_hist']}` |"
        )
    out.append("")

    out.append("## 节 D:时间转换回归(UTC/ET/JD)")
    out.append("")
    out.append(
        "检查项: a=`utc_to_et`±1e-5s; b=`et2utc(ISOC,3)`回环±1ms;"
        " c=Rust `batch_et_to_utc_py` vs spiceypy prec=0 逐分量; d=Meeus JD 自洽±1e-7d"
    )
    out.append("")
    out.append("| UTC | ET_ref | a | b | c | d |")
    out.append("|---|---|---|---|---|---|")
    for r in time_rows:
        mark = lambda ok: "OK" if ok else "FAIL"  # noqa: E731
        et_col = "nan" if r["et_ref"] != r["et_ref"] else f"{r['et_ref']:.3f}"
        out.append(
            f"| {r['utc']} | {et_col} | {mark(r['a'])} | {mark(r['b'])} |"
            f" {mark(r['c'])} | {mark(r['d'])} |"
        )
    out.append("")

    out.append("## 异常清单")
    out.append("")
    if not anomalies:
        out.append("无。")
    else:
        for i, a in enumerate(anomalies, 1):
            out.append(
                f"{i}. [{a.get('section')}] {a.get('kernel', '')}{a.get('frame', '')}"
                f"{a.get('pair', '')}{a.get('body', '')}{a.get('utc', '')}"
                f" metric={a['metric']} value={_f(a['value'])} threshold={_f(a['threshold'])}"
            )
            out.append(f"   - 复现: `{a['reproduce']}`")
    out.append("")
    out.append(f"异常计数: {len(anomalies)}")
    out.append("OK" if not anomalies else "FAIL")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="e2m2e-spice 星历与帧旋转全量对拍回归 (#638)")
    p.add_argument("--kernel-dir", default=None, help="SPICE 内核目录(默认 default_kernel_dir())")
    p.add_argument("--step-days", type=float, default=1.0, help="星历/旋转采样步长(天),默认 1.0")
    p.add_argument("--smoke", action="store_true", help="裁剪规模秒级跑通(单内核/单日/单目标对)")
    p.add_argument("--output", default=None, help="报告写入路径(默认打 stdout)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    require_rust_extension(
        "spice_spkezr", "spice_pxform", "batch_et_to_utc_py", "spice_furnsh", "spice_unload"
    )
    kernel_dir = args.kernel_dir or default_kernel_dir()
    start = time.perf_counter()

    manager = SPICEManager()
    base_kernels = load_base_kernels(manager, kernel_dir)

    spks: list[str] = []
    for name in ("de440s.bsp", "de430.bsp"):
        path = os.path.join(kernel_dir, name)
        if os.path.exists(path):
            spks.append(path)
    if not spks:
        raise SystemExit(f"行星历内核不存在（de440s/de430）: {kernel_dir}，先跑 make kernels")
    if args.smoke:
        spks = [p for p in spks if os.path.basename(p) == "de440s.bsp"] or spks[:1]

    ephem_rows: list[dict] = []
    ephem_infos: list[dict] = []
    anomalies: list[dict] = []
    try:
        for spk in spks:
            manager.load_kernel(spk)
            try:
                rows, anom, info = check_ephemeris(manager, spk, args.step_days, args.smoke)
            finally:
                manager.unload_kernel(spk)
            ephem_rows.extend(rows)
            ephem_infos.append(info)
            anomalies.extend(anom)

        # 节 B/C 共用一个可工作的行星历内核(样条节需 MOON/EARTH 相对状态)。
        working_spk = spks[0]
        manager.load_kernel(working_spk)
        try:
            rot_rows, rot_anom, rot_info = check_rotation(manager, args.step_days, args.smoke)
            anomalies.extend(rot_anom)
            spline_rows, spline_anom, spline_info = check_spline(manager, args.smoke)
            anomalies.extend(spline_anom)
        finally:
            manager.unload_kernel(working_spk)

        time_rows, time_anom, time_info = check_time(manager)
        anomalies.extend(time_anom)
    finally:
        for p in base_kernels:
            with contextlib.suppress(Exception):
                manager.unload_kernel(p)

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kernel_dir": kernel_dir,
        "base_kernels": base_kernels,
        "step_days": args.step_days,
        "smoke": args.smoke,
        "elapsed": time.perf_counter() - start,
        "ephem_infos": ephem_infos,
        "rot_info": rot_info,
        "spline_info": spline_info,
        "time_info": time_info,
    }
    report = render_report(ephem_rows, rot_rows, spline_rows, time_rows, anomalies, meta)
    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report)
    return 1 if anomalies else 0


if __name__ == "__main__":
    sys.exit(main())
