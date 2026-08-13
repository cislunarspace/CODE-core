"""ELFO 冻结轨道设计的经典力学转换与漂移分析。

本模块为 ``design_orbit`` 的 ELFO 分支提供两类辅助：

1. 经典轨道根数 ↔ 笛卡尔状态转换（``_oe2cart`` / ``_cart2oe``）——CR3BP
   管线不走这条路（初值来自轨道族生成器），ELFO 管线从六根数构造初值。
2. 月心惯性系根数提取与漂移统计（``_extract_moon_centric_elements`` /
   ``_compute_drift``）——传播在地心系进行，冻结特性看的是月心根数。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...data.constants import SECONDS_PER_JULIAN_YEAR

if TYPE_CHECKING:
    from ..data.kernels.manager import SPICEManager

# GRGM900C 月球引力场模型常数
R_MOON = 1738.0  # km（参考半径）
MU_MOON = 4902.799967088639  # km³/s²


def _oe2cart(
    a: float,
    e: float,
    i: float,
    raan: float,
    aop: float,
    nu: float,
    mu: float,
) -> np.ndarray:
    """经典轨道根数 → 笛卡尔状态（惯性系）。

    Args:
        a: 半长轴（km）。
        e: 离心率。
        i: 倾角（度）。
        raan: 升交点赤经（度）。
        aop: 近月点幅角（度）。
        nu: 真近点角（度）。
        mu: 中心天体引力参数（km³/s²）。

    Returns:
        ``(6,)`` 数组 ``[x, y, z, vx, vy, vz]``，单位 km / km·s⁻¹。
    """
    i_rad = np.radians(i)
    raan_rad = np.radians(raan)
    aop_rad = np.radians(aop)
    nu_rad = np.radians(nu)

    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * np.cos(nu_rad))
    r_pf = np.array([r_mag * np.cos(nu_rad), r_mag * np.sin(nu_rad), 0.0])
    v_pf = np.array(
        [
            -np.sqrt(mu / p) * np.sin(nu_rad),
            np.sqrt(mu / p) * (e + np.cos(nu_rad)),
            0.0,
        ]
    )

    cos_raan, sin_raan = np.cos(raan_rad), np.sin(raan_rad)
    cos_aop, sin_aop = np.cos(aop_rad), np.sin(aop_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)

    R_raan = np.array(
        [
            [cos_raan, -sin_raan, 0],
            [sin_raan, cos_raan, 0],
            [0, 0, 1],
        ]
    )
    R_aop = np.array(
        [
            [cos_aop, -sin_aop, 0],
            [sin_aop, cos_aop, 0],
            [0, 0, 1],
        ]
    )
    R_i = np.array(
        [
            [1, 0, 0],
            [0, cos_i, -sin_i],
            [0, sin_i, cos_i],
        ]
    )

    R = R_raan @ R_i @ R_aop
    return np.concatenate([R @ r_pf, R @ v_pf])


def _cart2oe(state: np.ndarray, mu: float) -> dict[str, float]:
    """笛卡尔状态 → 经典轨道根数。

    Args:
        state: ``(6,)`` 数组 ``[x, y, z, vx, vy, vz]``。
        mu: 中心天体引力参数（km³/s²）。

    Returns:
        包含 ``a`` （km）、``e``、``i`` （度）、``raan`` （度）、
        ``aop`` （度）、``rp`` （km）的字典。
    """
    r = np.asarray(state[:3], dtype=float)
    v = np.asarray(state[3:6], dtype=float)
    r_norm = float(np.linalg.norm(r))

    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))

    energy = float(np.dot(v, v)) / 2.0 - mu / r_norm
    a = -mu / (2.0 * energy) if abs(energy) > 1e-14 else float("inf")

    e_vec = np.cross(v, h) / mu - r / r_norm
    e = float(np.linalg.norm(e_vec))

    i = float(np.degrees(np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))))

    n_vec = np.cross([0, 0, 1], h)
    n_norm = float(np.linalg.norm(n_vec))
    if n_norm < 1e-12:
        raan = 0.0
    else:
        raan = float(np.degrees(np.arccos(np.clip(n_vec[0] / n_norm, -1.0, 1.0))))
        if n_vec[1] < 0:
            raan = 360.0 - raan

    if n_norm < 1e-12:
        aop = float(np.degrees(np.arctan2(e_vec[1], e_vec[0]))) % 360.0
    elif e < 1e-14:
        aop = 0.0
    else:
        cos_aop = np.dot(n_vec, e_vec) / (n_norm * e)
        aop = float(np.degrees(np.arccos(np.clip(cos_aop, -1.0, 1.0))))
        if e_vec[2] < 0:
            aop = 360.0 - aop

    return {"a": a, "e": e, "i": i, "raan": raan, "aop": aop, "rp": a * (1.0 - e)}


def _cart2oe_batch(rel: np.ndarray, mu: float) -> dict[str, np.ndarray]:
    """``(n, 6)`` 相对状态 → 经典根数序列（向量化，逐点语义对齐 :func:`_cart2oe`）。

    Args:
        rel: ``(n, 6)`` 数组 ``[x, y, z, vx, vy, vz]``（中心天体惯性系）。
        mu: 中心天体引力参数（km³/s²）。

    Returns:
        各键 → ``(n,)`` 数组的字典，键同 :func:`_cart2oe`。
    """
    rel = np.asarray(rel, dtype=float)
    n = rel.shape[0]
    r = rel[:, :3]
    v = rel[:, 3:]
    r_norm = np.linalg.norm(r, axis=1)

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h, axis=1)

    energy = np.einsum("ij,ij->i", v, v) / 2.0 - mu / r_norm
    a = np.where(np.abs(energy) > 1e-14, -mu / (2.0 * energy), np.inf)

    e_vec = np.cross(v, h) / mu - r / r_norm[:, None]
    e = np.linalg.norm(e_vec, axis=1)

    inc = np.degrees(np.arccos(np.clip(h[:, 2] / h_norm, -1.0, 1.0)))

    # 升交点矢量与保护除法（n_norm≈0 时分母换 1，分支掩码另行归零）
    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h)
    n_norm = np.linalg.norm(n_vec, axis=1)
    mask_n = n_norm >= 1e-12
    mask_e = e >= 1e-14

    raan = np.zeros(n)
    cos_raan = np.clip(n_vec[:, 0] / np.where(mask_n, n_norm, 1.0), -1.0, 1.0)
    raan = np.where(mask_n, np.degrees(np.arccos(cos_raan)), 0.0)
    raan = np.where(mask_n & (n_vec[:, 1] < 0), 360.0 - raan, raan)

    # aop 三分支对齐 _cart2oe：n_norm<1e-12 用 arctan2（与 e 无关），
    # 否则 e<1e-14 归零，其余走 arccos(dot/(n_norm·e)) 并处理 e_vec[2]<0。
    aop = np.zeros(n)
    aop_no_n = np.degrees(np.arctan2(e_vec[:, 1], e_vec[:, 0])) % 360.0
    aop = np.where(~mask_n, aop_no_n, aop)
    denom = np.where(mask_n & mask_e, n_norm * e, 1.0)
    cos_aop = np.clip(np.einsum("ij,ij->i", n_vec, e_vec) / denom, -1.0, 1.0)
    aop = np.where(mask_n & mask_e, np.degrees(np.arccos(cos_aop)), aop)
    aop = np.where(mask_n & mask_e & (e_vec[:, 2] < 0), 360.0 - aop, aop)

    return {"a": a, "e": e, "i": inc, "raan": raan, "aop": aop, "rp": a * (1.0 - e)}


def _extract_moon_centric_elements(
    times: np.ndarray,
    states: np.ndarray,
    spice: SPICEManager,
    mu: float = MU_MOON,
) -> dict[str, np.ndarray]:
    """从地心传播结果提取月心惯性系轨道根数序列（向量化 + Rust 批量星历）。

    传播在地心系进行；冻结特性看的是月心根数。对每个输出时刻，从地心状态
    中减去月球地心位置/速度，再换算为经典根数。月球状态批量下沉 Rust
    （``batch_body_states_py``），根数换算 numpy 向量化，无逐点 Python 循环。

    Args:
        times: ``(n,)`` ET 时间序列（秒）。
        states: ``(n, 6)`` 地心惯性系状态序列。
        spice: 已加载内核的 SPICEManager（保留签名兼容；查询已下沉 Rust 实例）。
        mu: 月球引力参数（km³/s²）。

    Returns:
        各键 → ``(n,)`` 数组的字典，键同 ``_cart2oe``。
    """
    from e2m2e.integrators import batch_body_states_py, require_rust_extension

    require_rust_extension("batch_body_states_py")
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    moon = np.asarray(
        batch_body_states_py("MOON", "EARTH", [float(t) for t in times]), dtype=float
    ).reshape(-1, 6)
    rel = np.concatenate([states[:, :3] - moon[:, :3], states[:, 3:] - moon[:, 3:]], axis=1)
    return _cart2oe_batch(rel, mu)


def _compute_drift(
    elements: dict[str, np.ndarray],
    times: np.ndarray | None = None,
    output_step_sec: float | None = None,
) -> dict[str, float | None]:
    """从月心根数序列计算长期漂移统计。

    Args:
        elements: ``_extract_moon_centric_elements`` 的返回值。
        times: ``(n,)`` ET 时间序列（秒）；与 ``output_step_sec`` 至少给一个，
            用于将每点拟合斜率转换为年漂移率。两者均缺时 secular 率返回 ``None``。
        output_step_sec: 采样间隔（秒）。

    Returns:
        包含以下键的字典：

        - ``drift_e``: 首末差 Δe。
        - ``drift_aop_deg``: 首末差 Δω（度，已处理 ±180° wraparound）。
        - ``drift_rp_km``: 首末差 Δrp（km）。
        - ``secular_aop_rate_deg_per_year``: ω 线性拟合年漂移率（剔除短周期）。
            无法计算时为 ``None``。
    """
    aop = elements["aop"]
    drift_aop = float(aop[-1] - aop[0])
    if drift_aop > 180:
        drift_aop -= 360
    elif drift_aop < -180:
        drift_aop += 360

    drift_e = float(elements["e"][-1] - elements["e"][0])
    drift_rp = float(elements["rp"][-1] - elements["rp"][0])

    # secular 率：对 ω 做 unwrap + 线性拟合，斜率（每点度数）转年率
    secular_rate = None
    n_pts = len(aop)
    if n_pts >= 3:
        aop_unwrap = np.unwrap(np.radians(aop))
        x = np.arange(n_pts, dtype=float)
        slope_per_point = float(np.polyfit(x, np.degrees(aop_unwrap), 1)[0])
        if output_step_sec is not None and output_step_sec > 0:
            points_per_year = SECONDS_PER_JULIAN_YEAR / output_step_sec
            secular_rate = slope_per_point * points_per_year
        elif times is not None and n_pts >= 2:
            total_sec = float(times[-1] - times[0])
            if total_sec > 0:
                slope_per_sec = slope_per_point * (n_pts - 1) / total_sec
                secular_rate = slope_per_sec * SECONDS_PER_JULIAN_YEAR

    return {
        "drift_e": drift_e,
        "drift_aop_deg": drift_aop,
        "drift_rp_km": drift_rp,
        "secular_aop_rate_deg_per_year": secular_rate,
    }
