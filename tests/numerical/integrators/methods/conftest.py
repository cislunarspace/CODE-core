"""积分器测试共享基准物理。

归一化 LEO+J2 问题（DU = 地球半径，TU s.t. mu = 1）是 RK 族、ABM、
Cowell 三类积分器共同的端到端基准——RK/ABM 测试直接对照本模块的
J2 物理，避免公式分散后基准不一致。

例外：Cowell 只接受仅位置的加速度签名 ``a(t, x)``，无法复用状态向量
形式的 ``j2_rhs``，其仅位置版在 test_cowell.py 内自带（docstring 标明
与本模块 ``j2_rhs`` 的加速度分量一致）。

常量来源：mu / re 取自项目常量层 ``Datum.WGS84``（ADR 0022，近地场景
默认基准）；J2 项目常量层暂无对应字段（生产代码从重力系数文件读），
此处保留 WGS-84/EGM96 值并标注来源。
"""

import numpy as np

from e2m2e.data.constants import Datum

# 地球引力参数与赤道半径：WGS-84 基准（Datum.WGS84）。
EARTH_MU = Datum.WGS84.earth_gm  # km³/s²
EARTH_RE = Datum.WGS84.earth_radius_km  # km
# WGS-84/EGM96 带谐系数 J2（项目常量层暂无 J2 字段，见 docstring 说明）。
EARTH_J2 = 1.0826261e-3


def j2_rhs(mu: float = EARTH_MU, re: float = EARTH_RE, j2: float = EARTH_J2):
    """二体 + J2 加速度右端项，状态向量 [x, y, z, vx, vy, vz]，单位 km/s。"""

    def f(t: float, state: np.ndarray) -> np.ndarray:  # noqa: ARG001
        r = state[:3]
        v = state[3:]
        r_norm = np.linalg.norm(r)
        r2 = r_norm**2
        a_2body = -mu * r / r_norm**3
        k = 1.5 * j2 * mu * re**2 / r_norm**5
        z2_over_r2 = r[2] ** 2 / r2
        a_j2 = -k * np.array(
            [
                r[0] * (1.0 - 5.0 * z2_over_r2),
                r[1] * (1.0 - 5.0 * z2_over_r2),
                r[2] * (3.0 - 5.0 * z2_over_r2),
            ]
        )
        return np.concatenate([v, a_2body + a_j2])

    return f


def normalized_leo_j2(altitude_du: float = 400.0 / EARTH_RE, days: float = 1.0):
    """归一化 LEO + J2 问题（长度单位 = 地球半径，时间单位取 mu = 1）。

    归一化使 ||y|| ~ O(1)，让相对容差直接控制误差，不被 km/s 量纲下
    ~7000 km 的状态量级放大。

    返回 ``(rhs, y0, t_span)``，``t_span`` 为归一化时间单位（1 天 ≈ 107.2 TU）。
    """
    tu_per_second = np.sqrt(EARTH_MU / EARTH_RE**3)  # 每秒的 TU 数
    t_span = (0.0, days * 86400.0 * tu_per_second)
    rhs = j2_rhs(mu=1.0, re=1.0, j2=EARTH_J2)
    r = 1.0 + altitude_du
    v = np.sqrt(1.0 / r)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
    return rhs, y0, t_span
