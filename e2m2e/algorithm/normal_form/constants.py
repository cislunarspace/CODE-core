"""Normal-form 归一化常数与平动点几何工具。

集中存放 qiao ``Global_File.py`` / ``Calc_LPstate.py`` 中固化下来的物理
与几何常量，以及从给定质量比反解平动点无量纲坐标、共线点 γ 值的辅助函数。

本模块刻意保持为零依赖（仅依赖 ``numpy``），方便上层模块在 ``__init__``
阶段按需加载。``sympy`` / ``joblib`` 的引入留待后续化简器内部惰性导入。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..dynamics import LibrationPoint

# ---------------------------------------------------------------------------
# qiao 全局归一化常量（见 qiao Python/crtbp/Subfunction/Global_File.py）
# ---------------------------------------------------------------------------

#: 归一化长度单位 LU（km）。qiao 约定；与 e2m2e 默认地月距离 (384405 km) 略有差异。
LU_KM: float = 384747.981

#: 归一化时间单位 TU（秒）。
TU_S: float = 375699.843898365

#: 归一化速度单位 VU（km/s），由 LU_KM / TU_S 推导。
VU_KMS: float = LU_KM / TU_S

#: 地月质量比 mu = mu_m / (mu_e + mu_m)，qiao 约定值。
MU: float = 1.215058560962404e-2

#: 归一化地球引力常数（无量纲，总质量归一）。
MU_E: float = 0.987849414390376

#: 归一化月球引力常数（无量纲）。
MU_M: float = 0.012150585609624

#: 归一化太阳引力常数（无量纲）。
MU_S: float = 328900.5614000000

#: qiao 流水线使用的基础频率 omega_1..omega_4（TU^-1），针对地月系统。
BASE_FREQUENCIES: tuple[float, ...] = (
    0.99154828857,
    0.07480066375,
    0.92519871658,
    1.00402177967,
)

#: 共线平动点 L1/L2/L3 的 γ 值（qiao Calc_LPstate.py 中固化）。
#: 物理含义：r_LP = (1 ± γ) * R_EM (L1/L2) 或 r_LP = -γ * R_EM (L3)，
#: 其中 γ 在 CR3BP 无量纲坐标下表示平动点距最近大天体的归一化距离。
_COLLINEAR_GAMMAS: dict[LibrationPoint, float] = {
    LibrationPoint.L1: 0.150934288618019,
    LibrationPoint.L2: 0.167832751054508,
    LibrationPoint.L3: 0.992912060200654,
}

#: qiao 中心流形频率与特征指数（按平动点分别给出）。
#: 对共线点 (L1/L2/L3) 分别对应 (nu1, nu2, lambda)；三角点 (L4/L5) 在
#: 严格 CR3BP 下 lambda≈0，但 qiao 在太阳扰动下给出微小正值。
_CENTRAL_PARAMS: dict[LibrationPoint, tuple[float, float, float]] = {
    LibrationPoint.L1: (2.33774371420711, 2.27427342163957, 2.93924602471),
    LibrationPoint.L2: (1.86464967793235, 1.79093984309149, 2.16475967850),
    LibrationPoint.L3: (1.00308425804420, 1.00934753444748, 0.17970712561),
    LibrationPoint.L4: (0.30259440630339, 1.00408270193080, 0.01416193941),
    LibrationPoint.L5: (0.30251624161526, 1.00403245203481, 0.01381362875),
}


# ---------------------------------------------------------------------------
# 历元与时间辅助
# ---------------------------------------------------------------------------


def jday(yr: int, mon: int, day: int, hr: int = 0, minute: int = 0, sec: float = 0.0) -> float:
    """公历日期 → 儒略日（Vallado 算法）。

    Args:
        yr: 年（1900–2100）。
        mon: 月（1–12）。
        day: 日（1–31）。
        hr: 时（0–23）。
        minute: 分（0–59）。
        sec: 秒（0.0–59.999）。

    Returns:
        对应时刻的儒略日（浮点数）。
    """
    jd = (
        367.0 * yr
        - np.floor(7.0 * (yr + np.floor((mon + 9.0) / 12.0)) * 0.25)
        + np.floor(275.0 * mon / 9.0)
        + day
        + 1721013.5
    )
    jdfrac = (sec + minute * 60.0 + hr * 3600.0) / 86400.0
    if jdfrac > 1.0:
        jd += np.floor(jdfrac)
        jdfrac -= np.floor(jdfrac)
    return float(jd + jdfrac)


#: 参考历元 J2000.0 的儒略日（2000-01-01 12:00:00）。
JD0_J2000: float = jday(2000, 1, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# 平动点几何
# ---------------------------------------------------------------------------


def compute_libration_position(
    point: LibrationPoint,
    mu: float,
    gamma: float | None = None,
) -> npt.NDArray[np.floating]:
    """计算无量纲会合系下平动点位置。

    对共线点 (L1/L2/L3)，qiao 使用固化 γ 值；如 ``gamma`` 为 ``None`` 则
    退回到 ``fsolve`` 求解与 e2m2e 内部 ``CR3BP_System.compute_libration_points``
    一致的非线性方程。三角点 (L4/L5) 给出解析精确位置，与 γ 无关。

    Args:
        point: 平动点枚举。
        mu: 系统质量比。
        gamma: 共线点的 γ 值；为 ``None`` 时自动求解。

    Returns:
        形状 ``(3,)`` 的无量纲会合系坐标 ``[x, y, z]``。
    """
    from scipy.optimize import fsolve  # 惰性导入以保持模块轻量

    if point in (LibrationPoint.L4, LibrationPoint.L5):
        y = np.sqrt(3.0) / 2.0
        sign = 1.0 if point is LibrationPoint.L4 else -1.0
        return np.array([0.5 - mu, sign * y, 0.0], dtype=float)

    if gamma is None:
        if point is LibrationPoint.L1:

            def f(x: float) -> float:
                return x - (1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2

            x0 = 1 - mu ** (1.0 / 3.0)
        elif point is LibrationPoint.L2:

            def f(x: float) -> float:
                return x - (1 - mu) / (x + mu) ** 2 - mu / (x - 1 + mu) ** 2

            x0 = 1 + mu ** (1.0 / 3.0)
        else:  # L3

            def f(x: float) -> float:
                return x + (1 - mu) / (x + mu) ** 2 + mu / (x - 1 + mu) ** 2

            x0 = -1 - (5.0 / 12.0) * mu
        x_lp = float(fsolve(f, x0)[0])
        return np.array([x_lp, 0.0, 0.0], dtype=float)

    if point is LibrationPoint.L1:
        return np.array([1.0 - gamma, 0.0, 0.0], dtype=float)
    if point is LibrationPoint.L2:
        return np.array([1.0 + gamma, 0.0, 0.0], dtype=float)
    # L3（地球外侧，距地球 γ；地心会合系地球在原点）
    return np.array([-gamma, 0.0, 0.0], dtype=float)


def libration_gamma(point: LibrationPoint) -> float:
    """返回共线平动点的 qiao 固化 γ 值。

    Args:
        point: 必须是 ``LibrationPoint.L1``/``L2``/``L3`` 之一。

    Raises:
        ValueError: ``point`` 不是共线平动点。
    """
    try:
        return _COLLINEAR_GAMMAS[point]
    except KeyError as exc:
        raise ValueError(f"γ 仅对共线平动点 (L1/L2/L3) 有定义，得到 {point!r}") from exc


def central_frequencies(
    point: LibrationPoint,
) -> tuple[float, float]:
    """返回 qiao 中心流形频率 ``(nu1, nu2)``。

    Args:
        point: 平动点枚举。
    """
    nu1, nu2, _ = _CENTRAL_PARAMS[point]
    return nu1, nu2


def characteristic_exponent(point: LibrationPoint) -> float:
    """返回 qiao 特征指数 λ。

    Args:
        point: 平动点枚举。
    """
    _, _, lam = _CENTRAL_PARAMS[point]
    return lam

