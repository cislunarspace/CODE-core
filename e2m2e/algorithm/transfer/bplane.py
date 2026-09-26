"""月心/地心 B-plane 与双曲渐近线参数化（PCN patched-conic 目标参数化，#635）。

本模块是 patched-conic（圆锥曲线拼接）目标参数化的纯数学内核：给定相对某
中心天体的二体惯性状态（GCRS 约定、km / km/s），解析地给出

- 双曲渐近线参数化 ``AsymptoteParams``：赤经 RHA、赤纬 DHA、C3 能量
  （``C3 = v∞²``）；
- B 平面参数化 ``BPlaneParams``：Vallado 定义——Ŝ 为入射渐近线速度方向，
  T̂ = normalize(ẑ×Ŝ)，R̂ = Ŝ×T̂，B 矢量为从焦点指向入射渐近线与 B 平面交点
  的向量（模为瞄准距离 b），B·R/B·T 为其沿 R̂/T̂ 的分量，θ 为 B 矢量角
  ``atan2(B·R, B·T)``；
- 正向映射 ``bplane_from_state`` 与解析雅可比
  ``bplane_jacobian_from_state``（打靶残差约束接口，#642）；
- 逆映射 ``state_from_bplane`` / ``perilune_state_from_bplane``（入射分支）；
- 近心距闭式 ``perilune_radius_closed_form`` 与近心点时间
  ``hyperbolic_time_to_periapsis``。

符号约定（入射双曲线分支）：航天器沿 −Ŝ 方向逼近，逐渐升高到近心点；
``state_from_bplane`` 给出的状态真近点角 ν<0，``hyperbolic_time_to_periapsis``
返回正的“到近心点时间”。

公式出处：Vallado《Fundamentals of Astrodynamics and Applications》双曲线
渐近线 / B 平面几何；Curtis (2008) 开普勒时间方程（双曲版）。

层级：algorithm 层，仅依赖 numpy/math；不得导入任何业务层。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AsymptoteParams",
    "BPlaneParams",
    "asymptote_from_vinf_vector",
    "bplane_from_state",
    "bplane_jacobian_from_state",
    "hyperbolic_time_to_periapsis",
    "perilune_radius_closed_form",
    "perilune_state_from_bplane",
    "state_from_bplane",
    "vinf_vector_from_asymptote",
]

#: 参考 z 轴（T̂ = normalize(ẑ×Ŝ) 的基准方向）。
_Z_HAT = np.array([0.0, 0.0, 1.0])

#: T̂ 构造的退化判据（|ẑ×Ŝ| 低于此视为 Ŝ∥ẑ）。
_BASIS_DEGENERATE_EPS = 1e-12


@dataclass(frozen=True)
class AsymptoteParams:
    """双曲渐近线参数化（地心/月心 GCRS 惯性系）。

    Attributes:
        rha_deg: 渐近线赤经 RHA [0, 360)（deg）。
        dha_deg: 渐近线赤纬 DHA [-90, 90]（deg）。
        c3_km2_s2: C3 能量 = v∞²（km²/s²）；双曲须 > 0。
    """

    rha_deg: float
    dha_deg: float
    c3_km2_s2: float


@dataclass(frozen=True, eq=False)  # ndarray 字段不用默认 __eq__（数组比较歧义）
class BPlaneParams:
    """月心/地心 B-plane 参数化（Vallado 定义，GCRS 惯性系）。

    Attributes:
        mu_km3_s2: 中心天体引力参数 (km³/s²)。
        v_inf_km_s: 双曲剩余速度 v∞ (km/s)。
        c3_km2_s2: C3 = v∞² (km²/s²)。
        rha_deg: 渐近线 Ŝ 赤经 [0, 360)（deg）。
        dha_deg: 渐近线 Ŝ 赤纬 [-90, 90]（deg）。
        bdot_r_km: B·R (km)。
        bdot_t_km: B·T (km)。
        b_mag_km: 瞄准距离 |B| (km)。
        theta_deg: B 矢量角 atan2(B·R, B·T)（deg）。
        perilune_radius_km: 近心距 r_p (km)。
        s_hat: 渐近线向 Ŝ (3,)（B-plane 基，响应/details 不携带）。
        t_hat: B-plane 基 T̂ = normalize(ẑ×Ŝ) (3,)。
        r_hat: B-plane 基 R̂ = Ŝ×T̂ (3,)。
    """

    mu_km3_s2: float
    v_inf_km_s: float
    c3_km2_s2: float
    rha_deg: float
    dha_deg: float
    bdot_r_km: float
    bdot_t_km: float
    b_mag_km: float
    theta_deg: float
    perilune_radius_km: float
    s_hat: NDArray[np.float64]
    t_hat: NDArray[np.float64]
    r_hat: NDArray[np.float64]


# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------


def _skew(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """反对称叉乘矩阵 [a]_×，满足 [a]_× u = a×u。"""
    return np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ],
        dtype=np.float64,
    )


def _unit_from_rha_dha(rha_deg: float, dha_deg: float) -> NDArray[np.float64]:
    """(RHA, DHA) → 单位球面方向（GCRS）。"""
    rha = math.radians(rha_deg)
    dha = math.radians(dha_deg)
    cd = math.cos(dha)
    return np.array([cd * math.cos(rha), cd * math.sin(rha), math.sin(dha)], dtype=np.float64)


def _rha_dha_from_unit(u: NDArray[np.float64]) -> tuple[float, float]:
    """单位球面方向（GCRS）→ (RHA [0,360), DHA [-90,90])。"""
    rha = math.degrees(math.atan2(float(u[1]), float(u[0]))) % 360.0
    if rha >= 360.0:  # 归一到 [0, 360)：−ε 经取模可能落在 360.0
        rha = 0.0
    dha = math.degrees(math.asin(float(np.clip(u[2], -1.0, 1.0))))
    return rha, dha


def _bplane_basis(s_hat: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """由渐近线方向 Ŝ 构造 B-plane 基 (T̂, R̂)。"""
    t_vec = np.cross(_Z_HAT, s_hat)
    tn = float(np.linalg.norm(t_vec))
    if tn < _BASIS_DEGENERATE_EPS:
        raise ValueError("渐近线方向平行于参考 z 轴，B-plane 的 T 矢量退化")
    t_hat = t_vec / tn
    r_hat = np.cross(s_hat, t_hat)
    return t_hat, r_hat


# ---------------------------------------------------------------------------
# 渐近线参数化
# ---------------------------------------------------------------------------


def vinf_vector_from_asymptote(a: AsymptoteParams) -> NDArray[np.float64]:
    """渐近线参数化 → 双曲剩余速度矢量 v⃗∞ = v∞·Ŝ (3,)，km/s。"""
    if a.c3_km2_s2 <= 0.0:
        raise ValueError(f"C3 须 > 0（双曲），得到 {a.c3_km2_s2}")
    return _unit_from_rha_dha(a.rha_deg, a.dha_deg) * math.sqrt(a.c3_km2_s2)


def asymptote_from_vinf_vector(v: ArrayLike) -> AsymptoteParams:
    """双曲剩余速度矢量 (3,) → 渐近线参数化 (RHA, DHA, C3)。"""
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        raise ValueError("v∞ 矢量模为零，无法参数化")
    rha, dha = _rha_dha_from_unit(arr / n)
    return AsymptoteParams(rha_deg=rha, dha_deg=dha, c3_km2_s2=n * n)


# ---------------------------------------------------------------------------
# 闭式近心距
# ---------------------------------------------------------------------------


def perilune_radius_closed_form(v_inf_km_s: float, b_mag_km: float, mu: float) -> float:
    """闭式近心距 r_p = (μ/v∞²)(√(1+(b·v∞²/μ)²) − 1)。

    等价于教科书形式 ``a(|e|−1)``（双曲 a<0），其中
    ``e = √(1+(b·v∞²/μ)²)``。
    """
    if v_inf_km_s <= 0.0:
        raise ValueError(f"v∞ 须 > 0（双曲），得到 {v_inf_km_s}")
    e = math.sqrt(1.0 + (b_mag_km * v_inf_km_s**2 / mu) ** 2)
    return (mu / v_inf_km_s**2) * (e - 1.0)


# ---------------------------------------------------------------------------
# 正向映射：状态 → B-plane
# ---------------------------------------------------------------------------


def bplane_from_state(state: ArrayLike, mu: float) -> BPlaneParams:
    """二体惯性状态 (6,) → 月心/地心 B-plane 参数（入射双曲分支）。

    Args:
        state: 相对中心天体的惯性状态 (6,)，km / km/s。
        mu: 中心天体引力参数 (km³/s²)。

    Returns:
        :class:`BPlaneParams`。

    Raises:
        ValueError: 到达轨道非双曲（v∞² ≤ 0 或 e ≤ 1）或 Ŝ∥ẑ 退化。
    """
    x = np.asarray(state, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(x)):
        raise ValueError("状态含非有限值，无法构造 B-plane")
    r_vec = x[:3]
    v_vec = x[3:]
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("状态位置矢量为零，无法构造 B-plane")
    v2 = float(v_vec @ v_vec)
    vinf2 = v2 - 2.0 * mu / r
    if vinf2 <= 0.0:
        raise ValueError(f"到达轨道非双曲（v∞²={vinf2:.6e} ≤ 0）")
    v_inf = math.sqrt(vinf2)

    e_vec = ((v2 - mu / r) * r_vec - float(r_vec @ v_vec) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))
    if e <= 1.0:
        raise ValueError(f"到达轨道偏心率 e={e:.6f} ≤ 1，非双曲")
    e_hat = e_vec / e

    h_vec = np.cross(r_vec, v_vec)
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm == 0.0:
        raise ValueError("角动量为零，轨道退化")
    h_hat = h_vec / h_norm
    q_hat = np.cross(h_hat, e_hat)

    s = math.sqrt(e * e - 1.0)
    s_hat = e_hat / e + (s / e) * q_hat  # 入射渐近线方向 Ŝ
    t_hat, r_hat = _bplane_basis(s_hat)

    b = mu * s / vinf2
    b_vec = b * np.cross(s_hat, h_hat)
    bdot_r = float(b_vec @ r_hat)
    bdot_t = float(b_vec @ t_hat)
    theta = math.degrees(math.atan2(bdot_r, bdot_t))
    rp = mu * (e - 1.0) / vinf2
    rha, dha = _rha_dha_from_unit(s_hat)

    return BPlaneParams(
        mu_km3_s2=float(mu),
        v_inf_km_s=v_inf,
        c3_km2_s2=vinf2,
        rha_deg=rha,
        dha_deg=dha,
        bdot_r_km=bdot_r,
        bdot_t_km=bdot_t,
        b_mag_km=float(np.linalg.norm(b_vec)),
        theta_deg=theta,
        perilune_radius_km=rp,
        s_hat=s_hat,
        t_hat=t_hat,
        r_hat=r_hat,
    )


def bplane_jacobian_from_state(state: ArrayLike, mu: float) -> NDArray[np.float64]:
    """正向映射 ``(B·R, B·T, r_p)`` 对状态 ``(r, v)`` 的**解析**雅可比 (3, 6)。

    ``#642`` 并入的打靶残差约束接口：返回 ``∂[B·R, B·T, r_p]/∂[r⃗, v⃗]``。
    逐链对 :func:`bplane_from_state` 的公式求导（每步均为初等向量代数），
    非有限差分。行顺序与残差 ``R = (B·R − B·R*, B·T − B·T*, r_p − r_p*)``
    一致。

    Args:
        state: 相对中心天体的惯性状态 (6,)，km / km/s。
        mu: 中心天体引力参数 (km³/s²)。

    Returns:
        (3, 6) 雅可比矩阵。

    Raises:
        ValueError: 到达轨道非双曲（v∞² ≤ 0）或 Ŝ∥ẑ 退化。
    """
    x = np.asarray(state, dtype=np.float64).reshape(6)
    r_vec = x[:3].copy()
    v_vec = x[3:].copy()
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("状态位置矢量为零，无法构造 B-plane 雅可比")
    pos_hat = r_vec / r
    v2 = float(v_vec @ v_vec)
    vinf2 = v2 - 2.0 * mu / r
    if vinf2 <= 0.0:
        raise ValueError(f"到达轨道非双曲（v∞²={vinf2:.6e} ≤ 0），雅可比无定义")

    def row(pos: NDArray[np.float64], vel: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.zeros(6)
        out[:3] = pos
        out[3:] = vel
        return out

    i36_r = np.zeros((3, 6))
    i36_r[:, :3] = np.eye(3)
    i36_v = np.zeros((3, 6))
    i36_v[:, 3:] = np.eye(3)

    # --- 偏心率矢量及其导数 ---
    a_energy = v2 - mu / r
    bc = float(r_vec @ v_vec)
    d_a_dx = row((mu / r**2) * pos_hat, 2.0 * v_vec)
    d_bc_dx = row(v_vec, r_vec)
    e_vec = (a_energy * r_vec - bc * v_vec) / mu
    e = float(np.linalg.norm(e_vec))
    if e <= 1.0:
        raise ValueError(f"到达轨道偏心率 e={e:.6f} ≤ 1，非双曲")
    e_hat = e_vec / e
    de_vec_dx = (
        np.outer(r_vec, d_a_dx) + a_energy * i36_r - np.outer(v_vec, d_bc_dx) - bc * i36_v
    ) / mu
    de_dx = e_hat @ de_vec_dx
    d_e_hat_dx = (de_vec_dx - np.outer(e_hat, de_dx)) / e

    # --- 角动量方向及其导数 ---
    h_vec = np.cross(r_vec, v_vec)
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm == 0.0:
        raise ValueError("角动量为零，雅可比无定义")
    h_hat = h_vec / h_norm
    d_h_vec_dx = np.hstack([-_skew(v_vec), _skew(r_vec)])
    d_hhat_dx = (np.eye(3) - np.outer(h_hat, h_hat)) @ d_h_vec_dx / h_norm

    q_hat = np.cross(h_hat, e_hat)
    d_q_hat_dx = -_skew(e_hat) @ d_hhat_dx + _skew(h_hat) @ d_e_hat_dx

    # --- 入射渐近线方向 Ŝ ---
    s = math.sqrt(e * e - 1.0)
    d_s_dx = (e / s) * de_dx
    pcoef = s / e
    d_pcoef_dx = de_dx / (s * e * e)
    inv_e = 1.0 / e
    d_inv_e_dx = -de_dx / (e * e)
    s_hat = inv_e * e_hat + pcoef * q_hat
    d_s_hat_dx = (
        inv_e * d_e_hat_dx
        + np.outer(e_hat, d_inv_e_dx)
        + np.outer(q_hat, d_pcoef_dx)
        + pcoef * d_q_hat_dx
    )

    # --- B-plane 基 T̂, R̂ ---
    t_vec = np.cross(_Z_HAT, s_hat)
    tn = float(np.linalg.norm(t_vec))
    if tn < _BASIS_DEGENERATE_EPS:
        raise ValueError("渐近线方向平行于参考 z 轴，B-plane 的 T 矢量退化")
    t_hat = t_vec / tn
    d_t_vec_dx = _skew(_Z_HAT) @ d_s_hat_dx
    d_t_hat_dx = (np.eye(3) - np.outer(t_hat, t_hat)) @ d_t_vec_dx / tn
    r_hat = np.cross(s_hat, t_hat)
    d_r_hat_dx = -_skew(t_hat) @ d_s_hat_dx + _skew(s_hat) @ d_t_hat_dx

    # --- B 矢量及其导数 ---
    d_vinf2_dx = row(2.0 * mu / r**2 * pos_hat, 2.0 * v_vec)
    b = mu * s / vinf2
    d_b_dx = mu * (d_s_dx / vinf2 - s * d_vinf2_dx / vinf2**2)
    s_cross_h = np.cross(s_hat, h_hat)
    d_s_cross_h_dx = -_skew(h_hat) @ d_s_hat_dx + _skew(s_hat) @ d_hhat_dx
    b_vec = b * s_cross_h
    d_b_vec_dx = np.outer(s_cross_h, d_b_dx) + b * d_s_cross_h_dx

    # --- 残差行 (B·R, B·T, r_p) ---
    d_bdot_r_dx = r_hat @ d_b_vec_dx + b_vec @ d_r_hat_dx
    d_bdot_t_dx = t_hat @ d_b_vec_dx + b_vec @ d_t_hat_dx
    d_rp_dx = mu * (de_dx / vinf2 - (e - 1.0) * d_vinf2_dx / vinf2**2)
    return np.vstack([d_bdot_r_dx, d_bdot_t_dx, d_rp_dx])


# ---------------------------------------------------------------------------
# 逆映射：B-plane → 状态
# ---------------------------------------------------------------------------


def _hyperbolic_elements_from_bplane(
    a: AsymptoteParams, bdot_r_km: float, bdot_t_km: float, mu: float
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """由渐近线 + B-plane 目标解出双曲元素。

    Returns:
        (s_hat, t_hat, r_hat, b_hat, e, e_hat, q_hat, rp)。
    """
    s_hat = _unit_from_rha_dha(a.rha_deg, a.dha_deg)
    t_hat, r_hat = _bplane_basis(s_hat)
    b_vec = bdot_t_km * t_hat + bdot_r_km * r_hat
    b = float(np.linalg.norm(b_vec))
    if b <= 0.0:
        raise ValueError("B 矢量模为零，逆映射退化（B·R = B·T = 0）")
    b_hat = b_vec / b
    vinf2 = a.c3_km2_s2
    if vinf2 <= 0.0:
        raise ValueError(f"C3 须 > 0（双曲），得到 {vinf2}")
    e = math.sqrt(1.0 + (b * vinf2 / mu) ** 2)
    h_hat = np.cross(b_hat, s_hat)
    s = math.sqrt(e * e - 1.0)
    e_hat = (s_hat + s * b_hat) / e
    q_hat = np.cross(h_hat, e_hat)
    rp = mu * (e - 1.0) / vinf2
    return s_hat, t_hat, r_hat, b_hat, e, e_hat, q_hat, rp


def state_from_bplane(
    a: AsymptoteParams, bdot_r_km: float, bdot_t_km: float, mu: float, radius_km: float
) -> NDArray[np.float64]:
    """B-plane 目标 → 给定球心距处的惯性状态 (6,)（入射双曲分支）。

    真近点角取负根（航天器尚未到近心点）。

    Args:
        a: 渐近线参数化。
        bdot_r_km: B·R (km)。
        bdot_t_km: B·T (km)。
        mu: 中心天体引力参数 (km³/s²)。
        radius_km: 目标状态的球心距 (km)，须 > 近心距。

    Returns:
        (6,) 惯性状态，km / km/s。

    Raises:
        ValueError: B 矢量模为零（退化）或 radius_km ≤ 近心距。
    """
    s_hat, _t_hat, _r_hat, _b_hat, e, e_hat, q_hat, rp = _hyperbolic_elements_from_bplane(
        a, bdot_r_km, bdot_t_km, mu
    )
    if radius_km <= rp:
        raise ValueError(f"半径 {radius_km} 不大于近心距 {rp}，状态不可达")
    vinf2 = a.c3_km2_s2
    p = mu * (e * e - 1.0) / vinf2
    cos_nu = float(np.clip((p / radius_km - 1.0) / e, -1.0, 1.0))
    sin_nu = -math.sqrt(max(0.0, 1.0 - cos_nu * cos_nu))  # 入射分支（ν < 0）
    r_vec = radius_km * (cos_nu * e_hat + sin_nu * q_hat)
    v_vec = math.sqrt(mu / p) * (-sin_nu * e_hat + (e + cos_nu) * q_hat)
    return np.concatenate([r_vec, v_vec])


def perilune_state_from_bplane(
    a: AsymptoteParams, bdot_r_km: float, bdot_t_km: float, mu: float
) -> NDArray[np.float64]:
    """B-plane 目标 → 闭式近心点惯性状态 (6,)，km / km/s。

    近心点位置沿 ê、速度沿 q̂（运动方向），速度模
    ``√(v∞² + 2μ/r_p)``。
    """
    _s_hat, _t_hat, _r_hat, _b_hat, _e, e_hat, q_hat, rp = _hyperbolic_elements_from_bplane(
        a, bdot_r_km, bdot_t_km, mu
    )
    v_mag = math.sqrt(a.c3_km2_s2 + 2.0 * mu / rp)
    r_vec = rp * e_hat
    v_vec = v_mag * q_hat
    return np.concatenate([r_vec, v_vec])


# ---------------------------------------------------------------------------
# 双曲开普勒时间
# ---------------------------------------------------------------------------


def hyperbolic_time_to_periapsis(state: ArrayLike, mu: float) -> float:
    """双曲状态 → 到近心点的飞行时间 (s)（入射分支，返回正值）。

    双曲开普勒方程：``tanh(H/2) = √((e−1)/(e+1))·tan(ν/2)``，
    ``M = e·sinh(H) − H``，``n = √(μ/|a|³)``，``t = M/n``；入射分支 ν<0
    ⇒ M<0 ⇒ 到近心点时间 ``−M/n > 0``。

    Args:
        state: 相对中心天体的惯性状态 (6,)，km / km/s（双曲）。
        mu: 中心天体引力参数 (km³/s²)。

    Returns:
        到近心点时间 (s)，≥ 0。
    """
    x = np.asarray(state, dtype=np.float64).reshape(6)
    r_vec = x[:3]
    v_vec = x[3:]
    r = float(np.linalg.norm(r_vec))
    v2 = float(v_vec @ v_vec)
    vinf2 = v2 - 2.0 * mu / r
    if vinf2 <= 0.0:
        raise ValueError(f"状态非双曲（v∞²={vinf2:.6e} ≤ 0）")
    e_vec = ((v2 - mu / r) * r_vec - float(r_vec @ v_vec) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))
    e_hat = e_vec / e
    h_hat = np.cross(r_vec, v_vec)
    h_hat = h_hat / float(np.linalg.norm(h_hat))
    q_hat = np.cross(h_hat, e_hat)
    cos_nu = float(e_hat @ r_vec) / r
    sin_nu = float(q_hat @ r_vec) / r
    nu = math.atan2(sin_nu, cos_nu)

    tanh_half = math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu / 2.0)
    tanh_half = float(np.clip(tanh_half, -1.0 + 1e-15, 1.0 - 1e-15))
    hyp_anomaly = 2.0 * math.atanh(tanh_half)
    a_hyp = -mu / vinf2  # a = −μ/v∞²（双曲，负）
    n = math.sqrt(mu / abs(a_hyp) ** 3)
    mean_anomaly = e * math.sinh(hyp_anomaly) - hyp_anomaly
    return -mean_anomaly / n
