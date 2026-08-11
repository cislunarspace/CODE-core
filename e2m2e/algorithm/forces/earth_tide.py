"""固体潮修正（天体无关 Step1 + 地球专用 Step2/极潮/永久潮）。

``solid_tide_step1`` 天体无关:对任意中心天体,把扰动体位置 + 该天体的 Love
数表喂进去即可算 ΔC/ΔS(对齐 GMAT ``HarmonicGravity::IncrementSolidTide``)。
Step2(频率相关)、极潮、永久潮修正均为地球专用,保留原样。

公式与系数取自 IERS Technical Note 32 (Conventions 2003)，与 GMAT R2026a
``HarmonicGravity`` 对齐。单位一致：位置 km、GM km³/s²、参考半径 km。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from ...data.constants import DAYS_PER_JULIAN_CENTURY

# ----------------------------------------------------------------------------
# Love 数(GMAT LM_SetDefaultEarthTide 默认值,硬编码)
# KEarth[n][m]:n=2,3 阶位移 Love 数;n=0,1,4 为零
# KPlusEarth[m]:n=2 时的弹性 3 阶位移(LoveMax=4,GMAT 数组)
# ----------------------------------------------------------------------------

_K_EARTH: npt.NDArray[np.floating] = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],  # n=0
        [0.0, 0.0, 0.0, 0.0, 0.0],  # n=1
        [0.30190, 0.29830, 0.30102, 0.0, 0.0],  # n=2(K20,K21,K22)
        [0.093, 0.093, 0.093, 0.094, 0.0],  # n=3
        [0.0, 0.0, 0.0, 0.0, 0.0],  # n=4
    ],
    dtype=float,
)

_K_PLUS_EARTH: npt.NDArray[np.floating] = np.array(
    [-0.00087, -0.00079, -0.00057, 0.0, 0.0], dtype=float
)

# ----------------------------------------------------------------------------
# 月球 Love 数:从 grgm900c.tide 读取(k 2 m value 格式)。月球只有 k₂=0.024116
# 三项(m=0,1,2),没有弹性 3 阶位移(k_plus=None)。k₂≈0.024 比地球 0.30 小一阶,
# 潮汐效应是二阶小量。
# ----------------------------------------------------------------------------

# Love 数表的阶数固定为 5(n=0..4),与地球 _K_EARTH 对齐,便于共用 solid_tide_step1。
_LOVE_TABLE_SIZE = 5


def load_love_number_file(path: str | Path) -> npt.NDArray[np.floating]:
    """读取 GMAT 风格 Love 数文件(如 ``grgm900c.tide``)。

    文件格式:每行 ``k <n> <m> <value>``(可带 ``%`` 注释行与空行)。返回
    ``_LOVE_TABLE_SIZE``×``_LOVE_TABLE_SIZE`` 的 ``K[n][m]`` 表,n=0,1,4 行
    默认为零;文件中未给出的项也为零。

    Args:
        path: Love 数文件路径。

    Returns:
        ``K[n][m]`` 表,形状 (5,5)。
    """
    path = Path(path)
    k = np.zeros((_LOVE_TABLE_SIZE, _LOVE_TABLE_SIZE), dtype=float)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4 or parts[0].lower() != "k":
            continue
        n = int(parts[1])
        m = int(parts[2])
        val = float(parts[3])
        if 0 <= n < _LOVE_TABLE_SIZE and 0 <= m < _LOVE_TABLE_SIZE:
            k[n, m] = val
    return k


# ----------------------------------------------------------------------------
# 时间与角度常量
# ----------------------------------------------------------------------------

_JD_J2000 = 2451545.0
_RAD_PER_DEG = np.pi / 180.0

# ----------------------------------------------------------------------------
# Table 6.3(IERS TN32 p.64/66,迁移 GMAT HarmonicGravity.cpp 静态数组)
# 每行:[l, l', F, D, O, ...系数]
#   l,l' : Doodson 幅角阶数(用于幅角组合)
#   F,D,O: 月亮升交点经度 F、月亮平近点角... 实际每行前 5 列是 5 个 Delaunay
#          幅角的整数乘子(N1..N5 对应 F[0..4])
#   ip,op: C/S(或 cos/sin)的同相、正交系数(单位 1e-12)
# Table63c 只有 amp(1 列),(2,2) 公式不同
# ----------------------------------------------------------------------------

# Table63a:48 行,C21/S21。列 [N1,N2,N3,N4,N5, ip, op]
_TABLE_63A: npt.NDArray[np.floating] = np.array(
    [
        [2, 0, 2, 0, 2, -0.1, 0],
        [0, 0, 2, 2, 2, -0.1, 0],
        [1, 0, 2, 0, 1, -0.1, 0],
        [1, 0, 2, 0, 2, -0.7, 0.1],
        [-1, 0, 2, 2, 2, -0.1, 0],
        [0, 0, 2, 0, 1, -1.3, 0.1],
        [0, 0, 2, 0, 2, -6.8, 0.6],
        [0, 0, 0, 2, 0, 0.1, 0],
        [1, 0, 2, -2, 2, 0.1, 0],
        [-1, 0, 2, 0, 1, 0.1, 0],
        [-1, 0, 2, 0, 2, 0.4, 0],
        [1, 0, 0, 0, 0, 1.3, -0.1],
        [1, 0, 0, 0, 1, 0.3, 0],
        [-1, 0, 0, 2, 0, 0.3, 0],
        [-1, 0, 0, 2, 1, 0.1, 0],
        [0, 1, 2, -2, 2, -1.9, 0.1],
        [0, 0, 2, -2, 1, 0.5, 0],
        [0, 0, 2, -2, 2, -43.4, 2.9],
        [0, -1, 2, -2, 2, 0.6, 0],
        [0, 1, 0, 0, 0, 1.6, -0.1],
        [-2, 0, 2, 0, 1, 0.1, 0],
        [0, 0, 0, 0, -2, 0.1, 0],
        [0, 0, 0, 0, -1, -8.8, 0.5],
        [0, 0, 0, 0, 0, 470.9, -30.2],
        [0, 0, 0, 0, 1, 68.1, -4.6],
        [0, 0, 0, 0, 2, -1.6, 0.1],
        [-1, 0, 0, 1, 0, 0.1, 0],
        [0, -1, 0, 0, -1, -0.1, 0],
        [0, -1, 0, 0, 0, -20.6, -0.3],
        [0, 1, -2, 2, -2, 0.3, 0],
        [0, -1, 0, 0, 1, -0.3, 0],
        [-2, 0, 0, 2, 0, -0.2, 0],
        [-2, 0, 0, 2, 1, -0.1, 0],
        [0, 0, -2, 2, -2, -5.0, 0.3],
        [0, 0, -2, 2, -1, 0.2, 0],
        [0, -1, -2, 2, -2, -0.2, 0],
        [1, 0, 0, -2, 0, -0.5, 0],
        [1, 0, 0, -2, 1, -0.1, 0],
        [-1, 0, 0, 0, -1, 0.1, 0],
        [-1, 0, 0, 0, 0, -2.1, 0.1],
        [-1, 0, 0, 0, 1, -0.4, 0],
        [0, 0, 0, -2, 0, -0.2, 0],
        [-2, 0, 0, 0, 0, -0.1, 0],
        [0, 0, -2, 0, -2, -0.6, 0],
        [0, 0, -2, 0, -1, -0.4, 0],
        [0, 0, -2, 0, 0, -0.1, 0],
        [-1, 0, -2, 0, -2, -0.1, 0],
        [-1, 0, -2, 0, -1, -0.1, 0],
    ],
    dtype=float,
)

# Table63b:21 行,C20。列 [N1,N2,N3,N4,N5, ip, op]
_TABLE_63B: npt.NDArray[np.floating] = np.array(
    [
        [0, 0, 0, 0, 1, 16.6, -6.7],
        [0, 0, 0, 0, 2, -0.1, 0.1],
        [0, -1, 0, 0, 0, -1.2, 0.8],
        [0, 0, -2, 2, -2, -5.5, 4.3],
        [0, 0, -2, 2, -1, 0.1, -0.1],
        [0, -1, -2, 2, -2, -0.3, 0.2],
        [1, 0, 0, -2, 0, -0.3, 0.7],
        [-1, 0, 0, 0, -1, 0.1, -0.2],
        [-1, 0, 0, 0, 0, -1.2, 3.7],
        [-1, 0, 0, 0, 1, 0.1, -0.2],
        [1, 0, -2, 0, -2, 0.1, -0.2],
        [0, 0, 0, -2, 0, 0.0, 0.6],
        [-2, 0, 0, 0, 0, 0.0, 0.3],
        [0, 0, -2, 0, -2, 0.6, 6.3],
        [0, 0, -2, 0, -1, 0.2, 2.6],
        [0, 0, -2, 0, 0, 0.0, 0.2],
        [1, 0, -2, -2, -2, 0.1, 0.2],
        [-1, 0, -2, 0, -2, 0.4, 1.1],
        [-1, 0, -2, 0, -1, 0.2, 0.5],
        [0, 0, -2, -2, -2, 0.1, 0.2],
        [-2, 0, -2, 0, -2, 0.1, 0.1],
    ],
    dtype=float,
)

# Table63c:2 行,C22/S22。列 [N1,N2,N3,N4,N5, amp]
_TABLE_63C: npt.NDArray[np.floating] = np.array(
    [
        [1, 0, 2, 0, 2, -0.3],
        [0, 0, 2, 0, 2, -1.2],
    ],
    dtype=float,
)


def _delaunay_args_and_gmst(jd: float) -> tuple[npt.NDArray[np.floating], float]:
    """计算 5 个 Delaunay 幅角(度)与 GMST(度)。

    公式取自 IERS TN32 p.48/60，与 GMAT IncrementEarthTide 幅角段对齐。
    用 et→JD(TDB) 近似，忽略 TDB 与 TDT 的秒级差异（潮汐是日级振荡，
    该近似在框架层面可接受）。

    Args:
        jd: 儒略日(TDB 近似)。

    Returns:
        (F, GMST):F 为长度 5 的数组(度),GMST 为标量(度)。
    """
    t = (jd - _JD_J2000) / DAYS_PER_JULIAN_CENTURY
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t

    # 5 个 Delaunay 幅角(度)
    F = np.zeros(5, dtype=float)
    F[0] = (
        134.96340251e3 + 1717915923.2178 * t + 31.8792 * t2 + 0.051635 * t3 - 0.00024470 * t4
    ) / 3600.0
    F[1] = (
        357.52910918e3 + 129596581.0481 * t - 0.5532 * t2 + 0.000136 * t3 - 0.00001149 * t4
    ) / 3600.0
    F[2] = (
        93.27209062e3 + 1739527262.8478 * t - 12.7512 * t2 - 0.001037 * t3 + 0.00000417 * t4
    ) / 3600.0
    F[3] = (
        297.85019547e3 + 1602961601.2090 * t - 6.3706 * t2 + 0.006593 * t3 - 0.00003169 * t4
    ) / 3600.0
    F[4] = (
        125.04455501e3 - 6962890.5431 * t + 7.4722 * t2 + 0.007702 * t3 - 0.00005939 * t4
    ) / 3600.0

    # GMST(IERS p.60):先算秒,再 /240 转度
    gmst_sec = 67310.54841 + 3164400184.812866 * t + 0.093104 * t2 - 6.2e-06 * t3
    gmst_deg = gmst_sec / 240.0

    return F, gmst_deg


def solid_tide_step2(et: float) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """固体潮 Step 2（频率相关）。

    只影响 (2,0)/(2,1)/(2,2)。用 5 个 Delaunay 幅角 + GMST + Table 6.3a/b/c。
    量级 ~1e-10（GMAT ``freq_dep * 1e-12`` 缩放）。

    实现：1:1 移植到 Rust（``crates/e2m2e-integrators/src/solid_tide.rs``），
    Python 侧仅做一次 FFI 调用与 reshape。精度回归：< 1e-15（机器精度）。

    Args:
        et: SPICE et 秒(past J2000)。

    Returns:
        (DeltaC, DeltaS),各为 5×5 数组;仅 (2,0)/(2,1)/(2,2) 非零。
    """
    from e2m2e.integrators import solid_tide_step2 as _rust_step2

    out = _rust_step2(float(et))
    arr = np.asarray(out, dtype=float)
    return arr[:25].reshape(5, 5), arr[25:].reshape(5, 5)


def _legendre_23(s: float, c: float) -> npt.NDArray[np.floating]:
    """计算 n=2,3 的完全正规化 associated Legendre(与 GMAT PolarToLegendre 一致)。

    Args:
        s: sin(地心纬度)。
        c: cos(地心纬度)。

    Returns:
        5×5 数组,P[n][m],n,m ∈ {0..4};未计算的项为零。
    """
    P = np.zeros((5, 5), dtype=float)
    sqrt5 = np.sqrt(5.0)
    sqrt5over3 = np.sqrt(5.0 / 3.0)
    sqrt7 = np.sqrt(7.0)
    sqrt7over6 = np.sqrt(7.0 / 6.0)
    sqrt7over15 = np.sqrt(7.0 / 15.0)
    sqrt_point7 = np.sqrt(0.7)

    # n=2
    P[2, 0] = sqrt5 * (1.5 * s * s - 0.5)
    P[2, 1] = 3.0 * sqrt5over3 * c * s
    P[2, 2] = 1.5 * sqrt5over3 * c * c
    # n=3
    P[3, 0] = sqrt7 * (2.5 * s * s * s - 1.5 * s)
    P[3, 1] = sqrt7over6 * c * (7.5 * s * s - 1.5)
    P[3, 2] = 7.5 * sqrt7over15 * c * c * s
    P[3, 3] = 2.5 * sqrt_point7 * c * c * c
    return P


def solid_tide_step1(
    perturbers: list[tuple[npt.ArrayLike, float]] | tuple[npt.ArrayLike, float],
    k_love: npt.NDArray[np.floating],
    k_plus: npt.NDArray[np.floating] | None,
    mu_central: float,
    r_central: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """固体潮 Step 1（频率无关，天体无关）。

    对一组扰动天体累加 ΔC/ΔS。公式本身天体无关（IERS TN32 eqn 1, p.59；
    eqn 4, p.60），Love 数由调用方按中心天体传入。

    对每个扰动天体:
        ΔC[n][m] += K[n][m]/(2n+1) · (μ_p/μ_c) · (R_c/r)^(n+1) · P_nm(sinφ) · cos(mλ)
        ΔS[n][m] += K[n][m]/(2n+1) · (μ_p/μ_c) · (R_c/r)^(n+1) · P_nm(sinφ) · sin(mλ)
    n=2 且 ``k_plus`` 非空时额外（弹性 Love 数，3 阶位移，IERS eqn 4）:
        ΔC[4][m] += KPlus[m]/5 · ... ; ΔS[4][m] += KPlus[m]/5 · ...

    与 GMAT ``HarmonicGravity::IncrementEarthTide`` 对齐:其内部对 Sun、Moon
    各调一次 ``IncrementSolidTide`` 并累加;本函数把"逐体累加"内化,调用方
    一次性传完整扰动体列表。

    Args:
        perturbers: 扰动天体列表 ``[(position, gm), ...]``,``position`` 为扰动
            体相对中心天体的位置(中心天体 body-fixed 系,如 ITRF93/MOON_PA),
            形状 (3,)、单位 km;``gm`` 为扰动体 GM(km³/s²)。也接受单个
            ``(position, gm)`` 元组以兼容旧调用方。
        k_love: n=2,3 阶位移 Love 数表 ``K[n][m]``,形状 (5,5)(n=0,1,4 行用零
            填充,与 GMAT LoveMax+1=5 对齐)。地球用 ``_K_EARTH``。
        k_plus: n=2 时的弹性 3 阶位移 ``KPlus[m]``,形状 (5,)(m>2 处为零)。
            地球用 ``_K_PLUS_EARTH``;月球等无此贡献时传 ``None``。
        mu_central: 中心天体 GM,km³/s²。
        r_central: 中心天体参考半径,km。

    Returns:
        (DeltaC, DeltaS),各为 5×5 数组。

    实现：1:1 移植到 Rust（``crates/e2m2e-integrators/src/solid_tide.rs``）。
    Python 侧仅做参数打包（perturbers → [px,py,pz,gm,...] 扁平化）+ reshape。
    """
    from e2m2e.integrators import solid_tide_step1 as _rust_step1

    # 兼容单个 (position, gm) 元组:不期望第一个元素本身是 (3,) ndarray。
    if isinstance(perturbers, tuple) and len(perturbers) == 2 and np.isscalar(perturbers[1]):
        perturbers = [perturbers]

    # 扁平化 perturbers: [px, py, pz, gm, px, py, pz, gm, ...]
    flat: list[float] = []
    perturber_list: list[tuple[npt.ArrayLike, float]] = (
        list(perturbers) if isinstance(perturbers, list) else [perturbers]
    )
    for pos_perturber, mu_perturber in perturber_list:
        pos = np.asarray(pos_perturber, dtype=float).ravel()
        if pos.size != 3:
            raise ValueError(f"perturber position must have 3 elements, got {pos.size}")
        flat.extend(pos.tolist())
        flat.append(float(mu_perturber))

    k_love_flat = np.asarray(k_love, dtype=float).ravel().tolist()
    k_plus_flat = None if k_plus is None else np.asarray(k_plus, dtype=float).ravel().tolist()

    out = _rust_step1(
        flat,
        k_love_flat,
        k_plus_flat,
        float(mu_central),
        float(r_central),
    )
    arr = np.asarray(out, dtype=float)
    return arr[:25].reshape(5, 5), arr[25:].reshape(5, 5)


def pole_tide(
    et: float,
    xp: float,
    yp: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """极潮（固体极潮 IERS p.65 + Desai 海洋极潮 TN32 §6.3）。

    只影响 (2,1)。对齐 GMAT ``ETide::SolidAndPole`` 档（固体极潮 + 海洋极潮都做）；
    ``Solid`` 档不做极潮。

    公式:
        ym2000 = (JD - JD_J2000) / 365.25
        xp_bar = 0.054 + ym2000·0.00083  (IERS p.84 mean pole)
        yp_bar = 0.357 + ym2000·0.00395
        m1 = xp - xp_bar;m2 = -(yp - yp_bar)
        固体极潮:ΔC21 -= 1.333e-9·(m1+0.0115·m2);ΔS21 -= 1.333e-9·(m2-0.0115·m1)
        海洋极潮:ΔC21 -= 2.2344e-10·(m1-0.01737·m2);ΔS21 -= 1.7680e-10·(m2-0.03351·m1)

    Args:
        et: SPICE et 秒(past J2000)。
        xp: 极移 x 分量(arcsec,IERS EOP C04)。
        yp: 极移 y 分量(arcsec)。

    Returns:
        (DeltaC, DeltaS),各为 5×5 数组;仅 (2,1) 非零。

    实现：1:1 移植到 Rust（``crates/e2m2e-integrators/src/solid_tide.rs``）。
    """
    from e2m2e.integrators import pole_tide as _rust_pole_tide

    out = _rust_pole_tide(float(et), float(xp), float(yp))
    arr = np.asarray(out, dtype=float)
    return arr[:25].reshape(5, 5), arr[25:].reshape(5, 5)


def permanent_tide_correction(
    mu_sun: float,
    mu_moon: float,
    mu_earth: float,
    r_earth: float,
    a_sun: float,
    a_moon: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """永久潮汐修正（IERS TN32 Step 3，时间平均）。

    zero-tide 系数约定下，``GravityField`` 在叠加固体潮后减去此值——因为
    zero-tide 系数已含永久潮汐，运行时若再加完整固体潮（含永久分量）会重复。

    用 ``solid_tide_step1`` 在 Sun/Moon 半长轴距离 + 赤道（零纬度，时间平均
    近似）计算。GMAT 把永久潮汐处理放在系数加载 setup；e2m2e 用运行时减除，
    公式等价。

    Args:
        mu_sun, mu_moon: Sun/Moon GM，km³/s²。
        mu_earth: 地球 GM，km³/s²。
        r_earth: 地球参考半径，km。
        a_sun, a_moon: Sun/Moon 轨道半长轴，km（时间平均距离近似）。

    Returns:
        (DeltaC, DeltaS)，各为 5×5 数组。
    """
    sun_pos = np.array([a_sun, 0.0, 0.0])
    moon_pos = np.array([a_moon, 0.0, 0.0])
    return solid_tide_step1(
        [(sun_pos, mu_sun), (moon_pos, mu_moon)],
        k_love=_K_EARTH,
        k_plus=_K_PLUS_EARTH,
        mu_central=mu_earth,
        r_central=r_earth,
    )
