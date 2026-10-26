"""地月 L2 halo 族的周期倍化分岔定位与两条子族生成（#821）。

从 L2 halo 族的周期倍化分岔（单值矩阵乘子过负一）起链两条子族：

- 8 字形族（``figure_eight``）：从 x0 约 1.0118 处分岔（母 halo 周期
  1.3743 TU，约 5.98 d），族成员周期约 2.75 到 3.42 TU；x-y 投影为 8
  字，两瓣分居 L1 侧与 L2 侧。文献对照：Grebow 2006 表 3.13（Northern
  L2 Butterfly Family），每周期 2 次垂直 y=0 穿越。
- 0 形族（``ring``）：从 x0 约 1.1093 处分岔（母 halo 周期约 2.765 TU，
  约 12.0 d），出生成员周期 5.527 TU、已验证段到 6.37 TU；x-y 投影为
  椭圆、y-z 投影为闭合圆环。成员每周期 4 次 y=0 穿越：出生极限下全垂
  直（即母 halo 倍周期形态），有限振幅下 2 次垂直加 2 次斜穿越，按
  ADR 0042 判据分类为 halo 南北标签（与 issue 预期的 butterfly 标签不
  同，详见 ``design_ring_family`` docstring 的口径记录；判据不改，族
  名闭值与几何标签并存）。

分岔定位是族生成内部的起链能力，不构成稳定性或分岔分析的接口面或独立
算法层模块（#687 边界，ADR 0060 决策 4 的先例）。

三条分岔判据（同一成员上同时成立，逐条返回数值）：

1. Broucke 图判据 ``f = 2 + β − 2α`` 归零。α、β 为单值矩阵 M 的两个
   Broucke 稳定性标量（Broucke 1969；本文口径取 Li、Tao、Jiang 2022，
   Aerospace 9:454，表 1）：``α = 2 − tr(M)``、
   ``β = ½[α² + 2 − tr(M²)]``。可分解为
   ``f = (s₂ + 2)(s₃ + 2)``（``sᵢ = λᵢ + 1/λᵢ`` 为两个非平凡乘子对
   的迹型指数），任一非平凡乘子对到达负一即严格归零；周期倍化线即
   Broucke 图中的 ``β + 2 = 2α``。f 有符号且跨分岔变号，定位二分直接
   用它。
2. 约化雅可比最小奇异值归零。约化雅可比取 2T 闭合雅可比 ``M(2T) − I``
   的对称扇区 3×3 子块（行列取 XZ 对称形态的自由槽位 x、z、vy；两条子
   族都保持 XZ 对称，分岔方向落在该扇区）。母族切向恒在该子块的核中
   （族在对称扇区内演化），故判据取扣除该恒零核后的最小奇异值（三个
   奇异值中的第二小）。它在正则成员处为 O(0.1) 到 O(1)，在分岔成员处
   因子族方向并入核而归零；反称（面内）扇区在相关成员处谱条件数可达
   1e3，直接用全 6×6 或反称扇区会把成员数值噪声放大到 1e-2 量级，对称
   扇区子块无此问题。
3. 单值矩阵存在特征值负一：``min |λ(M) + 1|`` 归零（平凡乘子对贴着
   +1，不干扰）。

数值口径（地月 DE421 基准，μ = 0.012150585350562453）：

- 0 形族分岔位置高精度复算值为 x0 = 1.109352298816（issue #821 记录，
  库内同基准复核一致，母 halo 周期 2.7653 TU 即 12.0 d）；本实现定位
  值 1.1093522988，与参考值相差 3e-11。
- 8 字形族分岔位置文献值为 x0 = 1.0118（Grebow 2006 表 3.13 首行，四
  位小数，对应舍入量级 5e-5）；本实现定位值 1.011829，与文献值相差
  2.9e-5，在文献舍入量级内但不满足 1e-6 逐位一致（Grebow 使用的
  μ = 0.0121505856 与特征尺度同库内 DE421 基准存在微小差异，四位小数
  文献值本身不承载更高精度，不调参凑数）。
- 定位成员处的三判据实测：f 在 4e-10 到 2e-9、约化奇异值在 3e-11 到
  5e-9、乘子距离在 4e-6 到 4e-5（乘子以 sqrt(参数距离) 速率趋近负一，
  残余距离受 x0 定位精度的平方根下限约束）。
- 同一扫描段（x0 从约 1.148 到 1.005）上共四处 f 过零：除上述两处定
  义分岔外，还有约 1.10872（0 形族实负乘子区间的下端，与 1.109352 相
  隔约 6e-4，同一乘子对的第二次负一交汇）与约 1.06877（面内对的负一
  交汇）。后两处不属于本模块两条族的定义分岔，定位结果一并返回、由
  族入口按参考值就近选用。

成员生成路线（#819 记录的原生路线，全部复用库内既有机器）：分岔成员
沿负一特征向量做 deflection 阶梯加双向穷举，用共享全周期 PAL 内核
（``spatial_full_period_pal_py``，n_revolutions=2，即 tau 取母 halo 整
周期的 2T 闭合）固定 x0 精化种子，非平凡校验（单圈末端不回到初态）
择优后交给同一内核伪弧长行走。成员统一存为最大 x 的垂直 y=0 穿越点最
小形态（单行状态加闭合周期）。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ....data.types.orbit import Orbit, OrbitFamily
from ....status import ConvergenceState
from ...dynamics import CR3BP_Dynamics
from ...results import FamilyGenerationResult
from ..resonant_branch import monodromy, per_revolution_closure
from .halo import _correct_halo_x0, _halo_seed_walk
from .walk import Cr3bpOrbitError, earth_moon_system

#: 8 字形族定义分岔的参考位置（Grebow 2006 表 3.13 首行，四位小数文献值）。
FIGURE_EIGHT_BIFURCATION_X0 = 1.0118

#: 0 形族定义分岔的参考位置（issue #821 高精度复算值，12 位有效数字）。
RING_BIFURCATION_X0 = 1.109352298816

#: 三判据在定位成员处的门限（二分把 f 收敛到宽度 1e-10 后的实测：f 在
#: 4e-10 到 2e-9、约化奇异值在 3e-11 到 5e-9、乘子距离在 4e-6 到 4e-5；
#: 正则成员处 f 与约化奇异值为 O(0.1) 以上、乘子距离为 O(0.1) 以上，
#: 门限各留两个量级以上余量。乘子距离的平方根下限：乘子以 sqrt(参数
#: 距离) 的速率趋近负一，残余距离受 x0 定位精度的平方根下限约束）。
BROUCKE_F_TOLERANCE = 1e-6
REDUCED_SIGMA_TOLERANCE = 1e-6
MULTIPLIER_GAP_TOLERANCE = 5e-4

#: 起链 deflection 幅度阶梯与方向组合（负一特征向量实部、双向）。
_DEFLECTION_LADDER = (1e-4, 3e-4, 1e-3, 3e-3)

#: 种子非平凡判据：单圈末端偏差低于此值说明精化落回了平凡母族。
_TRIVIALITY_TOLERANCE = 1e-8


@dataclass
class PeriodDoublingPoint:
    """母 halo 族上的周期倍化成员与三条判据数值。"""

    #: 分岔成员的初始状态（6 维，y=0 穿越点形态）。
    state: np.ndarray
    #: 分岔成员的单圈周期（TU）。
    period: float
    #: 分岔成员的族参数 x0。
    x0: float
    #: Broucke 图判据 f = 2 + β − 2α（分岔处为零）。
    broucke_f: float
    #: Broucke 稳定性标量 α = 2 − tr(M)。
    broucke_alpha: float
    #: Broucke 稳定性标量 β = ½[α² + 2 − tr(M²)]。
    broucke_beta: float
    #: 约化雅可比（2T 闭合雅可比对称扇区子块）扣除族切向恒零核后的
    #: 最小奇异值。
    reduced_sigma: float
    #: 单值矩阵最接近负一的特征值距离 min |λ + 1|。
    multiplier_gap: float
    #: 负一乘子的特征向量（6 维复向量；实部与虚部张分岔平面，起链
    #: deflection 两个待试方向）。
    eigenvector: np.ndarray


def _broucke_indices(matrix: np.ndarray) -> tuple[float, float]:
    """Broucke 稳定性标量 (α, β) = (2 − tr M, ½[α² + 2 − tr M²])。"""
    alpha = 2.0 - float(np.trace(matrix))
    beta = 0.5 * (alpha * alpha + 2.0 - float(np.trace(matrix @ matrix)))
    return alpha, beta


def broucke_period_doubling_f(matrix: np.ndarray) -> float:
    """Broucke 图周期倍化判据 f = 2 + β − 2α（任一乘子对到负一即归零）。

    f = (s₂ + 2)(s₃ + 2)，sᵢ 为两个非平凡乘子对的迹型指数；跨分岔变
    号，可直接作二分判据。
    """
    alpha, beta = _broucke_indices(matrix)
    return 2.0 + beta - 2.0 * alpha


#: 约化雅可比对称扇区的行列槽位（x、z、vy；XZ 对称形态的自由变量）。
_SYMMETRIC_SLOTS = (0, 2, 4)


def reduced_jacobian_sigma(dynamics: CR3BP_Dynamics, state: np.ndarray, period: float) -> float:
    """2T 闭合雅可比对称扇区子块扣除族切向恒零核后的最小奇异值。

    子块取 ``M(2T) − I`` 的行列 (x, z, vy) 槽位（XZ 对称形态的自由变
    量）。母族在对称扇区内演化，族切向恒为该子块零向量，最小奇异值恒
    近零；返回三个奇异值的第二小者，分岔成员处因子族起链方向并入核而
    归零（正则成员 O(0.1) 以上）。
    """
    matrix = monodromy(dynamics, np.asarray(state, dtype=float), float(period))
    squared = matrix @ matrix
    block = squared[np.ix_(_SYMMETRIC_SLOTS, _SYMMETRIC_SLOTS)] - np.eye(3)
    singular = np.linalg.svd(block, compute_uv=False)
    return float(singular[1])


def _bifurcation_criteria(
    dynamics: CR3BP_Dynamics, orbit: Orbit
) -> tuple[float, np.ndarray, float, float]:
    """成员处的 (f, M, min|λ+1|, 约化奇异值)；M 为单圈单值矩阵。"""
    assert orbit.period is not None
    matrix = monodromy(dynamics, orbit.states[0], float(orbit.period))
    f = broucke_period_doubling_f(matrix)
    gap = float(np.min(np.abs(np.linalg.eigvals(matrix) + 1.0)))
    sigma = reduced_jacobian_sigma(dynamics, orbit.states[0], float(orbit.period))
    return f, matrix, gap, sigma


def _correct_mother_member(
    dynamics: CR3BP_Dynamics, x0: float, libration_point: int, guess: Orbit
) -> Orbit:
    """固定 x0 修正母 halo 成员，带长周期伪解拒绝（同 ``_correct_halo`` 口径）。"""
    orbit = _correct_halo_x0(dynamics, x0, libration_point, guess)
    assert orbit.period is not None and guess.period is not None
    if orbit.period > 1.2 * guess.period:
        raise Cr3bpOrbitError(
            f"Halo(L{libration_point}, x0={x0:.6f}) 修正跳到长周期伪解"
            f"（T={orbit.period:.3f}，初猜 {guess.period:.3f}）"
        )
    return orbit


def _walk_mother_family(
    dynamics: CR3BP_Dynamics,
    libration_point: int,
    z_sign: float,
    *,
    x0_floor: float,
    dx0: float,
    start: Orbit | None = None,
) -> Iterator[Orbit]:
    """母 halo 族固定 x0 向低 x0 方向行走，逐成员产出。

    修正失败（含伪解拒绝）时步长减半重试（下限 1e-5，与
    ``_walk_family`` 一致）。``start`` 给出时从该成员继续（局部加密用）。
    """
    guess = start if start is not None else _halo_seed_walk(dynamics, libration_point, z_sign)
    x0 = float(guess.states[0, 0])
    while x0 > x0_floor:
        step = dx0
        while True:
            try:
                nxt = _correct_mother_member(dynamics, x0 - step, libration_point, guess)
                break
            except Cr3bpOrbitError:
                step *= 0.5
                if step < 1e-5:
                    raise
        x0 -= step
        guess = nxt
        yield nxt


def _closed_period(orbit: Orbit) -> float:
    """取成员闭合周期；本模块族链路上的成员必已填周期（None 属契约破坏）。"""
    assert orbit.period is not None
    return float(orbit.period)


def _bisect_crossing(
    dynamics: CR3BP_Dynamics,
    libration_point: int,
    low_x0: float,
    high_x0: float,
    guess: Orbit,
    *,
    width_tolerance: float = 1e-10,
) -> Orbit:
    """在 ``[low_x0, high_x0]`` 上二分 Broucke f 的变号区间，返回定位成员。"""
    low_orbit = _correct_mother_member(dynamics, low_x0, libration_point, guess)
    low_f = broucke_period_doubling_f(
        monodromy(dynamics, low_orbit.states[0], _closed_period(low_orbit))
    )
    current = low_orbit
    for _ in range(60):
        if high_x0 - low_x0 < width_tolerance:
            break
        mid = 0.5 * (low_x0 + high_x0)
        orbit = _correct_mother_member(dynamics, mid, libration_point, current)
        f = broucke_period_doubling_f(monodromy(dynamics, orbit.states[0], _closed_period(orbit)))
        current = orbit
        if (f > 0.0) == (low_f > 0.0):
            low_x0, low_f = mid, f
        else:
            high_x0 = mid
    x_star = 0.5 * (low_x0 + high_x0)
    return _correct_mother_member(dynamics, x_star, libration_point, current)


#: 粗扫描中触发局部加密的 |f| 门限：远离分岔时 f 为 O(1) 量级，低于该值
#: 且为局部极小处可能藏着宽度小于粗步长的负 f 窄区间（0 形族分岔的实负
#: 乘子区间宽约 6e-4，粗步长 5e-3 需要加密才能分辨）。
_DIP_MAGNITUDE = 0.5


def _locate_crossing_brackets(
    dynamics: CR3BP_Dynamics,
    libration_point: int,
    z_sign: float,
    *,
    x0_floor: float,
    dx0: float,
    fine_dx0: float,
) -> list[tuple[float, float, Orbit]]:
    """扫描母族收集 f 的变号区间 (low, high, 暖启动成员)，按 x0 降序。"""
    members = list(
        _walk_mother_family(dynamics, libration_point, z_sign, x0_floor=x0_floor, dx0=dx0)
    )
    values = [
        (
            float(o.states[0, 0]),
            broucke_period_doubling_f(monodromy(dynamics, o.states[0], _closed_period(o))),
        )
        for o in members
    ]
    brackets: list[tuple[float, float, Orbit]] = []

    def overlaps(lo: float, hi: float) -> bool:
        return any(not (hi < b_lo or lo > b_hi) for b_lo, b_hi, _ in brackets)

    for i, ((xa, fa), (xb, fb)) in enumerate(zip(values, values[1:], strict=False)):
        if (fa > 0.0) != (fb > 0.0):
            brackets.append((xb, xa, members[i + 1]))
    for i in range(1, len(values) - 1):
        xa, fa = values[i - 1]
        xb, fb = values[i]
        xc, fc = values[i + 1]
        if not (abs(fb) < _DIP_MAGNITUDE and abs(fb) < abs(fa) and abs(fb) < abs(fc)):
            continue
        if overlaps(xc, xa):
            continue
        start = members[i - 1]
        fine_members = [
            o
            for o in _walk_mother_family(
                dynamics,
                libration_point,
                z_sign,
                x0_floor=xc,
                dx0=fine_dx0,
                start=start,
            )
            if xc <= float(o.states[0, 0]) <= xa
        ]
        fine_values = [
            (
                float(o.states[0, 0]),
                broucke_period_doubling_f(monodromy(dynamics, o.states[0], _closed_period(o))),
            )
            for o in fine_members
        ]
        for j, ((ya, fya), (yb, fyb)) in enumerate(zip(fine_values, fine_values[1:], strict=False)):
            if (fya > 0.0) != (fyb > 0.0):
                brackets.append((yb, ya, fine_members[j + 1]))
    brackets.sort(key=lambda b: -b[0])
    return brackets


def locate_period_doubling_bifurcations(
    dynamics: CR3BP_Dynamics | None = None,
    *,
    libration_point: int = 2,
    north_south: int = 1,
    dx0: float = 0.005,
    fine_dx0: float = 0.0002,
    x0_floor: float = 1.005,
) -> list[PeriodDoublingPoint]:
    """沿 L2 halo 族定位周期倍化分岔成员（算法层内部能力，不进接口面）。

    从固定 z0 安全校验上界成员（x0 约 1.148）出发固定 x0 向低 x0 行走到
    ``x0_floor``，逐成员算 Broucke f；变号区间二分到 1e-9 宽，粗步长分
    辨不了的窄负区间（0 形族分岔的实负乘子区间宽约 6e-4）按 |f| 局部极
    小触发加密重扫。返回按 x0 降序的全部定位点（该段上有四处过零，见
    模块 docstring；两条族的定义分岔由族入口按参考值就近选用）。

    Raises:
        Cr3bpOrbitError: 族行走失败或该段未找到任何过零。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    if libration_point != 2:
        raise ValueError(f"周期倍化族仅支持 L2，当前 libration_point={libration_point}")
    if north_south not in (1, 2):
        raise ValueError(f"north_south 必须为 1 或 2，当前 {north_south}")
    z_sign = 1.0 if north_south == 1 else -1.0
    brackets = _locate_crossing_brackets(
        dynamics,
        libration_point,
        z_sign,
        x0_floor=x0_floor,
        dx0=dx0,
        fine_dx0=fine_dx0,
    )
    if not brackets:
        raise Cr3bpOrbitError(
            f"沿 L2 halo 族扫描到 x0={x0_floor} 未发现 Broucke f 过零，周期倍化分岔不在该参数区间内"
        )
    points: list[PeriodDoublingPoint] = []
    for low, high, warm in brackets:
        bisected = _bisect_crossing(dynamics, libration_point, low, high, warm)
        f, matrix, gap, sigma = _bifurcation_criteria(dynamics, bisected)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        index = int(np.argmin(np.abs(eigenvalues + 1.0)))
        vector = eigenvectors[:, index]
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            raise Cr3bpOrbitError("负一乘子特征向量为零，起链方向退化")
        alpha, beta = _broucke_indices(matrix)
        points.append(
            PeriodDoublingPoint(
                state=np.asarray(bisected.states[0], dtype=float).copy(),
                period=_closed_period(bisected),
                x0=float(bisected.states[0, 0]),
                broucke_f=float(f),
                broucke_alpha=float(alpha),
                broucke_beta=float(beta),
                reduced_sigma=float(sigma),
                multiplier_gap=float(gap),
                eigenvector=vector / norm,
            )
        )
    return points


def _nearest_point(points: list[PeriodDoublingPoint], reference_x0: float) -> PeriodDoublingPoint:
    """按参考 x0 就近选用定义分岔点。"""
    return min(points, key=lambda p: abs(p.x0 - reference_x0))


def _refine_child_seed(dynamics: CR3BP_Dynamics, state: np.ndarray, closure_period: float):
    """固定 x0 的 2T 闭合阻尼最小二乘精化（仅精化不行走）。"""
    from ....integrators import spatial_full_period_pal_py

    return spatial_full_period_pal_py(
        float(dynamics.system.mu),
        np.asarray(state, dtype=float).tolist(),
        float(closure_period),
        0,
        0.005,
        "increase-x0",
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
        n_revolutions=2,
    )


def _seed_child_branch(
    dynamics: CR3BP_Dynamics, point: PeriodDoublingPoint
) -> tuple[np.ndarray, float]:
    """从分岔成员起链子族种子，返回 (种子状态, 闭合周期)。

    deflection 沿负一特征向量的实部与虚部（分岔平面的两组基）按阶梯从
    小到大、双向穷举，逐个用固定 x0 的 2T 闭合精化，非平凡校验（单圈
    末端偏差）取离开母族最远者为种子。全部组合落回平凡族时报硬失败。
    定位成员处乘子对通常仍为贴着负一的复数对，虚部方向即另一支候选；
    乘子对已为实数时虚部为零、自然跳过。
    """
    closure_period = 2.0 * float(point.period)
    parts = {
        "real": np.real(point.eigenvector),
        "imag": np.imag(point.eigenvector),
    }
    best: tuple[float, np.ndarray, float] | None = None
    for vector in parts.values():
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            continue
        unit = vector / norm
        for epsilon in _DEFLECTION_LADDER:
            for sign in (1.0, -1.0):
                state = np.asarray(point.state, dtype=float) + sign * epsilon * unit
                refined = _refine_child_seed(dynamics, state, closure_period)
                if refined.status is not ConvergenceState.CONVERGED:
                    continue
                seed_state = np.asarray(refined.states[0], dtype=float)
                seed_period = float(refined.periods[0])
                member = Orbit(
                    states=seed_state.reshape(1, -1),
                    times=np.array([0.0]),
                    system=dynamics.system,
                )
                member.period = seed_period
                deviation = per_revolution_closure(dynamics, member, 2)
                if deviation <= _TRIVIALITY_TOLERANCE:
                    continue
                if best is None or deviation > best[0]:
                    best = (deviation, seed_state, seed_period)
    if best is None:
        raise Cr3bpOrbitError(
            "起链失败：负一特征向量 deflection 的全部组合在 2T 闭合精化后"
            "仍落回母族（单圈末端回到初态），子族未被命中"
        )
    return best[1], best[2]


#: 垂直穿越判别门限：穿越处 ``|vx| + |vz|`` 低于此值为垂直穿越。真垂
#: 直穿越在事件精化后残差 1e-10 量级，斜穿越（如 0 形族离出生后的中位
#: 穿越）在 1e-2 量级，两者相隔多个数量级。
_PERPENDICULAR_GATE = 1e-6


def _perpendicular_crossing(
    dynamics: CR3BP_Dynamics, state: np.ndarray, period: float
) -> np.ndarray:
    """成员在最大 x 的垂直 y=0 穿越点处的状态（事件积分精化）。

    先按 ``|vx| + |vz|`` 门限筛出垂直穿越（两族成员都有 2 个），再取其
    中 x 最大者作规范化存储形态；无垂直穿越时取最接近垂直的穿越（防御
    性回退，正常成员不会触发）。
    """

    def surface(_t: float, s: np.ndarray) -> float:
        return float(s[1])

    initial = np.asarray(state, dtype=float)
    out = dynamics.propagate(
        initial,
        (0.0, float(period)),
        events=surface,
        backend="scipy",
    )
    states = np.asarray(out["y_events"][0], dtype=float)
    if abs(float(initial[1])) < 1e-6:
        # 成员相位点常贴着 y=0 截面（相位条件钉在种子穿越方向上）；y(0)
        # 为负时事件窗口 (0, T) 漏掉 t=0 处的穿越，这里显式补入初态。
        states = np.vstack([initial.reshape(1, -1), states])
    if len(states) == 0:
        raise Cr3bpOrbitError("成员周期内没有 y=0 穿越点，垂直穿越形态不可用")
    obliquity = np.abs(states[:, 3]) + np.abs(states[:, 5])
    perpendicular = states[obliquity < _PERPENDICULAR_GATE]
    pool = perpendicular if len(perpendicular) > 0 else states[np.argmin(obliquity)][None, :]
    index = int(np.argmax(pool[:, 0]))
    return np.asarray(pool[index], dtype=float).copy()


def _generate_period_doubled_family(
    dynamics: CR3BP_Dynamics,
    libration_point: int,
    north_south: int,
    *,
    family_type: str,
    reference_x0: float,
    x0_min: float,
    x0_max: float,
    n_orbits: int,
    step_size: float,
    initial_direction: str,
) -> FamilyGenerationResult:
    """两条周期倍化族的共用生成管线：定位、起链、PAL 行走、穿越点规范化。"""
    from ....integrators import spatial_full_period_pal_py

    points = locate_period_doubling_bifurcations(
        dynamics, libration_point=libration_point, north_south=north_south
    )
    point = _nearest_point(points, reference_x0)
    seed_state, seed_period = _seed_child_branch(dynamics, point)
    # PAL 行走器返回种子加 n 条新成员；n_orbits 是含种子的总数上限
    # （与响应契约 generated ≤ requested 一致），故行走请求 n−1 条。
    raw = spatial_full_period_pal_py(
        float(dynamics.system.mu),
        seed_state.tolist(),
        seed_period,
        max(n_orbits - 1, 0),
        step_size,
        initial_direction,
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
        n_revolutions=2,
    )
    orbits: list[Orbit] = []
    for member_state, period, closure_error, drift in zip(
        raw.states, raw.periods, raw.closure_errors, raw.jacobi_drifts, strict=True
    ):
        crossing = _perpendicular_crossing(
            dynamics, np.asarray(member_state, dtype=float), float(period)
        )
        # 族参数（规范化穿越 x）单调走出 [x0_min, x0_max] 即停；种子成员
        # 无条件保留。
        if orbits and not x0_min <= float(crossing[0]) <= x0_max:
            break
        orbit = Orbit(
            states=crossing.reshape(1, -1),
            times=np.array([0.0]),
            system=dynamics.system,
        )
        orbit.period = float(period)
        orbit.family_type = family_type
        orbit.closure_error = float(closure_error)
        orbit.parameters["libration_point"] = libration_point
        orbit.parameters["north_south"] = north_south
        orbit.parameters["bifurcation_x0"] = point.x0
        orbit.parameters["jacobi_drift"] = float(drift)
        orbits.append(orbit)
    family = OrbitFamily(orbits=orbits, family_type=family_type, system=dynamics.system)
    family.metadata["bifurcation_seeding"] = "period-doubling-deflection"
    family.metadata["bifurcation_x0"] = point.x0
    family.metadata["broucke_f"] = point.broucke_f
    family.metadata["multiplier_gap"] = point.multiplier_gap
    family.metadata["reduced_sigma"] = point.reduced_sigma
    return FamilyGenerationResult(
        status=raw.status,
        cause=raw.cause,
        message=raw.message,
        family=family,
        requested_members=n_orbits,
        generated_members=len(family),
    )


def design_figure_eight_family(
    libration_point: int,
    north_south: int,
    min_x0_km: float,
    max_x0_km: float,
    *,
    n_orbits: int = 50,
    continuation_direction: str = "increase-x0",
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成 8 字形族：L2 halo 族 x0 约 1.0118 处周期倍化分岔的子族。

    族参数为成员最大 x 垂直 y=0 穿越点的 x 坐标（km，会合系 DU × 特征
    长度）。母 halo 周期 1.3743 TU（约 5.98 d），子族成员在 2T 上闭合，
    周期约 2.75 到 3.42 TU；x-y 投影为 8 字，两瓣分居 L1 侧与 L2 侧。
    成员每周期 2 次垂直 y=0 穿越，按 ADR 0042 判据分类为 halo 标签
    （族名闭值 figure_eight 与几何标签并存，判据不改）。文献对照见
    Grebow 2006 表 3.13 与 ``scripts/compare_grebow_table_3_13.py``。

    Args:
        libration_point: 平动点编号（仅 2）。
        north_south: 1=北族，2=南族（南族为 z 镜像）。
        min_x0_km: 族参数下界（km）。
        max_x0_km: 族参数上界（km）。
        n_orbits: 族成员数量上限（含种子）。
        continuation_direction: 延拓方向，increase-x0 或 decrease-x0。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。

    Returns:
        :class:`FamilyGenerationResult`；``family`` 为
        ``family_type="figure_eight"`` 的 ``OrbitFamily``，软失败保留部
        分成员。
    """
    return _design_period_doubled_family(
        "figure_eight",
        FIGURE_EIGHT_BIFURCATION_X0,
        libration_point,
        north_south,
        min_x0_km,
        max_x0_km,
        n_orbits=n_orbits,
        continuation_direction=continuation_direction,
        dynamics=dynamics,
        step_size=0.005,
    )


def design_ring_family(
    libration_point: int,
    north_south: int,
    min_x0_km: float,
    max_x0_km: float,
    *,
    n_orbits: int = 50,
    continuation_direction: str = "increase-x0",
    dynamics: CR3BP_Dynamics | None = None,
) -> FamilyGenerationResult:
    """生成 0 形族：L2 halo 族 x0 约 1.1093 处周期倍化分岔的子族。

    族参数为成员最大 x 垂直 y=0 穿越点的 x 坐标（km，会合系 DU × 特征
    长度）。母 halo 周期约 2.765 TU（约 12.0 d），子族成员在 2T 上闭合，
    出生成员周期 5.527 TU，已验证段周期 5.53 到 6.37 TU；x-y 投影为椭圆、
    y-z 投影为闭合圆环（成员跨越南北半球，北族成员也南探到 z 约 −0.06）。

    穿越形态口径（与 issue #821 预期的差异，如实记录）：成员每周期 4 次
    y=0 穿越，其中 2 次垂直、2 次斜穿越。出生极限成员即母 halo 的倍周
    期形态，4 次穿越全垂直；离开出生后两个中位穿越的斜度随族参数立即增
    大（种子成员处 ``|vx| + |vz|`` 约 9e-3，远超 ADR 0042 分类器的
    5e-4 垂直门限），有限振幅成员经 ``classify_orbit`` 分类为 halo 南北
    标签而非 butterfly。分岔两侧的实部与虚部 deflection、大幅 deflection
    与区间内部起链均已穷举验证，该分岔处不存在有限振幅下 4 次全垂直的
    子族；分类判据按裁定不改，族名闭值 ring 与几何标签并存。

    Args:
        libration_point: 平动点编号（仅 2）。
        north_south: 1=北族，2=南族（南族为 z 镜像）。
        min_x0_km: 族参数下界（km）。
        max_x0_km: 族参数上界（km）。
        n_orbits: 族成员数量上限（含种子）。
        continuation_direction: 延拓方向，increase-x0 或 decrease-x0。
        dynamics: CR3BP 动力学；缺省构造标准地月系统。

    Returns:
        :class:`FamilyGenerationResult`；``family`` 为
        ``family_type="ring"`` 的 ``OrbitFamily``，软失败保留部分成员。
    """
    return _design_period_doubled_family(
        "ring",
        RING_BIFURCATION_X0,
        libration_point,
        north_south,
        min_x0_km,
        max_x0_km,
        n_orbits=n_orbits,
        continuation_direction=continuation_direction,
        dynamics=dynamics,
        step_size=0.005,
    )


def _design_period_doubled_family(
    family_type: str,
    reference_x0: float,
    libration_point: int,
    north_south: int,
    min_x0_km: float,
    max_x0_km: float,
    *,
    n_orbits: int,
    continuation_direction: str,
    dynamics: CR3BP_Dynamics | None,
    step_size: float,
) -> FamilyGenerationResult:
    """参数校验加共用管线的族入口壳。"""
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    if libration_point != 2:
        raise ValueError(f"{family_type} 族仅支持 L2，当前 libration_point={libration_point}")
    if north_south not in (1, 2):
        raise ValueError(f"north_south 必须为 1 或 2，当前 {north_south}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits 必须大于 0，当前 {n_orbits}")
    if continuation_direction not in ("increase-x0", "decrease-x0"):
        raise ValueError(
            "continuation_direction 必须为 increase-x0 或 decrease-x0，"
            f"当前 {continuation_direction!r}"
        )
    length = dynamics.system.characteristic_length
    assert length is not None
    x0_min = float(min_x0_km) / float(length)
    x0_max = float(max_x0_km) / float(length)
    if not x0_min < x0_max:
        raise ValueError(f"min_x0_km 必须小于 max_x0_km，当前 {min_x0_km} >= {max_x0_km}")
    return _generate_period_doubled_family(
        dynamics,
        libration_point,
        north_south,
        family_type=family_type,
        reference_x0=reference_x0,
        x0_min=x0_min,
        x0_max=x0_max,
        n_orbits=n_orbits,
        step_size=step_size,
        initial_direction=continuation_direction,
    )


__all__ = [
    "FIGURE_EIGHT_BIFURCATION_X0",
    "RING_BIFURCATION_X0",
    "BROUCKE_F_TOLERANCE",
    "REDUCED_SIGMA_TOLERANCE",
    "MULTIPLIER_GAP_TOLERANCE",
    "PeriodDoublingPoint",
    "broucke_period_doubling_f",
    "design_figure_eight_family",
    "design_ring_family",
    "locate_period_doubling_bifurcations",
    "reduced_jacobian_sigma",
]
