"""周期轨道邻域相对运动的 Floquet 模态基础解集与有界性判据（主题 3）。

给定一条带周期的 CR3BP 周期轨道，沿轨道传播一个周期得到单值矩阵
M = Φ(T, t₀)，把它的六个特征值组织为模态基础解集：

- 切向链：λ = 1 的二重亏损根。相位方向 v1 = ẋ(t₀) 是 M 的平凡特征向量
  （精确恒等 M v1 = v1），广义特征向量 v2 由 (M − I) v2 = v1 的最小二乘
  解投影得到。v1 方向的解是周期解（沿轨道的相位平移），v2 方向的解线性
  增长（沿轨道族移动带来的相位漂移）。
- 面内对与法向对：模在单位圆上的共轭复对 λ = e^(±iα)。由特征向量
  w = a + i b 构造周期化模态::

      u(t) = cos(αt/T) Φ(t, t₀) a + sin(αt/T) Φ(t, t₀) b
      v(t) = −sin(αt/T) Φ(t, t₀) a + cos(αt/T) Φ(t, t₀) b

  二者均为 T 周期函数，把时变线性相对运动化为常系数旋转。初值为 a 时
  相对运动满足 x(t) = cos(θ) u(t) − sin(θ) v(t)，初值为 b 时满足
  x(t) = sin(θ) u(t) + cos(θ) v(t)，θ = αt/T；子空间内一般初值
  c₁a + c₂b 按这两支线性组合。面内与法向的归属取自分块结构：CR3BP 的
  平面子空间不变，单值矩阵在面内块（x、y、vx、vy）与法向块（z、vz）
  上解耦，两块分别特征分解，块归属即 kind。本模块只支持平面周期轨道：
  跨块耦合超过门限的三维轨道确定性报错，不按分量支撑排序贴标签。

``basis[i, :, k]`` 为第 k 个模态在 ``times[i]`` 的取值，列序为
[切向, 面内 u, 面内 v, 法向 u, 法向 v, 漂移]。前五个模态张成有界子空间，
第六列为漂移方向；相对运动长期有界的充要条件是初值不含漂移分量，即一
个线性约束 n·x(t₀) = 0，n 为 ``bounded_normal``（对模态初值矩阵取逆的
漂移对偶行，n·v2 = 1）。

数值口径：亏损的平凡对在 numpy.linalg.eig 下数值分裂为 1 ± 1e-6 量级的
近实对（``family/resonant_branch`` 记录过同一现象），因此切向方向不取数
值特征向量，用解析相位方向 ẋ(t₀) 与 |M v1 − v1| 残差把关；偏离单位模的
复对或 λ ≠ 1 的非平凡实对（不稳定周期轨道的双曲结构）不在本模块形态内，
确定性报错，留给后续扩展。三维周期轨道同样拒绝：其单值矩阵的面内块与
法向块不分离，两对特征向量的分量互相混杂，按支撑排序贴标签会静默出错，
形态内的三维模态构造留给后续工单。

论文对照（杨驰航等，航空学报 2023 年 44 卷 5 期 326563）：式(13) 的
e3(0) 按论文的随动导数速度变换可独立复算，作为定义性 oracle 由
``tests/algorithm/proximity/test_lvlh_reference_dro21.py`` 断言，最大
残差约 2.0e-4，容差 1e-3。特征值与有界条件属于结果性对照，只进
docstring，不进断言（ADR 0055）：2:1 地月 DRO 取月心近侧穿越 −0.192 LU
（会合系 x0 = 1 − μ − 0.192），周期约 14.75 d，面内对
λ1,2 ≈ −0.7422 ± 0.6702i，法向对 λ5,6 ≈ 0.1387 ± 0.9903i，与本文复现
成员（λ1,2 = −0.7424 ± 0.6699i、λ5,6 = 0.1392 ± 0.9903i）在前三位小数
一致；末位差异属复现陷阱，特征值末位随轨道成员的钉住位置（初值精度）
漂移，本文钉住月心穿越精确 −0.192 LU，论文成员的收敛位置略有不同。论
文另给出有界条件 ẋ(0) ≈ −2.04 x(0)；本模块在该成员上给出的充要约束化
为 δvy = −0.73 δx（δvx 不受约束），与论文数值不属同一初值参数化，
未作对齐。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from ...exceptions import E2M2EError

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...algorithm.dynamics import CR3BP_Dynamics
    from ...data.types.orbit import Orbit

#: 平凡乘子贴近 1 的判定容差（亏损双根的数值分裂在 1e-6 量级）。
_TRIVIAL_EIGENVALUE_TOL = 1e-4

#: 解析相位方向的恒等残差 |M v1 − v1| 门限，超过即非平凡结构。
_TANGENT_RESIDUAL_TOL = 1e-6

#: 非平凡复对的单位模容差，超过视为双曲形态。
_UNIT_MODULUS_TOL = 1e-3

#: 复对旋转恒等 M a = cosα·a − sinα·b 的校验门限。
_ROTATION_TOL = 1e-6

#: 面内块（x、y、vx、vy）与法向块（z、vz）的跨块耦合门限。平面轨道的
#: 两块精确解耦（实测跨块元素为零），三维轨道在 1e2 量级。
_PLANAR_DECOUPLING_TOL = 1e-8

#: 广义特征向量投影后的范数下限，低于此值说明切向链退化。
_DRIFT_NORM_TOL = 1e-8

#: 平面块与法向块的分量下标。
_PLANAR_COMPONENTS = (0, 1, 3, 4)
_NORMAL_COMPONENTS = (2, 5)

#: 默认采样网格点数（恰好覆盖一个周期，含两端）。
_DEFAULT_GRID_POINTS = 201


class FloquetModeError(E2M2EError, ValueError):
    """Floquet 模态构造的确定性失败（输入或特征结构非预期）。"""


class FloquetPairKind(Enum):
    """共轭复对的分类。"""

    IN_PLANE = "in_plane"
    OUT_OF_PLANE = "out_of_plane"


@dataclass(frozen=True)
class FloquetPair:
    """一对单位模共轭复乘子及其周期化基向量。

    Attributes:
        kind: 分类（面内或法向），按单值矩阵的面内块与法向块归属确定。
        eigenvalue: 辐角为正的复乘子。
        alpha: 每周期转角（rad），等于 eigenvalue 的辐角，落在 (0, π)。
        a: 复特征向量的实部（复向量整体单位化，最大模分量旋为正实）。
        b: 复特征向量的虚部。
    """

    kind: FloquetPairKind
    eigenvalue: complex
    alpha: float
    a: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class FloquetModeSet:
    """周期轨道邻域的 Floquet 模态基础解集。

    Attributes:
        state0: 相位原点处的轨道状态（6 维，无量纲）。
        t_origin: 相位原点（轨道 times 首点）。
        period: 轨道周期（TU）。
        monodromy: 单值矩阵 Φ(T, t₀)，形状 ``(6, 6)``。
        eigenvalues: 单值矩阵特征值（numpy.linalg.eig 原始次序）。
        tangent: 切向单位向量 v1 = ẋ(t₀)/|ẋ(t₀)|（周期解初值）。
        drift: 漂移单位向量 v2（广义特征向量，线性增长解初值）。
        in_plane: 面内共轭复对。
        out_of_plane: 法向共轭复对。
        times: 采样网格，从 t_origin 到 t_origin + period。
        basis: 模态采样，形状 ``(n, 6, 6)``，列序为
            [切向, 面内 u, 面内 v, 法向 u, 法向 v, 漂移]。
        bounded_normal: 有界子空间法向量 n，约束 n·x(t₀) = 0 与相对运动
            长期有界互为充要（n·v2 = 1）。
    """

    state0: np.ndarray
    t_origin: float
    period: float
    monodromy: np.ndarray
    eigenvalues: np.ndarray
    tangent: np.ndarray
    drift: np.ndarray
    in_plane: FloquetPair
    out_of_plane: FloquetPair
    times: np.ndarray
    basis: np.ndarray
    bounded_normal: np.ndarray


def floquet_modes(
    orbit: Orbit,
    dynamics: CR3BP_Dynamics,
    *,
    times: npt.ArrayLike | None = None,
) -> FloquetModeSet:
    """构造周期轨道邻域相对运动的 Floquet 模态基础解集。

    Args:
        orbit: 平面周期轨道，``period`` 必须已设置为单圈周期真值；只读取
            首点状态与首点时刻，单点轨道即可。
        dynamics: 地月 CR3BP 动力学，用于整周期 STM 传播与状态方程求值。
        times: 采样网格，须严格递增、起于轨道 times 首点、止于首点加一个
            周期；缺省为等间距 201 点。

    Returns:
        :class:`FloquetModeSet`。

    Raises:
        FloquetModeError: ``period`` 未设置或非正、网格不合约、单值矩阵缺
            少平凡切向结构（周期偏离真值或轨道未收敛）、单值矩阵面内块与
            法向块不分离（三维周期轨道不在本模块形态内）、块内非平凡特征
            值不是单位模共轭复对（不稳定轨道的双曲结构不在形态内）。
    """
    if orbit.period is None or float(orbit.period) <= 0.0:
        raise FloquetModeError("orbit.period 未设置或非正，须先给出单圈周期真值")
    period = float(orbit.period)
    state0 = np.asarray(orbit.states[0], dtype=float)
    t_origin = float(orbit.times[0])
    grid = _validated_grid(times, t_origin, period)

    result = dynamics.propagate(state0, (t_origin, t_origin + period), t_eval=grid, with_stm=True)
    stm = np.asarray(result["stm"], dtype=float)
    monodromy = stm[-1]

    tangent = _tangent_direction(dynamics, t_origin, state0, monodromy)
    drift = _drift_direction(monodromy, tangent)
    eigenvalues, _ = np.linalg.eig(monodromy)
    in_plane, out_of_plane = _classify_pairs(eigenvalues, monodromy)

    basis = np.empty((grid.shape[0], 6, 6))
    basis[:, :, 0] = stm @ tangent
    basis[:, :, 5] = stm @ drift
    for pair, col_u, col_v in ((in_plane, 1, 2), (out_of_plane, 3, 4)):
        theta = pair.alpha * (grid - t_origin) / period
        cos, sin = np.cos(theta), np.sin(theta)
        phi_a = stm @ pair.a
        phi_b = stm @ pair.b
        basis[:, :, col_u] = cos[:, None] * phi_a + sin[:, None] * phi_b
        basis[:, :, col_v] = -sin[:, None] * phi_a + cos[:, None] * phi_b

    # 有界子空间法向量：模态初值矩阵（basis[0]，Φ(t₀)=I）的漂移对偶行，
    # 满足 n·ê_k(t₀) = 0（k 取前五列）且 n·v2 = 1。
    bounded_normal = np.linalg.solve(basis[0].T, np.array([0.0, 0, 0, 0, 0, 1.0]))

    return FloquetModeSet(
        state0=state0,
        t_origin=t_origin,
        period=period,
        monodromy=monodromy,
        eigenvalues=eigenvalues,
        tangent=tangent,
        drift=drift,
        in_plane=in_plane,
        out_of_plane=out_of_plane,
        times=grid,
        basis=basis,
        bounded_normal=bounded_normal,
    )


def _validated_grid(times: npt.ArrayLike | None, t_origin: float, period: float) -> np.ndarray:
    """校验并返回采样网格：严格递增，起于 t_origin 止于 t_origin + period。"""
    if times is None:
        return np.linspace(t_origin, t_origin + period, _DEFAULT_GRID_POINTS)
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.shape[0] < 2:
        raise FloquetModeError("采样网格须为一维且至少两个点")
    if not np.all(np.diff(grid) > 0.0):
        raise FloquetModeError("采样网格须严格递增")
    if abs(grid[0] - t_origin) > 1e-9 or abs(grid[-1] - (t_origin + period)) > 1e-9:
        raise FloquetModeError(
            "采样网格须从相位原点起、止于原点加一个周期，"
            f"当前为 [{grid[0]:.6f}, {grid[-1]:.6f}]，"
            f"要求 [{t_origin:.6f}, {t_origin + period:.6f}]"
        )
    return grid


def _tangent_direction(
    dynamics: CR3BP_Dynamics, t_origin: float, state0: np.ndarray, monodromy: np.ndarray
) -> np.ndarray:
    """解析相位方向 v1 = ẋ(t₀)，并把关单值矩阵的平凡切向结构。"""
    tangent_raw = np.asarray(dynamics.equations_of_motion(t_origin, state0), dtype=float)
    norm = float(np.linalg.norm(tangent_raw))
    if norm < 1e-12:
        raise FloquetModeError("相位方向 ẋ(t₀) 为零向量，轨道状态退化")
    tangent = tangent_raw / norm
    residual = float(np.linalg.norm(monodromy @ tangent - tangent))
    if residual > _TANGENT_RESIDUAL_TOL:
        raise FloquetModeError(
            f"切向恒等 |M v1 − v1| = {residual:.2e} 超过门限 {_TANGENT_RESIDUAL_TOL:.0e}，"
            "单值矩阵无平凡切向结构；确认 period 为该周期解的真值且轨道已收敛"
        )
    return tangent


def _drift_direction(monodromy: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """广义特征向量 v2：(M − I) v2 = v1 的最小二乘解，投影去 v1 分量。"""
    identity = np.eye(6)
    solution, *_ = np.linalg.lstsq(monodromy - identity, tangent, rcond=None)
    drift = solution - float(tangent @ solution) * tangent
    norm = float(np.linalg.norm(drift))
    if norm < _DRIFT_NORM_TOL:
        raise FloquetModeError(
            "切向链的广义特征向量退化（投影后范数近零），单值矩阵无预期的亏损结构"
        )
    drift = drift / norm
    pivot = int(np.argmax(np.abs(drift)))
    if drift[pivot] < 0.0:
        drift = -drift
    return drift


def _classify_pairs(
    eigenvalues: np.ndarray, monodromy: np.ndarray
) -> tuple[FloquetPair, FloquetPair]:
    """把非平凡特征值组织为面内与法向两对单位模共轭复对。

    面内与法向的归属按单值矩阵的分块结构确定：CR3BP 的平面子空间不变，
    平面轨道的单值矩阵在面内块（x、y、vx、vy）与法向块（z、vz）上解耦，
    两块分别做特征分解，块归属即 kind。跨块耦合超过门限时轨道是三维的，
    两对特征向量的面内与面外分量互相混杂，按支撑排序贴标签会静默出错，
    因此确定性拒绝（三维模态分类属后续扩展）。

    每个块内的非平凡特征值须为单位模共轭复对，否则是不稳定或分岔形态，
    同样确定性报错。
    """
    off_block = max(
        float(np.max(np.abs(monodromy[np.ix_(_NORMAL_COMPONENTS, _PLANAR_COMPONENTS)]))),
        float(np.max(np.abs(monodromy[np.ix_(_PLANAR_COMPONENTS, _NORMAL_COMPONENTS)]))),
    )
    if off_block > _PLANAR_DECOUPLING_TOL:
        raise FloquetModeError(
            f"单值矩阵的面内块与法向块不分离（跨块最大元素 {off_block:.2e}），"
            "本模块只支持平面周期轨道；三维周期轨道的模态分类属后续扩展"
        )

    near_one = np.abs(eigenvalues - 1.0) <= _TRIVIAL_EIGENVALUE_TOL
    trivial_count = int(np.count_nonzero(near_one))
    if trivial_count != 2:
        raise FloquetModeError(
            f"贴近 1 的平凡乘子有 {trivial_count} 个（应为 2），"
            "轨道处于分岔或退化形态，不在本模块形态内"
        )

    in_plane = _pair_from_block(monodromy, _PLANAR_COMPONENTS, FloquetPairKind.IN_PLANE)
    out_of_plane = _pair_from_block(monodromy, _NORMAL_COMPONENTS, FloquetPairKind.OUT_OF_PLANE)
    return in_plane, out_of_plane


def _pair_from_block(
    monodromy: np.ndarray,
    components: tuple[int, ...],
    kind: FloquetPairKind,
) -> FloquetPair:
    """在单值矩阵的指定分量子块上取单位模共轭复对，块归属即 kind。"""
    block = monodromy[np.ix_(components, components)]
    block_eigenvalues, block_eigenvectors = np.linalg.eig(block)
    near_one = np.abs(block_eigenvalues - 1.0) <= _TRIVIAL_EIGENVALUE_TOL
    upper = np.flatnonzero(~near_one & (block_eigenvalues.imag > 0.0))
    if int(np.count_nonzero(~near_one)) != 2 or upper.shape[0] != 1:
        raise FloquetModeError(
            f"{kind.value} 块的非平凡特征值不是一对共轭复乘子；双曲实对"
            "（不稳定周期轨道）或退化结构不在本模块形态内"
        )
    eigenvalue = complex(block_eigenvalues[upper[0]])
    if abs(abs(eigenvalue) - 1.0) > _UNIT_MODULUS_TOL:
        raise FloquetModeError(
            f"{kind.value} 块的乘子 {eigenvalue:.6f} 偏离单位模；双曲形态"
            "（不稳定周期轨道）不在本模块形态内"
        )
    column = np.zeros(6, dtype=complex)
    column[list(components)] = block_eigenvectors[:, upper[0]]
    alpha, a, b = _canonical_pair(eigenvalue, column, monodromy)
    return FloquetPair(kind=kind, eigenvalue=eigenvalue, alpha=alpha, a=a, b=b)


def _canonical_pair(
    eigenvalue: complex, column: np.ndarray, monodromy: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """特征向量单位化并把最大模分量旋为正实，校验旋转恒等。"""
    vector = column / float(np.linalg.norm(column))
    pivot = int(np.argmax(np.abs(vector)))
    vector = vector * np.exp(-1j * np.angle(vector[pivot]))
    a = np.real(vector).copy()
    b = np.imag(vector).copy()
    alpha = float(np.angle(eigenvalue))
    cos, sin = np.cos(alpha), np.sin(alpha)
    rotation_error = max(
        float(np.linalg.norm(monodromy @ a - (cos * a - sin * b))),
        float(np.linalg.norm(monodromy @ b - (sin * a + cos * b))),
    )
    if rotation_error > _ROTATION_TOL:
        raise FloquetModeError(f"复对旋转恒等校验失败（误差 {rotation_error:.2e}），特征结构非预期")
    return alpha, a, b


__all__ = [
    "FloquetModeError",
    "FloquetModeSet",
    "FloquetPair",
    "FloquetPairKind",
    "floquet_modes",
]
