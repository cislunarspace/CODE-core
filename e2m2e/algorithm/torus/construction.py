"""二维不变环面的频闪构造（#842）。

以种子周期轨道与其中心乘子对为输入，求解以种子周期为频闪周期的离散
映射的不变闭曲线与旋转数：曲线上的点经一个频闪周期落到同一曲线的
φ+ρ 处（invariance），参数化与纯旋转共轭（conjugacy）。未知量为相位
网格上的曲线采样与旋转数 ρ，φ+ρ 平移用 DFT 在带限插值意义上精确实
现，频闪周期固定为种子周期（该离散映射的不变曲线即频闪口径的环面截
面，不需要把频闪周期当未知量）。

方程组结构：不变性条件在两倍过采样相位上求值（配点数多于未知量，高
次模由超定最小二乘钉住），加上振幅钉扎与两条规范。不变性方程有三个
连续对称：快相位切片（曲线沿流推进仍是解，流与自身可交换）、慢相位
参数化原点（参数旋转）与振幅（环面族参数）；前两者各配一条规范方程
（切片取曲线沿流方向的平均偏离为零，参数取一阶正弦分量过零），振幅
由钉扎条件约束。求解用高斯牛顿迭代加 Armijo 回溯线搜索（SVD 分解滤
奇异值），与共享 PAL 内核的 SVD 牛顿加回溯线搜索机器同口径；配合从
小振幅起的同伦阶梯把解爬到目标振幅。

振幅按环面相对种子轨道的最大法向偏离（月球距离单位）参数化。术语见
CONTEXT.md 的拟周期不变环面一节，决策与口径见 ADR 0063。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from e2m2e.algorithm.family.resonant_branch import classify_multiplier_pairs, monodromy
from e2m2e.data.types.orbit import Orbit
from e2m2e.exceptions import E2M2EError
from e2m2e.status import ConvergenceState, FailureCause, ResultStatus

if TYPE_CHECKING:
    from e2m2e.algorithm.dynamics import CR3BP_Dynamics

__all__ = [
    "CenterMultiplier",
    "InvariantTorus",
    "TorusError",
    "central_multiplier_pair",
    "construct_invariant_torus",
    "evaluate_torus_curve",
]

#: 非平凡复对单位模的判定容差（与 proximity.floquet_modes 同口径）。
_UNIT_MODULUS_TOL = 1e-3

#: 复对判定的虚部下限：实特征值对（双曲或恰在分岔处）不构成中心对。
_COMPLEX_IMAG_TOL = 1e-12

#: 平面轨道法向支撑（z、vz 分量占比）的判定阈值。
_NORMAL_SUPPORT_TOL = 1e-6

#: 最小网格点数：带限闭曲线至少要分辨 k=±1 主模，且须为奇数（见
#: _band_limited 的 Nyquist 说明）。
_MIN_POINTS = 9

#: 不变性条件的配点过采样倍数：配点数为网格点数的两倍。
_OVERSAMPLE = 2

#: 振幅同伦的起始振幅与逐级放大倍数。起始振幅取线性种子的高斯牛顿
#: 收敛域内（1e-4）；级差按标定取 1.25，大振幅处映射曲率增大，更陡的
#: 级差（1.6）会在 0.02 附近失去收敛域。
_HOMOTOPY_START = 1e-4
_HOMOTOPY_FACTOR = 1.25

#: 残差停滞下限：低于此值视为积分噪声地板。
_STAGNATION_LIMIT = 1e-14

#: 残差发散上限。
_DIVERGENCE_LIMIT = 1e10

#: 回溯线搜索的最大折半次数。
_MAX_LINE_SEARCH = 24

#: 振幅细网格的过采样倍数。
_FINE_OVERSAMPLE = 8


class TorusError(E2M2EError, ValueError):
    """不变环面构造的确定性输入错误（周期未设、振幅非正、特征结构不合）。"""


@dataclass(frozen=True)
class CenterMultiplier:
    """种子周期轨道的中心乘子对与环面种子基。

    Attributes:
        eigenvalue: 辐角为正的单位模复乘子。
        alpha: 每周期转角（rad），等于 eigenvalue 的辐角，落在 (0, π)。
        a: 复特征向量的实部（单位化）。
        b: 复特征向量的虚部（单位化）。
    """

    eigenvalue: complex
    alpha: float
    a: np.ndarray
    b: np.ndarray


@dataclass
class InvariantTorus:
    """不变环面构造结果：状态三元组加环面数据。

    Attributes:
        status: 收敛状态（CAUSE_STATUS 口径）。
        cause: 失败原因码，收敛时为 NONE。
        message: 结局说明。
        curve_states: 频闪截面闭曲线采样，形状 (n_points, 6)。
        phases: 采样相位网格，0 到 2π 等间距（不含端点）。
        rotation_number: 旋转数（rad），每频闪周期慢相位推进辐角。
        strobe_period: 频闪周期（TU），本构造固定为种子周期。
        second_period: 第二周期 T·2π/ρ（TU）。
        second_period_days: 第二周期换算物理时间（天），系统未定尺度时为
            None。
        amplitude: 环面相对种子轨道的最大法向偏离（月球距离单位）。
        multiplier: 中心乘子（辐角正向成员）。
        multiplier_argument: 中心乘子辐角 α（rad），旋转数的零振幅极限。
        monodromy: 种子周期轨道的单值矩阵。
        invariance_residual: 不变性残差（不变性行的无穷范数）。
        conjugacy_residual: 共轭性残差（rad，各采样点推进角与旋转数偏离
            的最大绝对值），收敛时随结果一并给出。
        newton_iterations: 牛顿迭代次数。
        residual_history: 逐次迭代的求解系统残差无穷范数（含钉扎与规范行）。
    """

    status: ConvergenceState
    cause: FailureCause
    message: str
    curve_states: np.ndarray
    phases: np.ndarray
    rotation_number: float
    strobe_period: float
    second_period: float
    second_period_days: float | None
    amplitude: float
    multiplier: complex
    multiplier_argument: float
    monodromy: np.ndarray
    invariance_residual: float
    conjugacy_residual: float
    newton_iterations: int
    residual_history: list[float]

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass(frozen=True)
class _Problem:
    """牛顿系统的不可变上下文（振幅钉扎权重随外层刷新重建）。"""

    state0: np.ndarray
    amplitude: float
    n: int
    free_directions: int
    m: int
    strobe_period: float
    sin_phi: np.ndarray
    gauge: np.ndarray
    flow: np.ndarray
    position_projection: np.ndarray
    weights: np.ndarray
    os_phases: np.ndarray
    interp: np.ndarray
    tolerance: float
    dynamics: CR3BP_Dynamics


def central_multiplier_pair(matrix: np.ndarray, flow_direction: np.ndarray) -> CenterMultiplier:
    """从单值矩阵谱中选取环面生长方向的非平凡中心乘子对。

    规则：平凡对（按特征向量与流方向的投影判别）排除后，取单位模复对。
    恰有一个时直接采用（空间周期轨道的常态）；多于一个时按平面轨道分块
    结构判别，取法向支撑（z、vz 分量占比）完整的对（平面种子的法向环面
    生长方向）；仍不可判时确定性报错。

    Args:
        matrix: 6×6 单值矩阵。
        flow_direction: 种子状态处的流方向（CR3BP 向量场）。

    Returns:
        :class:`CenterMultiplier`。

    Raises:
        TorusError: 无非平凡单位模乘子对，或多个中心对且无平面分块结构
            可判别。
    """
    flow = np.asarray(flow_direction, dtype=float)
    pairs = classify_multiplier_pairs(matrix, flow, flow)
    candidates = []
    for pair in pairs:
        if pair.is_trivial:
            continue
        values = pair.eigenvalues
        if min(abs(abs(value) - 1.0) for value in values) > _UNIT_MODULUS_TOL:
            continue
        if min(abs(value.imag) for value in values) <= _COMPLEX_IMAG_TOL:
            continue
        candidates.append(pair)
    if not candidates:
        raise TorusError("无非平凡单位模乘子对：种子轨道在环面生长方向双曲或近简并")
    if len(candidates) > 1:
        normal = [
            pair
            for pair in candidates
            if min(
                float(np.linalg.norm(vector[[2, 5]])) / float(np.linalg.norm(vector))
                for vector in pair.eigenvectors
            )
            > 1.0 - _NORMAL_SUPPORT_TOL
        ]
        if len(normal) != 1:
            raise TorusError(
                f"存在 {len(candidates)} 个非平凡单位模乘子对且无法按平面分块判别"
                f"（法向支撑合格 {len(normal)} 个），须换单中心对种子或显式指定方向"
            )
        candidates = normal
    pair = candidates[0]
    if pair.eigenvalues[0].imag > 0.0:
        value, vector = pair.eigenvalues[0], pair.eigenvectors[0]
    else:
        value, vector = pair.eigenvalues[1], pair.eigenvectors[1]
    unit = np.asarray(vector, dtype=complex) / np.linalg.norm(vector)
    return CenterMultiplier(
        eigenvalue=complex(value),
        alpha=float(np.angle(value)),
        a=np.real(unit),
        b=np.imag(unit),
    )


def _band_limited(curve: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """网格采样的带限重建与相位导数（fft 序对称模式集，单一份实现）。

    网格须为奇数：偶数网格的 Nyquist 模（k = n/2）在实带限表示里没有
    共轭搭档，旋转该模需要频带外的 sin 部分，数值解把残余垃圾堆进该模
    后旋转不一致性直接变成残差地板（两种种子实测 1e-6 量级）。奇数网格
    的全部模式共轭成对，旋转逐模精确。
    """
    n = curve.shape[0]
    coef = np.fft.rfft(curve, axis=0)
    weights = np.full(n // 2 + 1, 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    spectrum = coef * weights[:, None]
    modes = np.arange(n // 2 + 1)
    wave = np.exp(1j * np.outer(phi, modes))
    values = np.real(wave @ spectrum) / n
    slope = np.real((1j * modes[None, :] * wave) @ spectrum) / n
    return values, slope


def evaluate_torus_curve(curve_states: npt.ArrayLike, phases: npt.ArrayLike) -> np.ndarray:
    """闭曲线的带限插值：在任意相位处求值网格采样的环面截面。

    Args:
        curve_states: 网格采样，形状 (n_points, 6)，须来自奇数点等间距
            网格（偶数网格含不可一致旋转的 Nyquist 模，见 _band_limited）。
        phases: 求值相位（rad），标量或一维数组。

    Returns:
        形状 (len(phases), 6) 的状态，相位相差 2π 时取值相同。
    """
    curve = np.asarray(curve_states, dtype=float)
    n = curve.shape[0]
    if n < 5 or n % 2 == 0:
        raise TorusError(f"curve_states 须为不小于 5 行的奇数行网格采样，当前 {n} 行")
    phi = np.atleast_1d(np.asarray(phases, dtype=float))
    return _band_limited(curve, phi)[0]


def _interpolation_row(phase: float, n: int) -> np.ndarray:
    """带限插值在相位 phase 处对网格采样的线性权重。

    模式集取 fft 序（0 到 n/2，再取负模），与 rfft 的对称重建一致；
    自然序 0 到 n−1 的和只在网格相位上与它重合，离网格相位会相差
    相位卷绕，不可用。
    """
    modes = np.fft.fftfreq(n) * n
    delta = phase - np.arange(n) * (2.0 * np.pi / n)
    kernel = np.exp(1j * np.outer(modes, delta))
    return np.real(kernel.sum(axis=0)) / n


def _interpolation_matrix(phases: np.ndarray, n: int, *, derivative: bool = False) -> np.ndarray:
    """带限插值矩阵（或其对相位导数）：行 j 为相位 phases[j] 的权重。"""
    modes = np.fft.fftfreq(n) * n
    delta = np.asarray(phases)[:, None] - np.arange(n)[None, :] * (2.0 * np.pi / n)
    kernel = np.exp(1j * np.outer(modes, delta.ravel())).reshape(n, -1)
    factors = 1j * modes[:, None] if derivative else np.ones((n, 1))
    return np.real((factors * kernel).sum(axis=0)).reshape(delta.shape) / n


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _strobe_images(
    dynamics: CR3BP_Dynamics, curve: np.ndarray, tau: float, *, with_stm: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """把曲线各采样传播一个频闪周期，返回终点状态与（可选）单值矩阵。"""
    count = curve.shape[0]
    images = np.empty((count, 6))
    stms = np.empty((count, 6, 6)) if with_stm else None
    for j in range(count):
        out = dynamics.propagate(curve[j], (0.0, tau), t_eval=[tau], with_stm=with_stm)
        images[j] = np.asarray(out["states"])[-1]
        if stms is not None:
            stms[j] = np.asarray(out["stm"])[-1]
    return images, stms


def _anchor_deviation(problem: _Problem, curve: np.ndarray) -> np.ndarray:
    """振幅钉扎点的法向偏离（位置三分量）。"""
    anchor = problem.weights @ curve
    return problem.position_projection @ (anchor - problem.state0)


def _residual(problem: _Problem, q: np.ndarray) -> np.ndarray:
    """牛顿系统残差：过采样配点上的不变性条件加振幅钉扎与两条规范。"""
    n = problem.n
    m = problem.m
    curve = q[: 6 * n].reshape(n, 6)
    rho = float(q[-1])
    os_curve = problem.interp @ curve
    images, _ = _strobe_images(problem.dynamics, os_curve, problem.strobe_period, with_stm=False)
    shifted = evaluate_torus_curve(curve, problem.os_phases + rho)
    invariance = (images - shifted).ravel()
    pin = float(np.linalg.norm(_anchor_deviation(problem, curve))) - problem.amplitude
    deviation = os_curve - problem.state0
    gauge = 2.0 / m * float(problem.sin_phi @ (deviation @ problem.gauge))
    slice_gauge = float(np.mean(deviation @ problem.flow))
    return np.concatenate([invariance, [pin, gauge, slice_gauge]])


def _jacobian(
    problem: _Problem,
    q: np.ndarray,
    images: np.ndarray,
    stms: np.ndarray,
) -> np.ndarray:
    """牛顿系统雅可比：解析组装（传播 STM、带限插值）。"""
    n = problem.n
    m = problem.m
    curve = q[: 6 * n].reshape(n, 6)
    rho = float(q[-1])
    jac = np.zeros((6 * m + 3, 6 * n + 1))
    shifted = _interpolation_matrix(problem.os_phases + rho, n)
    for j in range(m):
        jac[6 * j : 6 * j + 6, : 6 * n] = np.kron(problem.interp[j], stms[j])
        jac[6 * j : 6 * j + 6, : 6 * n] -= np.kron(shifted[j], np.eye(6))
    dshift = _interpolation_matrix(problem.os_phases + rho, n, derivative=True)
    jac[: 6 * m, -1] = -(dshift @ curve).ravel()
    deviation = _anchor_deviation(problem, curve)
    unit = problem.position_projection.T @ (deviation / np.linalg.norm(deviation))
    jac[6 * m, : 6 * n] = np.outer(problem.weights, unit).ravel()
    column = problem.interp.T @ (2.0 / m * problem.sin_phi)
    jac[6 * m + 1, : 6 * n] = np.outer(column, problem.gauge).ravel()
    jac[6 * m + 2, : 6 * n] = np.outer(problem.interp.T @ np.full(m, 1.0 / m), problem.flow).ravel()
    return jac


@dataclass(frozen=True)
class _Outcome:
    """牛顿求解结局。"""

    q: np.ndarray
    status: ConvergenceState
    cause: FailureCause
    message: str
    iterations: int
    history: list[float]


def _newton(problem: _Problem, q0: np.ndarray, max_iterations: int) -> _Outcome:
    """高斯牛顿迭代加 Armijo 回溯线搜索。

    步长控制取纯高斯牛顿方向上的 t 缩放而不是 Levenberg 阻尼：阻尼先削
    平坦方向的分量，而远离解时需要的修正恰恰整体住在平坦方向上，按奇
    异值阻尼会得到方向错乱的步，实测在振幅阶梯上撞 1e-6 量级残差地板；
    t 缩放保持方向、只收步长，配合振幅同伦可爬到 1e-2 量级振幅。线搜索
    用 Armijo 判据（平方残差充分下降），并守卫钉扎偏离的正性。
    """
    n = problem.n
    q = q0.copy()
    history: list[float] = []
    residual: np.ndarray | None = None
    for iteration in range(1, max_iterations + 1):
        try:
            if residual is None:
                residual = _residual(problem, q)
        except E2M2EError as error:
            return _Outcome(
                q,
                ConvergenceState.FAILED,
                FailureCause.INTEGRATION_FAILED,
                f"频闪传播失败：{error}",
                iteration - 1,
                history,
            )
        # 收敛判据看全残差：只看不变性行会在钉扎与规范行未满足时误报
        # 收敛（振幅阶梯曾因此空转）。
        norm = float(np.max(np.abs(residual)))
        history.append(norm)
        if norm <= problem.tolerance:
            return _Outcome(
                q,
                ConvergenceState.CONVERGED,
                FailureCause.NONE,
                f"求解系统残差 {norm:.2e} 达容差",
                iteration,
                history,
            )
        if norm > _DIVERGENCE_LIMIT:
            return _Outcome(
                q,
                ConvergenceState.DIVERGED,
                FailureCause.DIVERGENCE_DETECTED,
                f"残差发散到 {norm:.2e}",
                iteration,
                history,
            )
        if norm < _STAGNATION_LIMIT:
            return _Outcome(
                q,
                ConvergenceState.STAGNATED,
                FailureCause.STAGNATION_DETECTED,
                f"残差 {norm:.2e} 已低于积分噪声地板",
                iteration,
                history,
            )
        curve = q[: 6 * n].reshape(n, 6)
        try:
            images, stms = _strobe_images(
                problem.dynamics,
                problem.interp @ curve,
                problem.strobe_period,
                with_stm=True,
            )
        except E2M2EError as error:
            return _Outcome(
                q,
                ConvergenceState.FAILED,
                FailureCause.INTEGRATION_FAILED,
                f"频闪传播失败：{error}",
                iteration,
                history,
            )
        assert stms is not None
        jac = _jacobian(problem, q, images, stms)
        left, singular, right_t = np.linalg.svd(jac, full_matrices=False)
        # 丢弃最小的 free_directions 个奇异方向：双中心对种子的面内振幅
        # 方向是精确零空间（数值 σ 极小但非零），按 σ/σ² 放大会把步长
        # 爆到该方向上；该方向的解族成员都满足方程，显式忽略即可。
        cutoff = len(singular) - problem.free_directions
        keep = np.arange(len(singular)) < cutoff
        filter_ = np.where(keep & (singular > singular[0] * 1e-12), singular / (singular**2), 0.0)
        step = right_t.T @ (filter_ * (left.T @ (-residual)))
        base_square = float(residual @ residual)
        accepted = False
        for attempt in range(_MAX_LINE_SEARCH):
            factor = 0.5**attempt
            trial = q + factor * step
            trial_curve = trial[: 6 * n].reshape(n, 6)
            usable = (
                np.all(np.isfinite(trial))
                and float(np.linalg.norm(_anchor_deviation(problem, trial_curve))) > 1e-12
            )
            if usable:
                try:
                    trial_residual = _residual(problem, trial)
                except E2M2EError:
                    trial_residual = None
            else:
                trial_residual = None
            if (
                trial_residual is not None
                and float(trial_residual @ trial_residual) <= (1.0 - 1e-4 * factor) * base_square
            ):
                q = trial
                residual = trial_residual
                accepted = True
                break
        if not accepted:
            return _Outcome(
                q,
                ConvergenceState.STAGNATED,
                FailureCause.STAGNATION_DETECTED,
                f"回溯线搜索 {_MAX_LINE_SEARCH} 次折半未能充分下降残差（{norm:.2e}）",
                iteration,
                history,
            )
    assert residual is not None
    if float(np.max(np.abs(residual))) <= problem.tolerance:
        # 末步恰达容差也要按收敛报告：收敛判据在循环顶，末步通过时
        # 直接落出循环会误报迭代上限。
        final = float(np.max(np.abs(residual)))
        return _Outcome(
            q,
            ConvergenceState.CONVERGED,
            FailureCause.NONE,
            f"求解系统残差 {final:.2e} 达容差",
            max_iterations,
            history,
        )
    return _Outcome(
        q,
        ConvergenceState.MAX_ITERATIONS,
        FailureCause.MAX_ITERATIONS_REACHED,
        f"达到迭代上限 {max_iterations}，残差 {history[-1]:.2e}",
        max_iterations,
        history,
    )


def _initial_curve(
    state0: np.ndarray,
    center: CenterMultiplier,
    amplitude: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """线性化种子曲线、中心平面投影与慢相位参数化规范方向。

    零振幅极限下不变曲线为 x0 + r(a cosφ − b sinφ)（符号取推进角为
    +α 的一支），r 按目标振幅标定。
    """
    a, b = center.a, center.b
    b_perp = b - float(b @ a) * a
    b_perp_norm = float(np.linalg.norm(b_perp))
    if b_perp_norm < 1e-12:
        raise TorusError("中心乘子对特征向量实部与虚部共线，无法张成中心平面")
    basis = np.column_stack([a, b_perp / b_perp_norm])
    projection = basis @ basis.T
    direction = a[:3] if float(np.linalg.norm(a[:3])) > 1e-8 else b[:3]
    gauge = np.zeros(6)
    gauge[:3] = direction / np.linalg.norm(direction)
    fine = np.arange(512) * (2.0 * np.pi / 512)
    shapes = a[None, :] * np.cos(fine)[:, None] - b[None, :] * np.sin(fine)[:, None]
    span = float(np.max(np.linalg.norm(shapes[:, :3], axis=1)))
    if span < 1e-12:
        raise TorusError("中心乘子对无位置分量，无法按位置振幅参数化")
    phi = np.arange(n) * (2.0 * np.pi / n)
    curve = state0[None, :] + (amplitude / span) * (
        a[None, :] * np.cos(phi)[:, None] - b[None, :] * np.sin(phi)[:, None]
    )
    return curve, projection, gauge


def _nearest_resonance(rho: float, *, max_denominator: int = 32) -> tuple[int, int, float] | None:
    """最邻近的低阶共振 p/q·2π（q 不超过 max_denominator）及其偏差。"""
    best: tuple[int, int, float] | None = None
    for denominator in range(2, max_denominator + 1):
        numerator = round(rho * denominator / (2.0 * np.pi))
        if numerator <= 0:
            continue
        gap = abs(_wrap_angle(rho - numerator * 2.0 * np.pi / denominator))
        if best is None or gap < best[2]:
            best = (int(numerator), denominator, gap)
    return best


def construct_invariant_torus(
    orbit: Orbit,
    amplitude: float,
    *,
    n_points: int = 25,
    dynamics: CR3BP_Dynamics | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 50,
) -> InvariantTorus:
    """在周期轨道邻域构造二维不变环面。

    求解以种子周期为频闪周期的离散映射的不变闭曲线：未知量为相位网格
    上的曲线采样与旋转数 ρ，方程为两倍过采样相位上的不变性条件、振幅
    钉扎条件（细网格最大法向偏离等于目标振幅）与两条规范条件。中心乘
    子对由内部按谱自动选取，选不出唯一对时以状态三元组拒绝。

    Args:
        orbit: 带单圈周期真值的周期轨道，只读取首点状态与周期。
        amplitude: 目标振幅：环面相对种子轨道的最大法向偏离（月球距离
            单位），平面种子即最大 |z|。
        n_points: 相位网格点数（奇数，不小于 9），即曲线分辨率。网格
            密度须随振幅放大：环面截面的高次谐波随振幅增长，带宽不足
            时残差停在截断地板。L2 Halo 标定：振幅到 1e-2 量级用 17 点，
            到 0.031 用 33 点。
        dynamics: 地月 CR3BP 动力学；缺省用标准地月系统。积分容差取
            dynamics 自带 rtol/atol，默认残差容差 1e-9 已贴研究级积分的
            噪声地板，收紧前先收紧积分容差。
        tolerance: 收敛容差：不变性残差无穷范数上限。
        max_iterations: 牛顿迭代上限。

    Returns:
        :class:`InvariantTorus`，不可行或未收敛以状态三元组如实报告。

    Raises:
        TorusError: 周期未设、振幅非正、网格点数不合等确定性输入错误。
    """
    from e2m2e.algorithm.dynamics import CR3BP_Dynamics
    from e2m2e.algorithm.family.orbits.walk import earth_moon_system

    if orbit.period is None or float(orbit.period) <= 0.0:
        raise TorusError("orbit.period 未设置或非正，须先给出单圈周期真值")
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise TorusError(f"amplitude 须为正有限值，当前 {amplitude}")
    if n_points % 2 == 0 or n_points < _MIN_POINTS:
        raise TorusError(f"n_points 须为不小于 {_MIN_POINTS} 的奇数，当前 {n_points}")
    if max_iterations < 1:
        raise TorusError(f"max_iterations 须为正，当前 {max_iterations}")
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())

    state0 = np.asarray(orbit.states[0], dtype=float).copy()
    seed_period = float(orbit.period)
    matrix = monodromy(dynamics, state0, seed_period)
    flow = np.asarray(dynamics.equations_of_motion(0.0, state0), dtype=float)
    phi = np.arange(n_points) * (2.0 * np.pi / n_points)
    fine_count = _FINE_OVERSAMPLE * n_points
    fine = np.arange(fine_count) * (2.0 * np.pi / fine_count)
    os_count = _OVERSAMPLE * n_points
    os_phases = np.arange(os_count) * (2.0 * np.pi / os_count)
    interp = _interpolation_matrix(os_phases, n_points)

    try:
        center = central_multiplier_pair(matrix, flow)
    except TorusError as error:
        return InvariantTorus(
            status=ConvergenceState.FAILED,
            cause=FailureCause.INVALID_INPUT,
            message=str(error),
            curve_states=np.tile(state0, (n_points, 1)),
            phases=phi,
            rotation_number=0.0,
            strobe_period=seed_period,
            second_period=float("nan"),
            second_period_days=None,
            amplitude=float("nan"),
            multiplier=complex("nan"),
            multiplier_argument=float("nan"),
            monodromy=matrix,
            invariance_residual=float("nan"),
            conjugacy_residual=float("nan"),
            newton_iterations=0,
            residual_history=[],
        )

    _, projection, gauge = _initial_curve(state0, center, amplitude, n_points)
    position_projection = projection[:3, :]
    # 未规范方向数：每个非平凡单位模对贡献一个环面族振幅参数，钉扎只固定
    # 所选对（平面种子的法向）的振幅，其余对（面内）的振幅方向是方程组的
    # 精确零空间，高斯牛顿求解显式丢弃（见 _newton）。
    unit_pairs = sum(
        1
        for pair in classify_multiplier_pairs(matrix, flow, flow)
        if not pair.is_trivial
        and min(abs(abs(value) - 1.0) for value in pair.eigenvalues) <= _UNIT_MODULUS_TOL
        and min(abs(value.imag) for value in pair.eigenvalues) > _COMPLEX_IMAG_TOL
    )
    free_directions = max(0, unit_pairs - 1)

    def fine_amplitude(curve: np.ndarray) -> tuple[float, float]:
        values = evaluate_torus_curve(curve, fine)
        deviations = np.linalg.norm((values - state0) @ position_projection.T, axis=1)
        peak = int(np.argmax(deviations))
        return float(deviations[peak]), float(fine[peak])

    # 振幅同伦：近月空间映射的二阶曲率大（等效 |D²P| 在千量级），线性种子
    # 在目标振幅处已落在牛顿二次收敛域外。从小振幅起沿阶梯爬升，逐级以
    # 已收敛解热启动，每级的修正量保持在收敛域内。
    ladder = [amplitude]
    if amplitude > _HOMOTOPY_START:
        steps: list[float] = []
        value = _HOMOTOPY_START
        while value < amplitude:
            steps.append(value)
            value *= _HOMOTOPY_FACTOR
        ladder = [*steps, amplitude]
    start_amplitude = ladder[0]
    seed_curve, _, _ = _initial_curve(state0, center, start_amplitude, n_points)
    _, phi_star = fine_amplitude(seed_curve)
    q = np.concatenate([seed_curve.ravel(), [center.alpha]])
    outcome: _Outcome | None = None
    for level in ladder:
        for _ in range(3):
            problem = _Problem(
                state0=state0,
                amplitude=level,
                n=n_points,
                free_directions=free_directions,
                m=os_count,
                strobe_period=seed_period,
                sin_phi=np.sin(os_phases),
                gauge=gauge,
                flow=flow,
                position_projection=position_projection,
                weights=_interpolation_row(phi_star, n_points),
                os_phases=os_phases,
                interp=interp,
                tolerance=tolerance,
                dynamics=dynamics,
            )
            outcome = _newton(problem, q, max_iterations)
            q = outcome.q
            if outcome.status is not ConvergenceState.CONVERGED:
                break
            _, new_star = fine_amplitude(q[: 6 * n_points].reshape(n_points, 6))
            if abs(_wrap_angle(new_star - phi_star)) <= 2.0 * np.pi / fine_count:
                break
            phi_star = new_star
        if outcome is None or outcome.status is not ConvergenceState.CONVERGED:
            break
    assert outcome is not None

    curve = q[: 6 * n_points].reshape(n_points, 6).copy()
    rotation_number = float(q[-1])
    achieved, _ = fine_amplitude(curve)
    try:
        final_rows = _residual(problem, q)
        invariance_norm = float(np.max(np.abs(final_rows[: 6 * os_count])))
    except E2M2EError:
        invariance_norm = float("nan")
    converged = outcome.status is ConvergenceState.CONVERGED
    conjugacy_norm = float("nan")
    if converged:
        from e2m2e.algorithm.torus.diagnostics import conjugacy_residual

        try:
            conjugacy_norm = conjugacy_residual(curve, rotation_number, seed_period, dynamics)
        except E2M2EError:
            conjugacy_norm = float("nan")
    message = outcome.message
    if not converged and math.isfinite(rotation_number) and rotation_number > 0.0:
        # 不收敛时报告终止旋转数与最邻近低阶共振，共振不可行可从消息分辨。
        resonance = _nearest_resonance(rotation_number % (2.0 * np.pi))
        if resonance is not None:
            numerator, denominator, gap = resonance
            message = (
                f"{message}；终止旋转数 {rotation_number % (2.0 * np.pi):.6f} rad，"
                f"最邻近低阶共振 {numerator}/{denominator}·2π 偏差 {gap:.2e} rad"
            )
    if math.isfinite(rotation_number) and rotation_number > 0.0:
        second_period = seed_period * 2.0 * np.pi / rotation_number
    else:
        second_period = float("nan")
    characteristic_time = getattr(dynamics.system, "characteristic_time", None)
    if characteristic_time is not None and math.isfinite(second_period):
        second_period_days = float(second_period) * float(characteristic_time) / 86400.0
    else:
        second_period_days = None
    return InvariantTorus(
        status=outcome.status,
        cause=outcome.cause,
        message=message,
        curve_states=curve,
        phases=phi,
        rotation_number=rotation_number % (2.0 * np.pi) if converged else rotation_number,
        strobe_period=seed_period,
        second_period=second_period,
        second_period_days=second_period_days,
        amplitude=achieved,
        multiplier=center.eigenvalue,
        multiplier_argument=center.alpha,
        monodromy=matrix,
        invariance_residual=invariance_norm,
        conjugacy_residual=conjugacy_norm,
        newton_iterations=outcome.iterations,
        residual_history=outcome.history,
    )
