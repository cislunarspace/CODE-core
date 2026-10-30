"""共振点与乘子合并的定位、起链（#820、#825）。

两部分能力，都是算法层内部自含，不构成接口层的稳定性或分岔分析工具
（#687 的边界）。

共振起链（#820）：在平凡族上定位 p:n 共振成员（非平凡复数乘子对的辐
角扫过 2π/n，其 n 次幂落回 1），沿共振平面的实向量扰动出种子，交给全
周期 PAL 在 n 圈闭合形式下精化并行走。共振成员处平凡族的 n 圈闭合雅
可比退化，普通牛顿的收敛盆把有限振幅种子吸回平凡叶。起链用共振特征
向量的实部与虚部（共振平面的两组基）做 deflection，给出分支切向的一
阶近似，再用阻尼最小二乘（固定 x0 加相位条件）把种子收到分岔支上，
精化后校验种子确已离开平凡族。

乘子合并定位（#825）：沿 L2 南 Halo 族逐成员计算单值矩阵，按特征向量
与流方向、族切向的投影显式判别平凡乘子对，连续跟踪非平凡对的稳定性
指数 ν，定位 ν 跨越 +2（乘子过 +1）与 −2（乘子过 −1）的族成员，再用
固定 x0 微分修正二分精化。+1 合并的临界成员构造 ResonancePoint，供
spatial 起链走 butterfly 类非对称分支。

平凡乘子对是这两件事共同的陷阱。单值矩阵恒有等于 1 的平凡乘子对（流
方向与族切向），其数值分裂随成员闭合容差放大：PAL 原始成员的闭合只
到 1e-7 量级时，平凡对可分裂成 1e-3 量级的实对或复对，不比真实临界
乘子小。判别因此不得用“最接近 1 的乘子”，必须按特征向量投影把平凡
对显式排除，再在非平凡对上跟踪 ν。逐成员谱直接取 PAL 原始成员的行走
状态，判别靠投影而不靠乘子接近度，平凡分裂的幅度因此不影响结论，只有
精化端点用固定 x0 紧修正收紧到 1e-13 量级。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...data.templates.seed import _HALO_SEED_Z0
from ...data.types.orbit import Orbit, OrbitFamily
from ...integrators import planar_full_period_pal_py, spatial_full_period_pal_py
from ...status import ConvergenceState, FailureCause, ResultStatus
from ..dynamics import CR3BP_Dynamics
from ..results import ContinuationResult
from ..solver.continuation import Continuation
from ..solver.differential_correction import DifferentialCorrection
from .halo_family import halo_pseudo_arclength_continuation
from .orbits.dro import _correct_dro
from .orbits.halo import _correct_halo, _correct_halo_x0
from .orbits.walk import Cr3bpOrbitError, earth_moon_system

#: 共振乘子偏差门限：n 次幂乘子与 1 的差距超过此值时，平凡成员处的分支
#: 振幅过小且牛顿盆退化加剧，起链按不可行拒绝（标定脚本可复核）。
CRITICAL_MULTIPLIER_TOLERANCE = 5e-3

#: 起链 deflection 幅度阶梯与方向组合（共振平面的实部与虚部、双向）。
#: 全部组合精化后取单圈偏差最大者（分支振幅最大），确定性择优。
_DEFLECTION_LADDER = (1e-4, 3e-4, 1e-3, 3e-3)

#: 平面问题的状态分量（x、y、vx、vy）。
_PLANAR_COMPONENTS = (0, 1, 3, 4)


@dataclass
class ResonancePoint:
    """平凡族上的 p:n 共振成员及其起链信息。"""

    #: 共振成员的初始状态（6 维）。
    state: np.ndarray
    #: 共振成员的单圈周期（TU）。
    period: float
    #: 共振成员的族参数 x0。
    x0: float
    #: 共振乘子对（取辐角正向成员）的辐角（度）。
    phase_deg: float
    #: n 次幂乘子与 1 的偏差 |λ^n − 1|（平凡对数值分裂不计入）。
    multiplier_gap: float
    #: 共振特征向量（6 维，复数；其实部与虚部张成共振平面）。
    eigenvector: np.ndarray
    #: 共振圈数 n。
    revolutions: int


def monodromy(dynamics: CR3BP_Dynamics, state: np.ndarray, period: float) -> np.ndarray:
    """传播一个周期，返回 6×6 单值矩阵。"""
    out = dynamics.propagate(
        np.asarray(state, dtype=float),
        (0.0, float(period)),
        t_eval=[float(period)],
        with_stm=True,
    )
    return np.asarray(out["stm"][0], dtype=float)


def _planar_block(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.ix_(_PLANAR_COMPONENTS, _PLANAR_COMPONENTS)]


def _resonant_pair(planar_block: np.ndarray, target_deg: float) -> complex:
    """平面块中辐角最接近目标角的单位圆邻域复数乘子（取虚部为正成员）。

    平凡乘子对数值分裂为 1±1e-6 量级的近实对，其虚部为正的成员辐角贴
    近 0 或 180 度且模同样贴近 1，按“模最接近 1”选择会被它抢占。目标
    角 360/n（n 大于 2）远离 0 与 180 度，按辐角最近选取即稳定锁定共振
    对。
    """
    eigenvalues = np.linalg.eigvals(planar_block)
    best = None
    for value in eigenvalues:
        if value.imag <= 0.0 or abs(abs(value) - 1.0) > 0.05:
            continue
        angle = math.degrees(math.atan2(value.imag, value.real)) % 180.0
        distance = abs(angle - target_deg)
        if distance > 90.0:
            continue
        if best is None or distance < best[0]:
            best = (distance, value)
    if best is None:
        raise Cr3bpOrbitError(
            f"平面块未找到辐角贴近 {target_deg:.1f} 度的单位圆乘子对，无法定位共振"
        )
    return complex(best[1])


def _resonant_eigenvector(matrix: np.ndarray, target: complex) -> np.ndarray:
    """全 6×6 单值矩阵中与 ``target`` 最接近特征值对应的特征向量。"""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(eigenvalues - target)))
    vector = eigenvectors[:, index]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise Cr3bpOrbitError("共振特征向量为零，起链方向退化")
    return vector / norm


def locate_dro_revolution_resonance(
    dynamics: CR3BP_Dynamics | None = None,
    *,
    revolutions: int = 5,
    dx0: float = 0.01,
    max_members: int = 60,
) -> ResonancePoint:
    """沿月心 DRO 族定位 n 圈共振成员。

    从标准 DRO 种子邻域沿 x0 步进（两个方向都可扫到：默认从 0.80 向增
    方向走，小振幅侧辐角单调下降），跟踪平面块共振复数对的辐角，检测
    其跨越 360/n 度（方向无关），二分 x0 到区间宽 1e-7。返回共振成员与
    共振特征向量，供起链使用。

    Raises:
        Cr3bpOrbitError: 走完 ``max_members`` 仍未命中目标辐角，或修正
            失败。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    if revolutions < 2:
        raise ValueError(f"revolutions 必须大于 1，当前为 {revolutions}")
    target_deg = 360.0 / revolutions
    if not 0.0 < target_deg < 180.0:
        raise ValueError(f"360/n 必须落在 (0, 180) 度内，当前为 {target_deg}")

    def phase_at(orbit: Orbit) -> float:
        assert orbit.period is not None
        pair = _resonant_pair(
            _planar_block(monodromy(dynamics, orbit.states[0], float(orbit.period))),
            target_deg,
        )
        return math.degrees(math.atan2(pair.imag, pair.real)) % 180.0

    guess: Orbit | None = None
    x0 = 0.80
    prev_x0 = x0
    prev_phase: float | None = None
    bracket: tuple[float, float, float, float] | None = None
    for _ in range(max_members):
        orbit = _correct_dro(dynamics, x0, guess)
        guess = orbit
        phase = phase_at(orbit)
        if prev_phase is not None and (
            prev_phase < target_deg <= phase or phase < target_deg <= prev_phase
        ):
            bracket = (prev_x0, prev_phase, x0, phase)
            break
        prev_phase = phase
        prev_x0 = x0
        x0 += dx0
    if bracket is None:
        raise Cr3bpOrbitError(
            f"沿 DRO 族走 {max_members} 个成员（x0 步长 {dx0}）未扫到辐角 "
            f"{target_deg:.1f} 度，共振点不在该参数区间内"
        )

    low_x0, low_phase, high_x0, high_phase = bracket
    assert guess is not None
    for _ in range(50):
        if abs(high_x0 - low_x0) < 1e-7:
            break
        mid_x0 = 0.5 * (low_x0 + high_x0)
        orbit = _correct_dro(dynamics, mid_x0, guess)
        guess = orbit
        phase = phase_at(orbit)
        if (phase < target_deg) == (low_phase < target_deg):
            low_x0, low_phase = mid_x0, phase
        else:
            high_x0, high_phase = mid_x0, phase
    final_x0 = low_x0 if abs(low_phase - target_deg) <= abs(high_phase - target_deg) else high_x0
    orbit = _correct_dro(dynamics, final_x0, guess)

    state = orbit.states[0].copy()
    assert orbit.period is not None
    period = float(orbit.period)
    matrix = monodromy(dynamics, state, period)
    pair = _resonant_pair(_planar_block(matrix), target_deg)
    vector = _resonant_eigenvector(matrix, pair)
    gap = abs(pair**revolutions - 1.0)
    return ResonancePoint(
        state=state,
        period=period,
        x0=float(state[0]),
        phase_deg=float(math.degrees(math.atan2(pair.imag, pair.real)) % 180.0),
        multiplier_gap=float(gap),
        eigenvector=vector,
        revolutions=revolutions,
    )


def per_revolution_closure(dynamics: CR3BP_Dynamics, orbit: Orbit, revolutions: int) -> float:
    """成员在不足 n 圈处闭合强度的度量：单圈末端状态与初态的最大偏差。

    分岔支成员是真正的 n 圈闭合轨道（单圈末端不回到初态）；平凡族成员
    单圈即闭合（偏差在数值噪声量级）。用于区分两者（n 大于 1 时）。
    """
    assert orbit.period is not None
    single = float(orbit.period) / revolutions
    out = dynamics.propagate(
        orbit.states[0],
        (0.0, single),
        t_eval=[single],
    )
    states = np.asarray(out["states"], dtype=float)
    return float(np.max(np.abs(states[-1] - orbit.states[0])))


def _branch_nontriviality(
    dynamics: CR3BP_Dynamics, member: Orbit, resonance: ResonancePoint
) -> float:
    """分支成员离开平凡族的程度。

    n 圈共振分支（n 大于 1）：平凡族成员单圈即闭合，分岔支成员单圈末端
    不回到初态，用单圈闭合偏差度量。单圈 pitchfork 分支（n 等于 1，如
    butterfly）：任何周期轨道单圈闭合都近零，改用平凡族（XZ 对称）恒为
    零的对称破缺分量 y0 与 vx0 度量。
    """
    state = member.states[0]
    if resonance.revolutions > 1:
        return per_revolution_closure(dynamics, member, resonance.revolutions)
    return max(abs(float(state[1])), abs(float(state[3])))


def deflect_resonance_seed(
    resonance: ResonancePoint,
    *,
    epsilon: float,
    part: str = "real",
    sign: float = 1.0,
) -> np.ndarray:
    """沿共振平面的指定方向扰动出分支种子（6 维状态）。

    ``part`` 取 ``"real"`` 或 ``"imag"``（共振特征向量的实部或虚部）。
    闭合周期为 ``resonance.revolutions * resonance.period``，由调用方组
    装后交给 PAL。标定与演示脚本用它探索更大的初始振幅。
    """
    if part not in ("real", "imag"):
        raise ValueError(f"part 必须为 real 或 imag，当前为 {part!r}")
    vector = np.real(resonance.eigenvector) if part == "real" else np.imag(resonance.eigenvector)
    unit = vector / float(np.linalg.norm(vector))
    return np.asarray(resonance.state, dtype=float) + sign * epsilon * unit


def _refine_branch_seed(
    dynamics: CR3BP_Dynamics,
    resonance: ResonancePoint,
    state: np.ndarray,
    closure_period: float,
    spatial: bool,
):
    """仅精化种子（不行走）：固定 x0 的 n 圈闭合阻尼最小二乘。"""
    entry = spatial_full_period_pal_py if spatial else planar_full_period_pal_py
    return entry(
        float(dynamics.system.mu),
        np.asarray(state, dtype=float).tolist(),
        float(closure_period),
        0,
        0.005,
        "increase-x0",
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
        n_revolutions=resonance.revolutions,
    )


def generate_resonant_branch_family(
    resonance: ResonancePoint,
    *,
    n_orbits: int,
    step_size: float,
    initial_direction: str,
    dynamics: CR3BP_Dynamics | None = None,
    family_type: str = "dro",
    spatial: bool = False,
) -> ContinuationResult:
    """从共振点起链并沿分岔支生成周期轨道族。

    平面分支（默认）走平面 n 圈闭合 PAL；``spatial=True`` 走三维全周期
    PAL（如 butterfly 类非对称分支，特征向量含面外分量时使用）。共振
    乘子偏差超过门限时显式拒绝起链，不静默落到平凡族。

    deflection 在共振平面（特征向量的实部与虚部）内从小到大、双向重
    试，精化后校验种子确已离开平凡族，全部尝试落回平凡族时报失败。

    Returns:
        ContinuationResult: 族包含精化后的种子（分岔支首成员）加完成的
        部分族；软失败保留已收敛成员。
    """
    if dynamics is None:
        dynamics = CR3BP_Dynamics(earth_moon_system())
    if resonance.multiplier_gap > CRITICAL_MULTIPLIER_TOLERANCE:
        raise Cr3bpOrbitError(
            f"共振乘子 n 次幂与 1 的偏差 {resonance.multiplier_gap:.3e} 超过门限 "
            f"{CRITICAL_MULTIPLIER_TOLERANCE}，该平凡成员处起链不可行"
        )
    closure_period = resonance.revolutions * float(resonance.period)
    parts = {
        "real": np.real(resonance.eigenvector),
        "imag": np.imag(resonance.eigenvector),
    }
    best: tuple[float, np.ndarray] | None = None
    for vector in parts.values():
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            # 退化方向（如实特征向量的虚部为零）：无扰动可用，跳过。
            continue
        unit = vector / norm
        for epsilon in _DEFLECTION_LADDER:
            for sign in (1.0, -1.0):
                state = np.asarray(resonance.state, dtype=float) + sign * epsilon * unit
                refined = _refine_branch_seed(dynamics, resonance, state, closure_period, spatial)
                if refined.status is not ConvergenceState.CONVERGED:
                    continue
                member = Orbit(
                    states=np.asarray(refined.states[0], dtype=float).reshape(1, -1),
                    times=np.array([0.0]),
                    system=dynamics.system,
                )
                member.period = float(refined.periods[0])
                single = _branch_nontriviality(dynamics, member, resonance)
                if single <= 1e-8:
                    continue
                if best is None or single > best[0]:
                    best = (single, state)
    if best is None:
        raise Cr3bpOrbitError(
            "起链失败：共振平面 deflection 的全部组合在 n 圈闭合精化后仍落回"
            "平凡族（单圈末端回到初态），分岔支未被命中"
        )
    seed_state = best[1]

    entry = spatial_full_period_pal_py if spatial else planar_full_period_pal_py
    raw = entry(
        float(dynamics.system.mu),
        seed_state.tolist(),
        closure_period,
        n_orbits,
        step_size,
        initial_direction,
        rtol=dynamics.rtol,
        atol=dynamics.atol,
        max_step=dynamics.max_step,
        n_revolutions=resonance.revolutions,
    )
    orbits: list[Orbit] = []
    for member_state, period, closure_error, drift in zip(
        raw.states, raw.periods, raw.closure_errors, raw.jacobi_drifts, strict=True
    ):
        orbit = Orbit(
            states=np.asarray(member_state, dtype=float).reshape(1, -1),
            times=np.array([0.0]),
            system=dynamics.system,
        )
        orbit.period = float(period)
        orbit.family_type = family_type
        orbit.closure_error = float(closure_error)
        orbit.parameters["resonance_revolutions"] = resonance.revolutions
        orbit.parameters["jacobi_drift"] = float(drift)
        orbits.append(orbit)
    family = OrbitFamily(orbits=orbits, family_type=family_type, system=dynamics.system)
    family.metadata["bifurcation_seeding"] = "resonance-deflection"
    family.metadata["multiplier_gap"] = resonance.multiplier_gap
    return ContinuationResult(
        status=raw.status,
        cause=raw.cause,
        message=raw.message,
        family=family,
        steps=raw.steps,
        step_size=raw.step_size,
    )


# ---------------------------------------------------------------------------
# 乘子合并定位（#825）
# ---------------------------------------------------------------------------

#: 乘子对的特征向量与流方向（或族切向）的投影超过此值时，视为平凡方向候选。
TRIVIAL_PROJECTION_TOLERANCE = 0.9

#: 相邻成员之间同一乘子对 ν 的允许跳变量（相对量）。
_PAIR_JUMP_THRESHOLD = 0.3

#: 精化终止条件：|ν 减目标| 不超过此值，或 x0 区间宽度不超过 _MERGER_PARAMETER_TOL。
_MERGER_INDICATOR_TOL = 1e-6
_MERGER_PARAMETER_TOL = 1e-7
_MAX_REFINE_ITER = 50

#: 二分中点修正失败时依次尝试的区间比例（相对低端 x0）。
_REFINE_FRACTIONS = (0.5, 0.25, 0.75)

#: 参考相位下对称破缺分量所在的状态分量（y、vx、vz）。
_ANTISYMMETRIC_COMPONENTS = (1, 3, 5)

#: 横截特征向量增广矩阵最小奇异值的上限：超过它说明成员不在 +1 合并处。
_TRANSVERSE_SINGULAR_TOL = 0.5

#: 复特征值与共轭配对的容差。
_CONJUGATE_IMAG_TOL = 1e-10

#: 实数谱配对时，特征向量平凡性不一致的惩罚（仅用于打破近简并的平局）。
_CLASS_MISMATCH_PENALTY = 1e-6

_DEFAULT_SCAN_STEP = 0.0045
_DEFAULT_MAX_MEMBERS = 600

MergeKind = Literal["+1", "-1"]


@dataclass(frozen=True)
class MultiplierPair:
    """单值矩阵的一对乘子及其平凡性判别。"""

    #: 配对的两个特征值（复共轭对或实倒数对）。
    eigenvalues: tuple[complex, complex]
    #: 稳定性指数 ν = λ + 1/λ（取两成员的平均实部）。
    nu: float
    #: 两个特征值对应的特征向量（未归一化，复数）。
    eigenvectors: tuple[np.ndarray, np.ndarray]
    #: 是否为平凡乘子对（流方向与族方向张成的那对单位乘子）。
    is_trivial: bool
    #: 两个特征向量与流方向投影的较大值。
    flow_projection: float
    #: 两个特征向量与族切向投影的较大值。
    tangent_projection: float


@dataclass(frozen=True)
class MultiplierMerge:
    """一次乘子合并事件（非平凡对合到 +1 或 −1）。"""

    #: 合并种类：``"+1"`` 或 ``"-1"``。
    kind: MergeKind
    #: 临界成员的族参数 x0。
    parameter: float
    #: 事件在行走序列中的成员下标（精化前区间的高端）。
    member_index: int
    #: 临界成员的初始状态（6 维）。
    state: np.ndarray
    #: 临界成员的周期（TU）。
    period: float
    #: 靠近目标的那个特征值（复数）。
    multiplier: complex
    #: 乘子与目标的偏差：``|λ − 1|`` 或 ``|λ + 1|``。
    multiplier_gap: float
    #: 该乘子对的稳定性指数。
    nu: float
    #: 横截特征向量（6 维，归一化）。+1 合并能解析出对称破缺方向时取该方向，
    #: 退化时取所选乘子对特征向量的反对称部分。
    eigenvector: np.ndarray
    #: 横截特征向量与流方向的投影。
    flow_projection: float
    #: 横截特征向量与族切向的投影。
    tangent_projection: float
    #: 精化前的 x0 区间（升序）。
    bracket: tuple[float, float]
    #: 二分精化是否达到指示量容差。
    refined: bool


@dataclass(frozen=True)
class MultiplierTrace:
    """单个族成员的非平凡乘子轨迹。"""

    #: 成员的族参数 x0。
    parameter: float
    #: 成员的周期（TU）。
    period: float
    #: 该成员非平凡乘子对的 ν（升序）。
    nus: tuple[float, ...]


@dataclass(frozen=True)
class HaloMergeScan:
    """沿 Halo 族扫描乘子合并的结果。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    #: 沿途记录到的全部 ±1 合并事件（按行走顺序）。
    events: tuple[MultiplierMerge, ...]
    #: 首个可起链的 +1 合并构造的共振点，没有可用横截特征向量时为 None。
    resonance: ResonancePoint | None
    #: 实际计算的成员数（含种子）。
    members_scanned: int
    #: 扫描覆盖的 x0 区间（升序）。
    parameter_interval: tuple[float, float]
    #: 扫描覆盖的周期区间（升序，TU）。
    period_interval: tuple[float, float]
    #: 逐成员的非平凡乘子轨迹。
    trace: tuple[MultiplierTrace, ...]

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


def _unit_vector(vector: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm < 1e-12:
        raise ValueError(f"{label}向量为零，无法判别乘子对")
    return array / norm


def _projection(vector: np.ndarray, unit: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return 0.0
    return float(abs(np.vdot(vector, unit)) / norm)


def _reciprocal_cost(first: complex, second: complex) -> float:
    if abs(first) < 1e-12 or abs(second) < 1e-12:
        return math.inf
    return abs(first - 1.0 / second) + abs(second - 1.0 / first)


def _matchings(indices: tuple[int, ...]):
    if not indices:
        yield ()
        return
    first = indices[0]
    for position in range(1, len(indices)):
        rest = indices[1:position] + indices[position + 1 :]
        for tail in _matchings(rest):
            yield ((first, indices[position]), *tail)


def _match_eigenvalues(eigenvalues: np.ndarray, aligned: list[bool]) -> tuple[tuple[int, int], ...]:
    """把 6 个特征值配成 3 对：复特征值先按共轭配对，实特征值按倒数最近配对。

    共轭配对优先，避免在 +1 附近的四重近简并簇里把平凡对与非平凡对
    交叉配错。实特征值的配对代价以倒数接近度为主，特征向量的平凡性
    不一致只作微小平局惩罚。
    """
    count = len(eigenvalues)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for index in range(count):
        if index in used or abs(eigenvalues[index].imag) <= _CONJUGATE_IMAG_TOL:
            continue
        partners = [j for j in range(count) if j != index and j not in used]
        partner = min(partners, key=lambda j: abs(eigenvalues[j] - np.conj(eigenvalues[index])))
        pairs.append((index, partner))
        used.update((index, partner))
    rest = [index for index in range(count) if index not in used]
    if len(rest) == 2:
        pairs.append((rest[0], rest[1]))
    elif rest:
        best: tuple[tuple[int, int], ...] | None = None
        best_cost = math.inf
        for matching in _matchings(tuple(rest)):
            cost = 0.0
            for first, second in matching:
                cost += _reciprocal_cost(eigenvalues[first], eigenvalues[second])
                if aligned[first] != aligned[second]:
                    cost += _CLASS_MISMATCH_PENALTY
            if cost < best_cost:
                best_cost = cost
                best = matching
        assert best is not None
        pairs.extend(best)
    return tuple(pairs)


def classify_multiplier_pairs(
    matrix: np.ndarray,
    flow_direction: np.ndarray,
    family_tangent: np.ndarray,
    *,
    trivial_projection: float = TRIVIAL_PROJECTION_TOLERANCE,
) -> tuple[MultiplierPair, ...]:
    """把一个 6×6 单值矩阵的谱分成乘子对并显式判别平凡对。

    平凡对的判别不看乘子是否最接近 1，而看特征向量方向：特征向量与
    流方向或族切向的投影超过 ``trivial_projection`` 的乘子对才是平凡对
    候选。候选可能不止一对（非平凡对接近 +1 时其特征向量也贴近流方
    向），此时取 ν 最接近 2 的那对：平凡对的 ν 恒为 2，接近 +1 的临界
    非平凡对仍在 2 的一侧。

    Args:
        matrix: 6×6 单值矩阵。
        flow_direction: 参考点处的流方向（CR3BP 向量场）。
        family_tangent: 族切向（相邻成员状态差）。
        trivial_projection: 判定特征向量贴近平凡方向的投影阈值。

    Returns:
        三对乘子。``is_trivial`` 恰好标记一对（若存在对齐候选）。
    """
    values = np.asarray(matrix, dtype=float)
    if values.shape != (6, 6):
        raise ValueError(f"matrix 必须为 6×6，当前为 {values.shape}")
    flow = _unit_vector(flow_direction, "流方向")
    tangent = _unit_vector(family_tangent, "族切向")
    eigenvalues, eigenvectors = np.linalg.eig(values)
    projections = [
        max(
            _projection(eigenvectors[:, index], flow),
            _projection(eigenvectors[:, index], tangent),
        )
        for index in range(6)
    ]
    aligned = [value > trivial_projection for value in projections]
    pairs: list[MultiplierPair] = []
    for first, second in _match_eigenvalues(eigenvalues, aligned):
        nu = float(
            np.real(
                0.5
                * (
                    (eigenvalues[first] + 1.0 / eigenvalues[first])
                    + (eigenvalues[second] + 1.0 / eigenvalues[second])
                )
            )
        )
        pair = MultiplierPair(
            eigenvalues=(complex(eigenvalues[first]), complex(eigenvalues[second])),
            nu=nu,
            eigenvectors=(eigenvectors[:, first], eigenvectors[:, second]),
            is_trivial=False,
            flow_projection=max(
                _projection(eigenvectors[:, first], flow),
                _projection(eigenvectors[:, second], flow),
            ),
            tangent_projection=max(
                _projection(eigenvectors[:, first], tangent),
                _projection(eigenvectors[:, second], tangent),
            ),
        )
        pairs.append(pair)
    candidates = [
        index
        for index, pair in enumerate(pairs)
        if pair.flow_projection > trivial_projection or pair.tangent_projection > trivial_projection
    ]
    if candidates:
        trivial = min(candidates, key=lambda index: abs(pairs[index].nu - 2.0))
        pairs[trivial] = MultiplierPair(
            eigenvalues=pairs[trivial].eigenvalues,
            nu=pairs[trivial].nu,
            eigenvectors=pairs[trivial].eigenvectors,
            is_trivial=True,
            flow_projection=pairs[trivial].flow_projection,
            tangent_projection=pairs[trivial].tangent_projection,
        )
    return tuple(pairs)


def _merge_target(kind: MergeKind) -> float:
    return 2.0 if kind == "+1" else -2.0


def _merge_indicator(pair: MultiplierPair, kind: MergeKind) -> float:
    return pair.nu - _merge_target(kind)


def _sign_changed(previous: float, current: float) -> bool:
    if previous == 0.0 or current == 0.0:
        return previous != current
    return (previous > 0.0) != (current > 0.0)


def _merge_pair(pairs: tuple[MultiplierPair, ...], kind: MergeKind) -> MultiplierPair:
    """取非平凡对中 ν 最接近目标的一对。"""
    nontrivial = [pair for pair in pairs if not pair.is_trivial]
    return min(nontrivial, key=lambda pair: abs(pair.nu - _merge_target(kind)))


def _pair_spectrum(
    dynamics: CR3BP_Dynamics, orbit: Orbit, tangent: np.ndarray
) -> tuple[MultiplierPair, ...]:
    assert orbit.period is not None
    state = np.asarray(orbit.states[0], dtype=float)
    matrix = monodromy(dynamics, state, float(orbit.period))
    flow = np.asarray(dynamics.equations_of_motion(0.0, state), dtype=float)
    return classify_multiplier_pairs(matrix, flow, tangent)


def _tight_halo(dynamics: CR3BP_Dynamics, orbit: Orbit, guess: Orbit | None) -> tuple[Orbit, bool]:
    """用固定 x0 微分修正在同一 x0 处把成员收紧到闭合 1e-13 量级。

    原始 PAL 成员的闭合只到 1e-7 量级，平凡对的数值分裂可达 1e-3，
    与真实临界乘子同量级。精化成员必须先收紧，ν 才有意义。修正失败
    时退回原成员并返回 ``False``，调用方据此把该端点记为未达精化容差。
    """
    x0 = float(orbit.states[0, 0])
    try:
        return _correct_halo_x0(dynamics, x0, 2, guess if guess is not None else orbit), True
    except Cr3bpOrbitError:
        return orbit, False


def _transverse_direction(matrix: np.ndarray, flow: np.ndarray) -> np.ndarray | None:
    """求 +1 合并处分岔支的横截特征向量（对称破缺方向）。

    临界成员处单值矩阵在 λ=1 有约当链：流方向 f 在核空间，横截特征
    向量 v 满足 ``(M − I) v = c f``。相位条件 ``f·v = 0`` 排除沿流平移，
    分岔支切向落在 7×7 增广矩阵 ``[[M − I, f], [fᵀ, 0]]`` 的零空间里。
    零空间可能同时含族切向（对称、纯 x/z/vy 分量），因此取零空间基底
    中反对称分量最大、且去掉流分量后仍非零的组合。零空间没有可用的反
    对称方向时返回 None（该合并处无对称破缺分支方向）。
    """
    flow = np.asarray(flow, dtype=float)
    flow_norm = float(np.linalg.norm(flow))
    if flow_norm < 1e-12:
        return None
    f_unit = flow / flow_norm
    augmented = np.zeros((7, 7))
    augmented[:6, :6] = np.asarray(matrix, dtype=float) - np.eye(6)
    augmented[:6, 6] = f_unit
    augmented[6, :6] = f_unit
    _, singular, right = np.linalg.svd(augmented)
    anti = list(_ANTISYMMETRIC_COMPONENTS)
    best: np.ndarray | None = None
    best_norm = 0.0
    for row, sigma in zip(right, singular, strict=True):
        if sigma > _TRANSVERSE_SINGULAR_TOL:
            continue
        vector = np.asarray(row[:6], dtype=float)
        candidate = np.zeros(6)
        candidate[anti] = vector[anti]
        candidate -= float(np.dot(candidate, f_unit)) * f_unit
        length = float(np.linalg.norm(candidate))
        if length > best_norm:
            best_norm = length
            best = candidate / length
    if best is None or best_norm < 0.1:
        return None
    if best[int(np.argmax(np.abs(best)))] < 0.0:
        best = -best
    return best


def _pair_eigenvector(pair: MultiplierPair, target: float) -> np.ndarray:
    """乘子对里靠近目标的特征向量的反对称部分（归一化）。"""
    index = 0 if abs(pair.eigenvalues[0] - target) <= abs(pair.eigenvalues[1] - target) else 1
    vector = np.asarray(pair.eigenvectors[index], dtype=complex)
    anti = np.zeros(6, dtype=complex)
    anti[list(_ANTISYMMETRIC_COMPONENTS)] = vector[list(_ANTISYMMETRIC_COMPONENTS)]
    if np.linalg.norm(anti) < 0.5 * np.linalg.norm(vector):
        anti = vector
    length = float(np.linalg.norm(anti))
    if length < 1e-12:
        raise Cr3bpOrbitError("临界特征向量为零，无法构造合并事件")
    return anti / length


def _build_merge(
    dynamics: CR3BP_Dynamics,
    orbit: Orbit,
    pair: MultiplierPair,
    kind: MergeKind,
    tangent: np.ndarray,
    bracket: tuple[float, float],
    member_index: int,
    refined: bool,
) -> MultiplierMerge:
    assert orbit.period is not None
    state = np.asarray(orbit.states[0], dtype=float)
    matrix = monodromy(dynamics, state, float(orbit.period))
    flow = np.asarray(dynamics.equations_of_motion(0.0, state), dtype=float)
    target = 1.0 if kind == "+1" else -1.0
    multiplier = min(pair.eigenvalues, key=lambda value: abs(value - target))
    eigenvector = (
        _transverse_direction(matrix, flow) if kind == "+1" else _pair_eigenvector(pair, target)
    )
    if eigenvector is None:
        eigenvector = _pair_eigenvector(pair, target)
    eigenvector = np.asarray(eigenvector)
    flow_unit = _unit_vector(flow, "流方向")
    tangent_unit = _unit_vector(tangent, "族切向")
    return MultiplierMerge(
        kind=kind,
        parameter=float(state[0]),
        member_index=member_index,
        state=state,
        period=float(orbit.period),
        multiplier=complex(multiplier),
        multiplier_gap=float(abs(multiplier - target)),
        nu=pair.nu,
        eigenvector=eigenvector,
        flow_projection=_projection(eigenvector, flow_unit),
        tangent_projection=_projection(eigenvector, tangent_unit),
        bracket=bracket,
        refined=refined,
    )


def _refine_merge(
    dynamics: CR3BP_Dynamics,
    low_orbit: Orbit,
    high_orbit: Orbit,
    tangent: np.ndarray,
    kind: MergeKind,
    high_index: int,
) -> MultiplierMerge:
    """在 x0 区间内二分精化一次合并事件。"""
    low_orbit, low_tightened = _tight_halo(dynamics, low_orbit, None)
    high_orbit, high_tightened = _tight_halo(dynamics, high_orbit, low_orbit)
    low_x0 = float(low_orbit.states[0, 0])
    high_x0 = float(high_orbit.states[0, 0])
    low_pair = _merge_pair(_pair_spectrum(dynamics, low_orbit, tangent), kind)
    high_pair = _merge_pair(_pair_spectrum(dynamics, high_orbit, tangent), kind)
    low_indicator = _merge_indicator(low_pair, kind)
    high_indicator = _merge_indicator(high_pair, kind)
    guess = low_orbit
    best = (abs(low_indicator), low_orbit, low_pair, low_tightened)
    if abs(high_indicator) < best[0]:
        best = (abs(high_indicator), high_orbit, high_pair, high_tightened)
    bracket = (min(low_x0, high_x0), max(low_x0, high_x0))
    if low_x0 > high_x0:
        low_x0, high_x0 = high_x0, low_x0
        low_indicator, high_indicator = high_indicator, low_indicator
    crossed = _sign_changed(low_indicator, high_indicator)
    for _ in range(_MAX_REFINE_ITER if crossed else 0):
        width = high_x0 - low_x0
        if best[0] <= _MERGER_INDICATOR_TOL or width <= _MERGER_PARAMETER_TOL:
            break
        midpoint = None
        for fraction in _REFINE_FRACTIONS:
            candidate_x0 = low_x0 + fraction * width
            try:
                candidate = _correct_halo_x0(dynamics, candidate_x0, 2, guess)
            except Cr3bpOrbitError:
                continue
            midpoint = (candidate_x0, candidate)
            break
        if midpoint is None:
            break
        candidate_x0, candidate = midpoint
        guess = candidate
        pair = _merge_pair(_pair_spectrum(dynamics, candidate, tangent), kind)
        indicator = _merge_indicator(pair, kind)
        if abs(indicator) < best[0]:
            best = (abs(indicator), candidate, pair, True)
        if _sign_changed(low_indicator, indicator):
            high_x0, high_indicator = candidate_x0, indicator
        else:
            low_x0, low_indicator = candidate_x0, indicator
    refined = best[0] <= _MERGER_INDICATOR_TOL and best[3]
    return _build_merge(
        dynamics,
        best[1],
        best[2],
        kind,
        tangent,
        bracket,
        high_index,
        refined,
    )


@dataclass(frozen=True)
class _Crossing:
    kind: MergeKind
    low_index: int
    high_index: int


def _is_startable(event: MultiplierMerge) -> bool:
    """判断一个 +1 合并能否用于起链。

    横截特征向量与流方向、族切向的投影都要低于判别阈值，事件达到精化
    容差，乘子偏差也在 ``CRITICAL_MULTIPLIER_TOLERANCE`` 以内：下游
    ``generate_resonant_branch_family`` 对超出该门限的共振点直接拒绝。
    """
    return (
        max(event.flow_projection, event.tangent_projection) < TRIVIAL_PROJECTION_TOLERANCE
        and event.refined
        and event.multiplier_gap <= CRITICAL_MULTIPLIER_TOLERANCE
    )


def locate_halo_multiplier_merges(
    dynamics: CR3BP_Dynamics,
    *,
    seed_orbit: Orbit | None = None,
    max_members: int = _DEFAULT_MAX_MEMBERS,
    step_size: float = _DEFAULT_SCAN_STEP,
) -> HaloMergeScan:
    """沿 L2 南 Halo 族定位乘子过 +1 与 −1 的族成员。

    从 Richardson 小振幅种子（或调用方给定的族成员）沿振幅增长支单次
    PAL 行走，逐成员计算单值矩阵，按特征向量投影判别平凡乘子对，连续
    跟踪非平凡对的稳定性指数 ν，记录 ν 跨越 +2（乘子过 +1）与 −2
    （乘子过 −1）的相邻成员区间，再用固定 x0 微分修正二分精化。首个
    +1 合并构造 ResonancePoint 供 spatial 起链使用。

    种子须位于目标合并沿行走方向之前：行走方向固定为振幅增长支
    （``positive``），越过折叠点后族继续走向近月 NRHO 段。

    Args:
        dynamics: CR3BP 动力学。
        seed_orbit: 起始族成员，缺省用 L2 南 Halo 的 Richardson 种子。成员带
            ``parameters["libration_point"]`` 时校验必须为 L2，缺少该键时按 L2
            南族处理并重新标注，调用方须自行确认传入的是 L2 族成员。
        max_members: 成员数上限（含种子）。
        step_size: PAL 伪弧长步长。

    Returns:
        :class:`HaloMergeScan`。定位到可起链的 +1 合并（横截特征向量与流
        方向、族切向的投影都低于判别阈值，事件达到精化容差，乘子偏差不
        超过 ``CRITICAL_MULTIPLIER_TOLERANCE``）时状态为 CONVERGED。否则
        状态为 INFEASIBLE、原因为 NO_INTERSECTION，消息如实给出扫描区间、
        非平凡乘子对到 ±1 的最小偏差、终止情况与不可起链的原因。
    """
    if max_members < 2:
        raise ValueError(f"max_members 必须大于 1，当前为 {max_members}")
    if step_size <= 0.0:
        raise ValueError(f"step_size 必须为正数，当前为 {step_size}")
    seed = _prepare_halo_seed(dynamics, seed_orbit)
    continuation = Continuation(corrector=DifferentialCorrection(dynamics))
    family = halo_pseudo_arclength_continuation(
        continuation,
        seed_orbit=seed,
        n_orbits=max_members - 1,
        direction="positive",
        step_size=step_size,
        verbose=False,
    )
    members = list(family.orbits)[:max_members]
    if len(members) < 2:
        seed_state = np.asarray(members[0].states[0], dtype=float)
        assert members[0].period is not None
        seed_period = float(members[0].period)
        message = (
            f"沿 L2 南 Halo 族行走在 {len(members)} 条成员后提前中断（请求 {max_members} 条），"
            f"不足以计算相邻成员间的乘子轨迹。种子 x0={seed_state[0]:.6f}，"
            f"T={seed_period:.6f} TU"
        )
        return HaloMergeScan(
            status=ConvergenceState.INFEASIBLE,
            cause=FailureCause.NO_INTERSECTION,
            message=message,
            events=(),
            resonance=None,
            members_scanned=len(members),
            parameter_interval=(float(seed_state[0]), float(seed_state[0])),
            period_interval=(seed_period, seed_period),
            trace=(),
        )

    states = [np.asarray(member.states[0], dtype=float) for member in members]
    tangents: list[np.ndarray] = []
    for index in range(len(members)):
        if index + 1 < len(members):
            tangents.append(states[index + 1] - states[index])
        else:
            tangents.append(states[index] - states[index - 1])

    spectra = [
        _pair_spectrum(dynamics, member, tangents[index]) for index, member in enumerate(members)
    ]

    traces: list[MultiplierTrace] = []
    minimum_gap = math.inf
    for index, member in enumerate(members):
        assert member.period is not None
        nontrivial = [pair for pair in spectra[index] if not pair.is_trivial]
        traces.append(
            MultiplierTrace(
                parameter=float(states[index][0]),
                period=float(member.period),
                nus=tuple(sorted(pair.nu for pair in nontrivial)),
            )
        )
        for pair in nontrivial:
            for value in pair.eigenvalues:
                minimum_gap = min(minimum_gap, abs(value - 1.0), abs(value + 1.0))

    crossings: list[_Crossing] = []
    tracks: list[tuple[int, float, int]] = []
    next_track = 0
    for index, pairs in enumerate(spectra):
        nontrivial = [pair for pair in pairs if not pair.is_trivial]
        candidates: list[tuple[float, int, int]] = []
        for track_id, nu, _previous in tracks:
            for pair_index, pair in enumerate(nontrivial):
                distance = abs(pair.nu - nu)
                if distance <= _PAIR_JUMP_THRESHOLD * max(1.0, abs(nu)):
                    candidates.append((distance, track_id, pair_index))
        candidates.sort()
        matched_tracks: set[int] = set()
        matched_pairs: set[int] = set()
        next_tracks: list[tuple[int, float, int]] = []
        for _distance, track_id, pair_index in candidates:
            if track_id in matched_tracks or pair_index in matched_pairs:
                continue
            matched_tracks.add(track_id)
            matched_pairs.add(pair_index)
            previous_nu = next(nu for uid, nu, _ in tracks if uid == track_id)
            pair = nontrivial[pair_index]
            for kind in ("+1", "-1"):
                target = _merge_target(kind)
                if _sign_changed(previous_nu - target, pair.nu - target):
                    crossings.append(_Crossing(kind=kind, low_index=index - 1, high_index=index))
            next_tracks.append((track_id, pair.nu, index))
        for pair_index, pair in enumerate(nontrivial):
            if pair_index in matched_pairs:
                continue
            next_tracks.append((next_track, pair.nu, index))
            next_track += 1
        tracks = next_tracks

    events: list[MultiplierMerge] = []
    for crossing in crossings:
        tangent = states[crossing.high_index] - states[crossing.low_index]
        events.append(
            _refine_merge(
                dynamics,
                members[crossing.low_index],
                members[crossing.high_index],
                tangent,
                crossing.kind,
                crossing.high_index,
            )
        )

    parameter_interval = (
        min(trace.parameter for trace in traces),
        max(trace.parameter for trace in traces),
    )
    period_interval = (
        min(trace.period for trace in traces),
        max(trace.period for trace in traces),
    )
    plus_events = [event for event in events if event.kind == "+1"]
    minus_events = [event for event in events if event.kind == "-1"]
    usable = [event for event in plus_events if _is_startable(event)]
    resonance = _resonance_from_merge(usable[0]) if usable else None
    walked = len(members)
    gap_text = f"{minimum_gap:.3e}" if math.isfinite(minimum_gap) else "未找到非平凡乘子对"
    termination = (
        f"PAL 延拓在 {walked} 条成员后提前中断（请求 {max_members} 条）"
        if walked < max_members
        else f"PAL 延拓走满请求的 {max_members} 条成员"
    )
    if usable:
        first = usable[0]
        message = (
            f"沿 L2 南 Halo 族扫描 {walked} 个成员（x0 区间 "
            f"[{parameter_interval[0]:.6f}, {parameter_interval[1]:.6f}]），定位到 "
            f"{len(plus_events)} 个乘子过 +1 的临界成员，首个可起链的在 x0="
            f"{first.parameter:.9f}（乘子偏差 {first.multiplier_gap:.3e}，"
            f"ν={first.nu:.9f}，族切向投影 {first.tangent_projection:.3f}）。"
            f"沿途记录 {len(minus_events)} 个过 −1 的合并"
        )
        status, cause = ConvergenceState.CONVERGED, FailureCause.NONE
    elif plus_events:
        first = plus_events[0]
        refined_text = "达到" if first.refined else "未达到"
        message = (
            f"沿 L2 南 Halo 族扫描 {walked} 个成员（x0 区间 "
            f"[{parameter_interval[0]:.6f}, {parameter_interval[1]:.6f}]），定位到 "
            f"{len(plus_events)} 个乘子过 +1 的成员，但没有可起链的成员：要求横截特征向量"
            f"与流方向、族切向的投影都低于 {TRIVIAL_PROJECTION_TOLERANCE:.2f}，事件达到精化"
            f"容差，乘子偏差不超过 {CRITICAL_MULTIPLIER_TOLERANCE:.0e}。首个事件在 "
            f"x0={first.parameter:.9f}，流方向投影 {first.flow_projection:.3f}，族切向投影 "
            f"{first.tangent_projection:.3f}，精化容差{refined_text}，乘子偏差 "
            f"{first.multiplier_gap:.3e}。非平凡乘子对到 ±1 的最小偏差为 {gap_text}。"
            f"{termination}。沿途记录 {len(minus_events)} 个过 −1 的合并"
        )
        status, cause = ConvergenceState.INFEASIBLE, FailureCause.NO_INTERSECTION
    else:
        message = (
            f"沿 L2 南 Halo 族扫描 {walked} 个成员（x0 区间 "
            f"[{parameter_interval[0]:.6f}, {parameter_interval[1]:.6f}]，周期区间 "
            f"[{period_interval[0]:.6f}, {period_interval[1]:.6f}] TU），未定位到乘子过 "
            f"+1 的临界成员，非平凡乘子对到 ±1 的最小偏差为 {gap_text}。"
            f"{termination}。沿途记录 {len(minus_events)} 个过 −1 的合并"
        )
        status, cause = ConvergenceState.INFEASIBLE, FailureCause.NO_INTERSECTION
    return HaloMergeScan(
        status=status,
        cause=cause,
        message=message,
        events=tuple(events),
        resonance=resonance,
        members_scanned=walked,
        parameter_interval=parameter_interval,
        period_interval=period_interval,
        trace=tuple(traces),
    )


def _prepare_halo_seed(dynamics: CR3BP_Dynamics, seed_orbit: Orbit | None) -> Orbit:
    """复制并补全 Halo 种子（PAL 需要 libration_point、halo_class 与振幅参数）。"""
    if seed_orbit is None:
        orbit = _correct_halo(dynamics, -_HALO_SEED_Z0, 2, None)
    else:
        supplied = seed_orbit.parameters.get("libration_point")
        if supplied is not None and float(supplied) != 2.0:
            raise ValueError(
                f"种子族的平动点必须为 L2，当前为 {supplied}，本次定位沿 L2 南 Halo 族行走"
            )
        orbit = Orbit(
            states=np.asarray(seed_orbit.states, dtype=float).copy(),
            times=np.asarray(seed_orbit.times, dtype=float).copy(),
            system=dynamics.system,
        )
        orbit.period = seed_orbit.period
    if orbit.period is None:
        raise ValueError("种子轨道缺少周期，无法作为 Halo 族行走起点")
    z0 = float(orbit.states[0, 2])
    orbit.family_type = "halo"
    orbit.parameters = {
        **orbit.parameters,
        "libration_point": 2,
        "halo_class": 0 if z0 > 0.0 else 1,
        "amplitude_z": abs(z0),
    }
    return orbit


def _resonance_from_merge(event: MultiplierMerge) -> ResonancePoint:
    """把首个可起链的 +1 合并事件转成共振点（单圈，横截特征向量作为起链方向）。"""
    vector = np.asarray(event.eigenvector)
    if np.iscomplexobj(vector):
        vector = np.real(vector)
    return ResonancePoint(
        state=np.asarray(event.state, dtype=float).copy(),
        period=float(event.period),
        x0=float(event.parameter),
        phase_deg=0.0,
        multiplier_gap=float(event.multiplier_gap),
        eigenvector=np.asarray(vector, dtype=float).copy(),
        revolutions=1,
    )


__all__ = [
    "CRITICAL_MULTIPLIER_TOLERANCE",
    "HaloMergeScan",
    "MergeKind",
    "MultiplierMerge",
    "MultiplierPair",
    "MultiplierTrace",
    "ResonancePoint",
    "TRIVIAL_PROJECTION_TOLERANCE",
    "classify_multiplier_pairs",
    "deflect_resonance_seed",
    "generate_resonant_branch_family",
    "locate_dro_revolution_resonance",
    "locate_halo_multiplier_merges",
    "monodromy",
    "per_revolution_closure",
]
