"""分岔共振点的定位与起链（#820）。

在平凡族上定位 p:n 共振成员（非平凡复数乘子对的辐角扫过 2π/n，其 n
次幂落回 1），沿共振平面的实向量扰动出种子，交给全周期 PAL 在 n 圈闭
合形式下精化并行走。定位是算法层内部自含能力，不构成接口层的稳定性或
分岔分析工具（#687 的边界）。

机制说明：共振成员处平凡族的 n 圈闭合雅可比退化（共振乘子对的 n 次幂
落到 1 上），普通牛顿的收敛盆把有限振幅种子吸回平凡叶。起链用共振特
征向量的实部与虚部（共振平面的两组基）做 deflection，给出分支切向的
一阶近似，再用阻尼最小二乘（固定 x0 加相位条件）把种子收到分岔支上，
精化后校验种子确已离开平凡族（单圈末端不回到初态）。

单值矩阵恒有等于 1 的平凡乘子对（相位方向与族切向），数值上常分裂为
1±1e-6 量级的实对。共振判据因此不得用“最接近 1 的乘子”，必须显式跟
踪非平凡的共振复数对（issue #820 记录的 1.0000214 正是平凡对数值分裂
的五次幂，不是共振乘子）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...data.types.orbit import Orbit, OrbitFamily
from ...integrators import planar_full_period_pal_py, spatial_full_period_pal_py
from ...status import ConvergenceState
from ..dynamics import CR3BP_Dynamics
from ..results import ContinuationResult
from .orbits.dro import _correct_dro
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


__all__ = [
    "CRITICAL_MULTIPLIER_TOLERANCE",
    "ResonancePoint",
    "deflect_resonance_seed",
    "generate_resonant_branch_family",
    "locate_dro_revolution_resonance",
    "monodromy",
    "per_revolution_closure",
]
