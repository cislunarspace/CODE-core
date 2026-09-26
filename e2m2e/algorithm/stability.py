"""
稳定性分析模块

提供轨道稳定性分析功能，包括单值矩阵计算、Floquet乘子分析、分岔检测等。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data.types.orbit import Orbit
from .dynamics import CR3BP_Dynamics

logger = logging.getLogger(__name__)

__all__ = [
    "BifurcationType",
    "BranchJump",
    "FamilyBifurcationPoint",
    "FamilyBifurcationScan",
    "MemberAnalysisFailure",
    "OrbitStability",
    "StabilityAnalysis",
    "StabilityType",
]


class StabilityType(Enum):
    """稳定性类型枚举"""

    STABLE = "stable"
    UNSTABLE = "unstable"
    MARGINALLY_STABLE = "marginally_stable"
    HYPERBOLIC = "hyperbolic"
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"


class BifurcationType(Enum):
    """分岔类型枚举"""

    NONE = "none"
    PERIOD_DOUBLING = "period_doubling"
    SADDLE_NODE = "saddle_node"
    TORUS = "torus"
    PITCHFORK = "pitchfork"
    TRANSCRITICAL = "transcritical"
    SECONDARY_HOPF = "secondary_hopf"


@dataclass(frozen=True)
class OrbitStability:
    """轨道稳定性分析结果。

    由 ``StabilityAnalysis.analyze()`` 返回的结果容器，包含单值矩阵、
    Floquet 乘子、稳定性指数、分类与分岔信息。

    Attributes:
        monodromy_matrix: 单值矩阵，形状 (6, 6)
        eigenvalues: 单值矩阵特征值数组
        stability_indices: 稳定性指数字典
        classification: 稳定性分类结果
        bifurcation: 分岔分析结果
        numerical_errors: 数值误差估计
    """

    monodromy_matrix: np.ndarray | None
    eigenvalues: np.ndarray | None
    stability_indices: dict[str, float | None]
    classification: dict[str, Any]
    bifurcation: dict[str, Any]
    numerical_errors: dict[str, float | None]


#: 倒数对配对的乘积容差（沿用贪心配对的历史值 0.01）。
_PAIR_PRODUCT_TOL = 0.01

#: 平凡乘子对判据：自治 Hamilton 流的单值矩阵恒有 λ=1（时间平移/能量
#: 方向），其 ν = λ + 1/λ = 2 不随族参数变化；数值噪声量级 ~1e-11（λ 数值
#: 劈裂 ~1e-5 时 ν−2 ~1e-10）。该阈值用于逐成员识别平凡对。
_TRIVIAL_NU_TOL = 1e-6

#: 逐类型的实轴分岔判据零点偏移：判据 = Re ν + offset，过零即分岔。
#: 检测（`_crossing_type`）与精化（`_bifurcation_indicator`）共用同一映射，
#: 防止两侧漂移导致「检测到的类型」与「精化用的判据」不一致。
_CRITERION_OFFSETS: dict[BifurcationType, float] = {
    BifurcationType.SADDLE_NODE: -2.0,
    BifurcationType.PERIOD_DOUBLING: 2.0,
}


def _reciprocal_pairs(eigenvalues: np.ndarray) -> list[tuple[complex, complex]]:
    """辛矩阵乘子的倒数对配对（贪心：λ_i·λ_j ≈ 1）。

    Args:
        eigenvalues: 单值矩阵特征值数组。

    Returns:
        倒数对列表 ``[(λ_i, λ_j), ...]``；未配对成功的乘子不出现在结果中。
    """
    pairs: list[tuple[complex, complex]] = []
    used: set[int] = set()

    for i in range(len(eigenvalues)):
        if i in used:
            continue
        for j in range(i + 1, len(eigenvalues)):
            if j in used:
                continue
            product = eigenvalues[i] * eigenvalues[j]
            if abs(product - 1.0) < _PAIR_PRODUCT_TOL:
                pairs.append((eigenvalues[i], eigenvalues[j]))
                used.add(i)
                used.add(j)
                break

    return pairs


def _pair_stability_index(pair: tuple[complex, complex]) -> complex:
    """倒数对的稳定性指数 ``ν = λ_far + 1/λ_far``（保留虚部）。

    取 |λ| ≥ 1 的成员 λ_far（并列时任取）。物理口径：实对 λ>1 ↔ ν 为
    >2 的实数；单位圆共轭对 ↔ ν = 2cosθ ∈ (−2, 2) 的实数；离圆复四元组
    ↔ Im ν ≠ 0。
    """
    lam1, lam2 = pair
    far = lam1 if abs(lam1) >= abs(lam2) else lam2
    if far == 0:
        far = lam2 if far is lam1 else lam1
        if far == 0:
            raise ValueError("特征值对包含零乘子，稳定性指数无定义")
    return complex(far) + 1.0 / complex(far)


def _member_nus(eigenvalues: np.ndarray) -> list[complex]:
    """成员乘子 → 三条倒数对的稳定性指数。

    乘子未组成 3 个倒数对（6×6 辛矩阵的常规形态）即视为该成员分析失败，
    由调用方记入 failures。
    """
    pairs = _reciprocal_pairs(eigenvalues)
    if len(pairs) != 3:
        raise ValueError(f"Floquet 乘子未组成 3 个倒数对（得到 {len(pairs)} 对）")
    return [_pair_stability_index(pair) for pair in pairs]


def _drop_trivial_pair(nus: Sequence[complex]) -> list[complex]:
    """剔除平凡乘子对（自治 Hamilton 流的时间平移/能量对，ν ≡ 2）。

    平凡对的 ν 恒为 2（数值噪声 ~1e-11），而真实分岔对**穿过** ν = 2：按值
    跟踪时两者在临界成员上互换次序，逐成员剔除平凡对才不会让跟踪在临界处
    交换身份而吞掉穿越（L2 Lyapunov 实测：面外 ν 1.9859 → 2.0056 与平凡对
    2.000000000000 在临界成员上分配退化）。
    """
    if len(nus) < 2:
        return list(nus)
    nearest = min(range(len(nus)), key=lambda index: abs(nus[index] - 2.0))
    if abs(nus[nearest] - 2.0) > _TRIVIAL_NU_TOL:
        return list(nus)
    return [nu for index, nu in enumerate(nus) if index != nearest]


def _bifurcation_indicator(kind: BifurcationType, nu: complex, ns_imag_tol: float) -> float:
    """分岔判据函数：穿越判据的标量形式（过零即分岔）。"""
    offset = _CRITERION_OFFSETS.get(kind)
    if offset is not None:
        return nu.real + offset
    return abs(nu.imag) - ns_imag_tol


def _crossing_type(
    nu_lo: complex,
    nu_hi: complex,
    ns_imag_tol: float,
    *,
    nu_before: complex | None = None,
) -> BifurcationType | None:
    """单区间、单条 ν 轨迹上的穿越判据（变号或离圆 onset）。

    实轴穿越分两种：判据在区间两端**严格异号**；或判据恰为零的**左端点**成员
    （网格节点恰好落在临界参数上）——后者只在该节点确为符号变化点时报出，即前一个
    成员（``nu_before``）与右端点的判据异号；``nu_before`` 缺省或也为零时不报。
    因此零判据节点若只是被触及而未穿过（前后同号）不报点，真穿越也只有一个区间
    认领。族参数域首端的零判据成员没有前一个成员可比，不被认领（其判据端点值为
    零，调用方可从原始成员表读到）。
    """
    if abs(nu_lo.imag) <= ns_imag_tol and abs(nu_hi.imag) <= ns_imag_tol:
        for kind, offset in _CRITERION_OFFSETS.items():
            lower = nu_lo.real + offset
            upper = nu_hi.real + offset
            if (lower < 0.0 < upper) or (upper < 0.0 < lower):
                return kind
            if lower == 0.0 and upper != 0.0 and nu_before is not None:
                before = nu_before.real + offset
                if before != 0.0 and (before < 0.0) != (upper < 0.0):
                    return kind
    if abs(nu_lo.imag) <= ns_imag_tol < abs(nu_hi.imag):
        return BifurcationType.TORUS
    return None


def _nearest_nu(nus: Sequence[complex], reference: complex) -> complex:
    """在成员的三条 ν 中认领最接近参考值的跟踪对。"""
    return min(nus, key=lambda nu: abs(nu - reference))


def _refine_family_point(
    *,
    kind: BifurcationType,
    interval: tuple[float, float],
    nus: tuple[complex, complex],
    member_at: Callable[[float], Orbit],
    analyze: Callable[[Orbit], np.ndarray],
    ns_imag_tol: float,
    parameter_tol: float,
    indicator_tol: float,
    max_refine_iter: int,
    causes: list[str],
) -> tuple[float, Orbit, np.ndarray, float] | None:
    """对分岔穿越区间二分精化，返回 (族参数, 轨道, 乘子, |判据|)。

    分点修正失败时依次试 1/4、3/4 分点（``_walk_family`` 先例，绕开共振
    缝隙）；三点全失败返回 None，由调用方记入失败清单并放弃该点。

    Args:
        causes: 失败原因收集器（出参）：逐个分点试算的异常按出现次序、去重后
            追加，供调用方写进 ``MemberAnalysisFailure.message``，与成员循环
            同为「``类名: 消息``」口径。
    """
    lo, hi = interval
    nu_lo, nu_hi = nus
    ind_lo = _bifurcation_indicator(kind, nu_lo, ns_imag_tol)
    best: tuple[float, Orbit, np.ndarray, float] | None = None
    for _ in range(max_refine_iter):
        sampled = False
        for fraction in (0.5, 0.25, 0.75):
            p_try = lo + fraction * (hi - lo)
            try:
                orbit_try = member_at(p_try)
                eigenvalues = np.asarray(analyze(orbit_try), dtype=complex)
                candidates = _drop_trivial_pair(_member_nus(eigenvalues))
                nu_try = _nearest_nu(candidates, 0.5 * (nu_lo + nu_hi))
            except Exception as exc:
                cause = f"{type(exc).__name__}: {exc}"
                if cause not in causes:
                    causes.append(cause)
                continue
            sampled = True
            ind_try = _bifurcation_indicator(kind, nu_try, ns_imag_tol)
            residual = abs(ind_try)
            if best is None or residual < best[3]:
                best = (p_try, orbit_try, eigenvalues, residual)
            if residual <= indicator_tol or (hi - lo) <= parameter_tol:
                return best
            if (ind_try > 0.0) == (ind_lo > 0.0):
                lo, ind_lo, nu_lo = p_try, ind_try, nu_try
            else:
                hi, nu_hi = p_try, nu_try
            break
        if not sampled:
            return None
    return best


@dataclass(frozen=True)
class FamilyBifurcationPoint:
    """族级分岔点（沿族参数定位的单个分岔穿越）。

    Attributes:
        parameter: 族参数值（精化收敛后的取值；粗扫时为判据过零的线性插值估计）。
        parameter_name: 族参数名（如 ``x0``、``vz0``）。
        type: 分岔类型。乘子层面无法区分 pitchfork 与 saddle-node，穿越
            ν=+2 统一报 :attr:`BifurcationType.SADDLE_NODE`。
        orbit: 临界处的族成员轨道。
        multipliers: 临界处的全部 6 个 Floquet 乘子。
        residual: 分岔判据的绝对值 ``|判据|``。
    """

    # 跳支不产生分岔点：伴随跳支的区间记入 ``FamilyBifurcationScan.branch_jumps``
    # （区间级、带两端族参数与最大位移），因此本类型不带逐点 ``branch_jump`` 标记。
    parameter: float
    parameter_name: str
    type: BifurcationType
    orbit: Orbit
    multipliers: np.ndarray
    residual: float


@dataclass(frozen=True)
class BranchJump:
    """族成员间的跳支区间：乘子位移超阈，不产生分岔点。"""

    parameter_lo: float
    parameter_hi: float
    max_displacement: float


@dataclass(frozen=True)
class MemberAnalysisFailure:
    """族成员分析失败记录（该成员两侧的区间都不参与配对与判据）。"""

    parameter: float
    message: str


@dataclass(frozen=True)
class FamilyBifurcationScan:
    """沿族参数的分岔扫描结果。

    Attributes:
        points: 分岔点，按 ``parameter`` 严格升序；同一族参数上不同类型的穿越各
            保留一条（同参数同类型的重复检测只留残差最小者）。
        branch_jumps: 跳支区间（乘子位移超阈）；区间内不报穿越，故分岔点上不再
            逐点标注是否伴随跳支。
        failures: 成员分析失败记录。
    """

    points: tuple[FamilyBifurcationPoint, ...]
    branch_jumps: tuple[BranchJump, ...]
    failures: tuple[MemberAnalysisFailure, ...]


class StabilityAnalysis:
    """Monodromy-based stability analysis of a single periodic orbit. / 轨道稳定性分析

    计算轨道的单值矩阵、Floquet乘子、稳定性指数等，
    并进行稳定性分类和分岔检测。

    Attributes:
        orbit: Orbit对象
        dynamic: CR3BP_Dynamics对象
        monodromy_matrix: 单值矩阵
        eigenvalues: 特征值
        stability_indices: 稳定性指数
    """

    # 类属性
    STABILITY_THRESHOLD = 1e-6
    BIFURCATION_TOLERANCE = 1e-8

    def __init__(self, orbit: Orbit, dynamics: CR3BP_Dynamics | None = None) -> None:
        """初始化分析器

        Args:
            orbit: Orbit对象
            dynamics: CR3BP_Dynamics对象（可选，如果orbit关联了system则自动创建）
        """
        self.orbit = orbit
        system = getattr(orbit, "system", None)
        if dynamics is None and system is not None:
            dynamics = CR3BP_Dynamics(system)
        self.dynamics = dynamics

        self.monodromy_matrix: np.ndarray | None = None
        self.stm_history: list[np.ndarray] = []

        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors = None
        self.eigenvalue_magnitudes = None
        self.eigenvalue_arguments = None
        self.sorted_eigenvalues = None
        self.eigenvalue_pairs: list[tuple[complex, complex]] = []

        self.stability_indices: dict[str, float | None] = {
            "nu1": None,
            "nu2": None,
            "nu3": None,
            "broucke": None,
        }

        self.floquet_multipliers = None
        self.floquet_exponents = None

        self.lyapunov_exponents: list[float] | np.ndarray = []
        self.max_lyapunov_exponent = None

        self.stability_type = None
        self.is_stable = False
        self.is_unstable = False
        self.is_critical = False
        self.stability_margin = None

        self.bifurcation_type = BifurcationType.NONE
        self.bifurcation_detected = False

        self.numerical_errors: dict[str, float | None] = {
            "determinant_error": None,
            "symplectic_error": None,
        }

        self.has_monodromy = False
        self.has_eigenvalues = False
        self.analysis_complete = False

    def compute_monodromy(self):
        """计算单值矩阵

        通过积分一个完整周期的状态转移矩阵获得单值矩阵。

        Returns:
            np.ndarray: 6x6 单值矩阵
        """
        if self.dynamics is None:
            raise ValueError("需要提供dynamics对象才能计算单值矩阵")

        if self.orbit.period is None:
            raise ValueError("轨道周期未知，无法计算单值矩阵")

        initial_state = self.orbit.states[0]
        period = self.orbit.period

        # 使用动力学对象计算STM
        # 从轨道初始状态出发，积分一个完整周期得到 STM(T)，即单值矩阵
        self.monodromy_matrix = self.dynamics.compute_state_transition_matrix(initial_state, period)

        self.has_monodromy = True

        # 辛矩阵行列式恒为 1，若偏差过大则说明积分精度不够
        det = np.linalg.det(self.monodromy_matrix)
        self.numerical_errors["determinant_error"] = abs(det - 1.0)

        # 检查辛性质：M^T J M - J 应为零矩阵
        # 其中 J = [[0, I], [-I, 0]] 为辛结构矩阵
        J = np.zeros((6, 6))
        J[:3, 3:] = np.eye(3)
        J[3:, :3] = -np.eye(3)
        symplectic_residual = self.monodromy_matrix.T @ J @ self.monodromy_matrix - J
        self.numerical_errors["symplectic_error"] = np.linalg.norm(symplectic_residual)

        return self.monodromy_matrix

    def compute_floquet_multipliers(self):
        """计算Floquet乘子（特征值）

        Returns:
            np.ndarray: Floquet乘子
        """
        if not self.has_monodromy:
            self.compute_monodromy()

        # 求单值矩阵的特征值，即 Floquet 乘子 λ
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.monodromy_matrix)

        self.eigenvalue_magnitudes = np.abs(self.eigenvalues)  # |λ|
        self.eigenvalue_arguments = np.angle(self.eigenvalues)  # arg(λ)

        # 按幅值降序排列，方便识别主导模态
        sort_idx = np.argsort(-self.eigenvalue_magnitudes)
        self.sorted_eigenvalues = self.eigenvalues[sort_idx]

        self.floquet_multipliers = self.eigenvalues.copy()

        if self.orbit.period is not None and self.orbit.period > 0:
            self.floquet_exponents = np.log(self.eigenvalues + 0j) / self.orbit.period

        self.has_eigenvalues = True

        self._pair_eigenvalues()

        return self.floquet_multipliers

    def _pair_eigenvalues(self):
        """配对特征值（辛矩阵特征值成倒数对）。

        辛矩阵的特征值成倒数对出现，即 λ_i * λ_j ≈ 1。
        使用贪心匹配将每个特征值与最接近其倒数的特征值配对。

        Returns:
            None（结果存入 self.eigenvalue_pairs）
        """
        assert self.eigenvalues is not None  # 调用前必已求特征值
        self.eigenvalue_pairs = _reciprocal_pairs(self.eigenvalues)

    def compute_stability_index(self):
        """计算稳定性指数

        Broucke稳定性参数定义为：ν = λ + 1/λ，
        其中λ是单值矩阵的特征值。

        Returns:
            dict: 稳定性指数字典
        """
        if not self.has_eigenvalues:
            self.compute_floquet_multipliers()

        for i, (lam1, lam2) in enumerate(self.eigenvalue_pairs):
            # Broucke 稳定性参数：ν = λ + 1/λ（倒数对之和）
            nu = np.real(lam1 + lam2)  # ν = λ + 1/λ
            key = f"nu{i + 1}"
            if key in self.stability_indices:
                self.stability_indices[key] = nu

        # Broucke稳定性指数
        if len(self.eigenvalue_pairs) >= 2:
            nu1 = self.stability_indices.get("nu1", 0)
            nu2 = self.stability_indices.get("nu2", 0)
            if nu1 is not None and nu2 is not None:
                self.stability_indices["broucke"] = abs(nu1) + abs(nu2)

        return self.stability_indices

    def classify_orbit(self):
        """对轨道进行稳定性分类

        Returns:
            dict: 稳定性分类结果
        """
        if not self.has_eigenvalues:
            self.compute_floquet_multipliers()

        magnitudes = self.eigenvalue_magnitudes
        max_magnitude = np.max(magnitudes)
        min_magnitude = np.min(magnitudes)

        threshold = self.STABILITY_THRESHOLD

        # 所有 Floquet 乘子都在单位圆上 → Lyapunov 稳定
        all_on_unit_circle = np.all(np.abs(magnitudes - 1.0) < threshold)

        if all_on_unit_circle:
            self.stability_type = StabilityType.STABLE
            self.is_stable = True
        elif max_magnitude > 1.0 + threshold:
            self.stability_type = StabilityType.UNSTABLE
            self.is_unstable = True
        else:
            self.stability_type = StabilityType.MARGINALLY_STABLE
            self.is_critical = True

        # 更精细分类：区分双曲型（实数不稳定）和椭圆型（复数不稳定）
        has_real_unstable = False
        has_complex_unstable = False

        for lam in self.eigenvalues:
            if abs(lam) > 1.0 + threshold:
                if abs(np.imag(lam)) < threshold:
                    has_real_unstable = True
                else:
                    has_complex_unstable = True

        if has_real_unstable and not has_complex_unstable:
            self.stability_type = StabilityType.HYPERBOLIC
        elif has_complex_unstable:
            # 复数不稳定模态（螺旋型）
            pass

        # 稳定裕度
        self.stability_margin = 1.0 - max_magnitude

        # 计算Lyapunov指数
        if self.orbit.period is not None and self.orbit.period > 0:
            self.lyapunov_exponents = np.log(magnitudes) / self.orbit.period
            self.max_lyapunov_exponent = np.max(self.lyapunov_exponents)

        self.analysis_complete = True

        return {
            "stability_type": self.stability_type,
            "is_stable": self.is_stable,
            "is_unstable": self.is_unstable,
            "stability_margin": self.stability_margin,
            "max_eigenvalue_magnitude": max_magnitude,
            "min_eigenvalue_magnitude": min_magnitude,
            "lyapunov_exponents": self.lyapunov_exponents,
            "max_lyapunov_exponent": self.max_lyapunov_exponent,
        }

    def analyze_bifurcation(self):
        """分析分岔类型

        通过检查单值矩阵特征值的分布判断是否存在分岔。

        Returns:
            dict: 分岔分析结果
        """
        if not self.has_eigenvalues:
            self.compute_floquet_multipliers()

        tol = self.BIFURCATION_TOLERANCE

        for lam in self.eigenvalues:
            mag = abs(lam)

            # 鞍结分岔：特征值穿过 +1（稳定分支消失或创建）
            if abs(lam - 1.0) < tol:
                self.bifurcation_type = BifurcationType.SADDLE_NODE
                self.bifurcation_detected = True
                break

            # 倍周期分岔：特征值穿过 -1
            if abs(lam + 1.0) < tol:
                self.bifurcation_type = BifurcationType.PERIOD_DOUBLING
                self.bifurcation_detected = True
                break

            # 环面（Neimark-Sacker）分岔：复轨特征值穿过单位圆
            if abs(mag - 1.0) < tol and abs(np.imag(lam)) > tol:
                self.bifurcation_type = BifurcationType.TORUS
                self.bifurcation_detected = True
                break

        return {
            "bifurcation_type": self.bifurcation_type,
            "bifurcation_detected": self.bifurcation_detected,
            "eigenvalues": self.eigenvalues,
        }

    def analyze(self) -> OrbitStability:
        """执行完整稳定性分析并返回不可变结果对象。

        Returns:
            OrbitStability: 包含单值矩阵、特征值、稳定性指数、
            分类、分岔与数值误差的独立结果对象。
        """
        self.compute_monodromy()
        self.compute_floquet_multipliers()
        self.compute_stability_index()
        classification = self.classify_orbit()
        bifurcation = self.analyze_bifurcation()

        return OrbitStability(
            monodromy_matrix=self.monodromy_matrix,
            eigenvalues=self.eigenvalues,
            stability_indices=self.stability_indices.copy(),
            classification=classification,
            bifurcation=bifurcation,
            numerical_errors=self.numerical_errors.copy(),
        )

    def full_analysis(self):
        """执行完整的稳定性分析

        Returns:
            dict: 完整分析结果
        """
        result = self.analyze()
        return {
            "monodromy_matrix": result.monodromy_matrix,
            "eigenvalues": result.eigenvalues,
            "stability_indices": result.stability_indices,
            "classification": result.classification,
            "bifurcation": result.bifurcation,
            "numerical_errors": result.numerical_errors,
        }

    def __str__(self):
        status = self.stability_type.value if self.stability_type else "未分析"
        return f"StabilityAnalysis(type={status})"

    def __repr__(self):
        return (
            f"StabilityAnalysis(orbit={self.orbit}, "
            f"type={self.stability_type}, complete={self.analysis_complete})"
        )

    @staticmethod
    def detect_bifurcation_in_family(
        orbits: list[Orbit],
        parameters: Sequence[float],
        *,
        parameter_name: str = "parameter",
        member_at: Callable[[float], Orbit] | None = None,
        dynamics: CR3BP_Dynamics | None = None,
        multiplier_fn: Callable[[Orbit], np.ndarray] | None = None,
        parameter_tol: float = 1e-6,
        indicator_tol: float = 1e-6,
        ns_imag_tol: float = 1e-6,
        jump_threshold: float = 0.3,
        max_refine_iter: int = 50,
    ) -> FamilyBifurcationScan:
        """沿族参数扫描分岔穿越（多成员、结构化结果）。

        逐成员算 Floquet 乘子 → 倒数对配对 → 稳定性指数 ν = λ_far + 1/λ_far，
        相邻成员间用最小总位移分配跟踪 ν 轨迹，再按穿越判据在各区间定分岔：

        - ``Re ν − 2`` 变号且两端 |Im ν| 均在容差内 → SADDLE_NODE（乘子层面
          无法区分 pitchfork 与 saddle-node，统一报 saddle-node）；
        - ``Re ν + 2`` 变号且两端 |Im ν| 均在容差内 → PERIOD_DOUBLING；
        - |Im ν| 由容差内升到容差外（复四元组离开单位圆）→ TORUS。

        平凡乘子对（自治 Hamilton 流的时间平移/能量对，ν ≡ 2）逐成员剔除，
        不参与跟踪与判据——分岔对会**穿过** ν = 2，两者在临界成员上按值不可
        区分，逐成员剔除才不会让跟踪在临界处交换身份而吞掉穿越。区间内跟踪对的
        乘子位移超过 ``jump_threshold`` 的**量级缩放阈值**
        （``|Δν| > jump_threshold · max(1, |ν|)``）判为跳支，记入
        ``branch_jumps``，且不在该区间报穿越。缩放是必需的：共线 Lyapunov
        族的面内稳定性指数量级达 10³ 且逐步变化数十，属平滑延拓，绝对阈值
        会把每个区间都误判为跳支，连真实穿越一并屏蔽。

        两种模式：

        - 粗扫（``member_at=None``）：只用给定成员定位，分岔点的 ``parameter``
          取判据过零的线性插值估计，``orbit``/``multipliers`` 取 |判据| 较小的
          区间端成员；
        - 精化（给出 ``member_at``）：对穿越区间二分，每步取分点轨道重算判据，
          保留变号子区间，收敛到 ``|判据| ≤ indicator_tol``、区间宽
          ``≤ parameter_tol`` 或迭代上限（取历次 |判据| 最小者）。分点修正失败
          时依次试 1/4、3/4 分点，全失败记 :class:`MemberAnalysisFailure` 并放弃
          该点。

        Args:
            orbits: 族成员轨道，与 ``parameters`` 一一对应。
            parameters: 族参数（严格递增；``x0``、``vz0``、振幅等皆可）。
            parameter_name: 族参数名，写入结果点。
            member_at: 给定族参数 → 该处已修正轨道（精化用；缺省走粗扫）。
            dynamics: CR3BP 动力学；缺省时 ``StabilityAnalysis`` 从 ``orbit.system``
                自建。
            multiplier_fn: 轨道 → Floquet 乘子（缺省走真实单值矩阵；自定义
                monodromy 来源与合成族的正式注入点）。
            parameter_tol: 精化结束的族参数区间宽阈值。
            indicator_tol: 精化结束的 |判据| 阈值。
            ns_imag_tol: 判据的 |Im ν| 容差（区分实对与离圆复四元组）。
            jump_threshold: 跳支判据的 |Δν| 阈值（按 ν 量级缩放，见上）。
            max_refine_iter: 二分精化迭代上限。

        Returns:
            :class:`FamilyBifurcationScan`；``points`` 按族参数升序。

        Raises:
            ValueError: 成员数与参数数不等、成员少于 2 或参数非严格递增。
        """
        params = [float(value) for value in parameters]
        if len(orbits) != len(params):
            raise ValueError(f"orbits 与 parameters 长度必须一致（{len(orbits)} != {len(params)}）")
        if len(orbits) < 2:
            raise ValueError("族分岔扫描至少需要 2 个族成员")
        for lower, upper in zip(params[:-1], params[1:], strict=True):
            if not upper > lower:
                raise ValueError(f"族参数必须严格递增，实际出现 {lower} -> {upper}")

        # 缺省乘子来源：真实单值矩阵（Rust STM 传播，无 scipy 回退）
        analyze: Callable[[Orbit], np.ndarray] = (
            multiplier_fn
            if multiplier_fn is not None
            else lambda orbit: StabilityAnalysis(orbit, dynamics).compute_floquet_multipliers()
        )

        failures: list[MemberAnalysisFailure] = []
        member_nus: list[list[complex] | None] = []
        member_multipliers: list[np.ndarray | None] = []
        for parameter, orbit in zip(params, orbits, strict=True):
            try:
                eigenvalues = np.asarray(analyze(orbit), dtype=complex)
                nus = _drop_trivial_pair(_member_nus(eigenvalues))
            except Exception as exc:
                failures.append(
                    MemberAnalysisFailure(
                        parameter=parameter, message=f"{type(exc).__name__}: {exc}"
                    )
                )
                member_nus.append(None)
                member_multipliers.append(None)
                continue
            member_nus.append(nus)
            member_multipliers.append(eigenvalues)

        # ν 轨迹：逐成员 3 槽位，相邻成员间用最小总位移分配保序配对；
        # 成员分析失败的缺口两侧不跨接（后续成员另起新轨迹）。
        tracks: list[dict[int, complex]] = []
        track_of: list[list[int] | None] = []
        previous_slots: list[int] | None = None
        for index, slot_nus in enumerate(member_nus):
            if slot_nus is None:
                track_of.append(None)
                previous_slots = None
                continue
            if previous_slots is None:
                slots = []
                for nu in slot_nus:
                    tracks.append({index: nu})
                    slots.append(len(tracks) - 1)
            else:
                cost = np.array(
                    [
                        [abs(nu - tracks[slot][index - 1]) for nu in slot_nus]
                        for slot in previous_slots
                    ]
                )
                rows, columns = linear_sum_assignment(cost)
                # 逐成员剔除平凡对后 ν 条数可为 2 或 3（成员缺平凡对时不会被剔除），
                # 此时 cost 是矩形矩阵，linear_sum_assignment 只给出 min(行,列) 个分配：
                # 未分配的列用 -1 占位并另起一条新轨迹，绝不复用 0 这类合法 track id
                # （否则会与既有轨迹撞号——同一 track 被写两次、且该区间比较出现重复槽位）。
                slots = [-1] * len(slot_nus)
                for row, column in zip(rows, columns, strict=True):
                    slot = previous_slots[row]
                    tracks[slot][index] = slot_nus[column]
                    slots[column] = slot
                for column, slot in enumerate(slots):
                    if slot < 0:
                        tracks.append({index: slot_nus[column]})
                        slots[column] = len(tracks) - 1
            track_of.append(slots)
            previous_slots = slots

        detections: list[tuple[int, BifurcationType, complex, complex]] = []
        branch_jumps: list[BranchJump] = []
        for index in range(len(params) - 1):
            slots_lo = track_of[index]
            slots_hi = track_of[index + 1]
            if slots_lo is None or slots_hi is None:
                continue
            # dict.fromkeys：同一区间两侧共有的槽位去重（保序），避免重复报点。
            shared = [slot for slot in dict.fromkeys(slots_lo) if slot in set(slots_hi)]
            # 跳支判据按 ν 量级缩放（|Δν| > jump_threshold·max(1, |ν|)）：共线
            # Lyapunov 族的面内稳定性指数量级达 1e3 且逐步变化数十，本身是平滑
            # 延拓，绝对阈值会把每个区间误判为跳支、连真实穿越一并屏蔽。
            displacements = [
                (slot, abs(tracks[slot][index + 1] - tracks[slot][index])) for slot in shared
            ]
            jumps = [
                (slot, displacement)
                for slot, displacement in displacements
                if displacement
                > jump_threshold
                * max(
                    1.0,
                    abs(tracks[slot][index]),
                    abs(tracks[slot][index + 1]),
                )
            ]
            if jumps:
                branch_jumps.append(
                    BranchJump(
                        parameter_lo=params[index],
                        parameter_hi=params[index + 1],
                        max_displacement=float(max(displacement for _, displacement in jumps)),
                    )
                )
                continue
            for slot in shared:
                nu_lo = tracks[slot][index]
                nu_hi = tracks[slot][index + 1]
                # nu_before：判据恰为零的左端点成员需要它确认符号确实变化
                kind = _crossing_type(
                    nu_lo, nu_hi, ns_imag_tol, nu_before=tracks[slot].get(index - 1)
                )
                if kind is not None:
                    detections.append((index, kind, nu_lo, nu_hi))

        points: list[FamilyBifurcationPoint] = []
        for index, kind, nu_lo, nu_hi in detections:
            p_lo, p_hi = params[index], params[index + 1]
            if member_at is None:
                f_lo = _bifurcation_indicator(kind, nu_lo, ns_imag_tol)
                f_hi = _bifurcation_indicator(kind, nu_hi, ns_imag_tol)
                span = f_lo - f_hi
                weight = 0.5 if span == 0.0 else f_lo / span
                near = index if abs(f_lo) <= abs(f_hi) else index + 1
                multipliers = member_multipliers[near]
                assert multipliers is not None  # 该端成员已成功分析
                points.append(
                    FamilyBifurcationPoint(
                        parameter=p_lo + weight * (p_hi - p_lo),
                        parameter_name=parameter_name,
                        type=kind,
                        orbit=orbits[near],
                        multipliers=multipliers,
                        residual=min(abs(f_lo), abs(f_hi)),
                    )
                )
                continue
            causes: list[str] = []
            refined = _refine_family_point(
                kind=kind,
                interval=(p_lo, p_hi),
                nus=(nu_lo, nu_hi),
                member_at=member_at,
                analyze=analyze,
                ns_imag_tol=ns_imag_tol,
                parameter_tol=parameter_tol,
                indicator_tol=indicator_tol,
                max_refine_iter=max_refine_iter,
                causes=causes,
            )
            if refined is None:
                reason = f"；样本失败原因：{'；'.join(causes[:2])}" if causes else ""
                failures.append(
                    MemberAnalysisFailure(
                        parameter=0.5 * (p_lo + p_hi),
                        message=f"分岔精化区间 [{p_lo}, {p_hi}] 内成员修正均失败{reason}",
                    )
                )
                continue
            parameter, orbit, multipliers, residual = refined
            points.append(
                FamilyBifurcationPoint(
                    parameter=parameter,
                    parameter_name=parameter_name,
                    type=kind,
                    orbit=orbit,
                    multipliers=multipliers,
                    residual=residual,
                )
            )

        # 同参数、同类型的重复检测（同一条穿越被两次认领）只保留残差最小的一条，
        # 保证 ``points`` 的族参数严格递增；参数相同但类型不同者各留一条。
        points.sort(key=lambda point: point.parameter)
        deduped: list[FamilyBifurcationPoint] = []
        for point in points:
            previous = deduped[-1] if deduped else None
            if (
                previous is not None
                and previous.parameter == point.parameter
                and previous.type is point.type
            ):
                if point.residual < previous.residual:
                    deduped[-1] = point
                continue
            deduped.append(point)
        points = deduped
        if failures:
            logger.warning("族分岔扫描中 %d/%d 个成员或精化点分析失败", len(failures), len(params))

        return FamilyBifurcationScan(
            points=tuple(points),
            branch_jumps=tuple(branch_jumps),
            failures=tuple(failures),
        )

    @staticmethod
    def find_nearest_bifurcation(
        scan: FamilyBifurcationScan, target: float
    ) -> FamilyBifurcationPoint | None:
        """取扫描结果中族参数最接近 ``target`` 的分岔点；无点返回 None。

        Args:
            scan: :meth:`detect_bifurcation_in_family` 的结果。
            target: 目标族参数值。

        Returns:
            分岔点，或 None（扫描无点）。
        """
        if not scan.points:
            return None
        return min(scan.points, key=lambda point: abs(point.parameter - target))
