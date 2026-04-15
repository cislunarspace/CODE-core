"""
稳定性分析模块

提供轨道稳定性分析功能，包括单值矩阵计算、Floquet乘子分析、分岔检测等。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from ..core.dynamics import CR3BP_Dynamics
from ..core.orbit import Orbit


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


class StabilityAnalysis:
    """轨道稳定性分析

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
        self.dynamics = dynamics

        self.monodromy_matrix = None
        self.stm_history = []

        self.eigenvalues = None
        self.eigenvectors = None
        self.eigenvalue_magnitudes = None
        self.eigenvalue_arguments = None
        self.sorted_eigenvalues = None
        self.eigenvalue_pairs = []

        self.stability_indices = {
            "nu1": None,
            "nu2": None,
            "nu3": None,
            "broucke": None,
        }

        self.floquet_multipliers = None
        self.floquet_exponents = None

        self.lyapunov_exponents = []
        self.max_lyapunov_exponent = None

        self.stability_type = None
        self.is_stable = False
        self.is_unstable = False
        self.is_critical = False
        self.stability_margin = None

        self.bifurcation_type = BifurcationType.NONE
        self.bifurcation_detected = False

        self.numerical_errors = {
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
        """配对特征值（辛矩阵特征值成倒数对）"""
        self.eigenvalue_pairs = []
        used = set()

        for i in range(len(self.eigenvalues)):
            if i in used:
                continue
            for j in range(i + 1, len(self.eigenvalues)):
                if j in used:
                    continue
                # 检查是否为倒数对
                product = self.eigenvalues[i] * self.eigenvalues[j]
                if abs(product - 1.0) < 0.01:
                    self.eigenvalue_pairs.append((self.eigenvalues[i], self.eigenvalues[j]))
                    used.add(i)
                    used.add(j)
                    break

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
            # Broucke 稳定性参数：ν = λ + 1/λ = λ + λ_conj（倒数关系）
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

    def full_analysis(self):
        """执行完整的稳定性分析

        Returns:
            dict: 完整分析结果
        """
        self.compute_monodromy()
        self.compute_floquet_multipliers()
        self.compute_stability_index()
        classification = self.classify_orbit()
        bifurcation = self.analyze_bifurcation()

        return {
            "monodromy_matrix": self.monodromy_matrix,
            "eigenvalues": self.eigenvalues,
            "stability_indices": self.stability_indices,
            "classification": classification,
            "bifurcation": bifurcation,
            "numerical_errors": self.numerical_errors,
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
        dynamics: CR3BP_Dynamics,
        tolerance: float = 1e-8,
    ) -> list[dict[str, Any]]:
        """检测轨道族中的分岔点

        遍历轨道族中的每条轨道，计算其单值矩阵特征值，
        检测是否有特征值接近 +1（切分岔/saddle-node bifurcation）。

        Args:
            orbits: Orbit对象列表（轨道族）
            dynamics: CR3BP_Dynamics对象
            tolerance: 特征值接近 +1 的容差，默认 1e-8

        Returns:
            List[Dict[str, Any]]: 分岔点列表，每个元素包含：
                - orbit_index: 轨道在族中的索引
                - orbit: Orbit对象
                - eigenvalues: 特征值数组
                - eigenvalue_diff: |λ - 1| 的最小值
                - bifurcation_type: 分岔类型
        """
        bifurcation_points = []

        for i, orbit in enumerate(orbits):
            try:
                analysis = StabilityAnalysis(orbit=orbit, dynamics=dynamics)
                analysis.compute_floquet_multipliers()

                for j, lam in enumerate(analysis.eigenvalues):
                    diff = abs(lam - 1.0)
                    if diff < tolerance:
                        bifurcation_points.append(
                            {
                                "orbit_index": i,
                                "orbit": orbit,
                                "eigenvalues": analysis.eigenvalues,
                                "eigenvalue_diff": diff,
                                "bifurcation_type": BifurcationType.SADDLE_NODE,
                                "eigenvalue_index": j,
                                "eigenvalue": lam,
                            }
                        )

            except Exception:
                continue

        return bifurcation_points

    @staticmethod
    def find_nearest_bifurcation(
        orbits: list[Orbit],
        dynamics: CR3BP_Dynamics,
        target_x0: float | None = None,
        tolerance: float = 1e-4,
    ) -> dict[str, Any] | None:
        """在轨道族中找到最接近目标参数的分岔点

        Args:
            orbits: Orbit对象列表
            dynamics: CR3BP_Dynamics对象
            target_x0: 目标x0坐标（可选）
            tolerance: 搜索容差

        Returns:
            分岔点字典，如果未找到则返回None
        """
        bifurcation_points = StabilityAnalysis.detect_bifurcation_in_family(
            orbits=orbits,
            dynamics=dynamics,
            tolerance=tolerance,
        )

        if not bifurcation_points:
            return None

        if target_x0 is None:
            return bifurcation_points[0]

        best_bp = None
        best_dist = float("inf")

        for bp in bifurcation_points:
            x0 = bp["orbit"].states[0][0]
            dist = abs(x0 - target_x0)
            if dist < best_dist:
                best_dist = dist
                best_bp = bp

        return best_bp
