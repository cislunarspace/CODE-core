"""
微分修正算法模块

提供用于求解周期轨道的微分修正算法，支持多种对称性配置。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ...data.types.orbit import Orbit
from ..dynamics import CR3BP_Dynamics

# Re-export Richardson approximation functions for backward compatibility.
# These moved to halo_initial_guess.py in v4.3.
from ..family.halo_initial_guess import (  # noqa: F401
    compute_halo_coefficients,
    compute_halo_initial_guess,
    halo_third_order_approximation,
)

if TYPE_CHECKING:
    from ..family.strategies.base import CorrectionConfig

logger = logging.getLogger(__name__)


class DifferentialCorrection:
    """微分修正算法

    通过迭代修正初始条件，使轨道满足指定的约束条件（如周期性、对称性等）。

    支持的对称性配置：
    - 2D对称X固定X0: 平面对称周期轨道，固定初始x坐标
    - 2D对称X固定T: 平面对称周期轨道，固定轨道周期
    - 3D对称X固定X0: 空间对称周期轨道（Halo轨道等）
    - 3D对称XZ固定X0: 空间XZ对称周期轨道
    - 3D对称XZ固定Z0: 空间XZ对称周期轨道，固定Z0

    Attributes:
        dynamic: CR3BP_Dynamics对象
        target_conditions: 目标约束条件字典
        free_variables: 自由变量列表
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        convergence_history: 收敛历史
    """

    # 类属性
    DEFAULT_TOLERANCE = 1e-12
    DEFAULT_MAX_ITERATIONS = 50
    DEFAULT_DAMPING_FACTOR = 1.0
    VALID_SETUP_TYPES = [
        "2D_symmetric_x_fixed_x0",
        "2D_symmetric_x_fixed_t",
        "2D_symmetric_y_fixed_y0",
        "3D_symmetric_x_fixed_x0",
        "3D_symmetric_xz_fixed_x0",
        "3D_symmetric_xz_fixed_z0",
        "axial_orbit_fixed_vz0",
        "halo_orbit_fixed_z0",
        "halo_orbit_fixed_x0",
        "spo_fixed_x0",
        "lpo_fixed_x0",
    ]

    def __init__(
        self,
        dynamic: CR3BP_Dynamics,
        target: dict[str, Any] | None = None,
        free_vars: list[str] | None = None,
    ) -> None:
        """初始化修正器

        Args:
            dynamic: CR3BP_Dynamics对象
            target: 目标约束条件字典（可选）
            free_vars: 自由变量列表（可选）
        """
        self.dynamics: CR3BP_Dynamics = dynamic
        self.target_conditions = target or {}
        self.free_variables = free_vars or []

        self.tolerance = self.DEFAULT_TOLERANCE
        self.max_iterations = self.DEFAULT_MAX_ITERATIONS
        self.damping_factor = self.DEFAULT_DAMPING_FACTOR

        self.convergence_history: list[float] = []
        self.error_history: list[float] = []
        self.correction_history: list[float] = []
        self.iteration_count = 0
        self.converged = False

        self.current_error = None

        self.initial_guess = None
        self.final_solution = None
        self.solution_time = None

        self.jacobian_matrix = None

        self.constraint_indices: list[int] = []
        self.free_variable_indices: list[int] = []

        self.setup_type: str | None = None
        self.symmetry_condition: str | None = None
        self.fixed_parameters: dict[str, float] = {}

        self.finite_difference_step = 1e-7

        self.stagnation_limit = 1e-14
        self.divergence_limit = 1e10

        self.performance_stats = {
            "total_time": 0.0,
            "stm_evaluations": 0,
            "constraint_evaluations": 0,
            "jacobian_evaluations": 0,
        }

        self.termination_reason = None
        self.success = False

    def setup_2D_symmetric_x_fixed_x0(self, x0=0.0):
        """配置平面问题中固定初始x坐标的对称周期轨道搜索

        在平面圆形限制性三体问题（PCRTBP）模型中，动力学方程关于会合坐标系的x轴具有对称性。
        利用这一性质，周期轨道的搜索可以简化为寻找合适的初始条件：
        从x轴上一点垂直出发（y=0, x_dot=0），经过半周期T/2后再次垂直穿越x轴（y=0, x_dot=0）。

        本函数针对这种对称性设置微分修正问题，固定初始x坐标x0，将初始y方向速度y_dot0
        和半周期T/2作为自由变量进行调整，以满足终点处的垂直穿越条件。

        Args:
            x0: 固定的初始x坐标，轨道从点(x0, 0)垂直出发。
                默认值 0.0 对应 L1 平动点附近的典型初始位置

        Returns:
            配置好的微分修正器实例

        Note:
            - 自由变量: [y_dot0, T_half] - 初始y方向速度和半周期时间
            - 目标约束: [y(T/2)=0, x_dot(T/2)=0] - 终点处再次垂直穿越x轴
            - 状态向量索引: [1, 3] 分别对应y坐标和x方向速度

            上述配置对应于Broucke(1968)等经典文献中寻找对称周期轨道的基本方法，
            可用于生成围绕平动点或主天体的各类周期轨道家族。

        Reference:
            Broucke R A. Periodic orbits in the restricted three body problem
            with Earth-moon masses[R]. 1968.
        """
        from ..family.strategies import symmetric_2d_fixed_x0

        config = symmetric_2d_fixed_x0(x0)
        self._apply_config(config)

        self._reset_history()

        logger.debug(
            "2D对称x轴配置完成：固定x0=%s，自由变量=%s，目标约束=%s",
            x0,
            self.free_variables,
            list(self.target_conditions.keys()),
        )

        return self

    def setup_2D_symmetric_x_fixed_t(self, t_half):
        """配置平面问题中固定半周期的对称周期轨道搜索

        固定半周期T/2，调整初始条件x0和y_dot0满足约束。

        Args:
            t_half: 固定的半周期

        Returns:
            配置好的微分修正器实例
        """
        from ..family.strategies import symmetric_2d_fixed_t

        config = symmetric_2d_fixed_t(t_half)
        self._apply_config(config)

        self._reset_history()
        return self

    def setup_2D_symmetric_y_fixed_y0(self, y0=0.0):
        """配置平面问题中固定初始y坐标的y轴对称周期轨道搜索

        适用于共振轨道(RO)等从y轴出发的周期轨道。
        轨道从点(0, y0)出发（x=0, x_dot=0），经过半周期T/2后再次穿越y轴（x=0, x_dot=0）。

        Args:
            y0: 固定的初始y坐标

        Returns:
            配置好的微分修正器实例

        Note:
            - 自由变量: [x_dot0, T_half] - 初始x方向速度和半周期时间
            - 目标约束: [x(T/2)=0, x_dot(T/2)=0] - 终点处再次穿越y轴
            - 状态向量索引: [0, 3] 分别对应x坐标和x方向速度
        """
        from ..family.strategies import symmetric_2d_fixed_y0

        config = symmetric_2d_fixed_y0(y0)
        self._apply_config(config)

        self._reset_history()

        logger.debug(
            "2D对称y轴配置完成：固定y0=%s，自由变量=%s，目标约束=%s",
            y0,
            self.free_variables,
            list(self.target_conditions.keys()),
        )

        return self

    def setup_3D_symmetric_x_fixed_x0(self, x0):
        """配置空间问题中固定初始x坐标的对称周期轨道搜索（如Halo轨道）

        Args:
            x0 (float): 固定的初始x坐标

        Returns:
            self: 配置好的微分修正器实例
        """
        from ..family.strategies import symmetric_3d_fixed_x0

        config = symmetric_3d_fixed_x0(x0)
        self._apply_config(config)

        self._reset_history()
        return self

    def setup_3D_symmetric_xz_fixed_x0(self, x0):
        """配置空间XZ对称周期轨道搜索，固定X0

        Args:
            x0 (float): 固定的初始x坐标

        Returns:
            self: 配置好的微分修正器实例
        """
        from ..family.strategies import symmetric_xz_fixed_x0

        config = symmetric_xz_fixed_x0(x0)
        self._apply_config(config)

        self._reset_history()
        return self

    def setup_3D_symmetric_xz_fixed_z0(self, z0):
        """配置空间XZ对称周期轨道搜索，固定Z0

        Args:
            z0 (float): 固定的初始z坐标

        Returns:
            self: 配置好的微分修正器实例
        """
        from ..family.strategies import symmetric_xz_fixed_z0

        config = symmetric_xz_fixed_z0(z0)
        self._apply_config(config)

        self._reset_history()
        return self

    def setup_halo_orbit_fixed_z0(self, z0, libration_point=1):
        """配置 Halo 轨道微分修正，固定初始 Z0（XZ 对称）

        Halo 轨道具有 XZ 平面对称性，利用该对称性可以将问题简化为：
        从 XZ 平面上一点 (x0, 0, z0) 出发，经过半周期 T/2 后再次到达 XZ 平面。

        Args:
            z0 (float): 固定的初始 z 坐标
            libration_point (int): 平动点编号 (1=L1, 2=L2)，默认 L1

        Returns:
            self: 配置好的微分修正器实例

        Note:
            - 自由变量: [x0, y_dot0, T_half] - 初始 x 坐标、y 方向速度和半周期时间
            - 目标约束: [y(T/2)=0, x_dot(T/2)=0, z_dot(T/2)=0] - 半周期处再次位于 XZ 平面
            - 状态向量索引: [0, 4, 6] 分别对应 x0、y_dot0 和时间 T_half
            - 注意: z 在半周期时会改变符号，不作为约束
        """
        from ..family.strategies import halo_fixed_z0

        config = halo_fixed_z0(z0, libration_point)
        self._apply_config(config)

        self._reset_history()

        logger.debug(
            "Halo 轨道配置完成（固定 Z0）：z0=%s，平动点=L%s，自由变量=%s，目标约束=%s",
            z0,
            libration_point,
            self.free_variables,
            list(self.target_conditions.keys()),
        )

        return self

    def setup_halo_orbit_fixed_x0(self, x0, libration_point=1):
        """配置 Halo 轨道微分修正，固定初始 X0（XZ 对称）

        Halo 轨道具有 XZ 平面对称性，利用该对称性可以将问题简化为：
        从 XZ 平面上一点 (x0, 0, z0) 出发，经过半周期 T/2 后再次到达 XZ 平面。

        Args:
            x0 (float): 固定的初始 x 坐标
            libration_point (int): 平动点编号 (1=L1, 2=L2)，默认 L1

        Returns:
            self: 配置好的微分修正器实例

        Note:
            - 自由变量: [z0, y_dot0, T_half] - 初始 z 坐标、y 方向速度和半周期时间
            - 目标约束: [y(T/2)=0, x_dot(T/2)=0, z_dot(T/2)=0] - 半周期处再次位于 XZ 平面且垂直穿越
            - 状态向量索引: [2, 4, 6] 分别对应 z0、y_dot0 和时间 T_half
        """
        from ..family.strategies import halo_fixed_x0

        config = halo_fixed_x0(x0, libration_point)
        self._apply_config(config)

        self._reset_history()

        logger.debug(
            "Halo 轨道配置完成（固定 X0）：x0=%s，平动点=L%s，自由变量=%s，目标约束=%s",
            x0,
            libration_point,
            self.free_variables,
            list(self.target_conditions.keys()),
        )

        return self

    def setup_axial_orbit_fixed_vz0(self, vz0, libration_point=1):
        """配置 Axial 轨道微分修正，固定初始 vz0（x 轴对称，Type B）。

        Axial 轨道关于 x 轴对称，初始状态 (x0, 0, 0, 0, y_dot0, vz0)，
        半周期处回到 x 轴 (y=0, z=0, x_dot=0)。

        与 Halo（Type A, z0≠0, vz0=0）的区别：Axial 从 xy 平面出发，
        获得面外速度后在半周期返回 xy 平面。

        Args:
            vz0 (float): 固定的初始 z 方向速度
            libration_point (int): 平动点编号 (1=L1, 2=L2)，默认 L1

        Returns:
            self: 配置好的微分修正器实例

        Note:
            - 自由变量: [x0, y_dot0, T_half]
            - 目标约束: [y(T/2)=0, z(T/2)=0, x_dot(T/2)=0]
        """
        from ..family.strategies import axial_fixed_vz0

        config = axial_fixed_vz0(vz0, libration_point)
        self._apply_config(config)

        self._reset_history()

        logger.debug(
            "Axial 轨道配置完成（固定 vz0）：vz0=%s，平动点=L%s，自由变量=%s，目标约束=%s",
            vz0,
            libration_point,
            self.free_variables,
            list(self.target_conditions.keys()),
        )

        return self

    def setup_spo_fixed_x0(self, x0, libration_point=5):
        """配置 SPO 通用平面周期修正，固定 x₀（无对称性假设）。

        L4/L5 短周期族是 xy 平面内围绕三角平动点的周期轨道，
        不具有 x 轴或 xz 平面对称性（y₀≠0）。直接求解全周期闭合
        条件：state(T) - state(0) = 0。

        Args:
            x0 (float): 固定的初始 x 坐标（族参数）
            libration_point (int): 平动点编号 (4=L4, 5=L5)，默认 5

        Returns:
            self: 配置好的微分修正器实例

        Note:
            - 自由变量: [y₀, ẋ₀, ẏ₀, T]（4 个）
            - 目标约束: [Δy=0, Δẋ=0, Δẏ=0]（3 个，全周期闭合）
            - Δx/Δz 约束省略：x₀ 已固定（Δx 自动满足），z 方向解耦（Δz 行恒零）
            - 4 自由 vs 3 约束 → 欠定系统，用 lstsq 求最小范数修正
            - 使用 iterate_full_period_correction 方法（非 iterate_correction）
        """
        from ..family.strategies import spo_fixed_x0

        config = spo_fixed_x0(x0, libration_point)
        self._apply_config(config)
        self._reset_history()

        logger.debug(
            "SPO 配置完成（固定 x0）：x0=%s，平动点=L%s，自由变量=%s，目标约束=%s",
            x0,
            libration_point,
            self.free_variables,
            list(self.target_conditions.keys()),
        )

        return self

    def setup_lpo_fixed_x0(self, x0, libration_point=5):
        """配置 LPO 通用平面周期修正，固定 x₀（无对称性假设）。

        与 setup_spo_fixed_x0 同构。LPO 是 L4/L5 长周期族，大振幅成员
        呈马蹄形（Horseshoe）。使用 iterate_full_period_correction 方法。

        Args:
            x0 (float): 固定的初始 x 坐标（族参数）
            libration_point (int): 平动点编号 (4=L4, 5=L5)，默认 5

        Returns:
            self: 配置好的微分修正器实例
        """
        from ..family.strategies import lpo_fixed_x0

        config = lpo_fixed_x0(x0, libration_point)
        self._apply_config(config)
        self._reset_history()

        logger.debug(
            "LPO 配置完成（固定 x0）：x0=%s，平动点=L%s，自由变量=%s，目标约束=%s",
            x0,
            libration_point,
            self.free_variables,
            list(self.target_conditions.keys()),
        )

        return self

    def _apply_config(self, config: CorrectionConfig) -> None:
        """将不可变的 CorrectionConfig 应用到当前修正器实例。

        Args:
            config: 策略函数生成的 CorrectionConfig 对象。
        """
        self.setup_type = config.setup_type
        self.symmetry_condition = config.symmetry_condition
        self.fixed_parameters = dict(config.fixed_parameters)
        self.free_variables = list(config.free_variables)
        self.free_variable_indices = list(config.free_variable_indices)
        self.target_conditions = dict(config.target_conditions)
        self.constraint_indices = list(config.constraint_indices)

    def _reset_history(self):
        """重置收敛历史"""
        self.convergence_history = []
        self.error_history = []
        self.correction_history = []
        self.iteration_count = 0
        self.converged = False
        self.termination_reason = None
        self.success = False

    def _compute_jacobian_finite_diff(self, current_state, current_time):
        """使用有限差分法计算雅可比矩阵

        Args:
            current_state: 当前初始状态
            current_time: 当前半周期时间

        Returns:
            jacobian: 雅可比矩阵
        """
        n_constraints = len(self.constraint_indices)
        n_variables = len(self.free_variable_indices)
        jacobian = np.zeros((n_constraints, n_variables))
        eps = self.finite_difference_step

        for j, var_idx in enumerate(self.free_variable_indices):
            if var_idx < 6:
                state_fwd = current_state.copy()
                state_fwd[var_idx] += eps
                result_fwd = self.dynamics.propagate(
                    state_fwd,
                    (0, current_time),
                    t_eval=[current_time],
                )
                final_fwd = result_fwd["states"][-1]

                state_bwd = current_state.copy()
                state_bwd[var_idx] -= eps
                result_bwd = self.dynamics.propagate(
                    state_bwd,
                    (0, current_time),
                    t_eval=[current_time],
                )
                final_bwd = result_bwd["states"][-1]

                sensitivity = (final_fwd - final_bwd) / (2 * eps)
                for i, c_idx in enumerate(self.constraint_indices):
                    jacobian[i, j] = sensitivity[c_idx]

            elif var_idx == 6:
                t_fwd = current_time + eps
                result_fwd = self.dynamics.propagate(
                    current_state,
                    (0, t_fwd),
                    t_eval=[t_fwd],
                )
                final_fwd = result_fwd["states"][-1]

                t_bwd = current_time - eps
                result_bwd = self.dynamics.propagate(
                    current_state,
                    (0, t_bwd),
                    t_eval=[t_bwd],
                )
                final_bwd = result_bwd["states"][-1]

                sensitivity = (final_fwd - final_bwd) / (2 * eps)
                for i, c_idx in enumerate(self.constraint_indices):
                    jacobian[i, j] = sensitivity[c_idx]

        self.performance_stats["jacobian_evaluations"] += 1
        return jacobian

    def iterate_correction(self, initial_guess, verbose=False, callback=None):
        """迭代修正主算法（基于STM的牛顿法）

        通过状态转移矩阵(STM)构建雅可比矩阵，使用牛顿迭代法修正自由变量，
        使终点状态满足目标约束条件，从而找到精确的周期轨道。

        Args:
            initial_guess (Orbit):
                初始猜测轨道，或初始状态向量
            verbose (bool):
                是否打印迭代过程信息
            callback (Callable[[int, float, bool], None] | None):
                每次迭代结束后的回调函数，参数为
                (iteration, error, converged)。converged 为 True 表示
                本次迭代后已收敛或因发散/停滞而终止。

        Returns:
            Orbit | None: 修正后的周期轨道对象；
                若修正失败（发散、雅可比奇异、周期无效等）则返回 None。
        """
        _STATE_INDEX_TO_KEY = {
            0: "x",
            1: "y",
            2: "z",
            3: "x_dot",
            4: "y_dot",
            5: "z_dot",
        }

        self.initial_guess = initial_guess.states[0]
        self.iteration_count = 0
        self.converged = False
        self.success = False

        # 固定T模式下使用预设的T_half，否则从轨道周期计算
        if "T_half" in self.fixed_parameters:
            half_period_time = self.fixed_parameters["T_half"]
        else:
            half_period_time = initial_guess.period / 2

        current_state = self.initial_guess.copy()
        current_time = half_period_time

        if verbose:
            logger.info("=" * 60)
            logger.info("开始微分修正迭代（STM牛顿法）...")
            logger.info("=" * 60)
            logger.info(
                "初始状态: x=%.6f, y=%.6f, z=%.6f",
                self.initial_guess[0],
                self.initial_guess[1],
                self.initial_guess[2],
            )
            logger.info(
                "         x_dot=%.6f, y_dot=%.6f, z_dot=%.6f",
                self.initial_guess[3],
                self.initial_guess[4],
                self.initial_guess[5],
            )
            logger.info("初始半周期: T/2=%.6f", half_period_time)
            logger.info("=" * 60)

        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1

            # 1. 带STM传播到半周期时间（使用修正后的当前状态）
            final_state = None
            try:
                result = self.dynamics.propagate(
                    current_state,
                    (0, current_time),
                    t_eval=np.linspace(0, current_time, 1000),
                    with_stm=True,
                    with_jacobi=False,
                )

                final_state = result["states"][-1]
                final_stm = result["stm"][-1]

                self.performance_stats["stm_evaluations"] += 1

            except Exception as e:
                if verbose:
                    logger.info("  积分失败: %s", e)
                self.termination_reason = f"积分失败: {e}"
                break

            # 2. 计算约束残差
            constraint = np.array([final_state[idx] for idx in self.constraint_indices])
            target = np.array(
                [
                    self.target_conditions[_STATE_INDEX_TO_KEY[idx]]
                    for idx in self.constraint_indices
                ]
            )
            error_vector = constraint - target
            current_error = np.linalg.norm(error_vector)

            self.error_history.append(current_error)

            self.convergence_history.append(
                {
                    "iteration": iteration + 1,
                    "error": current_error,
                    "state": current_state.copy(),
                    "time": current_time,
                    "final_state": final_state.copy(),
                }
            )

            if verbose:
                logger.info("迭代 %d: 约束残差范数 = %.2e", iteration + 1, current_error)

            if current_error < self.tolerance:
                self.converged = True
                self.termination_reason = "收敛成功：误差小于容差"
                self.current_error = current_error  # 保存误差值
                if verbose:
                    logger.info("[OK] 收敛成功！最终误差: %.2e", current_error)
                if callback:
                    callback(iteration + 1, current_error, True)
                break

            # 4. 检查发散
            if current_error > self.divergence_limit:
                self.termination_reason = "发散：误差超过限制"
                if verbose:
                    logger.warning("[WARN] 迭代发散，误差 = %.2e", current_error)
                if callback:
                    callback(iteration + 1, current_error, False)
                break

            # 5. 构建雅可比矩阵（基于STM和终点状态导数）
            # 计算终点处的状态导数（速度和加速度）
            state_derivative = self.dynamics.equations_of_motion(current_time, final_state)

            n_constraints = len(self.constraint_indices)
            n_variables = len(self.free_variable_indices)
            self.jacobian_matrix = np.zeros((n_constraints, n_variables))

            for j, var_idx in enumerate(self.free_variable_indices):
                if var_idx < 6:  # 状态变量：从STM获取偏导 ∂(终点状态)/∂(初始状态)
                    for i, c_idx in enumerate(self.constraint_indices):
                        self.jacobian_matrix[i, j] = final_stm[c_idx, var_idx]
                elif var_idx == 6:  # 时间变量：从状态导数获取 ∂(终点状态)/∂t
                    for i, c_idx in enumerate(self.constraint_indices):
                        self.jacobian_matrix[i, j] = state_derivative[c_idx]

            self.performance_stats["jacobian_evaluations"] += 1

            # 6. 求解牛顿修正量: J * delta = F => delta = J^{-1} * F
            try:
                if n_constraints == n_variables:
                    delta = np.linalg.solve(self.jacobian_matrix, error_vector)
                else:
                    delta = np.linalg.lstsq(self.jacobian_matrix, error_vector, rcond=None)[0]
            except np.linalg.LinAlgError:
                if verbose:
                    logger.warning("  雅可比矩阵奇异，无法求解修正量。")
                self.termination_reason = "雅可比矩阵奇异"
                if callback:
                    callback(iteration + 1, current_error, False)
                break

            correction_norm = np.linalg.norm(delta)
            self.correction_history.append(correction_norm)

            # 7. 牛顿更新: X_new = X_old - J^{-1} * F
            for j, var_idx in enumerate(self.free_variable_indices):
                if var_idx < 6:  # 更新状态变量
                    current_state[var_idx] -= delta[j]
                elif var_idx == 6:  # 更新时间变量
                    current_time -= delta[j]

            # Halo/Axial：仅当 T/2 已崩溃到极小（寄生根）时拉回，
            # 避免一律钳位导致无法收敛到真实 ~0.92 TU
            if self.setup_type in (
                "halo_orbit_fixed_x0",
                "halo_orbit_fixed_z0",
                "axial_orbit_fixed_vz0",
            ):
                if current_time < 0.02:
                    current_time = 0.25
            elif current_time <= 0:
                current_time = 1e-6
                if verbose:
                    logger.warning("  时间调整为正值")

            if verbose:
                logger.info("  修正量范数: %.2e", correction_norm)
                logger.info("  新状态: x=%.6f, y_dot=%.6f", current_state[0], current_state[4])
                logger.info("  新半周期: T/2=%.6f", current_time)

            # 检查停滞（仅在未收敛的情况下检查）
            if not self.converged and correction_norm < self.stagnation_limit:
                # 修正量过小时，若误差已足够小，也视为收敛成功
                if current_error < 1e-8:
                    self.converged = True
                    self.termination_reason = "收敛成功：修正量过小但误差足够小"
                    self.current_error = current_error
                    if verbose:
                        logger.info(
                            "  收敛成功：修正量过小(%.2e)但误差已足够小(%.2e)",
                            correction_norm,
                            current_error,
                        )
                    if callback:
                        callback(iteration + 1, current_error, True)
                    break
                else:
                    self.termination_reason = "停滞：修正量过小"
                    if verbose:
                        logger.info("  停滞：修正量 = %.2e", correction_norm)
                    if callback:
                        callback(iteration + 1, current_error, False)
                    break

            if callback:
                callback(iteration + 1, current_error, False)

        if self.converged:
            # 验证周期合理性（防止收敛到无效解如周期接近0的轨道）
            if self.setup_type in (
                "halo_orbit_fixed_z0",
                "halo_orbit_fixed_x0",
                "axial_orbit_fixed_vz0",
            ):
                min_valid_period = 0.5
            else:
                min_valid_period = 1e-6
            if 2 * current_time < min_valid_period:
                self.converged = False
                self.success = False
                self.termination_reason = (
                    f"收敛但周期无效: T={2 * current_time:.6e} < {min_valid_period}"
                )
                if verbose:
                    logger.info("微分修正失败: %s", self.termination_reason)
                return None

            self.success = True
            self.final_solution = current_state.copy()
            self.solution_time = current_time

            if verbose:
                logger.info("=" * 60)
                logger.info("微分修正成功完成")
                logger.info("=" * 60)
                logger.info("  最终周期: T = %.6f", 2 * current_time)
                logger.info("  最终误差: %.2e", current_error)
                logger.info("  迭代次数: %d", self.iteration_count)
                logger.info("=" * 60)

            result_dict = self._build_result(current_state, current_time)
            return self._create_corrected_orbit(result_dict)
        else:
            if verbose:
                logger.info("微分修正失败: %s", self.termination_reason)

    def iterate_full_period_correction(self, initial_guess, verbose=False, callback=None):
        """全周期闭合修正（适用于无对称性的周期轨道，如 SPO）。

        与 iterate_correction 的区别：
        - iterate_correction：传播 T/2，利用对称性检查半周期约束
        - 本方法：传播完整周期 T，检查 state(T) - state(0) = 0

        约束语义：target_conditions 的值为 0（闭合残差为零），
        实际残差 = final_state[c_idx] - initial_state[c_idx]。

        Args:
            initial_guess: 初始猜测轨道（states[0] 为初始状态，period 为周期）
            verbose: 是否打印迭代过程
            callback: 每次迭代后的回调函数

        Returns:
            Orbit | None: 修正后的周期轨道；失败返回 None
        """
        self.initial_guess = initial_guess.states[0].copy()
        self.iteration_count = 0
        self.converged = False
        self.success = False

        current_state = self.initial_guess.copy()
        current_period = initial_guess.period  # 全周期（非半周期）

        if verbose:
            logger.info("=" * 60)
            logger.info("开始全周期闭合修正迭代...")
            logger.info(
                "初始状态: x=%.6f, y=%.6f, z=%.6f",
                current_state[0],
                current_state[1],
                current_state[2],
            )
            logger.info(
                "         ẋ=%.6f, ẏ=%.6f, ż=%.6f",
                current_state[3],
                current_state[4],
                current_state[5],
            )
            logger.info("初始周期: T=%.6f", current_period)
            logger.info("=" * 60)

        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1

            # 1. 带STM传播完整周期
            try:
                result = self.dynamics.propagate(
                    current_state,
                    (0, current_period),
                    t_eval=np.linspace(0, current_period, 1000),
                    with_stm=True,
                    with_jacobi=False,
                )
                final_state = result["states"][-1]
                final_stm = result["stm"][-1]
                self.performance_stats["stm_evaluations"] += 1
            except Exception as e:
                if verbose:
                    logger.info("  积分失败: %s", e)
                self.termination_reason = f"积分失败: {e}"
                break

            # 2. 计算闭合残差：state(T) - state(0) 在约束分量上
            error_vector = np.array(
                [final_state[idx] - current_state[idx] for idx in self.constraint_indices]
            )
            current_error = np.linalg.norm(error_vector)

            self.error_history.append(current_error)
            self.convergence_history.append(
                {
                    "iteration": iteration + 1,
                    "error": current_error,
                    "state": current_state.copy(),
                    "time": current_period,
                    "final_state": final_state.copy(),
                }
            )

            if verbose:
                logger.info("迭代 %d: 闭合残差范数 = %.2e", iteration + 1, current_error)

            if current_error < self.tolerance:
                self.converged = True
                self.termination_reason = "收敛成功：闭合残差小于容差"
                self.current_error = current_error
                if verbose:
                    logger.info("[OK] 收敛成功！最终误差: %.2e", current_error)
                if callback:
                    callback(iteration + 1, current_error, True)
                break

            if current_error > self.divergence_limit:
                self.termination_reason = "发散：误差超过限制"
                if callback:
                    callback(iteration + 1, current_error, False)
                break

            # 3. 构建雅可比矩阵
            # 对于闭合约束：∂(final_state[i] - initial_state[i]) / ∂free_var[j]
            #   状态变量 (var_idx < 6): ∂final/∂var = STM[c_idx, var_idx],
            #     若 var_idx == c_idx 则还需 -1（∂(-initial)/∂var = -δ）
            #   时间变量 (var_idx == 6): ∂final/∂T = dstate/dt(T)
            state_derivative = self.dynamics.equations_of_motion(current_period, final_state)

            n_constraints = len(self.constraint_indices)
            n_variables = len(self.free_variable_indices)
            jacobian = np.zeros((n_constraints, n_variables))

            for j, var_idx in enumerate(self.free_variable_indices):
                if var_idx < 6:
                    for i, c_idx in enumerate(self.constraint_indices):
                        jacobian[i, j] = final_stm[c_idx, var_idx]
                        if var_idx == c_idx:
                            jacobian[i, j] -= 1.0
                elif var_idx == 6:
                    for i, c_idx in enumerate(self.constraint_indices):
                        jacobian[i, j] = state_derivative[c_idx]

            self.performance_stats["jacobian_evaluations"] += 1

            # 4. 求解牛顿修正量（方阵用 solve，欠定用最小二乘）
            try:
                if n_constraints == n_variables:
                    delta = np.linalg.solve(jacobian, error_vector)
                else:
                    delta = np.linalg.lstsq(jacobian, error_vector, rcond=None)[0]
            except np.linalg.LinAlgError:
                if verbose:
                    logger.warning("  雅可比矩阵奇异，无法求解修正量。")
                self.termination_reason = "雅可比矩阵奇异"
                if callback:
                    callback(iteration + 1, current_error, False)
                break

            correction_norm = np.linalg.norm(delta)
            self.correction_history.append(correction_norm)

            # 5. 更新自由变量
            for j, var_idx in enumerate(self.free_variable_indices):
                if var_idx < 6:
                    current_state[var_idx] -= delta[j]
                elif var_idx == 6:
                    current_period -= delta[j]

            if current_period <= 0:
                current_period = 0.1

            if verbose:
                logger.info("  修正量范数: %.2e", correction_norm)
                logger.info("  新周期: T=%.6f", current_period)

            # 6. 停滞检查
            if correction_norm < self.stagnation_limit:
                if current_error < 1e-8:
                    self.converged = True
                    self.termination_reason = "收敛成功：修正量过小但误差足够小"
                    self.current_error = current_error
                    if callback:
                        callback(iteration + 1, current_error, True)
                    break
                self.termination_reason = "停滞：修正量过小"
                if callback:
                    callback(iteration + 1, current_error, False)
                break

            if callback:
                callback(iteration + 1, current_error, False)

        if self.converged:
            if current_period < 1e-6:
                self.converged = False
                self.success = False
                self.termination_reason = f"收敛但周期无效: T={current_period:.6e}"
                return None

            self.success = True
            self.final_solution = current_state.copy()
            self.solution_time = current_period

            if verbose:
                logger.info("=" * 60)
                logger.info("全周期修正成功完成")
                logger.info("  最终周期: T = %.6f", current_period)
                logger.info("  最终误差: %.2e", current_error)
                logger.info("  迭代次数: %d", self.iteration_count)
                logger.info("=" * 60)

            # 全周期模式：period 就是 current_period（非 2*half_period）
            result_dict = {
                "state": current_state,
                "period": current_period,
                "half_period": current_period / 2,
                "setup_type": self.setup_type,
                "converged": self.converged,
                "error": self.current_error,
            }
            return self._create_corrected_orbit(result_dict)
        else:
            if verbose:
                logger.info("微分修正失败: %s", self.termination_reason)

    def _build_result(self, final_state, half_period):
        """构建修正结果字典

        Args:
            final_state: 修正后的最终状态向量
            half_period: 修正后的半周期时间

        Returns:
            dict: 包含状态、周期等信息的字典
        """
        return {
            "state": final_state,
            "period": 2 * half_period,
            "half_period": half_period,
            "setup_type": self.setup_type,
            "converged": self.converged,
            "error": self.current_error if hasattr(self, "current_error") else None,
        }

    def _create_corrected_orbit(self, result):
        """根据修正结果积分生成完整周期轨道。

        Args:
            result: 包含修正结果的字典，包含 state (半周期对称点状态) 和 period (完整周期)

        Returns:
            Orbit: 完整的周期轨道对象
        """
        full_period = result["period"]
        initial_state = result["state"]

        prop_result = self.dynamics.propagate(
            initial_state,
            (0, full_period),
            t_eval=np.linspace(0, full_period, 1000),
        )

        final_state = prop_result["states"][-1]
        closure_error = np.linalg.norm(final_state - initial_state)

        # 对于 Halo/Axial/SPO 轨道，周期对称性或全周期闭合已由微分修正精确保证；
        # 全周期闭合误差通常来自积分截断（~2e-6），用速度调整反而破坏修正结果。
        if closure_error > 1e-10 and self.setup_type not in (
            "halo_orbit_fixed_x0",
            "halo_orbit_fixed_z0",
            "axial_orbit_fixed_vz0",
            "spo_fixed_x0",
        ):
            closure_error_vector = final_state - initial_state
            pos_error = np.linalg.norm(closure_error_vector[:3])
            vel_error = np.linalg.norm(closure_error_vector[3:])

            if pos_error > 1e-14 and vel_error > 1e-14:
                # 调整初始速度来修正位置误差
                adjustment = -0.5 * closure_error_vector[3:]
                new_state = initial_state.copy()
                new_state[4] += adjustment[1]

                # 重新积分
                prop_result = self.dynamics.propagate(
                    new_state,
                    (0, full_period),
                    t_eval=np.linspace(0, full_period, 1000),
                )
                final_state = prop_result["states"][-1]
                new_closure_error = np.linalg.norm(final_state - new_state)

                if new_closure_error < closure_error:
                    initial_state = new_state
                    closure_error = new_closure_error

        # 确保 states 是独立的副本
        orbit_states = np.array(prop_result["states"], copy=True)

        orbit = Orbit(
            states=orbit_states,
            times=prop_result["time"].copy(),
            system=self.dynamics.system,
        )
        orbit.period = full_period
        orbit.is_periodic = bool(closure_error < 1e-8)
        orbit.family_type = self._infer_family_type()

        # 保存修正结果信息到 Orbit 对象
        orbit.correction_success = self.success
        orbit.correction_iterations = self.iteration_count
        orbit.correction_error = result.get("error")
        orbit.correction_termination_reason = self.termination_reason
        orbit.closure_error = float(closure_error)

        return orbit

    def _infer_family_type(self):
        """根据配置推断轨道族类型"""
        if self.setup_type and "axial" in self.setup_type:
            return "axial"
        elif self.setup_type and ("3D" in self.setup_type or "halo" in self.setup_type):
            return "halo"
        elif self.setup_type and "2D" in self.setup_type:
            return "lyapunov"
        return None

    def check_convergence(self):
        """检查收敛性

        Returns:
            bool: 是否收敛
        """
        return self.converged

    def get_convergence_history(self):
        """获取收敛历史

        Returns:
            dict: 收敛历史数据
        """
        return {
            "errors": self.error_history,
            "corrections": self.correction_history,
            "iterations": self.iteration_count,
            "converged": self.converged,
            "termination_reason": self.termination_reason,
        }

    def __str__(self):
        return f"DifferentialCorrection(setup={self.setup_type}, converged={self.converged})"

    def __repr__(self):
        return (
            f"DifferentialCorrection(dynamic={self.dynamics}, "
            f"setup={self.setup_type}, tol={self.tolerance})"
        )
