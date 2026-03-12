"""
微分修正算法模块

提供用于求解周期轨道的微分修正算法，支持多种对称性配置。
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy.typing as npt

from ..core.orbit import Orbit
from ..core.dynamics import CR3BP_Dynamics


class DifferentialCorrection:
    """微分修正算法

    通过迭代修正初始条件，使轨道满足指定的约束条件（如周期性、对称性等）。

    支持的对称性配置：
    - 2D对称X固定X0: 平面对称周期轨道，固定初始x坐标
    - 2D对称X固定T: 平面对称周期轨道，固定轨道周期
    - 3D对称X固定X0: 空间对称周期轨道（Halo轨道等）
    - 3D对称XZ固定X0: 空间XZ对称周期轨道
    - 3D对称XZ固定Z0: 空间XZ对称周期轨道，固定Z0

    属性：
        dynamics: CR3BP_Dynamics对象
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
        "3D_symmetric_x_fixed_x0",
        "3D_symmetric_xz_fixed_x0",
        "3D_symmetric_xz_fixed_z0",
    ]

    def __init__(
        self,
        dynamics: CR3BP_Dynamics,
        target: Optional[Dict[str, Any]] = None,
        free_vars: Optional[List[str]] = None,
    ) -> None:
        """初始化修正器

        参数：
        - dynamics: CR3BP_Dynamics对象
        - target: 目标约束条件字典（可选）
        - free_vars: 自由变量列表（可选）
        """
        # 核心对象
        self.dynamics = dynamics
        self.target_conditions = target or {}
        self.free_variables = free_vars or []

        # 收敛控制参数
        self.tolerance = self.DEFAULT_TOLERANCE
        self.max_iterations = self.DEFAULT_MAX_ITERATIONS
        self.damping_factor = self.DEFAULT_DAMPING_FACTOR
        self.use_adaptive_damping = True
        self.min_damping = 0.1
        self.max_damping = 2.0

        # 收敛历史记录
        self.convergence_history = []
        self.error_history = []
        self.correction_history = []
        self.iteration_count = 0
        self.converged = False

        # 当前状态
        self.current_state = None
        self.current_time = None
        self.current_constraints = None
        self.current_error = None

        # 解
        self.initial_guess = None
        self.final_solution = None
        self.solution_time = None

        # 矩阵
        self.jacobian_matrix = None
        self.correction_matrix = None
        self.pseudoinverse_matrix = None

        # 约束设置
        self.constraint_indices = []
        self.constraint_weights = {}
        self.constraint_types = {}
        self.free_variable_indices = []

        # 配置类型
        self.setup_type = None
        self.symmetry_condition = None
        self.fixed_parameters = {}

        # 数值微分设置
        self.use_analytic_stm = True
        self.finite_difference_step = 1e-7
        self.finite_difference_method = "central"

        # 迭代控制
        self.stagnation_limit = 1e-14
        self.divergence_limit = 1e10
        self.step_size_limit = 1.0

        # 性能统计
        self.performance_stats = {
            "total_time": 0.0,
            "stm_evaluations": 0,
            "constraint_evaluations": 0,
            "jacobian_evaluations": 0,
        }

        # 终止条件
        self.termination_reason = None
        self.success = False

    def setup_2D_symmetric_x_fixed_x0(self, x0=0.0):
        """配置平面问题中固定初始x坐标的对称周期轨道搜索

        在平面圆形限制性三体问题（PCRTBP）模型中，动力学方程关于会合坐标系的x轴具有对称性。
        利用这一性质，周期轨道的搜索可以简化为寻找合适的初始条件：
        从x轴上一点垂直出发（y=0, x_dot=0），经过半周期T/2后再次垂直穿越x轴（y=0, x_dot=0）。

        本函数针对这种对称性设置微分修正问题，固定初始x坐标x0，将初始y方向速度y_dot0
        和半周期T/2作为自由变量进行调整，以满足终点处的垂直穿越条件。

        参数:
            x0 (float): 固定的初始x坐标，轨道从点(x0, 0)垂直出发，这里将值设置为0.0，是因为在这个函数中只需要使用x0进行初始化

        返回:
            self: 返回配置好的微分修正器实例

        配置说明:
            - 自由变量: [y_dot0, T_half] - 初始y方向速度和半周期时间
            - 目标约束: [y(T/2)=0, x_dot(T/2)=0] - 终点处再次垂直穿越x轴
            - 状态向量索引: [1, 3] 分别对应y坐标和x方向速度

        应用场景:
            此配置对应于Broucke(1968)等经典文献中寻找对称周期轨道的基本方法，
            可用于生成围绕平动点或主天体的各类周期轨道家族。

        参考文献：
            [1] Broucke R A. Periodic orbits in the restricted three body problem with Earth-moon masses[R]. 1968.
        """
        # 设置配置类型
        self.setup_type = "2D_symmetric_x_fixed_x0"
        self.symmetry_condition = "x_axis"
        self.fixed_parameters = {"x0": x0}

        # 定义自由变量
        # 在2D对称x轴的情况下，从x轴垂直出发的初始条件为: [x0, 0, 0, y_dot]
        # 自由变量是初始y方向速度 y_dot 和飞行时间 T/2
        self.free_variables = ["y_dot0", "T_half"]
        self.free_variable_indices = [
            4,
            6,
        ]  # 状态向量中索引4是y_dot，索引6表示时间（作为变量）

        # 定义目标约束条件
        # 对于对称x轴的周期轨道，在半周期处应满足：y(T/2)=0, x_dot(T/2)=0
        # 即轨道再次垂直穿越x轴
        self.target_conditions = {
            "y": 0.0,  # 终点y坐标为0
            "x_dot": 0.0,  # 终点x方向速度为0
        }

        # 设置约束索引
        # 状态向量为 [x, y, z, x_dot, y_dot, z_dot]
        self.constraint_indices = [1, 3]  # y和x_dot在状态向量中的索引

        # 设置约束权重（可选，用于加权最小二乘）
        self.constraint_weights = {"y": 1.0, "x_dot": 1.0}

        # 设置约束类型
        self.constraint_types = {"y": "equality", "x_dot": "equality"}

        # 更新固定参数到目标条件中（可选）
        self.fixed_parameters.update({"x0": x0})

        # 重置收敛历史
        self._reset_history()

        print(
            f"2D对称x轴配置完成：固定x0={x0}，自由变量={self.free_variables}，目标约束={list(self.target_conditions.keys())}"
        )

        return self

    def setup_2D_symmetric_x_fixed_t(self, t_half):
        """配置平面问题中固定半周期的对称周期轨道搜索

        固定半周期T/2，调整初始条件x0和y_dot0满足约束。

        参数:
            t_half (float): 固定的半周期

        返回:
            self: 配置好的微分修正器实例
        """
        self.setup_type = "2D_symmetric_x_fixed_t"
        self.symmetry_condition = "x_axis"
        self.fixed_parameters = {"T_half": t_half}

        self.free_variables = ["x0", "y_dot0"]
        self.free_variable_indices = [0, 4]  # x0索引0, y_dot索引4

        self.target_conditions = {"y": 0.0, "x_dot": 0.0}
        self.constraint_indices = [1, 3]
        self.constraint_weights = {"y": 1.0, "x_dot": 1.0}
        self.constraint_types = {"y": "equality", "x_dot": "equality"}

        self._reset_history()
        return self

    def setup_3D_symmetric_x_fixed_x0(self, x0):
        """配置空间问题中固定初始x坐标的对称周期轨道搜索（如Halo轨道）

        参数:
            x0 (float): 固定的初始x坐标

        返回:
            self: 配置好的微分修正器实例
        """
        self.setup_type = "3D_symmetric_x_fixed_x0"
        self.symmetry_condition = "x_axis"
        self.fixed_parameters = {"x0": x0}

        self.free_variables = ["z0", "y_dot0", "T_half"]
        self.free_variable_indices = [2, 4, 6]

        self.target_conditions = {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}
        self.constraint_indices = [1, 3, 5]
        self.constraint_weights = {"y": 1.0, "x_dot": 1.0, "z_dot": 1.0}
        self.constraint_types = {"y": "equality", "x_dot": "equality", "z_dot": "equality"}

        self._reset_history()
        return self

    def setup_3D_symmetric_xz_fixed_x0(self, x0):
        """配置空间XZ对称周期轨道搜索，固定X0

        参数:
            x0 (float): 固定的初始x坐标

        返回:
            self: 配置好的微分修正器实例
        """
        self.setup_type = "3D_symmetric_xz_fixed_x0"
        self.symmetry_condition = "xz_plane"
        self.fixed_parameters = {"x0": x0}

        self.free_variables = ["z0", "y_dot0", "T_half"]
        self.free_variable_indices = [2, 4, 6]

        self.target_conditions = {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}
        self.constraint_indices = [1, 3, 5]

        self._reset_history()
        return self

    def setup_3D_symmetric_xz_fixed_z0(self, z0):
        """配置空间XZ对称周期轨道搜索，固定Z0

        参数:
            z0 (float): 固定的初始z坐标

        返回:
            self: 配置好的微分修正器实例
        """
        self.setup_type = "3D_symmetric_xz_fixed_z0"
        self.symmetry_condition = "xz_plane"
        self.fixed_parameters = {"z0": z0}

        self.free_variables = ["x0", "y_dot0", "T_half"]
        self.free_variable_indices = [0, 4, 6]

        self.target_conditions = {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}
        self.constraint_indices = [1, 3, 5]

        self._reset_history()
        return self

    def _reset_history(self):
        """重置收敛历史"""
        self.convergence_history = []
        self.error_history = []
        self.correction_history = []
        self.iteration_count = 0
        self.converged = False
        self.termination_reason = None
        self.success = False

    def _compute_error_vector(self, final_state):
        """计算约束误差向量

        参数：
            final_state: 终点状态向量

        返回：
            error_vector: 误差向量
        """
        constraints = np.array([final_state[idx] for idx in self.constraint_indices])
        targets = np.zeros(len(self.constraint_indices))

        # 从target_conditions获取目标值
        keys = list(self.target_conditions.keys())
        for i, key in enumerate(keys):
            targets[i] = self.target_conditions[key]

        return constraints - targets

    def _compute_jacobian_finite_diff(self, current_state, current_time):
        """使用有限差分法计算雅可比矩阵

        参数：
            current_state: 当前初始状态
            current_time: 当前半周期时间

        返回：
            jacobian: 雅可比矩阵
        """
        n_constraints = len(self.constraint_indices)
        n_variables = len(self.free_variable_indices)
        jacobian = np.zeros((n_constraints, n_variables))
        eps = self.finite_difference_step

        for j, var_idx in enumerate(self.free_variable_indices):
            if var_idx < 6:  # 对初始状态的敏感性
                # 正向扰动
                state_fwd = current_state.copy()
                state_fwd[var_idx] += eps
                result_fwd = integrate.solve_ivp(
                    self.dynamics.equations_of_motion,
                    (0, current_time),
                    state_fwd,
                    method="DOP853",
                    t_eval=[current_time],
                    rtol=1e-12,
                    atol=1e-12,
                )
                final_fwd = result_fwd.y[:, -1]

                # 负向扰动
                state_bwd = current_state.copy()
                state_bwd[var_idx] -= eps
                result_bwd = integrate.solve_ivp(
                    self.dynamics.equations_of_motion,
                    (0, current_time),
                    state_bwd,
                    method="DOP853",
                    t_eval=[current_time],
                    rtol=1e-12,
                    atol=1e-12,
                )
                final_bwd = result_bwd.y[:, -1]

                # 中心差分
                sensitivity = (final_fwd - final_bwd) / (2 * eps)
                for i, c_idx in enumerate(self.constraint_indices):
                    jacobian[i, j] = sensitivity[c_idx]

            elif var_idx == 6:  # 对时间的敏感性
                # 正向扰动
                t_fwd = current_time + eps
                result_fwd = integrate.solve_ivp(
                    self.dynamics.equations_of_motion,
                    (0, t_fwd),
                    current_state,
                    method="DOP853",
                    t_eval=[t_fwd],
                    rtol=1e-12,
                    atol=1e-12,
                )
                final_fwd = result_fwd.y[:, -1]

                # 负向扰动
                t_bwd = current_time - eps
                result_bwd = integrate.solve_ivp(
                    self.dynamics.equations_of_motion,
                    (0, t_bwd),
                    current_state,
                    method="DOP853",
                    t_eval=[t_bwd],
                    rtol=1e-12,
                    atol=1e-12,
                )
                final_bwd = result_bwd.y[:, -1]

                # 中心差分
                sensitivity = (final_fwd - final_bwd) / (2 * eps)
                for i, c_idx in enumerate(self.constraint_indices):
                    jacobian[i, j] = sensitivity[c_idx]

        self.performance_stats["jacobian_evaluations"] += 1
        return jacobian

    def iterate_correction(self, initial_guess, verbose=False):
        """迭代修正主算法（基于STM的牛顿法）

        通过状态转移矩阵(STM)构建雅可比矩阵，使用牛顿迭代法修正自由变量，
        使终点状态满足目标约束条件，从而找到精确的周期轨道。

        参数:
            initial_guess (Orbit):
                初始猜测轨道，或初始状态向量
            verbose (bool):
                是否打印迭代过程信息

        返回:
            - 返回修正后的 Orbit 对象
        """
        # 状态索引到目标条件键的映射
        _STATE_INDEX_TO_KEY = {
            0: "x",
            1: "y",
            2: "z",
            3: "x_dot",
            4: "y_dot",
            5: "z_dot",
        }

        # 保存初始猜测
        self.initial_guess = initial_guess.states[0]
        self.iteration_count = 0
        self.converged = False
        self.success = False
        half_period_time = initial_guess.period / 2

        # 初始化当前状态和时间（用于牛顿迭代）
        current_state = self.initial_guess.copy()
        current_time = half_period_time

        if verbose:
            print(f"\n{'=' * 60}")
            print("开始微分修正迭代（STM牛顿法）...")
            print(f"{'=' * 60}")
            print(
                f"初始状态: x={self.initial_guess[0]:.6f}, y={self.initial_guess[1]:.6f}, z={self.initial_guess[2]:.6f}"
            )
            print(
                f"         x_dot={self.initial_guess[3]:.6f}, y_dot={self.initial_guess[4]:.6f}, z_dot={self.initial_guess[5]:.6f}"
            )
            print(f"初始半周期: T/2={half_period_time:.6f}")
            print(f"{'=' * 60}")

        # 迭代循环
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1

            # 1. 带STM传播到半周期时间（使用修正后的当前状态）
            try:
                result = self.dynamics.propagate(
                    current_state,
                    (0, current_time),
                    t_eval=np.linspace(0, current_time, 1000),
                    with_stm=True,
                )

                final_state = result["states"][-1]
                final_stm = result["stm"][-1]

                self.performance_stats["stm_evaluations"] += 1

            except Exception as e:
                if verbose:
                    print(f"  积分失败: {e}")
                self.termination_reason = f"积分失败: {e}"

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

            # 记录误差历史
            self.error_history.append(current_error)

            # 记录收敛历史
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
                print(f"\n迭代 {iteration + 1}: 约束残差范数 = {current_error:.2e}")

            if current_error < self.tolerance:
                self.converged = True
                self.termination_reason = "收敛成功：误差小于容差"
                self.current_error = current_error  # 保存误差值
                if verbose:
                    print(f"[OK] 收敛成功！最终误差: {current_error:.2e}")
                break

            # 4. 检查发散
            if current_error > self.divergence_limit:
                self.termination_reason = "发散：误差超过限制"
                if verbose:
                    print(f"[WARN] 警告：迭代发散，误差 = {current_error:.2e}")
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
                    print("  雅可比矩阵奇异，无法求解修正量。")
                self.termination_reason = "雅可比矩阵奇异"
                break

            correction_norm = np.linalg.norm(delta)
            self.correction_history.append(correction_norm)

            # 7. 牛顿更新: X_new = X_old - J^{-1} * F
            for j, var_idx in enumerate(self.free_variable_indices):
                if var_idx < 6:  # 更新状态变量
                    current_state[var_idx] -= delta[j]
                elif var_idx == 6:  # 更新时间变量
                    current_time -= delta[j]

            # 确保时间正数
            if current_time <= 0:
                current_time = 1e-6
                if verbose:
                    print("  警告：时间调整为正值")

            if verbose:
                print(f"  修正量范数: {correction_norm:.2e}")
                print(f"  新状态: x={current_state[0]:.6f}, y_dot={current_state[4]:.6f}")
                print(f"  新半周期: T/2={current_time:.6f}")

            # 检查停滞
            if correction_norm < self.stagnation_limit:
                self.termination_reason = "停滞：修正量过小"
                if verbose:
                    print(f"  停滞：修正量 = {correction_norm:.2e}")
                break

        # 迭代结束，处理结果
        if self.converged:
            self.success = True
            self.final_solution = current_state.copy()
            self.solution_time = current_time

            if verbose:
                print(f"\n{'=' * 60}")
                print("微分修正成功完成")
                print(f"{'=' * 60}")
                print(f"  最终周期: T = {2 * current_time:.6f}")
                print(f"  最终误差: {current_error:.2e}")
                print(f"  迭代次数: {self.iteration_count}")
                print(f"{'=' * 60}")

            result_dict = self._build_result(current_state, current_time)
            return self._create_corrected_orbit(result_dict)
        else:
            if verbose:
                print(f"\n微分修正失败: {self.termination_reason}")

    def _build_result(self, final_state, half_period):
        """构建修正结果字典

        参数:
            final_state: 修正后的最终状态向量
            half_period: 修正后的半周期时间

        返回:
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
        """根据修正结果积分生成完整周期轨道。"""
        full_period = result["period"]
        propagation = integrate.solve_ivp(
            self.dynamics.equations_of_motion,
            (0, full_period),
            result["state"],
            method="DOP853",
            t_eval=np.linspace(0, full_period, 1000),
            rtol=1e-12,
            atol=1e-12,
        )

        orbit = Orbit(
            states=propagation.y.T,
            times=propagation.t,
            system=self.dynamics.system,
        )
        orbit.period = full_period
        orbit.is_periodic = True
        orbit.family_type = self._infer_family_type()

        # 保存修正结果信息到 Orbit 对象
        orbit.correction_success = self.success
        orbit.correction_iterations = self.iteration_count
        orbit.correction_error = result.get("error")
        orbit.correction_termination_reason = self.termination_reason

        return orbit

    def _infer_family_type(self):
        """根据配置推断轨道族类型"""
        if self.setup_type and "3D" in self.setup_type:
            return "halo"
        elif self.setup_type and "2D" in self.setup_type:
            return "lyapunov"
        return None

    def check_convergence(self):
        """检查收敛性

        返回:
            bool: 是否收敛
        """
        return self.converged

    def get_convergence_history(self):
        """获取收敛历史

        返回:
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
            f"DifferentialCorrection(dynamics={self.dynamics}, "
            f"setup={self.setup_type}, tol={self.tolerance})"
        )
