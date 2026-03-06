"""
微分修正算法模块

提供用于求解周期轨道的微分修正算法，支持多种对称性配置。
"""

import numpy as np
from scipy import integrate


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

    def __init__(self, dynamics, target=None, free_vars=None):
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

    def setup_2D_symmetric_x_fixed_x0(self, x0):
        """配置平面问题中固定初始x坐标的对称周期轨道搜索

        利用CR3BP关于x轴的对称性，搜索从x轴垂直出发的周期轨道。
        固定初始x坐标x0，调整y_dot0和T/2满足终点垂直穿越x轴条件。

        参数:
            x0 (float): 固定的初始x坐标

        返回:
            self: 配置好的微分修正器实例
        """
        self.setup_type = "2D_symmetric_x_fixed_x0"
        self.symmetry_condition = "x_axis"
        self.fixed_parameters = {"x0": x0}

        self.free_variables = ["y_dot0", "T_half"]
        self.free_variable_indices = [4, 6]  # y_dot索引4, 时间作为额外变量索引6

        self.target_conditions = {"y": 0.0, "x_dot": 0.0}
        self.constraint_indices = [1, 3]  # y和x_dot在状态向量中的索引
        self.constraint_weights = {"y": 1.0, "x_dot": 1.0}
        self.constraint_types = {"y": "equality", "x_dot": "equality"}

        self._reset_history()
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

    def iterate_correction(self, initial_state, t_half, verbose=True):
        """迭代修正主算法

        参数:
            initial_state (np.ndarray): 初始状态向量 [x, y, z, vx, vy, vz]
            t_half (float): 初始半周期估计
            verbose (bool): 是否打印迭代信息

        返回:
            dict: 包含修正结果的字典
                - 'state': 修正后的初始状态
                - 'period': 修正后的完整周期
                - 'success': 是否收敛
                - 'iterations': 迭代次数
                - 'error': 最终误差
                - 'history': 收敛历史
        """
        self._reset_history()
        current_state = initial_state.copy()
        current_time = t_half

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"开始微分修正迭代 (配置: {self.setup_type})")
            print(f"{'=' * 60}")

        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1

            # 1. 积分到半周期
            try:
                result = integrate.solve_ivp(
                    self.dynamics.equations_of_motion,
                    (0, current_time),
                    current_state,
                    method="DOP853",
                    t_eval=np.linspace(0, current_time, 1000),
                    rtol=1e-12,
                    atol=1e-12,
                )
                if not result.success:
                    self.termination_reason = f"积分失败: {result.message}"
                    if verbose:
                        print(f"  积分失败: {result.message}")
                    return self._build_result(current_state, current_time)

                final_state = result.y[:, -1]
                self.performance_stats["stm_evaluations"] += 1

            except Exception as e:
                self.termination_reason = f"积分异常: {e}"
                return self._build_result(current_state, current_time)

            # 2. 计算误差
            error_vector = self._compute_error_vector(final_state)
            current_error = np.linalg.norm(error_vector)

            # 保存历史
            self.error_history.append(current_error)
            self.convergence_history.append(
                {
                    "iteration": iteration,
                    "error": current_error,
                    "state": current_state.copy(),
                    "time": current_time,
                    "final_state": final_state.copy(),
                }
            )

            if verbose:
                print(f"  迭代 {iteration + 1}: 误差 = {current_error:.4e}")

            # 3. 检查收敛
            if current_error < self.tolerance:
                self.converged = True
                self.success = True
                self.termination_reason = "收敛成功"
                if verbose:
                    print(f"\n✓ 收敛成功！最终误差: {current_error:.2e}")
                break

            # 4. 检查发散
            if current_error > self.divergence_limit:
                self.termination_reason = "发散"
                if verbose:
                    print(f"\n✗ 迭代发散，误差 = {current_error:.2e}")
                break

            # 5. 计算雅可比矩阵
            self.jacobian_matrix = self._compute_jacobian_finite_diff(current_state, current_time)

            # 6. 计算修正量
            try:
                self.pseudoinverse_matrix = np.linalg.pinv(self.jacobian_matrix)
                correction = -self.pseudoinverse_matrix @ error_vector
            except np.linalg.LinAlgError:
                correction = -np.linalg.lstsq(self.jacobian_matrix, error_vector, rcond=None)[0]

            # 应用阻尼
            correction *= self.damping_factor

            # 限制步长
            correction_norm = np.linalg.norm(correction)
            if correction_norm > self.step_size_limit:
                correction *= self.step_size_limit / correction_norm

            self.correction_history.append(correction_norm)

            # 7. 更新自由变量
            for j, var_idx in enumerate(self.free_variable_indices):
                if var_idx < 6:
                    current_state[var_idx] += correction[j]
                elif var_idx == 6:
                    current_time += correction[j]

            # 确保时间为正
            if current_time <= 0:
                current_time = abs(current_time) if abs(current_time) > 1e-6 else 1e-6

            # 检查停滞
            if correction_norm < self.stagnation_limit:
                self.termination_reason = "停滞"
                break

        # 存储最终解
        self.final_solution = current_state.copy()
        self.solution_time = current_time

        return self._build_result(current_state, current_time)

    def _build_result(self, state, t_half):
        """构建结果字典"""
        return {
            "state": state.copy(),
            "period": 2 * t_half,
            "t_half": t_half,
            "success": self.success,
            "iterations": self.iteration_count,
            "error": self.error_history[-1] if self.error_history else float("inf"),
            "history": self.convergence_history,
            "termination_reason": self.termination_reason,
        }

    def correct_orbit(self, initial_state, t_half, verbose=True):
        """修正轨道并返回完整周期轨道

        参数:
            initial_state (np.ndarray): 初始状态向量
            t_half (float): 初始半周期估计
            verbose (bool): 是否打印信息

        返回:
            tuple: (Orbit对象或None, 修正结果字典)
        """
        from ..core.orbit import Orbit

        result = self.iterate_correction(initial_state, t_half, verbose)

        if result["success"]:
            # 积分完整周期获得周期轨道
            full_period = result["period"]
            propagation = integrate.solve_ivp(
                self.dynamics.equations_of_motion,
                (0, full_period),
                result["state"],
                method="DOP853",
                t_eval=np.linspace(0, full_period, 2000),
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

            return orbit, result
        else:
            return None, result

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
