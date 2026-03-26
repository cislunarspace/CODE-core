"""
微分修正算法模块

提供用于求解周期轨道的微分修正算法，支持多种对称性配置。
包括Halo轨道、Lyapunov轨道等的解析近似和微分修正功能。
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from typing import Dict, List, Optional, Any, Tuple

import numpy.typing as npt

from ..core.orbit import Orbit
from ..core.dynamics import CR3BP_Dynamics


# =============================================================================
# Halo轨道解析近似函数（Richardson三阶近似）
# =============================================================================


def compute_halo_coefficients(mu: float, L: int) -> Dict[str, float]:
    """计算Halo轨道三阶近似的系数

    参数：
        mu: 质量比
        L: 拉格朗日点 (1=L1, 2=L2)

    返回：
        dict: 包含所有系数的字典

    参考：Richardson, D. L. (1980). Analytic construction of periodic orbits
          about the collinear points. Celestial Mechanics.
    """
    if L not in [1, 2]:
        raise ValueError(f"L必须是1或2，当前为{L}")

    gamma_dict = {1: 0.012149, 2: -0.012149}
    gamma = gamma_dict[L]

    c1 = 1.0 - mu - (1 - 2 * mu) * gamma**3 / (1 - gamma) ** 3
    c2 = 2 * mu * (1 - mu)
    c3 = 3 * mu * (2 - mu)

    if L == 1:
        a21 = 1.0 / (2 * gamma)
        a22 = (3 * gamma + 1) / (4 * gamma**2)
        a23 = -(3 * gamma + 1) / (8 * gamma**3)
        a24 = -(3 * gamma - 1) / (8 * gamma**3)
        a31 = 1.0 / (8 * gamma**2)

        b21 = (3 * gamma + 2) / (4 * gamma)
        b22 = (3 * gamma - 1) / (4 * gamma)
        b31 = 1.0 / (16 * gamma**2)

        d21 = (3 * gamma + 1) / (4 * gamma**2)
        d31 = (3 * gamma + 2) / (32 * gamma**3)
        d32 = (3 * gamma - 1) / (32 * gamma**3)

        k = 1.0
        delta = -1.0

    else:
        a21 = -1.0 / (2 * gamma)
        a22 = (3 * gamma - 1) / (4 * gamma**2)
        a23 = (3 * gamma - 1) / (8 * gamma**3)
        a24 = (3 * gamma + 1) / (8 * gamma**3)
        a31 = 1.0 / (8 * gamma**2)

        b21 = -(3 * gamma - 2) / (4 * gamma)
        b22 = -(3 * gamma + 1) / (4 * gamma)
        b31 = 1.0 / (16 * gamma**2)

        d21 = (3 * gamma - 1) / (4 * gamma**2)
        d31 = (3 * gamma - 2) / (32 * gamma**3)
        d32 = (3 * gamma + 1) / (32 * gamma**3)

        k = -1.0
        delta = 1.0

    l1 = -1.0 / (2 * gamma)
    l2 = (3 * gamma**2 + 3 * gamma + 1) / (4 * gamma**2)
    l3 = (3 * gamma**2 + 9 * gamma + 4) / (32 * gamma**3)

    kappa1 = (3 * gamma**2 + 3 * gamma + 1) / (4 * gamma**2)
    kappa2 = (3 * gamma**2 + 9 * gamma + 4) / (32 * gamma**3)

    return {
        "gamma": gamma,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "a21": a21,
        "a22": a22,
        "a23": a23,
        "a24": a24,
        "a31": a31,
        "b21": b21,
        "b22": b22,
        "b31": b31,
        "d21": d21,
        "d31": d31,
        "d32": d32,
        "k": k,
        "delta": delta,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "kappa1": kappa1,
        "kappa2": kappa2,
    }


def halo_third_order_approximation(
    mu: float,
    Au: float,
    Aw: float,
    phi: float,
    L: int,
    tf: float,
    N: int,
    halo_class: int = 0,
) -> Tuple[npt.NDArray, npt.NDArray, float]:
    """计算Halo轨道三阶解析近似

    参数：
        mu: 质量比
        Au: U方向振幅
        Aw: W方向振幅
        phi: 相位偏移
        L: 拉格朗日点 (1=L1, 2=L2)
        tf: 终止时间
        N: 点数
        halo_class: 0=Class I (北), 1=Class II (南)

    返回：
        SV_uvw: 状态向量序列 (N, 6)，[u, v, w, u_dot, v_dot, w_dot]
        t: 时间序列
        T: 周期

    参考：Richardson, D. L. (1980). Analytic construction of periodic orbits
          about the collinear points. Celestial Mechanics.
    """
    if L not in [1, 2]:
        raise ValueError(f"L必须是1或2，当前为{L}")
    if N < 2:
        raise ValueError(f"N必须大于等于2，当前为{N}")
    if tf <= 0:
        raise ValueError(f"tf必须为正数，当前为{tf}")
    if halo_class not in [0, 1]:
        raise ValueError(f"halo_class必须是0或1，当前为{halo_class}")

    coeffs = compute_halo_coefficients(mu, L)
    gamma = coeffs["gamma"]
    L_position = 1 - mu - gamma if L == 1 else 1 - mu + gamma

    a21 = coeffs["a21"]
    a22 = coeffs["a22"]
    a23 = coeffs["a23"]
    a24 = coeffs["a24"]
    a31 = coeffs["a31"]
    b21 = coeffs["b21"]
    b22 = coeffs["b22"]
    b31 = coeffs["b31"]
    d21 = coeffs["d21"]
    d31 = coeffs["d31"]
    d32 = coeffs["d32"]
    k = coeffs["k"]
    delta = coeffs["delta"]
    kappa1 = coeffs["kappa1"]
    kappa2 = coeffs["kappa2"]

    if halo_class == 1:
        delta = -delta
        phi = phi + np.pi

    T = 2 * np.pi * (1 + kappa1 * Au**2 + kappa2 * Aw**2)
    tau = np.linspace(0, 2 * np.pi, N)
    t = np.linspace(0, tf, N)

    u = (
        a21 * Au**2
        + a22 * Aw**2
        - Au * np.cos(tau + phi)
        + (a23 * Au**2 - a24 * Aw**2) * np.cos(2 * (tau + phi))
        + a31 * Au**3 * np.cos(3 * (tau + phi))
    )

    v = (
        k * Au * np.sin(tau + phi)
        + (b21 * Au**2 - b22 * Aw**2) * np.sin(2 * (tau + phi))
        + b31 * Au**3 * np.sin(3 * (tau + phi))
    )

    w = delta * (
        Aw * np.cos(tau + phi)
        + d21 * Au * Aw * (np.cos(2 * (tau + phi)) - 3)
        + (d32 * Aw * Au**2 - d31 * Aw**3) * np.cos(3 * (tau + phi))
    )

    u_dot = Au * np.sin(tau + phi) + 2 * (a23 * Au**2 - a24 * Aw**2) * np.sin(2 * (tau + phi))
    v_dot = k * Au * np.cos(tau + phi) + 2 * (b21 * Au**2 - b22 * Aw**2) * np.cos(2 * (tau + phi))
    w_dot = -Aw * np.sin(tau + phi) - 2 * d21 * Au * Aw * np.sin(2 * (tau + phi))

    x = L_position + u
    y = v
    z = w

    x_dot = u_dot
    y_dot = v_dot
    z_dot = w_dot

    SV_uvw = np.column_stack([x, y, z, x_dot, y_dot, z_dot])

    return SV_uvw, t, T


def compute_halo_initial_guess(
    mu: float,
    z_amplitude: float,
    L: int = 1,
    halo_class: int = 0,
) -> Dict[str, float]:
    """计算Halo轨道初始猜测参数

    用于生成高质量的初始猜测，配合微分修正器使用。
    基于 MATLAB 参考值和缩放关系。

    参数：
        mu: 质量比
        z_amplitude: Z方向振幅
        L: 拉格朗日点 (1=L1, 2=L2)
        halo_class: 0=北Halo, 1=南Halo

    返回：
        dict: 包含初始猜测参数的字典
            - x0: 初始x坐标
            - y0: 初始y坐标 (0)
            - z0: 初始z坐标
            - vx0: 初始vx (0)
            - vy0: 初始vy
            - vz0: 初始vz (0)
            - T_half: 半周期
    """
    if z_amplitude <= 0:
        raise ValueError(f"z_amplitude必须为正数，当前为{z_amplitude}")

    if halo_class == 1:
        delta_sign = -1
    else:
        delta_sign = 1

    # 初始猜测基于 MATLAB 参考值
    # 参考: FAMILY_L1Halo_North.m
    # SV0 = [0.9305,0,0.2300,0,0.1043,0]', tf = 1.8397
    if L == 1:
        x0 = 0.9305
        vy0 = 0.1043 * z_amplitude / 0.23
        # 对于 L1，半周期随振幅变化不大，约为 0.92
        T_half = 0.91985
    else:
        x0 = 1.15
        vy0 = 0.1043 * z_amplitude / 0.23
        # 对于 L2，半周期也约为 0.92
        T_half = 0.91985

    return {
        "x0": x0,
        "y0": 0.0,
        "z0": delta_sign * z_amplitude,
        "vx0": 0.0,
        "vy0": vy0,
        "vz0": 0.0,
        "T_half": T_half,
        "Au": z_amplitude * 0.5,
        "Aw": z_amplitude,
    }


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
        "halo_orbit_fixed_z0",
        "halo_orbit_fixed_x0",
    ]

    def __init__(
        self,
        dynamic: CR3BP_Dynamics,
        target: Optional[Dict[str, Any]] = None,
        free_vars: Optional[List[str]] = None,
    ) -> None:
        """初始化修正器

        参数：
        - dynamic: CR3BP_Dynamics对象
        - target: 目标约束条件字典（可选）
        - free_vars: 自由变量列表（可选）
        """
        # 核心对象
        self.dynamics = dynamic
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

    def setup_2D_symmetric_y_fixed_y0(self, y0=0.0):
        """配置平面问题中固定初始y坐标的y轴对称周期轨道搜索

        适用于共振轨道(RO)等从y轴出发的周期轨道。
        轨道从点(0, y0)出发（x=0, x_dot=0），经过半周期T/2后再次穿越y轴（x=0, x_dot=0）。

        参数:
            y0 (float): 固定的初始y坐标

        返回:
            self: 配置好的微分修正器实例

        配置说明:
            - 自由变量: [x_dot0, T_half] - 初始x方向速度和半周期时间
            - 目标约束: [x(T/2)=0, x_dot(T/2)=0] - 终点处再次穿越y轴
            - 状态向量索引: [0, 3] 分别对应x坐标和x方向速度
        """
        self.setup_type = "2D_symmetric_y_fixed_y0"
        self.symmetry_condition = "y_axis"
        self.fixed_parameters = {"y0": y0}

        # 对于y轴对称轨道，从(0, y0)出发，初始状态: [0, y0, 0, x_dot, 0, 0]
        # 自由变量是初始x方向速度 x_dot 和半周期 T/2
        self.free_variables = ["x_dot0", "T_half"]
        self.free_variable_indices = [3, 6]  # x_dot索引3，索引6表示时间

        # 目标约束条件
        # 半周期处应满足: x(T/2)=0, x_dot(T/2)=0
        self.target_conditions = {
            "x": 0.0,  # 终点x坐标为0
            "x_dot": 0.0,  # 终点x方向速度为0
        }

        # 状态向量为 [x, y, z, x_dot, y_dot, z_dot]
        self.constraint_indices = [0, 3]  # x和x_dot在状态向量中的索引

        self.constraint_weights = {"x": 1.0, "x_dot": 1.0}
        self.constraint_types = {"x": "equality", "x_dot": "equality"}

        self._reset_history()

        print(
            f"2D对称y轴配置完成：固定y0={y0}，自由变量={self.free_variables}，目标约束={list(self.target_conditions.keys())}"
        )

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

    def setup_halo_orbit_fixed_z0(self, z0, libration_point=1):
        """配置 Halo 轨道微分修正，固定初始 Z0（XZ 对称）

        Halo 轨道具有 XZ 平面对称性，利用该对称性可以将问题简化为：
        从 XZ 平面上一点 (x0, 0, z0) 出发，经过半周期 T/2 后再次到达 XZ 平面。

        参数:
            z0 (float): 固定的初始 z 坐标
            libration_point (int): 平动点编号 (1=L1, 2=L2)，默认 L1

        返回:
            self: 配置好的微分修正器实例

        配置说明:
            - 自由变量: [x0, y_dot0, T_half] - 初始 x 坐标、y 方向速度和半周期时间
            - 目标约束: [y(T/2)=0, x_dot(T/2)=0, z_dot(T/2)=0] - 半周期处再次位于 XZ 平面
            - 状态向量索引: [0, 4, 6] 分别对应 x0、y_dot0 和时间 T_half
            - 注意: z 在半周期时会改变符号，不作为约束
        """
        self.setup_type = "halo_orbit_fixed_z0"
        self.symmetry_condition = "xz_plane"
        self.fixed_parameters = {"z0": z0, "libration_point": libration_point}

        self.free_variables = ["x0", "y_dot0", "T_half"]
        self.free_variable_indices = [0, 4, 6]

        self.target_conditions = {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}
        self.constraint_indices = [1, 3, 5]

        self._reset_history()

        print(
            f"Halo 轨道配置完成（固定 Z0）：z0={z0}，平动点=L{libration_point}，"
            f"自由变量={self.free_variables}，目标约束={list(self.target_conditions.keys())}"
        )

        return self

    def setup_halo_orbit_fixed_x0(self, x0, libration_point=1):
        """配置 Halo 轨道微分修正，固定初始 X0（XZ 对称）

        Halo 轨道具有 XZ 平面对称性，利用该对称性可以将问题简化为：
        从 XZ 平面上一点 (x0, 0, z0) 出发，经过半周期 T/2 后再次到达 XZ 平面。

        参数:
            x0 (float): 固定的初始 x 坐标
            libration_point (int): 平动点编号 (1=L1, 2=L2)，默认 L1

        返回:
            self: 配置好的微分修正器实例

        配置说明:
            - 自由变量: [z0, y_dot0, T_half] - 初始 z 坐标、y 方向速度和半周期时间
            - 目标约束: [y(T/2)=0, x_dot(T/2)=0, z_dot(T/2)=0] - 半周期处再次位于 XZ 平面且垂直穿越
            - 状态向量索引: [2, 4, 6] 分别对应 z0、y_dot0 和时间 T_half
        """
        self.setup_type = "halo_orbit_fixed_x0"
        self.symmetry_condition = "xz_plane"
        self.fixed_parameters = {"x0": x0, "libration_point": libration_point}

        self.free_variables = ["z0", "y_dot0", "T_half"]
        self.free_variable_indices = [2, 4, 6]

        self.target_conditions = {"y": 0.0, "x_dot": 0.0, "z_dot": 0.0}
        self.constraint_indices = [1, 3, 5]

        self._reset_history()

        print(
            f"Halo 轨道配置完成（固定 X0）：x0={x0}，平动点=L{libration_point}，"
            f"自由变量={self.free_variables}，目标约束={list(self.target_conditions.keys())}"
        )

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

        # 固定T模式下使用预设的T_half，否则从轨道周期计算
        if "T_half" in self.fixed_parameters:
            half_period_time = self.fixed_parameters["T_half"]
        else:
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
                    with_jacobi=False,
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

            # Halo：仅当 T/2 已崩溃到极小（寄生根）时拉回，避免一律钳位导致无法收敛到真实 ~0.92 TU
            if self.setup_type in ("halo_orbit_fixed_x0", "halo_orbit_fixed_z0"):
                if current_time < 0.02:
                    current_time = 0.25
            elif current_time <= 0:
                current_time = 1e-6
                if verbose:
                    print("  警告：时间调整为正值")

            if verbose:
                print(f"  修正量范数: {correction_norm:.2e}")
                print(f"  新状态: x={current_state[0]:.6f}, y_dot={current_state[4]:.6f}")
                print(f"  新半周期: T/2={current_time:.6f}")

            # 检查停滞（仅在未收敛的情况下检查）
            if not self.converged and correction_norm < self.stagnation_limit:
                # 修正量过小时，若误差已足够小，也视为收敛成功
                if current_error < 1e-8:
                    self.converged = True
                    self.termination_reason = "收敛成功：修正量过小但误差足够小"
                    self.current_error = current_error
                    if verbose:
                        print(
                            f"  收敛成功：修正量过小({correction_norm:.2e})但误差已足够小({current_error:.2e})"
                        )
                    break
                else:
                    self.termination_reason = "停滞：修正量过小"
                    if verbose:
                        print(f"  停滞：修正量 = {correction_norm:.2e}")
                    break

        # 迭代结束，处理结果
        if self.converged:
            # 验证周期合理性（防止收敛到无效解如周期接近0的轨道）
            if self.setup_type in ("halo_orbit_fixed_z0", "halo_orbit_fixed_x0"):
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
                    print(f"\n微分修正失败: {self.termination_reason}")
                return None

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
        """根据修正结果积分生成完整周期轨道。

        参数:
            result: 包含修正结果的字典，包含 state (半周期对称点状态) 和 period (完整周期)

        返回:
            Orbit: 完整的周期轨道对象
        """
        full_period = result["period"]
        initial_state = result["state"]

        # 第一次积分完整周期
        propagation = integrate.solve_ivp(
            self.dynamics.equations_of_motion,
            (0, full_period),
            initial_state,
            method="DOP853",
            t_eval=np.linspace(0, full_period, 1000),
            rtol=1e-12,
            atol=1e-12,
        )

        final_state = propagation.y[:, -1]
        closure_error = np.linalg.norm(final_state - initial_state)

        # 检查闭合性，如果不闭合则尝试修正
        if closure_error > 1e-10:
            closure_error_vector = final_state - initial_state
            pos_error = np.linalg.norm(closure_error_vector[:3])
            vel_error = np.linalg.norm(closure_error_vector[3:])

            if pos_error > 1e-14 and vel_error > 1e-14:
                # 调整初始速度来修正位置误差
                adjustment = -0.5 * closure_error_vector[3:]
                new_state = initial_state.copy()
                new_state[4] += adjustment[1]

                # 重新积分
                propagation = integrate.solve_ivp(
                    self.dynamics.equations_of_motion,
                    (0, full_period),
                    new_state,
                    method="DOP853",
                    t_eval=np.linspace(0, full_period, 1000),
                    rtol=1e-12,
                    atol=1e-12,
                )
                final_state = propagation.y[:, -1]
                new_closure_error = np.linalg.norm(final_state - new_state)

                if new_closure_error < closure_error:
                    initial_state = new_state
                    closure_error = new_closure_error

        # 确保 states 是独立的副本（避免与 propagation.y 共享内存）
        orbit_states = np.array(propagation.y.T, copy=True)

        orbit = Orbit(
            states=orbit_states,
            times=propagation.t.copy(),
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
            f"DifferentialCorrection(dynamic={self.dynamics}, "
            f"setup={self.setup_type}, tol={self.tolerance})"
        )
