"""
三体问题动力学模块

包含通用 Dynamics 基类和 CR3BP_Dynamics 类，用于计算和积分圆型限制性三体问题的动力学方程。

物理背景
--------
在圆型限制性三体问题 (CR3BP) 中，两个主天体（如地球和月球）绕其公共质心做圆周运动，
第三体（航天器）质量小到不影响两个主天体的运动。采用以质心为原点的旋转坐标系，
使得两个主天体固定在 x 轴上。

坐标系约定：
  - 原点：系统质心
  - x 轴：从质心指向较大天体（质量 1-μ）的方向
  - 较大天体位于 x = -μ，较小天体（质量 μ）位于 x = 1-μ
  - y 轴在轨道平面内垂直于 x 轴
  - z 轴与 x-y 平面正交

所有量均采用无量纲化单位（距离单位 DU = 主天体间距，时间单位 TU 使主天体角速度为 1）。
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional, Any, Callable, TYPE_CHECKING

import numpy.typing as npt

from .system import CR3BP_System

if TYPE_CHECKING:
    from .orbit import Orbit
    from .system import CR3BP_System as SystemType


class Dynamics:
    """通用天体系统动力学基类

    Attributes:
        system: 关联的系统对象
        integrator: 数值积分器类型
        rtol: 相对积分容差
        atol: 绝对积分容差
        max_step: 最大积分步长
        last_trajectory: 最近一次积分的轨迹 [t, y]
        last_stm: 最近一次积分的状态转移矩阵
        cross_section_tolerance: 截面检测容差
        last_crossing: 上次穿过截面的点和时间
        jacobi_history: Jacobi常数历史记录
        jacobi_error: Jacobi常数误差
        initialized: 初始化完成标志
    """

    DEFAULT_TOLERANCE = 1e-12  # 默认积分容差，双精度机器精度量级，确保数值解精度
    DEFAULT_MAX_STEP = 0.01  # 默认最大积分步长（无量纲时间），约为主天体轨道周期的 0.16%

    def __init__(self, system: CR3BP_System) -> None:
        """初始化动力学

        Args:
            system: CR3BP_System对象
        """
        self.system = system

        # --- 积分器配置 ---
        # 使用 RK45（显式 Runge-Kutta 4(5) 自适应步长法）作为默认积分器。
        # RK45 是非刚性问题的高效通用求解器，适合 CR3BP 轨道积分。
        # 对于高精度需求（如周期轨道修正），rtol=atol=1e-12 确保数值误差
        # 远小于微分修正的收敛阈值。
        self.integrator = "RK45"
        self.rtol = self.DEFAULT_TOLERANCE
        self.atol = self.DEFAULT_TOLERANCE
        self.max_step = self.DEFAULT_MAX_STEP

        # 缓存最近一次积分结果，供后续分析（如截面检测）使用
        self.last_trajectory = None  # (time_array, states_array)
        self.last_stm = None  # STM 矩阵数组，形状 (n_points, 6, 6)

        # 截面（Poincaré section）检测参数
        self.cross_section_tolerance = 1e-8  # 判断"穿过截面"的距离阈值
        self.last_crossing = None  # 上次穿越截面时的 (时间, 状态)

        # Jacobi 常数（CR3BP 唯一积分不变量）的沿轨迹监测
        self.jacobi_history = []  # 逐点 Jacobi 常数列表
        self.jacobi_error = 0.0  # Jacobi 常数的最大漂移量，用于评估积分精度

        self.initialized = True

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """运动方程（子类需实现）

        Args:
            t: 时间
            state: 状态向量

        Returns:
            状态导数

        Raises:
            NotImplementedError: 子类未实现此方法
        """
        raise NotImplementedError("子类必须实现此方法")

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[npt.ArrayLike] = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
    ) -> Dict[str, Any]:
        """传播轨迹

        Args:
            initial_state: 初始状态向量
            t_span: 时间区间 [t0, tf]
            t_eval: 评估时间点数组（可选）
            with_stm: 是否计算状态转移矩阵
            with_jacobi: 是否沿轨迹逐点计算 Jacobi 常数并写入
                ``jacobi`` / ``jacobi_error``（默认关闭以减轻粗积分负担）

        Returns:
            轨迹结果字典，包含 ``time`` 和 ``states`` 键；
            仅当 ``with_jacobi=True`` 时额外包含 ``jacobi`` 与 ``jacobi_error`` 键
        """
        # 调用 scipy.integrate.solve_ivp 进行数值积分。
        # max_step 限制积分器最大步长，防止在轨道曲率较大区域（如近天体飞越）
        # 步长过大导致精度下降。
        result = solve_ivp(
            self.equations_of_motion,
            t_span,
            initial_state,
            method=self.integrator,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
        )

        # result.y 形状为 (6, n_points)，转置为 (n_points, 6) 以便按行访问各时刻的状态
        states = result.y.T

        self.last_trajectory = (result.t, states)

        out: Dict[str, Any] = {
            "time": result.t,
            "states": states,
        }

        # 可选：逐点计算 Jacobi 常数，用于验证积分质量。
        # Jacobi 常数是 CR3BP 中唯一的运动积分（守恒量），其漂移量直接
        # 反映数值积分的精度。diff 取相邻点之差，max(abs) 取最大漂移。
        if with_jacobi and hasattr(self, "compute_jacobi_constant"):
            self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
            if len(self.jacobi_history) > 1:
                self.jacobi_error = float(np.max(np.abs(np.diff(self.jacobi_history))))
            else:
                self.jacobi_error = 0.0
            out["jacobi"] = self.jacobi_history
            out["jacobi_error"] = self.jacobi_error
        else:
            self.jacobi_history = []
            self.jacobi_error = 0.0

        return out

    def compute_jacobi_constant(self, state: npt.ArrayLike) -> float:
        """计算能量常数（子类需实现）

        Args:
            state: 状态向量

        Returns:
            能量常数

        Raises:
            NotImplementedError: 子类未实现此方法
        """
        raise NotImplementedError("子类必须实现此方法")

    def check_cross_section(self, state: npt.ArrayLike, plane: str, value: float) -> bool:
        """检查是否穿过指定截面

        Args:
            state: 状态向量
            plane: 截面平面 ('x', 'y', 'z')
            value: 平面值

        Returns:
            是否穿过截面

        Raises:
            ValueError: 无效的平面参数
        """
        state = np.asarray(state, dtype=float)
        if plane == "x":
            return abs(state[0] - value) < self.cross_section_tolerance
        elif plane == "y":
            return abs(state[1] - value) < self.cross_section_tolerance
        elif plane == "z":
            return abs(state[2] - value) < self.cross_section_tolerance
        else:
            raise ValueError(f"无效的平面: {plane}。可用平面: 'x', 'y', 'z'")

    def __str__(self):
        return f"{self.__class__.__name__}(system={self.system})"

    def __repr__(self):
        return f"{self.__class__.__name__}(system={self.system}, integrator='{self.integrator}', rtol={self.rtol})"


class CR3BP_Dynamics(Dynamics):
    """CR3BP动力学方程

    封装了CR3BP的动力学模型，提供状态传播、状态转移矩阵计算、
    Jacobi常数计算等核心功能。支持6维状态向量（位置+速度）和
    42维增广状态向量（状态+状态转移矩阵）的数值积分。

    CR3BP 运动方程（旋转坐标系中）：
        ẍ - 2ẏ = ∂Ω/∂x
        ÿ + 2ẋ = ∂Ω/∂y
        z̈       = ∂Ω/∂z

    其中 Ω 为伪势能（见 equations_of_motion 方法的详细注释），
    等号左侧的 2ẏ、-2ẋ 项为科里奥利力（Coriolis），伪势能中
    已包含离心力项 x²/2 + y²/2。

    Attributes:
        STM_DIMENSION: 增广状态向量维度（6状态 + 36个STM元素 = 42）
    """

    # 增广状态向量维度：6维状态 [x,y,z,vx,vy,vz] + 6x6=36维状态转移矩阵元素
    STM_DIMENSION = 42

    def __init__(self, system: CR3BP_System) -> None:
        """初始化CR3BP动力学

        Args:
            system: CR3BP_System对象，包含质量参数μ等系统常数
        """
        super().__init__(system)

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """6维状态向量的运动方程

        实现 CR3BP 在旋转坐标系中的运动方程。旋转坐标系以两个主天体的
        公共质心为原点，与主天体同步旋转（角速度 ω = 1），因此两个主天体
        在坐标系中固定不动。

        在旋转坐标系中，运动方程为：
            ẍ - 2ẏ = ∂Ω/∂x    (x 方向：离心力 + 引力 + 科里奥利力)
            ÿ + 2ẋ = ∂Ω/∂y    (y 方向：离心力 + 引力 + 科里奥利力)
            z̈       = ∂Ω/∂z    (z 方向：仅引力，无科里奥利力)

        伪势能 Ω = (x² + y²)/2 + (1-μ)/r₁ + μ/r₂，其偏导数为：
            ∂Ω/∂x = x - (1-μ)(x+μ)/r₁³ - μ(x-1+μ)/r₂³
            ∂Ω/∂y = y - (1-μ)y/r₁³     - μy/r₂³
            ∂Ω/∂z =   - (1-μ)z/r₁³     - μz/r₂³

        因此加速度各项的物理含义：
          - "x" / "y" 项：离心力（伪势能中的二次项贡献）
          - "(1-μ)(x+μ)/r₁³" 等：较大天体（如地球）的引力加速度
          - "μ(x-1+μ)/r₂³" 等：较小天体（如月球）的引力加速度
          - "2vy" / "-2vx"：科里奥利力（旋转坐标系中的虚拟力）

        Args:
            t: 时间（旋转坐标系中，CR3BP方程不显含时间，即自治系统）
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            状态导数 [vx, vy, vz, ax, ay, az]
        """
        mu = self.system.mu  # 质量参数 μ = m₂/(m₁+m₂)，m₂ 为较小天体质量

        x, y, z, vx, vy, vz = state

        # r₁：航天器到较大天体（质量 1-μ，位于 x=-μ）的距离
        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
        # r₂：航天器到较小天体（质量 μ，位于 x=1-μ）的距离
        r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)

        # --- x 方向加速度 ---
        # 2*vy：科里奥利力 x 分量（旋转坐标系效应）
        # x：离心力 x 分量（伪势能 ∂(x²/2)/∂x = x）
        # -(1-μ)*(x+μ)/r₁³：较大天体的引力加速度 x 分量
        # -μ*(x-1+μ)/r₂³：较小天体的引力加速度 x 分量
        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        # --- y 方向加速度 ---
        # -2*vx：科里奥利力 y 分量
        # y：离心力 y 分量（伪势能 ∂(y²/2)/∂y = y）
        # -(1-μ)*y/r₁³ 和 -μ*y/r₂³：两天体的引力加速度 y 分量
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
        # --- z 方向加速度 ---
        # z 方向无离心力和科里奥利力（旋转轴方向）
        # 仅受两天体引力作用
        az = -(1 - mu) * z / r1**3 - mu * z / r2**3

        return np.array([vx, vy, vz, ax, ay, az])

    def equations_with_stm(
        self, t: float, augmented_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """42维增广状态向量的运动方程（包含状态转移矩阵）

        同时积分状态向量和状态转移矩阵(STM)，满足 dΦ/dt = A(t)·Φ。

        状态转移矩阵 Φ(t, t₀) 将初始状态的微小扰动映射到当前时刻：
            δx(t) = Φ(t, t₀) · δx(t₀)

        通过将 Φ 拉伸为 36 维向量并与 6 维状态拼接为 42 维增广状态，
        可以用标准的 ODE 积分器同时求解轨道和 STM。

        矩阵微分方程 dΦ/dt = A(t)·Φ 中，A(t) 是运动方程右端函数对
        状态向量的雅可比矩阵（6×6），其结构为：

            A = | 0₃ₓ₃   I₃ₓ₃ |    上半部分：位置导数 = 速度（恒等映射）
                | U_ij   Ω   |    下半部分：速度导数 = 加速度线性化

        其中 U_ij = ∂²Ω/∂rᵢ∂rⱼ 是伪势能 Ω 的 Hessian 矩阵元素，
        Ω 包含科里奥利力项：
            Ω = |  0   2   0 |    ← 科里奥利力耦合（旋转坐标系效应）
                | -2   0   0 |
                |  0   0   0 |

        Args:
            t: 时间
            augmented_state: 增广状态向量 [6状态 + 36个STM元素]

        Returns:
            增广状态导数
        """
        mu = self.system.mu

        # 解包前 6 个分量为状态向量
        x, y, z, vx, vy, vz = augmented_state[:6]

        # --- 计算到两个主天体的距离 ---
        # 较大天体位于 (-μ, 0, 0)，较小天体位于 (1-μ, 0, 0)
        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)

        # --- 状态方程（与 equations_of_motion 完全相同）---
        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
        az = -(1 - mu) * z / r1**3 - mu * z / r2**3

        state_derivative = np.array([vx, vy, vz, ax, ay, az])

        # --- 状态转移矩阵部分 ---
        # 将 36 维向量重塑为 6×6 矩阵
        stm = augmented_state[6:].reshape((6, 6))

        # ================================================================
        # 伪势能二阶偏导数（Hessian 矩阵 U_ij = ∂²Ω/∂rᵢ∂rⱼ）
        # ================================================================
        # 伪势能 Ω = (x²+y²)/2 + (1-μ)/r₁ + μ/r₂
        # 其 Hessian 分为两部分：
        #   1) 离心力贡献：∂²((x²+y²)/2)/∂x² = 1，其余为 0
        #   2) 引力贡献：对 (1-μ)/r₁ 和 μ/r₂ 求二阶偏导
        #
        # 引力势的二阶偏导公式（以 r₁ 部分为例）：
        #   ∂²(1/r₁)/∂xᵢ∂xⱼ = -δᵢⱼ/r₁³ + 3·Δxᵢ·Δxⱼ/r₁⁵
        # 其中 Δx = x - x_body 为相对于天体的坐标差。

        # U_xx = 1 + 引力项
        # 离心力项贡献 1（∂²(x²/2)/∂x² = 1），
        # 引力项由 ∂²Ω_grav/∂x² = -(1-μ)[1/r₁³ - 3(x+μ)²/r₁⁵] - μ[1/r₂³ - 3(x-1+μ)²/r₂⁵] 给出
        U_xx = (
            1
            - (1 - mu) * (1 / r1**3 - 3 * (x + mu) ** 2 / r1**5)
            - mu * (1 / r2**3 - 3 * (x - 1 + mu) ** 2 / r2**5)
        )
        # U_yy = 1 + 引力项（离心力项 ∂²(y²/2)/∂y² = 1）
        U_yy = 1 - (1 - mu) * (1 / r1**3 - 3 * y**2 / r1**5) - mu * (1 / r2**3 - 3 * y**2 / r2**5)
        # U_zz：z 方向无离心力贡献（伪势能中无 z² 项），只有引力
        U_zz = -(1 - mu) / r1**3 - mu / r2**3
        # U_xy = U_yx：交叉项，由引力势产生（离心力势不含交叉项）
        U_xy = 3 * (1 - mu) * (x + mu) * y / r1**5 + 3 * mu * (x - 1 + mu) * y / r2**5
        # U_xz = U_zx
        U_xz = 3 * (1 - mu) * (x + mu) * z / r1**5 + 3 * mu * (x - 1 + mu) * z / r2**5
        # U_yz = U_zy
        U_yz = 3 * (1 - mu) * y * z / r1**5 + 3 * mu * y * z / r2**5

        # ================================================================
        # 状态方程雅可比矩阵 A(t)（6×6）
        # ================================================================
        # 结构：
        #   | 0₃ₓ₃  I₃ₓ₃ |    位置方程的雅可比：∂(v)/∂(r,v) = [0, I]
        #   | U_ij   Ω   |    速度方程的雅可比：∂(a)/∂(r,v) = [U, Ω]
        #
        # 其中 Ω 矩阵（科里奥利力对速度的偏导）为：
        #   ∂(2vy)/∂vx = 0,  ∂(2vy)/∂vy = 2,  ∂(-2vx)/∂vx = -2, ...
        # 即：
        #   Ω = | 0   2   0 |
        #       |-2   0   0 |
        #       | 0   0   0 |
        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [U_xx, U_xy, U_xz, 0, 2, 0],
                [U_xy, U_yy, U_yz, -2, 0, 0],
                [U_xz, U_yz, U_zz, 0, 0, 0],
            ]
        )

        # STM 的微分方程：dΦ/dt = A(t) · Φ
        # 这是矩阵微分方程，对 Φ 的每一列分别做矩阵-向量乘法
        stm_dot = A @ stm

        # 将状态导数和 STM 导数拼接为 42 维向量返回
        derivative = np.concatenate([state_derivative, stm_dot.flatten()])

        return derivative

    def propagate(
        self,
        initial_state: npt.ArrayLike,
        t_span: Tuple[float, float],
        t_eval: Optional[npt.ArrayLike] = None,
        with_stm: bool = False,
        with_jacobi: bool = False,
    ) -> Dict[str, Any]:
        """传播轨迹

        Args:
            initial_state: 初始状态向量
            t_span: 时间区间 [t0, tf]
            t_eval: 评估时间点数组（可选）
            with_stm: 是否计算状态转移矩阵
            with_jacobi: 是否逐点计算 Jacobi 常数（默认 ``False``，
                搜索/大量积分时可显著减少开销）

        Returns:
            轨迹结果字典。始终包含 ``time`` 和 ``states`` 键；
            当 ``with_stm=True`` 时额外包含 ``stm`` 键；
            当 ``with_jacobi=True`` 时额外包含 ``jacobi`` 与 ``jacobi_error`` 键
        """
        if with_stm:
            # --- 增广状态积分路径（42 维）---
            # 初始 STM 设为单位矩阵 Φ(t₀, t₀) = I₆，
            # 拉伸为 36 维向量后与初始状态拼接
            initial_stm = np.eye(6).flatten()
            augmented_state = np.concatenate([initial_state, initial_stm])

            # 使用 42 维增广方程同时积分状态和 STM。
            # 积分 42 维方程的开销约为纯状态方程的 7 倍（42/6），
            # 但可一次性获得完整的状态敏感度信息，用于微分修正等。
            result = solve_ivp(
                self.equations_with_stm,
                t_span,
                augmented_state,
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )

            # 从 42 维结果中分离状态（前 6 行）和 STM（后 36 行）
            states = result.y[:6, :].T  # (n_points, 6)
            stm_matrices = result.y[6:, :].T.reshape(-1, 6, 6)  # (n_points, 6, 6)

            self.last_trajectory = (result.t, states)
            self.last_stm = stm_matrices

            out: Dict[str, Any] = {
                "time": result.t,
                "states": states,
                "stm": stm_matrices,
            }

            if with_jacobi:
                self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
                if len(self.jacobi_history) > 1:
                    self.jacobi_error = float(np.max(np.abs(np.diff(self.jacobi_history))))
                else:
                    self.jacobi_error = 0.0
                out["jacobi"] = self.jacobi_history
                out["jacobi_error"] = self.jacobi_error
            else:
                self.jacobi_history = []
                self.jacobi_error = 0.0

            return out
        else:
            # --- 纯状态积分路径（6 维）---
            # 不需要 STM 时仅积分 6 维状态方程，计算量更小。
            # 适用于轨道搜索、批量积分等无需敏感度信息的场景。
            result = solve_ivp(
                self.equations_of_motion,
                t_span,
                initial_state,
                method=self.integrator,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )

            states = result.y.T

            self.last_trajectory = (result.t, states)

            out = {
                "time": result.t,
                "states": states,
            }

            if with_jacobi:
                self.jacobi_history = [self.compute_jacobi_constant(state) for state in states]
                if len(self.jacobi_history) > 1:
                    self.jacobi_error = float(np.max(np.abs(np.diff(self.jacobi_history))))
                else:
                    self.jacobi_error = 0.0
                out["jacobi"] = self.jacobi_history
                out["jacobi_error"] = self.jacobi_error
            else:
                self.jacobi_history = []
                self.jacobi_error = 0.0

            return out

    def propagate_orbit_state_at_time(
        self,
        orbit: Orbit,
        t: float,
        integration_dt: float = 0.01,
    ) -> npt.NDArray[np.floating]:
        """从轨道首点状态积分到给定时刻对应的相位（周期轨道上对周期取模）

        利用周期轨道的周期性，将目标时间对周期取模后从轨道起始状态
        重新积分，得到该相位处的精确状态。适用于需要轨道上任意时刻
        精确状态的场景（如多段拼接、状态约束等）。

        Args:
            orbit: 周期轨道数据（须含 ``states``、``times``、有效 ``period``）
            t: 与轨道 ``times`` 一致的时间坐标（绝对时间）
            integration_dt: 构造 ``t_eval`` 的步长

        Returns:
            积分末端状态 ``[x, y, z, vx, vy, vz]``

        Raises:
            ValueError: 轨道无状态或周期无效
        """
        if orbit.states.shape[0] < 1:
            raise ValueError("轨道无状态")
        if orbit.period is None or orbit.period <= 0:
            raise ValueError("轨道周期无效，无法沿周期外推")

        t0 = float(orbit.times[0])
        period = float(orbit.period)
        # 利用周期性：将绝对时间转换为轨道内的相对相位
        t_rel = float(np.mod(t - t0, period))
        # 若 t_rel ≈ 0（即恰好在起始点），直接返回初始状态，避免不必要的积分
        if t_rel < 1e-14:
            return np.asarray(orbit.states[0], dtype=float).copy()

        # 构造等间距的评估时间点，确保积分轨迹有足够的时间分辨率
        n_steps = max(int(np.ceil(t_rel / integration_dt)) + 1, 2)
        t_eval = np.linspace(t0, t0 + t_rel, n_steps)
        result = self.propagate(
            initial_state=orbit.states[0],
            t_span=(t0, t0 + t_rel),
            t_eval=t_eval,
            with_stm=False,
            with_jacobi=False,
        )
        # 返回积分最后一个时刻的状态
        return np.asarray(result["states"][-1], dtype=float)

    def compute_state_transition_matrix(
        self, initial_state: npt.ArrayLike, t: float
    ) -> npt.NDArray[np.floating]:
        """计算状态转移矩阵

        Args:
            initial_state: 初始状态向量
            t: 积分终止时间

        Returns:
            状态转移矩阵 (6x6)
        """
        result = self.propagate(initial_state, (0.0, float(t)), with_stm=True, with_jacobi=False)

        return result["stm"][-1]

    def compute_jacobi_constant(self, state: npt.ArrayLike) -> float:
        """计算Jacobi常数

        Args:
            state: 状态向量 [x, y, z, vx, vy, vz]

        Returns:
            Jacobi常数
        """
        return self.system.get_jacobi_constant(state)

    def check_cross_section(self, state: npt.ArrayLike, plane: str, value: float) -> bool:
        """检查是否穿过指定截面

        Args:
            state: 状态向量
            plane: 截面平面 ('x', 'y', 'z')
            value: 平面值

        Returns:
            是否穿过截面

        Raises:
            ValueError: 无效的平面参数
        """
        state = np.asarray(state, dtype=float)
        if plane == "x":
            return abs(state[0] - value) < self.cross_section_tolerance
        elif plane == "y":
            return abs(state[1] - value) < self.cross_section_tolerance
        elif plane == "z":
            return abs(state[2] - value) < self.cross_section_tolerance
        else:
            raise ValueError(f"无效的平面: {plane}。可用平面: 'x', 'y', 'z'")

    def __str__(self):
        return f"CR3BP_Dynamics(system={self.system}, integrator='{self.integrator}')"

    def __repr__(self):
        return (
            f"CR3BP_Dynamics(system={self.system}, integrator='{self.integrator}', "
            f"rtol={self.rtol}, atol={self.atol}, max_step={self.max_step})"
        )


def propagate_state_at_orbit_time(
    orbit: Any,
    t: float,
    dynamics: CR3BP_Dynamics,
    integration_dt: float = 0.01,
) -> npt.NDArray[np.floating]:
    """委托 :meth:`CR3BP_Dynamics.propagate_orbit_state_at_time`，便于顶层导入兼容

    Args:
        orbit: 周期轨道数据
        t: 目标时间（绝对时间）
        dynamics: CR3BP动力学对象
        integration_dt: 积分步长

    Returns:
        积分末端状态
    """
    return dynamics.propagate_orbit_state_at_time(orbit, t, integration_dt)
