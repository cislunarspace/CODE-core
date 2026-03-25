"""
DRO到RO转移轨道NLP优化模块

实现论文Cui et al. (2025)中的"搜索-优化"两步法的优化阶段。
优化变量: y = {α, T, t_ins}
目标函数: J(y) = Δv1 + Δv2
约束: 位置连续性、速度平行性、撞星约束
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Callable
from enum import Enum
from scipy.optimize import minimize, Bounds
import warnings

from ..core.orbit import Orbit
from ..core.dynamics import CR3BP_Dynamics
from ..core.system import CR3BP_System

try:
    import coptpy
except ImportError:
    coptpy = None
    NlpCallbackBase = None
else:
    NlpCallbackBase = coptpy.NlpCallbackBase


class TransferType(Enum):
    DIRECT = "direct"
    LGA = "lga"
    EXTERNAL = "external"


class NLPOptimizationVariables:
    """NLP优化变量

    优化变量: y = {α, T, t_ins}

    属性:
        alpha: 切向速度比
        transfer_time: 转移时间T
        t_ins: 从轨道远地点到插入点的时间
    """

    def __init__(self, alpha: float, transfer_time: float, t_ins: float):
        self.alpha = alpha
        self.transfer_time = transfer_time
        self.t_ins = t_ins

    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.alpha, self.transfer_time, self.t_ins])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "NLPOptimizationVariables":
        """从numpy数组创建"""
        return cls(alpha=arr[0], transfer_time=arr[1], t_ins=arr[2])


class NLPOptimizationResult:
    """NLP优化结果

    属性:
        alpha: 切向速度比
        transfer_time: 转移时间
        t_ins: 插入时间
        objective_value: 目标函数值(总ΔV)
        delta_v1: 出发脉冲
        delta_v2: 插入脉冲
        transfer_trajectory: 转移轨迹状态序列
        transfer_times: 转移轨迹时间序列
        departure_state: 出发点状态
        insertion_state: 插入点状态
        final_state: 转移轨迹末端状态
        success: 优化是否成功
        message: 结果消息
        transfer_type: 转移类型
    """

    def __init__(
        self,
        alpha: float = 0.0,
        transfer_time: float = 0.0,
        t_ins: float = 0.0,
        objective_value: float = 0.0,
        delta_v1: float = 0.0,
        delta_v2: float = 0.0,
        transfer_trajectory: Optional[np.ndarray] = None,
        transfer_times: Optional[np.ndarray] = None,
        departure_state: Optional[np.ndarray] = None,
        insertion_state: Optional[np.ndarray] = None,
        final_state: Optional[np.ndarray] = None,
        success: bool = False,
        message: str = "",
        transfer_type: TransferType = TransferType.DIRECT,
        constraints_violation: Optional[Dict[str, float]] = None,
    ):
        self.alpha = alpha
        self.transfer_time = transfer_time
        self.t_ins = t_ins
        self.objective_value = objective_value
        self.delta_v1 = delta_v1
        self.delta_v2 = delta_v2
        self.transfer_trajectory = transfer_trajectory
        self.transfer_times = transfer_times
        self.departure_state = departure_state
        self.insertion_state = insertion_state
        self.final_state = final_state
        self.success = success
        self.message = message
        self.transfer_type = transfer_type
        self.constraints_violation = constraints_violation or {}


class DROTRONLPOptimizer:
    """DRO到RO转移轨道NLP优化器

    实现论文Section III.B的优化阶段算法。
    使用SQP(序贯二次规划)方法求解NLP问题。

    属性:
        system: CR3BP系统对象
        dynamics: CR3BP动力学对象
        departure_orbit: 出发点轨道
        arrival_orbit: 目标轨道
        departure_state: 出发点状态
        alpha_range: α搜索范围
        velocity_angle_tolerance: 速度平行性容差(弧度)
        earth_radius: 地球半径(无量纲)
        moon_radius: 月球半径(无量纲)
    """

    # 默认参数
    DEFAULT_ALPHA_RANGE = (0.5, 2.5)
    DEFAULT_TRANSFER_TIME_RANGE = (1.0, 30.0)  # 无量纲时间
    DEFAULT_T_INS_RANGE = (0.0, 10.0)  # 无量纲时间

    # 撞星约束半径(无量纲)
    EARTH_RADIUS_ND = 1.0 / 389703.0 * 6378.137  # ~0.0163 DU
    MOON_RADIUS_ND = 1738.1 / 384400.0  # ~0.00452 DU

    # 速度平行性容差
    DEFAULT_VELOCITY_ANGLE_TOL = 1e-6  # 弧度

    def __init__(
        self,
        system: CR3BP_System,
        dynamics: CR3BP_Dynamics,
        departure_orbit: Orbit,
        arrival_orbit: Orbit,
        departure_state: np.ndarray,
    ):
        """初始化NLP优化器

        参数:
            system: CR3BP系统对象
            dynamics: CR3BP动力学对象
            departure_orbit: 出发点轨道
            arrival_orbit: 目标轨道
            departure_state: 出发点状态 [x, y, z, vx, vy, vz]
        """
        self.system = system
        self.dynamics = dynamics
        self.mu = system.mu

        self.departure_orbit = departure_orbit
        self.arrival_orbit = arrival_orbit
        self.departure_state = departure_state

        # 默认搜索范围
        self.alpha_range = self.DEFAULT_ALPHA_RANGE
        self.transfer_time_range = self.DEFAULT_TRANSFER_TIME_RANGE
        self.t_ins_range = self.DEFAULT_T_INS_RANGE

        # 约束容差
        self.velocity_angle_tol = self.DEFAULT_VELOCITY_ANGLE_TOL

        # 地球和月球半径(无量纲)
        self.earth_radius = self.EARTH_RADIUS_ND
        self.moon_radius = self.MOON_RADIUS_ND

        # 缓存
        self._last_trajectory: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def compute_departure_velocity(
        self, state: np.ndarray, alpha: float, beta: float = 0.0
    ) -> np.ndarray:
        """根据α计算出发速度

        参数:
            state: 出发点状态 [x, y, z, vx, vy, vz]
            alpha: 切向速度比

        返回:
            注入速度向量 [vx, vy, vz]
        """
        pos = state[:3]
        vel = state[3:]

        # 轨道面法向(CR3BP中为z轴)
        normal = np.array([0.0, 0.0, 1.0])

        # 速度大小
        v_mag = np.linalg.norm(vel)
        if v_mag < 1e-10:
            warnings.warn("出发点速度接近零")
            return vel

        # 切向单位向量(沿速度方向)
        tangential = vel / v_mag

        # 法向单位向量
        normal_dir = np.cross(tangential, normal)
        norm_nd = np.linalg.norm(normal_dir)
        if norm_nd < 1e-10:
            normal_dir = np.array([1.0, 0.0, 0.0])
        else:
            normal_dir = normal_dir / norm_nd

        # 注入速度
        v_injection = alpha * v_mag * tangential + beta * v_mag * normal_dir

        return v_injection

    def forward_integrate(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前向积分转移弧

        参数:
            initial_state: 初始状态 [x, y, z, vx, vy, vz]
            t_span: 积分时间范围 (t0, tf)
            t_eval: 评估时间点

        返回:
            (times, states): 时间序列和状态序列
        """
        if t_eval is None:
            n_steps = int((t_span[1] - t_span[0]) / 0.01) + 1
            t_eval = np.linspace(t_span[0], t_span[1], n_steps)

        result = self.dynamics.propagate(
            initial_state=initial_state, t_span=t_span, t_eval=t_eval, with_stm=False
        )

        times = result["time"]
        states = result["states"]

        # 缓存
        self._last_trajectory = (times, states)

        return times, states

    def compute_delta_v1(self, departure_state: np.ndarray, initial_velocity: np.ndarray) -> float:
        """计算出发脉冲ΔV1

        参数:
            departure_state: 出发点状态 [x, y, z, vx, vy, vz]
            initial_velocity: 注入速度 [vx, vy, vz]

        返回:
            ΔV1 大小
        """
        original_vel = departure_state[3:]
        dv1 = np.linalg.norm(initial_velocity - original_vel)
        return dv1

    def compute_delta_v2(self, final_velocity: np.ndarray, insertion_velocity: np.ndarray) -> float:
        """计算插入脉冲ΔV2

        参数:
            final_velocity: 转移轨迹末端速度
            insertion_velocity: 目标轨道插入点速度

        返回:
            ΔV2 大小
        """
        dv2 = np.linalg.norm(final_velocity - insertion_velocity)
        return dv2

    def get_arrival_state_at_t_ins(self, t_ins: float) -> Tuple[np.ndarray, np.ndarray]:
        """获取目标轨道上t_ins对应的状态

        参数:
            t_ins: 从远地点开始的时间

        返回:
            (position, velocity): 位置和速度
        """
        # 找到远地点(最大x的位置)
        x_max_idx = np.argmax(self.arrival_orbit.states[:, 0])

        # 从远地点开始计算插入点时间
        t_from_apogee = t_ins % self.arrival_orbit.period

        # 在目标轨道上插值
        arrival_state = self.arrival_orbit.interpolate_at_time(t_from_apogee)

        return arrival_state[:3], arrival_state[3:]

    def objective_function(self, y: np.ndarray) -> float:
        """目标函数 J(y) = Δv1 + Δv2

        参数:
            y: 优化变量 [alpha, T, t_ins]

        返回:
            总脉冲代价
        """
        alpha, transfer_time, t_ins = y

        # 构建出发速度
        v_injection = self.compute_departure_velocity(self.departure_state, alpha)

        # 完整初始状态
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        # 前向积分
        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return 1e10  # 积分失败

        # 获取转移轨迹末端状态
        final_state = states[-1]

        # 获取插入点状态
        insertion_state = self.arrival_orbit.interpolate_at_time(t_ins)

        # 计算ΔV
        dv1 = self.compute_delta_v1(self.departure_state, v_injection)
        dv2 = self.compute_delta_v2(final_state[3:], insertion_state[3:])

        return dv1 + dv2

    def constraint_position(self, y: np.ndarray) -> float:
        """位置连续性约束 Eq.(13)

        (x_f - x_ins)^2 + (y_f - y_ins)^2 + (z_f - z_ins)^2 = 0

        参数:
            y: 优化变量 [alpha, T, t_ins]

        返回:
            约束违反量
        """
        alpha, transfer_time, t_ins = y

        # 构建出发速度
        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        # 前向积分
        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return 1e6

        final_state = states[-1]
        insertion_state = self.arrival_orbit.interpolate_at_time(t_ins)

        # 位置连续性
        pos_diff = final_state[:3] - insertion_state[:3]
        constraint = np.dot(pos_diff, pos_diff)

        return constraint

    def constraint_velocity_parallel(self, y: np.ndarray) -> float:
        """速度平行性约束 Eq.(14) 或relaxed Eq.(17)

        v_f · v_ins / (||v_f|| ||v_ins||) - 1 = 0

        参数:
            y: 优化变量 [alpha, T, t_ins]

        返回:
            约束违反量
        """
        alpha, transfer_time, t_ins = y

        # 构建出发速度
        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        # 前向积分
        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return 1e6

        final_state = states[-1]
        insertion_state = self.arrival_orbit.interpolate_at_time(t_ins)

        v_f = final_state[3:]
        v_ins = insertion_state[3:]

        v_f_norm = np.linalg.norm(v_f)
        v_ins_norm = np.linalg.norm(v_ins)

        if v_f_norm < 1e-10 or v_ins_norm < 1e-10:
            return 1e6

        cos_angle = np.dot(v_f, v_ins) / (v_f_norm * v_ins_norm)

        # 约束: cos_angle - 1 = 0 (速度平行)
        constraint = cos_angle - 1.0

        return constraint

    def check_collision(self, y: np.ndarray) -> Tuple[bool, bool]:
        """检查是否撞击地球或月球

        参数:
            y: 优化变量 [alpha, T, t_ins]

        返回:
            (earth_collision, moon_collision): 是否撞击地球、月球
        """
        alpha, transfer_time, t_ins = y

        # 构建出发速度
        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        # 前向积分
        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return False, False

        earth_collision = False
        moon_collision = False

        for state in states:
            pos = state[:3]

            # 到地球距离(地球在-mu处)
            r_earth = np.sqrt((pos[0] + self.mu) ** 2 + pos[1] ** 2 + pos[2] ** 2)
            if r_earth < self.earth_radius:
                earth_collision = True

            # 到月球距离(月球在1-mu处)
            r_moon = np.sqrt((pos[0] - 1 + self.mu) ** 2 + pos[1] ** 2 + pos[2] ** 2)
            if r_moon < self.moon_radius:
                moon_collision = True

        return earth_collision, moon_collision

    def optimize(
        self,
        initial_guess: Optional[NLPOptimizationVariables] = None,
        alpha_range: Optional[Tuple[float, float]] = None,
        transfer_time_range: Optional[Tuple[float, float]] = None,
        t_ins_range: Optional[Tuple[float, float]] = None,
        use_relaxed_velocity_constraint: bool = False,
        velocity_angle_constraint: float = 0.0,
        verbose: bool = True,
    ) -> NLPOptimizationResult:
        """执行NLP优化

        参数:
            initial_guess: 初始猜测
            alpha_range: α范围
            transfer_time_range: 转移时间范围
            t_ins_range: 插入时间范围
            use_relaxed_velocity_constraint: 是否使用松弛速度约束
            velocity_angle_constraint: 松弛速度约束角度(弧度)
            verbose: 是否打印信息

        返回:
            优化结果
        """
        # 设置搜索范围
        if alpha_range is not None:
            self.alpha_range = alpha_range
        if transfer_time_range is not None:
            self.transfer_time_range = transfer_time_range
        if t_ins_range is not None:
            self.t_ins_range = t_ins_range

        # 默认初始猜测
        if initial_guess is None:
            alpha0 = 1.0
            T0 = 10.0
            t_ins0 = 5.0
        else:
            alpha0 = initial_guess.alpha
            T0 = initial_guess.transfer_time
            t_ins0 = initial_guess.t_ins

        y0 = np.array([alpha0, T0, t_ins0])

        if verbose:
            print("\n开始NLP优化:")
            print(f"  初始猜测: α={alpha0:.4f}, T={T0:.4f}, t_ins={t_ins0:.4f}")
            print(f"  α范围: [{self.alpha_range[0]}, {self.alpha_range[1]}]")
            print(f"  T范围: [{self.transfer_time_range[0]}, {self.transfer_time_range[1]}]")
            print(f"  t_ins范围: [{self.t_ins_range[0]}, {self.t_ins_range[1]}]")

        # 定义约束
        constraints = []

        # 位置连续性约束
        constraints.append({"type": "eq", "fun": self.constraint_position})

        # 速度平行性约束
        if use_relaxed_velocity_constraint:
            # 松弛约束: cos(θ) - cos(θ_max) <= 0
            cos_theta_max = np.cos(velocity_angle_constraint)
            constraints.append(
                {"type": "ineq", "fun": lambda y: cos_theta_max - self._compute_cos_angle(y)}
            )
        else:
            constraints.append({"type": "eq", "fun": self.constraint_velocity_parallel})

        # 边界
        bounds = Bounds(
            lb=[self.alpha_range[0], self.transfer_time_range[0], self.t_ins_range[0]],
            ub=[self.alpha_range[1], self.transfer_time_range[1], self.t_ins_range[1]],
        )

        # 执行优化
        try:
            result = minimize(
                self.objective_function,
                y0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 1000, "disp": verbose},
            )

            success = result.success
            message = result.message if success else result.message
            final_y = result.x

        except Exception as e:
            success = False
            message = f"优化失败: {str(e)}"
            final_y = y0

        # 构建优化结果
        opt_vars = NLPOptimizationVariables.from_array(final_y)
        opt_result = self._build_result(
            opt_vars, success, message, use_relaxed_velocity_constraint, velocity_angle_constraint
        )

        if verbose:
            print(f"\n优化结果:")
            print(f"  成功: {opt_result.success}")
            print(f"  消息: {opt_result.message}")
            print(f"  α={opt_result.alpha:.6f}")
            print(f"  T={opt_result.transfer_time:.6f}")
            print(f"  t_ins={opt_result.t_ins:.6f}")
            print(f"  ΔV1={opt_result.delta_v1:.6f}")
            print(f"  ΔV2={opt_result.delta_v2:.6f}")
            print(f"  总ΔV={opt_result.objective_value:.6f}")

        return opt_result

    def _compute_cos_angle(self, y: np.ndarray) -> float:
        """计算速度夹角余弦"""
        alpha, transfer_time, t_ins = y

        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return -1.0

        final_state = states[-1]
        insertion_state = self.arrival_orbit.interpolate_at_time(t_ins)

        v_f = final_state[3:]
        v_ins = insertion_state[3:]

        v_f_norm = np.linalg.norm(v_f)
        v_ins_norm = np.linalg.norm(v_ins)

        if v_f_norm < 1e-10 or v_ins_norm < 1e-10:
            return -1.0

        return np.dot(v_f, v_ins) / (v_f_norm * v_ins_norm)

    def _build_result(
        self,
        variables: NLPOptimizationVariables,
        success: bool,
        message: str,
        use_relaxed_constraint: bool = False,
        velocity_angle_constraint: float = 0.0,
    ) -> NLPOptimizationResult:
        """构建优化结果对象"""
        alpha = variables.alpha
        transfer_time = variables.transfer_time
        t_ins = variables.t_ins

        # 计算轨迹
        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])
        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        # 获取各种状态
        insertion_state = self.arrival_orbit.interpolate_at_time(t_ins)
        final_state = states[-1] if len(states) > 0 else None

        # 计算ΔV
        dv1 = self.compute_delta_v1(self.departure_state, v_injection)
        dv2 = self.compute_delta_v2(
            final_state[3:] if final_state is not None else np.zeros(3), insertion_state[3:]
        )

        # 检查撞星
        earth_col, moon_col = self.check_collision(variables.to_array())

        # 确定转移类型(基于转移时间和特征)
        transfer_type = self._classify_transfer(transfer_time, times, states, insertion_state)

        # 约束违反量
        violation = {}
        if success:
            violation["position"] = self.constraint_position(variables.to_array())
            if use_relaxed_constraint:
                violation["velocity"] = max(
                    0,
                    np.cos(velocity_angle_constraint)
                    - self._compute_cos_angle(variables.to_array()),
                )
            else:
                violation["velocity"] = abs(self.constraint_velocity_parallel(variables.to_array()))

        return NLPOptimizationResult(
            alpha=alpha,
            transfer_time=transfer_time,
            t_ins=t_ins,
            objective_value=dv1 + dv2,
            delta_v1=dv1,
            delta_v2=dv2,
            transfer_trajectory=states,
            transfer_times=times,
            departure_state=self.departure_state.copy(),
            insertion_state=insertion_state,
            final_state=final_state,
            success=success,
            message=message,
            transfer_type=transfer_type,
            constraints_violation=violation,
        )

    def _classify_transfer(
        self,
        transfer_time: float,
        times: np.ndarray,
        states: np.ndarray,
        insertion_state: np.ndarray,
    ) -> TransferType:
        """分类转移类型

        基于轨迹特征分类:
        - 直接转移: 短时间(<20天), 近地点变化小
        - LGA转移: 中等时间,有近月点
        - 外部转移: 长时间,远地点很大
        """
        if len(states) == 0:
            return TransferType.DIRECT

        # 计算远地点
        x_max_traj = np.max(states[:, 0])

        # 直接转移: 转移时间短,轨迹紧凑
        if transfer_time < 20.0 and x_max_traj < 1.5:
            return TransferType.DIRECT

        # 外部转移: 远地点非常大
        if x_max_traj > 3.0:
            return TransferType.EXTERNAL

        # LGA转移: 其他情况
        return TransferType.LGA


def optimize_transfer(
    system: CR3BP_System,
    dynamics: CR3BP_Dynamics,
    departure_orbit: Orbit,
    arrival_orbit: Orbit,
    departure_state: np.ndarray,
    initial_guess: Optional[NLPOptimizationVariables] = None,
    **kwargs,
) -> NLPOptimizationResult:
    """便捷函数: 优化DRO到RO转移

    参数:
        system: CR3BP系统
        dynamics: CR3BP动力学
        departure_orbit: 出发点轨道
        arrival_orbit: 目标轨道
        departure_state: 出发点状态
        initial_guess: 初始猜测
        **kwargs: 其他优化参数

    返回:
        优化结果
    """
    optimizer = DROTRONLPOptimizer(
        system=system,
        dynamics=dynamics,
        departure_orbit=departure_orbit,
        arrival_orbit=arrival_orbit,
        departure_state=departure_state,
    )

    return optimizer.optimize(initial_guess=initial_guess, **kwargs)


if NlpCallbackBase is not None:

    class COPTNLPCallback(NlpCallbackBase):
        """COPT NLP回调类

        用于COPT非线性优化问题的目标函数和约束计算。
        继承自coptpy.NlpCallbackBase以正确处理SWIG绑定。

        属性:
            optimizer: DROTRONLPOptimizer实例
            x: 当前变量值 [alpha, transfer_time, t_ins]
        """

        def __init__(self, optimizer: DROTRONLPOptimizer):
            super().__init__()
            self.optimizer = optimizer
            self.x = None

        def EvalObj(self, xdata, outdata):
            """计算目标函数值 J(y) = Δv1 + Δv2"""
            x = np.array(xdata)
            self.x = x
            obj = self.optimizer.objective_function(x)
            outdata[0] = obj
            return 0

        def EvalGrad(self, xdata, outdata):
            """计算目标函数梯度 (数值差分)"""
            x = np.array(xdata)
            self.x = x
            h = 1e-8
            grad = np.zeros(3)
            f0 = self.optimizer.objective_function(x)
            for i in range(3):
                x_pert = x.copy()
                x_pert[i] += h
                grad[i] = (self.optimizer.objective_function(x_pert) - f0) / h
            for i in range(3):
                outdata[i] = grad[i]
            return 0

        def EvalCon(self, xdata, outdata):
            """计算约束函数值"""
            x = np.array(xdata)
            self.x = x

            pos_con = self.optimizer.constraint_position(x)
            vel_con = self.optimizer.constraint_velocity_parallel(x)

            outdata[0] = pos_con
            outdata[1] = vel_con
            return 0

        def EvalJac(self, xdata, outdata):
            """计算约束函数Jacobian矩阵 (数值差分)"""
            x = np.array(xdata)
            self.x = x
            h = 1e-8

            pos_con = self.optimizer.constraint_position(x)
            grad_pos = np.zeros(3)
            for i in range(3):
                x_pert = x.copy()
                x_pert[i] += h
                grad_pos[i] = (self.optimizer.constraint_position(x_pert) - pos_con) / h

            vel_con = self.optimizer.constraint_velocity_parallel(x)
            grad_vel = np.zeros(3)
            for i in range(3):
                x_pert = x.copy()
                x_pert[i] += h
                grad_vel[i] = (self.optimizer.constraint_velocity_parallel(x_pert) - vel_con) / h

            outdata[0] = grad_pos[0]
            outdata[1] = grad_pos[1]
            outdata[2] = grad_pos[2]
            outdata[3] = grad_vel[0]
            outdata[4] = grad_vel[1]
            outdata[5] = grad_vel[2]
            return 0

        def EvalHess(self, xdata, sigma, lam, outdata):
            """计算Hessian矩阵 (数值差分)"""
            x = np.array(xdata)
            self.x = x
            h = 1e-6

            grad_f = np.zeros(3)
            f0 = self.optimizer.objective_function(x)
            for i in range(3):
                x_pert = x.copy()
                x_pert[i] += h
                grad_f[i] = (self.optimizer.objective_function(x_pert) - f0) / h

            hess_f = np.zeros((3, 3))
            for i in range(3):
                for j in range(i + 1):
                    x_pert = x.copy()
                    x_pert[i] += h
                    x_pert[j] += h
                    f_ij = self.optimizer.objective_function(x_pert)

                    x_i = x.copy()
                    x_i[i] += h
                    f_i = self.optimizer.objective_function(x_i)

                    x_j = x.copy()
                    x_j[j] += h
                    f_j = self.optimizer.objective_function(x_j)

                    hess_f[i, j] = (f_ij - f_i - f_j + f0) / (h * h)
                    hess_f[j, i] = hess_f[i, j]

            idx = 0
            for i in range(3):
                for j in range(i + 1):
                    outdata[idx] = sigma * hess_f[i, j]
                    idx += 1
            return 0


class COPTNLPSolver:
    """基于COPT的NLP求解器封装

    使用COPT求解器的内点法求解非线性规划问题。

    属性:
        optimizer: DROTRONLPOptimizer实例
        model: COPT模型
        callback: COPTNLPCallback实例
    """

    def __init__(self, optimizer: DROTRONLPOptimizer, options: dict = None):
        """初始化COPT NLP求解器

        参数:
            optimizer: DROTRONLPOptimizer实例
            options: 求解器选项
        """
        self.optimizer = optimizer
        self.options = options or {}
        self.model = None
        self.callback = None

    def _setup_model(self, x0: np.ndarray) -> bool:
        """设置NLP模型

        参数:
            x0: 初始猜测

        返回:
            是否成功
        """
        if coptpy is None:
            raise RuntimeError("COPT not installed. Install with: pip install coptpy")

        self.coptpy = coptpy

        # 创建环境
        env = self.coptpy.Envr()
        self.model = env.createModel("DRO_RO_Transfer_NLP")

        # 设置变量边界
        alpha_lb, alpha_ub = self.optimizer.alpha_range
        t_lb, t_ub = self.optimizer.transfer_time_range
        tins_lb, tins_ub = self.optimizer.t_ins_range

        col_lower = [alpha_lb, t_lb, tins_lb]
        col_upper = [alpha_ub, t_ub, tins_ub]

        # 约束边界 (等式约束 = 0)
        row_lower = [0.0, 0.0]
        row_upper = [0.0, 0.0]

        # 创建NLP回调
        self.callback = COPTNLPCallback(self.optimizer)

        # Jacobian稀疏结构 (2个约束 x 3个变量)
        n_jac = 6
        idx_jac_row = [0, 0, 0, 1, 1, 1]  # 约束索引
        idx_jac_col = [0, 1, 2, 0, 1, 2]  # 变量索引

        # Hessian稀疏结构 (3x3下三角 = 6个元素)
        n_hess = 6
        idx_hess_row = [0, 1, 1, 2, 2, 2]
        idx_hess_col = [0, 0, 1, 0, 1, 2]

        # 加载NLP数据
        self.model.loadNlData(
            nCols=3,
            nRows=2,
            sense=self.coptpy.COPT.MINIMIZE,
            nGrad=3,
            idxGrad=[0, 1, 2],
            nJac=n_jac,
            idxJacRow=idx_jac_row,
            idxJacCol=idx_jac_col,
            nHess=n_hess,
            idxHessRow=idx_hess_row,
            idxHessCol=idx_hess_col,
            colLower=col_lower,
            colUpper=col_upper,
            rowLower=row_lower,
            rowUpper=row_upper,
            initX=list(x0),
            evalType=-1,  # 所有回调都已实现
            cb=self.callback,
        )

        # 设置参数
        self.model.setParam(self.coptpy.COPT.Param.NLPTol, 1e-10)
        self.model.setParam(self.coptpy.COPT.Param.NLPIterLimit, self.options.get("max_iter", 1000))

        return True

    def solve(self, x0: np.ndarray) -> dict:
        """求解NLP问题

        参数:
            x0: 初始猜测 [alpha, transfer_time, t_ins]

        返回:
            结果字典
        """
        if self.model is None:
            self._setup_model(x0)

        try:
            # 求解
            self.model.solve()

            # 获取状态
            status = self.model.status
            obj_val = self.model.objval if status == self.coptpy.COPT.OPTIMAL else float("inf")
            solution = self.model.x if hasattr(self.model, "x") else x0

            return {
                "status": status,
                "objective": obj_val,
                "solution": solution,
                "success": status == self.coptpy.COPT.OPTIMAL,
            }
        except Exception as e:
            return {
                "status": -1,
                "objective": float("inf"),
                "solution": x0,
                "success": False,
                "message": str(e),
            }

    def get_result(self) -> "NLPOptimizationResult":
        """获取优化结果

        返回:
            NLPOptimizationResult对象
        """
        if self.model is None or self.callback.x is None:
            raise RuntimeError("Must call solve() first")

        opt_vars = NLPOptimizationVariables.from_array(self.callback.x)
        success = self.model.status == self.coptpy.COPT.OPTIMAL

        return self.optimizer._build_result(
            opt_vars, success, "COPT solution" if success else f"COPT status: {self.model.status}"
        )


def optimize_with_copt(
    optimizer: DROTRONLPOptimizer,
    initial_guess: Optional[NLPOptimizationVariables] = None,
    **kwargs,
) -> NLPOptimizationResult:
    """使用COPT求解器进行优化

    参数:
        optimizer: DROTRONLPOptimizer实例
        initial_guess: 初始猜测
        **kwargs: 其他选项

    返回:
        优化结果
    """
    if initial_guess is None:
        alpha0, T0, tins0 = 1.0, 10.0, 5.0
    else:
        alpha0 = initial_guess.alpha
        T0 = initial_guess.transfer_time
        tins0 = initial_guess.t_ins

    x0 = np.array([alpha0, T0, tins0])

    try:
        solver = COPTNLPSolver(optimizer, kwargs)
        result = solver.solve(x0)

        if result["success"]:
            return solver.get_result()
        else:
            # 回退到scipy
            return optimizer.optimize(initial_guess=initial_guess, verbose=False)
    except Exception as e:
        # COPT不可用，回退到scipy
        return optimizer.optimize(initial_guess=initial_guess, verbose=False)
