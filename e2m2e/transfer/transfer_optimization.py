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
    import coptpy as cp
    from coptpy import COPT

    NlpCallbackBase = cp.NlpCallbackBase
    coptpy = cp  # 与既有 ``_HAVE_COPT`` 等检查兼容
except ImportError:
    cp = None
    COPT = None
    coptpy = None
    NlpCallbackBase = None


class TransferType(Enum):
    DIRECT = "direct"
    LGA = "lga"
    EXTERNAL = "external"


class NLPOptimizationVariables:
    """NLP优化变量

    优化变量: y = {α, T, t_ins}

    Attributes:
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

    Attributes:
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

    Attributes:
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

    DEFAULT_ALPHA_RANGE = (0.5, 2.5)
    DEFAULT_TRANSFER_TIME_RANGE = (1.0, 30.0)
    DEFAULT_T_INS_RANGE = (0.0, 10.0)

    EARTH_RADIUS_ND = 1.0 / 389703.0 * 6378.137
    MOON_RADIUS_ND = 1738.1 / 384400.0

    DEFAULT_VELOCITY_ANGLE_TOL = 1e-6

    def __init__(
        self,
        system: CR3BP_System,
        dynamics: CR3BP_Dynamics,
        departure_orbit: Orbit,
        arrival_orbit: Orbit,
        departure_state: np.ndarray,
    ):
        """初始化NLP优化器

        Args:
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

        self.alpha_range = self.DEFAULT_ALPHA_RANGE
        self.transfer_time_range = self.DEFAULT_TRANSFER_TIME_RANGE
        self.t_ins_range = self.DEFAULT_T_INS_RANGE

        self.velocity_angle_tol = self.DEFAULT_VELOCITY_ANGLE_TOL

        self.earth_radius = self.EARTH_RADIUS_ND
        self.moon_radius = self.MOON_RADIUS_ND

        self._last_trajectory: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def compute_departure_velocity(
        self, state: np.ndarray, alpha: float, beta: float = 0.0
    ) -> np.ndarray:
        """根据α计算出发速度

        Args:
            state: 出发点状态 [x, y, z, vx, vy, vz]
            alpha: 切向速度比

        Returns:
            注入速度向量 [vx, vy, vz]
        """
        pos = state[:3]
        vel = state[3:]

        normal = np.array([0.0, 0.0, 1.0])

        v_mag = np.linalg.norm(vel)
        if v_mag < 1e-10:
            warnings.warn("出发点速度接近零")
            return vel

        tangential = vel / v_mag

        normal_dir = np.cross(tangential, normal)
        norm_nd = np.linalg.norm(normal_dir)
        if norm_nd < 1e-10:
            normal_dir = np.array([1.0, 0.0, 0.0])
        else:
            normal_dir = normal_dir / norm_nd

        v_injection = alpha * v_mag * tangential + beta * v_mag * normal_dir

        return v_injection

    def forward_integrate(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前向积分转移弧

        Args:
            initial_state: 初始状态 [x, y, z, vx, vy, vz]
            t_span: 积分时间范围 (t0, tf)
            t_eval: 评估时间点

        Returns:
            (times, states): 时间序列和状态序列
        """
        if t_eval is None:
            n_steps = int((t_span[1] - t_span[0]) / 0.01) + 1
            t_eval = np.linspace(t_span[0], t_span[1], n_steps)

        result = self.dynamics.propagate(
            initial_state=initial_state,
            t_span=t_span,
            t_eval=t_eval,
            with_stm=False,
            with_jacobi=False,
        )

        times = result["time"]
        states = result["states"]

        self._last_trajectory = (times, states)

        return times, states

    def compute_delta_v1(self, departure_state: np.ndarray, initial_velocity: np.ndarray) -> float:
        """计算出发脉冲ΔV1

        Args:
            departure_state: 出发点状态 [x, y, z, vx, vy, vz]
            initial_velocity: 注入速度 [vx, vy, vz]

        Returns:
            ΔV1 大小
        """
        original_vel = departure_state[3:]
        dv1 = np.linalg.norm(initial_velocity - original_vel)
        return dv1

    def compute_delta_v2(self, final_velocity: np.ndarray, insertion_velocity: np.ndarray) -> float:
        """计算插入脉冲ΔV2

        Args:
            final_velocity: 转移轨迹末端速度
            insertion_velocity: 目标轨道插入点速度

        Returns:
            ΔV2 大小
        """
        dv2 = np.linalg.norm(final_velocity - insertion_velocity)
        return dv2

    def get_arrival_state_at_t_ins(self, t_ins: float) -> Tuple[np.ndarray, np.ndarray]:
        """获取目标轨道上 t_ins（绝对时间）对应的状态

        Args:
            t_ins: 绝对时间（与 orbit.times 同一坐标系）

        Returns:
            (position, velocity): 位置和速度
        """
        arrival_state = self.dynamics.propagate_orbit_state_at_time(
            self.arrival_orbit, float(t_ins)
        )
        return arrival_state[:3], arrival_state[3:]

    def objective_function(self, y: np.ndarray) -> float:
        """目标函数 J(y) = Δv1 + Δv2

        Args:
            y: 优化变量 [alpha, T, t_ins]

        Returns:
            总脉冲代价
        """
        alpha, transfer_time, t_ins = y

        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return 1e10

        final_state = states[-1]

        insertion_state = self.dynamics.propagate_orbit_state_at_time(
            self.arrival_orbit, float(t_ins)
        )

        dv1 = self.compute_delta_v1(self.departure_state, v_injection)
        dv2 = self.compute_delta_v2(final_state[3:], insertion_state[3:])

        return dv1 + dv2

    def constraint_position(self, y: np.ndarray) -> float:
        """位置连续性约束 Eq.(13)

        (x_f - x_ins)^2 + (y_f - y_ins)^2 + (z_f - z_ins)^2 = 0

        Args:
            y: 优化变量 [alpha, T, t_ins]

        Returns:
            约束违反量
        """
        alpha, transfer_time, t_ins = y

        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return 1e6

        final_state = states[-1]
        insertion_state = self.dynamics.propagate_orbit_state_at_time(
            self.arrival_orbit, float(t_ins)
        )

        pos_diff = final_state[:3] - insertion_state[:3]
        constraint = np.dot(pos_diff, pos_diff)

        return constraint

    def constraint_velocity_parallel(self, y: np.ndarray) -> float:
        """速度平行性约束 Eq.(14) 或relaxed Eq.(17)

        v_f · v_ins / (||v_f|| ||v_ins||) - 1 = 0

        Args:
            y: 优化变量 [alpha, T, t_ins]

        Returns:
            约束违反量
        """
        alpha, transfer_time, t_ins = y

        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return 1e6

        final_state = states[-1]
        insertion_state = self.dynamics.propagate_orbit_state_at_time(
            self.arrival_orbit, float(t_ins)
        )

        v_f = final_state[3:]
        v_ins = insertion_state[3:]

        v_f_norm = np.linalg.norm(v_f)
        v_ins_norm = np.linalg.norm(v_ins)

        if v_f_norm < 1e-10 or v_ins_norm < 1e-10:
            return 1e6

        cos_angle = np.dot(v_f, v_ins) / (v_f_norm * v_ins_norm)

        constraint = cos_angle - 1.0

        return constraint

    def check_collision(self, y: np.ndarray) -> Tuple[bool, bool]:
        """检查是否撞击地球或月球

        Args:
            y: 优化变量 [alpha, T, t_ins]

        Returns:
            (earth_collision, moon_collision): 是否撞击地球、月球
        """
        alpha, transfer_time, t_ins = y

        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])

        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        if len(states) == 0:
            return False, False

        earth_collision = False
        moon_collision = False

        for state in states:
            pos = state[:3]

            r_earth = np.sqrt((pos[0] + self.mu) ** 2 + pos[1] ** 2 + pos[2] ** 2)
            if r_earth < self.earth_radius:
                earth_collision = True

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

        Args:
            initial_guess: 初始猜测
            alpha_range: α范围
            transfer_time_range: 转移时间范围
            t_ins_range: 插入时间范围
            use_relaxed_velocity_constraint: 是否使用松弛速度约束
            velocity_angle_constraint: 松弛速度约束角度(弧度)
            verbose: 是否打印信息

        Returns:
            优化结果
        """
        if alpha_range is not None:
            self.alpha_range = alpha_range
        if transfer_time_range is not None:
            self.transfer_time_range = transfer_time_range
        if t_ins_range is not None:
            self.t_ins_range = t_ins_range

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

        constraints = []

        constraints.append({"type": "eq", "fun": self.constraint_position})

        if use_relaxed_velocity_constraint:
            cos_theta_max = np.cos(velocity_angle_constraint)
            constraints.append(
                {"type": "ineq", "fun": lambda y: cos_theta_max - self._compute_cos_angle(y)}
            )
        else:
            constraints.append({"type": "eq", "fun": self.constraint_velocity_parallel})

        bounds = Bounds(
            lb=[self.alpha_range[0], self.transfer_time_range[0], self.t_ins_range[0]],
            ub=[self.alpha_range[1], self.transfer_time_range[1], self.t_ins_range[1]],
        )

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
        insertion_state = self.dynamics.propagate_orbit_state_at_time(
            self.arrival_orbit, float(t_ins)
        )

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

        v_injection = self.compute_departure_velocity(self.departure_state, alpha)
        initial_state = np.concatenate([self.departure_state[:3], v_injection])
        times, states = self.forward_integrate(
            initial_state=initial_state, t_span=(0.0, transfer_time)
        )

        insertion_state = self.dynamics.propagate_orbit_state_at_time(
            self.arrival_orbit, float(t_ins)
        )
        final_state = states[-1] if len(states) > 0 else None

        dv1 = self.compute_delta_v1(self.departure_state, v_injection)
        dv2 = self.compute_delta_v2(
            final_state[3:] if final_state is not None else np.zeros(3), insertion_state[3:]
        )

        earth_col, moon_col = self.check_collision(variables.to_array())

        transfer_type = self._classify_transfer(transfer_time, times, states, insertion_state)

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

        x_max_traj = np.max(states[:, 0])

        if transfer_time < 20.0 and x_max_traj < 1.5:
            return TransferType.DIRECT

        if x_max_traj > 3.0:
            return TransferType.EXTERNAL

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

    Args:
        system: CR3BP系统
        dynamics: CR3BP动力学
        departure_orbit: 出发点轨道
        arrival_orbit: 目标轨道
        departure_state: 出发点状态
        initial_guess: 初始猜测
        **kwargs: 其他优化参数

    Returns:
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
        继承自 cp.NlpCallbackBase 以正确处理 SWIG 绑定。

        Attributes:
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
            """拉格朗日函数 Hessian 下三角（与 COPT 文档一致）。

            L(x) = σ·f(x) + λᵀc(x)，须返回 ∇²L 在 idxHess 指定位置的值；
            仅 σ·∇²f 而忽略约束曲率会导致错误步长/收敛行为。
            """
            x = np.asarray(xdata, dtype=np.float64).ravel()
            lam_v = np.asarray(lam, dtype=np.float64).ravel()
            if lam_v.size < 2:
                lam_v = np.pad(lam_v, (0, max(0, 2 - lam_v.size)))
            sigma = float(sigma)
            l0, l1 = float(lam_v[0]), float(lam_v[1])
            self.x = x
            h = 1e-6

            def lagrangian(xv: np.ndarray) -> float:
                fv = self.optimizer.objective_function(xv)
                c1 = self.optimizer.constraint_position(xv)
                c2 = self.optimizer.constraint_velocity_parallel(xv)
                return sigma * fv + l0 * c1 + l1 * c2

            L0 = lagrangian(x)
            hess_L = np.zeros((3, 3))
            for i in range(3):
                for j in range(i + 1):
                    x_ij = x.copy()
                    x_ij[i] += h
                    x_ij[j] += h
                    L_ij = lagrangian(x_ij)

                    x_i = x.copy()
                    x_i[i] += h
                    L_i = lagrangian(x_i)

                    x_j = x.copy()
                    x_j[j] += h
                    L_j = lagrangian(x_j)

                    hess_L[i, j] = (L_ij - L_i - L_j + L0) / (h * h)
                    hess_L[j, i] = hess_L[i, j]

            idx = 0
            for i in range(3):
                for j in range(i + 1):
                    outdata[idx] = hess_L[i, j]
                    idx += 1
            return 0

    def _apply_copt_nlp_params(model: Any, options: Dict[str, Any]) -> None:
        """与参考脚本一致：``model.setParam(COPT.Param.*, ...)``（NLP 项 + 可选 TimeLimit）。"""
        assert COPT is not None
        model.setParam(COPT.Param.NLPTol, 1e-10)
        model.setParam(COPT.Param.NLPIterLimit, int(options.get("max_iter", 1000)))
        model.setParam(COPT.Param.Threads, int(options.get("threads", 1)))
        model.setParam(COPT.Param.BarThreads, int(options.get("bar_threads", 1)))
        tl = options.get("time_limit", None)
        if tl is not None:
            model.setParam(COPT.Param.TimeLimit, float(tl))

    class COPTNLPSolver:
        """基于 COPT 的 NLP 封装：``cp.Envr()`` → ``createModel`` → ``loadNlData`` → ``solve``。"""

        def __init__(self, optimizer: DROTRONLPOptimizer, options: Optional[Dict[str, Any]] = None):
            self.optimizer = optimizer
            self.options = options or {}
            self.model = None
            self.callback = None

        def _setup_model(self, x0: np.ndarray) -> bool:
            if cp is None or COPT is None:
                raise RuntimeError("COPT not installed. Install with: pip install coptpy")

            env = cp.Envr()
            self.model = env.createModel("DRO_RO_Transfer_NLP")
            _apply_copt_nlp_params(self.model, self.options)

            alpha_lb, alpha_ub = self.optimizer.alpha_range
            t_lb, t_ub = self.optimizer.transfer_time_range
            tins_lb, tins_ub = self.optimizer.t_ins_range

            col_lower = [alpha_lb, t_lb, tins_lb]
            col_upper = [alpha_ub, t_ub, tins_ub]

            row_lower = [0.0, 0.0]
            row_upper = [0.0, 0.0]

            self.callback = COPTNLPCallback(self.optimizer)

            n_jac = 6
            idx_jac_row = [0, 0, 0, 1, 1, 1]
            idx_jac_col = [0, 1, 2, 0, 1, 2]

            n_hess = 6
            idx_hess_row = [0, 1, 1, 2, 2, 2]
            idx_hess_col = [0, 0, 1, 0, 1, 2]

            self.model.loadNlData(
                nCols=3,
                nRows=2,
                sense=COPT.MINIMIZE,
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
                evalType=-1,
                cb=self.callback,
            )

            _apply_copt_nlp_params(self.model, self.options)

            return True

        def solve(self, x0: np.ndarray) -> dict:
            if self.model is None:
                self._setup_model(x0)

            try:
                self.model.solve()

                status = self.model.status
                assert COPT is not None
                obj_val = self.model.objval if status == COPT.OPTIMAL else float("inf")
                solution = self.model.x if hasattr(self.model, "x") else x0

                return {
                    "status": status,
                    "objective": obj_val,
                    "solution": solution,
                    "success": status == COPT.OPTIMAL,
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
            if self.model is None or self.callback.x is None:
                raise RuntimeError("Must call solve() first")

            assert COPT is not None
            opt_vars = NLPOptimizationVariables.from_array(self.callback.x)
            success = self.model.status == COPT.OPTIMAL

            return self.optimizer._build_result(
                opt_vars,
                success,
                "COPT solution" if success else f"COPT status: {self.model.status}",
            )


else:
    COPTNLPSolver = None  # type: ignore[misc, assignment]

    def _apply_copt_nlp_params(model: Any, options: Dict[str, Any]) -> None:
        pass


def optimize_with_copt(
    optimizer: DROTRONLPOptimizer,
    initial_guess: Optional[NLPOptimizationVariables] = None,
    *,
    fallback_to_scipy: bool = True,
    max_iter: int = 1000,
    threads: int = 1,
    bar_threads: int = 1,
    time_limit: Optional[float] = None,
    scipy_fallback_kwargs: Optional[Dict[str, Any]] = None,
) -> NLPOptimizationResult:
    """使用 COPT 求解 NLP（与 ``data_processing_module`` 中用法一致：``cp.Envr`` / ``createModel`` / ``COPT.Param`` / ``solve``）。

    数学形式与 ``DROTRONLPOptimizer.optimize`` 相同（等式约束 + 最小化 Δv）。

    Args:
        optimizer: 已设置 ``alpha_range`` / ``transfer_time_range`` / ``t_ins_range`` 的 ``DROTRONLPOptimizer``
        initial_guess: 初始猜测 ``(α, T, t_ins)``；默认 ``(1, 10, 5)``
        fallback_to_scipy: 未安装 COPT 或求解失败时是否回退 SciPy SLSQP
        max_iter: ``COPT.Param.NLPIterLimit``
        threads / bar_threads: ``COPT.Param.Threads`` / ``BarThreads``（Python 回调建议为 1）
        time_limit: 若给定，则设置 ``COPT.Param.TimeLimit``（秒），与参考脚本中 MILP 用法一致
        scipy_fallback_kwargs: 回退时传给 ``optimizer.optimize`` 的额外参数

    Returns:
        ``NLPOptimizationResult``
    """
    if scipy_fallback_kwargs is None:
        scipy_fallback_kwargs = {}

    def _run_scipy() -> NLPOptimizationResult:
        return optimizer.optimize(initial_guess=initial_guess, **scipy_fallback_kwargs)

    if cp is None or COPT is None or NlpCallbackBase is None:
        if fallback_to_scipy:
            return _run_scipy()
        raise RuntimeError(
            "coptpy 未安装，无法使用 COPT；请安装 coptpy 或设置 fallback_to_scipy=True"
        )

    if initial_guess is None:
        alpha0, T0, tins0 = 1.0, 10.0, 5.0
    else:
        alpha0 = initial_guess.alpha
        T0 = initial_guess.transfer_time
        tins0 = initial_guess.t_ins

    x0 = np.array([alpha0, T0, tins0], dtype=float)

    copt_options: Dict[str, Any] = {
        "max_iter": max_iter,
        "threads": threads,
        "bar_threads": bar_threads,
    }
    if time_limit is not None:
        copt_options["time_limit"] = time_limit

    try:
        solver = COPTNLPSolver(optimizer, copt_options)
        result = solver.solve(x0)

        if result["success"]:
            return solver.get_result()
        if fallback_to_scipy:
            return _run_scipy()
        try:
            return solver.get_result()
        except RuntimeError:
            return optimizer._build_result(
                NLPOptimizationVariables.from_array(np.asarray(x0, dtype=float)),
                False,
                "COPT 未收敛且无可用解向量",
            )
    except Exception:
        if fallback_to_scipy:
            return _run_scipy()
        raise
