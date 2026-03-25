"""
轨道族延拓算法模块

提供自然参数延拓和伪弧长延拓方法，用于生成轨道族。
包括Halo轨道、Lyapunov轨道等的生成功能。
"""

from __future__ import annotations
import numpy as np
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
from .differential_correction import (
    DifferentialCorrection,
    compute_halo_initial_guess,
)
from ..core.orbit import OrbitFamily, Orbit


# =============================================================================
# 辅助函数：计算约束向量和雅可比矩阵
# =============================================================================


def compute_F_and_dF_symmetric_xz_plane(
    X: np.ndarray,
    SV0: np.ndarray,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算XZ平面对称轨道的约束向量和雅可比矩阵

    对应MATLAB: computeFdF_symPeriodicPlanes_CR3BP(X, SV0i, mu, plane=13)

    X = [rx; rz; vy; tf2] - 自由变量向量
    SV0 = [rx, ry, rz, vx, vy, vz] - 初始状态向量

    F = [vx; vz; ry] - 约束向量（半周期终点状态）
    dF = ∂F/∂X - 约束雅可比矩阵 (3x4)

    参数:
        X: 自由变量向量 [rx, rz, vy, tf2]
        SV0: 初始状态向量 [x, y, z, vx, vy, vz]
        mu: 质量比

    返回:
        F: 约束向量 [vx, vz, ry]
        dF: 雅可比矩阵 (3, 4)
    """
    # 重建状态和飞行时间
    rx = X[0]
    rz = X[1]
    vy = X[2]
    tf2 = X[3]

    # 更新初始状态
    state = SV0.copy()
    state[0] = rx  # x
    state[2] = rz  # z
    state[4] = vy  # vy

    # 计算状态导数（用于雅可比矩阵的时间项）
    def state_derivative(t, s, mu_val):
        """CR3BP状态导数"""
        x, y, z, vx, vy, vz = s
        r1 = np.sqrt((x + mu_val) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu_val) ** 2 + y**2 + z**2)

        ax = 2 * vy + x - (1 - mu_val) * (x + mu_val) / r1**3 - mu_val * (x - 1 + mu_val) / r2**3
        ay = -2 * vx + y - (1 - mu_val) * y / r1**3 - mu_val * y / r2**3
        az = -(1 - mu_val) * z / r1**3 - mu_val * z / r2**3

        return np.array([vx, vy, vz, ax, ay, az])

    # 计算雅可比矩阵 A(t)
    def compute_jacobian_A(s, mu_val):
        """计算CR3BP雅可比矩阵A(t)"""
        x, y, z, vx, vy, vz = s
        r1 = np.sqrt((x + mu_val) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu_val) ** 2 + y**2 + z**2)

        U_xx = 1 - (1 - mu_val) * (1 / r1**3 - 3 * (x + mu_val) ** 2 / r1**5) - mu_val * (1 / r2**3 - 3 * (x - 1 + mu_val) ** 2 / r2**5)
        U_yy = 1 - (1 - mu_val) * (1 / r1**3 - 3 * y**2 / r1**5) - mu_val * (1 / r2**3 - 3 * y**2 / r2**5)
        U_zz = -(1 - mu_val) / r1**3 - mu_val / r2**3
        U_xy = 3 * (1 - mu_val) * (x + mu_val) * y / r1**5 + 3 * mu_val * (x - 1 + mu_val) * y / r2**5
        U_xz = 3 * (1 - mu_val) * (x + mu_val) * z / r1**5 + 3 * mu_val * (x - 1 + mu_val) * z / r2**5
        U_yz = 3 * (1 - mu_val) * y * z / r1**5 + 3 * mu_val * y * z / r2**5

        return np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [U_xx, U_xy, U_xz, 0, 2, 0],
            [U_xy, U_yy, U_yz, -2, 0, 0],
            [U_xz, U_yz, U_zz, 0, 0, 0],
        ])

    # 积分得到终点状态和STM
    from scipy.integrate import solve_ivp

    # 增广状态：[状态(6), STM(36)]
    initial_stm = np.eye(6).flatten()
    augmented_state = np.concatenate([state, initial_stm])

    def equations_with_stm(t, aug_s):
        """增广状态方程"""
        s = aug_s[:6]
        stm = aug_s[6:].reshape((6, 6))
        dsdt = state_derivative(t, s, mu)
        A = compute_jacobian_A(s, mu)
        stm_dot = A @ stm
        return np.concatenate([dsdt, stm_dot.flatten()])

    # 积分半周期
    result = solve_ivp(
        equations_with_stm,
        (0, tf2),
        augmented_state,
        method="DOP853",
        t_eval=[tf2],
        rtol=1e-12,
        atol=1e-12,
    )

    # 提取终点状态和STM
    final_state = result.y[:6, -1]
    final_stm = result.y[6:, -1].reshape((6, 6))

    # 计算终点状态导数
    dSV = state_derivative(tf2, final_state, mu)

    # 约束向量 F = [vx; vz; ry] = [final_state[3]; final_state[5]; final_state[1]]
    F = np.array([final_state[3], final_state[5], final_state[1]])

    # 雅可比矩阵 dF = ∂F/∂X
    # 对于 X = [rx, rz, vy, tf2]，dF是3x4矩阵
    # dF(i,j) = ∂F(i)/∂X(j)
    # 列1: ∂F/∂rx = STM([3,5,1], 0)  (对x的敏感性)
    # 列2: ∂F/∂rz = STM([3,5,1], 2)  (对z的敏感性)
    # 列3: ∂F/∂vy = STM([3,5,1], 4)  (对vy的敏感性)
    # 列4: ∂F/∂tf2 = dSV([3,5,1])  (对时间的敏感性)

    dF = np.zeros((3, 4))
    # 列0: 对rx的偏导 (STM第0列)
    dF[0, 0] = final_stm[3, 0]  # ∂vx/∂rx
    dF[1, 0] = final_stm[5, 0]  # ∂vz/∂rx
    dF[2, 0] = final_stm[1, 0]  # ∂ry/∂rx

    # 列1: 对rz的偏导 (STM第2列)
    dF[0, 1] = final_stm[3, 2]  # ∂vx/∂rz
    dF[1, 1] = final_stm[5, 2]  # ∂vz/∂rz
    dF[2, 1] = final_stm[1, 2]  # ∂ry/∂rz

    # 列2: 对vy的偏导 (STM第4列)
    dF[0, 2] = final_stm[3, 4]  # ∂vx/∂vy
    dF[1, 2] = final_stm[5, 4]  # ∂vz/∂vy
    dF[2, 2] = final_stm[1, 4]  # ∂ry/∂vy

    # 列3: 对tf2的偏导 (状态导数)
    dF[0, 3] = dSV[3]  # ∂vx/∂t
    dF[1, 3] = dSV[5]  # ∂vz/∂t
    dF[2, 3] = dSV[1]  # ∂ry/∂t

    return F, dF


def compute_tangent_vector(dF: np.ndarray) -> np.ndarray:
    """计算切向量（约束雅可比矩阵的零空间）

    对应MATLAB: Xdot = null(dF)

    参数:
        dF: 约束雅可比矩阵 (3, 4)

    返回:
        Xdot: 切向量 (4,)，单位化
    """
    # 使用SVD计算零空间
    _, _, Vh = np.linalg.svd(dF)
    # 零空间向量是V的最后一列（对应最小奇异值）
    tangent = Vh[-1, :]
    # 单位化
    norm = np.linalg.norm(tangent)
    if norm > 0:
        tangent = tangent / norm
    return tangent


class ContinuationMethod(Enum):
    """延拓方法枚举"""

    NATURAL = "natural"
    PSEUDO_ARCLENGTH = "pseudo_arclength"


class Continuation:
    """轨道族延拓

    通过延拓算法生成一族周期轨道，支持自然参数延拓和伪弧长延拓。

    属性：
        corrector: DifferentialCorrection对象
        continuation_parameter: 延拓参数名称
        step_size: 延拓步长
        family_orbits: 轨道族列表
    """

    # 类属性
    DEFAULT_STEP_SIZE = 0.01
    MIN_STEP_SIZE = 1e-6
    MAX_STEP_SIZE = 0.1
    DEFAULT_PREDICTOR_ORDER = 1

    def __init__(
        self,
        corrector: DifferentialCorrection,
        step: Optional[float] = None,
    ) -> None:
        """初始化延拓器

        参数：
        - corrector: DifferentialCorrection对象
        - param: 延拓参数（如 "energy", "period", "amplitude", "x0", "z0"）
        - step: 初始步长
        """
        self.correction = corrector
        self.dynamics = corrector.dynamics if corrector.dynamics is not None else None

        # 延拓参数
        if corrector.fixed_parameters:
            self.continuation_parameter = next(iter(corrector.fixed_parameters))
        else:
            self.continuation_parameter = None
        self.step_size = step or self.DEFAULT_STEP_SIZE
        self.initial_step_size = self.step_size
        self.method = ContinuationMethod.NATURAL

        # 轨道族
        self.family_orbits = []
        self.family_parameters = []
        self.family_states = []  # 初始状态列表
        self.family_periods = []  # 周期列表

        # 当前/历史轨道
        self.current_orbit = None
        self.current_parameter = None
        self.previous_orbit = None
        self.previous_parameter = None

        # 预测器
        self.predictor_order = self.DEFAULT_PREDICTOR_ORDER
        self.tangent_vector = None

        # 统计
        self.continuation_stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
        }

        # 步长控制
        self.step_reduction_factor = 0.5  # 步长缩减因子
        self.step_increase_factor = 1.2  # 步长增大因子
        self.min_step_size = 1e-5  # 最小步长
        self.max_step_size = 0.1  # 最大步长

        # 终止条件
        self.max_orbits = 100
        self.termination_reason = None

    def natural_continuation(
        self,
        seed_orbit,
        param_range,
        step_size,
        verbose=False,
    ):
        """自然参数延拓

        从种子轨道出发，逐步改变延拓参数，生成一族周期轨道。
        支持双向延拓：如果param_range的最小值小于种子轨道参数值，则向小值方向延拓；
        如果param_range的最大值大于种子轨道参数值，则向大值方向延拓。

        参数：
            seed_orbit: Orbit, 种子轨道
            param_range: tuple, 延拓参数范围 (param_min, param_max)
            step_size: float, 步长（始终为正值，延拓方向由参数范围自动确定）
            verbose: 是否打印信息

        返回：
            OrbitFamily: 包含轨道族的OrbitFamily对象
        """
        # 验证并限制步长
        if step_size > self.max_step_size:
            if verbose:
                print(
                    f"警告: 输入步长 {step_size} 超过最大限制 {self.max_step_size}，限制为 {self.max_step_size}"
                )
            step_size = self.max_step_size

        if step_size < self.min_step_size:
            if verbose:
                print(
                    f"警告: 输入步长 {step_size} 小于最小限制 {self.min_step_size}，限制为 {self.min_step_size}"
                )
            step_size = self.min_step_size

        # 创建轨道族对象
        orbit_family = OrbitFamily(seed_orbit)

        # 获取延拓参数索引
        param_index = self._infer_param_index()

        # 获取种子轨道的延拓参数值
        if param_index < 6:
            seed_param_value = seed_orbit.states[0, param_index]
        else:
            seed_param_value = seed_orbit.period

        param_min, param_max = param_range

        # 确定延拓方向
        forward = param_max > seed_param_value  # 向大值方向延拓
        backward = param_min < seed_param_value  # 向小值方向延拓

        print(f"\n{'=' * 60}")
        print(f"开始自然参数延拓 (参数: {self.continuation_parameter})")
        print(f"种子轨道参数值: {seed_param_value:.6f}")
        print(f"参数范围: {param_range}")
        print(f"延拓方向: {'正向' if forward else ''}{'反向' if backward else ''}")
        print(f"步长: {step_size}")
        print(f"{'=' * 60}")

        # 复用已初始化的corrector，确保修正模式（如setup_2D_symmetric_x_fixed_x0）被正确使用
        corrector = self.correction

        # 初始化当前轨道（使用种子轨道作为初始状态）
        current_orbit = seed_orbit.copy()

        # 临时存储轨道和对应步数（用于后续排序）
        # 步数定义：从种子轨道出发，正向延拓 step 1, 2, 3...，反向延拓 -1, -2, -3...
        temp_orbits_with_steps = []

        # 步长历史记录
        step_size_history = []

        # 执行正向延拓（向大值方向）
        if forward:
            if verbose:
                print("\n--- 正向延拓 (参数增大方向) ---")

            target_forward = param_max
            i = 0
            while True:
                # 获取当前参数值
                if param_index < 6:
                    current_param_value = current_orbit.states[0, param_index]
                else:
                    current_param_value = current_orbit.period

                # 检查是否到达目标
                if current_param_value >= target_forward:
                    break

                # 生成待修正的下一个延拓位置的初始猜测
                if param_index < 6:
                    # 修改状态分量
                    guess_orbit = current_orbit.copy()
                    guess_orbit.states[0, param_index] += step_size
                    orbit = corrector.iterate_correction(guess_orbit, verbose=verbose)
                else:
                    # 修改时间 - 通过复制并修改orbit的period属性
                    guess_orbit = current_orbit.copy()
                    guess_orbit.period = current_orbit.period + step_size * 2
                    orbit = corrector.iterate_correction(guess_orbit, verbose=verbose)

                if orbit is not None and orbit.correction_success:
                    # 临时存储轨道和步数（步数为正向 i+1）
                    orbit.metadata["continuation_step"] = i + 1
                    temp_orbits_with_steps.append((orbit, i + 1))

                    # 更新当前轨道
                    current_orbit = orbit

                    self.continuation_stats["successful_steps"] += 1

                    # 打印进度信息
                    if (i + 1) % 10 == 0:
                        if verbose:
                            print(
                                f"  第 {i + 1} 条轨道，参数值={orbit.states[0, param_index] if param_index < 6 else orbit.period:.6f}，周期={orbit.period:.4f}"
                            )
                        else:
                            print(f"  正向延拓进度：已完成 {i + 1} 条轨道")

                    # 自适应步长
                    if hasattr(self, "step_size_adaptation") and self.step_size_adaptation:
                        if orbit.correction_iterations < 3:
                            step_size = min(step_size * self.step_growth_factor, self.max_step_size)
                        elif orbit.correction_iterations > 8:
                            step_size = max(
                                step_size * self.step_reduction_factor, self.min_step_size
                            )
                else:
                    self.continuation_stats["failed_steps"] += 1

                    # 步长减半重试
                    step_size *= self.step_reduction_factor
                    if step_size < self.min_step_size:
                        self.termination_reason = "步长过小，延拓终止"
                        if verbose:
                            print(f"\n正向步长过小，延拓终止于第 {len(orbit_family)} 条轨道")
                        break

                    if verbose:
                        print(f"  第 {i + 1} 步修正失败，减小步长至 {step_size:.6f}")

                step_size_history.append(step_size)
                i += 1

        # 执行反向延拓（向小值方向）
        if backward:
            # 重置当前轨道为种子轨道
            current_orbit = seed_orbit.copy()

            if verbose:
                print("\n--- 反向延拓 (参数减小方向) ---")

            target_backward = param_min
            i = 0
            while True:
                # 获取当前参数值
                if param_index < 6:
                    current_param_value = current_orbit.states[0, param_index]
                else:
                    current_param_value = current_orbit.period

                # 检查是否到达目标
                if current_param_value <= target_backward:
                    break

                # 生成待修正的下一个延拓位置的初始猜测（减小参数值）
                if param_index < 6:
                    # 修改状态分量
                    guess_orbit = current_orbit.copy()
                    guess_orbit.states[0, param_index] -= step_size
                    orbit = corrector.iterate_correction(guess_orbit, verbose=verbose)
                else:
                    # 修改时间 - 通过复制并修改orbit的period属性
                    guess_orbit = current_orbit.copy()
                    guess_orbit.period = current_orbit.period - step_size * 2
                    orbit = corrector.iterate_correction(guess_orbit, verbose=verbose)

                if orbit is not None and orbit.correction_success:
                    # 添加到轨道族（步数为反向 -i-1）
                    orbit.metadata["continuation_step"] = -(i + 1)
                    temp_orbits_with_steps.append((orbit, -(i + 1)))

                    # 更新当前轨道
                    current_orbit = orbit

                    self.continuation_stats["successful_steps"] += 1

                    # 打印进度信息
                    if (i + 1) % 10 == 0:
                        if verbose:
                            print(
                                f"  第 {i + 1} 条轨道，参数值={orbit.states[0, param_index] if param_index < 6 else orbit.period:.6f}，周期={orbit.period:.4f}"
                            )
                        else:
                            print(f"  反向延拓进度：已完成 {i + 1} 条轨道")

                    # 自适应步长
                    if hasattr(self, "step_size_adaptation") and self.step_size_adaptation:
                        if orbit.correction_iterations < 3:
                            step_size = min(step_size * self.step_growth_factor, self.max_step_size)
                        elif orbit.correction_iterations > 8:
                            step_size = max(
                                step_size * self.step_reduction_factor, self.min_step_size
                            )
                else:
                    self.continuation_stats["failed_steps"] += 1

                    # 步长减半重试
                    step_size *= self.step_reduction_factor
                    if step_size < self.min_step_size:
                        self.termination_reason = "步长过小，延拓终止"
                        if verbose:
                            print(f"\n反向步长过小，延拓终止于第 {len(orbit_family)} 条轨道")
                        break

                    if verbose:
                        print(f"  第 {i + 1} 步修正失败，减小步长至 {step_size:.6f}")

                step_size_history.append(step_size)
                i += 1

        # 按步数排序：种子轨道(step=0)在最前，然后按绝对值从小到大排列
        # 排序顺序：0, 1, -1, 2, -2, 3, -3, ...
        seed_orbit.metadata["continuation_step"] = 0
        all_orbits_with_steps = [(seed_orbit, 0)] + temp_orbits_with_steps

        # 自定义排序：先按绝对值排序，再按正负排序（正数在前）
        def sort_key(item):
            orbit, step = item
            return (abs(step), step > 0)

        all_orbits_with_steps.sort(key=sort_key)

        # 清空并重新按排序顺序添加轨道
        orbit_family.orbits = []
        for orbit, step in all_orbits_with_steps:
            orbit_family.add_orbit(orbit)

        if verbose:
            print(f"\n延拓完成：共生成 {len(orbit_family)} 条轨道")
            stats = self.continuation_stats
            print(f"  成功: {stats['successful_steps']}, 失败: {stats['failed_steps']}")
            print("  轨道已按距离种子轨道的步数排序: 0, 1, -1, 2, -2, ...")

        return orbit_family

    def pseudo_arclength_continuation(
        self,
        seed_orbit,
        n_orbits: int = 50,
        step_size: float = 0.005,
        direction: str = "positive",
        verbose: bool = True,
        TolPAL: float = 1e-6,
        TolDiffCorr: float = 1e-6,
        IterMax: int = 100,
    ):
        """伪弧长延拓（基于MATLAB continuation_PAL_CR3BP）

        使用伪弧长参数化方法跟踪轨道族，可以处理折返点等复杂情况。

        参数：
            seed_orbit: 种子轨道（Orbit对象）
            n_orbits: 目标轨道数量
            step_size: 伪弧长步长 DeltaS
            direction: 延拓方向 ("positive", "negative", "both")
            verbose: 是否打印信息
            TolPAL: PAL Newton-Raphson 容差
            TolDiffCorr: 微分修正容差
            IterMax: PAL循环最大迭代次数

        返回：
            OrbitFamily: 包含轨道族的OrbitFamily对象
        """
        mu = self.dynamics.system.mu

        if verbose:
            print(f"\n{'=' * 60}")
            print("开始伪弧长延拓 (Pseudo Arc-Length Continuation)")
            print(f"{'=' * 60}")
            print(f"  质量比 mu = {mu:.10f}")
            print(f"  目标轨道数 = {n_orbits}")
            print(f"  步长 DeltaS = {step_size}")
            print(f"  延拓方向 = {direction}")

        orbit_family = OrbitFamily(seed_orbit)

        # 种子轨道初始状态
        SV0i = seed_orbit.states[0].copy()
        tfi = seed_orbit.period

        # 自由变量向量 X = [rx, rz, vy, tf2]
        X = np.array([SV0i[0], SV0i[2], SV0i[4], tfi / 2])

        # 计算初始约束雅可比
        _, dF = compute_F_and_dF_symmetric_xz_plane(X, SV0i, mu)

        # 计算切向量（零空间）
        Xdot = compute_tangent_vector(dF)

        if verbose:
            print(f"\n初始自由变量 X = [{X[0]:.6f}, {X[1]:.6f}, {X[2]:.6f}, {X[3]:.6f}]")
            print(f"初始切向量 Xdot = [{Xdot[0]:.6f}, {Xdot[1]:.6f}, {Xdot[2]:.6f}, {Xdot[3]:.6f}]")

        # 存储已修正的轨道状态和周期
        family_states = [SV0i.copy()]
        family_periods = [tfi]
        family_orbits = [seed_orbit]

        # 确定延拓方向
        step_sign = 1.0 if direction == "positive" else -1.0

        for n in range(n_orbits):
            if verbose and (n + 1) % 5 == 0:
                print(f"\n--- 延拓第 {n + 1}/{n_orbits} 条轨道 ---")

            # ============================================================
            # 预测步 (Predictor)
            # ============================================================
            # Xnew = X + DeltaS * Xdot
            DeltaX = step_sign * step_size * Xdot
            Xnew = X + DeltaX

            if verbose:
                print(f"  预测 Xnew = [{Xnew[0]:.6f}, {Xnew[1]:.6f}, {Xnew[2]:.6f}, {Xnew[3]:.6f}]")

            # ============================================================
            # PAL修正步 (Corrector using PAL constraint)
            # ============================================================
            # 组合约束: G = [F; dot(Xnew-X, Xdot) - DeltaS]
            # 组合雅可比: dG = [dF; Xdot']

            for iter_pal in range(IterMax):
                # 重建状态
                SV0_guess = SV0i.copy()
                SV0_guess[0] = Xnew[0]  # rx
                SV0_guess[2] = Xnew[1]  # rz
                SV0_guess[4] = Xnew[2]  # vy
                tf2_guess = Xnew[3]

                # 计算约束和雅可比
                F, dF_new = compute_F_and_dF_symmetric_xz_plane(Xnew, SV0_guess, mu)

                # 计算当前切向量
                Xdot_new = compute_tangent_vector(dF_new)

                # 组合约束 G = [F; dot(Xnew-X, Xdot) - DeltaS]
                G = np.zeros(4)
                G[:3] = F
                G[3] = np.dot(Xnew - X, Xdot) - step_size * step_sign

                # 组合雅可比 dG = [dF; Xdot']
                dG = np.zeros((4, 4))
                dG[:3, :] = dF_new
                dG[3, :] = Xdot

                # Newton-Raphson 修正
                try:
                    delta_X = np.linalg.solve(dG, G)
                except np.linalg.LinAlgError:
                    if verbose:
                        print(f"  PAL迭代 {iter_pal + 1}: 雅可比矩阵奇异")
                    break

                Xnew = Xnew - delta_X

                # 检查收敛
                if np.linalg.norm(F) < TolPAL:
                    if verbose:
                        print(f"  PAL迭代 {iter_pal + 1}: 收敛, ||F|| = {np.linalg.norm(F):.2e}")
                    break

            # 更新切向量
            Xdot = Xdot_new
            X = Xnew.copy()

            # ============================================================
            # 微分修正 (Differential Correction)
            # ============================================================
            # 重建状态向量
            SV0_corr = SV0i.copy()
            SV0_corr[0] = X[0]  # rx
            SV0_corr[2] = X[1]  # rz
            SV0_corr[4] = X[2]  # vy
            tf_corr = 2 * X[3]

            # 选择修正方法：根据 x 和 z 哪个变化更大来决定
            x0_last = family_states[-1][0] if len(family_states) > 0 else SV0_corr[0]
            z0_last = family_states[-1][2] if len(family_states) > 0 else SV0_corr[2]

            if abs(SV0_corr[0] - x0_last) > abs(SV0_corr[2] - z0_last):
                # 使用 fixedX 修正
                self.correction.setup_3D_symmetric_x_fixed_x0(x0=SV0_corr[0])
            else:
                # 使用 fixedZ 修正
                self.correction.setup_3D_symmetric_xz_fixed_z0(z0=SV0_corr[2])

            # 创建猜测轨道
            guess_orbit = Orbit(
                states=SV0_corr.reshape(1, -1),
                times=np.array([0.0]),
                system=self.dynamics.system,
            )
            guess_orbit.period = tf_corr

            # 执行微分修正
            orbit = self.correction.iterate_correction(guess_orbit, verbose=False)

            if orbit is not None and orbit.correction_success:
                family_orbits.append(orbit)
                family_states.append(orbit.states[0].copy())
                family_periods.append(orbit.period)

                # 更新自由变量
                X = np.array([orbit.states[0, 0], orbit.states[0, 2], orbit.states[0, 4], orbit.period / 2])

                if verbose and (n + 1) % 5 == 0:
                    print(f"  轨道 {n + 1}: x0={orbit.states[0, 0]:.4f}, z0={orbit.states[0, 2]:.4f}, T={orbit.period:.4f}")
            else:
                if verbose:
                    print(f"  轨道 {n + 1}: 微分修正失败")
                break

        # 添加所有轨道到族
        for orbit in family_orbits:
            orbit_family.add_orbit(orbit)

        if verbose:
            print(f"\n伪弧长延拓完成：共生成 {len(orbit_family)} 条轨道")

        return orbit_family

    def _infer_param_index(self):
        """根据延拓参数名称推断索引"""
        param_map = {
            "x0": 0,
            "y0": 1,
            "z0": 2,
            "vx0": 3,
            "vy0": 4,
            "vz0": 5,
            "period": 6,
            "energy": 6,
            "amplitude": 2,
        }
        return param_map.get(self.continuation_parameter, 0)

    def _build_family_result(self):
        """构建轨道族结果字典"""
        return {
            "orbits": self.family_orbits,
            "states": np.array(self.family_states),
            "periods": np.array(self.family_periods),
            "n_orbits": len(self.family_orbits),
            "stats": self.continuation_stats,
            "termination_reason": self.termination_reason,
        }

    # =============================================================================
    # Halo轨道生成方法
    # =============================================================================

    def generate_halo_seed_orbit(
        self,
        libration_point: int,
        amplitude_z: float,
        halo_class: int = 0,
        verbose: bool = False,
    ) -> Orbit:
        """生成Halo轨道初始猜测并修正

        使用Richardson三阶近似生成初始猜测，然后通过微分修正
        得到精确的周期轨道。

        参数：
            libration_point: 拉格朗日点 (1=L1, 2=L2)
            amplitude_z: Z方向振幅
            halo_class: 0=北Halo, 1=南Halo
            verbose: 是否打印详细信息

        返回：
            Orbit: Halo周期轨道
        """
        if libration_point not in [1, 2]:
            raise ValueError(f"libration_point必须是1或2，当前为{libration_point}")
        if amplitude_z <= 0:
            raise ValueError(f"amplitude_z必须为正数，当前为{amplitude_z}")
        if halo_class not in [0, 1]:
            raise ValueError(f"halo_class必须是0或1，当前为{halo_class}")

        if verbose:
            print(f"\n生成Halo轨道: L{libration_point} {'北' if halo_class == 0 else '南'} Halo")
            print(f"  Z振幅: {amplitude_z}")

        mu = self.dynamics.system.mu

        guess = compute_halo_initial_guess(
            mu=mu,
            z_amplitude=amplitude_z,
            L=libration_point,
            halo_class=halo_class,
        )

        if halo_class == 0:
            initial_z = amplitude_z
        else:
            initial_z = -amplitude_z

        initial_state = np.array(
            [
                guess["x0"],
                0.0,
                initial_z,
                guess["vx0"],
                guess["vy0"],
                guess["vz0"],
            ]
        )

        if halo_class == 0:
            self.correction.setup_halo_orbit_fixed_z0(
                z0=amplitude_z,
                libration_point=libration_point,
            )
        else:
            self.correction.setup_halo_orbit_fixed_z0(
                z0=-amplitude_z,
                libration_point=libration_point,
            )

        initial_orbit = Orbit(
            states=initial_state.reshape(1, -1),
            times=np.array([0.0]),
            system=self.dynamics.system,
        )
        initial_orbit.period = 2.0 * guess["T_half"]

        self.correction.max_iterations = 150
        self.correction.tolerance = 1e-5

        if verbose:
            print(f"  初始猜测: x0={guess['x0']:.6f}, vy0={guess['vy0']:.6f}")
            print(f"  预估周期: {initial_orbit.period:.4f} TU")

        orbit = self.correction.iterate_correction(
            initial_guess=initial_orbit,
            verbose=verbose,
        )

        if orbit is not None:
            orbit.family_type = "halo"
            orbit.parameters["libration_point"] = libration_point
            orbit.parameters["amplitude_z"] = amplitude_z
            orbit.parameters["halo_class"] = halo_class
            if verbose:
                print(f"[ok] Halo轨道生成成功: 周期={orbit.period:.6f} TU")

        return orbit

    def generate_halo_family(
        self,
        seed_orbit: Orbit,
        n_orbits: int = 50,
        direction: str = "positive",
        step_size: float = 0.001,
    ) -> List[Orbit]:
        """生成Halo轨道族

        使用自然参数延拓法生成Halo轨道族。

        参数：
            seed_orbit: 种子轨道
            n_orbits: 目标轨道数量
            direction: 延拓方向 ("positive", "negative", "both")
            step_size: 步长

        返回：
            List[Orbit]: Halo轨道族
        """
        if n_orbits < 1:
            raise ValueError(f"n_orbits必须大于0，当前为{n_orbits}")
        if direction not in ["positive", "negative", "both"]:
            raise ValueError(f"direction必须是positive/negative/both，当前为{direction}")

        family = [seed_orbit]

        directions = ["positive", "negative"] if direction == "both" else [direction]

        print(f"\n开始生成Halo轨道族: 目标数量={n_orbits}, 方向={direction}")
        print(
            f"  种子轨道: L{seed_orbit.parameters.get('libration_point', 1)} "
            f"{'北' if seed_orbit.parameters.get('halo_class', 0) == 0 else '南'} Halo"
        )

        for direction in directions:
            z_amplitude = seed_orbit.parameters.get("amplitude_z", 0.1)
            step = step_size if direction == "positive" else -step_size

            for i in range(n_orbits - 1):
                new_z = z_amplitude + step * (i + 1)
                if new_z <= 0:
                    break

                try:
                    new_orbit = self.generate_halo_seed_orbit(
                        libration_point=seed_orbit.parameters.get("libration_point", 1),
                        amplitude_z=new_z,
                        halo_class=seed_orbit.parameters.get("halo_class", 0),
                        verbose=False,
                    )
                    if new_orbit is not None:
                        family.append(new_orbit)
                except Exception:
                    break

        print(f"[ok] 轨道族生成完成: 共{len(family)}条轨道")

        return family

    def halo_pseudo_arclength_continuation(
        self,
        seed_orbit: Orbit,
        n_orbits: int = 50,
        direction: str = "both",
        step_size: float = 0.005,
        verbose: bool = True,
    ) -> OrbitFamily:
        """Halo轨道伪弧长延拓

        使用伪弧长参数化方法生成Halo轨道族。
        该方法利用Richardson三阶近似计算初始猜测，
        并使用轨道族切线方向进行预测，以更好地跟踪轨道族曲线。

        参数：
            seed_orbit: Orbit, 种子轨道（已修正的Halo轨道）
            n_orbits: int, 目标轨道数量
            direction: str, 延拓方向 ("positive", "negative", "both")
            step_size: float, 延拓步长（弧长）
            verbose: bool, 是否打印详细信息

        返回：
            OrbitFamily: 包含轨道族的OrbitFamily对象
        """
        mu = self.dynamics.system.mu
        libration_point = seed_orbit.parameters.get("libration_point", 1)
        halo_class = seed_orbit.parameters.get("halo_class", 0)
        seed_z_amplitude = seed_orbit.parameters.get("amplitude_z", 0.1)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"开始 Halo 轨道伪弧长延拓")
            print(f"  种子轨道: L{libration_point} {'北' if halo_class == 0 else '南'} Halo")
            print(f"  种子轨道 z_amplitude: {seed_z_amplitude:.4f}")
            print(f"  目标轨道数量: {n_orbits}")
            print(f"  延拓方向: {direction}")
            print(f"  步长: {step_size}")
            print(f"{'=' * 60}")

        orbit_family = OrbitFamily(seed_orbit)

        directions = ["positive", "negative"] if direction == "both" else [direction]

        self.correction.max_iterations = 100
        self.correction.tolerance = 1e-5

        for direction in directions:
            if verbose:
                print(f"\n--- {'正向' if direction == 'positive' else '反向'}延拓 ---")

            family_orbits = [seed_orbit]
            prev_orbit = seed_orbit

            step_sign = 1 if direction == "positive" else -1

            for i in range(n_orbits - 1):
                current_z = prev_orbit.parameters.get("amplitude_z", 0.1)
                next_z = current_z + step_sign * step_size * 10

                if next_z <= 0:
                    if verbose:
                        print(f"  轨道 {i + 1}: z={next_z:.4f} <= 0, 终止")
                    break

                # 基于上一个修正后的轨道状态来估计新轨道的初始状态
                prev_state = prev_orbit.states[0]
                prev_vy = prev_state[4]
                prev_period = prev_orbit.period

                # 对 z 方向的增量进行调整，vy 按比例变化
                # 假设 x 和 period 随 z 线性变化（简化假设）
                z_ratio = next_z / current_z if current_z > 0 else 1.0

                # 新的初始状态估计
                new_x0 = prev_state[0]  # x 变化不大
                new_vy = prev_vy * z_ratio  # vy 按比例调整
                new_period = prev_period * (0.9 + 0.1 * z_ratio)  # 周期也按比例调整

                if halo_class == 0:
                    initial_z = next_z
                else:
                    initial_z = -next_z

                initial_state = np.array(
                    [
                        new_x0,
                        0.0,
                        initial_z,
                        0.0,  # vx
                        new_vy,
                        0.0,  # vz
                    ]
                )

                guess_orbit = Orbit(
                    states=initial_state.reshape(1, -1),
                    times=np.array([0.0]),
                    system=self.dynamics.system,
                )
                guess_orbit.period = new_period

                self.correction.setup_halo_orbit_fixed_z0(
                    z0=initial_z,
                    libration_point=libration_point,
                )

                orbit = self.correction.iterate_correction(guess_orbit, verbose=False)

                if orbit is not None and orbit.correction_success:
                    orbit.family_type = "halo"
                    orbit.parameters["libration_point"] = libration_point
                    orbit.parameters["amplitude_z"] = next_z
                    orbit.parameters["halo_class"] = halo_class

                    family_orbits.append(orbit)
                    prev_orbit = orbit

                    if verbose and (i + 1) % 5 == 0:
                        print(
                            f"  轨道 {i + 1}: z={next_z:.4f}, x0={orbit.states[0, 0]:.4f}, 周期={orbit.period:.4f}"
                        )
                else:
                    if verbose:
                        print(f"  轨道 {i + 1}: z={next_z:.4f} 修正失败")
                    break

            for orbit in family_orbits[1:]:
                orbit_family.add_orbit(orbit)

        if verbose:
            print(f"\n延拓完成：共生成 {len(orbit_family)} 条轨道")
            if len(orbit_family) > 0:
                z_values = [o.parameters.get("amplitude_z", 0) for o in orbit_family]
                print(f"  z_amplitude 范围: [{min(z_values):.4f}, {max(z_values):.4f}]")

        return orbit_family

    def __str__(self):
        return (
            f"Continuation(param={self.continuation_parameter}, n_orbits={len(self.family_orbits)})"
        )

    def __repr__(self):
        return (
            f"Continuation(corrector={self.correction}, "
            f"param={self.continuation_parameter}, step={self.step_size})"
        )
