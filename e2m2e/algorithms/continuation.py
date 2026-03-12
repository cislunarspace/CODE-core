"""
轨道族延拓算法模块

提供自然参数延拓和伪弧长延拓方法，用于生成轨道族。
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

import numpy.typing as npt

import e2m2e
from .differential_correction import DifferentialCorrection
from ..core.orbit import OrbitFamily


class ContinuationMethod(Enum):
    """延拓方法枚举"""

    NATURAL = "natural"
    PSEUDO_ARCLENGTH = "pseudo_arclength"


class Continuation:
    """轨道族延拓

    通过延拓算法生成一族周期轨道，支持自然参数延拓和伪弧长延拓。

    属性：
        correction: DifferentialCorrection对象
        continuation_parameter: 延拓参数名称
        step_size: 当前步长
        direction: 延拓方向
        family_orbits: 轨道族列表
    """

    # 类属性
    DEFAULT_STEP_SIZE = 0.01
    MIN_STEP_SIZE = 1e-6
    MAX_STEP_SIZE = 0.1
    DEFAULT_PREDICTOR_ORDER = 1

    def __init__(
        self,
        correction: DifferentialCorrection,
        param: str = "energy",
        step: Optional[float] = None,
    ) -> None:
        """初始化延拓器

        参数：
        - correction: DifferentialCorrection对象
        - param: 延拓参数（如 "energy", "period", "amplitude", "x0", "z0"）
        - step: 初始步长
        """
        self.correction = correction
        self.dynamics = correction.dynamics if hasattr(correction, "dynamics") else None

        # 延拓参数
        self.continuation_parameter = param
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

        # 终止条件
        self.max_orbits = 100
        self.termination_reason = None

    def natural_continuation(
        self,
        seed_orbit,
        param_range,
        step_size,
        verbose=True,
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

        corrector = e2m2e.algorithms.DifferentialCorrection(self.dynamics)

        # 初始化当前轨道（使用种子轨道作为初始状态）
        current_orbit = seed_orbit.copy()

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
                    orbit = corrector.iterate_correction(guess_orbit)
                else:
                    # 修改时间 - 通过复制并修改orbit的period属性
                    guess_orbit = current_orbit.copy()
                    guess_orbit.period = current_orbit.period + step_size * 2
                    orbit = corrector.iterate_correction(guess_orbit, verbose=True)

                if orbit is not None and orbit.correction_success:
                    # 添加到轨道族
                    orbit_family.add_orbit(orbit)

                    # 更新当前轨道
                    current_orbit = orbit

                    self.continuation_stats["successful_steps"] += 1

                    if verbose and (i + 1) % 10 == 0:
                        print(f"  第 {i + 1} 条轨道，参数值={orbit.states[0, param_index] if param_index < 6 else orbit.period:.6f}，周期={orbit.period:.4f}")

                    # 自适应步长
                    if hasattr(self, "step_size_adaptation") and self.step_size_adaptation:
                        if orbit.correction_iterations < 3:
                            step_size = min(step_size * self.step_growth_factor, self.max_step_size)
                        elif orbit.correction_iterations > 8:
                            step_size = max(step_size * self.step_reduction_factor, self.min_step_size)
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
                    orbit = corrector.iterate_correction(guess_orbit)
                else:
                    # 修改时间 - 通过复制并修改orbit的period属性
                    guess_orbit = current_orbit.copy()
                    guess_orbit.period = current_orbit.period - step_size * 2
                    orbit = corrector.iterate_correction(guess_orbit, verbose=True)

                if orbit is not None and orbit.correction_success:
                    # 添加到轨道族
                    orbit_family.add_orbit(orbit)

                    # 更新当前轨道
                    current_orbit = orbit

                    self.continuation_stats["successful_steps"] += 1

                    if verbose and (i + 1) % 10 == 0:
                        print(f"  第 {i + 1} 条轨道，参数值={orbit.states[0, param_index] if param_index < 6 else orbit.period:.6f}，周期={orbit.period:.4f}")

                    # 自适应步长
                    if hasattr(self, "step_size_adaptation") and self.step_size_adaptation:
                        if orbit.correction_iterations < 3:
                            step_size = min(step_size * self.step_growth_factor, self.max_step_size)
                        elif orbit.correction_iterations > 8:
                            step_size = max(step_size * self.step_reduction_factor, self.min_step_size)
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

        if verbose:
            print(f"\n延拓完成：共生成 {len(orbit_family)} 条轨道")
            stats = self.continuation_stats
            print(f"  成功: {stats['successful_steps']}, 失败: {stats['failed_steps']}")

        return orbit_family

    def pseudo_arclength_continuation(self, seed_state, seed_t_half, n_orbits=50, verbose=True):
        """伪弧长延拓

        使用伪弧长参数化方法，可以跟踪轨道族中的折返点。

        参数：
            seed_state: 种子轨道初始状态
            seed_t_half: 种子轨道半周期
            n_orbits: 目标轨道数量
            verbose: 是否打印信息

        返回：
            dict: 包含轨道族数据的字典
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("开始伪弧长延拓")
            print(f"{'=' * 60}")

        corrector = e2m2e.algorithms.DifferentialCorrection(self.dynamics)

        # 首先用自然延拓获取前两条轨道
        seed_orbit = corrector.iterate_correction(seed_state, seed_t_half, verbose=False)
        if seed_orbit is None:
            print("种子轨道修正失败！")
            return None

        self.family_orbits.append(seed_orbit)
        self.family_states.append(seed_orbit.states[0].copy())
        self.family_periods.append(seed_orbit.period)

        # 获取第二条轨道（微小扰动）
        param_index = self._infer_param_index()
        state_2 = seed_orbit.states[0].copy()
        t_half_2 = seed_orbit.period / 2

        if param_index < 6:
            state_2[param_index] += self.step_size * 0.1
        else:
            t_half_2 += self.step_size * 0.1

        orbit_2 = corrector.iterate_correction(state_2, t_half_2, verbose=False)
        if orbit_2 is None:
            print("第二条轨道修正失败！")
            return None

        self.family_orbits.append(orbit_2)
        self.family_states.append(orbit_2.states[0].copy())
        self.family_periods.append(orbit_2.period)

        # 伪弧长延拓主循环
        for i in range(n_orbits - 2):
            self.continuation_stats["total_steps"] += 1

            # 计算切线方向
            state_prev = self.family_states[-2]
            state_curr = self.family_states[-1]
            t_prev = self.family_periods[-2] / 2
            t_curr = self.family_periods[-1] / 2

            # 切线向量（包含状态和时间）
            tangent_state = state_curr - state_prev
            tangent_time = t_curr - t_prev
            tangent = np.append(tangent_state, tangent_time)
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 0:
                tangent = tangent / tangent_norm

            self.tangent_vector = tangent

            # 预测步
            predicted_state = state_curr + self.step_size * tangent[:6]
            predicted_t_half = t_curr + self.step_size * tangent[6] if len(tangent) > 6 else t_curr

            # 修正步
            orbit = corrector.iterate_correction(predicted_state, predicted_t_half, verbose=False)

            if orbit is not None and orbit.correction_success:
                self.family_orbits.append(orbit)
                self.family_states.append(orbit.states[0].copy())
                self.family_periods.append(orbit.period)
                self.continuation_stats["successful_steps"] += 1

                if verbose and (i + 1) % 10 == 0:
                    print(f"  第 {i + 3}/{n_orbits} 条轨道")
            else:
                self.continuation_stats["failed_steps"] += 1
                self.step_size *= self.step_reduction_factor
                if self.step_size < self.min_step_size:
                    self.termination_reason = "步长过小"
                    break

        if verbose:
            print(f"\n伪弧长延拓完成：共生成 {len(self.family_orbits)} 条轨道")

        return self._build_family_result()

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

    def __str__(self):
        return (
            f"Continuation(param={self.continuation_parameter}, n_orbits={len(self.family_orbits)})"
        )

    def __repr__(self):
        return (
            f"Continuation(correction={self.correction}, "
            f"param={self.continuation_parameter}, step={self.step_size})"
        )
