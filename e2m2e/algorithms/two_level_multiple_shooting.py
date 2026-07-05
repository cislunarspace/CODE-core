"""两层多重打靶法求解器

将多重打靶问题分解为两层交替求解：
- Level 1（局部问题）：逐段调整出发速度使位置连续；
- Level 2（全局问题）：联合调整内部节点的位置和时间使速度连续。

这种分解将高维耦合问题拆解为交替求解的低维子问题，适用于
自由时间多段轨道设计（如多圈共振轨道、星际转移等）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from e2m2e.core.enums import BoundaryMode, TwoLevelMultipleShootingStatus


@dataclass(frozen=True)
class TwoLevelMultipleShootingResult:
    """两层多重打靶法的迭代结果。

    Attributes:
        t_patch: 修正后的时间节点数组
        state_patch: 修正后的状态量数组
        converged: 是否收敛
        status: 终止原因
        outer_iterations: 外层迭代次数
        level1_iterations: 每段 Level 1 迭代次数列表，形状为 ``list[list[int]]`` （外层迭代 × 弧段）
        final_position_residual: 最终最大位置残差
        final_velocity_residual: 最终最大速度残差
        per_patch_position_residual: 各段位置残差
        per_patch_velocity_residual: 各段速度残差
        residual_history: 每次外层迭代的 (最大位置残差, 最大速度残差) 记录
    """

    t_patch: np.ndarray
    state_patch: np.ndarray
    converged: bool
    status: TwoLevelMultipleShootingStatus
    outer_iterations: int
    level1_iterations: list[list[int]]
    final_position_residual: float
    final_velocity_residual: float
    per_patch_position_residual: np.ndarray
    per_patch_velocity_residual: np.ndarray
    residual_history: list[tuple[float, float]]


def _build_level1_constraint(
    propagated_position: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """构建 Level 1 位置约束残差。

    Args:
        propagated_position: 积分终端位置 (3,)
        target_position: 目标节点位置 (3,)

    Returns:
        位置残差向量 (3,)
    """
    return propagated_position - target_position


def _build_level1_jacobian(stm: np.ndarray) -> np.ndarray:
    """构建 Level 1 雅可比矩阵（位置对速度的灵敏度）。

    从 6×6 STM 中提取位置行、速度列的 3×3 子块，
    即 ∂r_final / ∂v_initial。

    Args:
        stm: 单段状态转移矩阵 (6, 6)

    Returns:
        3×3 灵敏度子块
    """
    return stm[0:3, 3:6].copy()


def _build_level2_constraint(
    departure_velocity: np.ndarray,
    arrival_velocity: np.ndarray,
) -> np.ndarray:
    """构建 Level 2 速度连续性约束残差。

    Args:
        departure_velocity: 节点出发速度 (3,)
        arrival_velocity: 前段到达速度 (3,)

    Returns:
        速度残差向量 (3,)
    """
    return departure_velocity - arrival_velocity


def _build_level2_patch_jacobian(
    left_stm: np.ndarray,
    right_stm: np.ndarray,
    left_departure_velocity: np.ndarray,
    left_arrival_velocity: np.ndarray,
    right_departure_velocity: np.ndarray,
    right_arrival_velocity: np.ndarray,
    left_arrival_acceleration: np.ndarray,
    right_departure_acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建单个内部节点的 Level 2 雅可比子块。

    对于内部节点 i，速度连续性约束对 (r_{i-1}, t_{i-1}, r_i, t_i, r_{i+1}, t_{i+1})
    六个修正量的偏导数，返回按此顺序排列的六个 3×3 / 3×1 子块。

    Args:
        left_stm: 左段（i-1→i）状态转移矩阵 (6, 6)
        right_stm: 右段（i→i+1）状态转移矩阵 (6, 6)
        left_departure_velocity: 左段出发速度 (3,)
        left_arrival_velocity: 左段到达速度 (3,)
        right_departure_velocity: 右段出发速度 (3,)
        right_arrival_velocity: 右段到达速度 (3,)
        left_arrival_acceleration: 左段终端加速度 (3,)
        right_departure_acceleration: 右段起始加速度 (3,)

    Returns:
        六个雅可比子块，顺序为 (prev_pos, prev_time, curr_pos, curr_time, next_pos, next_time)
    """
    # 用伪逆而非逆矩阵，STM 子块在周期轨道分岔点附近可能奇异
    left_inverse_stm = np.linalg.pinv(left_stm)
    left_b_inverse = np.linalg.pinv(left_inverse_stm[0:3, 3:6])
    left_a = left_inverse_stm[0:3, 0:3]
    right_b_inverse = np.linalg.pinv(right_stm[0:3, 3:6])
    right_a = right_stm[0:3, 0:3]

    previous_position_block = -left_b_inverse
    previous_time_block = left_b_inverse @ left_departure_velocity
    current_position_block = left_b_inverse @ left_a - right_b_inverse @ right_a
    current_time_block = -(
        left_arrival_acceleration + (left_b_inverse @ left_a) @ left_arrival_velocity
    ) + (right_departure_acceleration + (right_b_inverse @ right_a) @ right_departure_velocity)
    next_position_block = right_b_inverse
    next_time_block = -right_b_inverse @ right_arrival_velocity

    return (
        previous_position_block,
        previous_time_block,
        current_position_block,
        current_time_block,
        next_position_block,
        next_time_block,
    )


class TwoLevelMultipleShooting:
    """两层多重打靶法求解器。

    将自由时间多段轨迹修正分解为交替求解的两个子问题：
    Level 1 逐段调整出发速度使位置连续，Level 2 联合调整内部
    节点位置与时间使速度连续。

    Args:
        dynamics: 动力学模型对象，需提供 propagate 和 equations_of_motion 接口
    """

    def __init__(self, dynamics) -> None:
        """初始化两层多重打靶求解器。

        Args:
            dynamics: 动力学模型对象，需提供 propagate(state, t_span, with_stm) 和
                equations_of_motion(t, state) 接口

        Raises:
            TypeError: dynamics 为 None 或缺少必要接口
        """
        if dynamics is None:
            raise TypeError("dynamics must not be None")
        if not callable(getattr(dynamics, "propagate", None)):
            raise TypeError("dynamics must provide propagate")
        if not callable(getattr(dynamics, "equations_of_motion", None)):
            raise TypeError("dynamics must provide equations_of_motion")
        self.dynamics = dynamics

    def correct(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        *,
        max_outer_iterations: int = 10,
        max_level1_iterations: int = 20,
        position_tolerance: float = 1e-3,
        velocity_tolerance: float = 1e-6,
        level1_position_tolerance: float | None = None,
        boundary: BoundaryMode = BoundaryMode.FIXED_ENDPOINTS,
        verbose: bool = False,
    ) -> TwoLevelMultipleShootingResult:
        """执行两层多重打靶修正。

        交替运行 Level 1（位置连续）和 Level 2（速度连续），
        直到两者同时满足容差或达到最大外层迭代次数。

        Args:
            t_patch: 初始时间节点数组，长度 N，必须严格递增
            state_patch: 初始状态量数组，形状 (N, 6)，每行 [x, y, z, vx, vy, vz]
            max_outer_iterations: 外层（Level 1 + Level 2）最大迭代次数
            max_level1_iterations: Level 1 每段最大迭代次数
            position_tolerance: 位置残差收敛容差
            velocity_tolerance: 速度残差收敛容差
            level1_position_tolerance: Level 1 内部容差。
                若为 ``None``，则使用 ``position_tolerance``。
            boundary: 边界条件，目前仅支持 ``BoundaryMode.FIXED_ENDPOINTS``
            verbose: 是否显示进度条

        Returns:
            TwoLevelMultipleShootingResult 包含修正结果和收敛信息
        """
        # 两层修正策略：Level 1 逐段调整出发速度使位置连续（局部问题），
        # Level 2 联合调整内部节点的位置和时间使速度连续（全局问题）。
        # 这种分解将高维耦合问题拆解为交替求解的低维子问题。
        t_values = np.asarray(t_patch, dtype=float)
        states = np.asarray(state_patch, dtype=float)
        self._validate_inputs(
            t_values,
            states,
            max_outer_iterations,
            max_level1_iterations,
            position_tolerance,
            velocity_tolerance,
            boundary,
        )

        l1_tol = (
            position_tolerance if level1_position_tolerance is None else level1_position_tolerance
        )

        t_work = t_values.copy()
        state_work = states.copy()
        residual_history: list[tuple[float, float]] = []
        level1_iterations: list[list[int]] = []
        had_level1_failure = False
        final_position = np.full(len(t_work) - 1, np.inf)
        final_velocity = np.full(len(t_work) - 1, np.inf)

        iterator = tqdm(
            range(max_outer_iterations),
            desc="Two-Level Multiple Shooting",
            unit="iter",
            disable=not verbose,
        )
        for outer_index in iterator:
            state_work, segment_iterations, had_failure = self._run_level1(
                t_work,
                state_work,
                max_level1_iterations,
                l1_tol,
            )
            level1_iterations.append(segment_iterations)
            # Level 1 单段不收敛不中止，留给 Level 2 修复
            had_level1_failure = had_level1_failure or had_failure
            final_position, final_velocity = self._compute_residuals(t_work, state_work)
            position_residual = float(np.max(final_position))
            velocity_residual = float(np.max(final_velocity))
            residual_history.append((position_residual, velocity_residual))
            if self._is_converged(
                position_residual,
                velocity_residual,
                position_tolerance,
                velocity_tolerance,
            ):
                return self._result(
                    t_work,
                    state_work,
                    True,
                    TwoLevelMultipleShootingStatus.CONVERGED,
                    outer_index + 1,
                    level1_iterations,
                    final_position,
                    final_velocity,
                    residual_history,
                )

            t_candidate, state_candidate = self._run_level2(
                t_work,
                state_work,
            )
            t_work = t_candidate
            state_work = state_candidate
            final_position, final_velocity = self._compute_residuals(t_work, state_work)
            position_residual = float(np.max(final_position))
            velocity_residual = float(np.max(final_velocity))
            residual_history[-1] = (position_residual, velocity_residual)

            # Level 2 后也检查收敛——符合 legacy early-success 语义
            if self._is_converged(
                position_residual,
                velocity_residual,
                position_tolerance,
                velocity_tolerance,
            ):
                return self._result(
                    t_work,
                    state_work,
                    True,
                    TwoLevelMultipleShootingStatus.CONVERGED,
                    outer_index + 1,
                    level1_iterations,
                    final_position,
                    final_velocity,
                    residual_history,
                )

        status = (
            TwoLevelMultipleShootingStatus.LEVEL1_FAILED
            if had_level1_failure
            else TwoLevelMultipleShootingStatus.MAX_ITERATIONS
        )
        return self._result(
            t_work,
            state_work,
            False,
            status,
            max_outer_iterations,
            level1_iterations,
            final_position,
            final_velocity,
            residual_history,
        )

    @staticmethod
    def _validate_inputs(
        t_values: np.ndarray,
        states: np.ndarray,
        max_outer_iterations: int,
        max_level1_iterations: int,
        position_tolerance: float,
        velocity_tolerance: float,
        boundary: BoundaryMode,
    ) -> None:
        """校验 correct 方法的所有输入参数。

        Args:
            t_values: 时间节点数组
            states: 状态量数组 (N, 6)
            max_outer_iterations: 外层最大迭代次数
            max_level1_iterations: Level 1 每段最大迭代次数
            position_tolerance: 位置容差
            velocity_tolerance: 速度容差
            boundary: 边界条件类型

        Raises:
            TypeError: boundary 类型错误时
            ValueError: 任何参数不合法时
        """
        if not isinstance(boundary, BoundaryMode):
            raise TypeError(f"boundary must be a BoundaryMode enum, got {type(boundary).__name__}")
        if t_values.ndim != 1:
            raise ValueError("t_patch must be one-dimensional")
        if states.ndim != 2 or states.shape[1] != 6:
            raise ValueError("state_patch must have shape (n_points, 6)")
        if len(t_values) != len(states):
            raise ValueError("t_patch and state_patch must have the same length")
        if len(t_values) < 3:
            raise ValueError("at least 3 patch points are required")
        if np.any(np.diff(t_values) <= 0):
            raise ValueError("t_patch must be strictly increasing")
        if max_outer_iterations <= 0:
            raise ValueError("max_outer_iterations must be positive")
        if max_level1_iterations <= 0:
            raise ValueError("max_level1_iterations must be positive")
        if position_tolerance <= 0:
            raise ValueError("position_tolerance must be positive")
        if velocity_tolerance <= 0:
            raise ValueError("velocity_tolerance must be positive")
        if boundary != BoundaryMode.FIXED_ENDPOINTS:
            raise ValueError(f"unsupported boundary: {boundary.value}")

    def _run_level1(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        max_iterations: int,
        tolerance: float,
    ) -> tuple[np.ndarray, list[int], bool]:
        """执行 Level 1：逐段修正出发速度使位置连续。

        对每段弧段，用 STM 雅可比迭代修正初始速度，使积分终端位置
        与下一节点位置一致。

        Args:
            t_patch: 时间节点数组 (N,)
            state_patch: 当前状态量数组 (N, 6)
            max_iterations: 每段最大迭代次数
            tolerance: 位置残差容差

        Returns:
            (修正后的状态数组, 各段迭代次数列表, 是否存在未收敛段)
        """
        states = state_patch.copy()
        iterations: list[int] = []
        had_failure = False
        for segment_index in range(len(t_patch) - 1):
            velocity = states[segment_index, 3:6].copy()
            residual_norm = np.inf
            iteration_count = max_iterations
            for iteration_number in range(1, max_iterations + 1):
                initial_state = states[segment_index].copy()
                initial_state[3:6] = velocity
                result = self.dynamics.propagate(
                    initial_state,
                    (t_patch[segment_index], t_patch[segment_index + 1]),
                    with_stm=True,
                )
                final_state = np.asarray(result["states"])[-1]
                final_stm = np.asarray(result["stm"])[-1]
                residual = _build_level1_constraint(
                    final_state[:3],
                    states[segment_index + 1, :3],
                )
                residual_norm = float(np.linalg.norm(residual))
                iteration_count = iteration_number
                if residual_norm <= tolerance:
                    break
                jacobian = _build_level1_jacobian(final_stm)
                delta, _, _, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
                velocity = velocity + delta
            states[segment_index, 3:6] = velocity
            iterations.append(iteration_count)
            if residual_norm > tolerance:
                had_failure = True
        return states, iterations, had_failure

    def _run_level2(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行 Level 2：联合修正内部节点的位置和时间使速度连续。

        对每个内部节点构建速度连续性约束和雅可比矩阵，通过最小二乘
        求解修正量，并应用全步长修正。若全步长导致时间节点逆序，
        则使用几何衰减回溯保证时间单调递增。

        Args:
            t_patch: 时间节点数组 (N,)
            state_patch: 当前状态量数组 (N, 6)

        Returns:
            (修正后的时间数组, 修正后的状态数组)
        """
        t_next = t_patch.copy()
        states = state_patch.copy()
        n_points = len(t_next)
        n_internal = n_points - 2
        if n_internal <= 0:
            return t_next, states

        jacobian = np.zeros((3 * n_internal, 4 * n_internal))
        residuals = np.zeros(3 * n_internal)
        for patch_index in range(1, n_points - 1):
            left_result = self.dynamics.propagate(
                states[patch_index - 1],
                (t_next[patch_index - 1], t_next[patch_index]),
                with_stm=True,
            )
            right_result = self.dynamics.propagate(
                states[patch_index],
                (t_next[patch_index], t_next[patch_index + 1]),
                with_stm=True,
            )
            left_final_state = np.asarray(left_result["states"])[-1]
            right_final_state = np.asarray(right_result["states"])[-1]
            left_stm = np.asarray(left_result["stm"])[-1]
            right_stm = np.asarray(right_result["stm"])[-1]
            left_acceleration = self.dynamics.equations_of_motion(
                t_next[patch_index],
                left_final_state,
            )[3:6]
            right_acceleration = self.dynamics.equations_of_motion(
                t_next[patch_index],
                states[patch_index],
            )[3:6]
            blocks = _build_level2_patch_jacobian(
                left_stm,
                right_stm,
                states[patch_index - 1, 3:6],
                left_final_state[3:6],
                states[patch_index, 3:6],
                right_final_state[3:6],
                left_acceleration,
                right_acceleration,
            )
            row = 3 * (patch_index - 1)
            residuals[row : row + 3] = _build_level2_constraint(
                states[patch_index, 3:6],
                left_final_state[3:6],
            )
            for block_offset, position_block, time_block in (
                (-1, blocks[0], blocks[1]),
                (0, blocks[2], blocks[3]),
                (1, blocks[4], blocks[5]),
            ):
                target_patch = patch_index + block_offset
                if 0 < target_patch < n_points - 1:  # 跳过固定端点，仅修正内部节点
                    column = 4 * (target_patch - 1)
                    jacobian[row : row + 3, column : column + 3] = position_block
                    jacobian[row : row + 3, column + 3] = time_block

        delta, _, _, _ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        # 应用最小二乘修正量；优先全步长，若时间节点逆序则逐步折半。
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate_t = t_next.copy()
            candidate_states = states.copy()
            for internal_index in range(1, n_points - 1):
                column = 4 * (internal_index - 1)
                candidate_states[internal_index, :3] = (
                    candidate_states[internal_index, :3] + damping * delta[column : column + 3]
                )
                candidate_t[internal_index] = (
                    candidate_t[internal_index] + damping * delta[column + 3]
                )
            # 时间节点必须严格递增，否则继续折半
            if np.any(np.diff(candidate_t) <= 0):
                continue
            return candidate_t, candidate_states
        return t_next, states

    def _compute_residuals(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算各段的位置和速度残差。

        Args:
            t_patch: 时间节点数组 (N,)
            state_patch: 状态量数组 (N, 6)

        Returns:
            (各段位置残差的 2-范数数组 (n_seg,), 各段速度残差的 2-范数数组 (n_seg,))
        """
        position_residuals = []
        velocity_residuals = []
        for segment_index in range(len(t_patch) - 1):
            result = self.dynamics.propagate(
                state_patch[segment_index],
                (t_patch[segment_index], t_patch[segment_index + 1]),
                with_stm=True,
            )
            final_state = np.asarray(result["states"])[-1]
            difference = final_state - state_patch[segment_index + 1]
            position_residuals.append(float(np.linalg.norm(difference[:3])))
            velocity_residuals.append(float(np.linalg.norm(difference[3:6])))
        return np.array(position_residuals), np.array(velocity_residuals)

    @staticmethod
    def _is_converged(
        position_residual: float,
        velocity_residual: float,
        position_tolerance: float,
        velocity_tolerance: float,
    ) -> bool:
        """判断两层修正是否同时收敛。

        Args:
            position_residual: 最大位置残差
            velocity_residual: 最大速度残差
            position_tolerance: 位置容差
            velocity_tolerance: 速度容差

        Returns:
            两者均低于容差时返回 True
        """
        return position_residual <= position_tolerance and velocity_residual <= velocity_tolerance

    @staticmethod
    def _result(
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        converged: bool,
        status: TwoLevelMultipleShootingStatus,
        outer_iterations: int,
        level1_iterations: list[list[int]],
        per_patch_position_residual: np.ndarray,
        per_patch_velocity_residual: np.ndarray,
        residual_history: list[tuple[float, float]],
    ) -> TwoLevelMultipleShootingResult:
        """构建两层多重打靶结果对象（深拷贝所有数组）。

        Args:
            t_patch: 最终时间节点数组
            state_patch: 最终状态量数组
            converged: 是否收敛
            status: 终止原因枚举
            outer_iterations: 外层迭代次数
            level1_iterations: 各段 Level 1 迭代次数
            per_patch_position_residual: 各段位置残差
            per_patch_velocity_residual: 各段速度残差
            residual_history: 残差历史

        Returns:
            不可变的 TwoLevelMultipleShootingResult 实例
        """
        return TwoLevelMultipleShootingResult(
            t_patch=t_patch.copy(),
            state_patch=state_patch.copy(),
            converged=converged,
            status=status,
            outer_iterations=outer_iterations,
            level1_iterations=[list(seg) for seg in level1_iterations],
            final_position_residual=float(np.max(per_patch_position_residual)),
            final_velocity_residual=float(np.max(per_patch_velocity_residual)),
            per_patch_position_residual=per_patch_position_residual.copy(),
            per_patch_velocity_residual=per_patch_velocity_residual.copy(),
            residual_history=list(residual_history),
        )
