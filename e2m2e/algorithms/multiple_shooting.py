from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Tuple

import numpy.typing as npt


class MultipleShootingResult:
    """多重打靶法迭代修正的结果。

    Attributes:
        t_patch: 修正后的时间节点数组，形状 (N,)
        state_patch: 修正后的状态量数组，形状 (N, 6)，每行依次为 [x, y, z, vx, vy, vz]
        converged: 是否在最大迭代次数内收敛
        iterations: 实际迭代次数
        max_residual: 最终迭代的最大残差
        residual_history: 每次迭代最大残差的历史记录
    """

    def __init__(self, t_patch, state_patch, converged, iterations, max_residual, residual_history):
        self.t_patch = t_patch
        self.state_patch = state_patch
        self.converged = converged
        self.iterations = iterations
        self.max_residual = max_residual
        self.residual_history = residual_history


class MultipleShooting:
    """多重打靶法（Multiple Shooting）修正器。

    将一条轨迹分为 N 个节点、n_seg = N-1 段弧段，对每段独立积分后，
    通过匹配相邻段端点状态来构建残差向量，再利用雅可比矩阵（含 STM）
    进行最小二乘修正，反复迭代直到残差满足容差。

    当 var_time=True 时，时间节点也作为自由变量参与修正（适用于自由时间问题）。
    """

    def __init__(self, dynamics) -> None:
        """
        Args:
            dynamics: 动力学模型对象，需提供以下接口：
                - propagate(state, time_span, with_stm=True): 积分传播，返回含 "states" 和 "stm" 的字典
                - equations_of_motion(t, state): 计算状态导数（右端函数值）
        """
        if dynamics is None:
            raise TypeError("dynamics must not be None")
        self.dynamics = dynamics
        self.max_iter = 50
        self.tolerance = 1e-8

    def correct(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        var_time: bool = False,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        verbose: bool = False,
    ) -> MultipleShootingResult:
        """执行多重打靶修正。

        将整条轨迹分为若干弧段，对每段独立积分后检验节点处的状态连续性，
        利用状态转移矩阵（STM）组装雅可比矩阵，通过最小二乘求解修正量并迭代。

        Args:
            t_patch: 初始时间节点数组，长度 N
            state_patch: 初始状态量数组，形状 (N, 6)，每行 [x, y, z, vx, vy, vz]
            var_time: 是否允许时间节点作为自由变量参与修正
            max_iter: 最大迭代次数（默认使用 self.max_iter）
            tolerance: 收敛容差（默认使用 self.tolerance）
            verbose: 是否显示进度条

        Returns:
            MultipleShootingResult: 包含修正后的时间/状态、收敛标志、迭代次数和残差历史
        """
        t_patch = np.asarray(t_patch, dtype=float)
        state_patch = np.asarray(state_patch, dtype=float)

        if len(t_patch) != len(state_patch):
            raise ValueError("t_patch and state_patch must have the same length")
        if len(t_patch) == 0:
            raise ValueError("t_patch and state_patch must not be empty")

        _max_iter = max_iter if max_iter is not None else self.max_iter
        _tolerance = tolerance if tolerance is not None else self.tolerance

        # 工作副本，避免修改原始输入
        t_work = t_patch.copy()
        state_work = state_patch.copy()
        N = len(t_work)
        n_seg = N - 1  # 弧段数
        I6 = np.eye(6)

        residual_history = []
        converged = False

        pbar = tqdm(
            total=_max_iter,
            desc="Multiple Shooting",
            unit="iter",
            disable=not verbose,
        )

        try:
            for iteration in range(_max_iter):
                # === 第一步：逐段积分，收集 STM、终端状态和端点处的状态导数 ===
                stms = []  # 各段的状态转移矩阵 Φ(t_{i+1}; t_i)
                final_states = []  # 各段积分终端状态
                f_starts = []  # 各段起始点处的状态导数 f(t_i, x_i)
                f_ends = []  # 各段终止点处的状态导数 f(t_{i+1}, x_{i+1})

                for i in range(n_seg):
                    result = self.dynamics.propagate(
                        state_work[i],
                        (t_work[i], t_work[i + 1]),
                        with_stm=True,
                    )
                    final_state = result["states"][:, -1]
                    final_stm = result["stm"][:, :, -1]
                    final_states.append(final_state)
                    stms.append(final_stm)
                    f_starts.append(self.dynamics.equations_of_motion(t_work[i], state_work[i]))
                    f_ends.append(self.dynamics.equations_of_motion(t_work[i + 1], final_state))

                # === 第二步：构建残差向量 F ===
                # 残差定义：F_i = φ(t_{i+1}; t_i, x_i) - x_{i+1}
                # 即第 i 段积分终端状态与第 i+1 个节点状态的差值
                F = np.zeros(n_seg * 6)
                for i in range(n_seg):
                    F[i * 6 : (i + 1) * 6] = final_states[i] - state_work[i + 1]

                max_res = np.max(np.abs(F))
                residual_history.append(float(max_res))

                pbar.update(1)
                pbar.set_postfix(residual=f"{max_res:.2e}", refresh=False)

                # 判断收敛：最大残差是否小于容差
                if max_res < _tolerance:
                    converged = True
                    return MultipleShootingResult(
                        t_patch=t_work,
                        state_patch=state_work,
                        converged=True,
                        iterations=iteration + 1,
                        max_residual=max_res,
                        residual_history=residual_history,
                    )

                # === 第三步：构建雅可比矩阵 DF ===
                # 雅可比矩阵的每行块对应一个残差约束，每列块对应一个自由变量
                n_constraints = n_seg * 6  # 约束数量：每段 6 个状态分量

                if var_time:
                    # 自由时间修正：变量为 [x_0, x_1, ..., x_{N-1}, t_0, t_1, ..., t_{N-1}]
                    # 共 N*6 + N 个自由变量
                    n_vars = N * 6 + N
                    DF = np.zeros((n_constraints, n_vars))

                    for i in range(n_seg):
                        r_start = i * 6
                        r_end = (i + 1) * 6
                        # 对 x_i 的偏导：∂F_i/∂x_i = Φ_i（状态转移矩阵）
                        DF[r_start:r_end, i * 6 : (i + 1) * 6] = stms[i]
                        # 对 x_{i+1} 的偏导：∂F_i/∂x_{i+1} = -I_6
                        DF[r_start:r_end, (i + 1) * 6 : (i + 2) * 6] = -I6
                        # 对 t_i 的偏导：∂F_i/∂t_i = -f(t_i, x_i)（缩短起始时间的影响）
                        DF[r_start:r_end, N * 6 + i] = -f_starts[i]
                        # 对 t_{i+1} 的偏导：∂F_i/∂t_{i+1} = f(t_{i+1}, φ_i)（延长终止时间的影响）
                        DF[r_start:r_end, N * 6 + i + 1] = f_ends[i]
                else:
                    # 固定时间修正：变量仅为 [x_0, x_1, ..., x_{N-1}]
                    # 共 N*6 个自由变量
                    n_vars = N * 6
                    DF = np.zeros((n_constraints, n_vars))

                    for i in range(n_seg):
                        r_start = i * 6
                        r_end = (i + 1) * 6
                        # 对 x_i 的偏导：∂F_i/∂x_i = Φ_i
                        DF[r_start:r_end, i * 6 : (i + 1) * 6] = stms[i]
                        # 对 x_{i+1} 的偏导：∂F_i/∂x_{i+1} = -I_6
                        DF[r_start:r_end, (i + 1) * 6 : (i + 2) * 6] = -I6

                # === 第四步：最小二乘求解修正量并更新变量 ===
                # 求解 DF · dX = -F
                dX, _, _, _ = np.linalg.lstsq(DF, -F, rcond=None)

                # 应用状态修正量
                state_work = state_work.copy()
                t_work = t_work.copy()

                X_flat = state_work.flatten()
                X_flat += dX[: N * 6]
                state_work = X_flat.reshape(N, 6)

                # 应用时间修正量（仅自由时间模式）
                if var_time:
                    t_work += dX[N * 6 : N * 6 + N]
        finally:
            pbar.close()

        return MultipleShootingResult(
            t_patch=t_work,
            state_patch=state_work,
            converged=False,
            iterations=_max_iter,
            max_residual=residual_history[-1] if residual_history else float("inf"),
            residual_history=residual_history,
        )


def sample_patch_points(
    orbit,
    n_points: int,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """沿周期轨道均匀采样 patch points（打靶节点）。

    该方法用于多重打靶法（Multiple Shooting）的前处理，将一条周期轨道
    在时间上均匀分割为 n_points 个节点，并通过线性插值获取每个节点处的状态。

    Args:
        orbit: 轨道对象，需包含以下属性：
            - period: 轨道周期（归一化时间单位）
            - times: 时间数组，形状 (M,)
            - states: 状态数组，形状 (M, 6)，每行 [x, y, z, vx, vy, vz]
        n_points: 需要采样的节点数量

    Returns:
        Tuple[np.ndarray, np.ndarray]: 包含两个数组的元组：
            - t_patch: 采样时间节点数组，形状 (n_points,)，归一化时间单位
            - states: 采样状态数组，形状 (n_points, 6)，每行 [x, y, z, vx, vy, vz]

    Raises:
        ValueError: 当轨道对象没有 period 属性时抛出

    Notes:
        - 采样时间范围为 [0, period)，不包含周期终点（endpoint=False）
        - 使用线性插值从原始轨道数据中获取节点状态
        - 适用于 CR3BP 归一化坐标系下的周期轨道采样
    """
    if orbit.period is None:
        raise ValueError("Orbit must have a period attribute")

    # 在轨道周期内均匀生成 n_points 个时间节点
    t_patch = np.linspace(0, orbit.period, n_points, endpoint=False)

    # 为每个状态分量进行线性插值
    states = np.empty((n_points, 6))
    for i in range(6):
        states[:, i] = np.interp(t_patch, orbit.times, orbit.states[:, i])

    return t_patch, states


def convert_to_j2000(
    t_patch_syn: npt.ArrayLike,
    states_syn: npt.ArrayLike,
    syn_j2000,
    reference_et: float,
    tu_seconds: float,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """将 synodic 坐标系下的 patch points 转换到 J2000 惯性坐标系。

    该方法用于将 CR3BP 归一化 synodic 坐标系中的轨道节点转换到
    J2000 惯性坐标系，以便在星历模型（Ephemeris）中进行高精度轨道修正。

    Args:
        t_patch_syn: synodic 坐标系下的时间节点数组，归一化时间单位（TU）
        states_syn: synodic 坐标系下的状态数组，形状 (N, 6)，
                   每行 [x, y, z, vx, vy, vz]，归一化单位（DU, DU/TU）
        syn_j2000: SynodicJ2000Transformation 对象，提供坐标转换功能
        reference_et: 参考历元的 SPICE ephemeris time（ET），单位秒
        tu_seconds: 归一化时间单位（TU）对应的秒数

    Returns:
        Tuple[np.ndarray, np.ndarray]: 包含两个数组的元组：
            - t_patch_j2000: J2000 坐标系下的时间数组，SPICE ET（秒）
            - states_j2000: J2000 坐标系下的状态数组，形状 (N, 6)，
                            每行 [x, y, z, vx, vy, vz]，单位（km, km/s）

    Notes:
        - 时间转换公式：t_j2000 = reference_et + t_syn * tu_seconds
        - 状态转换使用 SynodicJ2000Transformation.batch_synodic_to_j2000() 方法
        - 适用于将 CR3BP 轨道转换到星历模型进行高精度修正的场景
        - 转换后的状态可用于 EphemerisDynamics 进行轨道传播
    """
    # 确保输入为 numpy 数组
    t_patch_syn = np.asarray(t_patch_syn, dtype=float)
    states_syn = np.asarray(states_syn, dtype=float)

    # 时间转换：归一化时间 → SPICE ephemeris time（秒）
    t_patch_j2000 = reference_et + t_patch_syn * tu_seconds

    # 状态转换：synodic 坐标系 → J2000 惯性坐标系
    states_j2000 = syn_j2000.batch_synodic_to_j2000(
        states_syn=states_syn,
        t_syn_arr=t_patch_syn,
        et0=reference_et,
    )

    return t_patch_j2000, states_j2000
