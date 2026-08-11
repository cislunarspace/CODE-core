"""多重打靶法模块

提供多重打靶法（Multiple Shooting）修正器，将一条轨迹分为多段弧段，
通过匹配相邻段端点状态构建残差向量，利用状态转移矩阵（STM）组装
雅可比矩阵进行最小二乘迭代修正。

支持串行、多线程和多进程（SPICE 内核独立加载）三种并行模式。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from ...data.constants import SECONDS_PER_DAY
from ...data.templates.enums import ConvergenceState, ReferenceFrame

if TYPE_CHECKING:
    from ...data.kernels.manager import SPICEManager

# Unicode sparkline 字符，用于在终端内渲染残差收敛曲线
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"

# ---------------------------------------------------------------------------
# 多进程 worker 支持
# ---------------------------------------------------------------------------
# SPICE 内核存储在进程级 C 全局状态（CSPICE KEEPER 子系统）中，无法跨进程共享。
# 每个子进程在启动时需通过 initializer 重新加载内核，并将 EphemerisDynamics
# 实例保存到进程全局变量 _worker_dynamics，供 worker 函数直接调用。
_worker_dynamics = None  # 进程全局：仅在子进程中被 _worker_init 赋值


def _load_worker_kernels(spice: SPICEManager, kernel_dir: str) -> str:
    """子进程内核加载：闰秒 + 星历，统一走 ``SPICEManager.load_kernel``。

    ``load_kernel`` 在 Python spiceypy 与 Rust cspice 两个独立 CSPICE 实例**双侧**
    furnsh，并对称注册行星名别名（类级 once）。多进程 worker 是新进程、内核池为空，
    必须经此入口加载，否则下沉到 Rust 的力模型查询会因 Rust 实例无内核报错。

    Returns:
        实际加载的 ``.bsp`` 内核路径。
    """
    import os

    # 加载闰秒内核（naif0012.tls）：与 bsp 同目录。load_kernel 内部 _ensure_leapseconds
    # 会自动搜索，但显式加载更稳（避免 search_dir 推断差异）。幂等。
    tls_path = os.path.join(kernel_dir, "naif0012.tls")
    if os.path.isfile(tls_path):
        spice.load_kernel(tls_path)

    # 加载星历内核（de440.bsp 或同目录下优先级最高的 .bsp 文件）
    bsp_path = spice.find_ephemeris_kernel(kernel_dir)
    spice.load_kernel(bsp_path)
    return bsp_path


def _worker_init(
    kernel_dir: str,
    bodies: list[str],
    origin: str,
    frame: ReferenceFrame,
    rtol: float,
    atol: float,
    max_step: float,
) -> None:
    """子进程初始化：经 ``SPICEManager`` 重载内核并构建 EphemerisDynamics。

    该函数由 ProcessPoolExecutor(initializer=...) 在每个工作进程启动时调用一次。
    内核加载经 :func:`_load_worker_kernels` → ``SPICEManager.load_kernel``，在
    Python spiceypy 与 Rust cspice 两侧 furnsh（不再直接调 ``spiceypy.furnsh``
    或手篡内部 once 标志）。动力学对象保存在 ``_worker_dynamics``。

    Args:
        kernel_dir: 包含 ``.bsp`` 和 ``.tls`` 内核文件的目录路径。
        bodies: 引力天体名称列表，如 ``["EARTH", "MOON", "SUN"]``。
        origin: 坐标原点天体，如 ``"EARTH"``。
        frame: 参考坐标系，如 ``"J2000"``。
        rtol: ODE 积分相对容差。
        atol: ODE 积分绝对容差。
        max_step: ODE 最大步长（秒）。
    """
    from ...data.kernels.manager import SPICEManager
    from ..dynamics import EphemerisDynamics, EphemerisSystem

    global _worker_dynamics

    spice = SPICEManager()
    _load_worker_kernels(spice, kernel_dir)

    # 构建动力学对象并覆盖积分参数
    eph_system = EphemerisSystem(bodies=bodies, spice=spice, origin=origin, frame=frame)
    dyn = EphemerisDynamics(system=eph_system)
    dyn.rtol = rtol
    dyn.atol = atol
    dyn.max_step = max_step

    _worker_dynamics = dyn


def _worker_propagate(state: np.ndarray, t_span: tuple[float, float]) -> dict:
    """子进程 worker：使用进程本地的 EphemerisDynamics 积分单段弧段（含 STM）。

    Args:
        state: 初始状态向量，形状 ``(6,)``，单位 km / km/s。
        t_span: ``(t0, tf)``，SPICE ET（秒）。

    Returns:
        包含 ``"states"``（6×n）、``"stm"``（6×6×n）、``"time"``（n,）的字典，
        但仅返回终端切片以减少 IPC 数据量：
        ``{"final_state": (6,), "final_stm": (6,6), "t_end": float}``。
    """
    result = _worker_dynamics.propagate(state, t_span, with_stm=True)  # type: ignore[union-attr]
    return {
        "final_state": result["states"][-1],
        "final_stm": result["stm"][-1],
    }


def _sparkline(values: list[float]) -> str:
    """将浮点序列渲染为单行 Unicode sparkline。

    对序列取 log10 后线性映射到 _SPARK_CHARS，以便在数量级跨越较大时
    也能清晰反映收敛趋势。序列长度为 0 或 1 时返回空字符串或单字符。
    """
    if not values:
        return ""
    import math

    logs = [math.log10(v) if v > 0 else -999.0 for v in values]
    lo, hi = min(logs), max(logs)
    span = hi - lo if hi != lo else 1.0
    n = len(_SPARK_CHARS) - 1
    return "".join(_SPARK_CHARS[max(0, min(n, int((x - lo) / span * n)))] for x in logs)


@dataclass(frozen=True)
class MultipleShootingResult:
    """多重打靶法迭代修正的结果。

    Attributes:
        t_patch: 修正后的时间节点数组，形状 (N,)
        state_patch: 修正后的状态量数组，形状 (N, 6)，每行依次为 [x, y, z, vx, vy, vz]
        converged: 是否在最大迭代次数内收敛
        status: 终止原因枚举
        outer_iterations: 实际迭代次数
        max_residual: 最终迭代的最大残差
        residual_history: 每次迭代最大残差的历史记录
    """

    t_patch: np.ndarray
    state_patch: np.ndarray
    converged: bool
    status: ConvergenceState
    outer_iterations: int
    max_residual: float
    residual_history: list[float]


class MultipleShooting:
    """多重打靶法（Multiple Shooting）修正器。

    将一条轨迹分为 N 个节点、n_seg = N-1 段弧段，对每段独立积分后，
    通过匹配相邻段端点状态来构建残差向量，再利用雅可比矩阵（含 STM）
    进行最小二乘修正，反复迭代直到残差满足容差。

    当 var_time=True 时，时间节点也作为自由变量参与修正（适用于自由时间问题）。

    动力学对象需提供以下接口：

    - ``propagate(state, time_span, with_stm=True)`` ——积分传播，
      返回含 ``"states"`` 和 ``"stm"`` 的字典。
    - ``equations_of_motion(t, state)`` ——计算状态导数（右端函数值）。

    并行策略
    --------
    - ``n_workers=1`` ：串行（默认）。
    - ``n_workers>1`` 且 ``kernel_dir=None`` ：多线程
      （``ThreadPoolExecutor`` ），适合 CR3BP 等纯 Python/NumPy 动力学，
      但受 GIL 限制，并发收益有限。
    - ``n_workers>1`` 且 ``kernel_dir`` 已设置：多进程
      （``ProcessPoolExecutor`` ），每个子进程重载 SPICE 内核，绕过 GIL，
      可充分利用多核 CPU，仅适用于 ``EphemerisDynamics`` （需 SPICE 内核）。
    """

    def __init__(
        self,
        dynamics,
        n_workers: int = 1,
        kernel_dir: str | None = None,
    ) -> None:
        """初始化多重打靶修正器。

        动力学对象接口与并行策略详见类文档字符串。

        Args:
            dynamics: 动力学模型对象，需提供 ``propagate`` 与
                ``equations_of_motion`` 接口。
            n_workers: 并行工作进程/线程数，默认 1 （串行）。
            kernel_dir: SPICE 内核目录路径（含 ``de440.bsp`` 和
                ``naif0012.tls`` ），仅在 ``n_workers>1`` 时需要。
        """
        if dynamics is None:
            raise TypeError("dynamics must not be None")
        self.dynamics = dynamics
        self.max_iter = 50
        self.tolerance = 1e-8
        self.n_workers = n_workers
        self.kernel_dir = kernel_dir

    @staticmethod
    def _propagate_segment(dynamics, state, t_span):
        """传播单段弧段（含 STM），供 ThreadPoolExecutor 调用。"""
        return dynamics.propagate(state, t_span, with_stm=True)

    def _make_process_pool(self) -> ProcessPoolExecutor:
        """构建并返回已初始化 SPICE 内核的 ProcessPoolExecutor。

        从 ``self.dynamics`` 中提取积分参数和系统配置，传递给每个子进程的
        ``_worker_init`` initializer，确保子进程拥有独立的 SPICE 内核池
        和 ``EphemerisDynamics`` 实例。
        """
        dyn = self.dynamics
        system = dyn.system
        return ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_worker_init,  # type: ignore[arg-type]
            initargs=(  # type: ignore[arg-type]
                self.kernel_dir,
                list(system.bodies),
                system.origin,
                system.frame,
                dyn.rtol,
                dyn.atol,
                dyn.max_step,
            ),
        )

    def correct(
        self,
        t_patch: np.ndarray,
        state_patch: np.ndarray,
        var_time: bool = False,
        max_iter: int | None = None,
        tolerance: float | None = None,
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

        residual_history: list[float] = []
        converged = False

        pbar = tqdm(
            total=_max_iter,
            desc="Multiple Shooting",
            unit="iter",
            disable=not verbose,
            leave=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}|"
                " {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )

        # 串行模式下，段级子进度条
        seg_pbar: tqdm | None = None  # type: ignore[type-arg]
        if verbose and self.n_workers <= 1 and n_seg > 1:
            seg_pbar = tqdm(
                total=n_seg,
                desc="  Segments",
                unit="seg",
                leave=False,
                bar_format="  {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            )

        # 决定并行后端：多进程（ProcessPool）或多线程（ThreadPool）
        use_multiprocess = self.n_workers > 1 and self.kernel_dir is not None
        use_multithread = self.n_workers > 1 and self.kernel_dir is None

        # 多进程模式：在 correct() 整个生命周期内保持一个 Pool，避免每次迭代重建子进程
        process_pool: ProcessPoolExecutor | None = None
        if use_multiprocess and n_seg > 1:
            process_pool = self._make_process_pool()
            process_pool.__enter__()

        try:
            prev_res: float | None = None
            for iteration in range(_max_iter):
                # === 第一步：逐段积分，收集 STM、终端状态和端点处的状态导数 ===
                stms = []  # 各段的状态转移矩阵 Φ(t_{i+1}; t_i)
                final_states = []  # 各段积分终端状态
                f_starts = []  # 各段起始点处的状态导数 f(t_i, x_i)
                f_ends = []  # 各段终止点处的状态导数 f(t_{i+1}, x_{i+1})

                # 逐段积分，收集每段的 (终端状态, 终端 STM)。三路后端差异仅在
                # "如何提交/收集任务"，统一产出 segment_pairs，公共的 4 列表
                # append 在下面只写一次。
                segment_pairs: list[tuple[np.ndarray, np.ndarray]] = []

                if use_multiprocess and n_seg > 1:
                    # 多进程：_worker_propagate 仅返回终端切片，减少 IPC 开销
                    future_to_idx = {
                        process_pool.submit(  # type: ignore[union-attr]
                            _worker_propagate,
                            state_work[i].copy(),
                            (float(t_work[i]), float(t_work[i + 1])),
                        ): i
                        for i in range(n_seg)
                    }
                    results: dict[int, dict] = {}
                    for done_count, future in enumerate(as_completed(future_to_idx)):
                        idx = future_to_idx[future]
                        results[idx] = future.result()
                        pbar.set_postfix_str(
                            f"propagating {done_count + 1}/{n_seg} segs [mp]",
                            refresh=True,
                        )
                    segment_pairs = [
                        (results[i]["final_state"], results[i]["final_stm"]) for i in range(n_seg)
                    ]

                elif use_multithread and n_seg > 1:
                    # 多线程：_propagate_segment 返回完整 result dict
                    seg_results: dict[int, dict] = {}
                    with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                        future_to_idx = {
                            executor.submit(
                                MultipleShooting._propagate_segment,
                                self.dynamics,
                                state_work[i],
                                (t_work[i], t_work[i + 1]),
                            ): i
                            for i in range(n_seg)
                        }
                        for done_count, future in enumerate(as_completed(future_to_idx)):
                            idx = future_to_idx[future]
                            seg_results[idx] = future.result()  # type: ignore[index]
                            pbar.set_postfix_str(
                                f"propagating {done_count + 1}/{n_seg} segs",
                                refresh=True,
                            )
                    segment_pairs = [
                        (seg_results[i]["states"][-1], seg_results[i]["stm"][-1])  # type: ignore[index]
                        for i in range(n_seg)
                    ]

                else:
                    # 串行
                    if seg_pbar is not None:
                        seg_pbar.reset()
                    for i in range(n_seg):
                        result = self.dynamics.propagate(
                            state_work[i],
                            (t_work[i], t_work[i + 1]),
                            with_stm=True,
                        )
                        segment_pairs.append((result["states"][-1], result["stm"][-1]))
                        if seg_pbar is not None:
                            seg_pbar.update(1)

                # 公共收集：每段的终端状态/STM + 两端点的状态导数
                for i, (final_state, final_stm) in enumerate(segment_pairs):
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

                # 构建丰富的 postfix 字符串
                if prev_res is not None and prev_res > 0:
                    ratio = max_res / prev_res
                    arrow = "↓" if ratio < 1 else "↑"
                    postfix = f"res={max_res:.2e} {arrow}{ratio:.2f} tol={_tolerance:.0e}"
                else:
                    postfix = f"res={max_res:.2e} tol={_tolerance:.0e}"
                pbar.set_postfix_str(postfix, refresh=False)
                prev_res = max_res

                pbar.update(1)

                # 判断收敛：最大残差是否小于容差
                if max_res < _tolerance:
                    converged = True
                    # 更新描述，明确标出收敛成功
                    pbar.set_description_str("Multiple Shooting [converged]")
                    pbar.set_postfix_str(
                        f"res={max_res:.2e} < tol={_tolerance:.0e}"
                        f"  spark={_sparkline(residual_history)}",
                        refresh=True,
                    )
                    if seg_pbar is not None:
                        seg_pbar.close()
                    pbar.close()
                    return MultipleShootingResult(
                        t_patch=t_work.copy(),
                        state_patch=state_work.copy(),
                        converged=True,
                        status=ConvergenceState.CONVERGED,
                        outer_iterations=iteration + 1,
                        max_residual=max_res,
                        residual_history=list(residual_history),
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
            if process_pool is not None:
                process_pool.__exit__(None, None, None)
            if seg_pbar is not None:
                seg_pbar.close()
            # 未收敛时在最终状态更新描述
            if not converged:
                pbar.set_description_str("Multiple Shooting [max_iter]")
                res_last = residual_history[-1] if residual_history else float("inf")
                pbar.set_postfix_str(
                    f"res={res_last:.2e} tol={_tolerance:.0e}"
                    f"  spark={_sparkline(residual_history)}",
                    refresh=True,
                )
            pbar.close()

        return MultipleShootingResult(
            t_patch=t_work.copy(),
            state_patch=state_work.copy(),
            converged=False,
            status=ConvergenceState.MAX_ITERATIONS,
            outer_iterations=_max_iter,
            max_residual=residual_history[-1] if residual_history else float("inf"),
            residual_history=list(residual_history),
        )


def sample_patch_points(
    orbit,
    n_points: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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


def sample_patch_points_perilune_clustered(
    orbit,
    dynamics,
    n_base: int = 8,
    n_perilune: int = 5,
    perilune_window: float = 0.15,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """在近月点附近加密采样 patch points。

    NRHO 近月点速度大、STM 条件数高，等时间间隔采样会让近月点落在节点
    之间而欠约束，导致多重打靶残差停滞。本函数先积分一圈定位近月点
    （离次天体最近的点），在其两侧 ``perilune_window·period`` 窗口内
    加密 ``n_perilune`` 个节点，其余 ``n_base`` 个节点等时间间隔分布在
    窗口外。

    Args:
        orbit: 周期轨道，需含 ``period``、``times``、``states``。
        dynamics: 动力学对象，用于积分定位近月点（需提供
            ``propagate_orbit_state_at_time``）。
        n_base: 窗口外的等时间间隔节点数。
        n_perilune: 近月点窗口内的加密节点数（含近月点本身）。
        perilune_window: 加密窗口半宽，占周期比例（如 0.15 表示近月点
            前后各 15% 周期）。

    Returns:
        (t_patch, states)：时间节点与对应状态，按时间升序排列。
    """
    if orbit.period is None or orbit.period <= 0:
        raise ValueError("orbit must have a positive period")
    period = float(orbit.period)

    # 积分一圈，密集采样定位近月点（离原点最远的次天体方向上最近）
    # NRHO 近月点 = 离月球最近的点。月球在 synodic 系 x = 1-mu 处。
    mu = getattr(dynamics.system, "mu", None)
    if mu is None:
        # 非 CR3BP：退化为等时间间隔
        return sample_patch_points(orbit, n_base + n_perilune - 1)
    moon_x = 1.0 - mu

    # 一次连续积分一圈（比逐点 propagate_orbit_state_at_time 快两个量级）
    n_probe = 200
    t_probe = np.linspace(0, period, n_probe, endpoint=False)
    probe_result = dynamics.propagate(orbit.states[0], (0, period), t_eval=t_probe)
    states_probe = probe_result["states"]
    dists = np.sqrt(
        (states_probe[:, 0] - moon_x) ** 2 + states_probe[:, 1] ** 2 + states_probe[:, 2] ** 2
    )
    i_perilune = int(np.argmin(dists))
    t_perilune = float(t_probe[i_perilune])

    # 近月点窗口 [t_p - w, t_p + w]，映射到 [0, period) 内
    half_w = perilune_window * period
    t_lo = (t_perilune - half_w) % period
    t_hi = (t_perilune + half_w) % period

    # 窗口内加密点（含近月点）
    t_dense = np.linspace(t_perilune - half_w, t_perilune + half_w, n_perilune)

    # 窗口外等分点：在 [t_hi, t_lo + period] 上等分 n_base 份
    # （绕开窗口，覆盖剩余弧段）
    t_outside = np.linspace(t_hi, t_lo + period, n_base + 1)[:-1] % period

    t_all = np.concatenate([t_dense, t_outside])
    t_all = np.sort(np.unique(np.round(t_all, 12)))  # 去重排序

    # 线性插值获取各节点状态
    states = np.empty((len(t_all), 6))
    for i in range(6):
        states[:, i] = np.interp(t_all, orbit.times, orbit.states[:, i])

    return t_all, states


def convert_to_j2000(
    t_patch_syn: npt.ArrayLike,
    states_syn: npt.ArrayLike,
    syn_j2000,
    reference_et: float,
    tu_days: float = 4.34811305,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """将 synodic 坐标系下的 patch points 转换到 J2000 惯性坐标系。

    该方法用于将 CR3BP 归一化 synodic 坐标系中的轨道节点转换到
    J2000 惯性坐标系，以便在星历模型（Ephemeris）中进行高精度轨道修正。

    Args:
        t_patch_syn: synodic 坐标系下的时间节点数组，归一化时间单位（TU）
        states_syn: synodic 坐标系下的状态数组，形状 (N, 6)，
                   每行 [x, y, z, vx, vy, vz]，归一化单位（DU, DU/TU）
        syn_j2000: SynodicJ2000System 对象，提供坐标转换功能
        reference_et: 参考历元的 SPICE ephemeris time（ET），单位秒
        tu_days: 归一化时间单位（TU）对应的天数，默认值为 4.34811305 天

    Returns:
        Tuple[np.ndarray, np.ndarray]: 包含两个数组的元组：
            - t_patch_j2000: J2000 坐标系下的时间数组，SPICE ET（秒）
            - states_j2000: J2000 坐标系下的状态数组，形状 (N, 6)，
                            每行 [x, y, z, vx, vy, vz]，单位（km, km/s）

    Notes:
        - 时间转换公式：t_j2000 = reference_et + t_syn * (tu_days * 86400)
        - 状态转换使用 SynodicJ2000System.batch_synodic_to_j2000() 方法
        - 适用于将 CR3BP 轨道转换到星历模型进行高精度修正的场景
        - 转换后的状态可用于 EphemerisDynamics 进行轨道传播
    """
    # 确保输入为 numpy 数组
    t_patch_syn = np.asarray(t_patch_syn, dtype=float)
    states_syn = np.asarray(states_syn, dtype=float)

    # 时间转换：归一化时间 → SPICE ephemeris time（秒）
    tu_seconds = tu_days * SECONDS_PER_DAY  # 将 TU 天数转换为秒数
    t_patch_j2000 = reference_et + t_patch_syn * tu_seconds

    # 状态转换：synodic 坐标系 → J2000 惯性坐标系
    states_j2000 = syn_j2000.batch_synodic_to_j2000(
        states_syn=states_syn,
        t_syn_arr=t_patch_syn,
        et0=reference_et,
    )

    return t_patch_j2000, states_j2000
