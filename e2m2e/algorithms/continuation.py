"""
轨道族延拓算法模块

提供自然参数延拓和伪弧长延拓方法，用于生成轨道族。
包括Halo轨道、Lyapunov轨道等的生成功能。
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.dynamics import CR3BP_Dynamics
from ..core.orbit import Orbit, OrbitFamily
from .halo_initial_guess import compute_halo_initial_guess

logger = logging.getLogger(__name__)


def compute_F_and_dF_symmetric_xz_plane(
    X: np.ndarray,
    SV0: np.ndarray,
    dynamics: CR3BP_Dynamics,
) -> tuple[np.ndarray, np.ndarray]:
    """计算XZ平面对称轨道的约束向量和雅可比矩阵

    对应MATLAB: computeFdF_symPeriodicPlanes_CR3BP(X, SV0i, mu, plane=13)

    X = [rx; rz; vy; tf2] - 自由变量向量
    SV0 = [rx, ry, rz, vx, vy, vz] - 初始状态向量

    F = [vx; vz; ry] - 约束向量（半周期终点状态）
    dF = ∂F/∂X - 约束雅可比矩阵 (3x4)

    Args:
        X: 自由变量向量 [rx, rz, vy, tf2]
        SV0: 初始状态向量 [x, y, z, vx, vy, vz]
        dynamics: CR3BP_Dynamics实例，提供运动方程和雅可比矩阵

    Returns:
        F: 约束向量 [vx, vz, ry]
        dF: 雅可比矩阵 (3, 4)
    """
    rx = X[0]
    rz = X[1]
    vy = X[2]
    tf2 = X[3]

    state = SV0.copy()
    state[0] = rx  # x
    state[2] = rz  # z
    state[4] = vy  # vy

    result = dynamics.propagate(
        state,
        (0, float(tf2)),
        t_eval=[float(tf2)],
        with_stm=True,
        with_jacobi=False,
    )

    final_state = result["states"][-1]
    final_stm = result["stm"][-1]

    dSV = dynamics.equations_of_motion(tf2, final_state)

    F = np.array([final_state[3], final_state[5], final_state[1]])

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

    Args:
        dF: 约束雅可比矩阵 (3, 4)

    Returns:
        Xdot: 切向量 (4,)，单位化
    """
    _, _, Vh = np.linalg.svd(dF)
    tangent = Vh[-1, :]
    norm = np.linalg.norm(tangent)
    if norm > 0:
        tangent = tangent / norm
    return tangent


class Continuation:
    """轨道族延拓

    通过延拓算法生成一族周期轨道，支持自然参数延拓和伪弧长延拓。

    Attributes:
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
        step: float | None = None,
    ) -> None:
        """初始化延拓器

        Args:
            corrector: DifferentialCorrection对象，已配置好对称性和约束
            step: 初始延拓步长（默认 0.01）
        """
        self.correction = corrector
        self.dynamics: CR3BP_Dynamics = corrector.dynamics  # type: ignore[assignment]

        self.continuation_parameter: str | None = None
        if corrector.fixed_parameters:
            self.continuation_parameter = next(iter(corrector.fixed_parameters))
        self.step_size = step or self.DEFAULT_STEP_SIZE
        self.initial_step_size = self.step_size

        # 轨道族
        self.family_orbits: list[Orbit] = []
        self.family_parameters: list[float] = []
        self.family_states: list[np.ndarray] = []
        self.family_periods: list[float] = []

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
        self.step_size_adaptation = True
        self.step_growth_factor = 1.2  # natural_continuation 专用：根据迭代次数自适应调整
        self.step_reduction_factor = 0.5  # 步长缩减因子
        self.step_increase_factor = 1.2  # 与 step_growth_factor 同值，供 generate_halo_family 使用
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

        Args:
            seed_orbit: Orbit, 种子轨道
            param_range: tuple, 延拓参数范围 (param_min, param_max)
            step_size: float, 步长（始终为正值，延拓方向由参数范围自动确定）
            verbose: 是否打印信息

        Returns:
            OrbitFamily: 包含轨道族的OrbitFamily对象
        """
        # 验证并限制步长
        if step_size > self.max_step_size:
            if verbose:
                logger.warning(
                    "输入步长 %.6f 超过最大限制 %.6f，限制为 %.6f",
                    step_size,
                    self.max_step_size,
                    self.max_step_size,
                )
            step_size = self.max_step_size

        if step_size < self.min_step_size:
            if verbose:
                logger.warning(
                    "输入步长 %.6f 小于最小限制 %.6f，限制为 %.6f",
                    step_size,
                    self.min_step_size,
                    self.min_step_size,
                )
            step_size = self.min_step_size

        orbit_family = OrbitFamily([seed_orbit])

        param_index = self._infer_param_index()

        if param_index < 6:
            seed_param_value = seed_orbit.states[0, param_index]
        else:
            seed_param_value = seed_orbit.period

        param_min, param_max = param_range

        forward = param_max > seed_param_value
        backward = param_min < seed_param_value

        logger.info("=" * 30)
        logger.info("开始自然参数延拓 (参数: %s)", self.continuation_parameter)
        logger.info("种子轨道参数值: %.6f", seed_param_value)
        logger.info("参数范围: %s", param_range)
        logger.info("延拓方向: %s%s", "正向" if forward else "", "反向" if backward else "")
        logger.info("步长: %s", step_size)
        logger.info("=" * 30)

        corrector = self.correction

        current_orbit = seed_orbit.copy()

        temp_orbits_with_steps = []

        step_size_history = []

        if forward:
            if verbose:
                logger.info("--- 正向延拓 (参数增大方向) ---")

            target_forward = param_max
            i = 0
            while True:
                if param_index < 6:
                    current_param_value = current_orbit.states[0, param_index]
                else:
                    current_param_value = current_orbit.period

                if current_param_value >= target_forward:
                    break

                if param_index < 6:
                    guess_orbit = current_orbit.copy()
                    guess_orbit.states[0, param_index] += step_size
                    orbit = corrector.iterate_correction(guess_orbit, verbose=verbose)
                else:
                    guess_orbit = current_orbit.copy()
                    guess_orbit.period = current_orbit.period + step_size * 2
                    orbit = corrector.iterate_correction(guess_orbit, verbose=verbose)

                if orbit is not None and orbit.correction_success:
                    orbit.metadata["continuation_step"] = i + 1
                    temp_orbits_with_steps.append((orbit, i + 1))

                    current_orbit = orbit

                    self.continuation_stats["successful_steps"] += 1

                    if (i + 1) % 10 == 0:
                        if verbose:
                            param_val = (
                                orbit.states[0, param_index] if param_index < 6 else orbit.period
                            )
                            logger.info(
                                "  第 %d 条轨道，参数值=%.6f，周期=%.4f",
                                i + 1,
                                param_val,
                                orbit.period,
                            )
                        else:
                            logger.info("  正向延拓进度：已完成 %d 条轨道", i + 1)

                    if hasattr(self, "step_size_adaptation") and self.step_size_adaptation:
                        if orbit.correction_iterations < 3:
                            step_size = min(step_size * self.step_growth_factor, self.max_step_size)
                        elif orbit.correction_iterations > 8:
                            step_size = max(
                                step_size * self.step_reduction_factor, self.min_step_size
                            )
                else:
                    self.continuation_stats["failed_steps"] += 1

                    step_size *= self.step_reduction_factor
                    if step_size < self.min_step_size:
                        self.termination_reason = "步长过小，延拓终止"
                        if verbose:
                            logger.info("正向步长过小，延拓终止于第 %d 条轨道", len(orbit_family))
                        break

                    if verbose:
                        logger.info("  第 %d 步修正失败，减小步长至 %.6f", i + 1, step_size)

                step_size_history.append(step_size)
                i += 1

                if len(orbit_family) >= self.max_orbits:
                    self.termination_reason = "达到最大轨道数限制"
                    break

        if backward:
            current_orbit = seed_orbit.copy()

            if verbose:
                logger.info("--- 反向延拓 (参数减小方向) ---")

            target_backward = param_min
            i = 0
            while True:
                if param_index < 6:
                    current_param_value = current_orbit.states[0, param_index]
                else:
                    current_param_value = current_orbit.period

                if current_param_value <= target_backward:
                    break

                if param_index < 6:
                    guess_orbit = current_orbit.copy()
                    guess_orbit.states[0, param_index] -= step_size
                    orbit = corrector.iterate_correction(guess_orbit, verbose=verbose)
                else:
                    guess_orbit = current_orbit.copy()
                    guess_orbit.period = current_orbit.period - step_size * 2
                    orbit = corrector.iterate_correction(guess_orbit, verbose=verbose)

                if orbit is not None and orbit.correction_success:
                    orbit.metadata["continuation_step"] = -(i + 1)
                    temp_orbits_with_steps.append((orbit, -(i + 1)))

                    current_orbit = orbit

                    self.continuation_stats["successful_steps"] += 1

                    if (i + 1) % 10 == 0:
                        if verbose:
                            param_val = (
                                orbit.states[0, param_index] if param_index < 6 else orbit.period
                            )
                            logger.info(
                                "  第 %d 条轨道，参数值=%.6f，周期=%.4f",
                                i + 1,
                                param_val,
                                orbit.period,
                            )
                        else:
                            logger.info("  反向延拓进度：已完成 %d 条轨道", i + 1)

                    if hasattr(self, "step_size_adaptation") and self.step_size_adaptation:
                        if orbit.correction_iterations < 3:
                            step_size = min(step_size * self.step_growth_factor, self.max_step_size)
                        elif orbit.correction_iterations > 8:
                            step_size = max(
                                step_size * self.step_reduction_factor, self.min_step_size
                            )
                else:
                    self.continuation_stats["failed_steps"] += 1

                    step_size *= self.step_reduction_factor
                    if step_size < self.min_step_size:
                        self.termination_reason = "步长过小，延拓终止"
                        if verbose:
                            logger.info("反向步长过小，延拓终止于第 %d 条轨道", len(orbit_family))
                        break

                    if verbose:
                        logger.info("  第 %d 步修正失败，减小步长至 %.6f", i + 1, step_size)

                step_size_history.append(step_size)
                i += 1

                if len(orbit_family) >= self.max_orbits:
                    self.termination_reason = "达到最大轨道数限制"
                    break

        seed_orbit.metadata["continuation_step"] = 0
        all_orbits_with_steps = [(seed_orbit, 0)] + temp_orbits_with_steps

        def sort_key(item):
            orbit, step = item
            return (abs(step), step > 0)

        all_orbits_with_steps.sort(key=sort_key)

        orbit_family.orbits = []
        for orbit, _step in all_orbits_with_steps:
            orbit_family.add_orbit(orbit)

        if verbose:
            logger.info("延拓完成：共生成 %d 条轨道", len(orbit_family))
            stats = self.continuation_stats
            logger.info("  成功: %d, 失败: %d", stats["successful_steps"], stats["failed_steps"])
            logger.info("  轨道已按距离种子轨道的步数排序: 0, 1, -1, 2, -2, ...")

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
        dc_scheme: str = "adaptive",
        libration_point: int = 1,
        directional_increment: bool = False,
        target_vector: int = 0,
        target_direction: int = 1,
        progress_callback=None,
    ):
        """伪弧长延拓（对应 MATLAB ``continuation_PAL_CR3BP``，plane=13 / XZ 对称）

        自由变量 ``X = [rx, rz, vy, T/2]``，``Xdot = null(dF)``，PAL 约束
        ``G = [F; (Xnew-X)·Xdot - DeltaS]``；内层用 ``Xnew`` 计算 ``F``（与 MATLAB 中
        仅用固定 ``X`` 相比更一致）。每步后用微分修正闭合。

        Args:
            seed_orbit: 种子轨道（Orbit对象）
            n_orbits: 本支要生成的新轨道条数（与 MATLAB 中 ``N`` 一致）
            step_size: 伪弧长步长 ``|DeltaS|``（正数；``direction`` 决定符号）
            direction: ``positive`` 或 ``negative``（双侧延拓请调用两次或使用
                ``halo_pseudo_arclength_continuation(..., direction='both')``）
            dc_scheme: 见 ``Continuation`` 文档字符串
            target_vector: 与 MATLAB ``TargetVector`` 对应的 0 基下标：``0=rx,1=rz,2=vy,3=T/2``

        Returns:
            OrbitFamily: 仅含种子 + 本支新轨道（不重复添加种子）
        """
        dynamics = self.dynamics

        if direction not in ("positive", "negative"):
            raise ValueError(
                "direction 须为 positive 或 negative（双侧请用 halo_pseudo_arclength_continuation）"
            )

        step_sign = 1.0 if direction == "positive" else -1.0

        if verbose:
            logger.info("=" * 30)
            logger.info("开始伪弧长延拓 (Pseudo Arc-Length Continuation)")
            logger.info("=" * 30)
            logger.info("  质量比 mu = %.10f", dynamics.system.mu)
            logger.info("  本支新轨道数 N = %d", n_orbits)
            logger.info("  步长 |DeltaS| = %s", step_size)
            logger.info("  延拓方向 = %s", direction)
            logger.info("  dc_scheme = %s", dc_scheme)

        orbit_family = OrbitFamily([seed_orbit])
        self.correction.tolerance = TolDiffCorr

        SV0i = seed_orbit.states[0].copy()
        tfi = float(seed_orbit.period)
        X = np.array([SV0i[0], SV0i[2], SV0i[4], tfi / 2])

        _, dF = compute_F_and_dF_symmetric_xz_plane(X, SV0i, dynamics)
        Xdot = compute_tangent_vector(dF)

        family_states: list[np.ndarray] = [SV0i.copy()]

        if verbose:
            logger.debug("初始自由变量 X = [%.6f, %.6f, %.6f, %.6f]", X[0], X[1], X[2], X[3])
            logger.debug(
                "初始切向量 Xdot = [%.6f, %.6f, %.6f, %.6f]",
                Xdot[0],
                Xdot[1],
                Xdot[2],
                Xdot[3],
            )

        ds = float(step_sign * step_size)
        tv = target_vector
        td = target_direction

        for n in range(n_orbits):
            if verbose and (n + 1) % 5 == 0:
                logger.info("--- 延拓第 %d/%d 条轨道 ---", n + 1, n_orbits)

            delta_dir = ds * Xdot
            if directional_increment:
                Xnew = X + ds * Xdot if td * delta_dir[tv] > 0 else X - ds * Xdot
            else:
                Xnew = X + ds * Xdot

            if verbose:
                logger.debug(
                    "  预测 Xnew = [%.6f, %.6f, %.6f, %.6f]",
                    Xnew[0],
                    Xnew[1],
                    Xnew[2],
                    Xnew[3],
                )

            # 仅欧拉预测时的自由变量（PAL 若跳入 F=0 的非物理根则回退到此）
            X_predictor_only = Xnew.copy()

            Xdot_new = Xdot
            for iter_pal in range(IterMax):
                SV0_guess = SV0i.copy()
                SV0_guess[0] = Xnew[0]
                SV0_guess[2] = Xnew[1]
                SV0_guess[4] = Xnew[2]

                F, dF_new = compute_F_and_dF_symmetric_xz_plane(Xnew, SV0_guess, dynamics)
                Xdot_new = compute_tangent_vector(dF_new)

                G = np.zeros(4)
                G[:3] = F
                G[3] = np.dot(Xnew - X, Xdot) - ds

                dG = np.zeros((4, 4))
                dG[:3, :] = dF_new
                dG[3, :] = Xdot

                # 与 MATLAB continuation_PAL_CR3BP 一致：先判收敛，再更新 Xnew
                if np.linalg.norm(F) < TolPAL:
                    if verbose:
                        logger.debug(
                            "  PAL迭代 %d: 收敛, ||F|| = %.2e",
                            iter_pal + 1,
                            np.linalg.norm(F),
                        )
                    break

                try:
                    delta_X = np.linalg.solve(dG, G)
                except np.linalg.LinAlgError:
                    if verbose:
                        logger.warning("  PAL迭代 %d: 雅可比矩阵奇异", iter_pal + 1)
                    break

                # 限制牛顿步，避免 PAL 收敛到 rx 极大的非 Halo 物理解（F=0 多根）
                max_step = np.array([0.04, 0.12, 0.12, 0.08], dtype=float)
                delta_X = np.clip(delta_X, -max_step, max_step)
                Xnew = Xnew - delta_X

            Xdot = Xdot_new
            X = Xnew.copy()

            # PAL 可能在 F=0 的另一支上“收敛”，偏离 L1 Halo 族；回退为欧拉预测初值
            _x, _z, _tf2 = X[0], X[1], X[3]
            pal_plausible = (
                0.75 < _x < 1.05
                and abs(_z) > 1e-3
                and abs(_z) < 0.55
                and 0.35 < _tf2 < 1.35
                and abs(_x - X_predictor_only[0]) < 0.25
                and abs(_z - X_predictor_only[1]) < 0.25
            )
            if not pal_plausible:
                if verbose:
                    logger.warning(
                        "  PAL 结果偏离物理 Halo 支 (x=%.4f, z=%.4f, T/2=%.4f)，回退为欧拉预测初值",
                        _x,
                        _z,
                        _tf2,
                    )
                X = X_predictor_only.copy()

            SV0_corr = SV0i.copy()
            SV0_corr[0] = X[0]
            SV0_corr[2] = X[1]
            SV0_corr[4] = X[2]
            tf_corr = 2 * X[3]

            x0_last = family_states[-1][0]
            z0_last = family_states[-1][2]

            if dc_scheme == "matlab_halo_type1":
                self.correction.setup_halo_orbit_fixed_x0(
                    x0=SV0_corr[0], libration_point=libration_point
                )
            elif dc_scheme == "matlab_halo_type2":
                if abs(SV0_corr[0] - x0_last) > abs(SV0_corr[2] - z0_last):
                    self.correction.setup_halo_orbit_fixed_x0(
                        x0=SV0_corr[0], libration_point=libration_point
                    )
                else:
                    self.correction.setup_halo_orbit_fixed_z0(
                        z0=SV0_corr[2], libration_point=libration_point
                    )
            else:
                if abs(SV0_corr[0] - x0_last) > abs(SV0_corr[2] - z0_last):
                    self.correction.setup_3D_symmetric_x_fixed_x0(x0=SV0_corr[0])
                else:
                    self.correction.setup_3D_symmetric_xz_fixed_z0(z0=SV0_corr[2])

            guess_orbit = Orbit(
                states=SV0_corr.reshape(1, -1),
                times=np.array([0.0]),
                system=self.correction.dynamics.system,
            )
            guess_orbit.period = tf_corr

            orbit = self.correction.iterate_correction(guess_orbit, verbose=False)

            # PAL 初值在固定 x0 下常落入寄生根或与 STM 牛顿不兼容；与种子生成一致改用固定 z0 再试
            if (orbit is None or not orbit.correction_success) and dc_scheme == "matlab_halo_type1":
                self.correction.setup_halo_orbit_fixed_z0(
                    z0=SV0_corr[2], libration_point=libration_point
                )
                orbit = self.correction.iterate_correction(guess_orbit, verbose=False)

            if orbit is not None and orbit.correction_success:
                orbit_family.add_orbit(orbit)
                family_states.append(orbit.states[0].copy())
                X = np.array(
                    [
                        orbit.states[0, 0],
                        orbit.states[0, 2],
                        orbit.states[0, 4],
                        orbit.period / 2,
                    ]
                )
                _, dF = compute_F_and_dF_symmetric_xz_plane(X, orbit.states[0].copy(), dynamics)
                Xdot = compute_tangent_vector(dF)

                if progress_callback is not None:
                    progress_callback(n + 1, n_orbits, orbit, direction)

                if verbose and (n + 1) % 5 == 0:
                    logger.info(
                        "  轨道 %d: x0=%.4f, z0=%.4f, T=%.4f",
                        n + 1,
                        orbit.states[0, 0],
                        orbit.states[0, 2],
                        orbit.period,
                    )
            else:
                if verbose:
                    logger.warning("  轨道 %d: 微分修正失败", n + 1)
                break

        if verbose:
            logger.info("伪弧长延拓完成：共生成 %d 条轨道", len(orbit_family))

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
        return param_map.get(self.continuation_parameter or "", 0)

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

        Args:
            libration_point: 拉格朗日点 (1=L1, 2=L2)
            amplitude_z: Z方向振幅
            halo_class: 0=北Halo, 1=南Halo
            verbose: 是否打印详细信息

        Returns:
            Orbit: Halo周期轨道
        """
        if libration_point not in [1, 2]:
            raise ValueError(f"libration_point必须是1或2，当前为{libration_point}")
        if amplitude_z <= 0:
            raise ValueError(f"amplitude_z必须为正数，当前为{amplitude_z}")
        if halo_class not in [0, 1]:
            raise ValueError(f"halo_class必须是0或1，当前为{halo_class}")

        if verbose:
            halo_label = "北" if halo_class == 0 else "南"
            logger.info("生成Halo轨道: L%d %s Halo", libration_point, halo_label)
            logger.info("  Z振幅: %s", amplitude_z)

        mu = self.correction.dynamics.system.mu

        guess = compute_halo_initial_guess(
            mu=mu,
            z_amplitude=amplitude_z,
            L=libration_point,
            halo_class=halo_class,
        )

        initial_z = amplitude_z if halo_class == 0 else -amplitude_z

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
            system=self.correction.dynamics.system,
        )
        initial_orbit.period = 2.0 * guess["T_half"]

        self.correction.max_iterations = 150
        self.correction.tolerance = 1e-5

        if verbose:
            logger.info("  初始猜测: x0=%.6f, vy0=%.6f", guess["x0"], guess["vy0"])
            logger.info("  预估周期: %.4f TU", initial_orbit.period)

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
                logger.info("[ok] Halo轨道生成成功: 周期=%.6f TU", orbit.period)

        return orbit

    def generate_halo_family(
        self,
        seed_orbit: Orbit,
        n_orbits: int = 50,
        direction: str = "positive",
        step_size: float = 0.001,
        z_range: tuple[float, float] | None = None,
        verbose: bool = False,
        progress_callback=None,
    ) -> list[Orbit]:
        """生成Halo轨道族

        使用自然参数延拓法生成Halo轨道族。每步以前一收敛轨道为初值，
        沿z方向推进，比每次都从Richardson近似重新开始更稳定。

        Args:
            seed_orbit: 种子轨道
            n_orbits: 目标轨道数量（含种子；每支各生成n_orbits-1条新轨道）
            direction: 延拓方向 ("positive", "negative", "both")
            step_size: z方向步长（正数；方向由direction控制）
            z_range: 延拓z振幅范围 (z_min, z_max)；提供时优先使用，
                代替硬编码边界，并自动推断延拓方向
            verbose: 是否打印详细信息

        Returns:
            List[Orbit]: Halo轨道族
        """
        if n_orbits < 1:
            raise ValueError(f"n_orbits必须大于0，当前为{n_orbits}")
        if direction not in ["positive", "negative", "both"]:
            raise ValueError(f"direction必须是positive/negative/both，当前为{direction}")

        family = [seed_orbit]
        libration_point = int(seed_orbit.parameters.get("libration_point", 1))
        halo_class = int(seed_orbit.parameters.get("halo_class", 0))
        seed_z = float(seed_orbit.states[0, 2])

        # z边界: 北Halo z>0, 南Halo z<0
        default_z_limit = 0.5 if halo_class == 0 else -0.5
        z_threshold = 1e-4 if halo_class == 0 else -1e-4

        # 处理 z_range
        if z_range is not None:
            z_min, z_max = z_range
            if z_min >= z_max:
                raise ValueError(f"z_range必须满足z_min < z_max，当前为({z_min}, {z_max})")
            # 自动推断方向：正向 = 往更大的 z 值走，反向 = 往更小的 z 值走
            forward = z_max > seed_z
            backward = z_min < seed_z
            dirs = []
            if forward:
                dirs.append("positive")
            if backward:
                dirs.append("negative")
            if not dirs:
                logger.warning("z_range不包含种子轨道z0，不延拓")
                return family
            directions = dirs
            logger.info(
                "开始生成Halo轨道族: z范围=[%.4f, %.4f], 方向=%s, 最大数量=%d",
                z_min,
                z_max,
                "/".join(directions),
                n_orbits,
            )
        else:
            directions = ["positive", "negative"] if direction == "both" else [direction]
            logger.info("开始生成Halo轨道族: 目标数量=%d, 方向=%s", n_orbits, direction)

        logger.info(
            "  种子轨道: L%d %s Halo, z0=%.6f",
            libration_point,
            "北" if halo_class == 0 else "南",
            seed_z,
        )

        # 自适应步长参数
        min_step = 1e-4
        max_step = 0.05
        growth = 1.2
        shrink = 0.5

        for dir_name in directions:
            current_orbit = seed_orbit
            current_step = float(step_size)
            current_z = float(current_orbit.states[0, 2])

            # z边界: z_range模式使用用户指定边界，否则使用默认值
            if z_range is not None:
                z_min_val, z_max_val = z_range
                z_limit = z_max_val if dir_name == "positive" else z_min_val
            else:
                z_limit = default_z_limit

            if verbose:
                dir_label = "正向" if dir_name == "positive" else "反向"
                logger.info("--- %s延拓 (边界=%.4f) ---", dir_label, z_limit)

            for i in range(n_orbits - 1):
                # 全局轨道数上限：n_orbits 是总轨道数（含种子），不是每方向上限
                if len(family) >= n_orbits:
                    if verbose:
                        logger.info("  达到全局轨道数上限 %d, 终止", n_orbits)
                    break

                # 计算目标z
                dz = current_step if dir_name == "positive" else -current_step
                target_z = current_z + dz

                # 边界检查
                if halo_class == 0:
                    if target_z <= z_threshold or target_z >= z_limit:
                        if verbose:
                            logger.info("  达到z边界 %.4f, 终止", z_limit)
                        break
                else:
                    if target_z >= z_threshold or target_z <= z_limit:
                        if verbose:
                            logger.info("  达到z边界 %.4f, 终止", z_limit)
                        break

                # 配置微分修正器
                self.correction.setup_halo_orbit_fixed_z0(
                    z0=target_z,
                    libration_point=libration_point,
                )
                self.correction.max_iterations = 150
                self.correction.tolerance = 1e-6

                # 用前一轨道构造初值猜测
                guess_state = current_orbit.states[0].copy()
                guess_state[2] = target_z
                guess = Orbit(
                    states=guess_state.reshape(1, -1),
                    times=np.array([0.0]),
                    system=self.correction.dynamics.system,
                )
                guess.period = current_orbit.period

                orbit = self.correction.iterate_correction(guess, verbose=False)

                if orbit is not None and orbit.correction_success:
                    orbit.family_type = "halo"
                    orbit.parameters["libration_point"] = libration_point
                    orbit.parameters["halo_class"] = halo_class
                    orbit.parameters["amplitude_z"] = abs(target_z)
                    family.append(orbit)
                    current_orbit = orbit
                    current_z = target_z

                    # 自适应步长
                    if self.step_size_adaptation:
                        if orbit.correction_iterations < 5:
                            current_step = min(current_step * growth, max_step)
                        elif orbit.correction_iterations > 20:
                            current_step = max(current_step * shrink, min_step)

                    if progress_callback is not None:
                        progress_callback(i + 1, n_orbits - 1, orbit, dir_name)

                    if verbose and (i + 1) % 5 == 0:
                        logger.info(
                            "  第%d条: z=%.5f, x=%.6f, T=%.4f",
                            i + 1,
                            target_z,
                            orbit.states[0, 0],
                            orbit.period,
                        )
                else:
                    # 修正失败，缩小步长重试一次
                    current_step = max(current_step * shrink, min_step)
                    if current_step <= min_step:
                        if verbose:
                            logger.warning(
                                "  第%d步修正失败且步长已达最小, 终止",
                                i + 1,
                            )
                        break
                    if verbose:
                        logger.info(
                            "  第%d步修正失败, 缩小步长至%.6f后重试",
                            i + 1,
                            current_step,
                        )
                    # 保持current_orbit不变，用更小的步长重新循环
                    # 不增加i，相当于重试
                    continue

        logger.info("[ok] 轨道族生成完成: 共%d条轨道", len(family))
        return family

    def halo_pseudo_arclength_continuation(
        self,
        seed_orbit: Orbit,
        n_orbits: int = 50,
        direction: str = "both",
        step_size: float = 0.0045,
        step_size_negative: float | None = None,
        verbose: bool = True,
        TolPAL: float = 1e-6,
        TolDiffCorr: float = 1e-6,
        IterMax: int = 100,
        dc_scheme: str = "adaptive",
        directional_increment: bool = True,
        progress_callback=None,
    ) -> OrbitFamily:
        """Halo 轨道族伪弧长延拓（对齐 ``CR3BP_MATLAB_Library``）

        对应 ``continuation_PAL_CR3BP`` + XZ 对称（``X = [rx,rz,vy,T/2]``），与
        ``examples/FAMILY_L1Halo_North.m`` 的 PAL 步一致。

        微分修正：MATLAB 脚本在 PAL 后使用 ``type=1``（固定 ``x0``）。当前 Python 在 PAL
        初值下 ``setup_halo_orbit_fixed_x0`` 易与 STM 牛顿耦合到非物理解，故默认
        ``dc_scheme='adaptive'``（按 Δx/Δz 在 fixed x / fixed z 间切换，与原
        ``pseudo_arclength_continuation`` 一致）。若需对齐 MATLAB 的 fixedX，可设
        ``matlab_halo_type1``（失败时会自动再试 ``fixed_z0``）。
        - ``DirectionalIncrement``、``TargetVector``、``TargetDirection`` 与脚本一致：
          正向支 ``TargetVector=2``（``rz``）、``TargetDirection=+1``；负向支
          ``TargetVector=1``（``rx``）、``TargetDirection=-1``（0 基下标见
          ``pseudo_arclength_continuation``）。

        ``FAMILY_L1Halo_North.m`` 中正负支 ``DeltaS`` 不同（0.0045 与 0.009），可用
        ``step_size_negative`` 单独指定负向步长模长。

        Args:
            seed_orbit: 已收敛的 Halo 种子轨道
            n_orbits: 每一支的新轨道条数 ``N``（与 MATLAB 一致；``direction=both`` 时
                正向、负向各生成 ``n_orbits`` 条）
            step_size: 正向伪弧长步长 ``|DeltaS|``（默认 0.0045）
            step_size_negative: 负向步长模长，默认与 ``step_size`` 相同
            dc_scheme: ``matlab_halo_type1`` / ``matlab_halo_type2`` / ``adaptive``

        Returns:
            OrbitFamily: 种子 + 各支新轨道（无重复种子）
        """
        libration_point = int(seed_orbit.parameters.get("libration_point", 1))
        halo_class = seed_orbit.parameters.get("halo_class", 0)
        seed_z_amplitude = seed_orbit.parameters.get("amplitude_z", 0.1)

        if direction not in ("positive", "negative", "both"):
            raise ValueError("direction 须为 positive / negative / both")

        if step_size_negative is None:
            step_size_negative = step_size

        self.correction.max_iterations = 150

        if verbose:
            logger.info("=" * 30)
            logger.info("Halo 伪弧长延拓（对齐 continuation_PAL_CR3BP + FAMILY_L1Halo_North）")
            logger.info("  种子: L%d %s Halo", libration_point, "北" if halo_class == 0 else "南")
            logger.info("  z_amplitude(参数): %.4f", seed_z_amplitude)
            logger.info("  每支新轨道数 N = %d", n_orbits)
            logger.info("  正向 |DeltaS| = %s, 负向 |DeltaS| = %s", step_size, step_size_negative)
            logger.info(
                "  dc_scheme = %s, DirectionalIncrement = %s",
                dc_scheme,
                directional_increment,
            )
            logger.info("=" * 30)

        orbit_family = OrbitFamily([seed_orbit])

        def _tag_halo_family(orb: Orbit) -> None:
            orb.family_type = "halo"
            orb.parameters["libration_point"] = libration_point
            orb.parameters["halo_class"] = halo_class
            z0 = float(orb.states[0, 2])
            orb.parameters["amplitude_z"] = abs(z0)

        branches: list[tuple[str, float, int, int]] = []
        if direction in ("positive", "both"):
            branches.append(("positive", step_size, 1, 1))
        if direction in ("negative", "both"):
            branches.append(("negative", step_size_negative, 0, -1))

        for br_name, ds_mag, tv, td in branches:
            if verbose:
                logger.info("--- Halo 延拓支: %s (|DeltaS|=%s) ---", br_name, ds_mag)

            sub = self.pseudo_arclength_continuation(
                seed_orbit,
                n_orbits=n_orbits,
                step_size=ds_mag,
                direction="positive" if br_name == "positive" else "negative",
                verbose=verbose,
                TolPAL=TolPAL,
                TolDiffCorr=TolDiffCorr,
                IterMax=IterMax,
                dc_scheme=dc_scheme,
                libration_point=libration_point,
                directional_increment=directional_increment,
                target_vector=tv,
                target_direction=td,
                progress_callback=progress_callback,
            )
            for o in sub.orbits[1:]:
                _tag_halo_family(o)
                orbit_family.add_orbit(o)

        if verbose:
            logger.info("延拓完成：共 %d 条轨道", len(orbit_family))
            z_values = [o.parameters.get("amplitude_z", 0) for o in orbit_family]
            if z_values:
                logger.info("  z_amplitude 范围: [%.4f, %.4f]", min(z_values), max(z_values))

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
