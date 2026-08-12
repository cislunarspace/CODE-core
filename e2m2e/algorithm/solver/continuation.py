"""轨道族延拓算法模块

提供自然参数延拓和伪弧长延拓方法，用于沿轨道族参数方向逐步生成相邻轨道。
"""

from __future__ import annotations

import logging

import numpy as np

from ...data.templates import ConvergenceState, FailureCause
from ...data.types.orbit import Orbit, OrbitFamily
from ..dynamics import CR3BP_Dynamics
from ..family.halo_family import (
    generate_halo_family,
    generate_halo_seed_orbit,
    halo_pseudo_arclength_continuation,
)
from ..results import ContinuationResult
from .differential_correction import DifferentialCorrection

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
        self._outcome_status: ConvergenceState | None = None
        self._outcome_cause: FailureCause | None = None
        self._outcome_message = ""

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

        temp_orbits_with_steps: list[tuple[Orbit, int]] = []

        if forward:
            if verbose:
                logger.info("--- 正向延拓 (参数增大方向) ---")
            temp_orbits_with_steps.extend(
                self._sweep(
                    seed_orbit=seed_orbit,
                    target=param_max,
                    direction_sign=+1,
                    direction_label="正向",
                    step_size=step_size,
                    param_index=param_index,
                    verbose=verbose,
                )
            )

        if backward:
            if verbose:
                logger.info("--- 反向延拓 (参数减小方向) ---")
            temp_orbits_with_steps.extend(
                self._sweep(
                    seed_orbit=seed_orbit,
                    target=param_min,
                    direction_sign=-1,
                    direction_label="反向",
                    step_size=step_size,
                    param_index=param_index,
                    verbose=verbose,
                )
            )

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

        return self._continuation_result(orbit_family)

    def _sweep(
        self,
        seed_orbit: Orbit,
        target: float,
        direction_sign: int,
        *,
        direction_label: str,
        step_size: float,
        param_index: int,
        verbose: bool,
    ) -> list[tuple[Orbit, int]]:
        """沿参数方向单向延拓（``natural_continuation`` 内部使用）

        ``direction_sign=+1`` 表示正向（参数增大），``-1`` 表示反向（参数减小）。
        由 ``natural_continuation`` 在正向、反向各调用一次，消除原先两块几乎
        完全重复的 forward/backward 代码。

        Args:
            seed_orbit: 起始轨道（不会被原地修改；内部 ``.copy()``）。
            target: 参数目标值；正向时延拓至 ``current >= target`` 终止，
                反向时延拓至 ``current <= target`` 终止。
            direction_sign: ``+1`` 或 ``-1``。
            direction_label: 仅用于日志（"正向"/"反向"）。
            step_size: 初始步长（函数内部按 ``step_size_adaptation`` 策略调整）。
            param_index: 0–5 对应 ``states[0, i]``，6 对应 ``orbit.period``。
            verbose: 是否打印详细进度日志。

        Returns:
            ``(orbit, signed_step_index)`` 列表，``signed_step_index =
            direction_sign * (i + 1)``。触底或达 ``max_orbits`` 上限后提前返回。
        """
        results: list[tuple[Orbit, int]] = []
        current_orbit = seed_orbit.copy()
        i = 0

        while True:
            if param_index < 6:
                current_param_value = current_orbit.states[0, param_index]
            else:
                current_param_value = current_orbit.period

            if direction_sign > 0:
                if current_param_value >= target:
                    break
            else:
                if current_param_value <= target:
                    break

            if param_index < 6:
                guess_orbit = current_orbit.copy()
                guess_orbit.states[0, param_index] += direction_sign * step_size
            else:
                # 延拓参数为周期时，period 必有值（微分修正后赋值）。
                assert current_orbit.period is not None
                guess_orbit = current_orbit.copy()
                guess_orbit.period = current_orbit.period + direction_sign * step_size * 2

            result = self.correction.iterate_correction(guess_orbit, verbose=verbose)
            orbit = result.orbit

            if orbit is not None and result.status is ConvergenceState.CONVERGED:
                signed_step = direction_sign * (i + 1)
                orbit.metadata["continuation_step"] = signed_step
                results.append((orbit, signed_step))

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
                        logger.info("  %s延拓进度：已完成 %d 条轨道", direction_label, i + 1)

                if self.step_size_adaptation:
                    if result.iterations < 3:
                        step_size = min(step_size * self.step_growth_factor, self.max_step_size)
                    elif result.iterations > 8:
                        step_size = max(step_size * self.step_reduction_factor, self.min_step_size)
            else:
                self.continuation_stats["failed_steps"] += 1

                step_size *= self.step_reduction_factor
                if step_size < self.min_step_size:
                    self._outcome_status = ConvergenceState.STAGNATED
                    self._outcome_cause = FailureCause.STAGNATION_DETECTED
                    self._outcome_message = "步长过小，延拓终止"
                    if verbose:
                        logger.info(
                            "%s步长过小，延拓终止于第 %d 条轨道",
                            direction_label,
                            1 + len(results),  # +1 to align with len(orbit_family)
                        )
                    break

                if verbose:
                    logger.info("  第 %d 步修正失败，减小步长至 %.6f", i + 1, step_size)

            i += 1

            if 1 + len(results) >= self.max_orbits:
                self._outcome_status = ConvergenceState.MAX_ITERATIONS
                self._outcome_cause = FailureCause.MAX_ITERATIONS_REACHED
                self._outcome_message = "达到最大轨道数限制"
                break

        return results

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
        ``G = [F; (Xnew-X)·Xdot - ΔS]``；内层用 ``Xnew`` 计算 ``F`` （与 MATLAB
        中仅用固定 ``X`` 相比更一致）。每步后用微分修正闭合。

        Args:
            seed_orbit: 种子轨道（Orbit 对象）。
            n_orbits: 本支要生成的新轨道条数（与 MATLAB 中 N 一致）。
            step_size: 伪弧长步长 ΔS 的模长（正数；direction 决定符号）。
            direction: 延拓方向，取 positive 或 negative；双侧延拓请调用两次，
                或改用 halo_pseudo_arclength_continuation(direction='both')。
            dc_scheme: 微分修正方案，见 Continuation 类文档字符串。
            target_vector: 与 MATLAB TargetVector 对应的 0 基下标
                （0=rx, 1=rz, 2=vy, 3=T/2）。

        Returns:
            OrbitFamily: 仅含种子与本支新轨道（不重复添加种子）。
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

        # [FIX] directional_increment 的原始实现每步根据 Xdot 重新判定方向,在
        # 族流形折叠点(fold)处 Xdot 渐近过零变号,形成 2-周期环振荡(steps 65+
        # 退回 z≈0.085,详见 tests/algorithm/correction/test_pal_stagnation.py 回归测试)。
        #
        # 修复方案:directional_increment 在 PAL 折叠点本质上不稳定。PAL 的
        # 核心优势就是能沿弧长穿过折叠,此时目标变量(tv)会自然反转。
        # 修复:让 directional_increment 仅在**延拓**的初始方向起作用,穿过
        # 折叠点后**不再强制方向**。具体:初始用 Xdot 决定 dir_sign 起点,
        # 一旦检测到"目标变量穿越期望方向"则不再翻转 — 信任 PAL 的自然
        # 行为。滞回参数保证噪声不触发反向。
        def _initial_dir_sign() -> float:
            if not directional_increment:
                return 1.0
            return 1.0 if td * (ds * Xdot)[tv] > 0 else -1.0

        dir_sign = _initial_dir_sign()
        prev_X_for_dir: np.ndarray | None = None
        # 滞回:翻转 dir_sign 后至少保持 K 步不再翻,防止噪声来回触发
        _hysteresis_steps_remaining = 0
        _HYSTERESIS_K = 5
        # 是否已经"穿越折叠点":穿越后不再使用 dir_sign 翻折,
        # 避免 dir_sign 在噪声中反复切换
        _crossed_fold = False

        # 用于停滞检测:上一条已收敛轨道的状态
        prev_orbit_state: np.ndarray | None = None
        # 连续无进展步数(用于触发步长缩减)
        stagnation_count = 0
        # 动态步长(支持自适应缩减)
        current_step_size = float(step_size)
        # 记录终止原因(给调用方诊断)
        self._outcome_status = None
        self._outcome_cause = None
        self._outcome_message = ""

        for n in range(n_orbits):
            if verbose and (n + 1) % 5 == 0:
                logger.info("--- 延拓第 %d/%d 条轨道 ---", n + 1, n_orbits)

            # [FIX] 反馈式 dir_sign + 滞回 + 单次穿越:
            # 1. 滞回 5 步内不重判(防噪声)
            # 2. 仅在"_crossed_fold=False"时检测翻转
            # 3. 一旦翻转一次,就标记为已穿越,不再翻转(避免来回)
            # 这让 PAL 在穿过折叠点后沿流形自然反向,不再振荡。
            if directional_increment and prev_X_for_dir is not None and not _crossed_fold:
                if _hysteresis_steps_remaining > 0:
                    _hysteresis_steps_remaining -= 1
                else:
                    _delta_tv_actual = X[tv] - prev_X_for_dir[tv]
                    # 当前 dir_sign 让目标变量沿 td 方向增大,若上一步 X[tv] 反向
                    # 说明已越过折叠点,翻转 dir_sign 让 PAL 沿流形继续走
                    if dir_sign * (td * _delta_tv_actual) < 0:
                        dir_sign = -dir_sign
                        _hysteresis_steps_remaining = _HYSTERESIS_K
                        _crossed_fold = True

            # [FIX] 使用 dir_sign 缩放预测(不每步重判 Xdot)
            Xnew = X + dir_sign * ds * Xdot

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
                # 切向量同向化(对 SVD 任意符号),确保 PAL 约束的 Newton 步稳定
                if np.dot(Xdot_new, Xdot) < 0:
                    Xdot_new = -Xdot_new

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

            # [FIX] PAL 牛顿迭代解本身已满足 PAL 约束 + F=0 物理约束,正常
            # 情况下它就在 Halo 流形上。仅检查绝对物理范围:
            #  - x ∈ (0.75, 1.05) : L1 halo 平面 Lyapunov 分岔附近的 x 范围
            #  - |z| ∈ (1e-3, 0.55) : z 振幅的非平凡物理范围
            #  - T/2 ∈ (0.35, π/2) : 周期下界 0.7(短周期近似),上界
            #    π ≈ 3.14 即 T/2 < π/2 ≈ 1.57(物理上 2:1 共振周期,实际
            #    L1 halo T/2 ≤ 1.38)
            # 旧版 T/2 < 1.35 把所有 L1 halo 折叠点附近的轨道误判
            # 触发回退,导致 158/160 步错误地"回退为欧拉预测初值",用户
            # 看到的"延拓到某范围不再继续"实际是回退后流形推进能力丧失。
            # **不要**再检查与欧拉预测的距离:PAL Newton 解是物理正解,
            # 欧拉预测只是切线一步,不能用作回退判据。
            _x, _z, _tf2 = X[0], X[1], X[3]
            pal_plausible = (
                0.75 < _x < 1.05 and abs(_z) > 1e-3 and abs(_z) < 0.55 and 0.35 < _tf2 < np.pi / 2
            )
            if not pal_plausible:
                if verbose:
                    logger.warning(
                        "  PAL 结果超出物理范围 (x=%.4f, z=%.4f, T/2=%.4f),回退为欧拉预测初值",
                        _x,
                        _z,
                        _tf2,
                    )
                X = X_predictor_only.copy()
                # [FIX] 同步把 Xdot 也回退到预测方向(以欧拉预测结果为基准重新求切向量)
                # 防止使用 stale Xdot 造成下一轮 PAL 牛顿迭代从错误初值起步
                # 实际不重新求,因为下面会通过差分修正收敛,这里不动 Xdot

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

            result = self.correction.iterate_correction(guess_orbit, verbose=False)
            orbit = result.orbit

            # PAL 初值在固定 x0 下常落入寄生根或与 STM 牛顿不兼容；与种子生成一致改用固定 z0 再试
            if (
                orbit is None or result.status is not ConvergenceState.CONVERGED
            ) and dc_scheme == "matlab_halo_type1":
                self.correction.setup_halo_orbit_fixed_z0(
                    z0=SV0_corr[2], libration_point=libration_point
                )
                result = self.correction.iterate_correction(guess_orbit, verbose=False)
            orbit = result.orbit

            if orbit is not None and result.status is ConvergenceState.CONVERGED:
                orbit_family.add_orbit(orbit)
                family_states.append(orbit.states[0].copy())

                # [FIX] 停滞检测:若新轨道与上一条几乎重合,认为 PAL 在折叠点处
                # 振荡,缩减步长重试(类似自然延拓的自适应策略)。阈值取当前
                # 步长的一定比例,避免在低步长下误判。
                new_state = orbit.states[0]
                if prev_orbit_state is not None:
                    progress = float(np.linalg.norm(new_state - prev_orbit_state))
                    stagnation_threshold = 0.1 * current_step_size
                    if progress < stagnation_threshold:
                        stagnation_count += 1
                        if stagnation_count >= 3 and current_step_size > 1e-5:
                            # 连续 3 步无实质进展,缩减步长
                            new_step = current_step_size * 0.5
                            if verbose:
                                logger.warning(
                                    "  轨道 %d: 连续 %d 步无实质进展 (Δ=%.2e),"
                                    " 步长从 %.5f 缩至 %.5f",
                                    n + 1,
                                    stagnation_count,
                                    progress,
                                    current_step_size,
                                    new_step,
                                )
                            current_step_size = new_step
                            # 同步更新 ds 和 dir_sign 应用的步长
                            ds = float(step_sign * current_step_size)
                            stagnation_count = 0
                    else:
                        stagnation_count = 0
                prev_orbit_state = new_state.copy()

                assert orbit.period is not None
                X = np.array(
                    [
                        orbit.states[0, 0],
                        orbit.states[0, 2],
                        orbit.states[0, 4],
                        orbit.period / 2,
                    ]
                )
                _, dF = compute_F_and_dF_symmetric_xz_plane(X, orbit.states[0].copy(), dynamics)
                Xdot_new = compute_tangent_vector(dF)
                # 切向量同向化(对 SVD 任意符号)
                if np.dot(Xdot_new, Xdot) < 0:
                    Xdot_new = -Xdot_new
                Xdot = Xdot_new
                # [FIX] 记录这一步收敛后的 X,供下一步反馈式方向调整
                prev_X_for_dir = X.copy()

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
                self._outcome_status = ConvergenceState.FAILED
                self._outcome_cause = FailureCause.UNKNOWN
                self._outcome_message = "微分修正失败"
                break

        if verbose:
            logger.info("伪弧长延拓完成：共生成 %d 条轨道", len(orbit_family))

        return self._continuation_result(orbit_family)

    def _continuation_result(self, family: OrbitFamily) -> ContinuationResult:
        if self._outcome_status is None or self._outcome_cause is None:
            status, cause, message = (
                ConvergenceState.CONVERGED,
                FailureCause.NONE,
                "延拓完成",
            )
        else:
            status, cause, message = (
                self._outcome_status,
                self._outcome_cause,
                self._outcome_message,
            )
        return ContinuationResult(
            status=status,
            cause=cause,
            message=message,
            family=family,
            steps=self.continuation_stats["successful_steps"],
            step_size=self.step_size,
        )

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
            "status": self._outcome_status,
            "cause": self._outcome_cause,
            "message": self._outcome_message,
        }

    # Halo 专用编排（generate_halo_seed_orbit / generate_halo_family /
    # halo_pseudo_arclength_continuation）已迁出到 ``halo_family`` 模块；
    # 本文件末尾以方法重绑定的形式对外保留同名 API。

    def __repr__(self):
        return (
            f"Continuation(corrector={self.correction}, "
            f"param={self.continuation_parameter}, step={self.step_size})"
        )


# Halo 专用方法在 ``halo_family`` 中以 ``(continuation, ...)`` 函数定义。
# 把它们以方法形式重绑到 ``Continuation`` 类，等价于调用方写
# ``continuation.halo_xxx(...)``（第一个参数由 Python 自动注入 ``self``）。
Continuation.generate_halo_seed_orbit = generate_halo_seed_orbit  # type: ignore[attr-defined]
Continuation.generate_halo_family = generate_halo_family  # type: ignore[attr-defined]
Continuation.halo_pseudo_arclength_continuation = halo_pseudo_arclength_continuation  # type: ignore[attr-defined]
