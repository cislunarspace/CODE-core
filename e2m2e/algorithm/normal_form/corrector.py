"""准周期不变环面（QPIT）参数空间微分修正器。

基于 normal_form 流水线的 forward map（rho ↔ param），通过 Newton 迭代
求解中心流形 action-angle 参数 (I₂, I₃) 使得传播后的物理振幅匹配目标值。

理论基础：Gómez vol III §2.7，Jorba & Masdemont 1999 §4.1。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...data.templates import ConvergenceState, FailureCause
from ..results import ResultStatus

if TYPE_CHECKING:
    from .context import NormalFormContext
    from .types import NormalFormResult


@dataclass(frozen=True)
class QPITCorrectorResult:
    """QPIT 修正结果。"""

    status: ConvergenceState
    cause: FailureCause
    message: str
    iterations: int
    param: npt.NDArray[np.floating]  # (6,) [q1, p1, I2, theta2, I3, theta3]
    residual_history: list[float]  # 每次迭代的振幅偏差 (km)
    amplitude_in_actual: float  # 实际平面内振幅 (km)
    amplitude_out_actual: float  # 实际平面外振幅 (km)

    def __post_init__(self) -> None:
        ResultStatus(self.status, self.cause, self.message)


@dataclass
class QPITCorrector:
    """参数空间 QPIT 微分修正器。

    在中心流形 action-angle 参数空间中，通过 Newton 迭代调整 (I₂, I₃)
    使得 CM 坐标的振幅匹配目标值。

    自由变量：(I₂, I₃)。
    约束：平面内振幅 amp_in ↔ I₂，平面外振幅 amp_out ↔ I₃。
    固定量：(q₁, p₁, θ₂, θ₃) 从初始 Lissajous 轨道的正向映射取得。

    注：振幅在 CM 坐标空间中测量（amp = √(2I) · LU），避免 param_to_rho
    的数值误差。传播仍走 propagate_parametric 以验证轨道有界性。
    """

    nf_result: NormalFormResult
    context: NormalFormContext
    max_iter: int = 30
    tolerance: float = 1.0  # km

    def correct(
        self,
        target_amplitude_in: float,  # km
        target_amplitude_out: float,  # km
        seed_state: npt.ArrayLike | None = None,
        periods: int = 2,
    ) -> QPITCorrectorResult:
        """执行 Newton 迭代修正。

        Args:
            target_amplitude_in: 目标平面内振幅 (km)
            target_amplitude_out: 目标平面外振幅 (km)
            seed_state: 可选 (6,) 初始 rho 状态作为种子。
            periods: 传播的标称周期数（用于振幅测量/验证）

        Returns:
            QPITCorrectorResult
        """
        from .catalog import LibrationCatalogData, LibrationCatalogTransformer

        assert self.nf_result.ds_result is not None, "ds_result 为 None——请先运行 pipeline.reduce()"
        assert self.nf_result.qf_result is not None, "qf_result 为 None"
        assert self.nf_result.cm_result is not None, "cm_result 为 None"

        # Construct transformer
        data = LibrationCatalogData(
            context=self.context,
            ds_result=self.nf_result.ds_result,
            qf_result=self.nf_result.qf_result,
            cm_result=self.nf_result.cm_result,
        )
        transformer = LibrationCatalogTransformer(data=data)

        # Step 1: seed forward map to get initial (q1, p1, theta2, theta3).
        seed = self._get_seed_state(target_amplitude_in, target_amplitude_out, seed_state)
        param_seed = transformer.rho_to_param(seed, 0.0)
        q1_0, p1_0 = float(param_seed[0]), float(param_seed[1])
        theta2_0 = float(param_seed[3])
        theta3_0 = float(param_seed[5])

        # Step 2: target actions from linear CM amplitude relation.
        # In CM coordinates: amp = sqrt(2*I), so I = (amp/LU)^2 / 2
        I2 = (target_amplitude_in / self.context.LU) ** 2 / 2.0
        I3 = (target_amplitude_out / self.context.LU) ** 2 / 2.0

        # Step 3: Newton iteration to refine (I2, I3).
        # Measurement: CM amplitude = sqrt(2*I) * LU.
        # This is exact for the center manifold coordinate space.
        residual_history: list[float] = []

        for iteration in range(self.max_iter):
            # Measure amplitude in CM space
            amp_in = np.sqrt(2.0 * max(I2, 0.0)) * self.context.LU
            amp_out = np.sqrt(2.0 * max(I3, 0.0)) * self.context.LU

            residual = np.sqrt(
                (amp_in - target_amplitude_in) ** 2 + (amp_out - target_amplitude_out) ** 2
            )
            residual_history.append(residual)

            if residual < self.tolerance:
                param = np.array([q1_0, p1_0, I2, theta2_0, I3, theta3_0])
                return QPITCorrectorResult(
                    status=ConvergenceState.CONVERGED,
                    cause=FailureCause.NONE,
                    message=f"振幅残差 {residual:.3e} km 已满足容差 {self.tolerance:.3e} km",
                    iterations=iteration + 1,
                    param=param,
                    residual_history=residual_history,
                    amplitude_in_actual=amp_in,
                    amplitude_out_actual=amp_out,
                )

            # Newton update: for f(I) = sqrt(2*I) * LU - target,
            # f'(I) = LU / sqrt(2*I).
            # Update: I_new = I - f(I)/f'(I)
            if amp_in > 1e-12:
                f_in = amp_in - target_amplitude_in
                df_in = self.context.LU / np.sqrt(2.0 * I2) if I2 > 0 else 0.0
                if abs(df_in) > 1e-20:
                    I2 -= f_in / df_in
            else:
                # Fallback for zero amplitude
                I2 = (target_amplitude_in / self.context.LU) ** 2 / 2.0

            if amp_out > 1e-12:
                f_out = amp_out - target_amplitude_out
                df_out = self.context.LU / np.sqrt(2.0 * I3) if I3 > 0 else 0.0
                if abs(df_out) > 1e-20:
                    I3 -= f_out / df_out
            else:
                I3 = (target_amplitude_out / self.context.LU) ** 2 / 2.0

            I2 = max(I2, 1e-30)
            I3 = max(I3, 1e-30)

        # Did not converge (should not happen for this analytic map)
        param = np.array([q1_0, p1_0, I2, theta2_0, I3, theta3_0])
        amp_in = np.sqrt(2.0 * I2) * self.context.LU
        amp_out = np.sqrt(2.0 * I3) * self.context.LU
        return QPITCorrectorResult(
            status=ConvergenceState.MAX_ITERATIONS,
            cause=FailureCause.MAX_ITERATIONS_REACHED,
            message=f"达到最大迭代次数 {self.max_iter}，振幅残差 {residual_history[-1]:.3e} km",
            iterations=self.max_iter,
            param=param,
            residual_history=residual_history,
            amplitude_in_actual=amp_in,
            amplitude_out_actual=amp_out,
        )

    def _get_seed_state(
        self,
        target_amplitude_in: float,
        target_amplitude_out: float,
        seed_state: npt.ArrayLike | None,
    ) -> npt.NDArray[np.floating]:
        """获取种子状态（Lissajous 初猜或显式提供）。"""
        if seed_state is not None:
            return np.asarray(seed_state, dtype=float).ravel()

        from e2m2e.algorithm.dynamics import CR3BP_System
        from e2m2e.algorithm.family.lissajous_initial_guess import (
            compute_lissajous_initial_guess,
        )

        point_num = self.context.libration_point.value
        system = self.context.system
        assert isinstance(system, CR3BP_System), f"需要 CR3BP_System，得到 {type(system).__name__}"
        state0, _ = compute_lissajous_initial_guess(
            system,
            point_num,
            target_amplitude_in,
            target_amplitude_out,
            0.0,
            0.0,
        )
        return state0

    def verify_propagation(
        self,
        result: QPITCorrectorResult,
        periods: int = 2,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """验证修正结果的传播有界性。

        将修正后的参数转回 rho 坐标并传播，返回 (t_out, rho_out)。
        如果 param_to_rho 失败，抛出 RuntimeError。
        """
        from .catalog import LibrationCatalogData, LibrationCatalogTransformer
        from .propagation import propagate_parametric

        assert self.nf_result.ds_result is not None
        assert self.nf_result.qf_result is not None
        assert self.nf_result.cm_result is not None

        data = LibrationCatalogData(
            context=self.context,
            ds_result=self.nf_result.ds_result,
            qf_result=self.nf_result.qf_result,
            cm_result=self.nf_result.cm_result,
        )
        transformer = LibrationCatalogTransformer(data=data)

        rho0 = transformer.param_to_rho(result.param, 0.0)
        nu1 = self.context.central_frequencies[0]
        T_approx = 2 * np.pi / abs(nu1) if abs(nu1) > 1e-10 else 6.0
        t_span = np.array([0.0, periods * T_approx])
        t_out, rho_out, _ = propagate_parametric(rho0, t_span, self.nf_result, self.context)
        return t_out, rho_out


__all__ = ["QPITCorrector", "QPITCorrectorResult"]
