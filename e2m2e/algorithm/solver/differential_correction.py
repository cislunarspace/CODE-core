"""CR3BP 周期轨道的微分修正问题构造与 Rust 内核适配。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from e2m2e.integrators import differential_correction_cr3bp_py

from ...data.types.orbit import Orbit
from ...status import ConvergenceState, FailureCause
from ..dynamics import CR3BP_Dynamics
from ..results import DifferentialCorrectionResult

logger = logging.getLogger(__name__)

_STATE_INDEX_TO_KEY = {
    0: "x",
    1: "y",
    2: "z",
    3: "x_dot",
    4: "y_dot",
    5: "z_dot",
}
_HALO_TIME_RECOVERY_SETUPS = {
    "halo_orbit_fixed_x0",
    "halo_orbit_fixed_z0",
    "axial_orbit_fixed_vz0",
}


@dataclass(frozen=True)
class CorrectionConfig:
    """微分修正的不可变配置。

    对称性、固定参数、自由变量与约束的单一数据对象：由
    ``family.strategies`` 工厂函数构造、``DifferentialCorrection.configure``
    消费。

    Attributes:
        setup_type: 修正配置类型标识符。
        symmetry_condition: 修正所利用的对称性（如 'x_axis'）。
        fixed_parameters: 修正过程中保持不变的参数值。
        free_variables: 牛顿求解器可调整的变量名列表。
        free_variable_indices: 自由变量在状态向量中对应的索引。
        target_conditions: 约束名称到目标值的映射。
        constraint_indices: 约束在状态向量中的求值索引。
        constraint_weights: 各约束的雅可比加权因子。
        constraint_types: 各约束的分类（如 'equality'）。
    """

    setup_type: str
    symmetry_condition: str
    fixed_parameters: dict[str, float] = field(default_factory=dict)
    free_variables: list[str] = field(default_factory=list)
    free_variable_indices: list[int] = field(default_factory=list)
    target_conditions: dict[str, float] = field(default_factory=dict)
    constraint_indices: list[int] = field(default_factory=list)
    constraint_weights: dict[str, float] = field(default_factory=dict)
    constraint_types: dict[str, str] = field(default_factory=dict)


class DifferentialCorrection:
    """Differential correction: iterative refinement of periodic-orbit initial guesses.

    周期轨道微分修正的问题构造入口。

    对称性配置、自由变量和结果编排保留在 Python；残差、STM 雅可比、Newton
    修正与收敛判定只由 Rust CR3BP 内核执行。
    """

    DEFAULT_TOLERANCE = 1e-12
    DEFAULT_MAX_ITERATIONS = 50
    DEFAULT_DAMPING_FACTOR = 1.0
    # Newton 迭代内增广状态（6+36 STM）传播的筛选级容差。修正
    # 闭环只需中间精度（残差评估/雅可比），研究级 1e-12 使单次 STM
    # 传播成本不可接受；1e-10 对 1e-6 级闭合判据精度足够。最终轨道
    # 编排（_create_corrected_orbit）仍用 dynamics 的研究级容差。
    DEFAULT_INTEGRATION_TOLERANCE = 1e-10
    VALID_SETUP_TYPES = [
        "2D_symmetric_x_fixed_x0",
        "2D_symmetric_x_fixed_t",
        "2D_symmetric_y_fixed_y0",
        "3D_symmetric_x_fixed_x0",
        "3D_symmetric_xz_fixed_x0",
        "3D_symmetric_xz_fixed_z0",
        "axial_orbit_fixed_vz0",
        "halo_orbit_fixed_z0",
        "halo_orbit_fixed_x0",
        "spo_fixed_x0",
        "lpo_fixed_x0",
    ]

    def __init__(
        self,
        dynamic: CR3BP_Dynamics,
        target: dict[str, Any] | None = None,
        free_vars: list[str] | None = None,
        integration_rtol: float | None = None,
        integration_atol: float | None = None,
    ) -> None:
        self.dynamics = dynamic
        self.target_conditions = target or {}
        self.free_variables = free_vars or []
        self.tolerance = self.DEFAULT_TOLERANCE
        # 迭代内积分容差：默认修正级 1e-10（DEFAULT_INTEGRATION_TOLERANCE），
        # 与 dynamics.rtol（研究级 1e-12，影响最终轨迹编排）解耦；显式传入
        # 时覆盖。收敛判据 self.tolerance 低于 100×integration_rtol 时 Newton
        # 可能因积分噪声停滞（由 stagnation_limit 兜底盘整为 MAX_ITERATIONS），
        # 需要更紧闭合时应同步调紧 integration_rtol。
        self.integration_rtol = (
            integration_rtol if integration_rtol is not None else self.DEFAULT_INTEGRATION_TOLERANCE
        )
        self.integration_atol = (
            integration_atol if integration_atol is not None else self.integration_rtol
        )
        self.max_iterations = self.DEFAULT_MAX_ITERATIONS
        self.damping_factor = self.DEFAULT_DAMPING_FACTOR
        self.convergence_history: list[dict[str, Any]] = []
        self.error_history: list[float] = []
        self.correction_history: list[float] = []
        self.iteration_count = 0
        self._converged = False
        self.current_error: float | None = None
        self.initial_guess: np.ndarray | None = None
        self.final_solution: np.ndarray | None = None
        self.solution_time: float | None = None
        self.jacobian_matrix: np.ndarray | None = None
        self.constraint_indices: list[int] = []
        self.free_variable_indices: list[int] = []
        self.setup_type: str | None = None
        self.symmetry_condition: str | None = None
        self.fixed_parameters: dict[str, float] = {}
        self.stagnation_limit = 1e-14
        self.divergence_limit = 1e10
        self.performance_stats = {
            "total_time": 0.0,
            "stm_evaluations": 0,
            "constraint_evaluations": 0,
            "jacobian_evaluations": 0,
        }
        self._outcome_status: ConvergenceState | None = None
        self._outcome_cause: FailureCause | None = None
        self._outcome_message = ""

    def configure(self, config: CorrectionConfig) -> DifferentialCorrection:
        """应用修正配置并重置迭代历史，返回 ``self`` 支持链式调用。"""
        self._apply_config(config)
        self._reset_history()
        return self

    def _apply_config(self, config: CorrectionConfig) -> None:
        self.setup_type = config.setup_type
        self.symmetry_condition = config.symmetry_condition
        self.fixed_parameters = dict(config.fixed_parameters)
        self.free_variables = list(config.free_variables)
        self.free_variable_indices = list(config.free_variable_indices)
        self.target_conditions = dict(config.target_conditions)
        self.constraint_indices = list(config.constraint_indices)

    def _reset_history(self) -> None:
        self.convergence_history = []
        self.error_history = []
        self.correction_history = []
        self.iteration_count = 0
        self._converged = False
        self.current_error = None
        self._outcome_status = None
        self._outcome_cause = None
        self._outcome_message = ""

    def _set_outcome(
        self,
        status: ConvergenceState,
        cause: FailureCause,
        message: str,
    ) -> None:
        self._outcome_status = status
        self._outcome_cause = cause
        self._outcome_message = message

    def _require_cr3bp_dynamics(self) -> CR3BP_Dynamics:
        if not isinstance(self.dynamics, CR3BP_Dynamics):
            raise TypeError("DifferentialCorrection 仅支持 CR3BP_Dynamics")
        return self.dynamics

    def _iterate_rust(
        self,
        initial_state: np.ndarray,
        initial_time: float,
        *,
        full_period: bool,
        verbose: bool,
        callback,
    ) -> DifferentialCorrectionResult:
        dynamics = self._require_cr3bp_dynamics()
        target_values = (
            [0.0] * len(self.constraint_indices)
            if full_period
            else [
                float(self.target_conditions[_STATE_INDEX_TO_KEY[index]])
                for index in self.constraint_indices
            ]
        )
        raw = differential_correction_cr3bp_py(
            mu=float(dynamics.system.mu),
            initial_state=[float(value) for value in initial_state],
            initial_time=float(initial_time),
            constraint_indices=self.constraint_indices,
            target_values=target_values,
            free_variable_indices=self.free_variable_indices,
            full_period=full_period,
            recover_halo_time=self.setup_type in _HALO_TIME_RECOVERY_SETUPS,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            stagnation_limit=self.stagnation_limit,
            divergence_limit=self.divergence_limit,
            rtol=self.integration_rtol,
            atol=self.integration_atol,
            max_step=dynamics.max_step,
            sample_count=1000,
        )
        self.error_history = [float(error) for error in raw["error_history"]]
        self.correction_history = [float(value) for value in raw["correction_history"]]
        self.iteration_count = int(raw["iterations"])
        self.current_error = None if raw["residual"] is None else float(raw["residual"])
        self.convergence_history = [
            {
                "iteration": index + 1,
                "error": float(error),
                "state": np.asarray(state, dtype=float).copy(),
                "time": float(time),
                "final_state": np.asarray(final_state, dtype=float).copy(),
            }
            for index, (error, state, time, final_state) in enumerate(
                zip(
                    raw["error_history"],
                    raw["state_history"],
                    raw["time_history"],
                    raw["final_state_history"],
                    strict=True,
                )
            )
        ]
        self.performance_stats["stm_evaluations"] += len(self.error_history)
        self.performance_stats["constraint_evaluations"] += len(self.error_history)
        self.performance_stats["jacobian_evaluations"] += len(self.correction_history)
        status = ConvergenceState(raw["status"])
        cause = FailureCause(raw["cause"])
        self._set_outcome(status, cause, str(raw["message"]))
        self._converged = status is ConvergenceState.CONVERGED

        for index, error in enumerate(self.error_history):
            is_final = index == len(self.error_history) - 1
            if verbose:
                logger.info("迭代 %d: 残差范数 = %.2e", index + 1, error)
            if callback:
                callback(index + 1, error, self._converged and is_final)

        if not self._converged:
            return self._result_from_state(None)

        solution = np.asarray(raw["solution_state"], dtype=float)
        solution_time = float(raw["solution_time"])
        period = solution_time if full_period else 2 * solution_time
        min_valid_period = (
            1e-6 if full_period or self.setup_type not in _HALO_TIME_RECOVERY_SETUPS else 0.5
        )
        if period < min_valid_period:
            self._converged = False
            self._set_outcome(
                ConvergenceState.INFEASIBLE,
                FailureCause.INVALID_PERIOD,
                f"收敛但周期无效: T={period:.6e} < {min_valid_period}",
            )
            return self._result_from_state(None)

        self.final_solution = solution.copy()
        self.solution_time = solution_time
        result = {
            "state": solution,
            "period": period,
            "half_period": period / 2,
            "setup_type": self.setup_type,
            "error": self.current_error,
        }
        return self._result_from_state(self._create_corrected_orbit(result))

    def iterate_correction(
        self, initial_guess, verbose=False, callback=None
    ) -> DifferentialCorrectionResult:
        """修正半周期对称轨道，数值迭代只在 Rust 中执行。"""
        self._require_cr3bp_dynamics()
        self.initial_guess = np.asarray(initial_guess.states[0], dtype=float).copy()
        self.iteration_count = 0
        self._converged = False
        self._outcome_status = None
        self._outcome_cause = None
        self._outcome_message = ""
        half_period = (
            self.fixed_parameters["T_half"]
            if "T_half" in self.fixed_parameters
            else initial_guess.period / 2
        )
        return self._iterate_rust(
            self.initial_guess.copy(),
            float(half_period),
            full_period=False,
            verbose=verbose,
            callback=callback,
        )

    def iterate_full_period_correction(
        self, initial_guess, verbose=False, callback=None
    ) -> DifferentialCorrectionResult:
        """修正无对称性假设的全周期闭合轨道，数值迭代只在 Rust 中执行。"""
        self._require_cr3bp_dynamics()
        self.initial_guess = np.asarray(initial_guess.states[0], dtype=float).copy()
        self.iteration_count = 0
        self._converged = False
        self._outcome_status = None
        self._outcome_cause = None
        self._outcome_message = ""
        return self._iterate_rust(
            self.initial_guess.copy(),
            float(initial_guess.period),
            full_period=True,
            verbose=verbose,
            callback=callback,
        )

    def _result_from_state(self, orbit: Orbit | None) -> DifferentialCorrectionResult:
        if self._converged:
            status, cause = ConvergenceState.CONVERGED, FailureCause.NONE
            message = self._outcome_message
        elif self._outcome_status is not None and self._outcome_cause is not None:
            status, cause = self._outcome_status, self._outcome_cause
            message = self._outcome_message
        else:
            status, cause = ConvergenceState.MAX_ITERATIONS, FailureCause.MAX_ITERATIONS_REACHED
            message = "达到最大迭代次数"
        return DifferentialCorrectionResult(
            status=status,
            cause=cause,
            message=message,
            orbit=orbit,
            iterations=self.iteration_count,
            residual=self.current_error,
            residual_history=tuple(float(error) for error in self.error_history),
        )

    def _create_corrected_orbit(self, result: dict[str, Any]) -> Orbit:
        """由 Rust 修正后的初始状态和周期编排完整轨迹。"""
        full_period = float(result["period"])
        initial_state = np.asarray(result["state"], dtype=float)
        prop_result = self.dynamics.propagate(
            initial_state,
            (0, full_period),
            t_eval=np.linspace(0, full_period, 1000),
        )
        final_state = prop_result["states"][-1]
        closure_error = float(np.linalg.norm(final_state - initial_state))

        # 非 Halo/Axial/SPO 的轨道可用一次速度微调消除积分截断误差。
        if closure_error > 1e-10 and self.setup_type not in {
            "halo_orbit_fixed_x0",
            "halo_orbit_fixed_z0",
            "axial_orbit_fixed_vz0",
            "spo_fixed_x0",
        }:
            closure_error_vector = final_state - initial_state
            if (
                np.linalg.norm(closure_error_vector[:3]) > 1e-14
                and np.linalg.norm(closure_error_vector[3:]) > 1e-14
            ):
                new_state = initial_state.copy()
                new_state[4] -= 0.5 * closure_error_vector[4]
                retry = self.dynamics.propagate(
                    new_state,
                    (0, full_period),
                    t_eval=np.linspace(0, full_period, 1000),
                )
                retry_final_state = retry["states"][-1]
                retry_error = float(np.linalg.norm(retry_final_state - new_state))
                # 只在微调确实更好时采用：否则 states 轨迹与 closure_error
                # 会分别来自两次传播（#646 实测 3:1 成员标称闭合 5.4e-7，
                # 数组首末差却达 3.8e-4，分类器据此误判 non_periodic）。
                if retry_error < closure_error:
                    prop_result = retry
                    initial_state = new_state
                    closure_error = retry_error

        orbit = Orbit(
            states=np.array(prop_result["states"], copy=True),
            times=prop_result["time"].copy(),
            system=self.dynamics.system,
        )
        orbit.period = full_period
        orbit.is_periodic = closure_error < self.tolerance
        orbit.family_type = self._infer_family_type()
        orbit.closure_error = closure_error
        return orbit

    def _infer_family_type(self):
        if self.setup_type and "axial" in self.setup_type:
            return "axial"
        if self.setup_type and ("3D" in self.setup_type or "halo" in self.setup_type):
            return "halo"
        if self.setup_type and "2D" in self.setup_type:
            return "lyapunov"
        return None

    def check_convergence(self):
        """返回上一次 Rust 修正是否收敛。"""
        return self._converged

    def get_convergence_history(self):
        """返回 Rust 内核回传的收敛历史。"""
        return {
            "errors": self.error_history,
            "corrections": self.correction_history,
            "iterations": self.iteration_count,
            "status": self._outcome_status,
            "cause": self._outcome_cause,
            "message": self._outcome_message,
        }

    def __str__(self):
        return f"DifferentialCorrection(setup={self.setup_type})"

    def __repr__(self):
        return (
            f"DifferentialCorrection(dynamic={self.dynamics}, "
            f"setup={self.setup_type}, tol={self.tolerance})"
        )
