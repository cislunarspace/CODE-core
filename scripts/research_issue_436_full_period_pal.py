"""Issue #436：平面 SPO/LPO 全周期伪弧长延拓数值原型。

此脚本是方法调研产物，不接入 ``Continuation``、Facade 或公开模型。它把
平面初态与周期写成 q=(x0, y0, vx0, vy0, T)，以完整周期闭合、相位条件和
伪弧长条件求解。输出 JSON 记录供 ADR 和后续实现 Issue 复现使用。

示例：
    make dev
    uv run python scripts/research_issue_436_full_period_pal.py \
        --family lpo --steps 5 --output /tmp/lpo-pal.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System
from e2m2e.algorithm.family.lpo_initial_guess import compute_lpo_initial_guess
from e2m2e.algorithm.family.spo_initial_guess import compute_spo_initial_guess
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from e2m2e.data.types.orbit import Orbit

EARTH_MOON_MU = 0.01215058560962404
PLANAR_STATE_INDICES = np.array([0, 1, 3, 4])
# q 的各分量量级不同；将周期按 10 TU 归一化后再计算弧长和条件数。
SCALES = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
# 积分和 STM 误差会把自治系统的理论零奇异值抬升；用相对阈值定义数值有效秩。
RANK_RELATIVE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class StepRecord:
    step: int
    initial_state: list[float]
    period: float
    closure_inf_norm: float
    phase_residual: float
    arc_residual: float
    pal_iterations: int
    tangent_system_rank: int
    tangent_system_condition: float
    augmented_system_rank: int
    augmented_system_condition: float
    jacobi_constant: float
    jacobi_drift: float
    libration_distance_mean_km: float


def _system_and_dynamics() -> tuple[CR3BP_System, CR3BP_Dynamics]:
    system = CR3BP_System(
        mu=EARTH_MOON_MU, primary="Earth", secondary="Moon"
    )._with_default_scales()
    return system, CR3BP_Dynamics(system=system)


def _orbit_from_state(system: CR3BP_System, state: np.ndarray, period: float) -> Orbit:
    orbit = Orbit(states=state.reshape(1, -1), times=np.array([0.0]), system=system)
    orbit.period = period
    return orbit


def _initial_seed(dynamics: CR3BP_Dynamics, family: str, point: int) -> Orbit:
    if family == "spo":
        state, period = compute_spo_initial_guess(dynamics.system, point, amplitude_km=1000.0)
        setup = DifferentialCorrection.setup_spo_fixed_x0
    else:
        state, period = compute_lpo_initial_guess(dynamics.system, point, amplitude_km=1000.0)
        setup = DifferentialCorrection.setup_lpo_fixed_x0

    corrector = DifferentialCorrection(dynamics)
    setup(corrector, x0=float(state[0]), libration_point=point)
    result = corrector.iterate_full_period_correction(
        _orbit_from_state(dynamics.system, state, period), verbose=False
    )
    if result.orbit is None:
        raise RuntimeError(f"{family.upper()} 种子全周期修正失败：{result.message}")
    return result.orbit


def _q_from_orbit(orbit: Orbit) -> np.ndarray:
    assert orbit.period is not None
    return np.array([*orbit.states[0, PLANAR_STATE_INDICES], orbit.period], dtype=float)


def _state_from_q(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], q[1], 0.0, q[2], q[3], 0.0])


def _rank_and_condition(matrix: np.ndarray) -> tuple[int, float]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = singular_values[0] * RANK_RELATIVE_TOLERANCE
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = float(singular_values[0] / singular_values[rank - 1])
    return rank, condition


class FullPeriodPal:
    """仅供调研的平面完整周期 PAL 实现。"""

    def __init__(self, dynamics: CR3BP_Dynamics, *, tolerance: float, max_iterations: int):
        self.dynamics = dynamics
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def closure_and_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = _state_from_q(q)
        result = self.dynamics.propagate(
            state,
            (0.0, float(q[4])),
            t_eval=[float(q[4])],
            with_stm=True,
            with_jacobi=False,
        )
        final_state = result["states"][-1]
        stm = result["stm"][-1]
        closure = final_state[PLANAR_STATE_INDICES] - state[PLANAR_STATE_INDICES]
        state_derivative = self.dynamics.equations_of_motion(float(q[4]), final_state)
        jacobian = np.column_stack(
            (
                stm[np.ix_(PLANAR_STATE_INDICES, PLANAR_STATE_INDICES)] - np.eye(4),
                state_derivative[PLANAR_STATE_INDICES],
            )
        )
        return closure, jacobian * SCALES

    def phase_and_jacobian(
        self, q: np.ndarray, reference_q: np.ndarray
    ) -> tuple[float, np.ndarray]:
        reference_flow = self.dynamics.equations_of_motion(0.0, _state_from_q(reference_q))[
            PLANAR_STATE_INDICES
        ]
        flow_norm = float(np.linalg.norm(reference_flow))
        if flow_norm == 0.0:
            raise RuntimeError("相位参考点的平面流速为零，不能定义相位条件")
        phase = float(np.dot(q[:4] - reference_q[:4], reference_flow) / flow_norm)
        jacobian = np.zeros(5)
        jacobian[:4] = reference_flow / flow_norm * SCALES[:4]
        return phase, jacobian

    def tangent(self, q: np.ndarray) -> tuple[np.ndarray, int, float]:
        _, closure_jacobian = self.closure_and_jacobian(q)
        _, phase_jacobian = self.phase_and_jacobian(q, q)
        tangent_system = np.vstack((closure_jacobian, phase_jacobian))
        _, _, vh = np.linalg.svd(tangent_system)
        tangent = vh[-1]
        return tangent / np.linalg.norm(tangent), *_rank_and_condition(tangent_system)

    def refine_seed(self, q: np.ndarray) -> np.ndarray:
        """以初始 x0 为锚点，把既有三残差种子收紧为全平面闭合解。"""
        reference_q = q.copy()
        x0 = float(q[0])
        for _ in range(self.max_iterations):
            closure, closure_jacobian = self.closure_and_jacobian(q)
            phase, phase_jacobian = self.phase_and_jacobian(q, reference_q)
            residual = np.concatenate((closure, [phase, q[0] - x0]))
            if np.max(np.abs(residual)) <= self.tolerance:
                return q
            jacobian = np.vstack(
                (
                    closure_jacobian,
                    phase_jacobian,
                    np.array([SCALES[0], 0.0, 0.0, 0.0, 0.0]),
                )
            )
            delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            q = q + delta * SCALES
            if q[4] <= 0.0:
                raise RuntimeError("种子收紧产生非正周期")
        raise RuntimeError("种子收紧未在最大迭代次数内收敛")

    def correct_step(
        self, previous_q: np.ndarray, tangent: np.ndarray, ds: float
    ) -> tuple[np.ndarray, int, int, float, int, float, float, float]:
        q = previous_q + ds * tangent * SCALES
        for iteration in range(1, self.max_iterations + 1):
            if q[4] <= 0.0:
                raise RuntimeError("PAL 预测产生非正周期")
            closure, closure_jacobian = self.closure_and_jacobian(q)
            phase, phase_jacobian = self.phase_and_jacobian(q, previous_q)
            arc = float(np.dot((q - previous_q) / SCALES, tangent) - ds)
            residual = np.concatenate((closure, [phase, arc]))
            augmented_jacobian = np.vstack((closure_jacobian, phase_jacobian, tangent))
            if np.max(np.abs(residual)) <= self.tolerance:
                tangent_rank, tangent_condition = self._tangent_metrics(previous_q)
                augmented_rank, augmented_condition = _rank_and_condition(augmented_jacobian)
                return (
                    q,
                    iteration,
                    tangent_rank,
                    tangent_condition,
                    augmented_rank,
                    augmented_condition,
                    phase,
                    arc,
                )
            delta = np.linalg.lstsq(augmented_jacobian, -residual, rcond=None)[0]
            q = q + delta * SCALES
        raise RuntimeError("PAL 校正未在最大迭代次数内收敛")

    def _tangent_metrics(self, q: np.ndarray) -> tuple[int, float]:
        _, rank, condition = self.tangent(q)
        return rank, condition


def _orbit_metrics(dynamics: CR3BP_Dynamics, q: np.ndarray, point: int) -> tuple[float, float]:
    state = _state_from_q(q)
    result = dynamics.propagate(
        state,
        (0.0, float(q[4])),
        t_eval=np.linspace(0.0, float(q[4]), 500),
        with_jacobi=True,
    )
    samples = result["states"]
    jacobi = np.asarray(result["jacobi"])
    if point not in (4, 5):
        raise ValueError(f"平动点必须是 L4 或 L5，当前为 L{point}")
    l45 = np.array(
        [0.5 - dynamics.system.mu, np.sqrt(3.0) / 2.0 if point == 4 else -np.sqrt(3.0) / 2.0]
    )
    distances = np.linalg.norm(samples[:, :2] - l45, axis=1)
    characteristic_length = dynamics.system.characteristic_length
    assert characteristic_length is not None
    distance_mean_km = float(0.5 * (distances.min() + distances.max()) * characteristic_length)
    jacobi_drift = float(np.max(np.abs(jacobi - jacobi[0])))
    return distance_mean_km, jacobi_drift


def _record(
    pal: FullPeriodPal,
    q: np.ndarray,
    *,
    step: int,
    point: int,
    pal_iterations: int,
    phase_residual: float,
    arc_residual: float,
) -> StepRecord:
    closure, closure_jacobian = pal.closure_and_jacobian(q)
    tangent, tangent_rank, tangent_condition = pal.tangent(q)
    _, phase_jacobian = pal.phase_and_jacobian(q, q)
    augmented_rank, augmented_condition = _rank_and_condition(
        np.vstack((closure_jacobian, phase_jacobian, tangent))
    )
    libration_distance_mean_km, jacobi_drift = _orbit_metrics(pal.dynamics, q, point)
    return StepRecord(
        step=step,
        initial_state=_state_from_q(q).tolist(),
        period=float(q[4]),
        closure_inf_norm=float(np.max(np.abs(closure))),
        phase_residual=phase_residual,
        arc_residual=arc_residual,
        pal_iterations=pal_iterations,
        tangent_system_rank=tangent_rank,
        tangent_system_condition=tangent_condition,
        augmented_system_rank=augmented_rank,
        augmented_system_condition=augmented_condition,
        jacobi_constant=float(pal.dynamics.compute_jacobi_constant(_state_from_q(q))),
        jacobi_drift=jacobi_drift,
        libration_distance_mean_km=libration_distance_mean_km,
    )


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #436 全周期 PAL 调研原型")
    parser.add_argument("--family", choices=("spo", "lpo"), default="lpo")
    parser.add_argument("--point", choices=(4, 5), type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--ds", type=float, default=0.01, help="归一化伪弧长步长")
    parser.add_argument(
        "--direction", choices=("increase-x0", "decrease-x0"), default="decrease-x0"
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.steps < 1 or args.ds <= 0.0:
        raise SystemExit("steps 必须大于零，ds 必须为正数")

    _, dynamics = _system_and_dynamics()
    seed = _initial_seed(dynamics, args.family, args.point)
    pal = FullPeriodPal(dynamics, tolerance=args.tolerance, max_iterations=args.max_iterations)
    q = pal.refine_seed(_q_from_orbit(seed))
    tangent, _, _ = pal.tangent(q)
    desired_x_sign = 1.0 if args.direction == "increase-x0" else -1.0
    if tangent[0] * desired_x_sign < 0.0:
        tangent = -tangent

    records = [
        _record(
            pal, q, step=0, point=args.point, pal_iterations=0, phase_residual=0.0, arc_residual=0.0
        )
    ]
    for step in range(1, args.steps + 1):
        q, iterations, _, _, _, _, phase, arc = pal.correct_step(q, tangent, args.ds)
        next_tangent, _, _ = pal.tangent(q)
        if np.dot(next_tangent, tangent) < 0.0:
            next_tangent = -next_tangent
        tangent = next_tangent
        records.append(
            _record(
                pal,
                q,
                step=step,
                point=args.point,
                pal_iterations=iterations,
                phase_residual=phase,
                arc_residual=arc,
            )
        )

    output = {
        "issue": 436,
        "commit": _git_commit(),
        "family": args.family,
        "libration_point": args.point,
        "formulation": {
            "unknowns": ["x0", "y0", "vx0", "vy0", "T"],
            "closure": "phi_T(q_state) - q_state = 0 (planar 4D)",
            "phase": "(q_state - q_state_ref) dot f(q_state_ref) / ||f(q_state_ref)|| = 0",
            "arclength": "((q - q_ref) / scales) dot tangent - ds = 0",
            "scales": SCALES.tolist(),
            "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        },
        "settings": vars(args) | {"output": str(args.output)},
        "records": [asdict(record) for record in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"写入 {args.output}：{len(records)} 条轨道记录")


if __name__ == "__main__":
    main()
