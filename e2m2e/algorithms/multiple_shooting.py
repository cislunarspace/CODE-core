from __future__ import annotations

import numpy as np
from typing import Optional


class MultipleShootingResult:
    def __init__(self, t_patch, state_patch, converged, iterations, max_residual, residual_history):
        self.t_patch = t_patch
        self.state_patch = state_patch
        self.converged = converged
        self.iterations = iterations
        self.max_residual = max_residual
        self.residual_history = residual_history


class MultipleShooting:
    def __init__(self, dynamics) -> None:
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
    ) -> MultipleShootingResult:
        t_patch = np.asarray(t_patch, dtype=float)
        state_patch = np.asarray(state_patch, dtype=float)

        if len(t_patch) != len(state_patch):
            raise ValueError("t_patch and state_patch must have the same length")
        if len(t_patch) == 0:
            raise ValueError("t_patch and state_patch must not be empty")

        _max_iter = max_iter if max_iter is not None else self.max_iter
        _tolerance = tolerance if tolerance is not None else self.tolerance

        t_work = t_patch.copy()
        state_work = state_patch.copy()
        N = len(t_work)
        n_seg = N - 1
        I6 = np.eye(6)

        residual_history = []
        converged = False

        for iteration in range(_max_iter):
            stms = []
            final_states = []
            f_starts = []
            f_ends = []

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

            F = np.zeros(n_seg * 6)
            for i in range(n_seg):
                F[i * 6 : (i + 1) * 6] = final_states[i] - state_work[i + 1]

            max_res = np.max(np.abs(F))
            residual_history.append(float(max_res))

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

            n_constraints = n_seg * 6

            if var_time:
                n_vars = N * 6 + N
                DF = np.zeros((n_constraints, n_vars))

                for i in range(n_seg):
                    r_start = i * 6
                    r_end = (i + 1) * 6
                    DF[r_start:r_end, i * 6 : (i + 1) * 6] = stms[i]
                    DF[r_start:r_end, (i + 1) * 6 : (i + 2) * 6] = -I6
                    DF[r_start:r_end, N * 6 + i] = -f_starts[i]
                    DF[r_start:r_end, N * 6 + i + 1] = f_ends[i]
            else:
                n_vars = N * 6
                DF = np.zeros((n_constraints, n_vars))

                for i in range(n_seg):
                    r_start = i * 6
                    r_end = (i + 1) * 6
                    DF[r_start:r_end, i * 6 : (i + 1) * 6] = stms[i]
                    DF[r_start:r_end, (i + 1) * 6 : (i + 2) * 6] = -I6

            dX, _, _, _ = np.linalg.lstsq(DF, -F, rcond=None)

            state_work = state_work.copy()
            t_work = t_work.copy()

            X_flat = state_work.flatten()
            X_flat += dX[: N * 6]
            state_work = X_flat.reshape(N, 6)

            if var_time:
                t_work += dX[N * 6 : N * 6 + N]

        return MultipleShootingResult(
            t_patch=t_work,
            state_patch=state_work,
            converged=False,
            iterations=_max_iter,
            max_residual=residual_history[-1] if residual_history else float("inf"),
            residual_history=residual_history,
        )
