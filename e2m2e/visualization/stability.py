from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..core import CR3BP_Dynamics


def _compute_single_stability(args):
    i, initial_state, period, system = args
    if period is None:
        return (i, 1.0)
    try:
        dynamics = CR3BP_Dynamics(system)
        monodromy = dynamics.compute_state_transition_matrix(initial_state, period)
        eigenvalues = np.linalg.eigvals(monodromy)
        magnitudes = np.abs(eigenvalues)
        stability_idx = float(np.max(magnitudes))
        return (i, stability_idx)
    except Exception:
        return (i, 1.0)


def compute_stability_for_family(family_result, system, max_workers=None):
    if family_result is None or len(family_result) == 0:
        return []

    for orbit in family_result:
        if orbit.system is None:
            orbit.system = system

    tasks = []
    for i, orbit in enumerate(family_result):
        tasks.append((i, orbit.states[0].copy(), orbit.period, system))

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(tasks))

    results: list[float] = [1.0] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_compute_single_stability, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_to_idx):
            idx, value = future.result()
            results[idx] = value

    return results
