import numpy as np
from typing import NamedTuple


class TransferCost(NamedTuple):
    dv1: float
    dv2: float
    total: float


def compute_transfer_cost(
    departure_state: np.ndarray,
    initial_velocity: np.ndarray,
    final_velocity: np.ndarray,
    insertion_velocity: np.ndarray,
) -> TransferCost:
    dv1 = float(np.linalg.norm(initial_velocity - departure_state[3:]))
    dv2 = float(np.linalg.norm(final_velocity - insertion_velocity))
    return TransferCost(dv1=dv1, dv2=dv2, total=dv1 + dv2)
