"""轨道搜索的几何核。

纯函数：仅 np.ndarray / Orbit 输入，不引入进度条与并行。
"""

from __future__ import annotations

import numpy as np

from ...data.types.orbit import Orbit

MAX_DISTANCE_PAIRS = 10_000_000


def compute_distance_series(
    trajectory_states: np.ndarray, arrival_orbit: Orbit
) -> tuple[np.ndarray, np.ndarray]:
    traj_positions = trajectory_states[:, :3]
    orbit_positions = arrival_orbit.states[:, :3]
    n_traj = len(traj_positions)
    n_orbit = len(orbit_positions)
    if n_traj * n_orbit > MAX_DISTANCE_PAIRS:
        return compute_distance_series_chunked(traj_positions, orbit_positions)
    diff = traj_positions[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    orbit_idx_per_step = np.argmin(distances, axis=1)
    d_per_step = distances[np.arange(n_traj), orbit_idx_per_step]
    return d_per_step, orbit_idx_per_step.astype(np.int64)


def compute_distance_series_chunked(
    traj_positions: np.ndarray, orbit_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_traj = len(traj_positions)
    n_orbit = len(orbit_positions)
    chunk_size = max(1, MAX_DISTANCE_PAIRS // n_orbit)
    d_per_step = np.empty(n_traj, dtype=np.float64)
    orbit_idx_per_step = np.empty(n_traj, dtype=np.int64)
    for start in range(0, n_traj, chunk_size):
        end = min(start + chunk_size, n_traj)
        chunk = traj_positions[start:end]
        diff = chunk[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        chunk_orbit_idx = np.argmin(distances, axis=1)
        chunk_d = distances[np.arange(end - start), chunk_orbit_idx]
        d_per_step[start:end] = chunk_d
        orbit_idx_per_step[start:end] = chunk_orbit_idx
    return d_per_step, orbit_idx_per_step


def compute_min_distance(
    trajectory_states: np.ndarray, arrival_orbit: Orbit
) -> tuple[float, int, int]:
    d_per_step, orbit_idx_per_step = compute_distance_series(trajectory_states, arrival_orbit)
    step_idx = int(np.argmin(d_per_step))
    return float(d_per_step[step_idx]), step_idx, int(orbit_idx_per_step[step_idx])


def detect_intersection(
    trajectory_states: np.ndarray, arrival_orbit: Orbit, threshold: float
) -> tuple[bool, np.ndarray | None, int]:
    min_dist, step_idx, _ = compute_min_distance(trajectory_states, arrival_orbit)
    if min_dist < threshold:
        return True, trajectory_states[step_idx], step_idx
    return False, None, -1


def detect_local_minimum(
    trajectory_states: np.ndarray, arrival_orbit: Orbit
) -> tuple[bool, float, int]:
    traj_positions = trajectory_states[:, :3]
    orbit_positions = arrival_orbit.states[:, :3]
    diff = traj_positions[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    min_distances = np.min(distances, axis=1)
    local_mins = []
    for i in range(1, len(min_distances) - 1):
        if min_distances[i + 1] > min_distances[i] and min_distances[i - 1] > min_distances[i]:
            local_mins.append((i, min_distances[i]))
    if local_mins:
        best = min(local_mins, key=lambda x: x[1])
        return True, best[1], best[0]
    return False, np.inf, -1


def check_collision(
    trajectory_states: np.ndarray,
    mu: float,
    collision_earth_radius: float,
    collision_moon_radius: float,
) -> tuple[bool, str | None, int]:
    positions = trajectory_states[:, :3]
    earth_center = np.array([-mu, 0.0, 0.0])
    moon_center = np.array([1.0 - mu, 0.0, 0.0])
    dist_earth = np.linalg.norm(positions - earth_center, axis=1)
    dist_moon = np.linalg.norm(positions - moon_center, axis=1)
    earth_collision_idx = np.where(dist_earth < collision_earth_radius)[0]
    moon_collision_idx = np.where(dist_moon < collision_moon_radius)[0]
    if len(earth_collision_idx) > 0:
        return True, "earth", int(earth_collision_idx[0])
    if len(moon_collision_idx) > 0:
        return True, "moon", int(moon_collision_idx[0])
    return False, None, -1


def is_feasible_result(
    result: dict,
    min_distance_threshold: float | None,
    default_min_distance_threshold: float,
) -> bool:
    mdt = (
        min_distance_threshold
        if min_distance_threshold is not None
        else default_min_distance_threshold
    )
    if result.get("collision_found", False):
        return False
    md = float(result.get("min_distance", float("inf")))
    lmd = float(result.get("local_minimum_distance", float("inf")))
    if result.get("intersection_found", False):
        return True
    if md < mdt:
        return True
    return bool(result.get("local_minimum_found", False) and lmd < mdt)

