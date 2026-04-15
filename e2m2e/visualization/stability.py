"""稳定性可视化模块

提供轨道族稳定性指标的并行计算工具。
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..core import CR3BP_Dynamics


def _compute_single_stability(args):
    """计算单条轨道的稳定性指标（Floquet 乘子最大模）。

    Args:
        args: 元组 (轨道索引, 初始状态, 周期, 系统对象)。

    Returns:
        (轨道索引, 稳定性指标) 元组。周期为 None 时返回 1.0。
    """
    i, initial_state, period, system = args
    if period is None:
        return (i, 1.0)
    try:
        # 为每条轨道创建独立的 dynamics 实例，支持多进程并行
        dynamics = CR3BP_Dynamics(system)
        monodromy = dynamics.compute_state_transition_matrix(initial_state, period)
        eigenvalues = np.linalg.eigvals(monodromy)  # Floquet 乘子
        magnitudes = np.abs(eigenvalues)  # |λ|
        stability_idx = float(np.max(magnitudes))
        return (i, stability_idx)
    except Exception:
        return (i, 1.0)


def compute_stability_for_family(family_result, system, max_workers=None):
    """计算轨道族的稳定性指标（Floquet 乘子最大模）。

    对轨道族中每条轨道，通过 monodromy matrix（单周期状态转移矩阵）
    计算 eigenvalue（特征值），取最大模作为稳定性指标。

    Args:
        family_result: OrbitFamily 或轨道列表
        system: CR3BP_System 实例
        max_workers: 并行进程数，默认为 min(cpu_count, 轨道数)

    Returns:
        list[float]: 每条轨道的稳定性指标（Floquet 乘子最大模）
    """
    if family_result is None or len(family_result) == 0:
        return []

    # 确保每条轨道关联了系统对象
    for orbit in family_result:
        if orbit.system is None:
            orbit.system = system

    # 构造并行任务列表
    tasks = []
    for i, orbit in enumerate(family_result):
        tasks.append((i, orbit.states[0].copy(), orbit.period, system))

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(tasks))

    # 使用多进程并行计算，每条轨道独立求解特征值
    results: list[float] = [1.0] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_compute_single_stability, task): task[0] for task in tasks
        }
        for future in as_completed(future_to_idx):
            idx, value = future.result()
            results[idx] = value

    return results
