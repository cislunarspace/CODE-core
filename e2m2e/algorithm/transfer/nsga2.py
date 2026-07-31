"""NSGA-II 多目标优化器（主题 8）。

经典 NSGA-II（Deb et al. 2002）：非支配排序 + 拥挤度选择 + 精英保留。
纯 Python 编排，适应度评估可选 ProcessPoolExecutor 并行（Windows spawn
安全，对齐 ``search_parallel.py`` 模式）。

约束处理用 Deb 可行支配规则：可行解支配不可行解；都不可行时按约束
违反量排序。无需罚因子。

用法::

    def my_objectives(x: np.ndarray) -> tuple[np.ndarray, float]:
        # x: 决策向量
        # 返回 (目标向量, 约束违反量)
        return np.array([x[0]**2, (x[0]-2)**2]), 0.0

    result = nsga2(
        objectives=my_objectives,
        bounds=[(-5.0, 5.0)],
        pop_size=100,
        n_gen=200,
        seed=42,
    )
    # result.x: (k, n_dim) 前沿决策向量
    # result.f: (k, n_obj) 前沿目标向量
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .porkchop import _non_dominated_sort

# 类型别名：目标函数签名
#   fn(x: np.ndarray) -> (objectives: np.ndarray, violation: float)
# objectives 长度 = n_obj，violation >= 0（0 = 可行）
ObjectiveFn = Callable[[npt.NDArray[np.floating]], tuple[npt.NDArray[np.floating], float]]


@dataclass
class NSGA2Result:
    """NSGA-II 优化结果。

    Attributes:
        x: Pareto 前沿决策向量，形状 ``(k, n_dim)``
        f: Pareto 前沿目标向量，形状 ``(k, n_obj)``
        rank: 前沿点非支配层级（恒为 0），形状 ``(k,)``
        crowding: 前沿点拥挤度，形状 ``(k,)``
        n_eval: 总目标函数评估次数
        history: 每代种群规模与前沿规模记录
    """

    x: np.ndarray
    f: np.ndarray
    rank: np.ndarray
    crowding: np.ndarray
    n_eval: int
    history: list[dict[str, Any]]


def nsga2(
    objectives: ObjectiveFn,
    bounds: list[tuple[float, float]],
    *,
    pop_size: int = 100,
    n_gen: int = 200,
    crossover_prob: float = 0.9,
    mutation_prob: float | None = None,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
    seed: int | None = None,
    n_workers: int | None = None,
    verbose: bool = False,
) -> NSGA2Result:
    """NSGA-II 多目标优化。

    Args:
        objectives: 目标函数，签名 ``fn(x) -> (objectives, violation)``。
            ``objectives`` 形状 ``(n_obj,)``，全部最小化；``violation >= 0``，
            0 表示可行。必须是**模块级可 pickle 函数**（并行评估要求）。
        bounds: 决策变量边界 ``[(lo, hi), ...]``，长度 = n_dim。
        pop_size: 种群规模。
        n_gen: 进化代数。
        crossover_prob: 交叉概率（SBX）。
        mutation_prob: 变异概率，None 时取 ``1 / n_dim``。
        eta_c: SBX 交叉分布指数（越大子代越贴近父代）。
        eta_m: 多项式变异分布指数。
        seed: 随机种子。
        n_workers: 并行进程数，None 时取 ``min(cpu_count(), 4)``；
            1 时退化为串行。
        verbose: 每代打印进度。

    Returns:
        :class:`NSGA2Result`（Pareto 前沿 + 诊断信息）。

    Raises:
        ValueError: bounds 为空、pop_size < 4、目标函数返回长度不一致。
    """
    if not bounds:
        raise ValueError("bounds 不能为空")
    n_dim = len(bounds)
    if pop_size < 4:
        raise ValueError(f"pop_size ({pop_size}) 须 >= 4")
    if mutation_prob is None:
        mutation_prob = 1.0 / n_dim

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    if np.any(lo >= hi):
        raise ValueError("bounds 下界须严格小于上界")

    rng = np.random.default_rng(seed)
    n_eval = 0

    # 初始化种群（拉丁超立方简化版：均匀随机）
    pop = rng.uniform(lo, hi, size=(pop_size, n_dim))
    fit, viol = _evaluate_population(pop, objectives, n_workers)
    n_eval += pop_size

    history: list[dict[str, Any]] = []

    for gen in range(n_gen):
        # 非支配排序 + 拥挤度
        rank, crowd = _constrained_non_dominated_sort(fit, viol)

        # 选择（二元锦标赛）
        mating_idx = _tournament_selection(rank, crowd, pop_size, rng)

        # SBX 交叉 + 多项式变异
        offspring = _sbx_crossover(pop[mating_idx], lo, hi, crossover_prob, eta_c, rng)
        offspring = _polynomial_mutation(offspring, lo, hi, mutation_prob, eta_m, rng)

        # 评估子代
        off_fit, off_viol = _evaluate_population(offspring, objectives, n_workers)
        n_eval += offspring.shape[0]

        # 合并父代 + 子代，环境选择
        combined_pop = np.vstack([pop, offspring])
        combined_fit = np.vstack([fit, off_fit])
        combined_viol = np.concatenate([viol, off_viol])

        rank_c, crowd_c = _constrained_non_dominated_sort(combined_fit, combined_viol)

        # 精英保留：按 (rank, -crowd) 排序取前 pop_size
        elite_idx = _environmental_selection(rank_c, crowd_c, pop_size)
        pop = combined_pop[elite_idx]
        fit = combined_fit[elite_idx]
        viol = combined_viol[elite_idx]

        # 记录
        front_size = int(np.sum(rank_c == 0))
        history.append({"gen": gen, "pop_size": pop_size, "front_size": front_size})
        if verbose and (gen % 10 == 0 or gen == n_gen - 1):
            print(f"gen {gen:4d}: front_size={front_size}, n_eval={n_eval}")

    # 最终前沿
    rank, crowd = _constrained_non_dominated_sort(fit, viol)
    front_mask = rank == 0
    return NSGA2Result(
        x=pop[front_mask],
        f=fit[front_mask],
        rank=rank[front_mask],
        crowding=crowd[front_mask],
        n_eval=n_eval,
        history=history,
    )


# ----------------------------------------------------------------------
# 内部：评估、排序、选择、交叉、变异
# ----------------------------------------------------------------------


def _evaluate_population(
    pop: np.ndarray,
    fn: ObjectiveFn,
    n_workers: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """评估种群，返回 (fitness, violation)。

    并行用 ProcessPoolExecutor，串行用 list comprehension。
    """
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 4)
    if n_workers <= 1 or pop.shape[0] <= 1:
        results = [fn(x) for x in pop]
    else:
        # ProcessPoolExecutor 要求 fn 可 pickle（Windows spawn 模式）。
        # 闭包/lambda/__main__ 内定义的函数不可 pickle，会抛 AttributeError。
        # 提前检测并给出清晰报错。
        import pickle

        try:
            pickle.dumps(fn)
        except (pickle.PicklingError, AttributeError) as e:
            raise ValueError(
                f"目标函数 {fn!r} 不可 pickle，无法用于并行评估。"
                "请把目标函数定义为模块级函数（非 lambda/闭包/__main__ 内定义），"
                "或设 n_workers=1 退化为串行。"
            ) from e
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_evaluate_single, [(fn, x) for x in pop]))

    fit = np.array([r[0] for r in results])
    viol = np.array([r[1] for r in results])
    return fit, viol


def _evaluate_single(packed: tuple[ObjectiveFn, np.ndarray]) -> tuple[np.ndarray, float]:
    """模块级 worker，供 ProcessPoolExecutor 调用（Windows spawn 安全）。"""
    fn, x = packed
    objs, viol = fn(x)
    return np.asarray(objs, dtype=float), float(viol)


def _constrained_non_dominated_sort(
    fit: np.ndarray, viol: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Deb 可行支配规则的非支配排序 + 拥挤度。

    可行解（viol == 0）支配不可行解；都不可行时按 viol 升序排（viol 小者
    支配 viol 大者）；都可行时按目标向量标准非支配排序。

    Returns:
        (rank, crowding)：rank 0 为最优前沿（可行且非支配）。
    """
    n = fit.shape[0]
    feasible = viol <= 0.0

    # 分层：可行层 + 不可行层
    # 可行层内部按目标非支配排序
    # 不可行层按 viol 升序排（viol 越小 rank 越高）
    rank = np.full(n, -1, dtype=int)

    if np.any(feasible):
        feas_idx = np.flatnonzero(feasible)
        feas_fit = fit[feasible]
        feas_rank = _non_dominated_sort(feas_fit)
        rank[feas_idx] = feas_rank

        # 不可行解排在可行解之后
        infeas_idx = np.flatnonzero(~feasible)
        if infeas_idx.size > 0:
            max_feas_rank = int(np.max(feas_rank))
            infeas_viol = viol[infeas_idx]
            # viol 小的排前面（rank 低）
            viol_order = np.argsort(infeas_viol)
            infeas_rank = np.empty(infeas_idx.size, dtype=int)
            # 按 viol 分组赋 rank：viol 相同的同 rank
            current_rank = max_feas_rank + 1
            prev_viol = -1.0
            for pos, idx in enumerate(viol_order):
                v = infeas_viol[idx]
                if v != prev_viol:
                    if pos > 0:
                        current_rank += 1
                    prev_viol = v
                infeas_rank[idx] = current_rank
            rank[infeas_idx] = infeas_rank
    else:
        # 全不可行：按 viol 排序
        order = np.argsort(viol)
        current_rank = 0
        prev_viol = -1.0
        for pos, idx in enumerate(order):
            v = viol[idx]
            if v != prev_viol:
                if pos > 0:
                    current_rank += 1
                prev_viol = v
            rank[idx] = current_rank

    crowding = _crowding_distance(fit, rank)
    return rank, crowding


def _crowding_distance(fit: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """计算每个点的拥挤度（仅用于同 rank 内比较）。"""
    n = fit.shape[0]
    crowd = np.zeros(n)
    for r in np.unique(rank):
        mask = rank == r
        if np.sum(mask) <= 2:
            crowd[mask] = np.inf
            continue
        idx = np.flatnonzero(mask)
        sub_fit = fit[mask]
        n_obj = sub_fit.shape[1]
        sub_crowd = np.zeros(idx.size)
        for m in range(n_obj):
            order = np.argsort(sub_fit[:, m])
            sub_crowd[order[0]] = np.inf
            sub_crowd[order[-1]] = np.inf
            f_min = sub_fit[order[0], m]
            f_max = sub_fit[order[-1], m]
            span = f_max - f_min
            if span < 1e-30:
                continue
            for i in range(1, idx.size - 1):
                sub_crowd[order[i]] += (sub_fit[order[i + 1], m] - sub_fit[order[i - 1], m]) / span
        crowd[idx] = sub_crowd
    return crowd


def _tournament_selection(
    rank: np.ndarray, crowd: np.ndarray, n_select: int, rng: np.random.Generator
) -> np.ndarray:
    """二元锦标赛选择：rank 低者胜；同 rank 拥挤度大者胜。"""
    n = rank.shape[0]
    selected = np.empty(n_select, dtype=int)
    for i in range(n_select):
        a, b = rng.integers(0, n, size=2)
        if rank[a] < rank[b]:
            selected[i] = a
        elif rank[b] < rank[a]:
            selected[i] = b
        elif crowd[a] >= crowd[b]:
            selected[i] = a
        else:
            selected[i] = b
    return selected


def _environmental_selection(rank: np.ndarray, crowd: np.ndarray, n_keep: int) -> np.ndarray:
    """精英保留：按 (rank, -crowd) 排序取前 n_keep。"""
    # 主键 rank 升序，副键 crowd 降序
    order = np.lexsort((-crowd, rank))
    return order[:n_keep]


def _sbx_crossover(
    parents: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    prob: float,
    eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """模拟二进制交叉（SBX）。"""
    n, n_dim = parents.shape
    offspring = parents.copy()
    for i in range(0, n - 1, 2):
        if rng.random() > prob:
            continue
        p1, p2 = parents[i], parents[i + 1]
        for j in range(n_dim):
            if rng.random() <= 0.5 and abs(p1[j] - p2[j]) > 1e-14:
                y1, y2 = min(p1[j], p2[j]), max(p1[j], p2[j])
                rand = rng.random()
                beta = 1.0 + (2.0 * (y1 - lo[j]) / (y2 - y1))
                alpha = 2.0 - beta ** -(eta + 1.0)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                beta = 1.0 + (2.0 * (hi[j] - y2) / (y2 - y1))
                alpha = 2.0 - beta ** -(eta + 1.0)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                # 边界钳制
                c1 = np.clip(c1, lo[j], hi[j])
                c2 = np.clip(c2, lo[j], hi[j])

                if rng.random() <= 0.5:
                    offspring[i, j] = c2
                    offspring[i + 1, j] = c1
                else:
                    offspring[i, j] = c1
                    offspring[i + 1, j] = c2
    return offspring


def _polynomial_mutation(
    pop: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    prob: float,
    eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """多项式变异。"""
    n, n_dim = pop.shape
    for i in range(n):
        for j in range(n_dim):
            if rng.random() <= prob:
                y = pop[i, j]
                delta1 = (y - lo[j]) / (hi[j] - lo[j])
                delta2 = (hi[j] - y) / (hi[j] - lo[j])
                rand = rng.random()
                mut_pow = 1.0 / (eta + 1.0)
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
                    deltaq = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
                    deltaq = 1.0 - val**mut_pow
                y = y + deltaq * (hi[j] - lo[j])
                pop[i, j] = np.clip(y, lo[j], hi[j])
    return pop

