"""任务综合评估：多指标加权（主题 8）。

在 Pareto 前沿上按用户权重把多目标标量化，辅助决策。对应规划文档
「丁百慧式多指标加权」——但文档未展开「动态权值」语义，本实现为
**静态加权** （与文档接口 stub 一致）。

用法::

    from e2m2e.transfer.mission_assessment import MissionAssessment

    ma = MissionAssessment()
    # 在 NSGA-II 前沿上评估
    scores = ma.evaluate(result.f, weights={"dv": 0.7, "tof": 0.3})
    best_idx = int(np.argmin(scores))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class MissionAssessment:
    """多指标加权综合评估器。

    把多目标解集（Pareto 前沿）按用户给定的权重标量化，
    返回每个解的综合得分（越小越优）。

    指标名与目标向量的列对应：默认按列序命名
    ``["obj_0", "obj_1", ...]``，也可显式传入 ``metric_names``。
    """

    metric_names: list[str] | None = None

    def evaluate(
        self,
        solutions: npt.NDArray[np.floating],
        weights: dict[str, float],
    ) -> np.ndarray:
        """对解集加权评分。

        Args:
            solutions: 目标向量矩阵，形状 ``(k, n_obj)``，每行一个解。
            weights: 指标权重，如 ``{"dv": 0.7, "tof": 0.3}``。
                权重和归一化到 1。未指定的指标权重为 0。

        Returns:
            综合得分数组，形状 ``(k,)``，越小越优。

        Raises:
            ValueError: solutions 为空、weights 含未知指标名、
                或 metric_names 长度与 solutions 列数不匹配。
        """
        solutions = np.asarray(solutions, dtype=float)
        if solutions.ndim == 1:
            solutions = solutions.reshape(1, -1)
        k, n_obj = solutions.shape
        if k == 0:
            raise ValueError("solutions 不能为空")

        names = self._resolve_names(n_obj)
        if len(names) != n_obj:
            raise ValueError(f"metric_names 长度 ({len(names)}) 与 solutions 列数 ({n_obj}) 不匹配")

        # 构建权重向量
        w = np.zeros(n_obj)
        for name, weight in weights.items():
            if name not in names:
                raise ValueError(f"未知指标名 {name!r}，可用: {names}")
            w[names.index(name)] = float(weight)
        w_sum = np.sum(np.abs(w))
        if w_sum < 1e-30:
            raise ValueError("weights 全为 0")
        w = w / w_sum

        # 加权求和（假设各指标已同向化：全部越小越优）
        return solutions @ w

    def rank(
        self,
        solutions: npt.NDArray[np.floating],
        weights: dict[str, float],
    ) -> np.ndarray:
        """返回按综合得分升序排序的索引（最优在前）。"""
        scores = self.evaluate(solutions, weights)
        return np.argsort(scores)

    def best(
        self,
        solutions: npt.NDArray[np.floating],
        weights: dict[str, float],
    ) -> tuple[int, float]:
        """返回最优解的索引与得分。"""
        scores = self.evaluate(solutions, weights)
        idx = int(np.argmin(scores))
        return idx, float(scores[idx])

    def _resolve_names(self, n_obj: int) -> list[str]:
        """确定指标名列表。"""
        if self.metric_names is not None:
            return self.metric_names
        return [f"obj_{i}" for i in range(n_obj)]

    @classmethod
    def from_pareto_front(
        cls, front: Any, metric_names: list[str] | None = None
    ) -> MissionAssessment:
        """从 ParetoFront 结果构造评估器（推断指标名）。

        支持 ``NSGA2Result`` （用 ``obj_0..obj_{n-1}``）和
        ``ParetoFront`` （用 ``["dv", "tof"]`` 等 porkchop 字段名）。
        """
        if metric_names is not None:
            return cls(metric_names=metric_names)
        # 从 front 对象推断
        if hasattr(front, "f") and isinstance(front.f, np.ndarray):
            n_obj = front.f.shape[1] if front.f.ndim > 1 else 1
            return cls(metric_names=[f"obj_{i}" for i in range(n_obj)])
        if hasattr(front, "total") and hasattr(front, "tof"):
            return cls(metric_names=["dv", "tof"])
        return cls()
