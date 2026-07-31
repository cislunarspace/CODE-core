"""NLP 优化公共抽象。

提供 :class:`NLPOptimizationVariables` 数据结构与后端无关的辅助类型，
作为 SciPy / COPT 后端之间共享的"问题描述"层。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NLPOptimizationVariables:
    """NLP 优化变量

    优化变量: ``y = (α, T, t_ins)``，分别表示切向速度比、转移时间与
    目标轨道上的插入时间。

    Attributes:
        alpha: 切向速度比
        transfer_time: 转移时间 T
        t_ins: 从轨道远地点到插入点的时间
    """

    alpha: float = 0.0
    transfer_time: float = 0.0
    t_ins: float = 0.0

    def to_array(self) -> np.ndarray:
        """转换为 numpy 数组。

        Returns:
            ``[alpha, transfer_time, t_ins]`` 一维数组。
        """
        return np.array([self.alpha, self.transfer_time, self.t_ins])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> NLPOptimizationVariables:
        """从 numpy 数组创建实例。

        Args:
            arr: ``[alpha, transfer_time, t_ins]`` 一维数组。

        Returns:
            对应的 :class:`NLPOptimizationVariables` 实例。
        """
        return cls(alpha=arr[0], transfer_time=arr[1], t_ins=arr[2])

