"""NLP 优化公共抽象。

提供 :class:`NLPOptimizationVariables` 与 :class:`NLPSpec` 两个数据结构，
作为 SciPy / COPT 后端与通用问题描述层之间的共享"问题"层：前者是 DRO→RO
专用变量元组，后者是求解器无关的 NLP 输入契约（#726）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.optimize import Bounds


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


@dataclass(frozen=True)
class NLPSpec:
    """通用 NLP 问题描述（:func:`.nlp_scipy.solve_slsqp` 的输入契约）。

    只描述问题（目标、约束及其解析雅可比、变量盒、回调），不绑定求解器；
    雅可比缺省 ``None`` 时由求解器数值差分。约束语义遵循 scipy：``eq`` 为
    ``fun(x) = 0``、``ineq`` 为 ``fun(x) >= 0``。

    Attributes:
        objective: 目标函数 ``x -> float``。
        objective_grad: 目标解析梯度；``None`` 退数值差分。
        eq: 等式约束残差向量；``None`` 表示无等式约束。
        eq_jac: 等式约束解析雅可比；``None`` 退数值差分。
        ineq: 不等式约束（``>= 0`` 可行）；``None`` 表示无不等式约束。
        ineq_jac: 不等式约束解析雅可比；``None`` 退数值差分。
        bounds: 决策变量盒（:class:`scipy.optimize.Bounds`）；``None`` 无界。
        callback: 逐迭代回调 ``xk -> None``；``None`` 不回调。
    """

    objective: Callable[[np.ndarray], float]
    objective_grad: Callable[[np.ndarray], np.ndarray] | None = None
    eq: Callable[[np.ndarray], np.ndarray] | None = None
    eq_jac: Callable[[np.ndarray], np.ndarray] | None = None
    ineq: Callable[[np.ndarray], np.ndarray] | None = None
    ineq_jac: Callable[[np.ndarray], np.ndarray] | None = None
    bounds: Bounds | None = None
    callback: Callable[[np.ndarray], None] | None = None
