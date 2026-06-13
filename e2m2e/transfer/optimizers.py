"""NLP 优化器 adapter：统一 SciPy / COPT 对外接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import TransferOptimizationResult
from .transfer_optimization import (
    DROTRONLPOptimizer,
    NLPOptimizationVariables,
    optimize_with_copt,
)


class TransferOptimizer(ABC):
    """DRO-RO 转移 NLP 优化器统一接口。"""

    @abstractmethod
    def optimize(
        self,
        initial_guess: NLPOptimizationVariables,
        **kwargs: Any,
    ) -> TransferOptimizationResult:
        """执行优化并返回统一结果类型。"""


class SciPyTransferOptimizer(TransferOptimizer):
    """基于 SciPy SLSQP 的优化器 adapter。"""

    def __init__(self, optimizer: DROTRONLPOptimizer):
        self.optimizer = optimizer

    def optimize(
        self,
        initial_guess: NLPOptimizationVariables,
        **kwargs: Any,
    ) -> TransferOptimizationResult:
        """调用底层 SciPy 路径，直接返回 TransferOptimizationResult。"""
        return self.optimizer.optimize(initial_guess=initial_guess, **kwargs)


class COPTTransferOptimizer(TransferOptimizer):
    """基于 COPT 的优化器 adapter，支持 SciPy 回退。"""

    def __init__(
        self,
        optimizer: DROTRONLPOptimizer,
        *,
        fallback_to_scipy: bool = True,
        **copt_options: Any,
    ):
        self.optimizer = optimizer
        self.fallback_to_scipy = fallback_to_scipy
        self.copt_options = copt_options

    def optimize(
        self,
        initial_guess: NLPOptimizationVariables,
        **kwargs: Any,
    ) -> TransferOptimizationResult:
        """调用底层 COPT 路径，失败时按配置回退 SciPy。"""
        return optimize_with_copt(
            self.optimizer,
            initial_guess=initial_guess,
            fallback_to_scipy=self.fallback_to_scipy,
            **self.copt_options,
            scipy_fallback_kwargs=kwargs,
        )
