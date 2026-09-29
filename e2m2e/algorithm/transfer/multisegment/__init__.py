"""通用多段参数优化框架（multisegment，issue #726 / ADR 0053）。

只做问题组装——leg 切分、节点变量、matchpoint 连续性约束、雅可比拼装
（含逐 leg 混档传播内核与跨 leg TOF 窗口平移灵敏度）；NLP 求解复用
``e2m2e.algorithm.transfer.nlp_scipy`` / ``nlp_copt``，方系统与 minimax
档在 :mod:`.solve`。Sims-Flanagan（单 leg / 多 leg）与 MGA 精化是本
框架的两个实例（SF 转录语义——成本函数、段可行域、flyby 决策变量——
留在各自模块）。
"""

from .chain import ChainLegEval, ChainLegRequest, evaluate_chain
from .kernels import LegKernel, propagate_half, tier_rhs
from .layout import VariableLayout
from .nodes import FlybyConstraintEval, flyby_node_constraints
from .solve import MinimaxSolveResult, SquareSolveResult, solve_minimax, solve_square

__all__ = [
    "ChainLegEval",
    "ChainLegRequest",
    "FlybyConstraintEval",
    "LegKernel",
    "MinimaxSolveResult",
    "SquareSolveResult",
    "VariableLayout",
    "evaluate_chain",
    "flyby_node_constraints",
    "propagate_half",
    "solve_minimax",
    "solve_square",
    "tier_rhs",
]
