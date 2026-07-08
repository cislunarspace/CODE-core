"""COPT NLP 后端。

封装杉数科技商业优化求解器 COPT（Cardinal Optimizer）的非线性规划调用，
供 :class:`~e2m2e.transfer.transfer_optimization.DROTRONLPOptimizer` 选用。

未安装 ``coptpy`` 时：本模块仍可被导入（``coptpy`` 退化为 ``None``），
但 :class:`COPTNLPSolver` 退化为占位实现，:func:`optimize_with_copt`
应通过 ``fallback_to_scipy`` 回退 SciPy 求解。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .config import TransferOptimizationResult
from .nlp_core import NLPOptimizationVariables

if TYPE_CHECKING:
    from .transfer_optimization import DROTRONLPOptimizer

# ---------------------------------------------------------------------------
# 条件导入 coptpy：与 ``_HAVE_COPT`` 等既有检查保持兼容。
# ---------------------------------------------------------------------------
try:
    import coptpy as cp
    from coptpy import COPT

    NlpCallbackBase = cp.NlpCallbackBase
    coptpy = cp  # 与既有 ``_HAVE_COPT`` 等检查兼容
except ImportError:
    cp = None
    COPT = None
    coptpy = None
    NlpCallbackBase = None


# ---------------------------------------------------------------------------
# COPT 可用时的真实实现
# ---------------------------------------------------------------------------
if NlpCallbackBase is not None:

    class COPTNLPCallback(NlpCallbackBase):  # type: ignore[misc, valid-type]
        """COPT NLP 回调类

        用于 COPT 非线性优化问题的目标函数和约束计算。
        继承自 ``cp.NlpCallbackBase`` 以正确处理 SWIG 绑定。

        Attributes:
            optimizer: DROTRONLPOptimizer 实例
            x: 当前变量值 ``[alpha, transfer_time, t_ins]``
        """

        def __init__(self, optimizer: DROTRONLPOptimizer):
            """初始化回调。

            Args:
                optimizer: 关联的 NLP 优化器实例。
            """
            super().__init__()
            self.optimizer = optimizer
            self.x = None

        def EvalObj(self, xdata, outdata):
            """计算目标函数值 J(y) = Δv1 + Δv2"""
            x = np.array(xdata)
            self.x = x
            obj = self.optimizer.objective_function(x)
            outdata[0] = obj
            return 0

        def EvalGrad(self, xdata, outdata):
            """计算目标函数梯度 (数值差分)"""
            x = np.array(xdata)
            self.x = x
            h = 1e-8
            grad = np.zeros(3)
            f0 = self.optimizer.objective_function(x)
            for i in range(3):
                x_pert = x.copy()
                x_pert[i] += h
                grad[i] = (self.optimizer.objective_function(x_pert) - f0) / h
            for i in range(3):
                outdata[i] = grad[i]
            return 0

        def EvalCon(self, xdata, outdata):
            """计算约束函数值"""
            x = np.array(xdata)
            self.x = x

            pos_con = self.optimizer.constraint_position(x)
            vel_con = self.optimizer.constraint_velocity_parallel(x)

            outdata[0] = pos_con
            outdata[1] = vel_con
            return 0

        def EvalJac(self, xdata, outdata):
            """计算约束函数 Jacobian 矩阵 (数值差分)"""
            x = np.array(xdata)
            self.x = x
            h = 1e-8

            pos_con = self.optimizer.constraint_position(x)
            grad_pos = np.zeros(3)
            for i in range(3):
                x_pert = x.copy()
                x_pert[i] += h
                grad_pos[i] = (self.optimizer.constraint_position(x_pert) - pos_con) / h

            vel_con = self.optimizer.constraint_velocity_parallel(x)
            grad_vel = np.zeros(3)
            for i in range(3):
                x_pert = x.copy()
                x_pert[i] += h
                grad_vel[i] = (self.optimizer.constraint_velocity_parallel(x_pert) - vel_con) / h

            outdata[0] = grad_pos[0]
            outdata[1] = grad_pos[1]
            outdata[2] = grad_pos[2]
            outdata[3] = grad_vel[0]
            outdata[4] = grad_vel[1]
            outdata[5] = grad_vel[2]
            return 0

        def EvalHess(self, xdata, sigma, lam, outdata):
            """拉格朗日函数 Hessian 下三角（与 COPT 文档一致）。

            ``L(x) = σ·f(x) + λᵀc(x)``，须返回 ∇²L 在 ``idxHess`` 指定位置的值；
            仅 σ·∇²f 而忽略约束曲率会导致错误步长/收敛行为。
            """
            x = np.asarray(xdata, dtype=np.float64).ravel()
            lam_v = np.asarray(lam, dtype=np.float64).ravel()
            if lam_v.size < 2:
                lam_v = np.pad(lam_v, (0, max(0, 2 - lam_v.size)))
            sigma = float(sigma)
            l0, l1 = float(lam_v[0]), float(lam_v[1])
            self.x = x
            h = 1e-6

            def lagrangian(xv: np.ndarray) -> float:
                fv = self.optimizer.objective_function(xv)
                c1 = self.optimizer.constraint_position(xv)
                c2 = self.optimizer.constraint_velocity_parallel(xv)
                return sigma * fv + l0 * c1 + l1 * c2

            L0 = lagrangian(x)
            hess_L = np.zeros((3, 3))
            for i in range(3):
                for j in range(i + 1):
                    x_ij = x.copy()
                    x_ij[i] += h
                    x_ij[j] += h
                    L_ij = lagrangian(x_ij)

                    x_i = x.copy()
                    x_i[i] += h
                    L_i = lagrangian(x_i)

                    x_j = x.copy()
                    x_j[j] += h
                    L_j = lagrangian(x_j)

                    hess_L[i, j] = (L_ij - L_i - L_j + L0) / (h * h)
                    hess_L[j, i] = hess_L[i, j]

            idx = 0
            for i in range(3):
                for j in range(i + 1):
                    outdata[idx] = hess_L[i, j]
                    idx += 1
            return 0

    def _apply_copt_nlp_params(model: Any, options: dict[str, Any]) -> None:
        """与参考脚本一致：``model.setParam(COPT.Param.*, ...)``（NLP 项 + 可选 TimeLimit）。"""
        assert COPT is not None
        model.setParam(COPT.Param.NLPTol, 1e-10)
        model.setParam(COPT.Param.NLPIterLimit, int(options.get("max_iter", 1000)))
        model.setParam(COPT.Param.Threads, int(options.get("threads", 1)))
        model.setParam(COPT.Param.BarThreads, int(options.get("bar_threads", 1)))
        tl = options.get("time_limit")
        if tl is not None:
            model.setParam(COPT.Param.TimeLimit, float(tl))

    class COPTNLPSolver:
        """基于 COPT 的 NLP 封装：``cp.Envr()`` → ``createModel`` → ``loadNlData`` → ``solve``。

        Args:
            optimizer: :class:`DROTRONLPOptimizer` 实例。
            options: COPT 求解参数（``max_iter``、``threads`` 等）。
        """

        def __init__(self, optimizer: DROTRONLPOptimizer, options: dict[str, Any] | None = None):
            self.optimizer = optimizer
            self.options = options or {}
            self.model: Any = None
            self.callback: COPTNLPCallback | None = None

        def _setup_model(self, x0: np.ndarray) -> bool:
            """构建 COPT NLP 模型（变量、约束、Jacobian/Hessian 结构）。

            Args:
                x0: 初始变量 ``[alpha, T, t_ins]``。

            Returns:
                ``True`` 表示模型构建成功。
            """
            if cp is None or COPT is None:
                raise RuntimeError("COPT not installed. Install with: pip install coptpy")

            env = cp.Envr()
            self.model = env.createModel("DRO_RO_Transfer_NLP")
            _apply_copt_nlp_params(self.model, self.options)

            alpha_lb, alpha_ub = self.optimizer.alpha_range
            t_lb, t_ub = self.optimizer.transfer_time_range
            tins_lb, tins_ub = self.optimizer.t_ins_range

            col_lower = [alpha_lb, t_lb, tins_lb]
            col_upper = [alpha_ub, t_ub, tins_ub]

            row_lower = [0.0, 0.0]
            row_upper = [0.0, 0.0]

            self.callback = COPTNLPCallback(self.optimizer)

            n_jac = 6
            idx_jac_row = [0, 0, 0, 1, 1, 1]
            idx_jac_col = [0, 1, 2, 0, 1, 2]

            n_hess = 6
            idx_hess_row = [0, 1, 1, 2, 2, 2]
            idx_hess_col = [0, 0, 1, 0, 1, 2]

            self.model.loadNlData(
                nCols=3,
                nRows=2,
                sense=COPT.MINIMIZE,
                nGrad=3,
                idxGrad=[0, 1, 2],
                nJac=n_jac,
                idxJacRow=idx_jac_row,
                idxJacCol=idx_jac_col,
                nHess=n_hess,
                idxHessRow=idx_hess_row,
                idxHessCol=idx_hess_col,
                colLower=col_lower,
                colUpper=col_upper,
                rowLower=row_lower,
                rowUpper=row_upper,
                initX=list(x0),
                evalType=-1,
                cb=self.callback,
            )

            _apply_copt_nlp_params(self.model, self.options)

            return True

        def solve(self, x0: np.ndarray) -> dict[str, Any]:
            """求解 NLP 模型。

            Args:
                x0: 初始变量 ``[alpha, T, t_ins]``。

            Returns:
                含 ``status``、``objective``、``solution``、``success`` 的字典。
            """
            if self.model is None:
                self._setup_model(x0)

            try:
                self.model.solve()

                status = self.model.status
                assert COPT is not None
                obj_val = self.model.objval if status == COPT.OPTIMAL else float("inf")
                solution = self.model.x if hasattr(self.model, "x") else x0

                return {
                    "status": status,
                    "objective": obj_val,
                    "solution": solution,
                    "success": status == COPT.OPTIMAL,
                }
            except Exception as e:
                return {
                    "status": -1,
                    "objective": float("inf"),
                    "solution": x0,
                    "success": False,
                    "message": str(e),
                }

        def get_result(self) -> TransferOptimizationResult:
            """从 COPT 求解结果构建 :class:`TransferOptimizationResult`。

            Raises:
                RuntimeError: 尚未调用 ``solve()`` 时。
            """
            if self.model is None or self.callback is None or self.callback.x is None:
                raise RuntimeError("Must call solve() first")

            assert COPT is not None
            assert self.callback is not None
            opt_vars = NLPOptimizationVariables.from_array(self.callback.x)
            success = self.model.status == COPT.OPTIMAL

            return self.optimizer._build_result(
                opt_vars,
                success,
                "COPT solution" if success else f"COPT status: {self.model.status}",
            )


else:
    # 未安装 coptpy：保留同名符号以满足类型注解与 ``is None`` 检查。
    COPTNLPSolver = None  # type: ignore[misc, assignment]

    def _apply_copt_nlp_params(model: Any, options: dict[str, Any]) -> None:
        """未安装 COPT 时的占位实现；调用方应避免到达此处。"""
        pass


def optimize_with_copt(
    optimizer: DROTRONLPOptimizer,
    initial_guess: NLPOptimizationVariables | None = None,
    *,
    fallback_to_scipy: bool = True,
    max_iter: int = 1000,
    threads: int = 1,
    bar_threads: int = 1,
    time_limit: float | None = None,
    scipy_fallback_kwargs: dict[str, Any] | None = None,
) -> TransferOptimizationResult:
    """使用 COPT 求解 NLP（与 ``data_processing_module`` 中用法一致：
    ``cp.Envr`` / ``createModel`` / ``COPT.Param`` / ``solve``）。

    数学形式与 :func:`e2m2e.transfer.nlp_scipy.solve_with_scipy` 相同
    （等式约束 + 最小化 Δv）。

    Args:
        optimizer: 已设置 ``alpha_range`` / ``transfer_time_range`` / ``t_ins_range`` 的
            :class:`DROTRONLPOptimizer`
        initial_guess: 初始猜测 ``(α, T, t_ins)``；默认 ``(1, 10, 5)``
        fallback_to_scipy: 未安装 COPT 或求解失败时是否回退 SciPy SLSQP
        max_iter: ``COPT.Param.NLPIterLimit`` （最大迭代数）
        threads / bar_threads: ``COPT.Param.Threads`` / ``BarThreads``
            （Python 回调建议为 1）
        time_limit: 若给定，则设置 ``COPT.Param.TimeLimit`` （秒），
            与参考脚本中 MILP 用法一致
        scipy_fallback_kwargs: 回退时传给 ``optimizer.optimize`` 的额外参数

    Returns:
        :class:`TransferOptimizationResult`
    """
    if scipy_fallback_kwargs is None:
        scipy_fallback_kwargs = {}

    def _run_scipy() -> TransferOptimizationResult:
        return optimizer.optimize(initial_guess=initial_guess, **scipy_fallback_kwargs)

    if cp is None or COPT is None or NlpCallbackBase is None:
        if fallback_to_scipy:
            return _run_scipy()
        raise RuntimeError(
            "coptpy 未安装，无法使用 COPT；请安装 coptpy 或设置 fallback_to_scipy=True"
        )

    if initial_guess is None:
        alpha0, T0, tins0 = 1.0, 10.0, 5.0
    else:
        alpha0 = initial_guess.alpha
        T0 = initial_guess.transfer_time
        tins0 = initial_guess.t_ins

    x0 = np.array([alpha0, T0, tins0], dtype=float)

    copt_options: dict[str, Any] = {
        "max_iter": max_iter,
        "threads": threads,
        "bar_threads": bar_threads,
    }
    if time_limit is not None:
        copt_options["time_limit"] = time_limit

    try:
        solver = COPTNLPSolver(optimizer, copt_options)
        result = solver.solve(x0)

        if result["success"]:
            return solver.get_result()
        if fallback_to_scipy:
            return _run_scipy()
        try:
            return solver.get_result()
        except RuntimeError:
            return optimizer._build_result(
                NLPOptimizationVariables.from_array(np.asarray(x0, dtype=float)),
                False,
                "COPT 未收敛且无可用解向量",
            )
    except Exception:
        if fallback_to_scipy:
            return _run_scipy()
        raise
