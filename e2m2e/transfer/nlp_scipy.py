"""SciPy SLSQP 后端。

把原先嵌入 :class:`~e2m2e.transfer.transfer_optimization.DROTRONLPOptimizer`
的 SciPy SLSQP 求解循环抽出为顶层函数 :func:`solve_with_scipy`，由
``DROTRONLPOptimizer.optimize`` 调用。SLSQP 是 DRO→RO 转移优化的默认求解器，
无需额外依赖，仅依赖 ``scipy>=1.10``。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, minimize

from .config import TransferOptimizationResult
from .nlp_core import NLPOptimizationVariables

if TYPE_CHECKING:
    from .transfer_optimization import DROTRONLPOptimizer


def solve_with_scipy(
    optimizer: DROTRONLPOptimizer,
    *,
    initial_guess: NLPOptimizationVariables | None = None,
    alpha_range: tuple[float, float] | None = None,
    transfer_time_range: tuple[float, float] | None = None,
    t_ins_range: tuple[float, float] | None = None,
    use_relaxed_velocity_constraint: bool | None = None,
    velocity_angle_constraint: float | None = None,
    verbose: bool | None = None,
) -> TransferOptimizationResult:
    """使用 SciPy SLSQP 求解 DRO→RO 转移 NLP。

    数学形式与论文 Cui et al. (2025) Section III.B 一致：最小化
    :math:`\\Delta v_1 + \\Delta v_2`，约束包括位置连续性、速度平行性
    （或松弛为不等式）以及变量范围。

    Args:
        optimizer: 已设置 ``alpha_range`` / ``transfer_time_range`` / ``t_ins_range``
            的 :class:`DROTRONLPOptimizer`。
        initial_guess: 初始猜测 ``(α, T, t_ins)``；默认 ``(1, 10, 5)``。
        alpha_range: 覆盖 ``optimizer.alpha_range``。
        transfer_time_range: 覆盖 ``optimizer.transfer_time_range``。
        t_ins_range: 覆盖 ``optimizer.t_ins_range``。
        use_relaxed_velocity_constraint: 是否使用松弛速度约束；``None`` 时取构造配置。
        velocity_angle_constraint: 松弛速度约束角度（弧度）；``None`` 时取构造配置。
        verbose: 是否打印迭代信息；``None`` 时取构造配置。

    Returns:
        :class:`TransferOptimizationResult`，包含优化详情与转移类型分类。
    """
    # 1. 范围覆盖
    if alpha_range is not None:
        optimizer.alpha_range = alpha_range
    if transfer_time_range is not None:
        optimizer.transfer_time_range = transfer_time_range
    if t_ins_range is not None:
        optimizer.t_ins_range = t_ins_range

    # 2. 默认值解析
    if use_relaxed_velocity_constraint is None:
        use_relaxed_velocity_constraint = optimizer._use_relaxed_velocity
    if velocity_angle_constraint is None:
        velocity_angle_constraint = optimizer.velocity_angle_tol
    if verbose is None:
        verbose = optimizer._verbose

    # 3. 初始猜测
    if initial_guess is None:
        alpha0, T0, t_ins0 = 1.0, 10.0, 5.0
    else:
        alpha0 = initial_guess.alpha
        T0 = initial_guess.transfer_time
        t_ins0 = initial_guess.t_ins
    y0 = np.array([alpha0, T0, t_ins0])

    # 4. 开启缓存（同一变量序列的积分结果可复用）
    optimizer.enable_cache(True)

    # 5. verbose 输出
    if verbose:
        print("\n开始NLP优化:")
        print(f"  初始猜测: α={alpha0:.4f}, T={T0:.4f}, t_ins={t_ins0:.4f}")
        print(f"  α范围: [{optimizer.alpha_range[0]}, {optimizer.alpha_range[1]}]")
        print(
            f"  T范围: [{optimizer.transfer_time_range[0]}, {optimizer.transfer_time_range[1]}]"
        )
        print(f"  t_ins范围: [{optimizer.t_ins_range[0]}, {optimizer.t_ins_range[1]}]")

    # 6. 构造约束
    constraints = [{"type": "eq", "fun": optimizer.constraint_position}]
    if use_relaxed_velocity_constraint:
        cos_theta_max = np.cos(velocity_angle_constraint)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda y: cos_theta_max - optimizer._compute_cos_angle(y),
            }
        )
    else:
        constraints.append({"type": "eq", "fun": optimizer.constraint_velocity_parallel})

    bounds = Bounds(
        lb=[
            optimizer.alpha_range[0],
            optimizer.transfer_time_range[0],
            optimizer.t_ins_range[0],
        ],
        ub=[
            optimizer.alpha_range[1],
            optimizer.transfer_time_range[1],
            optimizer.t_ins_range[1],
        ],
    )

    # 7. 进度回调
    iteration_counter = [0]

    def _scipy_callback(xk: np.ndarray) -> None:
        iteration_counter[0] += 1
        if optimizer._progress_callback is not None:
            alpha_k, T_k, tins_k = float(xk[0]), float(xk[1]), float(xk[2])
            obj_k = float(optimizer.objective_function(xk))
            optimizer._progress_callback(
                iteration_counter[0], obj_k, alpha_k, T_k, tins_k
            )

    # 8. 求解
    try:
        result = minimize(
            optimizer.objective_function,
            y0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000, "disp": verbose},
            callback=_scipy_callback,
        )

        success = result.success
        message = result.message
        final_y = result.x

    except Exception as e:
        success = False
        message = f"优化失败: {str(e)}"
        final_y = y0

    # 9. 组装结果
    opt_vars = NLPOptimizationVariables.from_array(final_y)
    opt_result = optimizer._build_result(
        opt_vars, success, message, use_relaxed_velocity_constraint, velocity_angle_constraint
    )

    # 10. verbose 结果输出
    if verbose:
        print("\n优化结果:")
        print(f"  成功: {opt_result.success}")
        print(f"  消息: {opt_result.message}")
        print(f"  α={opt_result.departure_alpha:.6f}")
        print(f"  T={opt_result.transfer_time:.6f}")
        print(f"  t_ins={opt_result.t_ins:.6f}")
        print(f"  ΔV1={opt_result.delta_v1:.6f}")
        print(f"  ΔV2={opt_result.delta_v2:.6f}")
        print(f"  总ΔV={opt_result.total_delta_v:.6f}")

    return opt_result
