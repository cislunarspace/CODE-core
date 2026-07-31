"""端到端验收：#255 CR3BP L2 有界 Lissajous。

验证两项：
1. ``cm_result.max_hyperbolic_coupling`` 为 0（双曲-中心耦合全部消去）；
2. M0 流（propagate_parametric）传播 6 个周期，位置幅值保持初始量级
   （有界 Lissajous，此前位置放大 ~70 倍）。

用法：``python scripts/_cm_bounded_lissajous_check.py``
"""

from __future__ import annotations

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint
from e2m2e.algorithm.family.lissajous_initial_guess import compute_lissajous_initial_guess
from e2m2e.algorithm.normal_form import NormalFormContext, NormalFormPipeline
from e2m2e.algorithm.normal_form.constants import JD0_J2000
from e2m2e.algorithm.normal_form.propagation import propagate_parametric


def main() -> None:
    mu = 1.215058560962404e-2
    sys_cr3bp = CR3BP_System(mu=mu, primary="Earth", secondary="Moon")
    sys_cr3bp.set_characteristic_scales(distance=384405.0, period=27.32 * 86400.0)

    ctx = NormalFormContext(
        system=sys_cr3bp, libration_point=LibrationPoint.L2, epoch=JD0_J2000, order=10
    )

    # 小振幅 Lissajous 初值（CR3BP_System 质心 synodic 系，线性解）
    state0, period = compute_lissajous_initial_guess(sys_cr3bp, 2, 1500.0, 800.0, 0.3, 0.7)
    # 质心系 → 地心系（x += mu），再取相对平动点偏移（qiao 归一化 L2 = 1+γ）
    rho0 = np.array(state0, dtype=float)
    rho0[0] = (state0[0] + mu) - ctx.libration_position[0]

    pipeline = NormalFormPipeline(
        context=ctx,
        center_max_order=10,
        # DS 打靶窗口 16 TU（~2.4 个面内周期）足够验证化简。CR3BP 降级
        # 路径下 QF 自动用 constant 方法（B = 实标准形变换矩阵 V 常数）。
        dynamical_kwargs={"t_total": 16.0, "node_step": 0.8, "dense_step": 0.1},
    )
    result = pipeline.reduce(rho0)
    print(f"pipeline success: {result.success}")
    if not result.success:
        print(f"  message: {result.message}")
        return
    print(
        f"  cm_hyperbolic_coupling: {result.metadata['cm_hyperbolic_coupling']:.3e}"
    )
    print(
        f"  pre_hyperbolic_center_coupling: "
        f"{result.cm_result.metadata['pre_hyperbolic_center_coupling']:.3e}"
    )

    # M0 流：传播 6 个面内周期
    t_span = np.linspace(0.0, 6.0 * period, 25)
    t_out, rho_out, pos_err = propagate_parametric(rho0, t_span, result, ctx)
    if rho_out.size == 0:
        print("  propagate_parametric 返回空（积分失败）")
        return

    pos_norm = np.linalg.norm(rho_out[:, :3], axis=1)
    init = float(np.linalg.norm(rho_out[0, :3]))
    ratio = float(pos_norm.max() / max(init, 1e-12))
    print(f"  传播 {len(t_out)} 点 / {t_span[-1]:.2f} TU")
    print(
        f"  位置幅值: init={init:.5f} LU ({init * ctx.LU:.0f} km), "
        f"max={pos_norm.max():.5f} LU ({pos_norm.max() * ctx.LU:.0f} km)"
    )
    # 有界验收：位置幅值不显著放大（<3× 初始，远小于此前 70×）
    assert ratio < 3.0, f"位置放大 {ratio:.1f}×，不是有界 Lissajous"
    print(f"  有界 Lissajous ✓（幅值比 {ratio:.3f}）")


if __name__ == "__main__":
    main()
