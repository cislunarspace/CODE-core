#!/usr/bin/env python3
"""法型化流水线最小示例（issue #175）。

把一条 rho 坐标初值送进 ``NormalFormPipeline``，打印整条流水线的收敛摘要
与表征参数输出。

无 SPICE 内核时（如 CI 环境）底层自动降级到纯 CR3BP，本示例在该降级下仍
可跑通（仅供演示，不用于生产数据）。
"""

import warnings

import numpy as np

from e2m2e.algorithms.normal_form import NormalFormContext, NormalFormPipeline
from e2m2e.core import CR3BP_System, LibrationPoint


def main():
    print("=" * 60)
    print("e2m2e 法型化流水线示例")
    print("=" * 60)

    # 1. 构造上下文：地月 CR3BP、L1 点、J2000 历元、4 阶展开
    system = CR3BP_System(mu=1.215058560962404e-2, primary="Earth", secondary="Moon")
    system.set_characteristic_scales(distance=384405.0, period=27.32 * 86400.0)
    context = NormalFormContext(
        system=system,
        libration_point=LibrationPoint.L1,
        epoch=2451545.0,  # J2000 儒略日
        order=4,
    )
    print(f"\n上下文：{context}")

    # 2. rho 坐标初值 [ρ, ρ̇]（无量纲，围绕平动点的小偏移）
    x0 = np.array([1e-3, -1e-3, 0.0, 0.0, 1e-4, -1e-4])
    print(f"初值 x0 = {x0}")

    # 3. 一行流水线：星历轨道 → 表征参数
    #    小窗口 + 低阶中心流形，让示例在数秒内跑完（非生产配置）。
    pipeline = NormalFormPipeline(
        context=context,
        quasi_floquet_method="matrix",
        center_max_order=5,
        dynamical_kwargs={
            "t_total": 4.0,
            "node_step": 0.8,
            "dense_step": 0.2,
            "max_iter": 3,
            "tolerance": 1e-6,
            "prefer": "fft",
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 屏蔽 SPICE 降级警告
        result = pipeline.reduce(x0)

    # 4. 打印流水线摘要
    print("\n流水线摘要")
    print("-" * 40)
    print(f"  success      = {result.success}")
    print(f"  message      = {result.message}")
    print(f"  residual     = {result.residual:.3e}")
    print(f"  spice_used   = {result.metadata.get('spice_available')}")
    print(f"  qf 辛误差   = {result.metadata.get('qf_symplectic_error', float('nan')):.3e}")
    print(f"  cm 双曲耦合 = {result.metadata.get('cm_hyperbolic_coupling', float('nan')):.3e}")

    if result.catalog_transformer is not None:
        # 5. rho 坐标 → 表征参数 [q1, p1, I2, θ2, I3, θ3]
        param = result.catalog_transformer.rho_to_param(x0, t=0.0)
        print("\n表征参数（t=0）")
        print("-" * 40)
        labels = ["q1 (双曲)", "p1 (双曲)", "I2 (平面)", "θ2 (平面)", "I3 (垂直)", "θ3 (垂直)"]
        for label, value in zip(labels, param, strict=True):
            print(f"  {label:14s} = {value:+.6e}")

        # 6. 往返自检：param → rho 应还原初值
        back = result.catalog_transformer.param_to_rho(param, t=0.0)
        print(f"\n往返误差 ‖param→rho − x0‖∞ = {np.max(np.abs(back - x0)):.3e}")
    else:
        print("\n流水线未跑完，无表征参数输出。")

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
