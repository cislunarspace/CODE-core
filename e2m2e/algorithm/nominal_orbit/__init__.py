"""名义轨道：NominalOrbit + Interpolator（FR1↔FR2 契约）。

NominalOrbit 数据契约在 ``data/types/trajectory.py``（ADR 0015）；本模块放
插值器与使用逻辑（Gómez vol I §8.2.3：等间距历元状态表 + Floquet 基 +
投影因子表 + 高次插值器 Lagrange r=5~6）。Floquet 基 + 投影因子由 FR1
预计算，控制全程插值不复算。

实现状态：插值器待 FR1 落地（当前为占位）。
"""

__all__: list[str] = []
