"""数据层：星历数据管理、时空参考系数据、数据模板、通用数据类型。

第 1 层，依赖方向：仅外部库（SPICE/r2s2/numpy），不依赖 e2m2e 其他层。

- ``kernels/``：SPICE 内核管理（``SPICEManager``）与 ``EphemerisProvider`` 抽象。
- ``frames/``：时空参考系**数据** （EOP、闰秒表、r2s2/SPICE 句柄）；转换**算法** 在
  ``algorithm/coordinate/``。
- ``templates/``：数据模板（轨道族种子、系统参数、摄动开关、力模型配置 schema、领域枚举）。
- ``types/``：通用数据类型（State/Epoch 类型别名；Orbit/EphemerisTable/NominalOrbit 容器类）。

实现状态：迁移完成（ADR 0011），自 core/io 迁入；旧包已在第 5 批清理中删除。
"""

__all__: list[str] = []
