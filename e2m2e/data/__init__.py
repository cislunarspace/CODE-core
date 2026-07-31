"""数据层：星历数据管理、时空参考系数据、数据模板、通用数据类型。

第 1 层，依赖方向：仅外部库（SPICE/r2s2/numpy），不依赖 e2m2e 其他层。

- ``kernels/``：SPICE 内核管理与 ``EphemerisProvider`` 抽象（对上层屏蔽数据来源）。
- ``frames/``：时空参考系**数据**（EOP、闰秒表、历表句柄）；转换**算法**在
  ``algorithm/coordinate/``。
- ``templates/``：数据模板（轨道族种子、系统参数、摄动开关、力模型配置 schema、领域枚举）。
- ``types/``：通用数据类型（State/Epoch 类型别名；Orbit/EphemerisTable/NominalOrbit 容器类）。

实现状态：骨架。模块逐个实现中，未实现模块的占位函数抛 ``NotImplementedError``。
"""

__all__: list[str] = []
