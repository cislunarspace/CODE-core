"""数据层：星历数据管理、时空参考系数据、数据模板、通用数据类型。

第 1 层，依赖方向：仅外部库（SPICE/r2s2/numpy），不依赖 e2m2e 其他层。

- ``kernels/``：SPICE 内核管理（``SPICEManager`` 加载/簿记；查询面 ``queries.py``、
  纯数据表 ``registry.py``）与 ``EphemerisProvider`` 抽象。
- ``frames/``：时空参考系**数据** （EOP、闰秒表、r2s2/SPICE 句柄）；转换**算法** 在
  ``algorithm/coordinate/``。
- ``templates/``：数据模板（轨道族种子、系统参数、摄动开关、力模型配置 schema、领域枚举）。
- ``types/``：通用数据类型（State/Epoch 类型别名；Orbit/EphemerisTable/NominalOrbit 容器类）。
- ``constants/``：声明式物理常数（``constants.toml`` 的 body/GM 表，Python loader 与
  Rust build script 同源生成）。
- ``catalog/``：轨道库建库基础设施（记录存储引擎；库目录与入库由调用方显式开启）。

仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。
"""

__all__: list[str] = []
