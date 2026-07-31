"""时空参考系数据：EOP、闰秒表、历表句柄。

只留**数据**（EOP 文件、闰秒表、r2s2/SPICE 句柄管理）；转换**算法**在
``algorithm/coordinate/``（强化现有 Axes/Origin/CoordinateSystem 抽象，不新增
Frame 抽象，见 ADR 0015）。

实现状态：骨架。各文件待从 ``core/coordinate/`` 迁入。
"""

__all__: list[str] = []
