"""坐标转换算法：IAU2006/synodic↔J2000/GCRS↔EBCRS。

转换**算法**归这里（ADR 0015），``data/frames/`` 只留数据（EOP/闰秒/历表句柄）。
强化现有 Axes/Origin/CoordinateSystem 抽象（不新增 Frame 抽象）：所有坐标系表达
为 Axes + Origin + CoordinateSystem，时空间联合转换作为 CoordinateSystem 扩展方法。
转换算法最终留 Python，Rust 下沉是后续性能优化。

实现状态：骨架。Axes/Origin/CoordinateSystem/IAU2006/SynodicJ2000System/
GCRSEBCRSSystem 待从 ``core/coordinate/`` 迁入。
"""

from __future__ import annotations

__all__: list[str] = []
