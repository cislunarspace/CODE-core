"""design_orbit 任务模板：星历修正方法的族级分派。

数据模板层（ADR 0011）：``DesignOrbitRequest`` 的请求校验与算法层
``design_orbit`` 的防御检查共用此表，保证族→方法映射唯一事实源。
"""

from __future__ import annotations

#: 星历修正强制 segmented 的不稳定轨道族：two_level/standard 的
#: "修正 1 圈 + 自由外推"对不稳定轨道必发散，只有 segmented（全程
#: 分段打靶）能产出不发散的标称参考轨道。圈间漂移是固有准周期特征，
#: 由 station_keeping 处理。
SEGMENTED_CORRECTION_ORBIT_TYPES: frozenset[str] = frozenset({"HALO", "NRHO", "DPO"})
