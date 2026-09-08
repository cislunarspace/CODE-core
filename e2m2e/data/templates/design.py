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

#: 支持设计的共振比 (p, q)（p:q = 卫星:月球，旋转系周期 T = (q/p)·T☾，
#: 与 orbit_taxonomy 的 resonant_p_q 术语及 ADR 0042 一致）：顺行内共振
#: 五档，即分类学 11 个比值中 p > q 的条目。初猜从共振周期条件构造
#: （Kepler 圆轨道，惯性频率 n = 1 + p/q），经固定半周期的 x 轴对称修正
#: 收敛到精确通约成员。q > p 的外共振档初猜不可靠（修正落入多圈伪解
#: 或长周期分支），不支持。
RO_SUPPORTED_RESONANCES: frozenset[tuple[int, int]] = frozenset(
    {(2, 1), (3, 1), (3, 2), (4, 1), (4, 3)}
)
