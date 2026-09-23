"""design_orbit 任务模板：星历修正方法的族级分派。

数据模板层（ADR 0011）：``DesignOrbitRequest`` 的请求校验与算法层
``design_orbit`` 的防御检查共用此表，保证族→方法映射唯一事实源。
"""

from __future__ import annotations

#: 星历修正强制 segmented 的不稳定轨道族：two_level/standard 的
#: "修正 1 圈 + 自由外推"对不稳定轨道必发散，只有 segmented（全程
#: 分段打靶）能产出不发散的标称参考轨道。圈间漂移是固有准周期特征，
#: 由 station_keeping 处理。
SEGMENTED_CORRECTION_ORBIT_TYPES: frozenset[str] = frozenset({"HALO", "NRHO", "DPO", "LYAPUNOV"})

#: 支持设计的恒星共振比 (p, q)（p:q = 航天器惯性圈数:月球圈数）：q 个
#: 恒星月内绕地 p 圈，会合系周期 ``T = 2πq``、净卷绕 ``w = p−q``
#: （Vaquero & Howell 2014 式（7），与 orbit_taxonomy 的 resonant_p_q
#: 标签及 resonant.csv 目录族同支）。五档均为顺行内共振且锚定偏心族，
#: 近圆 w=1 支仅在 |p−q|=1 时与目录族重合。q > p 的外共振档初猜不可靠
#: （修正落入多圈伪解或长周期分支），不支持。
RO_SUPPORTED_RESONANCES: frozenset[tuple[int, int]] = frozenset(
    {(2, 1), (3, 1), (3, 2), (4, 1), (4, 3)}
)
