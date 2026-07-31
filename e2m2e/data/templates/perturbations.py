"""摄动开关默认值与 DYB 系数默认值。

数据模板（ADR 0011 迁移，源：``io/inputs_dac.py`` 的 ``DEFAULT_*`` 常量）。
DFH 输入文件生成是开发期临时脚本，但默认值是通用契约，归数据层。
"""

from __future__ import annotations

#: 摄动开关默认值（与 MATLAB fmt_perturb_block.m 一致）
DEFAULT_PERTURBATION: dict[str, int] = {
    "sun_body": 1,
    "planets": 1,
    "earth_nonspherical": 1,
    "moon_nonspherical": 1,
    "solar_radiation": 2,
    "atmosphere": 0,
    "relativity": 0,
    "tide": 1,
    "coupling": 1,
}

#: DYB 系数默认值：DYB(1)=等效面质比 0.01 (m2/kg)，其余为相对 DYB(1) 的比值
DEFAULT_DYB: list[float] = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
