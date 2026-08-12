"""地月系统标准参数（已迁移至 ``e2m2e.data.constants``）。

本文件原为物理常量真值持有者，ADR 0022 阶段 3 已将其中的物理常数收编到
``e2m2e/data/constants/`` （单一来源）。为便于追溯，保留文件壳，不再定义
任何物理常量真值，也不再做 re-export shim。

历史说明：原 ``MU_EARTH``、``R_EARTH``、``AU``、``KM_TO_M``、
``EARTH_MOON_DISTANCE_KM``、``G``、``DAY``、``YEAR`` 等常量已迁移。
具体映射见 ADR 0022 或 ``e2m2e.data.constants`` 的导出表。
"""

from __future__ import annotations
