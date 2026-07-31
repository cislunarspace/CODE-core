"""地月系统标准参数：物理常量、特征尺度、标准系统常量。

数据模板（纯数据，ADR 0011 迁移，源：``core/constants.py`` 与
``core/cr3bp_system.py`` 的类常量）。模型类（``System``/``CR3BP_System``）
在 ``algorithm/dynamics/``。

取值统一采用 GMAT/WGS84 标准，与 GMAT R2026a 默认值及多数既有代码一致
（历史上 drag 用 6378.137 km 而 shadow/gravity_file 等用 6378.1363 km；
srp 用 149597870.691 km 而 cr3bp_system 用 149597870.7 km，已统一）。
"""

from __future__ import annotations

# 地球赤道半径（km）。
# 取 GMAT PCK / IAU 2015 常用值 6378.1363 km，与 gravity_file、shadow、
# relativistic_correction 等力模型一致。
R_EARTH: float = 6378.1363

# 天文单位（km）。
# 取 GMAT nominalSun 值 149597870.7 km（GMAT R2026a
# SolarRadiationPressure 默认）。
AU: float = 149597870.7

# 千米 → 米换算因子。
KM_TO_M: float = 1000.0

# 地月平均距离（km），Cui et al. 2025。
EARTH_MOON_DISTANCE_KM: float = 384405.0

# 万有引力常数（km³ / (kg · s²)）。
G: float = 6.67430e-20

# 一天的秒数。
DAY: float = 86400

# 一年的秒数（儒略年）。
YEAR: float = 365.25 * 86400
