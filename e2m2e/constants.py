"""共享物理常量（单一来源）。

所有力模型与系统统一从此模块引用地球半径、天文单位等共享物理常量，
避免各模块重复定义导致取值漂移（历史上 ``drag`` 用 6378.137 km 而
``shadow``/``gravity_file`` 等用 6378.1363 km；``srp`` 用
149597870.691 km 而 ``cr3bp_system`` 用 149597870.7 km）。

取值统一采用 GMAT/WGS84 标准，与 GMAT R2026a 默认值及多数既有代码一致。
"""

from __future__ import annotations

# 地球赤道半径（km）。
# 取 GMAT PCK / IAU 2015 常用值 6378.1363 km，与 gravity_file、shadow、
# relativistic_correction 等力模型一致。注意 drag 模型旧值 6378.137 km
# (WGS84) 已统一至此标准，地心高度差 0.0007 km，对大气密度影响可忽略。
R_EARTH: float = 6378.1363

# 天文单位（km）。
# 取 GMAT nominalSun 值 149597870.7 km（GMAT R2026a
# SolarRadiationPressure 默认），与 cr3bp_system 一致。
AU: float = 149597870.7

# 千米 → 米换算因子。
KM_TO_M: float = 1000.0
