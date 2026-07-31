"""轨道族种子参数与地月系统标准参数。

实现状态：骨架。标准参数待从 ``dfh/cr3bp_orbits.py``（EARTH_MOON_MU、
CHAR_LENGTH_KM、CHAR_PERIOD_SEC、MOON_RADIUS_KM、DRO/Halo 种子）与
``core/constants.py`` 迁入。
"""

from __future__ import annotations

__all__: list[str] = []

# 以下常量待迁入（当前在 dfh/cr3bp_orbits.py / core/constants.py）：
#   EARTH_MOON_MU = 0.0121506683
#   CHAR_LENGTH_KM = 384400.0
#   CHAR_PERIOD_SEC = 27.32 * 86400.0
#   MOON_RADIUS_KM = 1737.4
#   _DRO_SEED_X0 / _DRO_SEED_VY0 / _DRO_SEED_PERIOD
#   _HALO_SEED_Z0 / _HALO_FOLD_Z0
