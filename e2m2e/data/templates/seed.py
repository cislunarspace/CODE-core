"""轨道族种子参数与地月系统标准参数。

数据模板（ADR 0011 迁移，源：``dfh/cr3bp_orbits.py`` 常量）。初猜/族行走
算法在 ``algorithm/family/``。
"""

from __future__ import annotations

#: 地月 CR3BP 参数（与 examples/、tests/conftest.py 的标准系统一致）
EARTH_MOON_MU = 0.0121506683
CHAR_LENGTH_KM = 384400.0
CHAR_PERIOD_SEC = 27.32 * 86400.0

#: 月球平均半径（km），NRHO 近月点高度的起算面
MOON_RADIUS_KM = 1737.4

#: DRO 族标准种子（examples/main_design.py 等验证过的初值）
_DRO_SEED_X0 = 0.79188556619742
_DRO_SEED_VY0 = 0.53682
_DRO_SEED_PERIOD = 3.0

#: Halo 族种子面外振幅（无量纲，小振幅下 Richardson 近似精度高）
_HALO_SEED_Z0 = 0.001

#: 固定 z0 行走的安全上界（按平动点）：Halo 族折叠点 L1 在 |z0|≈0.085、
#: L2 在 |z0|≈0.20（PAL 测试实测），固定 z0 的修正接近折叠点即失效，
#: 超过此值改用固定 x0 行走
_HALO_FOLD_Z0 = {1: 0.07, 2: 0.15}
