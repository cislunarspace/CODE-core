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

#: Axial 族种子面外速度（无量纲 DU/TU，小振幅下线性化初猜精度高）
_AXIAL_SEED_VZ0 = 0.001

#: DPO 族标准种子（从 DRO seed 反转 vy0 微分修正收敛所得）
#: DPO 为顺行（prograde），xy 平面内围绕月球的周期轨道，与 DRO 对称。
#: 种子 x0=0.90 处：vy0 < 0（顺行），period≈2.50。
_DPO_SEED_X0 = 0.90
_DPO_SEED_VY0 = -0.247645
_DPO_SEED_PERIOD = 2.5022

#: SPO（Short-Period Orbit）族标准种子
#: 来源：Capdevila & Howell (2018), JGCD, Table 1
#: L4/L5 三角平动点短周期族成员，xy 平面内周期轨道（z₀=ż₀=0）。
#: 坐标为质心会合系无量纲值。L5 种子由 CR3BP 对称性得到
#: （y₀→-y₀, ẋ₀→-ẋ₀）。
_SPO_L4_SEED_X0 = -0.2255
_SPO_L4_SEED_Y0 = 0.8660
_SPO_L4_SEED_VX0 = -0.2384
_SPO_L4_SEED_VY0 = 0.2494
#: 周期 28.3488 天，无量纲：T_days / (CHAR_PERIOD_SEC / 86400) * 2π
#: CHAR_PERIOD_SEC = 27.32 * 86400，T* ≈ 4.3423 天/TU
_SPO_SEED_PERIOD = 6.529

#: LPO（Long-Period Orbit）族标准种子
#: 来源：Gómez et al. (2001) Vol. II, Orbit F
#: 中间方程 Orbit F：大振幅 LPO，周期 3T_S ≈ 88 天
#: 坐标以 L5 为中心的相对坐标（无量纲）
_LPO_SEED_F_DX = 0.4100071795043306
_LPO_SEED_F_DY = -0.1028862236602286
_LPO_SEED_F_VX = 0.1137594676750601
_LPO_SEED_F_VY = -0.3961422724835140
_LPO_SEED_PERIOD = 20.374
