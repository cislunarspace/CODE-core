"""星历修正的实验调参常量。

收敛容差、速度加权、每圈节点数、拼接点采样策略与固定时间打靶族
集合。取值依据与实测记录见各常量的注释块；策略性默认值（默认摄动
开关、内核清单）留在 ``design_orbit.py``。
"""

from __future__ import annotations

#: 星历修正的默认收敛容差（km，6 维状态 max 范数：位置 km + 速度 km/s）。
#:
#: 取 2e-2（20 m）。地月尺度（特征长度 3.84e5 km）下 10 m 已属高精度、
#: 1 km 都不错；更紧的容差超出星历模型与 CR3BP 初猜的物理可收敛底线
#: （紧凑近月 DRO 因月球引力梯度强，底线约 ~5.7e-5 km），会导致求解器
#: 停在 ~1.2e-2 km 永远报 not converged。容差 2e-2 给 solver 留裕度
#: （实测单圈收敛残差 1.7e-2、3 圈段 1.5e-2，均 < 2e-2 明确收敛），
#: 迭代数大减，保形不受影响。
CORRECTION_TOL_KM = 2e-2

#: 星历修正的速度连续目标（km/s）。0.01 m/s——patch point 速度跳变小于此即视为
#: 光滑轨道（非需脉冲拼接的断弧）。依据：DRO 等中性稳定轨道，速度连续的修正解
#: 才落在准周期轨道上，自由外推才有界（实测 amp=60000 DRO 坏历元，速度残差
#: 从 ~25 m/s 压到 <0.01 m/s 后，30 天传播从发散 200000 km 收敛到有界 ~80000 km）。
VELOCITY_TOL_KMS = 1e-5

#: 多重打靶速度残差加权 = 位置容差 / 速度容差。Rust 打靶残差向量把位置（km）与
#: 速度（km/s）混在一起取 ‖F‖²，cislunar 下位置项（几百 km）单边主导，速度项被
#: 忽略，求解器停在"位置连续 / 速度跳变数十 m/s"的局部极小。乘以 vel_weight 后
#: 两者在容差尺度可比，LM 真正压速度连续（见 Rust ``build_residual`` 注释）。
CORRECTION_VEL_WEIGHT = CORRECTION_TOL_KM / VELOCITY_TOL_KMS

#: 每圈 patch 节点数（均匀采样基线；DPO/Halo/NRHO 按族覆盖）
_POINTS_PER_REV = 8

#: DPO 的每圈 patch 节点数。默认振幅 20000 km 的 DPO 周期约 23 天，
#: 8 个等时间节点会产生约 2.9 天的自由传播弧，CR3BP→星历模型的节点
#: 跳变可达 4e4 km；64 个节点将弧段缩短至约 0.36 天，实测 GUI 默认场景
#: 可收敛。
_DPO_POINTS_PER_REV = 64

#: 拼接点采样策略（按轨道族覆盖，不暴露到请求模型）：
#: - uniform：等时间（NRHO 生产默认；其余族默认）
#: - perilune_clustered：近月点加密（Halo 长弧实测更稳）
#: - drop_near_perilune：删近月点附近节点（工具函数/对照，非生产默认：
#:   phase=0.5 约 1 个月弧 + 3 圈/段合并层易卡，且未钉历元会出前缀空洞）
_PATCH_SAMPLING_UNIFORM = "uniform"
_PATCH_SAMPLING_PERILUNE_CLUSTERED = "perilune_clustered"
_PATCH_SAMPLING_DROP_NEAR_PERILUNE = "drop_near_perilune"

#: 星历修正用固定时间打靶（var_time=False）的轨道族：Halo/NRHO/DPO/Lyapunov（不稳定，
#: 分段打靶全程固定时刻，对齐杨洪伟 2015）、拟周期/无周期闭合族
#: （Lissajous / 三角平动点 L4/L5）与 Axial。
#:
#: 拟周期族用固定时间的机理：CR3BP 初猜无周期闭合（Lissajous
#: 面内/面外频率不可约；L4/L5 短/长周期模态耦合），自由时间模式下时间
#: 自由度与沿流状态自由度近似线性相关（时间平移 δt ≈ 沿轨道移动 δt·f），
#: 雅可比列病态，LM 陷入线性收敛卡在 0.5–174 km（实测 L2/L4/L5 迭代到
#: 80 次上限不收敛）。固定时间下节点时刻保持 CR3BP 名义周期均匀采样，
#: 位置/速度修正直接吸收星历偏差，Gauss-Newton 二次收敛（实测 4–6 迭代，
#: 秒级到几十秒）。
#:
#: Axial 同属此病态：它从 Lyapunov 族 1:1 共振分岔（Gómez Type B）产生，
#: 分岔邻域面内周期 = 面外周期，时间平移与面外相位平移近似简并，自由
#: 时间打靶雅可比列病态——实测 L2/L1 默认参数 LM 停滞（
#: STAGNATION_DETECTED，15/17 次迭代后位置残差停在 1.5e-01 / 1.1e+01 km）；
#: 固定时间后两种修正方法均在约 10 s 内收敛到容差内。
#:
#: RO（共振轨道）同属此病态：近圆轨道时间平移 ≈ 沿轨相位旋转，自由
#: 时间打靶雅可比近似简并——实测 3:1 RO var_time 打靶 10 次迭代后
#: 停滞在位置残差 4.0 km（STAGNATED，554 s）；固定时间 5 次迭代收敛到
#: 2.7e-4 km（18 s）。
#: Lyapunov 平面轨道同属此病态：面内周期轨道时间平移与沿轨相位旋转简并，
#: 自由时间打靶雅可比病态；固定时间打靶收敛稳健。
_FIXED_TIME_ORBIT_TYPES = frozenset(
    {"HALO", "NRHO", "DPO", "LYAPUNOV", "LISSAJOUS", "L4", "L5", "AXIAL", "RO"}
)
