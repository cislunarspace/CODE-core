"""
生成 Halo 轨道

使用Richardson三阶近似生成初始猜测，结合微分修正生成精确的Halo周期轨道。

参考文献:
    Richardson, D. L. (1980). Analytic construction of periodic orbits
    about the collinear points. Celestial Mechanics.

初值来源:
    FAMILY_L1Halo_North.m (MATLAB参考)
    SV0 = [0.9305, 0, 0.2300, 0, 0.1043, 0]'
    tf = 1.8397 (完整周期)
"""

import sys
from pathlib import Path

from fontTools.misc.timeTools import timestampNow

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import e2m2e
from e2m2e.core import Orbit

from scripts.utils.common import MU, TU

project_root = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = project_root / "output" / "halo"

# =============================================================================
# 1. 系统与动力学模型初始化
# =============================================================================
system = e2m2e.core.system.CR3BP_System(mu=MU, primary="earth", secondary="moon")
dynamics = e2m2e.core.dynamics.CR3BP_Dynamics(system=system)

# =============================================================================
# 2. Halo轨道参数
# =============================================================================
# Halo轨道特征：关于XZ平面对称，在拉格朗日点(L1/L2)附近振荡
# 状态向量格式：[x, y, z, vx, vy, vz]，均为无量纲量

libration_point = 1  # 1=L1, 2=L2
amplitude_z = 0.23  # Z方向振幅
halo_class = 0  # 0=北Halo (Class I), 1=南Halo (Class II)

# 目标参数
target_period = 1.839732  # 完整周期（无量纲时间单位）
t_half = target_period / 2  # 半周期

print(f"目标轨道: L{libration_point} {'北' if halo_class == 0 else '南'} Halo")
print(f"Z振幅: {amplitude_z}")
print(f"目标周期: {target_period:.4f} TU ({target_period * TU:.2f} days)")
print(f"半周期: {t_half:.4f} TU")

# =============================================================================
# 3. 配置微分校正器
# =============================================================================
corrector = e2m2e.algorithms.DifferentialCorrection(dynamic=dynamics)
corrector.setup_halo_orbit_fixed_z0(
    z0=amplitude_z if halo_class == 0 else -amplitude_z, libration_point=libration_point
)

print(f"\n微分校正器配置:")
print(f"  模式: setup_halo_orbit_fixed_z0")
print(f"  固定参数: z0 = {amplitude_z if halo_class == 0 else -amplitude_z}")
print(f"  自由变量: {corrector.free_variables}")
print(f"  约束条件: {list(corrector.target_conditions.keys())}")

# =============================================================================
# 4. 初始猜测（来自Richardson三阶近似）
# =============================================================================
# x0 = 0.9305269194214338  # L1位置附近
# vy0 = 0.1043 * amplitude_z / 0.23  # 按振幅缩放
# z0 = amplitude_z if halo_class == 0 else -amplitude_z
# 采用计算之后的值
x0 = 0.9305269194214338  # L1位置附近
z0 = 0.23
vy0 = 0.10431508546142665


initial_state = [x0, 0.0, z0, 0.0, vy0, 0.0]
times = [0]

orbit_init = Orbit(states=[initial_state], times=times)
orbit_init.period = target_period

print(f"\n初始猜测:")
print(f"  状态: {initial_state}")
print(f"  预估周期: {orbit_init.period:.4f} TU")

# =============================================================================
# 5. 执行迭代修正
# =============================================================================
corrector.max_iterations = 150
corrector.tolerance = 1e-6

print(f"\n开始迭代修正...")
orbit_result = corrector.iterate_correction(initial_guess=orbit_init, verbose=True)

# =============================================================================
# 6. 保存结果
# =============================================================================
if orbit_result is not None:
    print(f"\n[ok] 成功找到 Halo 轨道!")
    print(f"  修正后周期: {orbit_result.period:.6f} TU")
    print(f"  周期误差: {abs(orbit_result.period - target_period):.6e}")
    print(f"  初始状态: {orbit_result.states[0].tolist()}")

    output_file = (
        OUTPUT_DIR
        / f"halo_L{libration_point}_{'N' if halo_class == 0 else 'S'}_{amplitude_z}_{timestampNow()}.json"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orbit_result.save_to_file(filename=str(output_file))
    print(f"  保存至: {output_file}")
else:
    print(f"\n[error] 修正失败: {corrector.termination_reason}")
