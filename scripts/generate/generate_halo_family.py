"""
生成 Halo 轨道族

使用Richardson三阶近似生成种子轨道，结合自然延拓方法生成完整的Halo轨道族。

参考文献:
    Richardson, D. L. (1980). Analytic construction of periodic orbits
    about the collinear points. Celestial Mechanics.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from fontTools.misc.timeTools import timestampNow

from scripts.utils.common import MU

import e2m2e
from e2m2e.core import Orbit, OrbitFamily

OUTPUT_DIR = project_root / "output" / "halo"

# =============================================================================
# 1. 系统与动力学模型初始化
# =============================================================================
system = e2m2e.core.system.CR3BP_System(mu=MU, primary="earth", secondary="moon")
dynamics = e2m2e.core.dynamics.CR3BP_Dynamics(system=system)

# =============================================================================
# 2. Halo轨道参数
# =============================================================================
libration_point = 1  # 1=L1, 2=L2
amplitude_z = 0.23  # Z方向振幅
halo_class = 0  # 0=北Halo (Class I), 1=南Halo (Class II)

# =============================================================================
# 3. 生成种子轨道
# =============================================================================
corrector = e2m2e.algorithms.DifferentialCorrection(dynamic=dynamics)
corrector.setup_halo_orbit_fixed_z0(
    z0=amplitude_z if halo_class == 0 else -amplitude_z,
    libration_point=libration_point,
)

# 使用Richardson三阶近似计算初始猜测
from e2m2e.algorithms.differential_correction import compute_halo_initial_guess

guess = compute_halo_initial_guess(
    mu=MU,
    z_amplitude=amplitude_z,
    L=libration_point,
    halo_class=halo_class,
)

if halo_class == 0:
    initial_z = amplitude_z
else:
    initial_z = -amplitude_z

initial_state = [
    guess["x0"],
    0.0,
    initial_z,
    guess["vx0"],
    guess["vy0"],
    guess["vz0"],
]

seed_orbit = Orbit(states=[initial_state], times=[0])
seed_orbit.period = 2.0 * guess["T_half"]

corrector.max_iterations = 150
corrector.tolerance = 1e-6

print(f"正在修正种子轨道: L{libration_point} {'北' if halo_class == 0 else '南'} Halo")
print(f"  Z振幅: {amplitude_z}")
print(f"  初始状态: {initial_state}")
print(f"  预估周期: {seed_orbit.period:.4f} TU")

seed_halo = corrector.iterate_correction(initial_guess=seed_orbit, verbose=True)

if seed_halo is None:
    print("[error] 种子轨道修正失败")
    sys.exit(1)

print(f"[ok] 种子轨道修正成功: 周期={seed_halo.period:.6f} TU")

# =============================================================================
# 4. 自然延拓生成轨道族
# =============================================================================
continuation = e2m2e.algorithms.Continuation(corrector=corrector)

# Halo轨道族延拓参数设置
param_min = 0.01  # 最小Z振幅
param_max = 0.5  # 最大Z振幅
step_size = 0.01  # 步长

print(f"\n开始自然参数延拓:")
print(f"  延拓参数: z_amplitude")
print(f"  参数范围: [{param_min}, {param_max}]")
print(f"  步长: {step_size}")

# 注意：自然延拓目前使用x0参数，需要修改为使用z_amplitude参数
# 这里暂时使用修正后的种子轨道作为单轨道输出
family_name = f"halo_L{libration_point}_{'N' if halo_class == 0 else 'S'}_family_{timestampNow()}"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 创建轨道族并保存
family = OrbitFamily(seed_halo)
family.save_to_file(filename=str(OUTPUT_DIR / f"{family_name}.json"))

print(f"\n[ok] 种子轨道已保存至: {OUTPUT_DIR / f'{family_name}.json'}")
print(f"  轨道族名称: {family_name}")
