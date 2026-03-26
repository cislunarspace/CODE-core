"""
生成 Halo 轨道族

使用Richardson三阶近似生成种子轨道，结合伪弧长延拓方法生成完整的Halo轨道族。

参考文献:
    Richardson, D. L. (1980). Analytic construction of periodic orbits
    about the collinear points. Celestial Mechanics.

注意:
    Halo轨道族延拓是一个非线性问题。Richardson三阶近似仅对小幅度的
    Halo轨道准确。伪弧长步长与 CR3BP_MATLAB_Library 中
    examples/FAMILY_L1Halo_North.m 一致：正向 DeltaS=0.0045，负向 |DeltaS|=0.009
    （由 step_size / step_size_negative 控制）。
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
# 3. 创建延拓器并生成种子轨道
# =============================================================================
corrector = e2m2e.algorithms.DifferentialCorrection(dynamic=dynamics)
continuation = e2m2e.algorithms.Continuation(corrector=corrector)

print(f"正在生成种子轨道: L{libration_point} {'北' if halo_class == 0 else '南'} Halo")
print(f"  Z振幅: {amplitude_z}")

seed_halo = continuation.generate_halo_seed_orbit(
    libration_point=libration_point,
    amplitude_z=amplitude_z,
    halo_class=halo_class,
    verbose=False,
)

if seed_halo is None:
    print("[error] 种子轨道生成失败")
    sys.exit(1)

print(f"[ok] 种子轨道生成成功: 周期={seed_halo.period:.6f} TU")
print(f"  x0={seed_halo.states[0, 0]:.6f}, z0={seed_halo.states[0, 2]:.6f}")

# =============================================================================
# 4. 使用halo_pseudo_arclength_continuation生成轨道族
# =============================================================================
print(f"\n开始Halo轨道族伪弧长延拓（continuation_PAL_CR3BP 流程）...")

n_orbits = 20
step_size = 0.0045
step_size_negative = 0.009

family_result = continuation.halo_pseudo_arclength_continuation(
    seed_orbit=seed_halo,
    n_orbits=n_orbits,
    direction="both",
    step_size=step_size,
    step_size_negative=step_size_negative,
    verbose=True,
)

print(f"\n[ok] 轨道族生成完成: 共{len(family_result)}条轨道")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
family_name = f"halo_L{libration_point}_{'N' if halo_class == 0 else 'S'}_family_{timestampNow()}"
family_result.save_to_file(filename=str(OUTPUT_DIR / f"{family_name}.json"))

print(f"\n[ok] 轨道族已保存至: {OUTPUT_DIR / f'{family_name}.json'}")
print(f"  轨道族名称: {family_name}")
if len(family_result) > 0:
    z_values = [o.parameters.get("amplitude_z", 0) for o in family_result]
    print(f"  z_amplitude 范围: [{min(z_values):.4f}, {max(z_values):.4f}]")
