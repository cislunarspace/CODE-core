"""DRO 设计 vs DFH 黄金样本差分统计（容差标定用，一次性脚本）。"""

import time

import numpy as np

from e2m2e.algorithm.design import design_orbit
from e2m2e.data.types.trajectory import read_ephemeris

t0 = time.time()
r = design_orbit(
    "DRO",
    amplitude=10000.0,
    phase=0.5001,
    epoch=(2024, 1, 1, 0, 0, 0.0),
    duration=0.1,
    output_step=3600.0,
    perturbation={
        "sun_body": 1,
        "planets": 1,
        "earth_nonspherical": 1,
        "moon_nonspherical": 1,
        "solar_radiation": 0,
        "atmosphere": 0,
        "relativity": 0,
        "tide": 0,
        "coupling": 0,
    },
)
print(f"设计耗时 {time.time() - t0:.0f}s", flush=True)
print(
    f"修正: converged={r.correction.converged}, iter={r.correction.iterations}, "
    f"res={r.correction.max_residual:.2e} km"
)
print(f"CR3BP: T={r.cr3bp_orbit.period:.4f} TU, C={r.cr3bp_jacobi:.4f}")
print(f"星历: {len(r.ephemeris)} 点")

golden = read_ephemeris("tests/dfh/fixtures/EPHEMERIDES_DESIGN_DRO.TXT")
ours = r.ephemeris
n = min(len(golden), len(ours))
# 逐点差分（前 n 点，时间网格同为 3600s 对齐）
pos_err = np.linalg.norm(ours.position_km[:n] - golden.position_km[:n], axis=1)
vel_err = np.linalg.norm(ours.velocity_mps[:n] - golden.velocity_mps[:n], axis=1)
print(f"公共 {n} 点:")
print(f"  位置差: max={pos_err.max():.2f} km @ {np.argmax(pos_err)}h, mean={pos_err.mean():.2f}")
print(f"  速度差: max={vel_err.max():.4f} m/s, mean={vel_err.mean():.4f}")
for h in [0, 24, 100, 200, 400, 600, 800, n - 1]:
    if h < n:
        print(f"  t={h:4d}h: pos={pos_err[h]:10.2f} km, vel={vel_err[h]:8.4f} m/s")
