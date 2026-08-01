"""基准：Rust 打靶器 有缓存 vs 无缓存 的单圈耗时对比。

确认缓存是否真被 Rust 积分内循环使用（用户要求"星历预处理成表格存内存"）。
单圈打靶（24 次积分 × 多迭代），有缓存应显著快于无缓存（否则缓存没生效）。
运行：``uv run python scripts/_bench_cache.py``
"""
from __future__ import annotations

import time

import numpy as np

from e2m2e._integrators import multiple_shooting_correct_py
from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
from e2m2e.algorithm.coordinate.synodic_j2000 import SynodicJ2000System
from e2m2e.algorithm.design.design_orbit import (
    _cr3bp_orbit_for,
    _epoch_to_iso,
    _sample_patch_points,
    _validate_params,
    default_kernel_dir,
    load_design_kernels,
)
from e2m2e.algorithm.dynamics import CR3BP_Dynamics, EphemerisSystem
from e2m2e.algorithm.family.cr3bp_orbits import earth_moon_system
from e2m2e.algorithm.forces.force_mapping import dfh_perturbation_to_force_config
from e2m2e.algorithm.forces.force_model import ForceModel
from e2m2e.data.kernels.manager import SPICEManager

PERTURBATION = {
    "sun_body": 1,
    "planets": 0,
    "earth_nonspherical": 1,
    "moon_nonspherical": 1,
    "solar_radiation": 1,
    "atmosphere": 0,
    "relativity": 0,
    "tide": 0,
    "coupling": 0,
}
EPOCH = (2024, 1, 1, 0, 0, 0.0)


def main() -> None:
    spice = SPICEManager()
    load_design_kernels(spice, default_kernel_dir())
    try:
        system = earth_moon_system()
        dynamics = CR3BP_Dynamics(system)
        params = _validate_params(
            "HALO",
            amplitude=30000.0,
            phase=0.0,
            collinear_point=2,
            north_south=None,
            perilune_height=None,
            amplitude_in=None,
            amplitude_out=None,
            phase_in=None,
            phase_out=None,
        )
        cr3bp_orbit = _cr3bp_orbit_for("HALO", params, dynamics)
        period = float(cr3bp_orbit.period)
        t_c = system.characteristic_time
        assert t_c is not None
        et0 = spice.utc_to_et(_epoch_to_iso(EPOCH))
        state0_syn = np.asarray(cr3bp_orbit.states[0], dtype=float)
        syn_j2000 = SynodicJ2000System(cr3bp_system=system, spice=spice)

        full_system = EphemerisSystem(
            bodies=["EARTH", "MOON", "SUN"], spice=spice, origin="EARTH"
        )
        full_system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )
        force_config = dfh_perturbation_to_force_config(
            PERTURBATION, earth_degree=10, moon_degree=10
        )
        fm = ForceModel.from_config(force_config, full_system)
        fm.rtol = 1e-12
        fm.atol = 1e-12
        fm.max_step = 600.0
        forces_py = []
        for entry in fm.list_forces():
            if not entry.enabled:
                continue
            spec = entry.force.to_rust_spec(full_system)
            if spec is not None:
                forces_py.append(spec)

        t_patch_syn, state_patch_syn = _sample_patch_points(
            dynamics, state0_syn, period, 1, perilune_clustered=False
        )
        seg_t = et0 + t_patch_syn * t_c
        seg_s = syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn, t_syn_arr=t_patch_syn, et0=et0
        )

        def run(label: str) -> None:
            t0 = time.perf_counter()
            r = multiple_shooting_correct_py(
                forces_py, "EARTH", list(seg_t), [list(map(float, x)) for x in seg_s],
                var_time=True, fix_first_node=False, fixed_node_mask=None,
                max_iter=50, tolerance=1e-4, rtol=1e-10,
            )
            dt = time.perf_counter() - t0
            print(f"{label}: {dt:.1f}s, conv={r.converged} iter={r.iterations} "
                  f"res={r.max_residual:.1e}")

        # 无缓存
        run("无缓存")
        # 有缓存
        spice.enable_ephem_cache(
            ["EARTH", "MOON", "SUN"], et0, et0 + 86400 * 400,
            dt=3600.0, observer="EARTH",
            frame_pairs=[("ITRF93", "J2000"), ("MOON_PA", "J2000")],
        )
        run("有缓存")
        spice.disable_ephem_cache()
    finally:
        spice.unload_kernel(default_kernel_dir())


if __name__ == "__main__":
    main()
