"""验证：Rust 新版原子打靶器（LM 阻尼 + 线搜索 + mask）在单圈 / 合并段的保形表现。

Python 实验（_exp_splice_v3）已证：var_time 全自由 + 线搜索收敛且保形。
本脚本用 Rust 原子打靶器复现，对比残差与保形指标，确认 Rust 实现与
Python 语义等价、可直接用于 segmented 分支。

运行：``uv run python scripts/_verify_rust_shooting.py``
"""
from __future__ import annotations

import time

import numpy as np

from e2m2e._integrators import multiple_shooting_correct_py
from e2m2e.algorithm.coordinate.synodic_j2000 import SynodicJ2000System
from e2m2e.algorithm.design.design_orbit import (
    _cr3bp_orbit_for,
    _epoch_to_iso,
    _sample_patch_points,
    _validate_params,
    default_kernel_dir,
    load_design_kernels,
)
from e2m2e.algorithm.dynamics import CR3BP_Dynamics
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


def _rust_forces(fm, full_system):
    """从 ForceModel 提取传给 Rust 的 force 元组列表。"""
    forces_py = []
    for entry in fm.list_forces():
        if not entry.enabled:
            continue
        spec = entry.force.to_rust_spec(full_system)
        if spec is not None:
            forces_py.append(spec)
    return forces_py


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
        du = system.characteristic_length
        assert du is not None
        mu = system.mu
        et0 = spice.utc_to_et(_epoch_to_iso(EPOCH))
        state0_syn = np.asarray(cr3bp_orbit.states[0], dtype=float)
        fm = _build_force_model(spice)
        syn_j2000 = SynodicJ2000System(cr3bp_system=system, spice=spice)

        # 构造 full_system 以提取 rust forces
        from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
        from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
        from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
        from e2m2e.algorithm.dynamics import EphemerisSystem

        full_system = EphemerisSystem(
            bodies=["EARTH", "MOON", "SUN"], spice=spice, origin="EARTH"
        )
        full_system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )
        forces_py = _rust_forces(fm, full_system)

        # 单圈 var_time 全自由
        t_patch_syn, state_patch_syn = _sample_patch_points(
            dynamics, state0_syn, period, 1, perilune_clustered=False
        )
        seg_t = et0 + t_patch_syn * t_c
        seg_s = syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn, t_syn_arr=t_patch_syn, et0=et0
        )
        t0 = time.perf_counter()
        r = multiple_shooting_correct_py(
            forces_py, "EARTH", list(seg_t), [list(map(float, x)) for x in seg_s],
            var_time=True, fix_first_node=False, fixed_node_mask=None,
            max_iter=50, tolerance=1e-4, rtol=1e-10,
        )
        dt = time.perf_counter() - t0
        sp = np.asarray(r.state_patch)
        rr = np.linalg.norm(sp[:, :3], axis=1) / du
        spread = (rr.max() - rr.min()) / rr.mean()
        print(f"Rust 单圈 var_time 全自由: conv={r.converged} iter={r.iterations} "
              f"res={r.max_residual:.1e} km ({dt:.1f}s)")
        print(f"  残差历史: {[f'{x:.1e}' for x in r.residual_history[:6]]}")
        print(f"  |r|/du 极差/均值 = {spread:.3f}（保形 < 0.3）")
        t_syn = (np.asarray(r.t_patch) - et0) / t_c
        syn_states = syn_j2000.batch_j2000_to_synodic(sp, t_syn, et0)[:, :3]
        syn_states[:, 0] += mu
        print(f"  会合系 x: [{syn_states[:, 0].min():.4f}, {syn_states[:, 0].max():.4f}]")

        # 两圈合并：各自独立转星历后拼接，合并段固定两端 var_time
        segs = []
        for k in range(2):
            t_patch_syn2, state_patch_syn2 = _sample_patch_points(
                dynamics, state0_syn, period, 1, perilune_clustered=False
            )
            seg_t2 = et0 + (k * period + t_patch_syn2) * t_c
            seg_s2 = syn_j2000.batch_synodic_to_j2000(
                states_syn=state_patch_syn2,
                t_syn_arr=(k * period + t_patch_syn2),
                et0=et0,
            )
            r1 = multiple_shooting_correct_py(
                forces_py, "EARTH", list(seg_t2), [list(map(float, x)) for x in seg_s2],
                var_time=True, fix_first_node=False, fixed_node_mask=None,
                max_iter=50, tolerance=1e-4, rtol=1e-10,
            )
            segs.append((np.asarray(r1.t_patch), np.asarray(r1.state_patch)))
            print(f"第 1 步段 {k + 1}: conv={r1.converged} res={r1.max_residual:.1e} km")

        t1, s1 = segs[0]
        t2, s2 = segs[1]
        merged_t = np.concatenate([t1, t2[1:]])
        merged_s = np.concatenate([s1, s2[1:]])
        n = len(merged_t)
        # 合并段固定首末两端（var_time：状态固定但时间自由）
        mask = [False] * n
        mask[0] = True
        mask[-1] = True
        t0 = time.perf_counter()
        rm = multiple_shooting_correct_py(
            forces_py, "EARTH", list(merged_t), [list(map(float, x)) for x in merged_s],
            var_time=True, fix_first_node=False, fixed_node_mask=mask,
            max_iter=50, tolerance=1e-4, rtol=1e-10,
        )
        dt = time.perf_counter() - t0
        spm = np.asarray(rm.state_patch)
        rm_ = np.linalg.norm(spm[:, :3], axis=1) / du
        spread_m = (rm_.max() - rm_.min()) / rm_.mean()
        print(f"\n合并段（{n} 节点，固定两端）: conv={rm.converged} iter={rm.iterations} "
              f"res={rm.max_residual:.1e} km ({dt:.1f}s)")
        print(f"  残差历史: {[f'{x:.1e}' for x in rm.residual_history[:8]]}")
        print(f"  |r|/du 极差/均值 = {spread_m:.3f}（保形 < 0.3）")
        t_syn2 = (np.asarray(rm.t_patch) - et0) / t_c
        syn_m = syn_j2000.batch_j2000_to_synodic(spm, t_syn2, et0)[:, :3]
        syn_m[:, 0] += mu
        print(f"  会合系 x: [{syn_m[:, 0].min():.4f}, {syn_m[:, 0].max():.4f}]")
    finally:
        spice.unload_kernel(default_kernel_dir())


def _build_force_model(spice: SPICEManager) -> ForceModel:
    from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
    from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
    from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
    from e2m2e.algorithm.dynamics import EphemerisSystem

    bodies = ["EARTH", "MOON", "SUN"]
    full_system = EphemerisSystem(bodies=bodies, spice=spice, origin="EARTH")
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
    return fm


if __name__ == "__main__":
    main()
