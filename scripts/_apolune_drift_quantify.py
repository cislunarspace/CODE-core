"""量化 CR3BP 远月点在星历模型下的偏离（决定"钉远月点"锚定策略是否可行）。

关键问题：诊断显示 CR3BP tile 的**相位敏感点**（圈末）在星历模型下第 1 圈
就偏 2e5 km。但远月点（apolune，距月球最远、速度小）可能相对稳定。若
CR3BP 远月点在星历模型下偏离小（<1e3 km），"钉 CR3BP 远月点"锚定可行；
若也偏 2e5 km，锚定会引入错误，必须换策略。

运行：``uv run python scripts/_apolune_drift_quantify.py``
"""
from __future__ import annotations

import numpy as np

from e2m2e.algorithm.coordinate.synodic_j2000 import SynodicJ2000System
from e2m2e.algorithm.design.design_orbit import (
    _cr3bp_orbit_for,
    _epoch_to_iso,
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
N_REV = 8
PTS_PER_REV = 8


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

        # CR3BP 一圈稠密积分，定位远月点（距月球最远）
        n_dense = 720
        t_dense = np.linspace(0.0, period, n_dense + 1)
        res = dynamics.propagate(state0_syn, (0.0, period), t_eval=t_dense)
        states_syn = np.asarray(res["states"])
        moon_x = 1.0 - mu
        dist_moon = np.sqrt(
            (states_syn[:, 0] - moon_x) ** 2 + states_syn[:, 1] ** 2 + states_syn[:, 2] ** 2
        )
        i_apo = int(np.argmax(dist_moon))
        t_apo_rel = float(t_dense[i_apo])  # 远月点在圈内的相对时刻
        apo_dist = dist_moon[i_apo]
        apo_geod = np.linalg.norm(states_syn[i_apo, :3])
        print(f"CR3BP 远月点: t/T = {t_apo_rel / period:.4f}, "
              f"距月 {apo_dist:.3f} du, 距地 {apo_geod:.4f} du")
        print(f"  synodic 坐标 = {states_syn[i_apo, :3]}")

        # 各圈远月点（CR3BP tile，转 J2000）
        syn_j2000 = SynodicJ2000System(cr3bp_system=system, spice=spice)
        apo_states_j2000 = []
        apo_t_j2000 = []
        for k in range(N_REV):
            t_syn = t_apo_rel + k * period
            s_syn = np.asarray(
                dynamics.propagate_orbit_state_at_time(cr3bp_orbit, t_syn), dtype=float
            )
            s_j2000 = syn_j2000.synodic_to_j2000(s_syn, t_syn, et0)
            apo_states_j2000.append(s_j2000)
            apo_t_j2000.append(et0 + t_syn * t_c)
        apo_states_j2000 = np.asarray(apo_states_j2000)
        apo_t_j2000 = np.asarray(apo_t_j2000)

        # 全摄动自由积分：从首圈远月点积分 N_REV 圈，对比每圈 CR3BP 远月点
        fm = _build_force_model(spice)
        t_span = (float(apo_t_j2000[0]), float(apo_t_j2000[-1]))
        out = fm.propagate(
            apo_states_j2000[0],
            t_span,
            t_eval=apo_t_j2000,
            with_stm=True,
            max_steps=2_000_000,
        )
        states_eph = np.asarray(out["states"])
        drift = np.linalg.norm(states_eph[:, :3] - apo_states_j2000[:, :3], axis=1)
        print("\n全摄动自由积分（从首圈 CR3BP 远月点出发）vs 各圈 CR3BP 远月点:")
        print(f"  {'圈':>3} {'|Δr| (km)':>14} {'|Δr|/du':>10} {'|Δv| (km/s)':>14}")
        for k in range(N_REV):
            dv = np.linalg.norm(
                states_eph[k, 3:] - apo_states_j2000[k, 3:]
            ) * (du / t_c)
            print(f"  {k:>3} {drift[k]:>14.3e} {drift[k] / du:>10.3f} {dv:>14.3e}")

        # 结论
        d0 = drift[1]  # 第 2 圈远月点偏离（第 1 圈积分终点）
        verdict = "远小于相位敏感点 2e5 km，钉远月点锚定可行" if d0 < 1e4 else "同样大，锚定需谨慎"
        print(f"\n结论: 第 1 圈后远月点偏离 = {d0:.3e} km（{verdict}）")
    finally:
        spice.unload_kernel(default_kernel_dir())


if __name__ == "__main__":
    main()
