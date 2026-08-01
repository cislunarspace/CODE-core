"""星历模型下 Halo 初猜质量诊断：CR3BP 闭合误差 / 星历偏离 / Lyapunov 增长。

前置调查（Step 0，不进 CI）：量化 CR3BP 初猜在星历模型下能支撑多长的
分段打靶，据此定 ``revs_per_group``（第 1 步每组圈数）的安全上限。三件事：

(a) CR3BP Halo 单圈闭合误差：CR3BP 周期轨道本身不是精确闭合（微分修正后
    残差 ~1e-6），量级决定首圈打靶需吸收的残差。
(b) 逐圈星历偏离：CR3BP 首点状态在**全摄动星历模型**下自由积分 N 圈，
    对比 CR3BP 周期轨道第 N 圈节点，量化第几圈开始指数发散（>1e4 km 即
    初猜失真，段长应明显短于此圈数）。
(c) STM 一圈增长：一圈状态转移矩阵的谱半径（Lyapunov 倍率），估分段段长
    上限——打靶的雅可比依赖 STM，倍率过大则超长段收敛差。

运行：``uv run python scripts/_halo_initial_guess_diagnostic.py``
依赖：SPICE 内核在 ``kernels/``（design_orbit 的默认目录）。
"""
from __future__ import annotations

import numpy as np

from e2m2e.algorithm.coordinate.synodic_j2000 import SynodicJ2000System
from e2m2e.algorithm.design.design_orbit import (
    _cr3bp_orbit_for,
    _dense_orbit,
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

#: 与 main_design.py 一致的摄动开关（太阳第三体 + 地月 10 阶非球形 + 炮弹光压）
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
N_REV = 16  # 诊断积分圈数


def _build_force_model(spice: SPICEManager):
    """构造与 design_orbit segmented 分支同款的全摄动 ForceModel。"""
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
        et0 = spice.utc_to_et(_epoch_to_iso(EPOCH))

        # 相位 0 → 历元状态 = CR3BP 参考状态（y=0 穿越点）
        state0_syn = np.asarray(cr3bp_orbit.states[0], dtype=float)
        print(f"Halo L2: period = {period:.4f} TU = {period * t_c / 86400:.2f} 天")
        print(f"特征长度 = {system.characteristic_length:.0f} km，"
              f"特征时间 = {t_c / 86400:.2f} 天")

        # --- (a) CR3BP 单圈闭合误差 ---
        dense = _dense_orbit(dynamics, state0_syn, period)
        r_close = np.linalg.norm(
            dense.states[-1, :3] - dense.states[0, :3]
        ) * system.characteristic_length
        v_close = np.linalg.norm(
            dense.states[-1, 3:] - dense.states[0, 3:]
        ) * (system.characteristic_length / t_c)
        print(f"\n(a) CR3BP 单圈闭合误差: |Δr| = {r_close:.3e} km, "
              f"|Δv| = {v_close:.3e} km/s")

        # --- (b) 逐圈星历偏离：全摄动自由积分 vs CR3BP tile ---
        t_patch_syn, state_patch_syn = _sample_patch_points(
            dynamics, state0_syn, period, N_REV, perilune_clustered=False
        )
        syn_j2000 = SynodicJ2000System(cr3bp_system=system, spice=spice)
        state_patch_j2000 = syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn, t_syn_arr=t_patch_syn, et0=et0
        )
        t_patch_j2000 = et0 + t_patch_syn * t_c

        fm = _build_force_model(spice)
        # 从首点全摄动积分 N_REV 圈（覆盖整条 tile）
        t_span = (float(et0), float(t_patch_j2000[-1]))
        out = fm.propagate(
            state_patch_j2000[0],
            t_span,
            t_eval=t_patch_j2000,
            with_stm=True,
            max_steps=2_000_000,
        )
        states_eph = np.asarray(out["states"])
        drift = np.linalg.norm(states_eph[:, :3] - state_patch_j2000[:, :3], axis=1)
        du = system.characteristic_length
        print(f"\n(b) 全摄动自由积分 vs CR3BP tile（每圈末点）:")
        print(f"    {'圈':>3} {'|Δr| (km)':>14} {'|Δv| (km/s)':>14}")
        for k in range(N_REV):
            i = (k + 1) * 8 - 1  # 每圈 8 点，取圈末
            if i >= len(drift):
                break
            dv = np.linalg.norm(
                states_eph[i, 3:] - state_patch_j2000[i, 3:]
            ) * (du / t_c)
            print(f"    {k + 1:>3} {drift[i]:>14.3e} {dv:>14.3e}")

        # --- (c) 一圈 STM 增长（Lyapunov 倍率）---
        stm = np.asarray(out["stm"])
        print(f"\n(c) 一圈 STM 谱半径（Lyapunov 倍率）:")
        for k in range(min(4, N_REV)):
            i = (k + 1) * 8 - 1
            M = stm[i]
            s = np.linalg.svd(M, compute_uv=False)
            print(f"    圈 {k + 1}: 最大奇异值 = {s[0]:.3e}, 最小 = {s[-1]:.3e}")

        # --- 结论：定 revs_per_group 上限 ---
        # 星历偏离 > 1e4 km 的圈数即段长安全上限（打靶需吸收的残差）
        exceed = np.where(drift > 1e4)[0]
        safe = int(exceed[0] // 8) + 1 if len(exceed) else N_REV
        print(f"\n结论: 星历偏离 > 1e4 km 出现于第 {safe} 圈 → "
              f"revs_per_group 建议 ≤ {max(1, safe - 1)}")
    finally:
        spice.unload_kernel(default_kernel_dir())


if __name__ == "__main__":
    main()
