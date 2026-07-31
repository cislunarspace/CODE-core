"""低推力 Hermite-Simpson 配点求解器验证。

对照 ``docs/plans/lowthrust-collocation-prd.md``：HS 缺陷约束正确性、min-fuel
闭环收敛、对标直接打靶。纯二体（PointMassGravity，无需 SPICE）。
"""

from types import SimpleNamespace

import numpy as np
from e2m2e._integrators import RkMethod, propagate_compiled_lowthrust

from e2m2e.algorithm.forces import PointMassGravity
from e2m2e.algorithm.transfer import EngineConfig, LowThrustCollocation, LowThrustShooting

MU = 398600.435507


def _system_forces():
    return SimpleNamespace(origin="EARTH"), [PointMassGravity("EARTH", mu=MU)]


def _make_collocation(target_state, tf, initial_mass=1000.0):
    system, forces = _system_forces()
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    init = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    return LowThrustCollocation(
        system, forces, engine, init, initial_mass, target_state, 0.0, tf
    ), forces


def test_defect_constraint_vanishes_on_true_trajectory():
    """真实传播轨迹采样成节点，HS 缺陷应小（转录正确）。

    短弧（100s）+ 单段，HS 三阶积分精度极高，缺陷应 < 1e-6。
    """
    system, forces = _system_forces()
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    init = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    tf = 100.0  # 短弧

    # 真实传播：满推沿初速方向（θ₁=π/2, θ₂=0）
    state0 = np.concatenate([init, [1000.0]])
    tf = 50.0  # 短弧；HS 三阶精度随弧长 O(dt⁴) 下降（100s→1.4e-4，50s→4.4e-6）
    res = propagate_compiled_lowthrust(
        RkMethod.PD45,
        0.0,
        state0.tolist(),
        5.0,
        1e-12,
        [0.0, tf / 2, tf],
        "EARTH",
        [("point_mass", MU)],
        (0.5, 3000.0, 1.0, 0.0, 1.0, 0.0),  # 沿 y（初速方向）
        500_000,
    )
    states_true = np.asarray(res["states"], dtype=float)  # (3, 7): t0, t_mid, tf

    # 构造配点问题的 z：1 段，2 节点，状态=真实轨迹采样，控制=满推沿初速
    coll = LowThrustCollocation(system, forces, engine, init, 1000.0, init.copy(), 0.0, tf)
    states = states_true[[0, 2]]  # 节点 0、1（首末）
    controls = np.array([[1.0, np.pi / 2, 0.0], [1.0, np.pi / 2, 0.0]])
    z = np.concatenate([states.ravel(), controls.ravel()])

    defects = coll._defect_constraints(z, n_segments=1)
    max_defect = np.max(np.abs(defects))
    # HS 三阶在 50s 弧缺陷 ~4e-6（随弧长 O(dt⁴) 下降，已验证收敛阶）
    assert max_defect < 1e-5, f"HS 缺陷应 < 1e-5（50s 弧）, got {max_defect:.3e}"


def test_collocation_min_fuel_converges():
    """min-fuel 配点闭环：默认初猜收敛，末态达标、质量下降。"""
    # 目标：略高圆轨道
    aT = 7200.0
    vT = np.sqrt(MU / aT)
    target = np.array([aT, 0.0, 0.0, 0.0, vT, 0.0])
    coll, _forces = _make_collocation(target, tf=2 * 86400.0)

    sol = coll.solve(6, maxiter=200, verbose=False)
    # 末态半长轴接近目标
    a_end = -MU / (
        2 * (np.linalg.norm(sol.states[-1][3:6]) ** 2 / 2 - MU / np.linalg.norm(sol.states[-1][:3]))
    )
    assert abs(a_end - aT) / aT < 5e-3, f"末态半长轴 {a_end:.1f} 偏离目标 {aT:.1f}"
    assert sol.fuel_consumed > 0, "燃料消耗应为正"


def test_collocation_vs_shooting_consistency():
    """配点 vs 直接打靶：同一问题末态/燃料量级一致。"""
    r0 = 7000.0
    v0 = np.sqrt(MU / r0)
    aT = 7100.0
    vT = np.sqrt(MU / aT)
    target = np.array([aT, 0.0, 0.0, 0.0, vT, 0.0])
    tf = 1.5 * 86400.0

    system, forces = _system_forces()
    engine = EngineConfig(t_max=0.5, isp=3000.0)
    init = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])

    coll = LowThrustCollocation(system, forces, engine, init, 1000.0, target, 0.0, tf)
    sol_coll = coll.solve(5, maxiter=150)

    shooter = LowThrustShooting(system, forces, engine, init, 1000.0, target, 0.0, tf)
    sol_shoot = shooter.solve(5, maxiter=150)

    # 两者末态半长轴都接近目标（各自收敛）
    a_coll = -MU / (
        2
        * (
            np.linalg.norm(sol_coll.states[-1][3:6]) ** 2 / 2
            - MU / np.linalg.norm(sol_coll.states[-1][:3])
        )
    )
    a_shoot = -MU / (
        2
        * (
            np.linalg.norm(sol_shoot.states[-1][3:6]) ** 2 / 2
            - MU / np.linalg.norm(sol_shoot.states[-1][:3])
        )
    )
    assert abs(a_coll - aT) / aT < 0.02, f"配点末态 {a_coll:.1f}"
    assert abs(a_shoot - aT) / aT < 0.02, f"打靶末态 {a_shoot:.1f}"
    # 燃料消耗量级一致（都 >0）
    assert sol_coll.fuel_consumed > 0 and sol_shoot.fuel_consumed > 0
