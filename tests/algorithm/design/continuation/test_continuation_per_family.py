"""每类家族的延拓 5 步测试（阶段 2）。

5 个家族（dpo/halo/axial/spo/lpo）各从标准种子起跑 5 步延拓，验证三点
（真行为，不算 Jacobi/stability 等物理不变量——那是 scenarios 的活）：

- 延拓链不断：5 步全部 ``correction_success``；
- 族参数随步单调连续变化（不断裂、不回头）；
- 不发散：每步状态有限、闭合误差远低于轨道量级（散度上界，非精度检查——
  精度由 correction/ 测试的 1e-6 把关；长弧 LPO 的全 6D 闭合地板约 1e-5）。

延拓方式按家族的对称性分流：

- dpo / halo / axial：x 轴或 xz 平面对称，用 ``Continuation.natural_continuation``
  （固定步长、``max_orbits`` 限到 5）。axial 的族参数是 vz0，而 Continuation
  从 ``fixed_parameters`` 推断出的首个键是 z0，需把 ``continuation_parameter``
  覆写为 "vz0"。
- spo / lpo：L4/L5 无 x 轴/xz 对称，必须用全周期闭合修正
  （``iterate_full_period_correction``）。当前 ``Continuation`` 类的
  ``natural_continuation`` 硬编码调用半周期的 ``iterate_correction``，
  与全周期闭合不兼容；故这两族用最小化的预测-修正循环（与源码
  ``cr3bp_orbits._correct_spo/_correct_lpo`` 同构）直接驱动 5 步。
"""

import numpy as np
import pytest

from e2m2e.algorithm.family.lpo_initial_guess import compute_lpo_initial_guess
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from tests.algorithm.design import seeds

#: 每族要走的延拓步数
N_STEPS = 5
#: 延拓链散度上界。区别于修正测试的 1e-6 精度检查：SPO/LPO 用 3 约束全周期
#: 修正器（不强制 dx 闭合），长弧 LPO（T≈21）成员的全 6D closure 地板约 1e-5；
#: 此处只判链不发散（远低于轨道量级 1 DU），精度由修正测试单独把关。
DIVERGENCE_LIMIT = 1e-3


def _natural_continuation_5(seed, dynamics, setup_corrector, param_name, step):
    """自然参数延拓恰好 5 步（固定步长、``max_orbits`` 硬限）。

    ``setup_corrector(corrector)`` 把修正器配置成该族的对称性 setup；
    ``param_name`` 是族参数名（用于 Continuation 的参数索引推断）。
    返回 5 条新轨道（不含种子）。
    """
    corrector = DifferentialCorrection(dynamics)
    setup_corrector(corrector)
    continuation = Continuation(corrector=corrector, step=step)
    continuation.step_size_adaptation = False  # 固定步长，保证 5 步均匀
    continuation.max_orbits = N_STEPS + 1  # 种子 + 最多 5 条新轨道
    if param_name is not None:
        continuation.continuation_parameter = param_name

    seed_param = float(seed.states[0, _param_index(param_name or _inferred_param(corrector))])
    # param_range 给一个宽上界，保证不因触界提前停；步数由 max_orbits 限到 5。
    family = continuation.natural_continuation(
        seed,
        param_range=(seed_param, seed_param + 1.0),
        step_size=step,
        verbose=False,
    )
    new_orbits = [o for o in family.orbits if o.metadata.get("continuation_step", 0) != 0]
    return new_orbits


def _full_period_continuation_5(seed, dynamics, setup_at, step):
    """全周期闭合延拓 5 步（SPO/LPO 专用，预测-修正循环）。

    ``setup_at(corrector, x0)`` 在族参数 ``x0`` 处配置修正器。每步把上一条
    轨道的 x0 推进 ``step``，全周期闭合修正；返回 5 条新轨道。
    """
    corrector = DifferentialCorrection(dynamics)
    current = seed.copy()
    new_orbits = []
    for _ in range(N_STEPS):
        guess = current.copy()
        guess.states[0, 0] = current.states[0, 0] + step
        setup_at(corrector, float(guess.states[0, 0]))
        orbit = corrector.iterate_full_period_correction(guess, verbose=False)
        new_orbits.append(orbit)
        current = orbit
    return new_orbits


def _param_index(name: str) -> int:
    return {"x0": 0, "y0": 1, "z0": 2, "vx0": 3, "vy0": 4, "vz0": 5}[name]


def _inferred_param(corrector: DifferentialCorrection) -> str:
    return next(iter(corrector.fixed_parameters))


# ---- 各族延拓运行器：返回 (5 条新轨道, 族参数索引) ----


def run_dpo(seed, dynamics):
    def setup(c):
        c.setup_2D_symmetric_x_fixed_x0(seeds.DPO_X0)

    return _natural_continuation_5(seed, dynamics, setup, None, step=0.003), 0


def run_halo_l1(seed, dynamics):
    def setup(c):
        c.setup_halo_orbit_fixed_z0(seeds.HALO_SEED_Z0, 1)

    return _natural_continuation_5(seed, dynamics, setup, None, step=0.003), 2


def run_axial_l1(seed, dynamics):
    def setup(c):
        c.setup_axial_orbit_fixed_vz0(seeds.AXIAL_SEED_VZ0, seeds.AXIAL_SEED_POINT)

    # axial 的 fixed_parameters 首键是 z0，族参数实为 vz0，需覆写。
    return _natural_continuation_5(seed, dynamics, setup, "vz0", step=0.001), 5


def run_spo_l4(seed, dynamics):
    def setup_at(c, x0):
        c.setup_spo_fixed_x0(x0, seeds.SPO_SEED_POINT)

    return _full_period_continuation_5(seed, dynamics, setup_at, step=0.005), 0


def run_lpo_l4(seed, dynamics):
    # LPO 不在 7 类修正组合内，无 session 缓存代表轨道；在此用线性化长周期
    # 模态种子修正一次作为延拓起点（同 cr3bp_orbits._correct_lpo 的 guess=None 分支）。
    state, period = compute_lpo_initial_guess(
        dynamics.system, seeds.SPO_SEED_POINT, seeds.SPO_SEED_AMPLITUDE_KM
    )
    seed_orbit = _make_seed(dynamics, state, period)
    corrector = DifferentialCorrection(dynamics)
    corrector.setup_lpo_fixed_x0(float(state[0]), seeds.SPO_SEED_POINT)
    lpo_seed = corrector.iterate_full_period_correction(seed_orbit, verbose=False)

    def setup_at(c, x0):
        c.setup_lpo_fixed_x0(x0, seeds.SPO_SEED_POINT)

    return _full_period_continuation_5(lpo_seed, dynamics, setup_at, step=-0.01), 0


def _make_seed(dynamics, state, period):
    from e2m2e.data.types.orbit import Orbit

    orbit = Orbit(
        states=np.asarray(state, dtype=float).reshape(1, -1),
        times=np.array([0.0]),
        system=dynamics.system,
    )
    orbit.period = period
    return orbit


# ---- parametrize：family_id → (种子 fixture 名, 运行器) ----
CONTINUATION_CASES = [
    ("dpo", "corrected_dpo", run_dpo),
    ("halo_l1", "corrected_halo_l1", run_halo_l1),
    ("axial_l1", "corrected_axial_l1", run_axial_l1),
    ("spo_l4", "corrected_triangular_l4", run_spo_l4),
    ("lpo_l4", "corrected_triangular_l4", run_lpo_l4),
]


@pytest.mark.parametrize("family_id, seed_fixture, runner", CONTINUATION_CASES)
def test_continuation_chain(family_id, seed_fixture, runner, request, earth_moon_dynamics):
    """每族延拓 5 步：链不断、族参数单调、不发散。"""
    seed = request.getfixturevalue(seed_fixture)
    new_orbits, param_idx = runner(seed, earth_moon_dynamics)

    # 1) 延拓链不断：恰好 5 步且全部成功
    assert len(new_orbits) == N_STEPS, (
        f"{family_id}: 延拓产出 {len(new_orbits)} 步（期望 {N_STEPS}），链断裂"
    )
    for i, orbit in enumerate(new_orbits):
        assert orbit is not None, f"{family_id} 第 {i + 1} 步返回 None"
        assert orbit.correction_success, f"{family_id} 第 {i + 1} 步 correction_success 不为 True"

    # 2) 族参数随步单调连续（不断裂、不回头）
    params = np.array([float(o.states[0, param_idx]) for o in new_orbits])
    diffs = np.diff(params)
    assert np.all(diffs != 0.0), f"{family_id}: 族参数相邻步出现相等（断裂）: {params}"
    assert np.all(diffs > 0) or np.all(diffs < 0), f"{family_id}: 族参数不单调（回头）: {params}"

    # 3) 不发散：每步状态有限、闭合误差远低于轨道量级（散度上界，非精度检查）
    for i, orbit in enumerate(new_orbits):
        assert np.all(np.isfinite(orbit.states)), f"{family_id} 第 {i + 1} 步状态含非有限值"
        assert orbit.closure_error < DIVERGENCE_LIMIT, (
            f"{family_id} 第 {i + 1} 步 closure_error={orbit.closure_error:.3e} "
            f"超过散度上界 {DIVERGENCE_LIMIT}"
        )
