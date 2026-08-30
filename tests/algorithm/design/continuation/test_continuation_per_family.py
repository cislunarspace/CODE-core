"""每类家族的延拓测试（阶段 2）。

家族（dpo/halo/spo/lpo）各从标准种子起跑延拓，验证三点
（真行为，不算 Jacobi/stability 等物理不变量——那是 scenarios 的活）：

- 延拓链不断：全部步 ``correction_success``；
- 族参数随步单调连续变化（不断裂、不回头）；
- 不发散：每步状态有限、闭合误差远低于轨道量级（散度上界，非精度检查——
  精度由 correction/ 测试的 1e-6 把关；长弧 LPO 的全 6D 闭合地板约 1e-5）。

延拓方式按家族的对称性分流：

- dpo / halo：x 轴或 xz 平面对称，用 ``Continuation.natural_continuation``
  （固定步长、``max_orbits`` 限到步数）。
- spo / lpo：L4/L5 无 x 轴/xz 对称，必须用全周期闭合修正
  （``iterate_full_period_correction``）。当前 ``Continuation`` 类的
  ``natural_continuation`` 硬编码调用半周期的 ``iterate_correction``，
  与全周期闭合不兼容；故这两族用最小化的预测-修正循环（与源码
  ``cr3bp_orbits._correct_spo/_correct_lpo`` 同构）直接驱动。

预算缩参（ADR 0037）：

- axial_l1 延拓链超预算移出默认套件：axial 族周期随 vz0 急剧增长，
  减步/缩步长/放宽到筛查级容差均无法压回单测预算（实测 >120s）。
- lpo_l4 长弧（T≈21）全周期传播成本高，缩为 3 步 + 修正容差放宽到
  筛查级 1e-9（散度筛查判据 ``DIVERGENCE_LIMIT`` 对此有充分余量）。
- dpo/halo/spo 链维持 5 步与默认 1e-12 研究级容差不动。
"""

import numpy as np
import pytest

from e2m2e.algorithm.family.lpo_initial_guess import compute_lpo_initial_guess
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection
from tests.algorithm.design import seeds

pytestmark = pytest.mark.orchestration


#: 每族默认要走的延拓步数
N_STEPS = 5
#: lpo_l4 链的缩参步数：长弧（T≈21）全周期传播 × 多步迭代实测接近单测
#: 预算，减步 + 筛查级容差（1e-9）压回预算内；链不断/单调/不发散三点
#: 验证语义不变。
N_STEPS_SLOW = 3
#: 筛查级修正容差（相对研究级默认 1e-12 放宽，减 Newton 迭代次数）
SCREENING_TOLERANCE = 1e-9
#: 筛查级 Newton 迭代内积分容差。须与 ``SCREENING_TOLERANCE`` 配套：
#: 收敛判据低于 100×integration_rtol 时 Newton 会因积分噪声停滞
#: （盘整跑满 max_iterations，见 DifferentialCorrection.__init__ 注释），
#: tolerance=1e-9 要求 rtol ≤ 1e-11。
SCREENING_INTEGRATION_RTOL = 1e-11
#: 延拓链散度上界。区别于修正测试的 1e-6 精度检查：SPO/LPO 用 3 约束全周期
#: 修正器（不强制 dx 闭合），长弧 LPO（T≈21）成员的全 6D closure 地板约 1e-5；
#: 此处只判链不发散（远低于轨道量级 1 DU），精度由修正测试单独把关。
DIVERGENCE_LIMIT = 1e-3


def _make_corrector(dynamics, tolerance):
    """构造缩参修正器：筛查级收敛判据 + 配套迭代内积分容差。"""
    corrector = DifferentialCorrection(
        dynamics,
        integration_rtol=SCREENING_INTEGRATION_RTOL if tolerance is not None else None,
    )
    if tolerance is not None:
        corrector.tolerance = tolerance
    return corrector


def _natural_continuation_5(
    seed, dynamics, setup_corrector, param_name, step, n_steps=N_STEPS, tolerance=None
):
    """自然参数延拓恰好 ``n_steps`` 步（固定步长、``max_orbits`` 硬限）。

    ``setup_corrector(corrector)`` 把修正器配置成该族的对称性 setup；
    ``param_name`` 是族参数名（用于 Continuation 的参数索引推断）；
    ``tolerance`` 非空时放宽修正器收敛判据（缩参家族用筛查级 1e-9）。
    返回 ``n_steps`` 条新轨道（不含种子）。
    """
    corrector = _make_corrector(dynamics, tolerance)
    setup_corrector(corrector)
    continuation = Continuation(corrector=corrector, step=step)
    continuation.step_size_adaptation = False  # 固定步长，保证步数均匀
    continuation.max_orbits = n_steps + 1  # 种子 + 最多 n_steps 条新轨道
    if param_name is not None:
        continuation.continuation_parameter = param_name

    seed_param = float(seed.states[0, _param_index(param_name or _inferred_param(corrector))])
    # param_range 给一个宽上界，保证不因触界提前停；步数由 max_orbits 限到 n_steps。
    # natural_continuation 返回 ContinuationResult 结果契约，
    # 轨道族在 result.family。
    result = continuation.natural_continuation(
        seed,
        param_range=(seed_param, seed_param + 1.0),
        step_size=step,
        verbose=False,
    )
    new_orbits = [o for o in result.family.orbits if o.metadata.get("continuation_step", 0) != 0]
    return new_orbits


def _full_period_continuation_5(seed, dynamics, setup_at, step, n_steps=N_STEPS, tolerance=None):
    """全周期闭合延拓 ``n_steps`` 步（SPO/LPO 专用，预测-修正循环）。

    ``setup_at(corrector, x0)`` 在族参数 ``x0`` 处配置修正器。每步把上一条
    轨道的 x0 推进 ``step``，全周期闭合修正；``tolerance`` 非空时放宽修正器
    收敛判据。返回 ``n_steps`` 条新轨道。
    """
    from e2m2e.data.templates import ConvergenceState

    corrector = _make_corrector(dynamics, tolerance)
    current = seed.copy()
    new_orbits = []
    for _ in range(n_steps):
        guess = current.copy()
        guess.states[0, 0] = current.states[0, 0] + step
        setup_at(corrector, float(guess.states[0, 0]))
        result = corrector.iterate_full_period_correction(guess, verbose=False)
        assert result.status is ConvergenceState.CONVERGED, (
            f"全周期修正失败（{result.status}/{result.cause}）"
        )
        assert result.orbit is not None, "全周期修正未产出轨道"
        orbit = result.orbit
        new_orbits.append(orbit)
        current = orbit
    return new_orbits


def _param_index(name: str) -> int:
    return {"x0": 0, "y0": 1, "z0": 2, "vx0": 3, "vy0": 4, "vz0": 5}[name]


def _inferred_param(corrector: DifferentialCorrection) -> str:
    return next(iter(corrector.fixed_parameters))


# ---- 各族延拓运行器：返回 (新轨道列表, 族参数索引) ----


def run_dpo(seed, dynamics):
    def setup(c):
        c.setup_2D_symmetric_x_fixed_x0(seeds.DPO_X0)

    return _natural_continuation_5(seed, dynamics, setup, None, step=0.003), 0


def run_halo_l1(seed, dynamics):
    def setup(c):
        c.setup_halo_orbit_fixed_z0(seeds.HALO_SEED_Z0, 1)

    return _natural_continuation_5(seed, dynamics, setup, None, step=0.003), 2


def run_spo_l4(seed, dynamics):
    def setup_at(c, x0):
        c.setup_spo_fixed_x0(x0, seeds.SPO_SEED_POINT)

    return _full_period_continuation_5(seed, dynamics, setup_at, step=0.005), 0


def run_lpo_l4(seed, dynamics):
    # LPO 不在 7 类修正组合内，无 session 缓存代表轨道；在此用线性化长周期
    # 模态种子修正一次作为延拓起点（同 cr3bp_orbits._correct_lpo 的 guess=None 分支）。
    # 缩参（ADR 0037）：3 步 + 筛查级容差（种子修正同容差，仅作延拓起点），
    # 长弧（T≈21）全周期传播压回单测预算。
    state, period = compute_lpo_initial_guess(
        dynamics.system, seeds.SPO_SEED_POINT, seeds.SPO_SEED_AMPLITUDE_KM
    )
    seed_orbit = _make_seed(dynamics, state, period)
    corrector = _make_corrector(dynamics, SCREENING_TOLERANCE)
    corrector.setup_lpo_fixed_x0(float(state[0]), seeds.SPO_SEED_POINT)
    lpo_seed = corrector.iterate_full_period_correction(seed_orbit, verbose=False)

    def setup_at(c, x0):
        c.setup_lpo_fixed_x0(x0, seeds.SPO_SEED_POINT)

    assert lpo_seed.orbit is not None, "LPO 种子修正未产出轨道"
    return (
        _full_period_continuation_5(
            lpo_seed.orbit,
            dynamics,
            setup_at,
            step=-0.01,
            n_steps=N_STEPS_SLOW,
            tolerance=SCREENING_TOLERANCE,
        ),
        0,
    )


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
# axial_l1 链超预算移出默认套件，见模块 docstring。
CONTINUATION_CASES = [
    ("dpo", "corrected_dpo", run_dpo),
    ("halo_l1", "corrected_halo_l1", run_halo_l1),
    ("spo_l4", "corrected_triangular_l4", run_spo_l4),
    ("lpo_l4", "corrected_triangular_l4", run_lpo_l4),
]

#: 每族期望的延拓步数（与各 runner 的 n_steps 保持一致）
FAMILY_N_STEPS = {
    "dpo": N_STEPS,
    "halo_l1": N_STEPS,
    "spo_l4": N_STEPS,
    "lpo_l4": N_STEPS_SLOW,
}


@pytest.mark.parametrize("family_id, seed_fixture, runner", CONTINUATION_CASES)
def test_continuation_chain(family_id, seed_fixture, runner, request, earth_moon_dynamics):
    """每族延拓：链不断、族参数单调、不发散。"""
    seed = request.getfixturevalue(seed_fixture)[0]
    new_orbits, param_idx = runner(seed, earth_moon_dynamics)
    expected_steps = FAMILY_N_STEPS[family_id]

    # 1) 延拓链不断：恰好期望步数且全部成功（修正失败已在运行器内中断）
    assert len(new_orbits) == expected_steps, (
        f"{family_id}: 延拓产出 {len(new_orbits)} 步（期望 {expected_steps}），链断裂"
    )
    for i, orbit in enumerate(new_orbits):
        assert orbit is not None, f"{family_id} 第 {i + 1} 步返回 None"

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
