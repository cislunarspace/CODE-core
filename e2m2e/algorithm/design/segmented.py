"""分段打靶（segmented）修正：拼接点采样与 Rust 分段打靶封装。

采样簇（``_patch_sampling_for``/``_points_per_rev_for``/
``_sample_patch_points``/``_sample_patch_points_from_trajectory``）把
CR3BP 周期解铺成多圈 patch tile；``_design_apolune_segmented`` 把整条
tile 按圈切段、逐段多重打靶转星历并分层合并（朱彦伟 2026，Rust 实现）。
段长策略见 ``_revs_per_group_for``。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...data.types.orbit import Orbit
from ...status import ConvergenceState, FailureCause
from ..dynamics import CR3BP_Dynamics
from ..solver.multiple_shooting import (
    sample_patch_points_drop_near_perilune,
    sample_patch_points_perilune_clustered,
)
from .tuning import (
    _DPO_POINTS_PER_REV,
    _PATCH_SAMPLING_DROP_NEAR_PERILUNE,
    _PATCH_SAMPLING_PERILUNE_CLUSTERED,
    _PATCH_SAMPLING_UNIFORM,
    _POINTS_PER_REV,
    CORRECTION_TOL_KM,
    CORRECTION_VEL_WEIGHT,
)


def _dense_orbit(
    dynamics: CR3BP_Dynamics, state0: np.ndarray, period: float, n_points: int = 720
) -> Orbit:
    """从 ``state0`` 传播一个周期的稠密轨道（供 patch points 采样）。"""
    t_eval = np.linspace(0.0, period, n_points + 1)
    result = dynamics.propagate(state0, (0.0, period), t_eval=t_eval)
    orbit = Orbit(states=result["states"], times=result["time"], system=dynamics.system)
    orbit.period = period
    return orbit


def _patch_sampling_for(orbit_type: str) -> str:
    """按轨道族选择拼接点采样策略（内部策略，不进请求契约）。

    NRHO 与 Halo 解耦：NRHO 默认等时间；Halo 近月点加密。
    DPO 使用等时间采样，并由 :func:`_points_per_rev_for` 提高节点密度。
    删近月点采样保留在 ``_sample_patch_points`` 分派中供对照，不作生产默认。
    """
    if orbit_type == "HALO":
        return _PATCH_SAMPLING_PERILUNE_CLUSTERED
    return _PATCH_SAMPLING_UNIFORM


def _points_per_rev_for(orbit_type: str) -> int:
    """返回轨道族的每圈拼接节点数。"""
    if orbit_type == "DPO":
        return _DPO_POINTS_PER_REV
    return _POINTS_PER_REV


def _sample_patch_points(
    dynamics: CR3BP_Dynamics,
    state0: np.ndarray,
    period: float,
    n_revolutions: int,
    *,
    sampling: str = _PATCH_SAMPLING_UNIFORM,
    points_per_rev: int = _POINTS_PER_REV,
) -> tuple[np.ndarray, np.ndarray]:
    """从历元状态出发，在 ``n_revolutions`` 圈上采样 patch points（synodic）。

    默认每圈 ``points_per_rev`` 个等时间点。``sampling`` 覆盖族相关策略：

    - ``perilune_clustered``：近月点加密（Halo）
    - ``drop_near_perilune``：删近月点附近节点（对照/研究，非生产默认）
    - ``uniform``：等时间（NRHO 与其余族默认）
    """
    dense = _dense_orbit(dynamics, state0, period)
    if sampling == _PATCH_SAMPLING_PERILUNE_CLUSTERED:
        t_rel, states = sample_patch_points_perilune_clustered(dense, dynamics)
    elif sampling == _PATCH_SAMPLING_DROP_NEAR_PERILUNE:
        t_rel, states = sample_patch_points_drop_near_perilune(
            dense, dynamics, n_points=points_per_rev
        )
    else:
        t_rel = np.linspace(0.0, period, points_per_rev, endpoint=False)
        states = np.empty((len(t_rel), 6))
        for i in range(6):
            states[:, i] = np.interp(t_rel, dense.times, dense.states[:, i])

    t_patch = np.concatenate([t_rel + k * period for k in range(n_revolutions)])
    state_patch = np.tile(states, (n_revolutions, 1))
    return t_patch, state_patch


def _sample_patch_points_from_trajectory(
    orbit: Orbit, period: float, n_revolutions: int
) -> tuple[np.ndarray, np.ndarray]:
    """从轨道自带稠密轨迹插值 patch points（准周期 Lissajous 用）。

    准周期 Lissajous 不能用 :func:`_sample_patch_points`——它原生 CR3BP
    重传播 ``states[0]``，会重新激发不稳定方向而发散。
    改从 ``orbit`` 的中心流形有界轨迹跨 ``n_revolutions`` 圈均匀采样
    ``_POINTS_PER_REV`` 点/圈、逐分量线性插值。调用方须保证
    ``orbit.times`` 覆盖 ``[0, n_revolutions·period]``。
    """
    n_points = _POINTS_PER_REV * n_revolutions
    t_patch = np.linspace(0.0, n_revolutions * period, n_points, endpoint=False)
    states = np.empty((n_points, 6))
    for i in range(6):
        states[:, i] = np.interp(t_patch, orbit.times, orbit.states[:, i])
    return t_patch, states


def _revs_per_group_for(sel: str, n_rev: int) -> int:
    """第 1 步段长（每组圈数）。Halo 与稳定轨道用多圈/段（上限 3）：
    长段节点密、段内约束强，各段修到正确星历弧（对齐朱彦伟 2026）。
    NRHO 单独 1 圈/段：默认相位 0.5、约 1 个月弧上
    revs_per_group=3 合并层残差可卡在约 10² km；1 圈/段与等时间
    采样组合下 GUI 默认量级收敛。DPO 的一个周期约 23 天，使用 64
    点/圈时两圈同组可避免逐圈独立修正后的 seam 残差。
    配合 var_time 固定时刻族（``_FIXED_TIME_ORBIT_TYPES``，含
    Halo/NRHO/DPO、拟周期族与 Axial）。

    对照论文方案：每段 9 圈（对应
    972 圈/15 年量级的 12→3→3 层级拼接），合并层节点稀疏化为每圈 1 个
    远月点。Halo 3 圈/段 + 全节点合并已覆盖 180 天站保基准；年量级
    若段数过多、全节点合并矩阵病态放大，再评估 9 圈/段与远月点稀疏化。
    """
    if sel == "NRHO":
        return 1
    elif sel == "DPO":
        return min(n_rev, 2)
    elif sel == "HALO":
        return min(n_rev, 3)
    else:
        return max(1, min(3, n_rev))


def _design_apolune_segmented(
    forces_py: list[Any],
    observer: str,
    t_patch_j2000: np.ndarray,
    state_patch_j2000: np.ndarray,
    revs_per_group: int,
    points_per_rev: int,
    *,
    max_iter: int = 50,
    tolerance: float = CORRECTION_TOL_KM,
    vel_weight: float = CORRECTION_VEL_WEIGHT,
    var_time: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """分段打靶星历转换（朱彦伟 2026 多重打靶拼接，Rust 实现）。

    将 CR3BP 周期解转换到星历模型：整条 CR3BP tile 按 ``revs_per_group``
    圈切段，每段独立多重打靶转星历；段数 >1 时分层两两合并（合并段全节点
    自由、最小范数更新，对齐文献）直至整条连续。

    计算全部下沉 Rust ``segmented_shooting_correct``：切段、第 1 步各段
    打靶（段间独立 rayon 并行）、分层合并（同层配对合并段 rayon 并行）。
    本函数仅做参数装配与结果归一。

    关键配置（实测 Halo/NRHO，对齐文献）：

    - **第 1 步段长**：Halo/NRHO 用多圈长段（调用方传 ``min(n_rev, 3)``）。
      长段节点密、段内约束强，单弧打靶即可收敛且各段不漂离真实动力学；
      1 圈短段各段独立修正后会漂走（seam 跳 ~1e5 km，合并层无法消除——
      STM 条件数分析见 Liu & Liu 2025 §3）。
    - **合并层**：合并段全节点自由（``fixed_node_mask=None``），LM 最小范数
      更新对齐文献。固定首末锚定会使合并层不收敛：各段独立打靶后
      seam 不连续，锚定把修正全压给内部节点（60 天合并层停在 7.5e-01 km）；
      去锚定后 60 天合并层收敛到 1.4e-03 km，180 天三层合并全程收敛
      （5.8e-03 / 5.7e-04 / 1.4e-02 km）。

      对照论文的合并层节点稀疏化（每圈仅 1 个远月点节点，约束更疏、矩阵
      更良态）：当前全节点合并 180 天 5 段 3 层已收敛到
      1.4e-02 km，说明当前圈数下全节点矩阵病态未显现，不跟进；年量级
      （50+ 圈）时矩阵规模与病态会放大，稀疏化留作该场景的前置评估。
    - **var_time**：Halo/NRHO 固定节点时刻（False，对齐杨洪伟 2015、
      刘刚 2017）；稳定轨道（DRO 等）保留自由时刻（True）吸收 CR3BP→星历
      的时间偏差。

    星历下 Halo/NRHO 的圈间漂移是标称轨道的固有准周期特征（星历非严格周期），
    由轨道保持（``algorithm.station_keeping``）处理，不在本转换范围内。

    Args:
        forces_py: 全摄动 Rust forces 序列（与长期预报同模型）。
        observer: 坐标系原点（"EARTH"）。
        t_patch_j2000 / state_patch_j2000: 整条 CR3BP tile（J2000, km/km/s）。
        revs_per_group: 第 1 步每组圈数。短期（≤ 该圈数）单段即收敛；
            长期分段后由分层合并拼接。
        points_per_rev: 每圈节点数（由采样 tile 推断；族策略下可能非整除 8）。
        var_time: 节点时刻是否作为自由变量。Halo/NRHO 传 False（见上）。

    Returns:
        ``(t_patch, state_patch, max_residual)`` （J2000），整条连续星历轨迹与
        全程各段/合并段的最大打靶残差（km）。
    """
    from e2m2e.integrators import segmented_shooting_correct_py

    from .design_orbit import DesignNotConvergedError

    try:
        result = segmented_shooting_correct_py(
            forces_py,
            observer,
            list(t_patch_j2000),
            [list(map(float, x)) for x in state_patch_j2000],
            revs_per_group=revs_per_group,
            per_rev=points_per_rev,
            var_time=var_time,
            max_iter_per_segment=max_iter,
            tolerance=tolerance,
            rtol=1e-10,
            vel_weight=vel_weight,
            verbose=verbose,
        )
    except RuntimeError as e:
        raise DesignNotConvergedError(
            f"分段打靶拼接积分失败: {e}",
            cause=FailureCause.INTEGRATION_FAILED,
        ) from e
    if result.status is not ConvergenceState.CONVERGED:
        raise DesignNotConvergedError(
            f"{result.message}（容差 {tolerance:.3e} km）",
            status=result.status,
            cause=result.cause,
        )
    return (
        np.asarray(result.t_patch, dtype=float),
        np.asarray(result.state_patch, dtype=float),
        float(result.max_residual),
    )
