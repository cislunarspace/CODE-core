"""二体 Lambert 求解器（Izzo 算法）的 Python 薄封装。

Rust 内核在 ``e2m2e-propagation`` crate（``crates/e2m2e-propagation/src/lambert.rs``），
经 ``e2m2e._integrators`` 暴露；本模块只做类型转换与结果封装。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from e2m2e.integrators import lambert_batch_py, lambert_izzo_py


@dataclass(frozen=True)
class LambertSolution:
    """单次 Lambert 求解结果。

    Attributes:
        v0: 出发速度，形状 ``(3,)``，km/s
        vf: 到达速度，形状 ``(3,)``，km/s
        converged: 是否收敛（不收敛时 Rust 侧直接抛错，故恒为 True）
        n_iter: Householder 迭代次数
        revs: 完整圈数
    """

    v0: np.ndarray
    vf: np.ndarray
    converged: bool
    n_iter: int
    revs: int


def _parse_direction(direction: str) -> bool:
    """把方向字符串转为 Rust 侧的 long_way 标志。"""
    if direction == "short":
        return False
    if direction == "long":
        return True
    raise ValueError(f"direction 必须是 'short' 或 'long'，得到 {direction!r}")


def solve_lambert(
    r0: npt.ArrayLike,
    rf: npt.ArrayLike,
    tof: float,
    mu: float,
    direction: str = "short",
    revs: int = 0,
) -> LambertSolution:
    """解二体 Lambert 问题（Izzo 2015 算法）。

    Args:
        r0: 出发位置 ``[x, y, z]``，km
        rf: 到达位置 ``[x, y, z]``，km
        tof: 飞行时间，s
        mu: 中心天体 GM，km³/s²
        direction: ``"short"`` 表示转移角 < π，``"long"`` 表示 > π
        revs: 完整圈数；≥ 1 时返回右分支（低能）解

    Returns:
        :class:`LambertSolution`

    Raises:
        ValueError: tof 低于该圈数最小转移时间或迭代不收敛
    """
    r0_arr = np.asarray(r0, dtype=float)
    rf_arr = np.asarray(rf, dtype=float)
    if r0_arr.shape != (3,) or rf_arr.shape != (3,):
        raise ValueError(f"r0/rf 必须是长度 3 的向量，得到 {r0_arr.shape} 与 {rf_arr.shape}")
    result = lambert_izzo_py(
        r0_arr.tolist(),
        rf_arr.tolist(),
        float(tof),
        float(mu),
        _parse_direction(direction),
        int(revs),
    )
    return LambertSolution(
        v0=np.asarray(result["v0"], dtype=float),
        vf=np.asarray(result["vf"], dtype=float),
        converged=True,
        n_iter=int(result["n_iter"]),
        revs=int(revs),
    )


def solve_lambert_batch(
    r0_list: npt.ArrayLike,
    rf_list: npt.ArrayLike,
    tof_grid: npt.ArrayLike,
    mu: float,
    direction: str = "short",
    revs: int = 0,
) -> np.ndarray:
    """N×M 网格批量求解：每个 (r0, rf) 几何对每个 tof 各解一次。

    Args:
        r0_list: 出发位置列表，形状 ``(N, 3)``，km
        rf_list: 到达位置列表，形状 ``(N, 3)``，km
        tof_grid: 飞行时间网格，形状 ``(M,)``，s
        mu: 中心天体 GM，km³/s²
        direction: ``"short"`` 或 ``"long"``
        revs: 完整圈数

    Returns:
        形状 ``(N, M, 2, 3)`` 的数组：``[..., 0, :]`` 为出发速度 v0，
        ``[..., 1, :]`` 为到达速度 vf；无解的组合为 NaN。
    """
    r0_arr = np.atleast_2d(np.asarray(r0_list, dtype=float))
    rf_arr = np.atleast_2d(np.asarray(rf_list, dtype=float))
    tofs = np.atleast_1d(np.asarray(tof_grid, dtype=float))
    if r0_arr.shape != rf_arr.shape or r0_arr.shape[1] != 3:
        raise ValueError(
            f"r0_list/rf_list 形状必须均为 (N, 3)，得到 {r0_arr.shape} 与 {rf_arr.shape}"
        )

    geometries = np.concatenate([r0_arr, rf_arr], axis=1).tolist()
    results = lambert_batch_py(
        geometries,
        tofs.tolist(),
        float(mu),
        _parse_direction(direction),
        int(revs),
    )

    n, m = r0_arr.shape[0], tofs.shape[0]
    out = np.full((n, m, 2, 3), np.nan)
    for i in range(n):
        for j in range(m):
            res = results[i * m + j]
            if res is not None:
                out[i, j, 0, :] = res["v0"]
                out[i, j, 1, :] = res["vf"]
    return out

