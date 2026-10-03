"""CR3BP 多项式流的一致半径-阶数曲线（issue #786；ADR 0059）。

对北族 L2 NRHO 种子做半周期弧的多项式流传播：逐阶（1..8）把终点
Taylor 映射编译为 CompiledDa，在 16 个固定伪随机 R⁶ 单位方向上比较
映射求值与 PD78 重积分的 ND 状态最大偏差，二分求每方向满足偏差
≤ TOL 的最大半径 ρ*，一致半径取各方向 ρ* 的最小值，落盘 CSV
``datasets/da_flow/uniform_radius_nrho_l2.csv`` 并打印表格。

一致性判据（ND 状态 max 范数，无量纲）——流多项式的常数项就是名义
状态，求值结果直接对拍重积分，不再另加名义项：
    error(ρ, dir) = max | compiled.eval(ρ·dir)
                            − propagate_cr3bp(SEED + ρ·dir).states[-1] |

半径区间 [1e-8, 1e-1]，二分 36 次（区间收敛到 ~1e-18 相对宽度）。
重积分容差 1e-13，远低于 TOL=1e-8，不构成瓶颈。

数据集随仓库留档（曲线数据不进默认 pytest；与
``datasets/catalog_baseline/`` 同级新增子目录）。阶数 8 在 debug 构建下
明显偏慢，建议先 ``make dev-release`` 再运行::

    .venv/bin/python scripts/da_uniform_radius_curve.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from e2m2e.integrators import (
    CompiledDa,
    da_init_py,
    propagate_cr3bp_da_py,
    propagate_cr3bp_py,
)

# 北族 L2 NRHO 折叠种子与 DE421 校准地月质量比，出处同
# crates/e2m2e-integrators/src/family_generation/nrho.rs（nrho.rs 的
# L2_SEED_STATE / L2_SEED_PERIOD / DE421_EARTH_MOON_MU）。
SEED = np.array([1.128103754424342, 0.0, 0.17883236940616654, 0.0, -0.22553424464298827, 0.0])
MU = 0.012150585350562453
PERIOD = 2.9899387540796956

ARC = 0.5 * PERIOD  # 半周期短弧
STEP = 1e-3  # RK4 固定步长
ORDERS = [1, 2, 3, 4, 5, 6, 8]
TOL = 1e-8  # 一致性判据（ND 状态 max 范数）
RHO_LO = 1e-8
RHO_HI = 1e-1
BISECTIONS = 36
N_DIRECTIONS = 16

OUTPUT = "datasets/da_flow/uniform_radius_nrho_l2.csv"


def re_integrated_final(deviation: np.ndarray) -> np.ndarray:
    """PD78 重积分（rtol=atol=1e-13）的终点状态。"""
    seed = SEED + deviation
    result = propagate_cr3bp_py(MU, (0.0, ARC), [ARC], seed.tolist(), 1e-13, 1e-13, None, None)
    return np.asarray(result["states"][-1])


def error_at(compiled: CompiledDa, rho: float, direction: np.ndarray) -> float:
    """多项式映射求值与重积分的 ND 状态最大偏差（流多项式已含名义常数项）。"""
    approx = np.asarray(compiled.eval((rho * direction).tolist()))
    return float(np.abs(approx - re_integrated_final(rho * direction)).max())


def uniform_radius_for_order(order: int) -> tuple[float, float]:
    """单阶一致半径与该半径处的最大偏差（各方向 ρ* 的最小值口径）。"""
    da_init_py(order, 6)
    flow = propagate_cr3bp_da_py(MU, (0.0, ARC), [ARC], SEED.tolist(), order, STEP)
    compiled = CompiledDa.from_das(flow["flows"][-1])

    rng = np.random.default_rng(786)
    directions = [d / np.linalg.norm(d) for d in rng.standard_normal((N_DIRECTIONS, 6))]

    radii = []
    for direction in directions:
        if error_at(compiled, RHO_HI, direction) <= TOL:
            radii.append(RHO_HI)  # 区间上界即满足
            continue
        lo, hi = RHO_LO, RHO_HI
        for _ in range(BISECTIONS):
            mid = 0.5 * (lo + hi)
            if error_at(compiled, mid, direction) <= TOL:
                lo = mid
            else:
                hi = mid
        radii.append(lo)
    radius = min(radii)
    max_err = max(error_at(compiled, radius, d) for d in directions)
    return radius, max_err


def main() -> None:
    rows = []
    for order in ORDERS:
        radius, max_err = uniform_radius_for_order(order)
        rows.append((order, radius, max_err))
        print(f"order={order:>2}  uniform_radius={radius:.6e}  max_err={max_err:.3e}")

    header = (
        f"# CR3BP DA 多项式流一致半径（issue #786；ADR 0059）\n"
        f"# seed=nrho L2 north fold ({', '.join(repr(float(v)) for v in SEED)})\n"
        f"# mu={MU!r}  period={PERIOD!r}  arc={ARC!r}  step={STEP!r}\n"
        f"# tol={TOL!r}  directions={N_DIRECTIONS} (rng=default_rng(786), 单位化)\n"
        f"# rho_interval=[{RHO_LO!r}, {RHO_HI!r}]  bisections={BISECTIONS}\n"
        f"# 生成命令：.venv/bin/python scripts/da_uniform_radius_curve.py\n"
    )
    out_path = Path(OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("order,uniform_radius,max_err_at_radius\n")
        for order, radius, max_err in rows:
            fh.write(f"{order},{radius!r},{max_err!r}\n")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
