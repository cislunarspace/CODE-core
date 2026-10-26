"""8 字形族与 Grebow 2006 表 3.13 的逐行初值对照（#821）。

文献：Daniel J. Grebow, Generating Periodic Orbits in the Circular
Restricted Three-Body Problem with Applications to Lunar South Pole
Coverage, M.S. Thesis, Purdue University, 2006, 表 3.13（Northern L2
Butterfly Family）。表给初值形式 q0 = (x0, 0, z0, 0, ydot0, 0) 与周期
T（无量纲），首行（x0 = 1.0118）是母 halo 分岔成员，对照按后十行成员。

复现陷阱（口径记录）：

- Grebow 使用 μ = 0.0121505856，库内 DE421 基准 μ = 0.012150585350562453；
  比较全部在无量纲量（DU、TU）上进行，两侧 μ 差异 2.5e-10 相对量级，
  对无量纲周期的影响远小于容差。
- 表值为四位小数文献值，舍入量级 5e-5；验收容差取 5e-3（z0、ydot0、T
  逐行），对应文献舍入并给动力学与参数化差异留两个量级余量。
- 族成员存储形态即最大 x 垂直 y=0 穿越点 (x0, 0, z0, 0, ydot0, 0)，
  与表列初值同形同号（北族 z0 大于 0、ydot0 小于 0）。行走网格步长
  0.005 DU 不逐行命中表列 x0，每行取最近成员为初猜、把 x0 钉到表列
  值后做固定 x0 的 2T 闭合精化，再比较 z0、ydot0、T。

用法（不要用会触发项目重建的 uv run）::

    .venv/bin/python scripts/compare_grebow_table_3_13.py

退出码 0 为十行全部达标，1 为存在超差行。
"""

from __future__ import annotations

import sys

import numpy as np

from e2m2e.algorithm.family.orbits.period_doubling import (
    _refine_child_seed,
    design_figure_eight_family,
)
from e2m2e.data.constants import EARTH_MOON_DISTANCE_KM

#: Grebow 2006 表 3.13 后十行成员：(x0, z0, ydot0, T)，全部无量纲。
GREBOW_TABLE_3_13 = (
    (1.0196, 0.1742, -0.0817, 2.7519),
    (1.0249, 0.1743, -0.0821, 2.7582),
    (1.0301, 0.1742, -0.0816, 2.7687),
    (1.0354, 0.1740, -0.0801, 2.7845),
    (1.0406, 0.1735, -0.0770, 2.8077),
    (1.0459, 0.1727, -0.0721, 2.8409),
    (1.0511, 0.1713, -0.0646, 2.8891),
    (1.0564, 0.1693, -0.0539, 2.9596),
    (1.0616, 0.1662, -0.0392, 3.0637),
    (1.0669, 0.1619, -0.0189, 3.2255),
)

#: 逐行验收容差（z0、ydot0、T 各自适用）。
TOLERANCE = 5e-3


def main() -> int:
    # 生成覆盖表列 x0 区间的北族成员链（L2、北族、步长 0.005）。
    from e2m2e.algorithm.dynamics import CR3BP_Dynamics
    from e2m2e.algorithm.family.orbits.walk import earth_moon_system

    dyn = CR3BP_Dynamics(earth_moon_system())
    lo = (GREBOW_TABLE_3_13[0][0] - 0.002) * EARTH_MOON_DISTANCE_KM
    hi = (GREBOW_TABLE_3_13[-1][0] + 0.002) * EARTH_MOON_DISTANCE_KM
    # n_orbits 需让链条走过末行 x0（步长 0.005，24 个成员到约 1.0697）。
    family = design_figure_eight_family(2, 1, lo, hi, n_orbits=24, dynamics=dyn)
    members = [
        (float(o.states[0, 0]), np.asarray(o.states[0], dtype=float), float(o.period))
        for o in family.family.orbits
    ]
    if not members:
        print(f"族生成失败：{family.status.name} {family.message}")
        return 1

    print("Grebow 2006 表 3.13 对照（北族，无量纲；Δ = 库内 − 文献）")
    print(f"{'x0':>8} {'字段':>6} {'文献':>10} {'库内':>12} {'Δ':>10} 判定")
    worst = {"z0": 0.0, "ydot0": 0.0, "T": 0.0}
    failed = 0
    for x0_lit, z0_lit, ydot0_lit, t_lit in GREBOW_TABLE_3_13:
        _, guess_state, guess_period = min(members, key=lambda m: abs(m[0] - x0_lit))
        guess = guess_state.copy()
        guess[0] = x0_lit  # x0 钉到表列值后固定 x0 精化
        refined = _refine_child_seed(dyn, guess, guess_period)
        if refined.status.value != "converged":
            print(f"{x0_lit:8.4f} 精化未收敛：{refined.status}")
            failed += 1
            continue
        state = np.asarray(refined.states[0], dtype=float)
        period = float(refined.periods[0])
        got = {"z0": float(state[2]), "ydot0": float(state[4]), "T": period}
        lit = {"z0": z0_lit, "ydot0": ydot0_lit, "T": t_lit}
        row_ok = True
        for field in ("z0", "ydot0", "T"):
            delta = got[field] - lit[field]
            worst[field] = max(worst[field], abs(delta))
            ok = abs(delta) <= TOLERANCE
            row_ok &= ok
            print(
                f"{x0_lit:8.4f} {field:>6} {lit[field]:10.4f} "
                f"{got[field]:12.6f} {delta:+10.2e} {'达标' if ok else '超差'}"
            )
        if not row_ok:
            failed += 1
    print(
        f"\n最大偏差：|Δz0|={worst['z0']:.2e} |Δydot0|={worst['ydot0']:.2e} "
        f"|ΔT|={worst['T']:.2e}（容差 {TOLERANCE:.0e}）"
    )
    if failed:
        print(f"结论：{failed} 行超差，对照未通过")
        return 1
    print("结论：十行全部达标，对照通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
