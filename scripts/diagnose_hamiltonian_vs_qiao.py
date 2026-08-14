"""与 qiao ``L1_EM_Hamilton.mat`` 的逐项比对（开发期交叉参考，手工运行）。

承接自 ``tests/algorithm/normal_form/test_hamiltonian.py`` 迁出的对拍逻辑
（ADR 0025 决策 1）：外部软件输出不进 pytest 正确性门，仅作本地诊断用。
正确性断言由定义级测试承担（``test_hamiltonian_constant_term_matches_
point_mass_definition`` 等）。

用法：

    uv run --extra normal-form python scripts/diagnose_hamiltonian_vs_qiao.py \
        /path/to/L1_EM_Hamilton.mat

环境要求：``kernels/``（或 ``$SPICE_KERNEL_DIR``）内有 ``.tls`` 闰秒内核与
``.bsp`` 星历内核（``make setup`` 拉取）；安装 ``normal-form`` extra（sympy）。

退出码：0 = order=4 内共有项匹配率达标；1 = 失配超阈或环境/文件缺失。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qiao_mat", help="qiao L1_EM_Hamilton.mat 的路径")
    parser.add_argument("--max-degree", type=int, default=4, help="本侧截断阶数（默认 4）")
    args = parser.parse_args()

    if not os.path.isfile(args.qiao_mat):
        print(f"qiao fixture 不存在：{args.qiao_mat}")
        return 1

    import scipy.io as sio

    from e2m2e.algorithm.dynamics import CR3BP_System, LibrationPoint
    from e2m2e.algorithm.normal_form import NormalFormContext
    from e2m2e.algorithm.normal_form.constants import JD0_J2000
    from e2m2e.algorithm.normal_form.hamiltonian import build_hamiltonian, evaluate_hamiltonian
    from e2m2e.algorithm.normal_form.legendre import expand_legendre_1_over_r
    from e2m2e.data.constants import Datum
    from e2m2e.data.kernels.manager import SPICEManager

    kernel_dir = os.environ.get("SPICE_KERNEL_DIR", "kernels")
    spice = SPICEManager()
    loaded = []
    for name in sorted(os.listdir(kernel_dir)):
        if name.endswith((".tls", ".bsp")):
            spice.load_kernel(os.path.join(kernel_dir, name))
            loaded.append(name)
    if not loaded:
        print(f"未加载到任何 SPICE 内核（目录：{kernel_dir}）")
        return 1
    print(f"已加载内核：{', '.join(loaded)}")

    system = CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")
    system = system._with_default_scales()
    context = NormalFormContext(
        system=system,
        libration_point=LibrationPoint.L1,
        epoch=JD0_J2000,
        order=args.max_degree,
    )
    legendre = expand_legendre_1_over_r(args.max_degree)

    qiao_h_poly = sio.loadmat(args.qiao_mat, squeeze_me=False)["H_poly"]
    qiao_lookup: dict[tuple[int, ...], float] = {}
    for i in range(qiao_h_poly.shape[0]):
        pv = qiao_h_poly[i, 0].ravel().astype(int)
        col = qiao_h_poly[i, 1].ravel()
        qiao_lookup[tuple(int(x) for x in pv)] = float(col[0])

    h = build_hamiltonian(context, legendre, max_degree=args.max_degree)
    evaled = evaluate_hamiltonian(h, np.array([0.0]), context)
    our_lookup = {
        tuple(int(x) for x in evaled.powers[j]): float(evaled.coefficients[0, j])
        for j in range(evaled.n_terms)
    }

    n_match = 0
    n_total = 0
    mismatches: list[tuple[tuple[int, ...], float, float]] = []
    for key, ref in qiao_lookup.items():
        ours = our_lookup.get(key)
        if ours is None:
            continue  # qiao N=15 的高阶项不在本侧截断内
        n_total += 1
        if abs(ref) < 1e-12 and abs(ours) < 1e-12:
            n_match += 1
            continue
        tol = max(1e-7, abs(ref) * 1e-6)
        if abs(ours - ref) < tol:
            n_match += 1
        else:
            mismatches.append((key, ref, ours))

    print(
        f"qiao 总项数 {len(qiao_lookup)}；本侧 order={args.max_degree} "
        f"共有项 {n_total}，匹配 {n_match}"
    )
    for key, ref, ours in mismatches[:10]:
        print(f"  失配 {key}: qiao={ref!r} ours={ours!r}")

    if n_total < 30:
        print(f"本侧共有项数过少：{n_total}（期望 ≥30）")
        return 1
    if n_match < n_total - 5:
        print(f"匹配率不达标：{n_match}/{n_total}")
        return 1
    print("匹配率达标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
