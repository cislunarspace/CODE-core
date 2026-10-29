"""长弧验证 2:1 地月 DRO 的 Floquet 模态与有界性判据（#832）。

默认 pytest 只做短弧与单周期验证（tests/algorithm/proximity/
test_floquet_modes.py），本脚本补多圈长弧：构造月心近侧穿越 −0.192 LU 的
2:1 DRO 复现成员，报告单值矩阵特征值与论文对照（结果性，人工判读），
再把中心模态初值与漂移初值各传播数十个周期，验证有界子空间内的相对
状态量级不变、漂移初值线性发散；另用单值矩阵幂把两对周期化模态外推到
数十圈，验证它们保持 T 周期（Φ(kT+τ) = M^k Φ(τ)）。

用法（不要用会触发项目重建的 uv run）::

    .venv/bin/python scripts/verify_floquet_modes_dro21.py
"""

from __future__ import annotations

import numpy as np

from e2m2e.algorithm.dynamics import CR3BP_Dynamics
from e2m2e.algorithm.family.orbits.dro import _correct_dro
from e2m2e.algorithm.family.orbits.walk import earth_moon_system
from e2m2e.algorithm.proximity import floquet_modes

#: 长弧圈数与初始扰动幅度（无量纲）。
REVOLUTIONS = 30
EPSILON = 1e-6

#: 论文对照值（杨驰航等，航空学报 2023 年 44 卷 5 期 326563，结果性）。
PAPER_IN_PLANE = complex(-0.7422, 0.6702)
PAPER_OUT_OF_PLANE = complex(0.1387, 0.9903)


def main() -> None:
    dynamics = CR3BP_Dynamics(earth_moon_system())
    seed = _correct_dro(dynamics, 0.79188556619742, None)
    orbit = _correct_dro(dynamics, (1.0 - dynamics.system.mu) - 0.192, seed)
    assert orbit.period is not None
    modes = floquet_modes(orbit, dynamics)

    print(f"周期 T = {modes.period:.9f} TU")
    print("特征值（论文对照为结果性，人工判读）：")
    print(f"  面内对 {modes.in_plane.eigenvalue:+.6f}  论文 {PAPER_IN_PLANE:+.4f}")
    print(f"  法向对 {modes.out_of_plane.eigenvalue:+.6f}  论文 {PAPER_OUT_OF_PLANE:+.4f}")

    period = modes.period
    samples = np.arange(1.0, REVOLUTIONS + 1) * period

    def relative_norms(initial: np.ndarray) -> np.ndarray:
        out = dynamics.propagate(modes.state0 + initial, (0.0, float(samples[-1])), t_eval=samples)
        states = np.asarray(out["states"], dtype=float)
        return np.linalg.norm(states - modes.state0, axis=1)

    bounded = relative_norms(EPSILON * modes.basis[0][:, 1])
    drift = relative_norms(EPSILON * modes.drift)
    print(f"\n{REVOLUTIONS} 圈长弧（ε = {EPSILON:.0e}）：")
    print(
        f"  中心模态初值：相对状态范数首末 {bounded[0]:.3e} / {bounded[-1]:.3e}，"
        f"全程最大 {bounded.max():.3e}"
    )
    print(f"  漂移初值    ：相对状态范数首末 {drift[0]:.3e} / {drift[-1]:.3e}")
    growth_slope = np.polyfit(samples / period, drift, 1)[0] / EPSILON
    print(f"  漂移增长斜率 {growth_slope:+.3f} ε/圈（线性增长判读）")

    bounded_ok = bounded.max() < 10.0 * EPSILON
    drift_ok = drift[-1] > 10.0 * drift[0]
    print(
        f"\n有界性判据：中心模态有界 {'通过' if bounded_ok else '失败'}，"
        f"漂移发散 {'通过' if drift_ok else '失败'}"
    )

    periodicization_long_arc(modes, dynamics)


def periodicization_long_arc(modes, dynamics: CR3BP_Dynamics) -> None:
    """把两对周期化模态外推到数十圈，验证保持 T 周期。

    线性化系统沿周期轨道是周期时变的，传播子按 Φ(kT+τ) = Φ(τ) M^k 组合，
    不能写成 M^k Φ(τ)。取轨道内相位点 τ，模态的旋转因子按 α(kT+τ)/T
    推进。若周期化构造正确，u(kT+τ) 与 u(τ) 相等，v 同理，偏差只来自
    M^k 与 Φ(τ) 的累积舍入。
    """
    tau = 0.37 * modes.period
    result = dynamics.propagate(modes.state0, (0.0, tau), t_eval=[tau], with_stm=True)
    phi_tau = np.asarray(result["stm"], dtype=float)[-1]
    print(f"\n周期化长弧（τ = 0.37T，{REVOLUTIONS} 圈）：")
    power = np.eye(6)
    powers = [power]
    for _ in range(REVOLUTIONS):
        power = modes.monodromy @ power
        powers.append(power)
    for label, pair in (("面内对", modes.in_plane), ("法向对", modes.out_of_plane)):
        theta = pair.alpha * tau / modes.period
        reference_u = np.cos(theta) * (phi_tau @ pair.a) + np.sin(theta) * (phi_tau @ pair.b)
        reference_v = -np.sin(theta) * (phi_tau @ pair.a) + np.cos(theta) * (phi_tau @ pair.b)
        deviation = 0.0
        for k in range(1, REVOLUTIONS + 1):
            theta_k = theta + pair.alpha * k
            extended_a = phi_tau @ (powers[k] @ pair.a)
            extended_b = phi_tau @ (powers[k] @ pair.b)
            u_k = np.cos(theta_k) * extended_a + np.sin(theta_k) * extended_b
            v_k = -np.sin(theta_k) * extended_a + np.cos(theta_k) * extended_b
            deviation = max(
                deviation,
                float(np.linalg.norm(u_k - reference_u)),
                float(np.linalg.norm(v_k - reference_v)),
            )
        print(f"  {label}：{REVOLUTIONS} 圈外推偏差最大 {deviation:.3e}")


if __name__ == "__main__":
    main()
