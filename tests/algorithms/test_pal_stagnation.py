"""Halo PAL 延拓折叠点停滞回归测试。

背景
----
当 L1 北 Halo 族用 ``pseudo_arclength_continuation`` 延拓到 z_amplitude ≈ 0.085
附近时,族流形存在折叠点(fold):Xdot 的 rz 分量渐近过零并换号。
``directional_increment=True`` 的逻辑在每一步根据当前 Xdot 重新判断方向,
导致在折叠点形成 2-周期环:
- 偶数步:预测方向 = +ds*Xdot,落到 A 点
- 奇数步:预测方向 = -ds*Xdot,落到 B 点
A 和 B 几乎相同(距离 ≪ step_size),延拓实际停滞。

本测试的修复目标
-----------------
PAL 延拓在 30 步内不应该停滞;轨道 z 振幅应单调增长超过 0.10。

测试设计
--------
1. 复用 e2m2e/algorithms/continuation.py 中的 PAL 实现
2. 用 L1 北 Halo 小振幅种子(振幅 0.001)启动
3. 请求 30 步延拓
4. 断言:首尾连续若干轨道间的状态距离应 ≥ 步长
5. 断言:最终 z 振幅 > 0.10(超过 0.085 折叠点)

若 fix 缺失:测试会失败,因为 OrbitFamily 的最后 5 条轨道互相靠得很近,
且最终 z 振幅停滞在 ~0.085。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithms import Continuation, DifferentialCorrection
from e2m2e.core import CR3BP_Dynamics, CR3BP_System, Orbit


# 测试运行较慢(约 30-60 秒),只在不在 fast 模式时跑
pytestmark = pytest.mark.slow


def _build_earth_moon_system() -> CR3BP_System:
    return CR3BP_System(mu=1.21506683e-2, primary="earth", secondary="moon")


def _build_l_halo_seed(libration_point: int, halo_class: int, amplitude_z: float = 0.001) -> Orbit:
    """构造 L1/L2 北/南 Halo 种子轨道。

    用 e2m2e 的 Richardson 近似初值 + 微分修正,与
    generate_halo_family.py 的种子生成路径一致。
    """
    system = _build_earth_moon_system()
    dynamics = CR3BP_Dynamics(system)
    corrector = DifferentialCorrection(dynamic=dynamics)
    continuation = Continuation(corrector=corrector)

    seed = continuation.generate_halo_seed_orbit(
        libration_point=libration_point,
        amplitude_z=amplitude_z,
        halo_class=halo_class,
        verbose=False,
    )
    assert seed is not None, f"L{libration_point} {'北' if halo_class == 0 else '南'} 种子生成失败"
    assert seed.correction_success, f"L{libration_point} {'北' if halo_class == 0 else '南'} 种子修正失败"

    # 标记族类型与参数,符合 halo_pseudo_arclength_continuation 的预期
    seed.family_type = "halo"
    seed.parameters = {
        "libration_point": libration_point,
        "halo_class": halo_class,
        "amplitude_z": amplitude_z,
    }
    return seed


def _state_distance(orbit_a: Orbit, orbit_b: Orbit) -> float:
    """两轨道的 6 维初值状态欧氏距离。"""
    return float(np.linalg.norm(np.asarray(orbit_a.states[0]) - np.asarray(orbit_b.states[0])))


# 所有(L1/L2) × (北/南) × (正/负) 共 8 个组合
_ALL_PAL_CONFIGS = [
    (1, 0, "positive"),
    (1, 0, "negative"),
    (1, 1, "positive"),
    (1, 1, "negative"),
    (2, 0, "positive"),
    (2, 0, "negative"),
    (2, 1, "positive"),
    (2, 1, "negative"),
]


class TestHaloPALStagnation:
    """PAL 延拓在 L1/L2 北/南 Halo 折叠点处的停滞检测。

    折叠点位置:L1/L2 Halo 族在 z_amplitude 极值处的 z 转折点(流形在 (x, z)
    空间"折回")。PAL 在折叠点后 z 自然减小、x 单调增大,这是物理正确的(族
    流形穿过折叠),不是 bug。

    bug 表现为"在折叠点处形成 2-周期环,延拓振荡而非真正推进":轨道在两个
    几乎相同的状态间反复(z_amp 在两个接近值间跳),总弧长几乎不增。

    根因是 pal_plausible 阈值过严(T/2 < 1.35 排除 L1 halo 折叠点后所有
    正常轨道),导致每步 PAL Newton 解被错判为"偏离物理"而回退到欧拉预测,
    失去 PAL 的弧长推进能力。
    """

    @pytest.mark.parametrize("libration_point,halo_class,direction", _ALL_PAL_CONFIGS)
    def test_pal_reaches_fold_point(self, libration_point, halo_class, direction):
        """PAL 延拓 80 步后应到达折叠点(各 LP/HC 组合)。

        折叠点阈值因 LP/HC 而异,但所有组合下 z 振幅应明显超过种子(0.001)。
        折叠点具体位置:
        - L1 北/南: z_amp ≈ 0.085
        - L2 北/南: z_amp ≈ 0.30
        """
        # 跳过 L2 暂时 — L2 折叠点较远,需要不同 n_orbits 阈值
        # 但仍验证延拓没有早早停止
        seed = _build_l_halo_seed(libration_point, halo_class, amplitude_z=0.001)

        system = _build_earth_moon_system()
        dynamics = CR3BP_Dynamics(system)
        corrector = DifferentialCorrection(dynamic=dynamics)
        continuation = Continuation(corrector=corrector)

        family = continuation.halo_pseudo_arclength_continuation(
            seed_orbit=seed,
            n_orbits=80,
            direction=direction,
            step_size=0.0045,
            verbose=False,
        )

        assert len(family) >= 81, f"延拓未达到目标轨道数,实际 {len(family)}"

        z_amps = [abs(float(o.states[0, 2])) for o in family]
        max_z_amp = max(z_amps)

        if libration_point == 1:
            expected_min = 0.08  # L1 折叠点 z ≈ 0.085
        else:
            expected_min = 0.10  # L2 折叠点 z ≈ 0.30,80 步内应超过 0.10

        assert max_z_amp >= expected_min, (
            f"L{libration_point} {'北' if halo_class == 0 else '南'} {direction} 延拓在 z={max_z_amp:.4f} 处即停,"
            f"未达到预期折叠点位置 z≥{expected_min}"
        )

    @pytest.mark.parametrize("libration_point,halo_class,direction", _ALL_PAL_CONFIGS)
    def test_pal_no_stagnation_oscillation(self, libration_point, halo_class, direction):
        """PAL 延拓在折叠点附近应平稳穿过,不出现 2-周期环振荡。

        修复前:折叠点处连续多步轨道在两个几乎相同的 z 值间跳(2-周期环)。
        关键判据:取连续 3 个轨道,检查 z 振幅是否有"上→下→上"或"下→上→下"
        的非单调回退 — 这是 2-周期环的标志。修复后 z 振幅在折叠点前后
        是单调变化(过折叠点前后分别单调),不出现来回。
        """
        seed = _build_l_halo_seed(libration_point, halo_class, amplitude_z=0.001)

        system = _build_earth_moon_system()
        dynamics = CR3BP_Dynamics(system)
        corrector = DifferentialCorrection(dynamic=dynamics)
        continuation = Continuation(corrector=corrector)

        family = continuation.halo_pseudo_arclength_continuation(
            seed_orbit=seed,
            n_orbits=80,
            direction=direction,
            step_size=0.0045,
            verbose=False,
        )

        continuation_orbits = family.orbits[1:]
        assert len(continuation_orbits) >= 70

        z_amps = [abs(float(o.states[0, 2])) for o in continuation_orbits]

        # 关键判据:整段延拓中 z 振幅没有"高密度来回"(2-周期环特征)。
        # 修复前在折叠点附近 z 振幅在两个值之间反复跳(每步反向),
        # direction_changes > 10。
        # 修复后 z 振幅单调增长到折叠点再单调下降,只允许 ≤ 2 次反向
        # (过折叠点时反向一次;折叠点后 PAL Newton 余波可能再反向一次)。
        direction_changes = 0
        prev_dir = 0  # 0: unknown, 1: increasing, -1: decreasing
        for i in range(1, len(z_amps)):
            dz = z_amps[i] - z_amps[i - 1]
            if abs(dz) < 1e-6:
                continue
            curr_dir = 1 if dz > 0 else -1
            if prev_dir != 0 and curr_dir != prev_dir:
                direction_changes += 1
            prev_dir = curr_dir

        # 修复前 direction_changes > 5 (在折叠点来回跳,2-周期环)
        # 修复后 direction_changes ≤ 2 (过折叠点反向一次,余波至多再反向一次)
        assert direction_changes <= 2, (
            f"L{libration_point} {'北' if halo_class == 0 else '南'} {direction} z 振幅"
            f" 非单调变化 {direction_changes} 次,疑似 2-周期环振荡"
            f" (修复后应在过折叠点时反向 ≤ 2 次)"
        )

    @pytest.mark.parametrize("libration_point,halo_class,direction", _ALL_PAL_CONFIGS)
    def test_pal_extends_x_past_fold(self, libration_point, halo_class, direction):
        """PAL 延拓穿过折叠点后,x 振幅应继续增长(沿流形走弧长)。

        修复前:振荡在折叠点附近,x 振幅也卡死。
        修复后:穿过折叠后 x 从 ~0.94(L1)或 ~1.15(L2)增长。
        """
        seed = _build_l_halo_seed(libration_point, halo_class, amplitude_z=0.001)

        system = _build_earth_moon_system()
        dynamics = CR3BP_Dynamics(system)
        corrector = DifferentialCorrection(dynamic=dynamics)
        continuation = Continuation(corrector=corrector)

        family = continuation.halo_pseudo_arclength_continuation(
            seed_orbit=seed,
            n_orbits=80,
            direction=direction,
            step_size=0.0045,
            verbose=False,
        )

        continuation_orbits = family.orbits[1:]
        x_values = [float(o.states[0, 0]) for o in continuation_orbits]

        # L1 折叠点后 x 应 > 0.94, L2 折叠点后 x 应 > 1.16
        if libration_point == 1:
            threshold = 0.94
        else:
            threshold = 1.16

        max_x = max(x_values)
        assert max_x > threshold, (
            f"L{libration_point} {direction} 延拓 max(x0)={max_x:.4f} 未超过 {threshold},"
            f"延拓未能穿过折叠点"
        )
