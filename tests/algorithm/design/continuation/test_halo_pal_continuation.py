"""Halo PAL 编排函数 halo_pseudo_arclength_continuation 冒烟测试。

#434 回归测试：#351 结果契约迁移后，编排层曾读取延拓结果的旧属性名
（``.orbits``，现契约下族在 ``ContinuationResult.family``），任何调用
必然 AttributeError——而套件对此零覆盖。此处以 session 缓存的 L1 Halo
种子直接驱动编排函数，小步数验证它返回带标签的 Halo 族。
"""

import pytest

from e2m2e.algorithm.family import halo_pseudo_arclength_continuation
from e2m2e.algorithm.solver.continuation import Continuation
from e2m2e.algorithm.solver.differential_correction import DifferentialCorrection

pytestmark = pytest.mark.orchestration

#: 冒烟步数：足以检验编排层结果读取与标签逻辑，又控制 STM 传播开销
N_NEW_ORBITS = 2


def test_pal_wrapper_returns_tagged_family(corrected_halo_l1, earth_moon_dynamics):
    """正向 PAL 编排返回 种子+N_NEW_ORBITS 条带 Halo 标签的族成员。"""
    seed, _result = corrected_halo_l1
    # 种子标签由调用方负责（生产调用方 _walk_pal_to_perilune 同样先打标签）
    seed.family_type = "halo"
    seed.parameters = {
        "libration_point": 1,
        "halo_class": 0,
        "amplitude_z": abs(float(seed.states[0, 2])),
    }
    continuation = Continuation(corrector=DifferentialCorrection(earth_moon_dynamics))
    family = halo_pseudo_arclength_continuation(
        continuation,
        seed_orbit=seed,
        n_orbits=N_NEW_ORBITS,
        direction="positive",
        step_size=0.0045,
        verbose=False,
    )

    # 统一族容器：种子 + N_NEW_ORBITS 条新成员
    assert len(family) == 1 + N_NEW_ORBITS

    # 成员带 Halo 族标签与参数
    for orbit in family.orbits:
        assert orbit.family_type == "halo"
        assert orbit.parameters["libration_point"] == 1
        assert orbit.parameters["halo_class"] == 0

    # 正向延拓（北族）：面外振幅单调增大
    amplitudes = [o.parameters["amplitude_z"] for o in family.orbits]
    assert amplitudes[1] > amplitudes[0]
    assert amplitudes[2] > amplitudes[1]

    # 每条新成员都是修正收敛产物（闭合误差已记录且远低于轨道量级）
    for orbit in family.orbits[1:]:
        assert orbit.closure_error is not None
        assert orbit.closure_error < 1e-3
