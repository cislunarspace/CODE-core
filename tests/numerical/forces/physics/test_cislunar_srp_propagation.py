"""cislunar SRP + 地影端到端传播测试。

验证赤道圆轨进出地影时 SRP 加速度跳变。
"""

from __future__ import annotations

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel, GravityField
from e2m2e.algorithm.forces.shadow import ConicalShadowModel
from e2m2e.algorithm.forces.srp import SolarRadiationPressure
from e2m2e.data.constants import SOLAR_PRESSURE_1AU
from tests.numerical.forces.conftest import EARTH_MU, EARTH_RE

pytestmark = pytest.mark.force


@pytest.mark.spice
def test_equatorial_leo_crosses_earth_shadow(earth_icrf_system) -> None:
    """秋分赤道圆轨传播 1.5 圈，进出地影时 SRP 加速度明显跳变。

    初值置于日侧正午；半圈后到反日点入地影。断言：
    - 经历全光照（flux≈1）与本影（flux≈0）；
    - 本影段 SRP 加速度 ≈ 0，全光照段 ≈ 满 SRP；
    - 半影过渡占采样比例小（窄跳变）。
    """
    system = earth_icrf_system
    spice = system.spice
    et0 = spice.utc_to_et("2025-09-22T12:00:00")  # 秋分附近，日赤纬≈0

    # 太阳方向（秋分时在赤道面内）
    sun_pos = spice.get_body_state("SUN", et0, "J2000", "EARTH")[:3]
    sun_dir = sun_pos / np.linalg.norm(sun_pos)

    # 赤道圆轨 400 km，初值在日侧正午
    a = EARTH_RE + 400.0
    v_circ = np.sqrt(EARTH_MU / a)
    r0 = sun_dir * a
    # 顺行：赤道面内垂直日向（z 轴为轨道法向）
    v0 = np.array([-sun_dir[1], sun_dir[0], 0.0]) * v_circ
    y0 = np.concatenate([r0, v0])

    # 力：J2 + SRP（含地影）
    gravity = GravityField(body="EARTH", degree=2, order=0)
    shadow = ConicalShadowModel(bodies=["EARTH"])
    srp = SolarRadiationPressure(area=20.0, mass=1000.0, cr=1.5, shadow=shadow)
    fm = ForceModel(system, forces=[gravity, srp])

    # 传播 1.5 圈（半圈到反日点入影，再半圈到正午出影后又入影）
    period = 2.0 * np.pi * np.sqrt(a**3 / EARTH_MU)
    t_span = (et0, et0 + 1.5 * period)
    t_eval = np.linspace(*t_span, 400)
    result = fm.propagate(y0, t_span, t_eval=t_eval, max_steps=200_000)

    times = result["time"]
    states = result["states"]

    # 逐点算阴影光照份额与 SRP 量级（Rust 单点绑定）
    from e2m2e.integrators import srp_acceleration

    flux = np.array([shadow.flux_factor(times[i], states[i], system) for i in range(len(times))])
    srp_mag = np.array(
        [
            np.linalg.norm(
                srp_acceleration(
                    times[i], states[i, :3].tolist(), srp.area, srp.mass, srp.cr, ["EARTH"], "EARTH"
                )
            )
            for i in range(len(times))
        ]
    )

    # 1) 经历全光照与本影
    assert flux.max() > 0.99, f"应经历全光照，max flux={flux.max():.4f}"
    assert flux.min() < 1e-6, f"应进入本影，min flux={flux.min():.4e}"

    # 2) 本影段 SRP 加速度 ≈ 0；全光照段 ≈ 满 SRP（P·Cr·A/m·(AU/r)²，r≈1AU）
    cr, area, mass = 1.5, 20.0, 1000.0
    full_mag_km = SOLAR_PRESSURE_1AU * cr * area / mass / 1000.0  # km/s²
    umbra_mask = flux < 1e-6
    sun_mask = flux > 0.99
    assert srp_mag[umbra_mask].max() < full_mag_km * 1e-6, "本影段 SRP 加速度应≈0"
    np.testing.assert_allclose(
        srp_mag[sun_mask].max(),
        full_mag_km,
        rtol=0.05,
        err_msg="全光照段 SRP 加速度应≈满量级",
    )

    # 3) 半影过渡占采样比例小（窄跳变，非渐变）
    penumbra_frac = np.mean((flux > 1e-6) & (flux < 0.99))
    assert penumbra_frac < 0.10, f"半影过渡占比应<10%，实际 {penumbra_frac:.1%}"
