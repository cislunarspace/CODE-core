#!/usr/bin/env python3
"""统一轨道族生成示例（纯 CR3BP，不依赖 SPICE）。

生成一组 L4 SPO 周期轨道，以及一组 L2 Lissajous 拟周期有界轨迹，
展示 ``OrbitFamily`` 统一容器如何通过 ``periodicity`` 区分结果语义。

用法：
    python examples/main_family_generation.py
"""

from __future__ import annotations

import time

from e2m2e.api.facade import Facade


def main() -> None:
    facade = Facade()

    print("生成 L4 SPO 周期族（振幅 5,000~20,000 km）")
    started = time.perf_counter()
    spo = facade.orbit_family_generation(
        orbit_type="SPO",
        libration_point=4,
        min_amplitude_km=5000.0,
        max_amplitude_km=20000.0,
        n_orbits=5,
    )
    print(f"  生成 {len(spo)} 个成员，耗时 {time.perf_counter() - started:.1f} s")
    print(f"  周期语义：{spo.periodicity}")
    for index, orbit in enumerate(spo, start=1):
        print(
            f"  {index}: amplitude={orbit.parameters['amplitude_km']:.0f} km, "
            f"period={orbit.period:.6f} TU"
        )

    print("\n生成 L2 Lissajous 拟周期采样族")
    started = time.perf_counter()
    lissajous = facade.orbit_family_generation(
        orbit_type="LISSAJOUS",
        libration_point=2,
        amplitude_in_km=2400.0,
        amplitude_out_km=7200.0,
        phase_in=0.01,
        phase_out=0.55,
        n_orbits=2,
    )
    print(f"  生成 {len(lissajous)} 个成员，耗时 {time.perf_counter() - started:.1f} s")
    print(f"  周期语义：{lissajous.periodicity}")
    print(f"  拟周期标注：{lissajous.is_quasi_periodic}")
    print(f"  每个成员状态点数：{[len(orbit.states) for orbit in lissajous]}")


if __name__ == "__main__":
    main()
