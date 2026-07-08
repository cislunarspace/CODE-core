"""对照 e2m2e 与 DFH 东方红软件的 cislunar 轨道外推结果。

DFH 基准：``EPHEMERIDES_DAC.TXT``（L1 Halo，振幅 40000 km，0.5 年）。
- 历元 2026-01-01 00:00:00 UTC
- 力模型：地球点质量 + 月球点质量 + 太阳第三体引力（DFH inputs-dac.txt
  Halo 段开关 ``[1 0 0 0 0 0 0 0 0]``，即仅太阳第三体打开）
- 输出：ECI（地心 J2000），位置 km、速度 m/s，每小时一点

e2m2e 用等价力模型：
    ``PointMassGravity(EARTH) + ThirdBodyGravity(MOON) + ThirdBodyGravity(SUN)``
在 ICRF（地心 J2000，与 DFH 的 ECI 一致）下积分。

差异来源（按预期量级）：
1. 星历表版本——DFH 用 DE430（JPLEPH 封装），e2m2e 仓库默认 DE440s。
   DE430/DE440 月球位置差异约米级，0.5 年累积公里量级。
2. 积分器——DFH 二进制不可见；e2m2e 用 Rust PD45。
3. 初值——DFH 首点状态作为 e2m2e 初值，无初始误差。
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.ephemeris_system import EphemerisSystem
from e2m2e.core.forces import (
    ForceModel,
    PointMassGravity,
    SolarRadiationPressure,
    ThirdBodyGravity,
)
from e2m2e.core.spice import SPICEManager
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

DFH_FILE = Path(
    "/home/ouyangjiahong/codes/qiao/OrbitDesign/EPHEMERIDES_DAC.TXT"
)


def main() -> None:
    # --- 加载 SPICE ---
    project_root = Path(__file__).resolve().parent.parent
    kernel_dir = project_root / "kernels"
    spice = SPICEManager()
    # DFH 用 DE430，但 DE430 对 2026 年只覆盖到金星（Mars+缺数据）
    # 用 DE440s 覆盖全行星；星历表差异（DE430 vs DE440）对行星影响很小
    de440s = kernel_dir / "de440s.bsp"
    de430 = kernel_dir / "de430.bsp"
    if de440s.is_file():
        ephem_kernel = str(de440s)
    elif de430.is_file():
        ephem_kernel = str(de430)
    else:
        ephem_kernel = spice.find_ephemeris_kernel(str(kernel_dir))
    pck_kernel = next((p for p in kernel_dir.glob("*.tpc") if p.is_file()), None)
    spice.load_kernel(ephem_kernel)
    if pck_kernel is not None:
        spice.load_kernel(str(pck_kernel))
    # DE430 包含全部行星数据，但缺少名称-ID映射，需手动注册
    from spiceypy import boddef
    for name, naif_id in [
        ("MERCURY", 1), ("VENUS", 2), ("EARTH", 399), ("MARS", 4),
        ("JUPITER", 5), ("SATURN", 6), ("URANUS", 7), ("NEPTUNE", 8),
        ("MOON", 301), ("SUN", 10),
    ]:
        boddef(name, naif_id)
    print(f"星历内核: {Path(ephem_kernel).name}")

    try:
        # --- 解析 DFH 输出 ---
        # 重新解析（上面的 parse 返回占位，这里正算 et）
        utc_list: list[str] = []
        dfh_states: list[list[float]] = []
        pat = re.compile(
            r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2}(?:\.\d+)?)"
            r"\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
            r"\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
        )
        with DFH_FILE.open() as f:
            for line in f:
                m = pat.match(line.strip())
                if not m:
                    continue
                y, mo, d, h, mi, s = (int(m.group(1)), int(m.group(2)),
                                      int(m.group(3)), int(m.group(4)),
                                      int(m.group(5)), float(m.group(6)))
                iso = f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:06.3f}"
                utc_list.append(iso)
                pos = [float(m.group(7)), float(m.group(8)), float(m.group(9))]
                vel = [float(m.group(10)) / 1000.0,
                       float(m.group(11)) / 1000.0,
                       float(m.group(12)) / 1000.0]  # m/s→km/s
                dfh_states.append(pos + vel)

        dfh_et = np.array([spice.utc_to_et(u) for u in utc_list])
        dfh_states = np.array(dfh_states)
        print(f"DFH 轨道: {len(dfh_et)} 点, "
              f"{utc_list[0]} → {utc_list[-1]}")
        print(f"  弧段长度: {(dfh_et[-1] - dfh_et[0]) / 86400:.2f} 天")

        # --- e2m2e 外推（九体第三体 + SRP，逐步量化各摄动贡献）---
        # 分两组对比，用不同力模型跑同一段：
        #   模式 A: 三体点质量（Earth+Moon+Sun），即之前的基准
        #   模式 B: 九体第三体（+七大行星）+ SRP(cannonball)
        # DFH 满配含: 太阳+大行星+地球非球形10×10+月球非球形
        #            +ECOM光压+潮汐+耦合项
        # 模式 B 与 DFH 满配的差异来自：月球非球形、ECOM vs cannonball、
        #   潮汐、耦合项、EGM96(n≤2) vs EGM2008(10×10)
        all_bodies = ["EARTH", "MOON", "SUN",
                      "MERCURY", "VENUS", "MARS",
                      "JUPITER", "SATURN", "URANUS", "NEPTUNE"]
        system = EphemerisSystem(bodies=all_bodies, spice=spice, origin="EARTH")
        system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )
        fm = ForceModel(system)
        fm.max_step = 600.0
        # 地球中心引力
        fm.add_force(PointMassGravity("EARTH"), name="earth")
        # 第三体引力：月球+太阳+七大行星
        for body in ["MOON", "SUN", "MERCURY", "VENUS", "MARS",
                     "JUPITER", "SATURN", "URANUS", "NEPTUNE"]:
            fm.add_force(ThirdBodyGravity(body), name=f"third_{body.lower()}")
        # 太阳光压（cannonball，最接近 ECOM 的简化替代）
        fm.add_force(SolarRadiationPressure(area=10.0, mass=1000.0, cr=1.4),
                     name="srp")

        et0 = dfh_et[0]
        # 采样间隔：取若干个时间点用于画误差曲线（每小时太密，每 6 小时）
        sample_idx = np.arange(0, len(dfh_et), 6)
        t_eval = dfh_et[sample_idx]

        print(f"e2m2e 外推中...")
        print(f"  力模型: PointMass(EARTH) + ThirdBody(Moon+Sun+七大行星) + SRP(cannonball)")
        print(f"  DFH 满配: 太阳+大行星+地球非球形10x10+月球非球形"
              f"+ECOM光压+潮汐+耦合(大气/相对论关)")
        print(f"  e2m2e 缺(相对DFH): 月球非球形、地球非球形、ECOM光压(用cannonball代替)、"
              f"潮汐、非球形-大天体耦合")
        result = fm.propagate(
            dfh_states[0],
            (et0, dfh_et[-1]),
            t_eval=t_eval,
            max_steps=500_000,
        )
        e2m2e_time = result["time"]
        e2m2e_states = result["states"]

        # --- 对比：把 DFH 状态插值到 e2m2e 输出时刻 ---
        dfh_pos_interp = np.empty_like(e2m2e_states[:, :3])
        for k in range(3):
            dfh_pos_interp[:, k] = np.interp(
                e2m2e_time, dfh_et, dfh_states[:, k]
            )
        dfh_vel_interp = np.empty_like(e2m2e_states[:, 3:])
        for k in range(3):
            dfh_vel_interp[:, k] = np.interp(
                e2m2e_time, dfh_et, dfh_states[:, 3 + k]
            )

        pos_err = np.linalg.norm(e2m2e_states[:, :3] - dfh_pos_interp, axis=1)
        vel_err = np.linalg.norm(e2m2e_states[:, 3:] - dfh_vel_interp, axis=1)
        days = (e2m2e_time - et0) / 86400.0

        # --- 汇总 ---
        print("\n===== e2m2e vs DFH 差异 =====")
        print(f"最大位置差: {pos_err.max():.3f} km  "
              f"@ t={days[np.argmax(pos_err)]:.2f} 天")
        print(f"最大速度差: {vel_err.max():.6f} km/s  "
              f"@ t={days[np.argmax(vel_err)]:.2f} 天")
        print(f"末态位置差: {pos_err[-1]:.3f} km")
        # 每 30 天采样
        for mark in [5, 30, 60, 90, 120, 150, len(days) - 1]:
            if mark < len(days):
                print(f"  t={days[mark]:6.1f} 天: 位置差 {pos_err[mark]:10.3f} km, "
                      f"速度差 {vel_err[mark]:.6f} km/s")

        # --- 画图 ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1 = axes[0]
        ax1.semilogy(days, pos_err, "r-", label="位置差")
        ax1.set_ylabel("位置差 [km]")
        ax1.set_title("e2m2e（地月日三体点质量）vs DFH（同配置）")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()

        ax2 = axes[1]
        ax2.semilogy(days, vel_err, "b-", label="速度差")
        ax2.set_ylabel("速度差 [km/s]")
        ax2.set_xlabel("时间 [天]")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        out = project_root / "scripts" / "dfh_vs_e2m2e_error.png"
        plt.savefig(out, dpi=120)
        print(f"\n误差曲线已保存: {out}")

        # 三维轨迹图
        fig2 = plt.figure(figsize=(10, 8))
        ax = fig2.add_subplot(111, projection="3d")
        ax.plot(dfh_states[:, 0], dfh_states[:, 1], dfh_states[:, 2],
                "k-", alpha=0.5, label="DFH")
        ax.plot(e2m2e_states[:, 0], e2m2e_states[:, 1], e2m2e_states[:, 2],
                "r--", alpha=0.7, label="e2m2e")
        ax.plot(0, 0, 0, "ko", markersize=5)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_zlabel("z [km]")
        ax.set_title("L1 Halo (振幅 40000 km, 0.5 年) — ECI 地心 J2000")
        ax.legend()
        plt.tight_layout()
        out2 = project_root / "scripts" / "dfh_vs_e2m2e_trajectory.png"
        plt.savefig(out2, dpi=120)
        print(f"轨迹图已保存: {out2}")

    finally:
        spice.unload_kernel(ephem_kernel)
        if pck_kernel is not None:
            spice.unload_kernel(str(pck_kernel))


if __name__ == "__main__":
    main()
