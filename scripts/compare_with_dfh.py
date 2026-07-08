"""对照 e2m2e 与 DFH 东方红软件的 cislunar 轨道外推结果。

DFH 基准：``EPHEMERIDES_DAC.TXT``（L1 Halo，振幅 40000 km，0.5 年）。
- 历元 2026-01-01 00:00:00 UTC
- 力模型（DFH 满配）：太阳+大行星第三体 + 地球非球形 10×10(EGM2008)
  + 月球非球形 10×10(GRM) + ECOM 光压 + 固体潮 + 非球形-大天体耦合项
  （大气阻力 / 相对论关）
- 输出：ECI（地心 J2000），位置 km、速度 m/s，每小时一点

e2m2e 力模型（issue #189 集成验收，对齐 DFH 满配）：
    ``PointMassGravity(EARTH) + GravityField(MOON, 10×10, solid)``
    ``+ IndirectTerm(MOON) + ThirdBodyGravity(SUN + 七大行星) + SRP(cannonball)``
在 ICRF（地心 J2000，与 DFH 的 ECI 一致）下积分。

选型说明：
- **地球用 PointMassGravity，不用 GravityField**：航天器在 L1（距地心 ~305,000 km），
  地球非球形项 ∝ (R_E/r)ⁿ 衰减极快，n=2 项仅为中心引力的 ~4e-4，对 cislunar
  外推可忽略。同时避开了 ``earth_latest_high_prec.bpc`` 只覆盖到 2026-06-12、
  无法外推满 0.5 年的问题（ITRF93 body-fixed 变换需要该 BPC）。
- **月球用 GravityField**：月球是近旁天体（距航天器 ~70,000 km），月球非球形
  是 cislunar 摄动的主导项，且 MOON_PA 帧（SPICELunaCurrentKernel.bpc）覆盖全弧段。
- **IndirectTerm(MOON)**：地心加速系下 N 体闭式需要月球间接项
  ``-μ_M·r_M/|r_M|³``。``GravityField`` 只算球谐直接引力（含 degree=0 中心项），
  不带间接项；不能用 ``ThirdBodyGravity("MOON")`` 替代（会与 degree=0 重复算
  月球点质量），故单独补 ``IndirectTerm``。
- **太阳/行星用 ThirdBodyGravity**：远天体，点质量（自带直接项 + 间接项）。

差异来源（e2m2e 相对 DFH 满配）：
1. ECOM 光压——e2m2e 用 cannonball 简化代替。
2. 地球非球形——e2m2e 用点质量代替（在 L1 可忽略；DFH 用 EGM2008 10×10）。
3. 非球形-大天体耦合项——e2m2e 未实现。
4. 星历表——两边均用 DE430。
5. 积分器——DFH 二进制不可见；e2m2e 用 Rust PD45。
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
    GravityField,
    PointMassGravity,
    SolarRadiationPressure,
    ThirdBodyGravity,
    PhysicalModel,
)
from e2m2e.core.spice import SPICEManager
from e2m2e.core.standard_axes import ICRSAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

DFH_FILE = Path(
    "/home/ouyangjiahong/codes/qiao/OrbitDesign/EPHEMERIDES_DAC.TXT"
)


class IndirectTerm(PhysicalModel):
    """第三体引力的间接项（geocentric 加速系所需）。

    在以某天体（如地球）为原点的非惯性系下传播时，运动方程需对每个摄动
    天体 :math:`i` 补一项 ``-μ_i · r_i / |r_i|³``（间接项），扣除摄动天体
    对原点的引力（见 ``EphemerisDynamics`` 的 N 体闭式公式）。

    ``ThirdBodyGravity`` 内部已自带间接项，但 ``GravityField`` 只算球谐
    直接引力（含中心项 degree=0），不带间接项。所以用 ``GravityField``
    模拟月球（中心+非球形）时，必须单独补月球间接项——既不能用
    ``ThirdBodyGravity("MOON")``（会与 ``GravityField`` 的 degree=0 中心项
    重复算月球点质量），也不能省略（地心系下物理不正确）。

    加速度：``-μ_body · r_body / |r_body|³``，其中 ``r_body`` 为摄动天体相对
    ``system.origin`` 的位置（由 ``system.get_body_position`` 自动以 origin
    为观察者计算）。与 ``ThirdBodyGravity`` 的间接项逐字一致。
    """

    def __init__(self, body: str, mu: float | None = None) -> None:
        self._body = body.upper()
        self._mu = float(mu) if mu is not None else None

    def compute_acceleration(self, t, state, system):
        mu = self._mu
        if mu is None:
            mu = system.gravitational_parameter(self._body)
        r_ob = np.asarray(system.get_body_position(self._body, t), dtype=float)
        n = float(np.linalg.norm(r_ob))
        if n < 1e-6:
            return np.zeros(3)
        return -mu * r_ob / n**3


def main() -> None:
    # --- 加载 SPICE ---
    project_root = Path(__file__).resolve().parent.parent
    kernel_dir = project_root / "kernels"
    spice = SPICEManager()
    # DFH 用 DE430（JPLEPH 封装），这里对齐。
    de430 = kernel_dir / "de430.bsp"
    de440s = kernel_dir / "de440s.bsp"
    if de430.is_file():
        ephem_kernel = str(de430)
    elif de440s.is_file():
        ephem_kernel = str(de440s)
    else:
        ephem_kernel = spice.find_ephemeris_kernel(str(kernel_dir))
    loaded_kernels: list[str] = [ephem_kernel]
    spice.load_kernel(ephem_kernel)
    # body-fixed 帧所需内核（issue #187）：
    #   地球 ITRF93  <- earth_latest_high_prec.bpc
    #   月球 MOON_PA <- SPICELunaFrameKernel.tf + SPICELunaCurrentKernel.bpc
    #   text PCK     <- pck00010.tpc
    body_fixed_names = [
        "naif0012.tls",
        "pck00010.tpc",
        "earth_latest_high_prec.bpc",
        "SPICELunaFrameKernel.tf",
        "SPICELunaCurrentKernel.bpc",
    ]
    for name in body_fixed_names:
        kpath = kernel_dir / name
        if kpath.is_file():
            spice.load_kernel(str(kpath))
            loaded_kernels.append(str(kpath))
    # DE430 内含全部行星数据，但缺少名称-ID映射，需手动注册（行星用 SSB ID）
    from spiceypy import boddef
    for name, naif_id in [
        ("MERCURY", 1), ("VENUS", 2), ("EARTH", 399), ("MARS", 4),
        ("JUPITER", 5), ("SATURN", 6), ("URANUS", 7), ("NEPTUNE", 8),
        ("MOON", 301), ("SUN", 10),
    ]:
        boddef(name, naif_id)
    print(f"星历内核: {Path(ephem_kernel).name}")
    print(f"body-fixed 内核: {[Path(p).name for p in loaded_kernels[1:]]}")

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

        # --- e2m2e 外推（issue #189：对齐 DFH 满配）---
        # 力模型：地球点质量 + 月球非球形 10×10 + 月球间接项
        #         + 太阳/行星第三体 + SRP
        # 坐标系约定（与 EphemerisDynamics 的 N 体闭式一致）：以地球为原点的
        # 加速系，每个摄动天体需补间接项 -μ_i·r_i/|r_i|³。GravityField 只算
        # 球谐直接引力（含中心项 degree=0），不带间接项；故：
        #   - 地球用 PointMassGravity（L1 处非球形可忽略；且 earth_latest_high_prec.bpc
        #     只覆盖到 2026-06-12，用 GravityField EARTH 会因 ITRF93 变换超覆盖而失败）
        #   - 月球用 GravityField（含中心 + 非球形 + 固体潮）+ IndirectTerm（补间接项）
        #   - 不用 ThirdBodyGravity("MOON")（会与 GravityField MOON 的 degree=0 重复）
        #   - 太阳/行星用 ThirdBodyGravity（自带直接项+间接项，点质量）
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
        # 地球中心引力（点质量；L1 处非球形可忽略）
        fm.add_force(PointMassGravity("EARTH"), name="earth")
        # 月球引力场（中心 + 非球形 GRGM900C 10×10 + 固体潮）
        moon_cof = str(project_root / "e2m2e" / "core" / "forces" / "data"
                       / "grgm900c.cof")
        fm.add_force(
            GravityField("MOON", degree=10, order=10, gravity_file=moon_cof,
                         tide_mode="solid"),
            name="moon_gravity",
        )
        # 月球间接项（地心加速系必需；不能省，也不能用 ThirdBodyGravity 替代）
        fm.add_force(IndirectTerm("MOON"), name="moon_indirect")
        # 太阳和七大行星第三体（点质量，自带直接项 + 间接项）
        for body in ["SUN", "MERCURY", "VENUS", "MARS",
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
        print(f"  力模型: PointMass(EARTH) + GravityField(MOON 10x10 solid)"
              f" + IndirectTerm(MOON) + ThirdBody(Sun+七大行星) + SRP(cannonball)")
        print(f"  DFH 满配: 太阳+大行星+地球非球形10x10(EGM2008)+月球非球形10x10"
              f"+ECOM光压+潮汐+耦合(大气/相对论关)")
        print(f"  e2m2e 缺(相对DFH): ECOM光压(用cannonball代替)、"
              f"地球非球形(用点质量代替)、非球形-大天体耦合")
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
        # 关键时刻采样：1/3/7/15/30 天 + 末态
        # 采样间隔 = 6 小时 → 每天 4 点；用最近索引
        def idx_for_day(d: float) -> int:
            return int(round(d * 4))
        key_marks = [1.0, 3.0, 7.0, 15.0, 30.0]
        print("  关键时刻：")
        for d in key_marks:
            i = idx_for_day(d)
            if i < len(days):
                print(f"  t={days[i]:6.1f} 天: 位置差 {pos_err[i]:10.3f} km, "
                      f"速度差 {vel_err[i]:.6f} km/s")
        print(f"  t={days[-1]:6.1f} 天(末态): 位置差 {pos_err[-1]:10.3f} km, "
              f"速度差 {vel_err[-1]:.6f} km/s")

        # --- 画图 ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1 = axes[0]
        ax1.semilogy(days, pos_err, "r-", label="Position error")
        ax1.set_ylabel("Position error [km]")
        ax1.set_title("e2m2e vs DFH: PointMass(Earth) + Moon gravity field 10x10 + Sun/planets + SRP")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()

        ax2 = axes[1]
        ax2.semilogy(days, vel_err, "b-", label="Velocity error")
        ax2.set_ylabel("Velocity error [km/s]")
        ax2.set_xlabel("Time [days]")
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
        ax.set_title("L1 Halo (amplitude 40000 km, 0.5 yr) - ECI geocentric J2000")
        ax.legend()
        plt.tight_layout()
        out2 = project_root / "scripts" / "dfh_vs_e2m2e_trajectory.png"
        plt.savefig(out2, dpi=120)
        print(f"轨迹图已保存: {out2}")

    finally:
        for kpath in reversed(loaded_kernels):
            spice.unload_kernel(kpath)


if __name__ == "__main__":
    main()
