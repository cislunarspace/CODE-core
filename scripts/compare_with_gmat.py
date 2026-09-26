"""将 e2m2e LEO 传播结果与 GMAT 参考输出做对比分析。

当前环境未安装可运行的 GMAT 二进制，因此本脚本默认读取由
``generate_gmat_leo_script.py`` 生成、并经 GMAT 运行后产出的报告文件。
若报告文件不存在，脚本会提示运行 GMAT 的命令并退出。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# 空间天气输入的单一来源：GMAT 脚本由 generate_gmat_leo_script.py 生成，这里
# 读同一组默认值构造 e2m2e 侧大气，避免两侧对拍不同配置（ADR 0049）。
from generate_gmat_leo_script import DEFAULT_FORCEMODEL


def _keplerian_to_cartesian(
    a: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    nu_deg: float,
    mu: float,
) -> npt.NDArray[np.floating]:
    """开普勒根数转笛卡尔状态（单位与 mu 一致）。"""
    i = np.radians(i_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    nu = np.radians(nu_deg)

    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pqw = np.array(
        [
            -np.sqrt(mu / p) * np.sin(nu),
            np.sqrt(mu / p) * (e + np.cos(nu)),
            0.0,
        ]
    )

    R3_raan = np.array(
        [
            [np.cos(raan), -np.sin(raan), 0.0],
            [np.sin(raan), np.cos(raan), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R1_i = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(i), -np.sin(i)],
            [0.0, np.sin(i), np.cos(i)],
        ]
    )
    R3_argp = np.array(
        [
            [np.cos(argp), -np.sin(argp), 0.0],
            [np.sin(argp), np.cos(argp), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = R3_raan @ R1_i @ R3_argp
    return np.concatenate([R @ r_pqw, R @ v_pqw])


def _parse_gmat_report(path: Path) -> dict[str, npt.NDArray[Any]]:
    """解析 GMAT ReportFile 输出，返回时间(et)、states、utc 字符串数组。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty report file: {path}")

    # 检测表头行：GMAT WriteHeaders=true 时第一行是列名
    header = lines[0].strip()
    data_lines = lines[1:] if "UTCGregorian" in header else lines

    utc_list: list[str] = []
    states: list[list[float]] = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        utc_list.append(parts[0] + "T" + parts[1] if len(parts[0]) == 10 else " ".join(parts[:2]))
        states.append([float(x) for x in parts[-6:]])

    if not states:
        raise ValueError(f"no data rows parsed from {path}")

    states_arr = np.asarray(states, dtype=float)
    return {"utc": np.array(utc_list), "states": states_arr}


def _utc_to_et(utc_strings: npt.NDArray[Any], spice: Any) -> npt.NDArray[np.floating]:
    """把 UTC 字符串数组转为 SPICE et 秒。"""
    return np.array([spice.utc_to_et(str(s)) for s in utc_strings], dtype=float)


def _state_to_elements(
    state: npt.NDArray[np.floating], et: float, mu: float
) -> npt.NDArray[np.floating]:
    """用 spiceypy.oscltx 提取经典轨道根数。"""
    import spiceypy

    return spiceypy.oscltx(state.copy(), et, mu)


def _rtn_error(
    ref_state: npt.NDArray[np.floating],
    test_state: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """把位置/速度差异投影到 RTN 框架（以 ref_state 为基准）。"""
    r_vec = ref_state[:3]
    v_vec = ref_state[3:6]
    h_vec = np.cross(r_vec, v_vec)
    r_hat = r_vec / np.linalg.norm(r_vec)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    R = np.stack([r_hat, t_hat, n_hat], axis=0)
    return R @ (test_state[:3] - ref_state[:3])


def _propagate_e2m2e(
    output_dir: Path,
    include_drag: bool = True,
    include_srp: bool = True,
    degree: int = 10,
    order: int = 10,
    atmosphere: str = "exponential",
) -> dict[str, Any]:
    """用 e2m2e 传播与 GMAT 脚本对应的 LEO 场景。

    ``atmosphere`` 选 ``"exponential"``（USSA76 分段指数，对应 GMAT
    ``Exponential``）或 ``"nrlmsise00"``（NRLMSISE-00；GMAT 侧最近的对应物是
    ``MSISE90``，两者模型版本与空间天气预处理不同，见 ADR 0049）。
    """
    from e2m2e.algorithm.coordinate.coordinate_system import CoordinateSystem
    from e2m2e.algorithm.coordinate.standard_axes import ICRSAxes
    from e2m2e.algorithm.coordinate.standard_origins import CelestialBodyOrigin
    from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
    from e2m2e.algorithm.forces import DragModel, ForceModel, GravityField, SolarRadiationPressure
    from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere, NRLMSISE00Atmosphere
    from e2m2e.data.kernels.manager import SPICEManager

    project_root = output_dir
    while project_root.name != "" and not (project_root / "kernels").is_dir():
        project_root = project_root.parent
    if not (project_root / "kernels").is_dir():
        project_root = Path.cwd()
    while project_root.name != "" and not (project_root / "kernels").is_dir():
        project_root = project_root.parent
    if not (project_root / "kernels").is_dir():
        raise FileNotFoundError(
            f"Could not find kernels directory from {output_dir} or {Path.cwd()}"
        )
    kernel_dir = project_root / "kernels"

    spice = SPICEManager()
    ephem_kernel = spice.find_ephemeris_kernel(str(kernel_dir))
    pck_kernel = next((p for p in kernel_dir.glob("*.tpc") if p.is_file()), None)
    # body-fixed 帧(ITRF93)需要 binary PCK;加载所有 .bpc 内核。
    bpc_kernels = sorted(p for p in kernel_dir.glob("*.bpc") if p.is_file())
    spice.load_kernel(ephem_kernel)
    if pck_kernel is not None:
        spice.load_kernel(str(pck_kernel))
    bpc_loaded = []
    for bpc in bpc_kernels:
        spice.load_kernel(str(bpc))
        bpc_loaded.append(str(bpc))

    try:
        system = EphemerisSystem(bodies=["EARTH"], spice=spice, origin="EARTH")
        system.coordinate_system = CoordinateSystem(
            axes=ICRSAxes(),
            origin=CelestialBodyOrigin(body="EARTH", spice=spice),
        )

        fm = ForceModel(system)
        fm.add_force(GravityField("EARTH", degree=degree, order=order), name="gravity")
        space_weather = "n/a（已关闭阻力）"
        if include_drag:
            if atmosphere == "nrlmsise00":
                # 与生成脚本的 MSISE90 分支同源：F10.7 与 Ap 都取
                # DEFAULT_FORCEMODEL 的默认值（脚本会把它写进
                # AtmosphereModel.F107 / .MagneticIndex）。
                drag_atmosphere = NRLMSISE00Atmosphere(
                    f107_daily=DEFAULT_FORCEMODEL["f107"],
                    f107_avg=DEFAULT_FORCEMODEL["f107"],
                    ap=DEFAULT_FORCEMODEL["ap"],
                )
                space_weather = (
                    f"f107_daily=f107_avg={drag_atmosphere.f107_daily:g} sfu、"
                    f"ap={drag_atmosphere.ap[0]:g}"
                )
            elif atmosphere == "exponential":
                drag_atmosphere = ExponentialAtmosphere()
                space_weather = f"f107={drag_atmosphere.f107:g} sfu、ap={drag_atmosphere.ap:g}"
            else:
                raise ValueError(
                    f"unknown atmosphere {atmosphere!r}; known: 'exponential', 'nrlmsise00'"
                )
            fm.add_force(
                DragModel(
                    atmosphere=drag_atmosphere,
                    area=10.0,
                    mass=1000.0,
                    cd=2.2,
                ),
                name="drag",
            )
        if include_srp:
            fm.add_force(
                SolarRadiationPressure(area=10.0, mass=1000.0, cr=1.4),
                name="srp",
            )

        mu = system.gravitational_parameter("EARTH")
        a0 = 6778.0
        y0 = _keplerian_to_cartesian(a0, 0.001, 51.6, 0.0, 0.0, 0.0, mu)
        et0 = spice.utc_to_et("2025-06-21T11:00:06")
        etf = et0 + 86400.0

        result = fm.propagate(
            y0,
            (et0, etf),
            t_eval=np.linspace(et0, etf, 200),
            max_steps=200_000,
        )
        return {
            "time": result["time"],
            "states": result["states"],
            "system": system,
            "spice": spice,
            "space_weather": space_weather,
        }
    finally:
        for bpc in reversed(bpc_loaded):
            spice.unload_kernel(bpc)
        spice.unload_kernel(ephem_kernel)
        if pck_kernel is not None:
            spice.unload_kernel(str(pck_kernel))


def _interpolate_states(
    source_time: npt.NDArray[np.floating],
    source_states: npt.NDArray[np.floating],
    target_time: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """对状态做逐分量线性插值。"""
    out = np.empty((target_time.shape[0], source_states.shape[1]))
    for i in range(source_states.shape[1]):
        out[:, i] = np.interp(target_time, source_time, source_states[:, i])
    return out


def _compute_errors(
    gmat_time: npt.NDArray[np.floating],
    gmat_states: npt.NDArray[np.floating],
    e2m2e_time: npt.NDArray[np.floating],
    e2m2e_states: npt.NDArray[np.floating],
) -> dict[str, npt.NDArray[np.floating]]:
    """计算 GMAT 与 e2m2e 之间的误差序列。"""
    e2m2e_at_gmat = _interpolate_states(e2m2e_time, e2m2e_states, gmat_time)

    pos_err = np.linalg.norm(gmat_states[:, :3] - e2m2e_at_gmat[:, :3], axis=1)
    vel_err = np.linalg.norm(gmat_states[:, 3:6] - e2m2e_at_gmat[:, 3:6], axis=1)

    rtn = np.array(
        [_rtn_error(gmat_states[i], e2m2e_at_gmat[i]) for i in range(gmat_states.shape[0])]
    )

    return {
        "time": gmat_time,
        "position_error_km": pos_err,
        "velocity_error_km_s": vel_err,
        "rtn_error_km": rtn,
    }


def _plot_errors(errors: dict[str, npt.NDArray[np.floating]], output_dir: Path) -> list[Path]:
    """生成误差曲线图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_hours = (errors["time"] - errors["time"][0]) / 3600.0
    figures: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_hours, errors["position_error_km"] * 1000.0)
    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("LEO Position Error: GMAT vs e2m2e")
    ax.grid(True)
    path = output_dir / "position_error.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_hours, errors["velocity_error_km_s"] * 1000.0)
    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_ylabel("Velocity Error (m/s)")
    ax.set_title("LEO Velocity Error: GMAT vs e2m2e")
    ax.grid(True)
    path = output_dir / "velocity_error.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Radial", "Transverse", "Normal"]
    for idx, label in enumerate(labels):
        ax.plot(time_hours, errors["rtn_error_km"][:, idx] * 1000.0, label=label)
    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_ylabel("RTN Error (m)")
    ax.set_title("LEO RTN Error: GMAT vs e2m2e")
    ax.legend()
    ax.grid(True)
    path = output_dir / "rtn_error.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)

    return figures


def _write_report(
    errors: dict[str, npt.NDArray[np.floating]],
    figure_paths: list[Path],
    output_dir: Path,
    atmosphere: str = "exponential",
    space_weather_e2m2e: str = "n/a",
    gmat_script_cfg: dict[str, str | None] | None = None,
) -> Path:
    """写 Markdown 报告。"""
    report_path = output_dir / "comparison_report.md"
    max_pos_m = float(np.max(errors["position_error_km"])) * 1000.0
    final_pos_m = float(errors["position_error_km"][-1]) * 1000.0
    max_vel_mm_s = float(np.max(errors["velocity_error_km_s"])) * 1e6

    lines: list[str] = []
    lines.append("# LEO 传播对比报告：GMAT vs e2m2e")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append(f"- 最大位置误差：{max_pos_m:.3f} m")
    lines.append(f"- 1 天末位置误差：{final_pos_m:.3f} m")
    lines.append(f"- 最大速度误差：{max_vel_mm_s:.3f} mm/s")
    lines.append("- 目标（参考性）：1 天末位置误差 < 100 m")
    lines.append("")
    lines.append("## 配置")
    lines.append("")
    lines.append("- 轨道：400 km 高度圆轨道，倾角 51.6°")
    lines.append("- 历元：2025-06-21T11:00:06 UTC")
    lines.append("- 坐标系：EarthICRF")
    lines.append(f"- 力模型：J2(10,10) + {atmosphere} 阻力 + SRP（无阴影）")
    lines.append(f"- 空间天气（e2m2e 侧）：{space_weather_e2m2e}")
    cfg = gmat_script_cfg or {}
    if cfg.get("drag") is None:
        lines.append("- GMAT 脚本侧：未找到 `leo_reference_gmat.script`，无法核对实际配置。")
    else:
        sw = (
            f"F107={cfg['f107']}、MagneticIndex={cfg['ap']}"
            if cfg["f107"] is not None
            else "F107/MagneticIndex 由 GMAT 内置默认（脚本内被注释）"
        )
        lines.append(f"- GMAT 脚本侧（读回脚本核对）：Drag={cfg['drag']}、{sw}")
    lines.append("- 积分器：RK89，MaxStep=60 s，Accuracy=1e-13")
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    for fig in figure_paths:
        lines.append(f"![{fig.name}]({fig.name})")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append(
        "本报告由 `scripts/compare_with_gmat.py` 自动生成。"
        "GMAT 脚本由 `scripts/generate_gmat_leo_script.py` 生成，"
        "需手动或用 GMAT CLI 运行后产生报告文件，再交由本脚本对比。"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _arc_semi_major_axes(data: dict[str, Any]) -> tuple[float, float]:
    """弧段首/末的 osculating 半长轴（km）。

    注意：J2(10,10) 的短周期项会让 osculating SMA 在单圈内起伏数 km（400 km LEO
    上 ~3.5 km），与 24 h 的阻力衰减（~0.36 km）同量级甚至更大。故本量**不能**
    直接当阻力衰减，必须与同弧段无阻力基线对比（见 `_print_e2m2e_summary`）。
    """
    states = np.asarray(data["states"], dtype=float)
    mu = data["system"].gravitational_parameter("EARTH")

    def sma(state: npt.NDArray[np.floating]) -> float:
        r = float(np.linalg.norm(state[:3]))
        v = float(np.linalg.norm(state[3:6]))
        return -mu / (2.0 * (0.5 * v * v - mu / r))

    return sma(states[0]), sma(states[-1])


def _print_e2m2e_summary(
    data: dict[str, Any], atmosphere: str, baseline: dict[str, Any] | None = None
) -> None:
    """打印 e2m2e 侧弧段摘要（GMAT 报告缺失时用于记录本方输出）。

    `baseline` 为同弧段、同 SRP、**关闭阻力**的传播结果：阻力衰减取两者 osculating
    SMA 差之差，扣掉重力场短周期项——否则打印值会被 J2 起伏主导（约 10 倍）。
    """
    time = np.asarray(data["time"], dtype=float)
    states = np.asarray(data["states"], dtype=float)
    a0, a1 = _arc_semi_major_axes(data)

    print(f"  atmosphere    : {atmosphere}")
    print(f"  space weather : {data.get('space_weather', 'n/a')}")
    print(f"  arc           : {(time[-1] - time[0]) / 3600.0:.3f} h, {states.shape[0]} samples")
    print(f"  SMA initial   : {a0:.6f} km")
    print(f"  SMA final     : {a1:.6f} km")
    if baseline is None:
        print(f"  SMA delta     : {a0 - a1:.6f} km（含重力场短周期项，勿作阻力衰减）")
    else:
        b0, b1 = _arc_semi_major_axes(baseline)
        print(f"  SMA delta     : {a0 - a1:.6f} km（含重力场短周期项）")
        print(f"  SMA delta 基线: {b0 - b1:.6f} km（同弧段无阻力）")
        print(f"  → 阻力衰减     : {(a0 - a1) - (b0 - b1):.6f} km")
    print(f"  all finite    : {bool(np.all(np.isfinite(states)))}")


def _read_gmat_script_config(script_path: Path) -> dict[str, str | None]:
    """从生成的 GMAT 脚本读回力模型侧配置（Drag 模型与空间天气）。

    报告里"GMAT 脚本侧"的输入必须取自**实际要跑的脚本**，而不是生成器的默认值：
    生成（`--drag-model`）与对拍（`--atmosphere`）分属两个工具的两个开关，脱钩时
    会静默把"输入不齐"记成"模型差异"（验收条件二要求同配置）。
    注释行（以 ``%`` 开头）跳过——Exponential 分支的 F107/MagneticIndex 即被注释，
    GMAT 用其内置默认。
    """
    cfg: dict[str, str | None] = {"drag": None, "f107": None, "ap": None}
    if not script_path.exists():
        return cfg
    patterns = {
        "drag": r"\.Drag\s*=\s*(\w+)\s*;",
        "f107": r"\.AtmosphereModel\.F107\s*=\s*(\S+?)\s*;",
        "ap": r"\.AtmosphereModel\.MagneticIndex\s*=\s*(\S+?)\s*;",
    }
    for raw in script_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        for key, pattern in patterns.items():
            if cfg[key] is None:
                match = re.search(pattern, line)
                if match:
                    cfg[key] = match.group(1)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare e2m2e LEO propagation with a GMAT reference report."
    )
    parser.add_argument(
        "--gmat-report",
        type=str,
        default="./gmat_leo_output/leo_reference_gmat_report.txt",
        help="Path to GMAT ReportFile output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gmat_leo_output",
        help="Directory for comparison plots and report.",
    )
    parser.add_argument(
        "--no-drag",
        action="store_true",
        help="Run e2m2e comparison without drag (for incremental analysis).",
    )
    parser.add_argument(
        "--no-srp",
        action="store_true",
        help="Run e2m2e comparison without SRP (for incremental analysis).",
    )
    parser.add_argument(
        "--atmosphere",
        type=str,
        choices=["exponential", "nrlmsise00"],
        default="exponential",
        help="Atmosphere model used on the e2m2e side (default: exponential).",
    )
    parser.add_argument(
        "--e2m2e-only",
        action="store_true",
        help=(
            "Only run the e2m2e side and print its summary, skipping the GMAT "
            "comparison (use when no GMAT binary/report is available)."
        ),
    )
    args = parser.parse_args()

    gmat_report = Path(args.gmat_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "leo_reference_gmat.script"

    # 对拍两侧必须同配置：核对实际要跑的 GMAT 脚本的大气模型与 e2m2e 侧选择。
    script_cfg = _read_gmat_script_config(script_path)
    expected_drag = "MSISE90" if args.atmosphere == "nrlmsise00" else "Exponential"
    if script_cfg["drag"] is not None and script_cfg["drag"] != expected_drag:
        raise SystemExit(
            f"GMAT 脚本 {script_path.name} 的 Drag = {script_cfg['drag']}，而 e2m2e 侧选择 "
            f"{args.atmosphere}（期望 GMAT 侧 {expected_drag}）。请先执行 "
            f"generate_gmat_leo_script.py --drag-model {expected_drag} 重新生成脚本。"
        )

    if args.e2m2e_only:
        print(f"Running e2m2e propagation (atmosphere={args.atmosphere})...")
        e2m2e_data = _propagate_e2m2e(
            output_dir,
            include_drag=not args.no_drag,
            include_srp=not args.no_srp,
            atmosphere=args.atmosphere,
        )
        baseline = None
        if not args.no_drag:
            # 同弧段、同 SRP、关闭阻力：扣掉重力场短周期项才是阻力衰减量。
            print("Running no-drag baseline for the SMA-decay metric...")
            baseline = _propagate_e2m2e(
                output_dir,
                include_drag=False,
                include_srp=not args.no_srp,
                atmosphere=args.atmosphere,
            )
        _print_e2m2e_summary(e2m2e_data, args.atmosphere, baseline=baseline)
        return

    if not gmat_report.exists():
        print(f"GMAT report not found: {gmat_report}")
        print("Please run the generated GMAT script first:")
        print(f"  gmat -s {script_path}")
        print("Then rerun this script.")
        print("To record only the e2m2e side, rerun with --e2m2e-only.")
        return

    print("Parsing GMAT report...")
    gmat_data = _parse_gmat_report(gmat_report)

    print("Running e2m2e propagation...")
    e2m2e_data = _propagate_e2m2e(
        output_dir,
        include_drag=not args.no_drag,
        include_srp=not args.no_srp,
        atmosphere=args.atmosphere,
    )

    print("Computing errors...")
    gmat_et = _utc_to_et(gmat_data["utc"], e2m2e_data["spice"])
    errors = _compute_errors(gmat_et, gmat_data["states"], e2m2e_data["time"], e2m2e_data["states"])

    print("Generating plots...")
    figures = _plot_errors(errors, output_dir)

    print("Writing report...")
    report_path = _write_report(
        errors,
        figures,
        output_dir,
        atmosphere=args.atmosphere,
        space_weather_e2m2e=e2m2e_data.get("space_weather", "n/a"),
        gmat_script_cfg=script_cfg,
    )

    print(f"Done. Report: {report_path}")
    print(f"Max position error: {float(np.max(errors['position_error_km'])) * 1000.0:.3f} m")
    print(f"Final position error: {float(errors['position_error_km'][-1]) * 1000.0:.3f} m")


if __name__ == "__main__":
    main()
