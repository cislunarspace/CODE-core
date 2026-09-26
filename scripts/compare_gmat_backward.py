"""将 e2m2e 反向传播结果与 GMAT backward 参考输出做对比。

手动对拍脚本，不进 CI（pytest testpaths 只收 ``tests/``）。当前环境未安装
可运行的 GMAT 二进制，本脚本默认读取由 ``generate_gmat_backward_script.py``
生成、并经 GMAT 运行后产出的报告文件；若报告文件不存在，脚本会提示运行
GMAT 的命令并退出（退出码 0，与 ``compare_with_gmat.py`` 一致）。

场景：400 km / 51.6° LEO，三体点质量力模型（Earth + Luna/Sun），自历元
2025-06-21T11:00:06 UTC 反向积分 1 天。两侧时刻均递减，按时刻插值对齐。

预期量级：星历口径差（GMAT DE405 vs e2m2e DE440 月/日位置差）主导，
1 天 LEO 约 10–100 m 量级为正常；数量级异常（km 级）即说明反向路径有问题。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

#: 与 e2m2e 默认三体配置一致的初值口径（compare_with_gmat.py 同款）。
_MU_EARTH = 398600.4415

#: J2000 历元的 TDB 儒略日（e2m2e ``times_jd_tdb`` 的零点）。
_J2000_JD_TDB = 2451545.0


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
    """解析 GMAT ReportFile 输出，返回 utc 字符串与 states 数组。"""
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


def _interpolate_states(
    source_time: npt.NDArray[np.floating],
    source_states: npt.NDArray[np.floating],
    target_time: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """对状态做逐分量线性插值。

    反向对拍两侧时刻均递减，故在取负的时间轴上插值（``-t`` 递增）。
    """
    out = np.empty((target_time.shape[0], source_states.shape[1]))
    for i in range(source_states.shape[1]):
        out[:, i] = np.interp(-target_time, -source_time, source_states[:, i])
    return out


def _find_kernel_dir(output_dir: Path) -> Path:
    """自 output_dir 向上寻找仓库 kernels/ 目录。"""
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
    return project_root / "kernels"


def _propagate_e2m2e_backward(output_dir: Path) -> dict[str, Any]:
    """用 e2m2e 反向传播与 GMAT 脚本对应的 LEO 场景。"""
    from e2m2e.algorithm.propagation import propagate_orbit
    from e2m2e.data.kernels.manager import SPICEManager

    kernel_dir = _find_kernel_dir(output_dir)
    spice = SPICEManager()
    ephem_kernel = spice.find_ephemeris_kernel(str(kernel_dir))
    tls_kernel = next((p for p in kernel_dir.glob("*.tls") if p.is_file()), None)
    spice.load_kernel(ephem_kernel)
    if tls_kernel is not None:
        spice.load_kernel(str(tls_kernel))

    try:
        y0 = _keplerian_to_cartesian(6778.0, 0.001, 51.6, 0.0, 0.0, 0.0, _MU_EARTH)
        result = propagate_orbit(
            initial_state=y0,
            epoch="2025-06-21T11:00:06",
            duration=86400.0,
            direction="backward",
            output_step=3600.0,
        )
        ephemeris = result.ephemeris
        times_jd = ephemeris.times_jd_tdb
        assert times_jd is not None  # propagate_orbit 始终填充
        time = (times_jd - _J2000_JD_TDB) * 86400.0
        states = np.hstack([ephemeris.position_km, ephemeris.velocity_mps / 1000.0])
        return {"time": time, "states": states, "spice": spice}
    finally:
        spice.unload_kernel(ephem_kernel)
        if tls_kernel is not None:
            spice.unload_kernel(str(tls_kernel))


def _write_report(
    e2m2e_time: npt.NDArray[np.floating],
    pos_err_m: npt.NDArray[np.floating],
    output_dir: Path,
) -> Path:
    """写纯文本对比报告。"""
    report_path = output_dir / "backward_comparison_report.txt"
    max_pos_m = float(np.max(pos_err_m))
    final_pos_m = float(pos_err_m[-1])
    et0 = float(e2m2e_time[0])

    lines: list[str] = []
    lines.append("# 反向传播对比报告：GMAT vs e2m2e")
    lines.append("")
    lines.append(f"- 最大位置误差：{max_pos_m:.3f} m")
    lines.append(f"- 末位置误差：{final_pos_m:.3f} m")
    lines.append("- 预期量级：星历口径差主导，1 天 LEO 约 10–100 m 为正常")
    lines.append("")
    lines.append("| elapsed_h | position_error_m |")
    lines.append("|---|---|")
    for et, err in zip(e2m2e_time, pos_err_m, strict=True):
        lines.append(f"| {(et - et0) / 3600.0:.4f} | {err:.6f} |")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare e2m2e backward propagation with a GMAT reference report."
    )
    parser.add_argument(
        "--gmat-report",
        type=str,
        default="./gmat_backward_output/backward_reference_gmat_report.txt",
        help="Path to GMAT ReportFile output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gmat_backward_output",
        help="Directory for the comparison report.",
    )
    args = parser.parse_args()

    gmat_report = Path(args.gmat_report).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gmat_report.exists():
        print(f"GMAT report not found: {gmat_report}")
        print("Please run the generated GMAT script first:")
        script_path = output_dir / "backward_reference_gmat.script"
        print(f"  gmat -s {script_path}")
        print("Then rerun this script.")
        return

    print("Parsing GMAT report...")
    gmat_data = _parse_gmat_report(gmat_report)

    print("Running e2m2e backward propagation...")
    e2m2e_data = _propagate_e2m2e_backward(output_dir)

    print("Aligning time and computing errors...")
    gmat_et = _utc_to_et(gmat_data["utc"], e2m2e_data["spice"])
    gmat_at_e2m2e = _interpolate_states(gmat_et, gmat_data["states"], e2m2e_data["time"])
    pos_err_m = np.linalg.norm(gmat_at_e2m2e[:, :3] - e2m2e_data["states"][:, :3], axis=1)

    print("Writing report...")
    report_path = _write_report(e2m2e_data["time"], pos_err_m, output_dir)

    print(f"Done. Report: {report_path}")
    print(f"Max position error: {float(np.max(pos_err_m)):.3f} m")
    print(f"Final position error: {float(pos_err_m[-1]):.3f} m")


if __name__ == "__main__":
    main()
