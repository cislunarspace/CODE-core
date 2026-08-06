"""NRHO 黄金样本重生成为有界轨道的参数探针（一次性脚本，用完即删）。

对若干 NRHO 参数变体跑 DFH_DAC.exe，检查输出星历是否始终维持在月球
附近（距月 < 150000 km），找出可用算例。现有 fixture
EPHEMERIDES_DESIGN_NRHO_L2_NORTH.TXT（perilune 3500）的轨迹第 4 天起
逃逸，不可用。

说明文档：inputs-dac.txt 第 33 行（NRHO 近月点高度行）为两列——
列 1 近月点高度、列 2 初始相位（0.01~0.99，近月点低时相位要明显大于 0）。
MATLAB 封装与 format_inputs_design 都只写一列，相位缺失导致 DFH 设计
出逃逸轨道。本探针在生成后手工补上第二列。
"""

import re
import shutil
import subprocess
from pathlib import Path

from scripts.dfh_inputs_dac import format_inputs_design, write_inputs_dac

ORBIT_DIR = Path(
    r"C:\Users\ouyangjiahong\Downloads\Compressed\orbit-design-module"
    r"\CislunarOrbitPack\orbit"
)
OUT_DIR = Path(__file__).resolve().parent / "_nrho_probe"
DYB = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PERT = {
    "sun_body": 1,
    "planets": 1,
    "earth_nonspherical": 1,
    "moon_nonspherical": 1,
    "solar_radiation": 0,
    "atmosphere": 0,
    "relativity": 0,
    "tide": 0,
    "coupling": 0,
}

# (名称, 近月点高度, 历元, 相位)
VARIANTS = [
    ("p5000_e2024_ph05", 5000.0, [2024, 1, 1, 0, 0, 0.0], 0.5),
    ("p5000_e2024_ph01", 5000.0, [2024, 1, 1, 0, 0, 0.0], 0.1),
    ("p3500_e2024_ph05", 3500.0, [2024, 1, 1, 0, 0, 0.0], 0.5),
    ("p3500_e2024_ph01", 3500.0, [2024, 1, 1, 0, 0, 0.0], 0.1),
]

_PERILUNE_ROW = re.compile(r"待设计NRHO轨道的近地点高度")


def _add_phase_column(lines, phase):
    """把 NRHO 近月点高度行改成两列（高度 + 相位）。"""
    out = []
    for ln in lines:
        if _PERILUNE_ROW.search(ln):
            val, _, comment = ln.partition("//")
            out.append(f"{val.strip()}  {phase:g}".ljust(30) + "//" + comment)
        else:
            out.append(ln)
    return out


def run_variant(name, perilune, epoch, phase):
    lines = format_inputs_design(
        "NRHO",
        collinear_point=2,
        north_south=1,
        perilune_height=perilune,
        epoch=epoch,
        duration=0.1,
        perturbation=PERT,
        dyb=DYB,
        earth_degree=10,
        moon_degree=10,
        output_step=3600.0,
    )
    lines = _add_phase_column(lines, phase)
    inputs_path = ORBIT_DIR / "inputs-dac.txt"
    write_inputs_dac(lines, inputs_path)
    try:
        proc = subprocess.run(
            [str(ORBIT_DIR / "DFH_DAC.exe")],
            cwd=ORBIT_DIR,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        err = ORBIT_DIR / "error.txt"
        err_msg = err.read_text(encoding="utf-8", errors="replace").strip() if err.is_file() else ""
        if proc.returncode != 0 or err_msg:
            print(f"[{name}] DFH 失败 rc={proc.returncode} err={err_msg[:200]}", flush=True)
            return
        eph = ORBIT_DIR / "EPHEMERIDES_DAC.TXT"
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / f"{name}.TXT"
        shutil.copyfile(eph, out)
        n_lines = sum(1 for ln in out.read_text().splitlines() if ln.strip())
        print(f"[{name}] {n_lines} 行 -> {out}", flush=True)
    finally:
        inputs_path.unlink(missing_ok=True)


for name, perilune, epoch, phase in VARIANTS:
    run_variant(name, perilune, epoch, phase)
print("DONE")
