"""生成 DFH 黄金样本（功能码 4 预报三档 + 功能码 1 设计算例 + 功能码 2 控制算例）。

流程：``e2m2e.io.format_inputs_propagate`` / ``format_inputs_design`` /
``format_inputs_control`` 生成 inputs-dac.txt 写入 DFH exe 所在目录（exe
从 cwd 读数据文件与 inputs-dac.txt、往 cwd 写输出）→ 运行 DFH_DAC.exe →
检查 error.txt（非空即失败）→ 把 EPHEMERIDES_DAC.TXT 复制回
``tests/dfh/fixtures/`` 并记录元数据 JSON。生成的 inputs-dac.txt 用后删除；
exe 自己管理的输出文件（EPHEMERIDES_DAC.TXT、error.txt）不动。

用法（在仓库根目录，.venv 的 Python）::

    python scripts/generate_dfh_golden.py minimal   # 预报单档
    python scripts/generate_dfh_golden.py design    # 设计算例全部
    python scripts/generate_dfh_golden.py dro       # 设计算例单个
    python scripts/generate_dfh_golden.py control   # 控制算例全部
    python scripts/generate_dfh_golden.py all       # 预报三档 + 设计算例全部

三档定义（初值/历元/弧长共用）：
- minimal：全开关关，仅地月质点引力；
- mid：+太阳/大行星第三体、地月非球形 10×10；
- high：mid 基础上 +光压炮弹档（solar_radiation=1，面质比取 dyb[0]）、
  +相对论、+地球潮汐。ECOM 光压与非球形×大天体耦合项属 #253，不开。

初值取 DFH 样例 EPHEMERIDES_DAC.TXT 首行（2024-01-01 00:00:00 UTC 的
L1 Halo 状态）。注意 ``format_inputs_propagate`` 按 MATLAB ``%g`` 语义
（6 位有效数字）渲染初始状态——DFH 实际读入的是舍入后的值，因此元数据
记录的 initial_state 从生成的 inputs-dac.txt 回读，e2m2e 侧传播必须用
回读值才能保证两侧初值逐位一致。

控制算例（功能码 2）：输入星历取设计档 NRHO 输出（先跑
``nrho_l2_north`` 或 ``design``），理论模型 2×2、实际模型 10×10、光压
球模型（ECOM 属 #253 不开）、关耦合项；三种控制模式（宽松/严格/特征
点）各一例。DFH 的蒙特卡洛随机数不可控，黄金对比对象是统计特征（总
Δv 均值、失败次数）而非逐样本值。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.dfh_inputs_dac import (
    format_inputs_control,
    format_inputs_design,
    format_inputs_propagate,
    write_inputs_dac,
)

ORBIT_DIR = Path(
    r"C:\Users\ouyangjiahong\Downloads\Compressed\orbit-design-module"
    r"\CislunarOrbitPack\orbit"
)
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "dfh" / "fixtures"
META_PATH = FIXTURE_DIR / "golden_meta.json"

# DFH 样例星历首行（2024-01-01 00:00:00 UTC，L1 Halo；位置 km、速度 m/s）
EPOCH = [2024, 1, 1, 0, 0, 0.0]
DURATION_DAY = 7.0
OUTPUT_STEP_SEC = 3600.0
INITIAL_STATE = [
    -437418.540296398744,
    154148.054367489938,
    132680.644840284513,
    -402.366323971841,
    -786.631509834248,
    -407.081341514286,
]
DYB = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_ALL_OFF = {
    "sun_body": 0,
    "planets": 0,
    "earth_nonspherical": 0,
    "moon_nonspherical": 0,
    "solar_radiation": 0,
    "atmosphere": 0,
    "relativity": 0,
    "tide": 0,
    "coupling": 0,
}

TIERS: dict[str, dict] = {
    "minimal": {"perturbation": _ALL_OFF},
    "mid": {
        "perturbation": {
            **_ALL_OFF,
            "sun_body": 1,
            "planets": 1,
            "earth_nonspherical": 1,
            "moon_nonspherical": 1,
        }
    },
    "high": {
        "perturbation": {
            **_ALL_OFF,
            "sun_body": 1,
            "planets": 1,
            "earth_nonspherical": 1,
            "moon_nonspherical": 1,
            "solar_radiation": 1,
            "relativity": 1,
            "tide": 1,
        }
    },
}

EARTH_DEGREE = 10
MOON_DEGREE = 10

# inputs-dac.txt 中初始状态 6 行的位置（功能码 4 布局，0 起始）
_STATE_ROWS = range(267, 273)


def _read_back_initial_state(lines: list[str]) -> list[float]:
    """从生成的 inputs-dac.txt 行列表回读 DFH 实际读入的初始状态。"""
    return [float(lines[i].split("//")[0].split()[0]) for i in _STATE_ROWS]


def _run_dfh(lines: list[str], out_name: str) -> Path:
    """写 inputs-dac.txt → 跑 DFH_DAC.exe → 把 EPHEMERIDES_DAC.TXT 复制为
    ``out_name`` 存入 fixtures，返回落地路径。

    只写/删 inputs-dac.txt；exe 自己管理的输出文件不动。
    """
    inputs_path = ORBIT_DIR / "inputs-dac.txt"
    exe_path = ORBIT_DIR / "DFH_DAC.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"DFH exe 不存在: {exe_path}")

    write_inputs_dac(lines, inputs_path)
    try:
        proc = subprocess.run(
            [str(exe_path)],
            cwd=ORBIT_DIR,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"DFH_DAC.exe 退出码 {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        error_txt = ORBIT_DIR / "error.txt"
        if error_txt.is_file() and error_txt.read_text(encoding="utf-8", errors="replace").strip():
            raise RuntimeError(f"DFH error.txt 非空:\n{error_txt.read_text(errors='replace')}")
        eph_path = ORBIT_DIR / "EPHEMERIDES_DAC.TXT"
        if not eph_path.is_file():
            raise RuntimeError("DFH 未产出 EPHEMERIDES_DAC.TXT")

        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIXTURE_DIR / out_name
        shutil.copyfile(eph_path, out_path)
        n_rows = sum(1 for ln in out_path.read_text().splitlines() if ln.strip())
        print(f"  {n_rows} 行 -> {out_path}")
        return out_path
    finally:
        # 只清理本脚本写入的 inputs-dac.txt；exe 的输出文件不动
        inputs_path.unlink(missing_ok=True)


def generate_tier(tier: str) -> dict:
    """生成单档基准，返回该档元数据。"""
    spec = TIERS[tier]
    lines = format_inputs_propagate(
        epoch=EPOCH,
        duration=DURATION_DAY,
        initial_state=INITIAL_STATE,
        perturbation=spec["perturbation"],
        dyb=DYB,
        earth_degree=EARTH_DEGREE,
        moon_degree=MOON_DEGREE,
        output_step=OUTPUT_STEP_SEC,
    )
    state_as_read = _read_back_initial_state(lines)

    print(f"[{tier}]")
    out_path = _run_dfh(lines, f"EPHEMERIDES_DAC_{tier.upper()}.TXT")

    return {
        "fixture": out_path.name,
        "perturbation": spec["perturbation"],
        "earth_degree": EARTH_DEGREE,
        "moon_degree": MOON_DEGREE,
        "dyb": DYB,
        "initial_state": state_as_read,
    }


# 功能码 1 设计基准算例：力模型取 mid 档同款（太阳/大行星第三体 +
# 地月非球形 10×10，其余关），e2m2e 侧经 dfh_perturbation_to_force_config
# 可逐项映射。维持时间 0.1 年，输出步长 3600 s。
DESIGN_PERTURBATION = {
    **_ALL_OFF,
    "sun_body": 1,
    "planets": 1,
    "earth_nonspherical": 1,
    "moon_nonspherical": 1,
}

DESIGN_CASES: dict[str, dict] = {
    "dro": {
        "orbit_type": "DRO",
        "params": {"amplitude": 10000.0, "phase": 0.5001},
        "epoch": EPOCH,
        "duration": 0.1,
        "output_step": OUTPUT_STEP_SEC,
    },
    "halo_l2": {
        "orbit_type": "Halo",
        "params": {"collinear_point": 2, "amplitude": 30000.0, "phase": 0.0},
        "epoch": EPOCH,
        "duration": 0.1,
        "output_step": OUTPUT_STEP_SEC,
    },
    "nrho_l2_north": {
        "orbit_type": "NRHO",
        "params": {"collinear_point": 2, "north_south": 1, "perilune_height": 3500.0, "phase": 0.5},
        "epoch": EPOCH,
        "duration": 0.1,
        "output_step": OUTPUT_STEP_SEC,
    },
}


def generate_design_case(name: str) -> dict:
    """生成单个设计基准（功能码 1），返回该算例元数据。"""
    case = DESIGN_CASES[name]
    lines = format_inputs_design(
        case["orbit_type"],
        **case["params"],
        epoch=case["epoch"],
        duration=case["duration"],
        perturbation=DESIGN_PERTURBATION,
        dyb=DYB,
        earth_degree=EARTH_DEGREE,
        moon_degree=MOON_DEGREE,
        output_step=case["output_step"],
    )
    print(f"[design:{name}]")
    out_path = _run_dfh(lines, f"EPHEMERIDES_DESIGN_{name.upper()}.TXT")
    return {
        "fixture": out_path.name,
        "orbit_type": case["orbit_type"],
        "params": case["params"],
        "epoch": case["epoch"],
        "duration": case["duration"],
        "output_step": case["output_step"],
        "perturbation": DESIGN_PERTURBATION,
        "earth_degree": EARTH_DEGREE,
        "moon_degree": MOON_DEGREE,
        "dyb": DYB,
    }


# 功能码 2 控制基准算例。
# 力模型双配置（对齐 e2m2e dfh/control_orbit 默认）：理论 2×2、实际 10×10，
# 光压球模型（ECOM 属 #253 不开）、关耦合项；其余开关与 MATLAB 默认一致。
CONTROL_THEORY_PERT = {
    **_ALL_OFF,
    "sun_body": 1,
    "planets": 1,
    "earth_nonspherical": 1,
    "moon_nonspherical": 1,
    "solar_radiation": 1,
    "tide": 1,
}
CONTROL_REAL_PERT = dict(CONTROL_THEORY_PERT)

#: 控制算例：三种控制模式，短弧段小样本（验证链路 + 统计对比够用）。
#: mode 3（特征点）用 Halo L2 星历 + is_nrho=0（NRHO 输入在双模型
#: 2×2 vs 10×10 下实测 DFH 特征点全部样本失败——NRHO 近月点对月球
#: 引力场阶次差异过于敏感）；mode 1/2 用 NRHO。
CONTROL_CASES: dict[str, dict] = {
    "loose": {
        "control_mode": 1,
        "is_nrho": 0,
        "special_mode": 1,
        "interval_day": 15.0,
        "input_eph": "EPHEMERIDES_DESIGN_NRHO_L2_NORTH_1Y.TXT",
        "input_design": (
            "NRHO",
            {"collinear_point": 2, "north_south": 1, "perilune_height": 3500.0, "phase": 0.5},
        ),
    },
    "tight": {
        "control_mode": 2,
        "is_nrho": 0,
        "special_mode": 1,
        "interval_day": 15.0,
        "input_eph": "EPHEMERIDES_DESIGN_NRHO_L2_NORTH_1Y.TXT",
        "input_design": (
            "NRHO",
            {"collinear_point": 2, "north_south": 1, "perilune_height": 3500.0, "phase": 0.5},
        ),
    },
    "special": {
        "control_mode": 3,
        "is_nrho": 0,
        "special_mode": 2,
        "interval_day": 7.0,
        "special_crossings": 1,
        "input_eph": "EPHEMERIDES_DESIGN_HALO_L2_1Y.TXT",
        "input_design": ("Halo", {"collinear_point": 2, "amplitude": 30000.0, "phase": 0.0}),
    },
}
CONTROL_NUM_CONTROLS = 12
CONTROL_FEEDBACK_ARC_DAY = 28.0
CONTROL_SPECIAL_CROSSINGS = 3
CONTROL_NUM_MONTE_CARLO = 5
CONTROL_OUTPUT_STEP_SEC = 86400.0

#: 受控星历输出文件名（按控制模式，对齐 DFH）
_CONTROL_EPH_NAMES = {
    1: "EPHEMERIDES_LOOSE.TXT",
    2: "EPHEMERIDES_TIGHT.TXT",
    3: "EPHEMERIDES_SPECIAL.TXT",
}


def _run_dfh_control(lines: list[str], prefix: str) -> tuple[Path, Path, Path]:
    """写 inputs-dac.txt → 跑 exe → 收集 SK_STATISTIC/MANEUVERS/受控星历。

    控制模式的输入星历（EPHEMERIDES_DAC.TXT）由调用方预先放好（exe 从
    cwd 读）。返回三个落地路径（sk/maneuvers/受控星历）。
    """
    inputs_path = ORBIT_DIR / "inputs-dac.txt"
    exe_path = ORBIT_DIR / "DFH_DAC.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"DFH exe 不存在: {exe_path}")

    write_inputs_dac(lines, inputs_path)
    # 运行前清除输出文件：特征点模式实测不重写 MANEUVERS.TXT，残留旧文件
    # 会被误复制（判断"本次产出"只能靠运行后文件存在）
    for name in ["SK_STATISTIC.TXT", "MANEUVERS.TXT", *_CONTROL_EPH_NAMES.values()]:
        (ORBIT_DIR / name).unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [str(exe_path)],
            cwd=ORBIT_DIR,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"DFH_DAC.exe 退出码 {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        error_txt = ORBIT_DIR / "error.txt"
        if error_txt.is_file() and error_txt.read_text(encoding="utf-8", errors="replace").strip():
            raise RuntimeError(f"DFH error.txt 非空:\n{error_txt.read_text(errors='replace')}")

        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        # L211（0 起始 210）首值为控制模式，决定受控星历输出文件名
        mode = int(lines[210].split("//")[0].split()[0])
        eph_src = ORBIT_DIR / _CONTROL_EPH_NAMES[mode]
        sk_out = FIXTURE_DIR / f"SK_STATISTIC_{prefix}.TXT"
        man_out = FIXTURE_DIR / f"MANEUVERS_{prefix}.TXT"
        for src, out in [
            (ORBIT_DIR / "SK_STATISTIC.TXT", sk_out),
            (ORBIT_DIR / "MANEUVERS.TXT", man_out),
        ]:
            if not src.is_file():
                # 实测特征点模式（mode 3）DFH 不重写 MANEUVERS.TXT：
                # 不复制旧文件，避免残留污染 fixtures
                if out.is_file():
                    out.unlink()
                print(f"  {src.name} 未产出（跳过）")
                continue
            shutil.copyfile(src, out)
            n_lines = sum(1 for ln in out.read_text().splitlines() if ln.strip())
            print(f"  {out.name}（{n_lines} 行）")
        if not eph_src.is_file():
            raise RuntimeError(f"DFH 未产出受控星历 {eph_src.name}")
        eph_out = FIXTURE_DIR / f"EPHEMERIDES_{prefix}.TXT"
        shutil.copyfile(eph_src, eph_out)
        print(f"  {eph_out.name}")
        return sk_out, man_out, eph_out
    finally:
        inputs_path.unlink(missing_ok=True)


def generate_control_input_ephemeris(name: str) -> Path:
    """生成控制算例的输入星历（1 年弧段，覆盖 12×30 天控制跨度）。

    控制仿真需要在每个控制时刻（含反馈弧末端）读取标称轨道，星历必须
    覆盖整个控制跨度（实测短星历下 DFH 提前终止、控制量全 0）。
    """
    out_path = FIXTURE_DIR / CONTROL_CASES[name]["input_eph"]
    if out_path.is_file():
        return out_path
    orbit_type, params = CONTROL_CASES[name]["input_design"]
    lines = format_inputs_design(
        orbit_type,
        **params,
        epoch=EPOCH,
        duration=1.0,
        perturbation=DESIGN_PERTURBATION,
        dyb=DYB,
        earth_degree=EARTH_DEGREE,
        moon_degree=MOON_DEGREE,
        output_step=3600.0,
    )
    print(f"[control:input-ephemeris:{orbit_type.lower()}]")
    _run_dfh(lines, out_path.name)
    return out_path


def generate_control_case(name: str) -> dict:
    """生成单个控制基准（功能码 2），返回该算例元数据。

    输入星历取 1 年设计输出（``generate_control_input_ephemeris`` 生成/
    复用），理论模型 2×2、实际模型 10×10、光压球模型。
    """
    case = CONTROL_CASES[name]
    src_eph = generate_control_input_ephemeris(name)
    shutil.copyfile(src_eph, ORBIT_DIR / "EPHEMERIDES_DAC.TXT")

    lines = format_inputs_control(
        control_mode=case["control_mode"],
        is_nrho=case["is_nrho"],
        special_mode=case["special_mode"],
        control_interval=case["interval_day"],
        feedback_arc=CONTROL_FEEDBACK_ARC_DAY,
        special_crossings=case.get("special_crossings", CONTROL_SPECIAL_CROSSINGS),
        num_controls=CONTROL_NUM_CONTROLS,
        num_monte_carlo=CONTROL_NUM_MONTE_CARLO,
        output_step=CONTROL_OUTPUT_STEP_SEC,
        perturbation=CONTROL_THEORY_PERT,
        earth_degree=2,
        moon_degree=2,
        real_perturbation=CONTROL_REAL_PERT,
        real_earth_degree=10,
        real_moon_degree=10,
    )
    print(f"[control:{name}]")
    sk_path, man_path, eph_path = _run_dfh_control(lines, name.upper())
    return {
        "fixture_sk": sk_path.name,
        "fixture_maneuvers": man_path.name,
        "fixture_ephemeris": eph_path.name,
        "fixture_input": src_eph.name,
        "control_mode": case["control_mode"],
        "is_nrho": case["is_nrho"],
        "special_mode": case["special_mode"],
        "control_interval_day": case["interval_day"],
        "feedback_arc_day": CONTROL_FEEDBACK_ARC_DAY,
        "special_crossings": case.get("special_crossings", CONTROL_SPECIAL_CROSSINGS),
        "num_controls": CONTROL_NUM_CONTROLS,
        "num_monte_carlo": CONTROL_NUM_MONTE_CARLO,
        "output_step_sec": CONTROL_OUTPUT_STEP_SEC,
        "perturbation": CONTROL_THEORY_PERT,
        "earth_degree": 2,
        "moon_degree": 2,
        "real_perturbation": CONTROL_REAL_PERT,
        "real_earth_degree": 10,
        "real_moon_degree": 10,
    }


def main() -> None:
    modes = list(TIERS) + ["all", "design", *DESIGN_CASES, "control", *CONTROL_CASES]
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg not in modes:
        raise SystemExit(f"未知参数 {arg!r}，可选: {modes}")

    meta: dict = {}
    if META_PATH.is_file():
        # 部分重跑时合并既有元数据
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    if arg in ("all", *TIERS):
        tiers = list(TIERS) if arg == "all" else [arg]
        meta.setdefault("epoch_utc", "2024-01-01T00:00:00")
        meta.setdefault("duration_day", DURATION_DAY)
        meta.setdefault("output_step_sec", OUTPUT_STEP_SEC)
        meta.setdefault("tiers", {})
        for tier in tiers:
            meta["tiers"][tier] = generate_tier(tier)

    if arg in ("all", "design", *DESIGN_CASES):
        cases = list(DESIGN_CASES) if arg in ("all", "design") else [arg]
        meta.setdefault("design", {})
        for name in cases:
            meta["design"][name] = generate_design_case(name)

    if arg in ("all", "control", *CONTROL_CASES):
        cases = list(CONTROL_CASES) if arg in ("all", "control") else [arg]
        meta.setdefault("control", {})
        for name in cases:
            meta["control"][name] = generate_control_case(name)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"元数据 -> {META_PATH}")


if __name__ == "__main__":
    main()
