"""端到端测试（移植 MATLAB ``tests/e2e/TestE2EDesignOrbit`` 等）。

用 Python 生成器写出 inputs-dac.txt，在 DFH exe 所在目录调用
``DFH_DAC.exe``，再解析输出，验证：exe 正常退出、星历结构完整、首行历元
匹配、位置量级合理、生成文件被清理。exe 不存在时整组跳过（CI 不跑）。

exe 定位：``DFH_ORBIT_ROOT`` 环境变量优先，其次 MATLAB 封装库的 orbit
目录（``orbit-design-module/CislunarOrbitPack/orbit``，含 JPLEPH 等数据
文件，exe 从 cwd 读取）。不要复制 exe 到临时目录——它依赖同目录数据。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from e2m2e.io import (
    format_inputs_control,
    format_inputs_design,
    format_inputs_propagate,
    read_ephemeris,
    read_maneuvers,
    read_sk_statistic,
    write_inputs_dac,
)

pytestmark = [pytest.mark.slow, pytest.mark.e2e]

EPOCH = [2024, 1, 1, 0, 0, 0.0]
DESIGN_PERTURBATION = {
    "sun_body": 1,
    "planets": 1,
    "earth_nonspherical": 1,
    "moon_nonspherical": 1,
    "solar_radiation": 2,
    "atmosphere": 0,
    "relativity": 0,
    "tide": 1,
    "coupling": 1,
}
DYB = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _default_orbit_root() -> Path | None:
    """MATLAB 封装库的 orbit 目录（含 DFH_DAC.exe 与数据文件）。"""
    candidates = [
        Path.home() / "Downloads/Compressed/orbit-design-module/CislunarOrbitPack/orbit",
        Path(
            r"C:\Users\ouyangjiahong\Downloads\Compressed\orbit-design-module\CislunarOrbitPack\orbit"
        ),
    ]
    for c in candidates:
        if (c / "DFH_DAC.exe").is_file():
            return c
    return None


@pytest.fixture(scope="module")
def orbit_root() -> Path:
    """定位 exe 目录；找不到则跳过 e2e 组。"""
    env = os.environ.get("DFH_ORBIT_ROOT")
    root = Path(env) if env else _default_orbit_root()
    if root is None or not (root / "DFH_DAC.exe").is_file():
        pytest.skip("DFH_DAC.exe 未找到；设置 DFH_ORBIT_ROOT 或按模块 docstring 放置")
    return root


def _run_exe(orbit_root: Path, lines: list[str], *, expected: list[str]) -> dict[str, Path]:
    """写 inputs-dac.txt → 跑 exe → 校验 error.txt → 返回输出文件路径。"""
    inputs_path = orbit_root / "inputs-dac.txt"
    write_inputs_dac(lines, inputs_path)
    try:
        proc = subprocess.run(
            [str(orbit_root / "DFH_DAC.exe")],
            cwd=orbit_root,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert proc.returncode == 0, (
            f"DFH_DAC.exe 退出码 {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        error_txt = orbit_root / "error.txt"
        if error_txt.is_file():
            assert not error_txt.read_text(encoding="utf-8", errors="replace").strip(), (
                f"DFH error.txt 非空:\n{error_txt.read_text(errors='replace')}"
            )
        outputs = {}
        for name in expected:
            path = orbit_root / name
            assert path.is_file(), f"DFH 未产出 {name}"
            outputs[name] = path
        return outputs
    finally:
        inputs_path.unlink(missing_ok=True)


def _assert_ephemeris_shape(eph, *, min_rows: int = 1):
    """星历结构：时间六列 + 位置/速度/会合系各三元组。"""
    for name in ("year", "month", "day", "hour", "minute", "second"):
        assert getattr(eph, name).size == len(eph)
    assert eph.position_km.shape == (len(eph), 3)
    assert eph.velocity_mps.shape == (len(eph), 3)
    assert eph.synodic_position.shape == (len(eph), 3)
    assert len(eph) >= min_rows


class TestE2EDesignOrbit:
    """功能码 1：六类轨道设计端到端。"""

    CASES = [
        ("DRO", dict(amplitude=10000.0, phase=0.5001)),
        ("NRHO", dict(collinear_point=2, north_south=1, perilune_height=3500.0)),
        ("Halo", dict(collinear_point=2, amplitude=30000.0, phase=0.0)),
        (
            "Lissajous",
            dict(
                collinear_point=2,
                amplitude_in=2500.0,
                amplitude_out=7500.0,
                phase_in=0.01,
                phase_out=0.55,
            ),
        ),
        ("L4", dict(amplitude_in=8000.0, amplitude_out=6000.0, phase_in=0.0, phase_out=0.0)),
        ("L5", dict(amplitude_in=8000.0, amplitude_out=6000.0, phase_in=0.0, phase_out=0.0)),
    ]

    @pytest.mark.parametrize(("orbit_type", "params"), CASES, ids=[c[0] for c in CASES])
    def test_design_by_type(self, orbit_root, orbit_type, params):
        lines = format_inputs_design(
            orbit_type,
            **params,
            epoch=EPOCH,
            duration=0.1,
            output_step=3600.0,
            perturbation=DESIGN_PERTURBATION,
            dyb=DYB,
        )
        outputs = _run_exe(orbit_root, lines, expected=["EPHEMERIDES_DAC.TXT"])
        eph = read_ephemeris(outputs["EPHEMERIDES_DAC.TXT"])

        _assert_ephemeris_shape(eph)
        # 起始历元匹配
        assert eph.year[0] == 2024
        assert eph.month[0] == 1
        assert eph.day[0] == 1
        # 位置量级合理（地月空间 1e4~1e6 km）
        pos_norm = np.linalg.norm(eph.position_km[0])
        assert 1e4 < pos_norm < 1e6
        # 清理断言：inputs-dac.txt 与 error.txt 不留存
        assert not (orbit_root / "inputs-dac.txt").exists()
        assert not (orbit_root / "error.txt").exists()


class TestE2EPropagateOrbit:
    """功能码 4：轨道预报端到端。"""

    def test_short_propagation(self, orbit_root):
        duration_day = 5.0
        lines = format_inputs_propagate(
            epoch=EPOCH,
            duration=duration_day,
            initial_state=[384400.0, 0.0, 0.0, 0.0, 1000.0, 0.0],
            output_step=3600.0,
            perturbation=DESIGN_PERTURBATION,
            dyb=DYB,
        )
        outputs = _run_exe(orbit_root, lines, expected=["EPHEMERIDES_DAC.TXT"])
        eph = read_ephemeris(outputs["EPHEMERIDES_DAC.TXT"])

        expected_rows = round(duration_day * 86400 / 3600)  # 120
        _assert_ephemeris_shape(eph, min_rows=int(0.95 * expected_rows))
        assert abs(len(eph) - expected_rows) / expected_rows < 0.05
        # 初值匹配（DFH 以 %g 6 位有效数字读入，容差按输入渲染误差放宽）
        assert eph.position_km[0, 0] == pytest.approx(384400.0, rel=1e-5)
        assert eph.position_km[0, 1] == pytest.approx(0.0, abs=1.0)
        assert eph.year[0] == 2024
        assert eph.month[0] == 1
        assert eph.day[0] == 1
        pos_norm = np.linalg.norm(eph.position_km[0])
        assert 1e4 < pos_norm < 1e6


class TestE2EControlOrbit:
    """功能码 2：轨道保持端到端（ControlMode=1 宽松，最小参数快速验证）。"""

    def test_mode1_loose(self, orbit_root):
        # 先用功能码 1 生成输入标称星历（NRHO，短弧段）
        design_lines = format_inputs_design(
            "NRHO",
            collinear_point=2,
            north_south=1,
            perilune_height=3500.0,
            epoch=EPOCH,
            duration=0.2,
            output_step=3600.0,
            perturbation=DESIGN_PERTURBATION,
            dyb=DYB,
        )
        design_out = _run_exe(orbit_root, design_lines, expected=["EPHEMERIDES_DAC.TXT"])
        nominal = read_ephemeris(design_out["EPHEMERIDES_DAC.TXT"])
        assert len(nominal) > 10, "控制输入星历行数不足"

        control_lines = format_inputs_control(
            control_mode=1,
            is_nrho=0,
            special_mode=1,
            control_interval=5.0,
            feedback_arc=5.0,
            num_controls=2,
            num_monte_carlo=1,
            output_step=86400.0,
            perturbation={**DESIGN_PERTURBATION, "solar_radiation": 1},
            earth_degree=2,
            moon_degree=2,
            real_perturbation={**DESIGN_PERTURBATION, "solar_radiation": 1},
            real_earth_degree=10,
            real_moon_degree=10,
        )
        outputs = _run_exe(
            orbit_root,
            control_lines,
            expected=["SK_STATISTIC.TXT", "MANEUVERS.TXT", "EPHEMERIDES_LOOSE.TXT"],
        )

        sk = read_sk_statistic(outputs["SK_STATISTIC.TXT"])
        assert len(sk) >= 1
        assert sk.rows.shape[1] >= 3
        assert sk.num_failed in (None, 0), "蒙特卡洛不应有失败"

        man = read_maneuvers(outputs["MANEUVERS.TXT"])
        assert len(man) >= 1

        controlled = read_ephemeris(outputs["EPHEMERIDES_LOOSE.TXT"])
        _assert_ephemeris_shape(controlled)
