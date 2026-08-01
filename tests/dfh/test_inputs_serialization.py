"""inputs-dac.txt 序列化测试（移植 MATLAB ``tests/unit`` 与 ``tests/integration``）。

覆盖三个生成器：``format_inputs_design``（六类轨道设计块）、
``format_inputs_propagate``（功能码 4 预报块）、``format_inputs_control``
（功能码 2 控制块）。断言风格照 MATLAB ``TestFmtInputs*`` 与
``Test*Serialization``：行数、功能码/类型码、各参数块逐行取值。

行号映射以说明文档《轨道设计仿真和目标光学定初轨输入输出.md》为准（1 起
始），两处与 MATLAB 版测试的偏差在此说明：

- NRHO 块第 33 行为**两列**（近月点高度 + 初始相位 0.01~0.99），说明文档
  明确如此；MATLAB golden 模板缺相位列、其测试断言 L34=近月点高度（实为
  起始历元）均不符说明文档，不照搬。
- 功能码 4 预报块按 e2m2e 的 274 行布局（说明文档 L245~273 后接 END），
  MATLAB 版多插一行分隔符导致 275 行布局，实测 exe 按定行号读取会错位
  （见 ``io/inputs_dac.py`` docstring），不照搬。
"""

import numpy as np
import pytest

from e2m2e.io import format_inputs_control, format_inputs_design, format_inputs_propagate

EPOCH = [2024, 1, 1, 0, 0, 0.0]

#: 与 MATLAB ``make_design_opts`` 一致的摄动/DYB 默认值
PERTURBATION = {
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
DYB = [0.01, 0, 0, 0, 0, 0, 0, 0, 0]


def _value(line: str) -> list[float]:
    """``//`` 之前的值词元。"""
    return [float(tok) for tok in line.split("//")[0].split()]


def _design_opts(orbit_type: str) -> dict:
    """六类轨道设计的形状参数（值照 MATLAB ``TestDesignSerialization``）。"""
    common = dict(
        epoch=EPOCH,
        duration=0.5,
        output_step=3600.0,
        perturbation=PERTURBATION,
        dyb=DYB,
    )
    if orbit_type == "DRO":
        return dict(orbit_type=orbit_type, amplitude=30000.0, phase=0.0, **common)
    if orbit_type == "NRHO":
        return dict(
            orbit_type=orbit_type,
            collinear_point=2,
            north_south=2,
            perilune_height=5000.0,
            **common,
        )
    if orbit_type == "Halo":
        return dict(
            orbit_type=orbit_type,
            collinear_point=2,
            amplitude=30000.0,
            phase=0.0,
            **common,
        )
    if orbit_type == "Lissajous":
        return dict(
            orbit_type=orbit_type,
            collinear_point=2,
            amplitude_in=2500.0,
            amplitude_out=7500.0,
            phase_in=0.01,
            phase_out=0.55,
            **common,
        )
    # L4 / L5
    return dict(
        orbit_type=orbit_type,
        amplitude_in=8000.0,
        amplitude_out=6000.0,
        phase_in=0.0,
        phase_out=0.0,
        **common,
    )


# =============================================================================
# 设计块序列化（MATLAB TestDesignSerialization + TestFmtInputsDesign）
# =============================================================================
class TestDesignSerialization:
    """六类 OrbitType 各一次：245 行、L1 功能码 1、L3 类型码、块值保真。"""

    #: (类型, 类型码, {说明文档行号(1 起) -> 期望值词元})
    CASES = [
        ("DRO", 1, {5: [30000.0], 6: [0.0]}),
        ("NRHO", 2, {33: [5000.0, 0.5]}),
        ("Halo", 3, {59: [30000.0], 60: [0.0]}),
        ("Lissajous", 4, {86: [2500.0], 87: [7500.0], 88: [0.01], 89: [0.55]}),
        ("L4", 5, {114: [8000.0], 115: [6000.0], 116: [0.0], 117: [0.0]}),
        ("L5", 6, {142: [8000.0], 143: [6000.0], 144: [0.0], 145: [0.0]}),
    ]

    @pytest.mark.parametrize(
        ("orbit_type", "type_code", "value_rows"), CASES, ids=[c[0] for c in CASES]
    )
    def test_six_types(self, orbit_type, type_code, value_rows):
        lines = format_inputs_design(**_design_opts(orbit_type))
        assert len(lines) == 245
        assert _value(lines[0]) == [1], "L1 功能码应为 1"
        assert _value(lines[2]) == [float(type_code)], f"L3 类型码应为 {type_code}"
        for row, expected in value_rows.items():
            np.testing.assert_allclose(
                _value(lines[row - 1]), expected, atol=1e-9, err_msg=f"L{row} 取值"
            )

    def test_halo_block_values(self):
        """Halo 块 L58-L62：共线点/振幅/相位/历元/维持时间与输入一致。"""
        lines = format_inputs_design(**_design_opts("Halo"))
        assert _value(lines[57]) == [2]
        assert _value(lines[58]) == [30000.0]
        assert _value(lines[59]) == [0.0]
        assert _value(lines[60]) == [2024.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        assert _value(lines[61]) == [0.5]

    def test_halo_perturbation_in_block(self):
        """Halo 块公共段 L63=太阳第三体、L67=光压（ECOM=2）。"""
        lines = format_inputs_design(**_design_opts("Halo"))
        assert _value(lines[62]) == [1]
        assert _value(lines[66]) == [2]

    def test_nrho_phase_default(self):
        """NRHO 相位缺省时为 0.5（说明文档取值 0.01~0.99）。"""
        lines = format_inputs_design(
            "NRHO",
            collinear_point=2,
            north_south=2,
            perilune_height=5000.0,
            epoch=EPOCH,
            duration=0.5,
        )
        assert _value(lines[32]) == [5000.0, 0.5]

    def test_perturbation_override(self):
        """摄动覆盖：L14 大气改为 1（golden 为 0）。"""
        lines = format_inputs_design(
            "DRO",
            amplitude=10000.0,
            phase=0.5,
            epoch=EPOCH,
            duration=0.1,
            perturbation={**PERTURBATION, "atmosphere": 1},
        )
        assert _value(lines[13]) == [1.0]


# =============================================================================
# 预报块序列化（MATLAB TestFmtInputsPropagate；按 274 行布局）
# =============================================================================
class TestFmtInputsPropagate:
    PROP_PARAMS = {
        "epoch": [2024, 1, 1, 0, 0, 0.0],
        "duration": 30.0,
        "initial_state": [384400.0, 0.0, 0.0, 0.0, 1000.0, 0.0],
        "perturbation": PERTURBATION,
        "dyb": DYB,
    }

    def test_total_lines(self):
        lines = format_inputs_propagate(**self.PROP_PARAMS)
        assert len(lines) == 274

    def test_function_code(self):
        lines = format_inputs_propagate(**self.PROP_PARAMS)
        assert _value(lines[0]) == [4.0], "L1 功能码应为 4"

    def test_epoch_line(self):
        lines = format_inputs_propagate(**{**self.PROP_PARAMS, "epoch": [2025, 6, 15, 12, 30, 0.0]})
        assert _value(lines[265]) == [2025.0, 6.0, 15.0, 12.0, 30.0, 0.0]
        assert "//" in lines[265], "L266 历元行应含 // 分隔符"

    def test_duration_value(self):
        lines = format_inputs_propagate(**{**self.PROP_PARAMS, "duration": 15.0})
        assert _value(lines[266]) == [15.0]

    def test_initial_state_positions(self):
        """L268-270 初始位置（km）。"""
        lines = format_inputs_propagate(
            **{**self.PROP_PARAMS, "initial_state": [100000.0, 200000.0, 300000.0, 0, 0, 0]}
        )
        assert _value(lines[267]) == [100000.0]
        assert _value(lines[268]) == [200000.0]
        assert _value(lines[269]) == [300000.0]

    def test_initial_state_velocities(self):
        """L271-273 初始速度（m/s）。"""
        lines = format_inputs_propagate(
            **{**self.PROP_PARAMS, "initial_state": [0, 0, 0, 500.0, -300.0, 100.0]}
        )
        assert _value(lines[270]) == [500.0]
        assert _value(lines[271]) == [-300.0]
        assert _value(lines[272]) == [100.0]

    def test_end_marker(self):
        lines = format_inputs_propagate(**self.PROP_PARAMS)
        assert lines[273] == "END OF THE INPUT FILE"

    def test_perturbation_block_values(self):
        """L245=太阳第三体、L250=大气，与覆盖值一致。"""
        lines = format_inputs_propagate(
            **{
                **self.PROP_PARAMS,
                "perturbation": {**PERTURBATION, "sun_body": 0, "atmosphere": 1},
            }
        )
        assert _value(lines[244]) == [0.0]
        assert _value(lines[249]) == [1.0]

    def test_earth_moon_degree(self):
        """L263/L264 阶次数与输入一致。"""
        lines = format_inputs_propagate(
            **{**self.PROP_PARAMS, "earth_degree": 20, "moon_degree": 15}
        )
        assert _value(lines[262]) == [20.0]
        assert _value(lines[263]) == [15.0]


# =============================================================================
# 控制块序列化（MATLAB TestFmtInputsControl）
# =============================================================================
class TestFmtInputsControl:
    def test_line_count_and_function_code(self):
        lines = format_inputs_control()
        assert len(lines) == 245
        assert _value(lines[0]) == [2.0], "L1 功能码应为 2"

    def test_control_mode_row(self):
        """L211 两列：控制模式 + NRHO 标志。"""
        lines = format_inputs_control(control_mode=3, is_nrho=1)
        assert _value(lines[210]) == [3.0, 1.0]

    def test_theory_real_perturbation_rows(self):
        """L170-178 理论模型摄动、L179-187 实际模型摄动，光压可分别覆盖。"""
        lines = format_inputs_control(
            perturbation={**PERTURBATION, "solar_radiation": 1},
            real_perturbation={**PERTURBATION, "solar_radiation": 1},
        )
        # 光压是摄动块第 5 行：L170+4=L174（理论）、L179+4=L183（实际）
        assert _value(lines[173]) == [1.0]
        assert _value(lines[182]) == [1.0]

    def test_monte_carlo_and_interval(self):
        """L213 控制间隔、L216 三值（控制次数/蒙特卡洛/输出间隔）。"""
        lines = format_inputs_control(
            control_interval=7.0, num_controls=12, num_monte_carlo=100, output_step=86400.0
        )
        assert _value(lines[212]) == [7.0]
        assert _value(lines[215]) == [12.0, 100.0, 86400.0]
