"""inputs-dac.txt 生成器测试：与 golden 模板逐行 diff。

约定（与 MATLAB fmt_inputs_*.m 一致）：重建的功能块之外，各行必须与
golden 模板逐字节一致；功能块内逐行比对数值与注释。
"""

from pathlib import Path

import pytest

import e2m2e.io
from e2m2e.io import format_inputs_design, format_inputs_propagate, write_inputs_dac

GOLDEN = Path(e2m2e.io.__file__).parent / "data" / "inputs-dac.golden"

# golden 中 DRO 设计块的参数（L5-L29）
DRO_PARAMS = {
    "amplitude": 10000,
    "phase": 0.5001,
    "epoch": [2024, 1, 1, 0, 0, 0.0],
    "duration": 0.1,
}


def _golden_lines():
    return GOLDEN.read_text(encoding="utf-8").splitlines()


def _value_tokens(line: str) -> list[float]:
    """``//`` 前的数值词元。"""
    return [float(tok) for tok in line.split("//")[0].split()]


def _comment(line: str) -> str:
    return line.split("//", 1)[1] if "//" in line else ""


class TestDesign:
    def test_line_count(self):
        lines = format_inputs_design("DRO", **DRO_PARAMS)
        assert len(lines) == 245

    def test_non_functional_lines_byte_identical(self):
        golden = _golden_lines()
        lines = format_inputs_design("DRO", **DRO_PARAMS)
        # 重建范围：L1（功能码）、L3（类型码）、L5-L29（DRO 设计块）
        rebuilt = {0, 2, *range(4, 29)}
        for i, (got, want) in enumerate(zip(lines, golden, strict=True)):
            if i in rebuilt:
                continue
            assert got == want, f"L{i + 1} 应与 golden 逐字节一致"

    def test_function_code_and_type_rows(self):
        lines = format_inputs_design("DRO", **DRO_PARAMS)
        assert _value_tokens(lines[0]) == [1]
        assert _value_tokens(lines[2]) == [1]  # DRO 类型编号

    def test_dro_block_values_and_comments(self):
        golden = _golden_lines()
        lines = format_inputs_design("DRO", **DRO_PARAMS)
        for i in range(4, 29):
            assert _value_tokens(lines[i]) == _value_tokens(golden[i]), f"L{i + 1} 数值不一致"
            assert _comment(lines[i]) == _comment(golden[i]), f"L{i + 1} 注释不一致"

    def test_nrho_block_line_count(self):
        lines = format_inputs_design(
            "NRHO",
            collinear_point=1,
            north_south=2,
            perilune_height=5000,
            phase=0.5,
            epoch=[2027, 12, 8, 5, 0, 0.0],
            duration=0.2,
        )
        assert len(lines) == 245
        assert _value_tokens(lines[2]) == [2]  # NRHO 类型编号
        assert _value_tokens(lines[30]) == [1]
        assert _value_tokens(lines[31]) == [2]
        # 说明文档：第 33 行两列——近月点高度 + 初始相位（0.01~0.99）
        assert _value_tokens(lines[32]) == [5000, 0.5]

    def test_bad_orbit_type(self):
        with pytest.raises(ValueError, match="orbit_type"):
            format_inputs_design("X", amplitude=1)

    def test_missing_required_param(self):
        with pytest.raises(ValueError, match="缺少必填参数"):
            format_inputs_design("DRO", amplitude=10000)

    def test_perturbation_override(self):
        lines = format_inputs_design("DRO", **DRO_PARAMS, perturbation={"atmosphere": 1})
        # DRO 块内摄动开关第 6 行（大气）：块起始 idx4 + 4 + 5 = idx13
        assert _value_tokens(lines[13]) == [1]
        golden = _golden_lines()
        assert _value_tokens(golden[13]) == [0]  # golden 中大气开关为 0


class TestPropagate:
    PROP_PARAMS = {
        "epoch": [2024, 1, 1, 0, 0, 0.0],
        "duration": 180.0,
        "initial_state": [
            -437418.540296398744,
            154148.054367489938,
            132680.644840284513,
            -402.366323971841,
            -786.631509834248,
            -407.081341514286,
        ],
    }

    def test_line_count(self):
        lines = format_inputs_propagate(**self.PROP_PARAMS)
        assert len(lines) == 274

    def test_prefix_byte_identical(self):
        golden = _golden_lines()
        lines = format_inputs_propagate(**self.PROP_PARAMS)
        assert _value_tokens(lines[0]) == [4]  # 功能码 4
        # L2-L244 与 golden 逐字节一致（原 L245 的 END 被丢弃）
        assert lines[1:244] == golden[1:244]

    def test_propagate_block_layout(self):
        # 行号布局（0 起始）：244~264 力模型（21 行）、265 历元、266 时长、
        # 267~272 初始状态、273 END——与说明文档 L245~265/266/267/268~273 对应
        lines = format_inputs_propagate(**self.PROP_PARAMS)
        assert lines[273] == "END OF THE INPUT FILE"
        assert _value_tokens(lines[265]) == [2024, 1, 1, 0, 0, 0]  # 起始历元
        assert _value_tokens(lines[266]) == [180]  # 时长（天）
        for j in range(6):
            # 初始状态按 MATLAB %g 渲染（6 位有效数字）
            assert _value_tokens(lines[267 + j]) == pytest.approx(
                [float(f"{self.PROP_PARAMS['initial_state'][j]:g}")]
            )

    def test_bad_state_size(self):
        with pytest.raises(ValueError, match="initial_state"):
            format_inputs_propagate(
                epoch=[2024, 1, 1, 0, 0, 0], duration=1, initial_state=[1, 2, 3]
            )


class TestWrite:
    def test_crlf_utf8_roundtrip(self, tmp_path):
        lines = format_inputs_design("DRO", **DRO_PARAMS)
        out = tmp_path / "inputs-dac.txt"
        write_inputs_dac(lines, out)
        raw = out.read_bytes()
        assert b"\r\n" in raw
        assert raw.decode("utf-8").splitlines() == lines


class TestControl:
    """功能码 2（轨道控制）段生成：默认参数与 golden 控制段逐值一致。"""

    def test_line_count(self):
        lines = e2m2e.io.format_inputs_control()
        assert len(lines) == 245

    def test_function_code_row(self):
        lines = e2m2e.io.format_inputs_control()
        assert _value_tokens(lines[0]) == [2]

    def test_non_control_lines_byte_identical(self):
        lines = e2m2e.io.format_inputs_control()
        golden = _golden_lines()
        # 控制段 L169-216（0 起始 168-215）之外逐字节保留
        assert lines[:168] == golden[:168]
        assert lines[216:] == golden[216:]

    def test_control_block_matches_golden_defaults(self):
        """默认参数生成的 48 行控制段与 golden 逐值一致。"""
        lines = e2m2e.io.format_inputs_control()
        golden = _golden_lines()
        for i in range(168, 216):
            if i == 168:  # L169 为 #### 分隔行
                continue
            assert _value_tokens(lines[i]) == pytest.approx(
                _value_tokens(golden[i])
            ), f"L{i + 1} 数值不一致"

    def test_parameter_override(self):
        lines = e2m2e.io.format_inputs_control(
            control_mode=3,
            is_nrho=1,
            special_mode=2,
            control_interval=7.0,
            num_controls=12,
            num_monte_carlo=100,
            thrust_min=0.05,
        )
        # L211：控制模式 + NRHO 标志；L213 间隔；L216 控制/蒙特卡洛次数
        assert _value_tokens(lines[210]) == [3, 1]
        assert _value_tokens(lines[212]) == [7.0]
        assert _value_tokens(lines[215]) == [12, 100, 86400.0]
        assert _value_tokens(lines[207]) == [0.05]

    def test_perturbation_override(self):
        lines = e2m2e.io.format_inputs_control(
            perturbation={"solar_radiation": 1, "coupling": 0},
            real_perturbation={"solar_radiation": 1, "coupling": 0},
        )
        # 理论/实际模型光压开关（L172/L181 的 solar_radiation 行）
        assert _value_tokens(lines[171]) == [1]
        assert _value_tokens(lines[180]) == [1]
