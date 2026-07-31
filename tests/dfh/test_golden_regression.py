"""golden 回归测试（移植 MATLAB ``tests/regression/TestGoldenSample.m``）。

``format_inputs_control`` 的输出与 ``data/inputs-dac.golden`` 逐行比对：值字段
（``//`` 之前）与注释字段（``//`` 之后）均须一致。golden 模板的控制段本身是
ControlMode=1 的默认参数，因此用默认参数调用即可复现。
"""

from pathlib import Path

import e2m2e.io
from e2m2e.io import format_inputs_control

GOLDEN = Path(e2m2e.io.__file__).parent / "data" / "inputs-dac.golden"


def _split_fields(line: str) -> tuple[str, str]:
    """拆成 (值字段, 注释字段)，均去首尾空白。"""
    if "//" not in line:
        return line.strip(), ""
    value, comment = line.split("//", 1)
    return value.strip(), "//" + comment.strip()


class TestGoldenSample:
    def test_golden_line_count(self):
        golden = GOLDEN.read_text(encoding="utf-8").splitlines()
        generated = format_inputs_control()
        assert len(generated) == len(golden), (
            f"行数不匹配：golden={len(golden)}, generated={len(generated)}"
        )

    def test_golden_value_fields_match(self):
        golden = GOLDEN.read_text(encoding="utf-8").splitlines()
        generated = format_inputs_control()
        diffs = []
        for k, (g, e) in enumerate(zip(golden, generated, strict=True), start=1):
            gv, _ = _split_fields(g)
            ev, _ = _split_fields(e)
            if gv != ev:
                diffs.append(f"L{k} VALUE:\n  golden  =[{gv}]\n  gen     =[{ev}]")
        assert not diffs, f"值字段有 {len(diffs)} 行差异，前几条：\n" + "\n".join(diffs[:5])

    def test_golden_comment_fields_match(self):
        golden = GOLDEN.read_text(encoding="utf-8").splitlines()
        generated = format_inputs_control()
        diffs = []
        for k, (g, e) in enumerate(zip(golden, generated, strict=True), start=1):
            _, gc = _split_fields(g)
            _, ec = _split_fields(e)
            if gc != ec:
                diffs.append(f"L{k} COMMENT:\n  golden  =[{gc}]\n  gen     =[{ec}]")
        assert not diffs, f"注释字段有 {len(diffs)} 行差异，前几条：\n" + "\n".join(diffs[:5])
