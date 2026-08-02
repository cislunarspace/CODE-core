"""输入序列化格式规格回归（ADR 0013：验证策略——按定义完成任务）。

``format_inputs_control`` 的输出与 ``data/inputs-dac.golden`` 逐行比对，验证
序列化格式规格（行数、字段布局、值精度）的正确性。golden 模板的控制段本身是
ControlMode=1 的默认参数，因此用默认参数调用即可复现。

**ADR 0013 对齐说明**：本测试验证的是"输入格式规格"（行结构、字段位置、注释
格式），而非算法正确性。算法正确性由解析解 + 物理不变量裁决，不依赖外部软件
输出。DFH 格式仅作开发期交叉参考（ADR 0013 §4）。
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
