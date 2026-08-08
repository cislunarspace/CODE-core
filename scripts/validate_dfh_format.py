"""DFH 输入格式规格验证脚本（ADR 0013：仅作开发期交叉参考，不进 CI）。

将 ``format_inputs_control`` 的输出与 ``data/inputs-dac.golden`` 逐行比对，
验证序列化格式规格（行数、字段布局、值精度）。golden 模板的控制段本身是
ControlMode=1 的默认参数，因此用默认参数调用即可复现。

用法::

    python scripts/validate_dfh_format.py

退出码 0 表示全部通过，非 0 表示存在差异。

**ADR 0013 对齐说明**：本脚本验证的是"输入格式规格"（行结构、字段位置、注释
格式），而非算法正确性。算法正确性由解析解 + 物理不变量裁决，不依赖外部软件
输出。本脚本不进 CI，仅作开发期手动交叉参考。
"""

import sys
from pathlib import Path

# 确保 scripts/ 在 sys.path 中，以便导入同目录的 dfh_inputs_dac
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dfh_inputs_dac import format_inputs_control  # noqa: E402

GOLDEN = _SCRIPT_DIR / "data" / "inputs-dac.golden"


def _split_fields(line: str) -> tuple[str, str]:
    """拆成 (值字段, 注释字段)，均去首尾空白。"""
    if "//" not in line:
        return line.strip(), ""
    value, comment = line.split("//", 1)
    return value.strip(), "//" + comment.strip()


def main() -> int:
    if not GOLDEN.exists():
        print(f"错误：golden 文件不存在：{GOLDEN}", file=sys.stderr)
        return 1

    golden = GOLDEN.read_text(encoding="utf-8").splitlines()
    generated = format_inputs_control()

    errors = 0

    # 1. 行数检查
    if len(generated) != len(golden):
        print(
            f"行数不匹配：golden={len(golden)}, generated={len(generated)}",
            file=sys.stderr,
        )
        errors += 1

    # 2. 值字段逐行对照
    value_diffs = []
    for k, (g, e) in enumerate(zip(golden, generated, strict=False), start=1):
        gv, _ = _split_fields(g)
        ev, _ = _split_fields(e)
        if gv != ev:
            value_diffs.append(f"  L{k} VALUE:\n    golden  =[{gv}]\n    gen     =[{ev}]")
    if value_diffs:
        print(
            f"值字段有 {len(value_diffs)} 行差异：",
            file=sys.stderr,
        )
        for d in value_diffs[:10]:
            print(d, file=sys.stderr)
        if len(value_diffs) > 10:
            print(f"  ... 共 {len(value_diffs)} 行", file=sys.stderr)
        errors += 1

    # 3. 注释字段逐行对照
    comment_diffs = []
    for k, (g, e) in enumerate(zip(golden, generated, strict=False), start=1):
        _, gc = _split_fields(g)
        _, ec = _split_fields(e)
        if gc != ec:
            comment_diffs.append(f"  L{k} COMMENT:\n    golden  =[{gc}]\n    gen     =[{ec}]")
    if comment_diffs:
        print(
            f"注释字段有 {len(comment_diffs)} 行差异：",
            file=sys.stderr,
        )
        for d in comment_diffs[:10]:
            print(d, file=sys.stderr)
        if len(comment_diffs) > 10:
            print(f"  ... 共 {len(comment_diffs)} 行", file=sys.stderr)
        errors += 1

    if errors == 0:
        print("全部通过：generated 与 golden 逐行一致。")
        return 0
    else:
        print(f"\n{errors} 项检查未通过。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
