"""文档引号口径门禁的自测（AGENTS.md 引号条款，issue #777）。

守护 ``scripts/check_doc_quotes.py``：合成违规文本要能按 ``相对路径:行号: 标签``
逐条定位，弯引号不得误报，豁免只覆盖 AGENTS.md 的规则正文（历史 ADR 不豁免），
整仓当前态必须过门（``main()`` 返回 0）。fixture 中的被禁字符一律用 ``\\u``
转义构造，本文件自身也在门禁扫描范围内。``scripts/`` 不是包，故用 ``importlib``
按文件路径加载该脚本（唯一等价手段）。
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.aux

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_doc_quotes.py"

CORNER_OPEN = "\u300c"
CORNER_CLOSE = "\u300d"
CORNER_SINGLE_OPEN = "\u300e"
CORNER_SINGLE_CLOSE = "\u300f"
CURVED_OPEN = "\u201c"
CURVED_CLOSE = "\u201d"
CURVED_SINGLE_OPEN = "\u2018"
CURVED_SINGLE_CLOSE = "\u2019"


def _load_gate():
    spec = importlib.util.spec_from_file_location("check_doc_quotes", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GATE = _load_gate()


def test_clean_text_reports_nothing():
    text = (
        f"中文引用词用弯引号：{CURVED_OPEN}正文写作约定{CURVED_CLOSE}，"
        f"嵌套用{CURVED_SINGLE_OPEN}内层{CURVED_SINGLE_CLOSE}。\n"
        'ASCII 直引号 "quoted" 与 ``code`` 也不是被禁字符。\n'
    )
    assert _GATE.violations_for("docs/clean.md", text) == []


def test_reports_corner_quotes_with_line_numbers():
    text = (
        "第一行正常\n"
        f"直角引号：{CORNER_OPEN}引用{CORNER_CLOSE}\n"
        f"直角单引号：{CORNER_SINGLE_OPEN}引用{CORNER_SINGLE_CLOSE}\n"
        f"弯引号行不应上报：{CURVED_OPEN}弯{CURVED_CLOSE}\n"
    )
    assert _GATE.violations_for("docs/x.md", text) == [
        f"docs/x.md:2: 直角引号 直角引号：{CORNER_OPEN}引用{CORNER_CLOSE}",
        f"docs/x.md:3: 直角单引号 直角单引号：{CORNER_SINGLE_OPEN}引用{CORNER_SINGLE_CLOSE}",
    ]


def test_curved_quotes_pass_everywhere():
    text = (
        f"弯引号{CURVED_OPEN}甲{CURVED_CLOSE}、"
        f"弯单引号{CURVED_SINGLE_OPEN}乙{CURVED_SINGLE_CLOSE}\n"
    )
    rels = ("docs/clean.md", "AGENTS.md", "docs/adr/0048-ephemeris-datum-gm-pairing.md")
    for rel in rels:
        assert _GATE.violations_for(rel, text) == []


def test_only_agents_rule_text_is_exempt():
    corner_line = (
        f"规则：{CORNER_OPEN}元引用{CORNER_CLOSE}与{CORNER_SINGLE_OPEN}单{CORNER_SINGLE_CLOSE}\n"
    )
    expected = [f"{CORNER_OPEN}元引用{CORNER_CLOSE}与{CORNER_SINGLE_OPEN}单{CORNER_SINGLE_CLOSE}"]

    # AGENTS.md 规则正文的元引用放行。
    assert _GATE.violations_for("AGENTS.md", corner_line) == []

    # 历史 ADR 与其它文档不再豁免。
    for rel in (
        "docs/adr/0046-contribution-workflow.md",
        "docs/adr/0048-ephemeris-datum-gm-pairing.md",
        "docs/adr/0055-acceptance-oracle-taxonomy.md",
        "docs/adr/0099-new-decision.md",
        "README.md",
    ):
        assert _GATE.violations_for(rel, corner_line) == [
            f"{rel}:1: 直角引号、直角单引号 规则：{expected[0]}"
        ]


def test_repo_tree_passes_gate():
    assert _GATE.main() == 0


def test_scan_surface_matches_policy():
    """扫描面须覆盖条款点名的文档与代码注释：漏登记会让条款静默失效。"""
    rels = {p.relative_to(_REPO_ROOT).as_posix() for p in _GATE.scan_paths()}
    for required in (
        "AGENTS.md",
        "NOTICE",
        ".out-of-scope/stability-bifurcation-analysis.md",
        "docs/adr/0048-ephemeris-datum-gm-pairing.md",
        "docs/tutorials/orbit-catalog.rst",
        "e2m2e/data/kernels/manager.py",
        "tests/data/kernels/test_spice_manager.py",
        "scripts/check_deleted_dir_refs.py",
        "crates/e2m2e-spice/src/lib.rs",
        "crates/e2m2e-spice/README.md",
        "examples/main_propagate.py",
    ):
        assert required in rels, f"门禁扫描面缺 {required}"
