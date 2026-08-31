"""ADR 0037 时长门禁：审计逻辑单测 + 端到端子进程验证。

单测部分直接测纯函数（合成时长，不跑真实重计算）；e2e 部分起真实 pytest
子进程（模拟用户跑测试的真实方式），验证违规退出码、违规报告输出、覆盖
标记语义与 xdist 跨 worker 聚合。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from time_budget import DEFAULT_BUDGET_S, DurationEntry, find_budget_violations

pytestmark = pytest.mark.aux

TESTS_DIR = Path(__file__).resolve().parent.parent


def entry(nodeid: str, call_s: float, override: float | None = None) -> DurationEntry:
    return DurationEntry(nodeid=nodeid, call_s=call_s, budget_override_s=override)


class TestFindBudgetViolations:
    def test_under_budget_not_flagged(self):
        assert find_budget_violations([entry("t::a", 9.9)]) == []

    def test_exactly_at_budget_not_flagged(self):
        # 上限语义是"超过即违规"：恰好压线不判违规。
        assert find_budget_violations([entry("t::a", DEFAULT_BUDGET_S)]) == []

    def test_just_over_default_budget_flagged(self):
        violations = find_budget_violations([entry("t::a", DEFAULT_BUDGET_S + 1e-6)])
        assert len(violations) == 1
        assert violations[0].nodeid == "t::a"
        assert violations[0].measured_s == DEFAULT_BUDGET_S + 1e-6
        assert violations[0].budget_s == DEFAULT_BUDGET_S

    def test_override_lowers_threshold(self):
        violations = find_budget_violations([entry("t::a", 2.5, override=2.0)])
        assert len(violations) == 1
        assert violations[0].budget_s == 2.0

    def test_override_raises_threshold(self):
        # ADR 0037 增补：NSGA-II 并行一致性 ~9.6s 不可再压，靠覆盖豁免。
        assert find_budget_violations([entry("t::a", 11.0, override=30.0)]) == []

    def test_override_exceeded_also_flagged(self):
        violations = find_budget_violations([entry("t::a", 31.0, override=30.0)])
        assert len(violations) == 1
        assert violations[0].budget_s == 30.0

    def test_all_violators_listed_sorted_by_excess_desc(self):
        violations = find_budget_violations(
            [
                entry("small", 10.5),
                entry("worst", 25.0),
                entry("ok", 1.0),
                entry("mid", 12.0),
            ]
        )
        assert [v.nodeid for v in violations] == ["worst", "mid", "small"]

    def test_empty_entries_no_violations(self):
        assert find_budget_violations([]) == []


# ---------- 端到端：真实 pytest 子进程 ----------

_CONFTEST = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, r"{tests_dir}")
    pytest_plugins = ["time_budget"]
    """
)


def _run_pytest(
    tmp_path: Path, files: dict[str, str], args: list[str]
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "conftest.py").write_text(_CONFTEST.format(tests_dir=TESTS_DIR), encoding="utf-8")
    for name, content in files.items():
        (tmp_path / name).write_text(textwrap.dedent(content), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestEndToEnd:
    def test_over_budget_fails_run_with_actionable_report(self, tmp_path):
        result = _run_pytest(
            tmp_path,
            {
                "test_x.py": """
                    import time
                    import pytest

                    @pytest.mark.time_budget(0.2)
                    def test_slow():
                        time.sleep(0.6)
                """,
            },
            args=[],
        )
        assert result.returncode == 1, result.stdout + result.stderr
        out = result.stdout + result.stderr
        assert "ADR 0037 time-budget gate" in out
        assert "test_x.py::test_slow" in out
        assert "budget 0.2s" in out
        # 处置指引与覆盖标记提示都在。
        assert "shrink" in out
        assert "time_budget" in out

    def test_within_override_budget_passes_clean(self, tmp_path):
        result = _run_pytest(
            tmp_path,
            {
                "test_x.py": """
                    import time
                    import pytest

                    @pytest.mark.time_budget(5)
                    def test_ok():
                        time.sleep(0.3)
                """,
            },
            args=[],
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "time-budget gate" not in result.stdout + result.stderr

    def test_multiple_violators_all_listed_sorted_by_excess(self, tmp_path):
        result = _run_pytest(
            tmp_path,
            {
                "test_x.py": """
                    import time
                    import pytest

                    @pytest.mark.time_budget(0.1)
                    def test_small():
                        time.sleep(0.3)

                    @pytest.mark.time_budget(0.1)
                    def test_worst():
                        time.sleep(0.7)
                """,
            },
            args=[],
        )
        assert result.returncode == 1, result.stdout + result.stderr
        out = result.stdout + result.stderr
        assert "test_small" in out and "test_worst" in out
        assert out.index("test_worst") < out.index("test_small"), "超额降序"

    def test_xdist_aggregates_violations_across_workers(self, tmp_path):
        result = _run_pytest(
            tmp_path,
            {
                "test_a.py": """
                    import time
                    import pytest

                    @pytest.mark.time_budget(0.1)
                    def test_slow_a():
                        time.sleep(0.4)
                """,
                "test_b.py": """
                    import time
                    import pytest

                    @pytest.mark.time_budget(0.1)
                    def test_slow_b():
                        time.sleep(0.4)
                """,
            },
            args=["-n", "2", "--dist=loadscope"],
        )
        assert result.returncode == 1, result.stdout + result.stderr
        out = result.stdout + result.stderr
        assert "test_a.py::test_slow_a" in out
        assert "test_b.py::test_slow_b" in out

    def test_malformed_marker_is_usage_error(self, tmp_path):
        result = _run_pytest(
            tmp_path,
            {
                "test_x.py": """
                    import pytest

                    @pytest.mark.time_budget
                    def test_bad_marker():
                        pass
                """,
            },
            args=[],
        )
        assert result.returncode != 0, result.stdout + result.stderr
        out = result.stdout + result.stderr
        assert "time_budget" in out
        assert "位置参数" in out
