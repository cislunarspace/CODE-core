"""ADR 0037 时间预算门禁：会话级时长审计 + per-test 覆盖标记。

ADR 0037 规定默认套件单用例墙钟上限 10s（call 阶段口径），超预算不得进入
默认 pytest 集。本模块把该预算变成机器强制：会话结束时审计每用例耗时，
凡超过自身预算（默认 10s，或 ``@pytest.mark.time_budget(<seconds>)`` 覆盖值）
的用例令本次运行以非零码失败，并输出可行动的违规报告。

钩子经 tests/conftest.py 再导出生效；纯函数 ``find_budget_violations`` 由
tests/_meta/test_time_budget.py 直接单测。端到端行为（真实子进程，含 xdist）
也在 tests/_meta/test_time_budget.py 覆盖。

机制要点（pytest 9.0.3 + xdist 3.8.0 实证，勿凭直觉改动）：

- **收集在 controller 侧**：xdist 下 controller 也会对每份测试报告触发
  ``pytest_runtest_logreport``（worker 侧同样触发，靠 config 上的
  ``workerinput`` 区分并跳过）；单进程运行时该进程即 controller。
- **时长跨进程保真**：``TestReport.duration`` 经 xdist 序列化原样到达
  （``_report_to_json`` 拷贝整个 ``__dict__``）。
- **覆盖值无法在 controller 读取 item**：xdist 下 controller 不做
  collection（``session.items`` 为空、collection 钩子只在 worker 触发），
  故在 ``pytest_runtest_makereport``（持有 item 的进程）把标记值塞进
  ``report.user_properties`` 随报告过河。副作用：junitxml 若启用会把该
  属性写进 <property>；本仓库不使用 junitxml。
- **退出码**：pytest 9 的 ``wrap_session`` 在 sessionfinish 阶段只认
  ``pytest.exit.Exception``（捕获后取 ``returncode`` 作为退出码）；在
  wrapper 型、trylast 的 ``pytest_sessionfinish`` 里 raise 它，可保证
  在内层 hook（含终端摘要）全部跑完之后才触发。重复审计用挂在 config
  上的 ``audited`` 标志去重（conftest 可能被双载成两份插件实例）。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple

import pytest
from pytest import StashKey

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.nodes import Item

# ADR 0037 决策 1：默认单用例墙钟上限 10 秒。
DEFAULT_BUDGET_S: float = 10.0

# per-test 覆盖标记：@pytest.mark.time_budget(<seconds>)。
MARKER_NAME = "time_budget"

# 覆盖值随报告过河所用的 user_properties 键。
_PROP_KEY = "e2m2e_time_budget_s"


class DurationEntry(NamedTuple):
    """一个用例的审计输入：nodeid、call 阶段实测秒数、覆盖预算（可为 None）。"""

    nodeid: str
    call_s: float
    budget_override_s: float | None


class Violation(NamedTuple):
    """一条违规：nodeid、实测秒数、适用预算。"""

    nodeid: str
    measured_s: float
    budget_s: float

    @property
    def excess_s(self) -> float:
        return self.measured_s - self.budget_s


def _budget_of(entry: DurationEntry) -> float:
    return entry.budget_override_s if entry.budget_override_s is not None else DEFAULT_BUDGET_S


def find_budget_violations(entries: Iterable[DurationEntry]) -> list[Violation]:
    """纯审计判定：超过自身预算的用例按超额降序返回。

    预算语义是"超过即违规"（严格大于）：恰好压线不判违规。
    """
    violations = [
        Violation(e.nodeid, e.call_s, _budget_of(e)) for e in entries if e.call_s > _budget_of(e)
    ]
    violations.sort(key=lambda v: v.excess_s, reverse=True)
    return violations


def _marker_seconds(item: Item) -> float | None:
    """读取用例上的覆盖标记；标记存在但不可用视为用法错误。"""
    marker = item.get_closest_marker(MARKER_NAME)
    if marker is None:
        return None
    if len(marker.args) != 1 or marker.kwargs:
        raise pytest.UsageError(
            f"{item.nodeid}: @{pytest.mark}.{MARKER_NAME} 需要恰好一个位置参数"
            f"（秒数），如 @{pytest.mark}.{MARKER_NAME}(30)；不接受关键字参数。"
        )
    seconds = float(marker.args[0])
    if seconds <= 0:
        raise pytest.UsageError(
            f"{item.nodeid}: @{pytest.mark}.{MARKER_NAME} 的秒数必须为正数，"
            f"得到 {marker.args[0]!r}。"
        )
    return seconds


# ---------- pytest 钩子（经 tests/conftest.py 再导出注册） ----------

# controller 侧的会话级配置。conftest 可能被双载成两份模块实例，各自持有
# 全局；审计状态挂在 config 的 stash 上（每会话一份，双载去重靠 audited）。
_config: Config | None = None

_StateKey = StashKey[dict[str, dict]]()


def pytest_configure(config: Config) -> None:
    global _config
    _config = config
    if _StateKey not in config.stash:
        config.stash[_StateKey] = {"entries": {}, "audited": False}


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """标记用法校验（有 item 的进程才会走到这里；xdist 下即 worker）。"""
    for item in items:
        if item.get_closest_marker(MARKER_NAME) is not None:
            _marker_seconds(item)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    override = _marker_seconds(item)
    if override is not None:
        # 随报告过河：controller 侧拿不到 item，只能从 report 读覆盖值。
        report.user_properties = [*report.user_properties, (_PROP_KEY, override)]


def pytest_runtest_logreport(report) -> None:
    if _config is None or hasattr(_config, "workerinput"):
        # worker 侧不入账：controller 会收到同一份报告。
        return
    if report.when != "call":
        return
    override = None
    for key, value in report.user_properties:
        if key == _PROP_KEY:
            override = float(value)
            break
    entries: dict = _config.stash[_StateKey]["entries"]
    entries[report.nodeid] = DurationEntry(report.nodeid, float(report.duration), override)


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_sessionfinish(session, exitstatus):
    # pluggy 1.6：非 firstresult 钩子的 wrapper 在 yield 处收到内层结果列表，
    # 无需消费；内层异常会原样在 yield 点抛出，不吞。
    yield
    config = session.config
    if _config is config and not hasattr(config, "workerinput"):
        state = config.stash.get(_StateKey, None)
        if state is not None and not state["audited"]:
            state["audited"] = True
            violations = find_budget_violations(state["entries"].values())
            if violations:
                _report_violations(config, violations)
                raise pytest.exit.Exception(
                    f"ADR 0037 time-budget gate: {len(violations)} test(s) over budget",
                    returncode=1,
                )


def _report_violations(config: Config, violations: list[Violation]) -> None:
    """把违规报告写到终端（会话摘要之后）。"""
    lines = [
        f"{len(violations)} test(s) exceeded their per-test ceiling:",
        *[
            f"  {v.measured_s:8.2f}s  {v.nodeid}  (budget {v.budget_s:g}s, +{v.excess_s:.2f}s)"
            for v in violations
        ],
        "",
        "Remedies (ADR 0037): shrink problem scale into budget (small amplitudes /",
        "short arcs / coarse grids / screening tolerances), or move the test to",
        "scripts/ manual diagnostics or benchmarks.",
        "",
        "If a test is genuinely irreducible, mark it with",
        f"@pytest.mark.{MARKER_NAME}(<seconds>) and a comment citing the reason --",
        "sparingly: overrides smuggle over-budget tests back into the default gate",
        "and defeat the budget.",
    ]
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        # 极端配置（-p no:terminalreporter）下不打详细报告；exit.Exception 的
        # 消息仍由 pytest 写 stderr，退出码不受影响。
        return
    reporter.write_sep("=", "ADR 0037 time-budget gate")
    for line in lines:
        reporter.write_line(line)
