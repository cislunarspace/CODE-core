"""tools/logging 配置工厂测试。"""

from __future__ import annotations

import logging

from e2m2e.tools.logging import KeyValueFormatter, configure_logging


class TestKeyValueFormatter:
    def test_appends_extras(self):
        formatter = KeyValueFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="correction_iter",
            args=(),
            exc_info=None,
        )
        record.iter = 3  # type: ignore[attr-defined]
        record.error = 1e-8  # type: ignore[attr-defined]
        out = formatter.format(record)
        assert "iter=3" in out
        assert "error=1e-08" in out

    def test_no_extras_plain(self):
        formatter = KeyValueFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="plain",
            args=(),
            exc_info=None,
        )
        assert formatter.format(record) == "plain"


class TestConfigureLogging:
    def test_idempotent(self):
        configure_logging(level="INFO")
        n_handlers = len(logging.getLogger().handlers)
        configure_logging(level="INFO")
        assert len(logging.getLogger().handlers) == n_handlers
