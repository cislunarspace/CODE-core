"""轨道搜索的 tqdm 进度条封装。"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from tqdm.auto import tqdm


class AggregatePbarWithSlot:
    __slots__ = ("_inner", "_lock", "_slot")

    def __init__(self, inner: Any, lock: threading.Lock | None, slot: int) -> None:
        self._inner = inner
        self._lock = lock
        self._slot = slot

    def update(self, n: int = 1) -> None:
        if self._lock is not None:
            with self._lock:
                self._inner.update(n)
        else:
            self._inner.update(n)

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        merged = f"W{self._slot} {s}"
        if self._lock is not None:
            with self._lock:
                self._inner.set_postfix_str(merged, refresh=refresh)
        else:
            self._inner.set_postfix_str(merged, refresh=refresh)


def open_search_progress_bar(total: int, desc: str) -> Any | None:
    if total <= 0:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit="it",
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.2,
    )


def use_multiline_worker_tqdm(n_workers: int) -> bool:
    env = os.environ.get("E2M2E_TQDM_MULTILINE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    if n_workers > 32:
        return False
    return sys.stderr.isatty()


def reset_tqdm_bar(bar: Any, total: int) -> None:
    if hasattr(bar, "reset"):
        bar.reset(total=total)
    else:
        bar.n = 0
        bar.total = total


def open_parallel_worker_progress_bars(n_workers: int, n_alpha: int) -> list[Any]:
    return [
        tqdm(
            total=n_alpha,
            position=i,
            desc=f"W{i}",
            leave=True,
            unit="α",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.1,
        )
        for i in range(n_workers)
    ]

