"""Ring buffer of engine logs the GUI and other apps can tail."""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any

LEVEL_ERROR = re.compile(r"\b(error|exception|traceback|failed|fail)\b", re.I)
LEVEL_WARN = re.compile(r"\b(warn(ing)?)\b", re.I)
LEVEL_PROGRESS = re.compile(
    r"(prompt processing|prefill|decode progress|keepalive|request completed)",
    re.I,
)
LEVEL_HTTP = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|OPTIONS)\b.*\bHTTP/\d")


def classify(text: str) -> str:
    if LEVEL_ERROR.search(text):
        return "error"
    if LEVEL_WARN.search(text):
        return "warn"
    if LEVEL_PROGRESS.search(text):
        return "progress"
    if LEVEL_HTTP.search(text):
        return "http"
    return "info"


class LogBuffer:
    def __init__(self, maxlen: int = 3000) -> None:
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._lines: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0

    def append(self, model: str, engine: str, text: str) -> None:
        line = text.rstrip("\n")
        if not line:
            return
        row = {
            "seq": 0,
            "ts": time.time(),
            "model": model,
            "engine": engine,
            "level": classify(line),
            "text": line,
        }
        with self._cv:
            self._seq += 1
            row["seq"] = self._seq
            self._lines.append(row)
            self._cv.notify_all()

    def snapshot(self, model: str | None = None, after: int = 0, limit: int = 500) -> dict[str, Any]:
        with self._lock:
            rows = list(self._lines)
            seq = self._seq
        if after:
            rows = [row for row in rows if int(row["seq"]) > after]
        if model:
            needle = model.strip().lower()
            rows = [row for row in rows if needle in str(row.get("model") or "").lower()]
        if limit and len(rows) > limit:
            rows = rows[-limit:]
        return {"object": "edge.logs", "seq": seq, "lines": rows}

    def wait(self, seq: int, timeout: float) -> int:
        with self._cv:
            if self._seq != seq:
                return self._seq
            self._cv.wait(timeout)
            return self._seq

    def seq(self) -> int:
        with self._lock:
            return self._seq

    def clear(self) -> None:
        with self._cv:
            self._lines.clear()
            self._seq += 1
            self._cv.notify_all()
