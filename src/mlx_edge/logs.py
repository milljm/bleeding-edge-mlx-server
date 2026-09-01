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

# Keep-hot embed "ok" and chat warmup "hi" (chat template → 9 tokens).
_RE_CACHE = re.compile(r"^Prompt Cache:", re.I)
_RE_CACHE_ROW = re.compile(r"^-\s+(assistant|user|system):", re.I)
_RE_LOOPBACK_HTTP = re.compile(
    r"(?:^INFO:\s+)?127\.0\.0\.1(?::\d+)?\s+.*\b(?:GET|POST|PUT|DELETE|PATCH)\b",
    re.I,
)
_RE_TINY_PREFILL = re.compile(r"Prompt processing progress:\s*(\d+)\s*/\s*(\d+)", re.I)
_RE_MAX_TOKENS_1 = re.compile(r"\bmax_tokens=1\b")
_RE_EMBED_KEEP_HOT = re.compile(r"\bprompt_tokens=3\b", re.I)
_RE_CHAT_WARM = re.compile(r"\bprompt_tokens=9\b", re.I)
_RE_ONE_TOKEN_CAP = re.compile(r"\bgenerated_tokens=1\b.*\bfinish_reason=length\b", re.I)
_RE_PREFILL_NINE = re.compile(r"Prefill (?:started|progress|completed):.*(?:prompt_tokens=9|tokens=\d+/9\b)", re.I)
_RE_GUI_POLL = re.compile(r"\bGET /v1/(?:progress|logs)(?:/|\?|\s)", re.I)


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


def is_noise(text: str) -> bool:
    """Drop keep-hot / warmup / cache chatter. Real client HTTP is kept."""
    line = text.strip()
    if not line:
        return True
    if _RE_CACHE.match(line) or _RE_CACHE_ROW.match(line):
        return True
    if _RE_LOOPBACK_HTTP.search(line):
        return True
    if _RE_GUI_POLL.search(line):
        return True
    tiny = _RE_TINY_PREFILL.search(line)
    if tiny and int(tiny.group(2)) <= 1:
        return True
    if _RE_MAX_TOKENS_1.search(line):
        return True
    if _RE_CHAT_WARM.search(line):
        return True
    if _RE_EMBED_KEEP_HOT.search(line) and "embed" in line.lower():
        return True
    if _RE_ONE_TOKEN_CAP.search(line):
        return True
    if _RE_PREFILL_NINE.search(line):
        return True
    return False


class LogBuffer:
    def __init__(self, maxlen: int = 3000) -> None:
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._lines: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0

    def append(self, model: str, engine: str, text: str) -> None:
        line = text.rstrip("\n")
        if not line or is_noise(line):
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
