"""Per-model prompt-processing progress.

OpenAI `/v1/chat/completions` stays untouched. This is the Edge-specific
snapshot the rest of your stack can poll or subscribe to while a prompt is
prefilled (the slow part) and while tokens decode.

GET /v1/progress          JSON snapshot
GET /v1/progress/stream   SSE of the same object as it changes
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

SCHEMA_VERSION = 1
OBJECT = "edge.progress"

PREFILL_RATIO = re.compile(
    r"(?:Prompt processing progress|Prefill progress):\s*(?:.*?tokens=)?(\d+)\s*/\s*(\d+)",
    re.I,
)
KEEPALIVE = re.compile(r":\s*keepalive\s+(\d+)\s*/\s*(\d+)", re.I)
PREFILL_STARTED = re.compile(r"Prefill started:.*?prompt_tokens=(\d+)", re.I)
PREFILL_DONE = re.compile(
    r"Prefill completed:.*?prompt_tokens=(\d+)(?:.*?cached_tokens=(\d+))?(?:.*?rate=([\d.]+))?",
    re.I,
)
DECODE_STARTED = re.compile(r"Decode started:", re.I)
DECODE_TOKENS = re.compile(r"Decode progress:.*?generated_tokens=(\d+)", re.I)
DECODE_DONE = re.compile(r"Decode completed:.*?generated_tokens=(\d+)", re.I)
REQUEST_DONE = re.compile(r"Request completed:", re.I)

LOAD_PCT = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")
LOAD_FETCH = re.compile(r"Fetching\s+(\d+)\s*/\s*(\d+)", re.I)
LOAD_BYTES = re.compile(
    r"(\d+(?:\.\d+)?)\s*([KMGT]i?B)\s*/\s*(\d+(?:\.\d+)?)\s*([KMGT]i?B)",
    re.I,
)
_BYTE_UNITS = {
    "B": 1.0,
    "KB": 1e3,
    "MB": 1e6,
    "GB": 1e9,
    "TB": 1e12,
    "KIB": 1024.0,
    "MIB": 1024.0**2,
    "GIB": 1024.0**3,
    "TIB": 1024.0**4,
}


def now() -> float:
    return time.time()


def _ratio(processed: float, total: float | None) -> float:
    if not total or total <= 0:
        return 0.0
    return round(min(1.0, max(0.0, float(processed) / float(total))), 4)


def row_progress(row: dict[str, Any]) -> float:
    """0.0 idle/unknown → 1.0 when prefill is done and tokens are decoding."""
    phase = row.get("phase")
    if phase == "loading":
        value = row.get("progress")
        return float(value) if isinstance(value, (int, float)) else 0.0
    if phase in {"decode", "done"}:
        return 1.0
    if phase == "error":
        prompt = row.get("prompt") or {}
        value = prompt.get("ratio")
        return float(value) if isinstance(value, (int, float)) else 0.0
    if phase == "prefill":
        prompt = row.get("prompt") or {}
        value = prompt.get("ratio")
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0
    return 0.0


def _tps(processed: int, started_at: float | None, at: float) -> float | None:
    if started_at is None or processed <= 0:
        return None
    elapsed = at - started_at
    if elapsed <= 0:
        return None
    return round(processed / elapsed, 1)


def empty_prompt() -> dict[str, Any]:
    return {
        "processed_tokens": 0,
        "total_tokens": None,
        "ratio": 0.0,
        "cached_tokens": None,
        "started_at": None,
        "updated_at": None,
        "tokens_per_second": None,
    }


def empty_generation() -> dict[str, Any]:
    return {
        "tokens": 0,
        "started_at": None,
        "updated_at": None,
        "tokens_per_second": None,
    }


def idle_row(model_id: str, engine: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "engine": engine,
        "phase": "idle",
        "status": "ready",
        "stream": None,
        "progress": 0.0,
        "prompt": empty_prompt(),
        "generation": empty_generation(),
        "error": None,
        "in_flight": 0,
    }


def parse_progress_text(text: str) -> dict[str, Any] | None:
    """Pick a progress event out of a log line or an SSE frame."""
    if not text:
        return None
    match = KEEPALIVE.search(text) or PREFILL_RATIO.search(text)
    if match:
        processed = int(match.group(1))
        total = int(match.group(2))
        return {"kind": "prefill", "processed": processed, "total": total}
    match = PREFILL_STARTED.search(text)
    if match:
        return {"kind": "prefill_start", "total": int(match.group(1))}
    match = PREFILL_DONE.search(text)
    if match:
        cached = int(match.group(2)) if match.group(2) else None
        tps = float(match.group(3)) if match.group(3) else None
        return {
            "kind": "prefill_done",
            "processed": int(match.group(1)),
            "total": int(match.group(1)),
            "cached": cached,
            "tps": tps,
        }
    if DECODE_STARTED.search(text):
        return {"kind": "decode_start"}
    match = DECODE_TOKENS.search(text)
    if match:
        return {"kind": "decode", "tokens": int(match.group(1))}
    match = DECODE_DONE.search(text)
    if match:
        return {"kind": "decode_done", "tokens": int(match.group(1))}
    if REQUEST_DONE.search(text):
        return {"kind": "done"}
    return None


def _to_bytes(value: float, unit: str) -> float:
    return value * _BYTE_UNITS.get(unit.upper(), 1.0)


def parse_load_text(text: str) -> dict[str, Any] | None:
    """Percent / file / byte progress from a child while Serve is in flight."""
    if not text:
        return None
    match = LOAD_FETCH.search(text)
    if match:
        processed = int(match.group(1))
        total = int(match.group(2))
        return {"kind": "load", "processed": processed, "total": total}
    match = LOAD_BYTES.search(text)
    if match:
        processed = _to_bytes(float(match.group(1)), match.group(2))
        total = _to_bytes(float(match.group(3)), match.group(4))
        if total > 0:
            return {"kind": "load", "processed": processed, "total": total}
    match = LOAD_PCT.search(text)
    if match:
        pct = float(match.group(1))
        if 0.0 <= pct <= 100.0:
            return {"kind": "load", "ratio": pct / 100.0}
    return None


def parse_sse_event(event: str) -> dict[str, Any] | None:
    """Interpret one SSE frame (comments + data lines)."""
    comment_hit = parse_progress_text(event)
    if comment_hit:
        return comment_hit
    payload = None
    for raw_line in event.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return {"kind": "done"}
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
    content = delta.get("content") if isinstance(delta, dict) else None
    finish = choice.get("finish_reason")
    if content:
        return {"kind": "decode_delta", "text": str(content), "finish": finish}
    if finish:
        return {"kind": "done"}
    return None


def _ids_match(left: str, right: str) -> bool:
    a = left.strip().replace("\\", "/").rstrip("/").lower()
    b = right.strip().replace("\\", "/").rstrip("/").lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if a.endswith("/" + b) or b.endswith("/" + a):
        return True
    return a.split("/")[-1] == b.split("/")[-1]


class ProgressTracker:
    def __init__(self, linger: float = 1.5) -> None:
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._models: dict[str, dict[str, Any]] = {}
        self._seq = 0
        self._linger = linger
        self._timers: dict[str, threading.Timer] = {}

    def ensure(self, model_id: str, engine: str) -> None:
        with self._cv:
            row = self._models.get(model_id)
            if row is None:
                self._models[model_id] = idle_row(model_id, engine)
                self._bump()
            elif row.get("engine") != engine:
                row["engine"] = engine
                self._bump()

    def drop(self, model_id: str) -> None:
        with self._cv:
            self._cancel_timer(model_id)
            if model_id in self._models:
                self._models.pop(model_id, None)
                self._bump()

    def begin(self, model_id: str, engine: str, stream: bool | None = None) -> None:
        at = now()
        with self._cv:
            self._cancel_timer(model_id)
            prev = self._models.get(model_id)
            inflight = int((prev or {}).get("in_flight") or 0) + 1
            row = idle_row(model_id, engine)
            row["in_flight"] = inflight
            row["phase"] = "prefill"
            row["status"] = "processing"
            row["stream"] = bool(stream) if stream is not None else None
            row["prompt"]["started_at"] = at
            row["prompt"]["updated_at"] = at
            self._models[model_id] = row
            self._bump()

    def begin_load(self, model_id: str, engine: str) -> None:
        with self._cv:
            self._cancel_timer(model_id)
            row = idle_row(model_id, engine)
            row["phase"] = "loading"
            row["status"] = "processing"
            row["progress"] = 0.0
            row["in_flight"] = 0
            self._models[model_id] = row
            self._bump()

    def end_load(self, model_id: str) -> None:
        with self._cv:
            row = self._models.get(model_id)
            if not row or row.get("phase") != "loading":
                return
            engine = str(row.get("engine") or "lm")
            self._models[model_id] = idle_row(model_id, engine)
            self._bump()

    def ingest_log(self, model_id: str, engine: str, line: str) -> None:
        loading = False
        with self._lock:
            row = self._models.get(model_id)
            loading = bool(row and row.get("phase") == "loading")
        if loading:
            event = parse_load_text(line)
            if event:
                self.apply(model_id, engine, event)
            return
        event = parse_progress_text(line)
        if event:
            self.apply(model_id, engine, event)

    def ingest_sse(self, model_id: str, buf: bytes) -> bytes:
        text = buf.decode("utf-8", errors="replace")
        parts = text.split("\n\n")
        rest = parts.pop() if parts else ""
        for frame in parts:
            event = parse_sse_event(frame)
            if event:
                engine = self.engine_of(model_id)
                self.apply(model_id, engine, event)
        return rest.encode("utf-8")

    def engine_of(self, model_id: str) -> str:
        with self._lock:
            row = self._models.get(model_id)
            return str(row.get("engine") or "lm") if row else "lm"

    def apply(self, model_id: str, engine: str, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if not kind:
            return
        at = now()
        with self._cv:
            row = self._models.get(model_id)
            if row is None:
                return
            if kind == "load":
                if row.get("phase") != "loading":
                    return
                ratio = event.get("ratio")
                if not isinstance(ratio, (int, float)):
                    ratio = _ratio(event.get("processed") or 0, event.get("total"))
                prev = float(row.get("progress") or 0.0)
                row["progress"] = max(prev, min(1.0, float(ratio)))
                row["status"] = "processing"
                row["engine"] = engine or row.get("engine") or "lm"
                self._bump()
                return
            if not row.get("in_flight"):
                # Keep-hot / warmup hits the child directly. Those logs must
                # not look like a user generation (stuck "generating" / green
                # embed dot).
                return
            row["engine"] = engine or row.get("engine") or "lm"
            prompt = row["prompt"]
            gen = row["generation"]
            if kind in {"prefill_start", "prefill", "prefill_done"}:
                if row["phase"] in {"idle", "done"}:
                    row["phase"] = "prefill"
                    row["status"] = "processing"
                    prompt["started_at"] = prompt["started_at"] or at
                if kind == "prefill_start":
                    row["phase"] = "prefill"
                    row["status"] = "processing"
                    prompt["started_at"] = prompt["started_at"] or at
                    total = event.get("total")
                    if isinstance(total, int):
                        prompt["total_tokens"] = total
                elif kind == "prefill":
                    row["phase"] = "prefill"
                    row["status"] = "processing"
                    processed = int(event.get("processed") or 0)
                    total = event.get("total")
                    prompt["processed_tokens"] = processed
                    if isinstance(total, int) and total > 0:
                        prompt["total_tokens"] = total
                    prompt["updated_at"] = at
                    prompt["started_at"] = prompt["started_at"] or at
                    prompt["ratio"] = _ratio(processed, prompt["total_tokens"])
                    prompt["tokens_per_second"] = _tps(processed, prompt["started_at"], at)
                    row["progress"] = row_progress(row)
                else:
                    processed = int(event.get("processed") or prompt["processed_tokens"] or 0)
                    total = event.get("total") or processed
                    prompt["processed_tokens"] = processed
                    prompt["total_tokens"] = total
                    prompt["ratio"] = _ratio(processed, total)
                    prompt["updated_at"] = at
                    if event.get("cached") is not None:
                        prompt["cached_tokens"] = event["cached"]
                    if event.get("tps") is not None:
                        prompt["tokens_per_second"] = event["tps"]
                    elif prompt["tokens_per_second"] is None:
                        prompt["tokens_per_second"] = _tps(processed, prompt["started_at"], at)
                    if row["phase"] == "prefill":
                        # Stay in prefill until the first generated token.
                        pass
                    row["progress"] = row_progress(row)
            elif kind in {"decode_start", "decode", "decode_delta"}:
                if row["phase"] != "decode":
                    row["phase"] = "decode"
                    gen["started_at"] = gen["started_at"] or at
                row["status"] = "processing"
                if prompt["total_tokens"] and prompt["processed_tokens"] < prompt["total_tokens"]:
                    prompt["processed_tokens"] = prompt["total_tokens"]
                    prompt["ratio"] = 1.0
                    prompt["updated_at"] = at
                row["progress"] = row_progress(row)
                if kind == "decode":
                    gen["tokens"] = int(event.get("tokens") or gen["tokens"])
                elif kind == "decode_delta":
                    gen["tokens"] = int(gen["tokens"] or 0) + 1
                    if event.get("finish"):
                        self._mark_done(row, at)
                        self._schedule_idle(model_id)
                        self._bump()
                        return
                gen["updated_at"] = at
                gen["started_at"] = gen["started_at"] or at
                gen["tokens_per_second"] = _tps(int(gen["tokens"] or 0), gen["started_at"], at)
            elif kind in {"decode_done", "done"}:
                if kind == "decode_done" and event.get("tokens") is not None:
                    gen["tokens"] = int(event["tokens"])
                    gen["updated_at"] = at
                self._mark_done(row, at)
                self._schedule_idle(model_id)
            elif kind == "error":
                row["phase"] = "error"
                row["status"] = "error"
                row["error"] = str(event.get("message") or "error")
            else:
                return
            self._bump()

    def complete(self, model_id: str) -> None:
        with self._cv:
            row = self._models.get(model_id)
            if not row:
                return
            row["in_flight"] = max(0, int(row.get("in_flight") or 0) - 1)
            if row["in_flight"] > 0:
                return
            self._mark_done(row, now())
            self._schedule_idle(model_id)
            self._bump()

    def fail(self, model_id: str, message: str) -> None:
        with self._cv:
            row = self._models.get(model_id)
            if not row:
                return
            row["in_flight"] = 0
            row["phase"] = "error"
            row["status"] = "error"
            row["error"] = message
            self._schedule_idle(model_id)
            self._bump()

    def cancel(self, model_id: str) -> None:
        """Stop is not a failure — drop back to idle so the generating pulse clears."""
        with self._cv:
            row = self._models.get(model_id)
            if not row:
                return
            if row.get("phase") == "loading":
                return
            engine = str(row.get("engine") or "lm")
            self._cancel_timer(model_id)
            self._models[model_id] = idle_row(model_id, engine)
            self._bump()

    def snapshot(self, needle: str | None = None) -> dict[str, Any]:
        with self._lock:
            rows = list(self._models.values())
        if needle:
            rows = [row for row in rows if _ids_match(str(row.get("id") or ""), needle)]
        active = any(row.get("status") == "processing" for row in rows)
        publics = [self._public(row) for row in rows]
        overall = 0.0
        if active:
            overall = max((float(row.get("progress") or 0.0) for row in publics), default=0.0)
        elif publics:
            overall = max((float(row.get("progress") or 0.0) for row in publics), default=0.0)
            if all(row.get("phase") == "idle" for row in publics):
                overall = 0.0
        return {
            "object": OBJECT,
            "version": SCHEMA_VERSION,
            "generated_at": now(),
            "active": active,
            "progress": round(float(overall), 4),
            "models": publics,
        }

    def wait(self, seq: int, timeout: float) -> int:
        with self._cv:
            if self._seq != seq:
                return self._seq
            self._cv.wait(timeout)
            return self._seq

    def seq(self) -> int:
        with self._lock:
            return self._seq

    def _public(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "engine": row["engine"],
            "phase": row["phase"],
            "status": row["status"],
            "stream": row.get("stream"),
            "progress": row_progress(row),
            "prompt": dict(row["prompt"]),
            "generation": dict(row["generation"]),
            "error": row.get("error"),
        }

    def _mark_done(self, row: dict[str, Any], at: float) -> None:
        row["phase"] = "done"
        row["status"] = "complete"
        prompt = row["prompt"]
        if prompt.get("total_tokens"):
            prompt["processed_tokens"] = prompt["total_tokens"]
            prompt["ratio"] = 1.0
            prompt["updated_at"] = at
        row["progress"] = row_progress(row)

    def _schedule_idle(self, model_id: str) -> None:
        self._cancel_timer(model_id)
        if self._linger <= 0:
            self._to_idle(model_id)
            return

        def fire() -> None:
            with self._cv:
                self._to_idle(model_id)
                self._bump()

        timer = threading.Timer(self._linger, fire)
        timer.daemon = True
        self._timers[model_id] = timer
        timer.start()

    def _to_idle(self, model_id: str) -> None:
        row = self._models.get(model_id)
        if not row:
            return
        engine = str(row.get("engine") or "lm")
        self._models[model_id] = idle_row(model_id, engine)

    def _cancel_timer(self, model_id: str) -> None:
        timer = self._timers.pop(model_id, None)
        if timer is not None:
            timer.cancel()

    def _bump(self) -> None:
        self._seq += 1
        self._cv.notify_all()
