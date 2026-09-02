"""In-RAM Playground transcript. One rolling thread shared by every model.

Lives with the gateway process — a browser reload keeps it, quitting Edge
drops it. Never written to disk.
"""

from __future__ import annotations

import threading
from typing import Any

MAX_TURNS = 200


def _clean_turn(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role") or "").strip()
    if role not in {"user", "assistant"}:
        return None
    text = str(raw.get("text") or "")
    thinking = str(raw.get("thinking") or "")
    out: dict[str, Any] = {"role": role, "text": text}
    if thinking:
        out["thinking"] = thinking
    metrics = _clean_metrics(raw.get("metrics"))
    if metrics:
        out["metrics"] = metrics
    return out


def _clean_metrics(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for key in ("ttft", "gen", "tps"):
        try:
            out[key] = float(raw[key])
        except (KeyError, TypeError, ValueError):
            continue
    try:
        out["tokens"] = int(raw["tokens"])
    except (KeyError, TypeError, ValueError):
        pass
    model = str(raw.get("model") or "").strip()
    if model:
        out["model"] = model
    return out or None


class PlaygroundStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: list[dict[str, Any]] = []

    def get(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._turns)

    def put(self, turns: list[Any]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for item in turns:
            row = _clean_turn(item)
            if row is None:
                continue
            cleaned.append(row)
            if len(cleaned) >= MAX_TURNS:
                break
        with self._lock:
            self._turns = cleaned
        return cleaned

    def clear(self) -> None:
        with self._lock:
            self._turns = []
