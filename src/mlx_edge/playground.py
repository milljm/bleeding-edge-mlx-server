"""In-RAM Playground transcript. One rolling thread shared by every model.

Lives with the gateway process — a browser reload keeps it, quitting Edge
drops it. Never written to disk.
"""

from __future__ import annotations

import threading
from typing import Any

MAX_TURNS = 200


def _clean_turn(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role") or "").strip()
    if role not in {"user", "assistant"}:
        return None
    text = str(raw.get("text") or "")
    thinking = str(raw.get("thinking") or "")
    out: dict[str, str] = {"role": role, "text": text}
    if thinking:
        out["thinking"] = thinking
    return out


class PlaygroundStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: list[dict[str, str]] = []

    def get(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._turns)

    def put(self, turns: list[Any]) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
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
