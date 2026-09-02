"""In-RAM Playground transcripts. Lives with the gateway process — reload keeps
them, quitting Edge drops them. Never written to disk."""

from __future__ import annotations

import threading
from typing import Any

MAX_TURNS = 200


def key_for(name: str) -> str:
    return name.strip().replace("\\", "/").rstrip("/").split("/")[-1].lower()


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
        self._by_model: dict[str, list[dict[str, str]]] = {}

    def get(self, model: str) -> list[dict[str, str]]:
        key = key_for(model)
        if not key:
            return []
        with self._lock:
            return list(self._by_model.get(key, []))

    def put(self, model: str, turns: list[Any]) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        for item in turns:
            row = _clean_turn(item)
            if row is None:
                continue
            cleaned.append(row)
            if len(cleaned) >= MAX_TURNS:
                break
        key = key_for(model)
        if not key:
            return cleaned
        with self._lock:
            if cleaned:
                self._by_model[key] = cleaned
            else:
                self._by_model.pop(key, None)
        return cleaned

    def clear(self, model: str | None = None) -> None:
        with self._lock:
            if model:
                self._by_model.pop(key_for(model), None)
            else:
                self._by_model.clear()
