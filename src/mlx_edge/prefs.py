"""Studio prefs that survive closing Edge (watch dirs + per-model flags)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFS_PATH = Path.home() / ".config" / "mlx-edge" / "studio.json"
ENGINES = {"lm", "vlm", "embed", "tts", "stt"}


def empty_prefs() -> dict[str, Any]:
    return {"watchDirs": [], "flagsByModel": {}, "engineByModel": {}}


def load_prefs(path: Path | None = None) -> dict[str, Any]:
    target = path or PREFS_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return empty_prefs()
    if not isinstance(raw, dict):
        return empty_prefs()
    return _clean(raw)


def save_prefs(data: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = path or PREFS_PATH
    out = _clean(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def _clean(raw: dict[str, Any]) -> dict[str, Any]:
    dirs_raw = raw.get("watchDirs") or raw.get("watch_dirs") or []
    dirs: list[str] = []
    if isinstance(dirs_raw, list):
        for item in dirs_raw:
            text = str(item).strip()
            if text and text not in dirs:
                dirs.append(text)
    flags_raw = raw.get("flagsByModel") or raw.get("flags_by_model") or {}
    flags: dict[str, Any] = {}
    if isinstance(flags_raw, dict):
        for key, value in flags_raw.items():
            name = str(key).strip()
            if name and isinstance(value, dict):
                flags[name] = value
    engines_raw = raw.get("engineByModel") or raw.get("engine_by_model") or {}
    engines: dict[str, str] = {}
    if isinstance(engines_raw, dict):
        for key, value in engines_raw.items():
            name = str(key).strip()
            engine = str(value or "").strip().lower()
            if name and engine in ENGINES:
                engines[name] = engine
    return {"watchDirs": dirs, "flagsByModel": flags, "engineByModel": engines}
