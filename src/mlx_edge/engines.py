"""Catalog of MLX engines this env is allowed to overlay."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Engine:
    id: str
    dist: str
    module: str
    conda: str
    repo: str
    owner_repo: str
    compiled: bool
    server_module: str | None
    branch: str = "main"


ENGINES: dict[str, Engine] = {
    "lm": Engine(
        id="lm",
        dist="mlx-lm",
        module="mlx_lm",
        conda="mlx-lm",
        repo="https://github.com/ml-explore/mlx-lm.git",
        owner_repo="ml-explore/mlx-lm",
        compiled=False,
        server_module="mlx_lm.server",
    ),
    "vlm": Engine(
        id="vlm",
        dist="mlx-vlm",
        module="mlx_vlm",
        conda="mlx-vlm",
        repo="https://github.com/Blaizzy/mlx-vlm.git",
        owner_repo="Blaizzy/mlx-vlm",
        compiled=False,
        server_module="mlx_vlm.server",
    ),
    "audio": Engine(
        id="audio",
        dist="mlx-audio",
        module="mlx_audio",
        conda="mlx-audio",
        repo="https://github.com/Blaizzy/mlx-audio.git",
        owner_repo="Blaizzy/mlx-audio",
        compiled=False,
        server_module=None,
    ),
    "mlx": Engine(
        id="mlx",
        dist="mlx",
        module="mlx",
        conda="mlx",
        repo="https://github.com/ml-explore/mlx.git",
        owner_repo="ml-explore/mlx",
        compiled=True,
        server_module=None,
    ),
}

PYTHON_ENGINES = ("lm", "vlm", "audio")


def get_engine(name: str) -> Engine:
    key = name.lower()
    aliases = {
        "mlx-lm": "lm",
        "mlx_lm": "lm",
        "mlx-vlm": "vlm",
        "mlx_vlm": "vlm",
        "mlx-audio": "audio",
        "mlx_audio": "audio",
        "tts": "audio",
        "stt": "audio",
    }
    key = aliases.get(key, key)
    if key not in ENGINES:
        known = ", ".join(ENGINES)
        raise SystemExit(f"unknown engine {name!r}. choose from: {known}")
    return ENGINES[key]


def resolve_targets(name: str | None) -> list[Engine]:
    if name is None or name in {"all", "engines"}:
        return [ENGINES[k] for k in PYTHON_ENGINES]
    return [get_engine(name)]

# --- Engine feature detection -----------------------------------------------
#
# Edge must not import engine packages in-process (the mlx wheel initializes
# Metal). Feature support is probed with a throwaway `python -c` that reads the
# engine's own metadata, then cached per gateway process. `mlx-edge update` /
# `build` swap engines underneath Edge, so a restart naturally refreshes it.

FEATURE_PROBE_TIMEOUT = 30.0

_FEATURE_PROBES: dict[str, str] = {
    # mlx-vlm: APC (Automatic Prefix Caching) is configured through server
    # settings / env vars, so its knob list is the capability surface.
    "vlm": (
        "import json\n"
        "from mlx_vlm.server import runtime_config as rc\n"
        "names = [k[0] for k in getattr(rc, 'KNOBS', ())]\n"
        "if not names:\n"
        "    spec = getattr(rc, '_KNOB_SPEC', None)\n"
        "    names = list(spec) if isinstance(spec, dict) else []\n"
        "print(json.dumps(sorted(set(names))))\n"
    ),
}

# Map a knob name found in an engine's metadata to an Edge feature id.
_KNOB_FEATURES: dict[str, str] = {"apc_enabled": "apc"}

_features_cache: dict[str, frozenset[str]] = {}


def engine_features(engine_id: str) -> frozenset[str]:
    """Feature ids the installed engine supports (cached; empty when unknown).

    Probes never raise and never import engines in-process: a missing or broken
    engine simply reports no features, so callers degrade gracefully.
    """
    if engine_id in _features_cache:
        return _features_cache[engine_id]
    source = _FEATURE_PROBES.get(engine_id)
    features: frozenset[str] = frozenset()
    if source:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", source],
                capture_output=True,
                text=True,
                timeout=FEATURE_PROBE_TIMEOUT,
            )
            if proc.returncode == 0:
                names = set(json.loads(proc.stdout.strip() or "[]"))
                features = frozenset(_KNOB_FEATURES[n] for n in names if n in _KNOB_FEATURES)
        except (OSError, subprocess.SubprocessError, ValueError):
            features = frozenset()
    _features_cache[engine_id] = features
    return features
