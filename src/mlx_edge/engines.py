"""Catalog of MLX engines this env is allowed to overlay."""

from __future__ import annotations

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
