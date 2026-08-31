"""In-process registry of hot-loaded mlx-lm / mlx-vlm children."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from mlx_edge.engines import get_engine


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def strip_bind_args(extra: list[str]) -> list[str]:
    """Children always bind 127.0.0.1:ephemeral. Drop competing --host/--port."""
    out: list[str] = []
    skip = False
    for arg in extra:
        if skip:
            skip = False
            continue
        if arg in {"--host", "--port"}:
            skip = True
            continue
        if arg.startswith("--host=") or arg.startswith("--port="):
            continue
        out.append(arg)
    return out


@dataclass
class LoadedModel:
    id: str
    engine: str
    model: str
    port: int
    started_at: float
    proc: subprocess.Popen[bytes] | None = None
    args: list[str] = field(default_factory=list)

    def as_openai(self) -> dict[str, object]:
        return {
            "id": self.model,
            "object": "model",
            "created": int(self.started_at),
            "owned_by": "mlx-vlm" if self.engine == "vlm" else "mlx-lm",
        }


SpawnFn = Callable[[str, str, int, list[str]], subprocess.Popen[bytes] | None]


def default_spawn(engine_id: str, model: str, port: int, extra: list[str]) -> subprocess.Popen[bytes]:
    engine = get_engine(engine_id)
    if not engine.server_module:
        raise RuntimeError(f"{engine.dist} has no server")
    cmd = [
        sys.executable,
        "-m",
        engine.server_module,
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *strip_bind_args(extra),
    ]
    return subprocess.Popen(cmd)


class ModelPool:
    def __init__(self, spawn: SpawnFn | None = None, wait: Callable[[int], None] | None = None) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._spawn = spawn or default_spawn
        self._wait = wait or wait_healthy

    def list(self) -> list[LoadedModel]:
        return sorted(self._models.values(), key=lambda m: m.started_at)

    def resolve(self, name: str | None) -> LoadedModel | None:
        loaded = self.list()
        if not loaded:
            return None
        if not name:
            return loaded[0]
        needle = name.strip()
        if needle in self._models:
            return self._models[needle]
        for item in loaded:
            if item.model == needle or item.id == needle:
                return item
            if item.model.endswith("/" + needle) or item.model.endswith(needle):
                return item
        return None

    def load(self, engine: str, model: str, extra: list[str] | None = None) -> LoadedModel:
        extra = list(extra or [])
        existing = self.resolve(model)
        if existing:
            self.unload(existing.id)
        port = free_port()
        proc = self._spawn(engine, model, port, extra)
        item = LoadedModel(
            id=model,
            engine=engine,
            model=model,
            port=port,
            started_at=time.time(),
            proc=proc,
            args=extra,
        )
        try:
            self._wait(port)
        except Exception as exc:
            code = proc.returncode if proc is not None else None
            self._kill(item)
            if code is not None:
                raise RuntimeError(f"{model} exited with code {code}") from exc
            raise
        self._models[item.id] = item
        return item

    def unload(self, name: str) -> LoadedModel | None:
        item = self.resolve(name)
        if not item:
            return None
        self._models.pop(item.id, None)
        self._kill(item)
        return item

    def unload_all(self) -> None:
        for key in list(self._models):
            self.unload(key)

    def _kill(self, item: LoadedModel) -> None:
        proc = item.proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def wait_healthy(port: int, timeout: float = 600.0) -> None:
    urls = (
        f"http://127.0.0.1:{port}/health",
        f"http://127.0.0.1:{port}/v1/models",
    )
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if 200 <= resp.status < 300:
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        time.sleep(0.2)
    raise TimeoutError(f"engine on :{port} did not become healthy") from last
