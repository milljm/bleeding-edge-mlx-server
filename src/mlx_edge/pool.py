"""In-process registry of hot-loaded mlx-lm / mlx-vlm children."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from mlx_edge.engines import get_engine
from mlx_edge.progress import ProgressTracker


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


def normalize_name(name: str) -> str:
    return name.strip().replace("\\", "/").rstrip("/").lower()


def basename_id(path: str) -> str:
    name = Path(path.replace("\\", "/").rstrip("/")).name
    return name or path


def unique_public_id(path: str, taken: list[str]) -> str:
    taken_n = {normalize_name(t) for t in taken if t}
    base = basename_id(path)
    if normalize_name(base) not in taken_n:
        return base
    parent = Path(path.replace("\\", "/")).parent.name
    candidate = f"{parent}/{base}" if parent else base
    if normalize_name(candidate) not in taken_n:
        return candidate
    return path


def names_for(item: "LoadedModel") -> list[str]:
    out = [item.id, item.model, item.public_id, basename_id(item.model), basename_id(item.public_id)]
    return [n for n in out if n]


def names_match(left: str, right: str) -> bool:
    a = normalize_name(left)
    b = normalize_name(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if a.endswith("/" + b) or b.endswith("/" + a):
        return True
    return a.split("/")[-1] == b.split("/")[-1]


def server_argv(engine_id: str) -> list[str]:
    """Prefer `python -m mlx_lm server` — `python -m mlx_lm.server` is deprecated."""
    engine = get_engine(engine_id)
    if not engine.server_module:
        raise RuntimeError(f"{engine.dist} has no server")
    if engine.id == "lm":
        return [sys.executable, "-u", "-m", "mlx_lm", "server"]
    return [sys.executable, "-u", "-m", engine.server_module]


@dataclass
class LoadedModel:
    id: str
    engine: str
    model: str
    port: int
    started_at: float
    proc: subprocess.Popen[bytes] | None = None
    args: list[str] = field(default_factory=list)
    public_id: str = ""

    def __post_init__(self) -> None:
        if not self.public_id:
            self.public_id = basename_id(self.model)
        if self.id == self.model:
            self.id = self.public_id

    def as_openai(self) -> dict[str, object]:
        return {
            "id": self.public_id,
            "object": "model",
            "created": int(self.started_at),
            "owned_by": "mlx-vlm" if self.engine == "vlm" else "mlx-lm",
        }


SpawnFn = Callable[[str, str, int, list[str]], subprocess.Popen[bytes] | None]


def default_spawn(engine_id: str, model: str, port: int, extra: list[str]) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [
        *server_argv(engine_id),
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *strip_bind_args(extra),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


class ModelPool:
    def __init__(
        self,
        spawn: SpawnFn | None = None,
        wait: Callable[..., None] | None = None,
        progress: ProgressTracker | None = None,
    ) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._spawn = spawn or default_spawn
        self._wait = wait or wait_healthy
        self.progress = progress or ProgressTracker()

    def list(self) -> list[LoadedModel]:
        return sorted(self._models.values(), key=lambda m: m.started_at)

    def resolve(self, name: str | None) -> LoadedModel | None:
        loaded = self.list()
        if not loaded:
            return None
        if not name:
            return loaded[0]
        needle = name.strip()
        for item in loaded:
            if any(names_match(candidate, needle) for candidate in names_for(item)):
                return item
        return None

    def load(self, engine: str, model: str, extra: list[str] | None = None) -> LoadedModel:
        extra = list(extra or [])
        existing = self.resolve(model)
        if existing:
            self.unload(existing.id)
        port = free_port()
        proc = self._spawn(engine, model, port, extra)
        public_id = unique_public_id(model, [m.public_id for m in self.list()])
        item = LoadedModel(
            id=public_id,
            engine=engine,
            model=model,
            port=port,
            started_at=time.time(),
            proc=proc,
            args=extra,
            public_id=public_id,
        )
        self.progress.ensure(item.public_id, engine)
        self._pump_logs(item)
        try:
            self._call_wait(port, proc)
        except Exception as exc:
            code = proc.returncode if proc is not None else None
            self._kill(item)
            self.progress.drop(item.public_id)
            label = item.public_id
            if code is not None:
                raise RuntimeError(f"{label} exited with code {code}") from exc
            raise RuntimeError(f"{label} failed to start: {exc}") from exc
        self._models[item.id] = item
        return item

    def _pump_logs(self, item: LoadedModel) -> None:
        proc = item.proc
        stdout = getattr(proc, "stdout", None) if proc is not None else None
        if proc is None or stdout is None:
            return
        public_id = item.public_id
        engine = item.engine
        progress = self.progress

        def run() -> None:
            try:
                while True:
                    line = stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", "replace") if isinstance(line, (bytes, bytearray)) else str(line)
                    try:
                        sys.stderr.write(text)
                        if not text.endswith("\n"):
                            sys.stderr.write("\n")
                        sys.stderr.flush()
                    except Exception:
                        pass
                    progress.ingest_log(public_id, engine, text)
            except Exception:
                pass

        threading.Thread(target=run, name=f"mlx-edge-log-{public_id}", daemon=True).start()

    def _call_wait(self, port: int, proc: subprocess.Popen[bytes] | None) -> None:
        try:
            self._wait(port, proc)
        except TypeError:
            self._wait(port)

    def unload(self, name: str) -> LoadedModel | None:
        item = self.resolve(name)
        if not item:
            return None
        self._models.pop(item.id, None)
        self.progress.drop(item.public_id)
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


def wait_healthy(
    port: int,
    proc: subprocess.Popen[bytes] | None = None,
    timeout: float = 600.0,
) -> None:
    urls = (
        f"http://127.0.0.1:{port}/health",
        f"http://127.0.0.1:{port}/v1/models",
    )
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        if proc is not None:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(f"engine exited with code {code}")
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if 200 <= resp.status < 300:
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        time.sleep(0.2)
    raise TimeoutError(f"engine on :{port} did not become healthy") from last
