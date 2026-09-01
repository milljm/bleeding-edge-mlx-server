"""In-process registry of hot-loaded mlx-lm / mlx-vlm children."""

from __future__ import annotations

import json
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
from mlx_edge.logs import LogBuffer
from mlx_edge.progress import ProgressTracker
from mlx_edge.templates import template_for_spawn

# After a real request the graphs are hot. Don't immediately 1-token-warm them.
REHEAT_COOLDOWN = 20.0
KEEP_HOT_INTERVAL = 30.0


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def strip_bind_args(extra: list[str]) -> list[str]:
    """Children always bind 127.0.0.1:ephemeral. Drop competing --host/--port."""
    return strip_named_args(extra, {"--host", "--port"})


def strip_named_args(extra: list[str], names: set[str]) -> list[str]:
    out: list[str] = []
    skip = False
    prefixes = tuple(name + "=" for name in names)
    for arg in extra:
        if skip:
            skip = False
            continue
        if arg in names:
            skip = True
            continue
        if arg.startswith(prefixes):
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
    if engine_id == "lm":
        return [sys.executable, "-u", "-m", "mlx_lm", "server"]
    if engine_id in {"vlm", "embed"}:
        return [sys.executable, "-u", "-m", "mlx_vlm.server"]
    engine = get_engine(engine_id)
    if not engine.server_module:
        raise RuntimeError(f"{engine.dist} has no server")
    return [sys.executable, "-u", "-m", engine.server_module]


def spawn_argv(engine_id: str, model: str, port: int, extra: list[str]) -> list[str]:
    extra = strip_bind_args(list(extra or []))
    extra = strip_named_args(extra, {"--model", "--embedding-model"})
    if engine_id == "embed":
        return [
            *server_argv("embed"),
            "--embedding-model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *extra,
        ]
    return [
        *server_argv(engine_id),
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *extra,
    ]


def owned_by(engine: str) -> str:
    return {"lm": "mlx-lm", "vlm": "mlx-vlm", "embed": "mlx-embed"}.get(engine, "mlx-lm")


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
            "owned_by": owned_by(self.engine),
        }


SpawnFn = Callable[[str, str, int, list[str]], subprocess.Popen[bytes] | None]


def default_spawn(engine_id: str, model: str, port: int, extra: list[str]) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = spawn_argv(engine_id, model, port, extra)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


def warmup_engine(item: LoadedModel, timeout: float = 120.0) -> None:
    """Compile Metal graphs so the first real request is not a cold start."""
    if item.engine == "embed":
        payload: dict[str, object] = {"model": item.model, "input": "ok"}
        path = "/v1/embeddings"
    else:
        payload = {
            "model": item.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
        path = "/v1/chat/completions"
    req = urllib.request.Request(
        f"http://127.0.0.1:{item.port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError, urllib.error.HTTPError):
        return


class ModelPool:
    def __init__(
        self,
        spawn: SpawnFn | None = None,
        wait: Callable[..., None] | None = None,
        progress: ProgressTracker | None = None,
        logs: LogBuffer | None = None,
        keep_hot: bool = True,
    ) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._spawn = spawn or default_spawn
        self._wait = wait or wait_healthy
        self.progress = progress or ProgressTracker()
        self.logs = logs or LogBuffer()
        self._keep_hot = keep_hot
        self._hot_stop = threading.Event()
        self._hot_thread: threading.Thread | None = None
        self._busy: set[str] = set()
        self._warm_at: dict[str, float] = {}
        self._warming: set[str] = set()
        self._warm_pending: set[str] = set()
        self._warm_lock = threading.Lock()
        self._warm_cv = threading.Condition(self._warm_lock)
        self._warm_thread: threading.Thread | None = None

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
        if engine == "lm":
            extra = template_for_spawn(model, extra)
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
        if proc is not None:
            warmup_engine(item)
            self._warm_at[item.public_id] = time.time()
            if engine == "embed":
                self._ensure_keep_hot()
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
        logs = self.logs

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
                    logs.append(public_id, engine, text)
                    progress.ingest_log(public_id, engine, text)
            except Exception:
                pass

        threading.Thread(target=run, name=f"mlx-edge-log-{public_id}", daemon=True).start()

    def _call_wait(self, port: int, proc: subprocess.Popen[bytes] | None) -> None:
        try:
            self._wait(port, proc)
        except TypeError:
            self._wait(port)

    def mark_busy(self, name: str, busy: bool) -> None:
        item = self.resolve(name)
        key = item.public_id if item else name
        if busy:
            self._busy.add(key)
        else:
            self._busy.discard(key)
            self._warm_at[key] = time.time()

    def _queue_warm(self, item: LoadedModel) -> bool:
        """At most one pending warmup per model. Skip busy / in-flight / cooldown."""
        if item.proc is None:
            return False
        key = item.public_id
        with self._warm_lock:
            if key in self._busy or key in self._warming or key in self._warm_pending:
                return False
            last = self._warm_at.get(key, 0.0)
            if time.time() - last < REHEAT_COOLDOWN:
                return False
            self._warm_pending.add(key)
            self._warm_cv.notify()
        self._ensure_warm_worker()
        return True

    def _ensure_warm_worker(self) -> None:
        with self._warm_lock:
            if self._warm_thread is not None:
                return
            self._warm_thread = threading.Thread(target=self._warm_loop, name="mlx-edge-warm", daemon=True)
            self._warm_thread.start()

    def _warm_loop(self) -> None:
        while not self._hot_stop.is_set():
            key = None
            with self._warm_cv:
                if not self._warm_pending:
                    self._warm_cv.wait(timeout=1.0)
                if self._warm_pending:
                    key = next(iter(self._warm_pending))
                    self._warm_pending.discard(key)
                    self._warming.add(key)
            if not key:
                continue
            item = self.resolve(key)
            try:
                if item is not None and item.proc is not None and item.public_id not in self._busy:
                    timeout = 15.0 if item.engine == "embed" else 45.0
                    warmup_engine(item, timeout=timeout)
                    self._warm_at[item.public_id] = time.time()
            finally:
                with self._warm_lock:
                    self._warming.discard(key)

    def reheat_others(self, item: LoadedModel) -> None:
        """After embeddings, warm chat graphs; after chat, keep embeddings hot.

        One worker, one pending slot per model, 20s cooldown. Rapid RAG embeds
        used to spawn a thread each and stack 1-token VL requests (in_flight=11).
        """
        if item.proc is None:
            return
        want_embed = item.engine != "embed"
        for other in self.list():
            if other.id == item.id or other.proc is None:
                continue
            if want_embed and other.engine == "embed":
                self._queue_warm(other)
            elif not want_embed and other.engine != "embed":
                self._queue_warm(other)

    def _ensure_keep_hot(self) -> None:
        if not self._keep_hot or self._hot_thread is not None:
            return

        def loop() -> None:
            while not self._hot_stop.wait(KEEP_HOT_INTERVAL):
                for item in self.list():
                    if item.engine != "embed" or item.proc is None:
                        continue
                    self._queue_warm(item)

        self._hot_thread = threading.Thread(target=loop, name="mlx-edge-keep-hot", daemon=True)
        self._hot_thread.start()

    def unload(self, name: str) -> LoadedModel | None:
        item = self.resolve(name)
        if not item:
            return None
        self._models.pop(item.id, None)
        self.progress.drop(item.public_id)
        with self._warm_lock:
            self._busy.discard(item.public_id)
            self._warming.discard(item.public_id)
            self._warm_pending.discard(item.public_id)
            self._warm_at.pop(item.public_id, None)
        self._kill(item)
        return item

    def unload_all(self) -> None:
        self._hot_stop.set()
        with self._warm_cv:
            self._warm_cv.notify_all()
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
