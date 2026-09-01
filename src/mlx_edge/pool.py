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
from mlx_edge.scan import context_window
from mlx_edge.templates import template_for_spawn


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


def annotate_load_error(message: str) -> str:
    """Point at `mlx-edge build --help` when an architecture is missing from the engine."""
    text = (message or "Serve failed").rstrip()
    if "mlx-edge build" in text:
        return text
    return (
        f"{text}\n"
        "Hint: if this architecture is missing from mlx-lm / mlx-vlm, "
        "run mlx-edge build --help"
    )


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
    context: int | None = None

    def __post_init__(self) -> None:
        if not self.public_id:
            self.public_id = basename_id(self.model)
        if self.id == self.model:
            self.id = self.public_id
        if self.context is None:
            self.context = context_window(self.model)

    def as_openai(self) -> dict[str, object]:
        row: dict[str, object] = {
            "id": self.public_id,
            "object": "model",
            "created": int(self.started_at),
            "owned_by": owned_by(self.engine),
        }
        if self.context and self.context > 0:
            n = int(self.context)
            # OpenRouter / Cline OpenAI-compat, vLLM, LM Studio native.
            row["context_length"] = n
            row["max_model_len"] = n
            row["max_context_length"] = n
        return row

    def as_lmstudio(self) -> dict[str, object]:
        kind = {"lm": "llm", "vlm": "vlm", "embed": "embeddings"}.get(self.engine, "llm")
        row: dict[str, object] = {
            "id": self.public_id,
            "object": "model",
            "type": kind,
            "publisher": owned_by(self.engine),
            "arch": self.engine,
            "state": "loaded",
        }
        if self.context and self.context > 0:
            n = int(self.context)
            row["max_context_length"] = n
            row["loaded_context_length"] = n
        return row


SpawnFn = Callable[[str, str, int, list[str]], subprocess.Popen[bytes] | None]


def default_spawn(engine_id: str, model: str, port: int, extra: list[str]) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = spawn_argv(engine_id, model, port, extra)
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)


def warmup_engine(item: LoadedModel, timeout: float = 120.0) -> None:
    """One 1-token (or tiny embed) request after Serve so Metal graphs compile.

    The child then sits idle until a real client request. No heartbeat.
    """
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


class Inflight:
    """One in-flight chat/embed so Stop / a dropped OpenAI client can abort it."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.abort = threading.Event()
        self._close: Callable[[], None] | None = None
        self._lock = threading.Lock()

    def set_close(self, close: Callable[[], None]) -> None:
        with self._lock:
            self._close = close
            if self.abort.is_set():
                close()

    def trigger(self) -> None:
        self.abort.set()
        with self._lock:
            close = self._close
        if close is None:
            return
        try:
            close()
        except OSError:
            pass


class ModelPool:
    def __init__(
        self,
        spawn: SpawnFn | None = None,
        wait: Callable[..., None] | None = None,
        progress: ProgressTracker | None = None,
        logs: LogBuffer | None = None,
    ) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._spawn = spawn or default_spawn
        self._wait = wait or wait_healthy
        self.progress = progress or ProgressTracker()
        self.logs = logs or LogBuffer()
        self._inflight: dict[str, Inflight] = {}
        self._inflight_lock = threading.Lock()

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
        if engine in {"lm", "vlm"}:
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
        self.progress.begin_load(item.public_id, engine)
        self._pump_logs(item)
        try:
            self._call_wait(port, proc)
        except Exception as exc:
            code = proc.returncode if proc is not None else None
            self._kill(item)
            self.progress.drop(item.public_id)
            label = item.public_id
            if code is not None:
                raise RuntimeError(annotate_load_error(f"{label} exited with code {code}")) from exc
            raise RuntimeError(annotate_load_error(f"{label} failed to start: {exc}")) from exc
        self.progress.end_load(item.public_id)
        if proc is not None:
            warmup_engine(item)
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

    def track_request(self, name: str) -> Inflight:
        item = self.resolve(name)
        key = item.public_id if item else name
        job = Inflight(key)
        with self._inflight_lock:
            prev = self._inflight.get(key)
            self._inflight[key] = job
        if prev is not None:
            prev.trigger()
        return job

    def untrack_request(self, job: Inflight) -> None:
        with self._inflight_lock:
            if self._inflight.get(job.model_id) is job:
                self._inflight.pop(job.model_id, None)

    def stop_generation(self, name: str | None = None) -> list[str]:
        """Abort in-flight chat/embed. Returns public ids that were signalled."""
        with self._inflight_lock:
            if name:
                item = self.resolve(name)
                key = item.public_id if item else name
                jobs = [self._inflight[key]] if key in self._inflight else []
            else:
                jobs = list(self._inflight.values())
        stopped: list[str] = []
        seen: set[str] = set()
        for job in jobs:
            if job.abort.is_set():
                continue
            job.trigger()
            self.progress.cancel(job.model_id)
            if job.model_id not in seen:
                seen.add(job.model_id)
                stopped.append(job.model_id)
        return stopped

    def unload(self, name: str) -> LoadedModel | None:
        item = self.resolve(name)
        if not item:
            return None
        self._models.pop(item.id, None)
        self.progress.drop(item.public_id)
        self.stop_generation(item.public_id)
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
