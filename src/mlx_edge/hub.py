"""Search and download MLX quants from Hugging Face into the local hub cache."""

from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from mlx_edge import __version__
from mlx_edge.scan import MLX_QUANT_BITS, _quant_bit_count

USER_AGENT = f"mlx-edge/{__version__}"
HF_API = "https://huggingface.co/api/models"
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
QUANT_TAIL = re.compile(r"(?i)[-_]?(?:mlx-?)?(?:\d+-?bit|mxfp4|bf16|fp16|fp32)$")
TOKEN_HELP = (
    "Create a Hugging Face account, copy a token from "
    "https://huggingface.co/settings/tokens, then launch edge-gui with HF_TOKEN set."
)


class HubCancelled(Exception):
    pass


def token_set() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def require_token() -> None:
    if not token_set():
        raise PermissionError(TOKEN_HELP)


def parse_repo(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise ValueError("paste a Hugging Face URL or org/name")
    text = text.split("?")[0].split("#")[0].strip()
    text = re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", text, flags=re.I)
    text = text.lstrip("/")
    if text.lower().startswith("models/"):
        text = text[7:]
    parts = [p for p in text.split("/") if p]
    if len(parts) >= 2:
        repo = f"{parts[0]}/{parts[1]}"
    else:
        repo = parts[0] if parts else ""
    if not repo or not REPO_RE.match(repo):
        raise ValueError(f"not a Hugging Face repo: {raw!r}")
    return repo


def search_stem(repo: str) -> str:
    name = repo.split("/")[-1]
    stem = QUANT_TAIL.sub("", name).strip("-_")
    return stem or name


def hf_headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def hf_search(search: str, *, author: str | None = None, filt: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    query: dict[str, str] = {"search": search, "limit": str(limit), "full": "full"}
    if author:
        query["author"] = author
    if filt:
        query["filter"] = filt
    url = f"{HF_API}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers=hf_headers())
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def looks_mlx(row: dict[str, Any]) -> bool:
    rid = str(row.get("id") or "").lower()
    lib = str(row.get("library_name") or "").lower()
    tags = [str(t).lower() for t in (row.get("tags") or [])]
    if lib == "mlx" or "mlx" in tags or "mlx-community" in rid:
        return True
    if "-mlx" in rid or rid.endswith("mlx") or "mlx-" in rid:
        return True
    return False


def _quant_rank(repo_id: str) -> tuple[int, str]:
    bits = _quant_bit_count({}, Path(repo_id), repo_id)
    order = {4: 0, 8: 1, 6: 2, 5: 3, 3: 4, 2: 5}
    if bits in order:
        return (order[bits], repo_id.lower())
    blob = repo_id.lower()
    if "bf16" in blob:
        return (6, repo_id.lower())
    if "fp16" in blob:
        return (7, repo_id.lower())
    return (8, repo_id.lower())


def search_quants(raw: str) -> dict[str, Any]:
    require_token()
    repo = parse_repo(raw)
    stem = search_stem(repo)
    seen: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    rows.extend(hf_search(stem, author="mlx-community"))
    rows.extend(hf_search(stem, filt="mlx"))
    rows.extend(hf_search(repo))
    for row in rows:
        rid = str(row.get("id") or "").strip()
        if not rid or rid in seen or not looks_mlx(row):
            continue
        bits = _quant_bit_count({}, Path(rid), rid)
        if bits is not None and bits not in MLX_QUANT_BITS:
            continue
        seen[rid] = {
            "id": rid,
            "quant": f"{bits}-bit" if bits else _pretty_quant(rid),
            "downloads": int(row.get("downloads") or 0),
        }
    results = sorted(seen.values(), key=lambda r: _quant_rank(str(r["id"])))
    if repo not in seen and "/" in repo:
        if looks_mlx({"id": repo, "tags": [], "library_name": ""}):
            bits = _quant_bit_count({}, Path(repo), repo)
            if bits is None or bits in MLX_QUANT_BITS:
                results.insert(0, {"id": repo, "quant": _pretty_quant(repo), "downloads": 0})
    return {
        "query": raw.strip(),
        "repo": repo,
        "stem": stem,
        "token": True,
        "results": results[:16],
    }


def _pretty_quant(repo_id: str) -> str:
    bits = _quant_bit_count({}, Path(repo_id), repo_id)
    if bits is not None:
        return f"{bits}-bit"
    blob = repo_id.lower()
    for token, label in (("mxfp4", "MXFP4"), ("bf16", "BF16"), ("fp16", "FP16")):
        if token in blob:
            return label
    return "MLX"


def hf_hub_root() -> Path:
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def hub_folder(repo: str) -> Path:
    return hf_hub_root() / ("models--" + repo.replace("/", "--"))


def resolve_hub_delete_target(raw: str) -> Path:
    """Only models--* folders under the Hugging Face hub cache."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("repo or cache path is required")
    root = hf_hub_root().resolve()
    candidate = Path(text).expanduser()
    if candidate.exists():
        cur = candidate.resolve()
        if cur.is_file():
            cur = cur.parent
        while True:
            if cur.name.startswith("models--") and (cur.parent == root or root in cur.parents):
                return cur
            if cur == cur.parent:
                break
            cur = cur.parent
        raise PermissionError("Edge only deletes Hugging Face cache models, not LM Studio or Ollama files")
    repo = parse_repo(text)
    folder = hub_folder(repo).resolve()
    if folder.parent != root and root not in folder.parents:
        raise PermissionError("Edge only deletes Hugging Face cache models")
    if not folder.name.startswith("models--"):
        raise PermissionError("not a Hugging Face hub model folder")
    return folder


def delete_hub_repo(raw: str, pool: Any | None = None) -> dict[str, Any]:
    import shutil

    folder = resolve_hub_delete_target(raw)
    if not folder.is_dir():
        raise FileNotFoundError(f"not on disk: {folder}")
    repo = folder.name.removeprefix("models--").replace("--", "/", 1)
    if pool is not None:
        item = pool.resolve(repo) or pool.resolve(raw) or pool.resolve(str(folder))
        if item is not None:
            pool.unload(item.id)
    shutil.rmtree(folder)
    return {"ok": True, "repo": repo, "path": str(folder)}


def is_active_hub(path: Path) -> bool:
    return QUEUE.is_active(path)


def hub_dir_incomplete(hub_dir: Path) -> bool:
    """True while huggingface_hub still has temp files for this repo."""
    inc = hub_dir / "incomplete"
    if inc.is_dir():
        try:
            if any(inc.iterdir()):
                return True
        except OSError:
            pass
    blobs = hub_dir / "blobs"
    if blobs.is_dir():
        try:
            for child in blobs.iterdir():
                if child.name.endswith(".incomplete"):
                    return True
        except OSError:
            pass
    return False


def repo_nbytes(repo: str) -> int:
    """Hub file sizes. Keep the slash in org/name — %2F 404s and we used to log 'size unknown'."""
    data = _hub_json(f"{HF_API}/{urllib.parse.quote(repo, safe='/')}?blobs=true")
    total = _siblings_nbytes(data)
    if total:
        return total
    if not data:
        data = _hub_json(f"{HF_API}/{urllib.parse.quote(repo, safe='/')}") or {}
        total = _siblings_nbytes(data)
        if total:
            return total
    try:
        stored = int((data or {}).get("usedStorage") or 0)
    except (TypeError, ValueError):
        stored = 0
    if stored:
        return stored
    tree = _hub_json(
        f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}/tree/main?recursive=true"
    )
    if isinstance(tree, list):
        return sum(_entry_size(row) for row in tree if isinstance(row, dict) and row.get("type") != "directory")
    return 0


def _hub_json(url: str) -> Any:
    req = urllib.request.Request(url, headers=hf_headers())
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _siblings_nbytes(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    return sum(_entry_size(sib) for sib in (data.get("siblings") or []) if isinstance(sib, dict))


def _entry_size(row: dict[str, Any]) -> int:
    sizes = [0]
    for raw in (row.get("size"),):
        try:
            sizes.append(int(raw or 0))
        except (TypeError, ValueError):
            pass
    lfs = row.get("lfs")
    if isinstance(lfs, dict):
        try:
            sizes.append(int(lfs.get("size") or 0))
        except (TypeError, ValueError):
            pass
    return max(sizes)


def hub_downloaded_bytes(hub_dir: Path) -> int:
    """Bytes on disk for this repo (blobs + incomplete). Snapshots are symlinks."""
    total = 0
    for sub in ("blobs", "incomplete"):
        folder = hub_dir / sub
        if not folder.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(folder):
            for name in filenames:
                path = Path(dirpath) / name
                try:
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_size
                except OSError:
                    continue
    return total


def human_bytes(n: int) -> str:
    n = max(0, int(n))
    if n < 1024:
        return f"{n} B"
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= size:
            value = n / size
            return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"
    return f"{n} B"


def _signal(proc: Any, sig: int) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.send_signal(sig)
    except (OSError, ProcessLookupError, AttributeError, ValueError):
        return


def _kill_proc(proc: Any) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGCONT)
    except (OSError, ProcessLookupError, AttributeError, ValueError):
        pass
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


_DOWNLOAD_PY = r"""
import os, sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
path = snapshot_download(repo_id=repo, token=token)
print(path)
"""


def _spawn_download(repo: str) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return subprocess.Popen(
        [sys.executable, "-u", "-c", _DOWNLOAD_PY, repo],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


class HubJob:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.repo = ""
        self.phase = "idle"
        self.n = 0
        self.total = 0
        self.error = ""
        self.path = ""
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.logs: Any | None = None
        self._logged_pct = -1
        self.finished_at = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            total = self.total
            n = self.n
            ratio = min(1.0, n / total) if total else (1.0 if self.phase == "done" else 0.0)
            pct = round(100.0 * n / total, 1) if total else None
            detail = f"{human_bytes(n)} / {human_bytes(total)}" if total else (human_bytes(n) if n else "")
            return {
                "repo": self.repo,
                "name": self.repo.split("/")[-1] if self.repo else "",
                "phase": self.phase,
                "bytes": n,
                "total": total,
                "ratio": round(ratio, 4),
                "pct": pct,
                "detail": detail,
                "error": self.error,
                "path": self.path,
                "token": token_set(),
            }

    def is_active(self, path: Path) -> bool:
        with self.lock:
            if self.phase not in {"downloading", "paused"}:
                return False
            if not self.repo:
                return False
            repo = self.repo
        try:
            return path.resolve() == hub_folder(repo).resolve()
        except OSError:
            return False

    def _poll_disk(self, folder: Path) -> None:
        n = hub_downloaded_bytes(folder)
        with self.lock:
            if self.total and n > self.total:
                self.total = n
            self.n = n
            total = self.total
            pct = int((n / total) * 100) if total else 0
            logs = self.logs
            last = self._logged_pct
            repo = self.repo
        if logs is not None and total and pct >= last + 5:
            with self.lock:
                self._logged_pct = pct
            logs.append(repo.split("/")[-1], "hub", f"{pct}% · {human_bytes(n)} / {human_bytes(total)}")
        elif logs is not None and not total and n and last < 0:
            with self.lock:
                self._logged_pct = 0
            logs.append(repo.split("/")[-1], "hub", f"{human_bytes(n)} (size unknown)")

    def launch(self, repo: str, logs: Any | None = None) -> dict[str, Any]:
        with self.lock:
            self.repo = repo
            self.phase = "downloading"
            self.n = 0
            self.total = 0
            self.error = ""
            self.path = ""
            self.logs = logs
            self.proc = None
            self._logged_pct = -1
            self.finished_at = 0.0
            self.cancel.clear()
        if logs is not None:
            logs.append(repo.split("/")[-1], "hub", f"Downloading {repo} into Hugging Face cache…")
        self.thread = threading.Thread(target=self._run, name=f"hf-download-{repo.split('/')[-1]}", daemon=True)
        self.thread.start()
        return self.snapshot()

    def request_pause(self) -> dict[str, Any]:
        with self.lock:
            if self.phase != "downloading":
                return self.snapshot()
            self.phase = "paused"
            proc = self.proc
        _signal(proc, signal.SIGSTOP)
        return self.snapshot()

    def request_resume(self) -> dict[str, Any]:
        with self.lock:
            if self.phase != "paused":
                return self.snapshot()
            self.phase = "downloading"
            proc = self.proc
        _signal(proc, signal.SIGCONT)
        return self.snapshot()

    def request_cancel(self) -> dict[str, Any]:
        self.cancel.set()
        with self.lock:
            proc = self.proc
            if self.phase in {"downloading", "paused"}:
                self.phase = "cancelled"
                self.finished_at = time.time()
        _kill_proc(proc)
        return self.snapshot()

    def _run(self) -> None:
        repo = self.repo
        folder = hub_folder(repo)
        try:
            expected = repo_nbytes(repo)
            with self.lock:
                if expected:
                    self.total = expected
            self._poll_disk(folder)
            proc = _spawn_download(repo)
            with self.lock:
                self.proc = proc
            chunks: list[bytes] = []
            stdout = proc.stdout
            while True:
                if self.cancel.is_set():
                    _kill_proc(proc)
                    raise HubCancelled()
                if stdout is not None:
                    ready, _, _ = select.select([stdout], [], [], 0.2)
                    if ready:
                        piece = stdout.read(4096)
                        if piece:
                            chunks.append(piece)
                else:
                    time.sleep(0.2)
                self._poll_disk(folder)
                code = proc.poll()
                if code is None:
                    continue
                if stdout is not None:
                    rest = stdout.read() or b""
                    if rest:
                        chunks.append(rest)
                text = b"".join(chunks).decode("utf-8", "replace").strip()
                if code != 0:
                    raise RuntimeError(text or f"download exited {code}")
                path = text.splitlines()[-1] if text else str(folder)
                with self.lock:
                    self.path = path
                    self.phase = "done"
                    self.finished_at = time.time()
                    if self.total:
                        self.n = self.total
                if self.logs is not None:
                    self.logs.append(repo.split("/")[-1], "hub", f"Downloaded {repo} → {path}")
                return
        except HubCancelled:
            with self.lock:
                self.phase = "cancelled"
                self.error = "cancelled"
                self.finished_at = time.time()
            if self.logs is not None:
                self.logs.append(repo.split("/")[-1], "hub", f"Cancelled {repo}")
        except Exception as extra:  # noqa: BLE001
            with self.lock:
                self.phase = "error"
                self.error = str(extra)
                self.finished_at = time.time()
            if self.logs is not None:
                self.logs.append(repo.split("/")[-1], "hub", f"Download failed: {extra}")
        finally:
            with self.lock:
                self.proc = None


class HubQueue:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, HubJob] = {}

    def is_active(self, path: Path) -> bool:
        with self.lock:
            jobs = list(self.jobs.values())
        return any(job.is_active(path) for job in jobs)

    def snapshots(self) -> list[dict[str, Any]]:
        now = time.time()
        with self.lock:
            keep: dict[str, HubJob] = {}
            rows: list[dict[str, Any]] = []
            for repo, job in self.jobs.items():
                snap = job.snapshot()
                done = snap["phase"] not in {"downloading", "paused"}
                if done and job.finished_at and now - job.finished_at > 45:
                    continue
                keep[repo] = job
                rows.append(snap)
            self.jobs = keep
        return rows

    def _get(self, repo: str | None) -> HubJob:
        with self.lock:
            if repo:
                key = parse_repo(repo)
                job = self.jobs.get(key)
                if job is None:
                    raise ValueError(f"no download for {key}")
                return job
            live = [j for j in self.jobs.values() if j.phase in {"downloading", "paused"}]
            if len(live) == 1:
                return live[0]
            if not live:
                raise ValueError("no active download")
            raise ValueError("repo is required when several downloads are running")

    def start(self, repo: str, logs: Any | None = None) -> dict[str, Any]:
        require_token()
        repo = parse_repo(repo)
        bits = _quant_bit_count({}, Path(repo), repo)
        if bits is not None and bits not in MLX_QUANT_BITS:
            raise ValueError(f"{repo} is {bits}-bit — mlx only loads 2/3/4/5/6/8")
        with self.lock:
            existing = self.jobs.get(repo)
            if existing is not None and existing.phase in {"downloading", "paused"}:
                return existing.snapshot()
            job = HubJob()
            self.jobs[repo] = job
        return job.launch(repo, logs=logs)

    def pause(self, repo: str | None = None) -> dict[str, Any]:
        return self._get(repo).request_pause()

    def resume(self, repo: str | None = None) -> dict[str, Any]:
        return self._get(repo).request_resume()

    def cancel(self, repo: str | None = None) -> dict[str, Any]:
        return self._get(repo).request_cancel()

    def clear(self) -> None:
        with self.lock:
            jobs = list(self.jobs.values())
            self.jobs = {}
        for job in jobs:
            job.cancel.set()
            _kill_proc(job.proc)


QUEUE = HubQueue()


def start_download(raw: str, logs: Any | None = None) -> dict[str, Any]:
    return QUEUE.start(raw, logs=logs)


def download_progress() -> dict[str, Any]:
    jobs = QUEUE.snapshots()
    return {"token": token_set(), "jobs": jobs}


def pause_download(repo: str | None = None) -> dict[str, Any]:
    return QUEUE.pause(repo)


def resume_download(repo: str | None = None) -> dict[str, Any]:
    return QUEUE.resume(repo)


def cancel_download(repo: str | None = None) -> dict[str, Any]:
    return QUEUE.cancel(repo)


def download_repo(raw: str, logs: Any | None = None, timeout: float = 3600.0) -> dict[str, Any]:
    """Blocking helper for tests / CLI. Studio uses start_download + poll."""
    snap = start_download(raw, logs=logs)
    repo = str(snap.get("repo") or "")
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = download_progress().get("jobs") or []
        match = next((row for row in jobs if row.get("repo") == repo), snap)
        if match.get("phase") in {"done", "error", "cancelled"}:
            snap = match
            break
        time.sleep(0.05)
    if snap.get("phase") == "done":
        return {"ok": True, "repo": snap["repo"], "path": snap["path"], "token": token_set()}
    raise RuntimeError(snap.get("error") or f"download {snap.get('phase')}")

