"""Search and download MLX quants from Hugging Face into the local hub cache."""

from __future__ import annotations

import json
import os
import re
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


def is_active_hub(path: Path) -> bool:
    return JOB.is_active(path)


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
    url = f"{HF_API}/{urllib.parse.quote(repo, safe='')}?blobs=true"
    req = urllib.request.Request(url, headers=hf_headers())
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return 0
    total = 0
    for sib in data.get("siblings") or []:
        if not isinstance(sib, dict):
            continue
        try:
            total += int(sib.get("size") or 0)
        except (TypeError, ValueError):
            continue
    return total


class HubTqdm:
    """tqdm stand-in. Blocking in update() pauses the huggingface_hub byte loop."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.n = 0
        self.total = int(kwargs.get("total") or 0)
        self.desc = str(kwargs.get("desc") or "")
        self.disable = True
        JOB.add_bar(self)

    def update(self, n: int | float = 1) -> None:
        JOB.wait_run()
        if JOB.cancel.is_set():
            raise HubCancelled()
        self.n += int(n)
        JOB.recompute()

    def close(self) -> None:
        JOB.drop_bar(self)

    def clear(self, *args: Any, **kwargs: Any) -> None:
        return None

    def refresh(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reset(self, total: int | None = None) -> None:
        if total is not None:
            self.total = int(total)
        self.n = 0

    def __enter__(self) -> HubTqdm:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __iter__(self):
        for item in self.iterable or []:
            self.update(1)
            yield item


class HubJob:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.repo = ""
        self.phase = "idle"
        self.n = 0
        self.total = 0
        self.error = ""
        self.path = ""
        self.pause = threading.Event()
        self.pause.set()
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.bars: list[HubTqdm] = []
        self.logs: Any | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            total = self.total
            n = self.n
            ratio = min(1.0, n / total) if total else (1.0 if self.phase == "done" else 0.0)
            return {
                "repo": self.repo,
                "phase": self.phase,
                "bytes": n,
                "total": total,
                "ratio": round(ratio, 4),
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
        try:
            return path.resolve() == hub_folder(self.repo).resolve()
        except OSError:
            return False

    def add_bar(self, bar: HubTqdm) -> None:
        with self.lock:
            self.bars.append(bar)

    def drop_bar(self, bar: HubTqdm) -> None:
        with self.lock:
            self.bars = [b for b in self.bars if b is not bar]

    def recompute(self) -> None:
        with self.lock:
            n = sum(max(0, b.n) for b in self.bars)
            bar_total = sum(max(0, b.total) for b in self.bars)
            if bar_total > self.total:
                self.total = bar_total
            if n > self.n:
                self.n = n

    def wait_run(self) -> None:
        while True:
            if self.cancel.is_set():
                raise HubCancelled()
            if self.pause.wait(timeout=0.2):
                return

    def start(self, repo: str, logs: Any | None = None) -> dict[str, Any]:
        require_token()
        repo = parse_repo(repo)
        bits = _quant_bit_count({}, Path(repo), repo)
        if bits is not None and bits not in MLX_QUANT_BITS:
            raise ValueError(f"{repo} is {bits}-bit — mlx only loads 2/3/4/5/6/8")
        with self.lock:
            if self.phase in {"downloading", "paused"} and self.repo == repo:
                self.pause.set()
                self.phase = "downloading"
                return self.snapshot()
            if self.phase in {"downloading", "paused"}:
                raise RuntimeError(f"already downloading {self.repo}")
            self.repo = repo
            self.phase = "downloading"
            self.n = 0
            self.total = 0
            self.error = ""
            self.path = ""
            self.bars = []
            self.logs = logs
            self.cancel.clear()
            self.pause.set()
        if logs is not None:
            logs.append(repo.split("/")[-1], "hub", f"Downloading {repo} into Hugging Face cache…")
        self.thread = threading.Thread(target=self._run, name="hf-download", daemon=True)
        self.thread.start()
        return self.snapshot()

    def request_pause(self) -> dict[str, Any]:
        with self.lock:
            if self.phase != "downloading":
                return self.snapshot()
            self.phase = "paused"
        self.pause.clear()
        return self.snapshot()

    def request_resume(self) -> dict[str, Any]:
        with self.lock:
            if self.phase != "paused":
                return self.snapshot()
            self.phase = "downloading"
        self.pause.set()
        return self.snapshot()

    def request_cancel(self) -> dict[str, Any]:
        self.cancel.set()
        self.pause.set()
        with self.lock:
            if self.phase in {"downloading", "paused"}:
                self.phase = "cancelled"
        return self.snapshot()

    def _run(self) -> None:
        repo = self.repo
        try:
            expected = repo_nbytes(repo)
            with self.lock:
                if expected:
                    self.total = expected
            path = _snapshot(repo, tqdm_class=HubTqdm)
            if self.cancel.is_set():
                raise HubCancelled()
            with self.lock:
                self.path = path
                self.phase = "done"
                if self.total:
                    self.n = self.total
            if self.logs is not None:
                self.logs.append(repo.split("/")[-1], "hub", f"Downloaded {repo} → {path}")
        except HubCancelled:
            with self.lock:
                self.phase = "cancelled"
                self.error = "cancelled"
            if self.logs is not None:
                self.logs.append(repo.split("/")[-1], "hub", f"Cancelled {repo}")
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.phase = "error"
                self.error = str(exc)
            if self.logs is not None:
                self.logs.append(repo.split("/")[-1], "hub", f"Download failed: {exc}")


JOB = HubJob()


def start_download(raw: str, logs: Any | None = None) -> dict[str, Any]:
    return JOB.start(raw, logs=logs)


def download_progress() -> dict[str, Any]:
    return JOB.snapshot()


def pause_download() -> dict[str, Any]:
    return JOB.request_pause()


def resume_download() -> dict[str, Any]:
    return JOB.request_resume()


def cancel_download() -> dict[str, Any]:
    return JOB.request_cancel()


def download_repo(raw: str, logs: Any | None = None, timeout: float = 3600.0) -> dict[str, Any]:
    """Blocking helper for tests / CLI. Studio uses start_download + poll."""
    snap = start_download(raw, logs=logs)
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = download_progress()
        if snap["phase"] in {"done", "error", "cancelled"}:
            break
        time.sleep(0.05)
    if snap["phase"] == "done":
        return {"ok": True, "repo": snap["repo"], "path": snap["path"], "token": token_set()}
    raise RuntimeError(snap.get("error") or f"download {snap['phase']}")


def _snapshot(repo: str, tqdm_class: type | None = None) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return _snapshot_cli(repo)
    kwargs: dict[str, Any] = {"repo_id": repo, "token": token, "max_workers": 1}
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class
    return str(snapshot_download(**kwargs))


def _snapshot_cli(repo: str) -> str:
    import shutil
    import subprocess

    exe = shutil.which("hf") or shutil.which("huggingface-cli")
    if not exe:
        raise RuntimeError("huggingface_hub is not installed (and no hf CLI)")
    cmd = [exe, "download", repo]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "hf download failed").strip())
    line = (proc.stdout or "").strip().splitlines()
    return line[-1] if line else repo
