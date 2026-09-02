"""Search and download MLX quants from Hugging Face into the local hub cache."""

from __future__ import annotations

import json
import os
import re
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


def token_set() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


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
        "token": token_set(),
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


def download_repo(raw: str, logs: Any | None = None) -> dict[str, Any]:
    repo = parse_repo(raw)
    bits = _quant_bit_count({}, Path(repo), repo)
    if bits is not None and bits not in MLX_QUANT_BITS:
        raise ValueError(f"{repo} is {bits}-bit — mlx only loads 2/3/4/5/6/8")
    if logs is not None:
        logs.append(repo.split("/")[-1], "hub", f"Downloading {repo} into Hugging Face cache…")
    path = _snapshot(repo)
    if logs is not None:
        logs.append(repo.split("/")[-1], "hub", f"Downloaded {repo} → {path}")
    return {"ok": True, "repo": repo, "path": path, "token": token_set()}


def _snapshot(repo: str) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or True
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return _snapshot_cli(repo)
    return str(snapshot_download(repo_id=repo, token=token))


def _snapshot_cli(repo: str) -> str:
    import shutil
    import subprocess

    exe = shutil.which("hf") or shutil.which("huggingface-cli")
    if not exe:
        raise RuntimeError("huggingface_hub is not installed (and no hf CLI)")
    cmd = [exe, "download", repo] if exe.endswith("hf") else [exe, "download", repo]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "hf download failed").strip())
    line = (proc.stdout or "").strip().splitlines()
    return line[-1] if line else repo
