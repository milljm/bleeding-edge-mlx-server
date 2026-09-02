"""Walk a model directory and describe MLX checkpoints the GUI can serve."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

MAX_DEPTH = 8
MAX_MODELS = 500
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".locks",
    "__pycache__",
    "node_modules",
    "blobs",
    "refs",
    "tmp",
    ".mplx",
}
WEIGHT_SUFFIXES = {".safetensors", ".npz", ".gguf", ".mlx"}
INDEX_NAMES = {
    "model.safetensors.index.json",
    "model.npz.index.json",
    "weights.safetensors.index.json",
}
# Text-only on mlx-lm even when the type name looks VL. MiniMax-M3
# (`minimax_m3_vl`) ships a vision tower mlx-lm ignores; the working loader is
# patched mlx-lm (`mlx-edge build`), not mlx-vlm.
LM_MODEL_TYPES = ("minimax_m3_vl",)
VLM_MARKERS = (
    "vision",
    "vlm",
    "_vl",
    "vl_",
    "llava",
    "idefics",
    "pixtral",
    "internvl",
    "florence",
    "molmo",
    "ocr",
    "omni",
    "paligemma",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "phi3v",
    "phi_3_vision",
    "deepseek_ocr",
)
EMBED_NAME_MARKERS = (
    "embedding",
    "embedder",
    "embed_",
    "_embed",
    "bge_",
    "_bge",
    "e5_",
    "_e5",
    "gte_",
    "_gte",
    "minilm",
    "nomic",
    "jina_embed",
    "arctic_embed",
)
EMBED_ARCH_MARKERS = (
    "bertmodel",
    "xlmrobertamodel",
    "nomicbert",
)
RERANK_NAME_MARKERS = (
    "rerank",
    "cross_encoder",
    "crossencoder",
)
IMAGE_NAME_MARKERS = (
    "flux",
    "z_image",
    "zimage",
    "stable_diffusion",
    "sdxl",
    "sd3",
    "mage_flow",
    "mageflow",
    "ernie_image",
    "hidream",
    "kolors",
    "auraflow",
    "text2image",
    "t2i",
    "image_gen",
)
TTS_NAME_MARKERS = (
    "tts",
    "kokoro",
    "orpheus",
    "chatterbox",
    "pocket_tts",
    "f5_tts",
    "xtts",
    "styletts",
    "vibevoice",
    "sesame",
    "spark_tts",
    "dia_",
    "_dia",
)
STT_NAME_MARKERS = (
    "whisper",
    "parakeet",
    "canary",
    "moonshine",
    "sensevoice",
    "faster_whisper",
    "speech_to_text",
    "_stt",
    "stt_",
    "asr",
)
STT_TYPES = ("whisper", "parakeet", "canary", "moonshine")


def scan_dirs(dirs: list[str]) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in dirs:
        typed = str(raw or "").strip()
        if not typed:
            continue
        root = _expand(typed)
        if not root.exists():
            errors.append({"dir": typed, "message": "not found"})
            continue
        if not root.is_dir():
            errors.append({"dir": typed, "message": "not a directory"})
            continue
        try:
            found = list_models(typed)
        except OSError as exc:
            errors.append({"dir": typed, "message": str(exc)})
            continue
        for item in found:
            key = item["path"]
            if key in seen:
                continue
            seen.add(key)
            models.append(item)
            if len(models) >= MAX_MODELS:
                return {"models": models, "errors": errors}
    return {"models": models, "errors": errors}


def list_models(typed_dir: str) -> list[dict[str, Any]]:
    root = _expand(typed_dir)
    if not root.is_dir():
        return []
    found: list[dict[str, Any]] = []
    rec = _describe(root, root, typed_dir)
    if rec:
        return [rec]
    _walk(root, root, typed_dir, 0, found)
    found.sort(key=lambda m: m["repo"].lower())
    return found


def _walk(root: Path, current: Path, typed_dir: str, depth: int, found: list[dict[str, Any]]) -> None:
    if len(found) >= MAX_MODELS or depth > MAX_DEPTH:
        return
    hub = _hub_snapshot(current)
    if hub is not None:
        rec = _describe(hub, root, typed_dir, hub_dir=current)
        if rec:
            found.append(rec)
        return
    try:
        children = sorted(current.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return
    for child in children:
        if len(found) >= MAX_MODELS:
            return
        if not child.is_dir() or child.is_symlink():
            continue
        if child.name in SKIP_DIRS or child.name.startswith("."):
            continue
        if child.name.startswith(("datasets--", "spaces--")):
            continue
        rec = _describe(child, root, typed_dir)
        if rec:
            found.append(rec)
            continue
        _walk(root, child, typed_dir, depth + 1, found)


def _hub_snapshot(path: Path) -> Path | None:
    snaps = path / "snapshots"
    if not snaps.is_dir():
        return None
    if not (path.name.startswith("models--") or (path / "refs").is_dir() or (path / "blobs").is_dir()):
        return None
    ref = path / "refs" / "main"
    if ref.is_file():
        try:
            sha = ref.read_text(encoding="utf-8").strip()
        except OSError:
            sha = ""
        if sha:
            cand = snaps / sha
            if cand.is_dir():
                return cand
    try:
        children = [p for p in snaps.iterdir() if p.is_dir() and not p.is_symlink()]
    except OSError:
        return None
    if not children:
        return None
    return max(children, key=lambda p: p.stat().st_mtime)


def _describe(
    path: Path,
    root: Path,
    typed_dir: str,
    hub_dir: Path | None = None,
) -> dict[str, Any] | None:
    if not _is_model_dir(path):
        return None
    cfg = _read_config(path / "config.json")
    repo = _infer_repo(path, root, hub_dir)
    engine = _infer_engine(cfg, repo, path.name)
    quant = _infer_quant(cfg, path, repo)
    size = _weight_size(path)
    return {
        "id": slug_model_id(engine, repo),
        "name": _pretty_name(path.name if hub_dir is None else repo.rsplit("/", 1)[-1]),
        "repo": repo,
        "path": str(path),
        "engine": engine,
        "size": _human_size(size),
        "quant": quant,
        "context": _infer_context(cfg),
        "watchDir": typed_dir,
        "source": "scan",
        "hasChatTemplate": _has_chat_template(path),
    }


def _has_chat_template(path: Path) -> bool:
    if (path / "chat_template.jinja").is_file():
        return True
    cfg = _read_config(path / "tokenizer_config.json")
    tmpl = cfg.get("chat_template")
    if isinstance(tmpl, str) and tmpl.strip():
        return True
    if isinstance(tmpl, list) and tmpl:
        return True
    return False


def _is_model_dir(path: Path) -> bool:
    cfg = path / "config.json"
    if not cfg.is_file():
        return False
    return _has_weights(path)


def _has_weights(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.name in INDEX_NAMES:
                return True
            if child.suffix.lower() in WEIGHT_SUFFIXES:
                return True
    except OSError:
        return False
    return False


def _read_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _infer_repo(path: Path, root: Path, hub_dir: Path | None) -> str:
    if hub_dir is not None:
        from_hub = _repo_from_hub_name(hub_dir.name)
        if from_hub:
            return from_hub
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    parts = [p for p in rel.parts if p not in {"snapshots"} and not _looks_like_sha(p)]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if parts:
        return parts[0]
    return path.name


def _repo_from_hub_name(name: str) -> str | None:
    if not name.startswith("models--"):
        return None
    body = name[len("models--") :]
    if "--" not in body:
        return body or None
    org, rest = body.split("--", 1)
    return f"{org}/{rest}" if org and rest else None


def _looks_like_sha(name: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{7,64}", name))


def _infer_engine(cfg: dict[str, Any], repo: str, folder: str) -> str:
    architectures = cfg.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [architectures]
    blob = " ".join(
        [
            str(cfg.get("model_type") or ""),
            " ".join(str(a) for a in architectures),
            repo,
            folder,
        ]
    ).lower().replace("-", "_")
    # Rerank before embed: bge-reranker matches both.
    if any(marker in blob for marker in RERANK_NAME_MARKERS):
        return "rerank"
    if any(marker in blob for marker in IMAGE_NAME_MARKERS):
        return "image"
    if any(marker in blob for marker in EMBED_NAME_MARKERS):
        return "embed"
    model_type = str(cfg.get("model_type") or "").lower().replace("-", "_")
    if model_type in LM_MODEL_TYPES:
        return "lm"
    # Omni VLMs (vision + audio_config) stay vlm. Dedicated speech checkpoints
    # have no vision tower — kokoro / whisper / parakeet.
    if not any(cfg.get(key) for key in ("vision_config", "image_config")):
        if model_type in STT_TYPES or any(marker in blob for marker in STT_NAME_MARKERS):
            return "stt"
        if any(marker in blob for marker in TTS_NAME_MARKERS):
            return "tts"
    if any(cfg.get(key) for key in ("vision_config", "image_config", "audio_config")):
        return "vlm"
    if any(marker in blob for marker in EMBED_ARCH_MARKERS):
        return "embed"
    if any(marker in blob for marker in VLM_MARKERS):
        return "vlm"
    return "lm"


CONTEXT_KEYS = (
    "max_position_embeddings",
    "max_sequence_length",
    "max_seq_len",
    "n_positions",
    "seq_length",
    "model_max_length",
    "max_length",
)


def _infer_context(cfg: dict[str, Any], depth: int = 0) -> int | None:
    if depth > 3 or not isinstance(cfg, dict):
        return None
    for key in CONTEXT_KEYS:
        value = cfg.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    for nested_key in ("text_config", "llm_config", "language_config", "decoder"):
        nested = cfg.get(nested_key)
        if isinstance(nested, dict):
            found = _infer_context(nested, depth + 1)
            if found:
                return found
    return None


def context_window(model: str | Path) -> int | None:
    """Read the checkpoint's context window from config.json, if present."""
    path = Path(str(model))
    cfg = path / "config.json"
    if not cfg.is_file():
        return None
    return _infer_context(_read_config(cfg))


def _infer_quant(cfg: dict[str, Any], path: Path, repo: str) -> str:
    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if isinstance(quant, dict):
        bits = quant.get("bits") or quant.get("n_bits")
        if bits:
            return f"{bits}-bit"
    blob = f"{path.name} {repo}"
    match = re.search(r"(?i)(?:^|[-_])(\d)\s*-?bit(?:$|[-_])", blob)
    if match:
        return f"{match.group(1)}-bit"
    for token, label in (
        ("mxfp4", "MXFP4"),
        ("bf16", "BF16"),
        ("fp16", "FP16"),
        ("fp32", "FP32"),
        ("8bit", "8-bit"),
        ("6bit", "6-bit"),
        ("5bit", "5-bit"),
        ("4bit", "4-bit"),
        ("3bit", "3-bit"),
        ("2bit", "2-bit"),
    ):
        if token in blob.lower():
            return label
    return "—"


def _pretty_name(raw: str) -> str:
    name = re.sub(r"(?i)[-_]?(?:\d-?bit|mxfp4|bf16|fp16|fp32)$", "", raw)
    name = name.replace("_", " ").replace("-", " ").strip()
    return name or raw


def _weight_size(path: Path) -> int:
    total = 0
    try:
        for child in path.iterdir():
            if child.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            try:
                total += child.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _human_size(n: int) -> str:
    if n <= 0:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if idx == 0:
        return f"{int(value)} B"
    if value >= 10:
        return f"{value:.0f} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def slug_model_id(engine: str, repo: str) -> str:
    slug = f"{engine}-{repo.lower()}"
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def _expand(raw: str) -> Path:
    return Path(os.path.expanduser(raw)).expanduser()
