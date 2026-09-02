"""OpenAI-compatible gateway in front of hot-loaded mlx engines."""

from __future__ import annotations

import http.client
import json
import re
import select
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from mlx_edge.channels import HarmonyFilter, assume_think_start, rewrite_completion_payload
from mlx_edge.playground import PlaygroundStore
from mlx_edge.pool import Inflight, LoadedModel, ModelPool
from mlx_edge.progress import ProgressTracker

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, content-type",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
}

# Dedicated mlx-vlm.server children (not chat). kind → (route noun, client hint)
SPECIAL_KINDS = {
    "embed": ("embeddings", "POST /v1/embeddings"),
    "tts": ("speech", "POST /v1/audio/speech"),
    "stt": ("transcriptions", "POST /v1/audio/transcriptions"),
    "rerank": ("rerank", "POST /v1/rerank"),
    "image": ("image generation", "POST /v1/images/generations"),
}

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".map": "application/json",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
    ".txt": "text/plain; charset=utf-8",
}

# GUI progress/log polls drown the log. Real POST /v1/chat stays visible.
_QUIET_ACCESS = re.compile(r"\b(?:GET|HEAD) /v1/(?:progress|logs|hub/progress)(?:/|\?|\s)", re.I)


def _quiet_access(line: str) -> bool:
    return bool(_QUIET_ACCESS.search(line))



def bundled_web_dir() -> Path | None:
    path = Path(__file__).resolve().parent / "web"
    if (path / "index.html").is_file():
        return path
    return None


def lan_ipv4() -> str | None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = str(sock.getsockname()[0])
        sock.close()
        return ip
    except OSError:
        return None


def public_base(host: str, port: int) -> dict[str, object]:
    raw = host or "127.0.0.1"
    wildcard = raw in {"0.0.0.0", "::", "[::]", "::0"}
    advertised = (lan_ipv4() or "127.0.0.1") if wildcard else raw
    return {
        "host": advertised,
        "port": int(port),
        "bind": f"{raw}:{int(port)}",
        "url": f"http://{advertised}:{int(port)}/v1",
    }


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON object required")
    return data


def _read_raw(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return b""
    return handler.rfile.read(length)


def _multipart_field(raw: bytes, name: str) -> str:
    marker = f'name="{name}"'.encode("utf-8")
    idx = raw.find(marker)
    if idx < 0:
        return ""
    rest = raw[idx + len(marker) :]
    sep = rest.find(b"\r\n\r\n")
    if sep < 0:
        return ""
    value = rest[sep + 4 :]
    end = value.find(b"\r\n")
    if end < 0:
        return value.decode("utf-8", "replace").strip()
    return value[:end].decode("utf-8", "replace").strip()


def wants_stream(body: dict[str, Any]) -> bool:
    value = body.get("stream")
    if value is True or value == 1:
        return True
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def request_has_tools(body: dict[str, Any]) -> bool:
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def prepare_chat_body(body: dict[str, Any], item: LoadedModel) -> dict[str, Any]:
    """Pin `model` to the child path and always request stream usage.

    Cline's context bar is `usage.prompt_tokens / context_length`. mlx-lm only
    emits usage on SSE when `stream_options.include_usage` is true. LM Studio
    always includes it; we do the same.
    """
    out = dict(body)
    out["model"] = item.model
    if wants_stream(out):
        opts = out.get("stream_options")
        opts = dict(opts) if isinstance(opts, dict) else {}
        opts["include_usage"] = True
        out["stream_options"] = opts
    return out


def make_handler(pool: ModelPool, static_dir: Path | str | None = None) -> type[BaseHTTPRequestHandler]:
    web = Path(static_dir).resolve() if static_dir else None
    playground = PlaygroundStore()

    class GatewayHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            try:
                rendered = fmt % args
            except Exception:
                rendered = str(fmt)
            if _quiet_access(rendered):
                return
            sys_stderr = __import__("sys").stderr
            sys_stderr.write("%s - %s\n" % (self.address_string(), rendered))

        def _cors(self) -> None:
            for key, value in CORS.items():
                self.send_header(key, value)

        def _json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _static(self, raw_path: str) -> bool:
            if web is None:
                return False
            rel = unquote(raw_path.split("?", 1)[0]).lstrip("/") or "index.html"
            if rel.startswith("v1/") or rel == "v1" or rel.startswith("edge/"):
                return False
            target = (web / rel).resolve()
            try:
                target.relative_to(web)
            except ValueError:
                return False
            if target.is_file():
                self._bytes(target.read_bytes(), STATIC_TYPES.get(target.suffix.lower(), "application/octet-stream"))
                return True
            index = web / "index.html"
            if index.is_file() and not Path(rel).suffix:
                self._bytes(index.read_bytes(), "text/html; charset=utf-8")
                return True
            return False

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            raw_path = urlparse(self.path).path
            path = raw_path.rstrip("/") or "/"
            if path == "/health":
                models = [item.public_id for item in pool.list()]
                bind_host, bind_port = self.server.server_address[:2]
                info = public_base(str(bind_host), int(bind_port))
                self._json({"status": "ok", "models": models, "model": models[0] if models else None, **info})
                return
            if path == "/v1/models":
                data = [item.as_openai() for item in pool.list()]
                self._json({"object": "list", "data": data})
                return
            if path.startswith("/v1/models/"):
                name = unquote(path[len("/v1/models/") :]).strip("/")
                if name and name not in {"load", "unload", "scan"}:
                    item = pool.resolve(name)
                    if item is None:
                        self._json(
                            {
                                "error": {
                                    "message": f"{name} is not loaded",
                                    "type": "invalid_request_error",
                                    "code": "model_not_found",
                                }
                            },
                            404,
                        )
                        return
                    self._json(item.as_openai())
                    return
            if path in {"/api/v0/models", "/api/v1/models"}:
                self._json({"object": "list", "data": [item.as_lmstudio() for item in pool.list()]})
                return
            if path in {"/v1/prefs", "/v1/studio"}:
                from mlx_edge.prefs import load_prefs

                self._json(load_prefs())
                return
            if path in {"/v1/progress", "/edge/progress"}:
                self._progress()
                return
            if path in {"/v1/progress/stream", "/edge/progress/stream"}:
                self._progress_stream()
                return
            if path in {"/v1/logs", "/edge/logs"}:
                self._logs()
                return
            if path in {"/v1/logs/stream", "/edge/logs/stream"}:
                self._logs_stream()
                return
            if path in {"/v1/template", "/v1/templates"}:
                self._template_inspect()
                return
            if path in {"/v1/playground", "/v1/chat/session"}:
                self._playground_get()
                return
            if path in {"/v1/hub"}:
                self._hub_status()
                return
            if path in {"/v1/hub/progress"}:
                self._hub_progress()
                return
            if self._static(raw_path):
                return
            self._json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)

        def do_PUT(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in {"/v1/prefs", "/v1/studio"}:
                self._prefs_save()
                return
            if path in {"/v1/playground", "/v1/chat/session"}:
                self._playground_put()
                return
            self._json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)

        def do_DELETE(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in {"/v1/playground", "/v1/chat/session"}:
                self._playground_clear()
                return
            self._json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in {"/v1/load", "/v1/models/load"}:
                self._load()
                return
            if path in {"/v1/unload", "/v1/models/unload"}:
                self._unload()
                return
            if path in {"/v1/scan", "/v1/models/scan"}:
                self._scan()
                return
            if path in {"/v1/hub/search", "/v1/hub"}:
                self._hub_search()
                return
            if path in {"/v1/hub/download"}:
                self._hub_download()
                return
            if path in {"/v1/hub/pause"}:
                self._hub_pause()
                return
            if path in {"/v1/hub/resume"}:
                self._hub_resume()
                return
            if path in {"/v1/hub/cancel"}:
                self._hub_cancel()
                return
            if path in {"/v1/hub/delete"}:
                self._hub_delete()
                return
            if path in {"/v1/prefs", "/v1/studio"}:
                self._prefs_save()
                return
            if path in {"/v1/chat/completions", "/v1/completions", "/chat/completions"}:
                self._proxy()
                return
            if path in {"/v1/embeddings", "/embeddings"}:
                self._kind_json("embed")
                return
            if path in {"/v1/rerank", "/rerank", "/v1/reranking"}:
                self._kind_json("rerank")
                return
            if path in {
                "/v1/images/generations",
                "/images/generations",
                "/v1/images/edits",
                "/images/edits",
            }:
                self._kind_json("image")
                return
            if path in {"/v1/audio/speech", "/audio/speech"}:
                self._audio("tts")
                return
            if path in {
                "/v1/audio/transcriptions",
                "/audio/transcriptions",
                "/v1/audio/translations",
                "/audio/translations",
            }:
                self._audio("stt")
                return
            if path in {"/v1/stop", "/v1/chat/stop", "/stop"}:
                self._stop()
                return
            if path in {"/v1/template", "/v1/templates"}:
                self._template_fetch()
                return
            if path in {"/v1/logs/clear", "/v1/logs"}:
                pool.logs.clear()
                self._json({"ok": True, "seq": pool.logs.seq()})
                return
            if path in {"/v1/playground/clear"}:
                self._playground_clear()
                return
            self._json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)

        def _hub_status(self) -> None:
            from mlx_edge.hub import TOKEN_HELP, token_set

            self._json({"token": token_set(), "help": TOKEN_HELP})

        def _hub_progress(self) -> None:
            from mlx_edge.hub import download_progress

            self._json(download_progress())

        def _hub_search(self) -> None:
            from mlx_edge.hub import TOKEN_HELP, search_quants, token_set

            if not token_set():
                self._json({"error": {"message": TOKEN_HELP, "type": "invalid_request_error"}}, 403)
                return
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            query = str(body.get("query") or body.get("url") or body.get("repo") or "").strip()
            try:
                self._json(search_quants(query))
            except PermissionError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error"}}, 403)
            except ValueError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error"}}, 400)

        def _hub_download(self) -> None:
            from mlx_edge.hub import TOKEN_HELP, start_download, token_set

            if not token_set():
                self._json({"error": {"message": TOKEN_HELP, "type": "invalid_request_error"}}, 403)
                return
            try:
                body = _read_json(self)
            except ValueError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error"}}, 400)
                return
            repo = str(body.get("repo") or body.get("query") or "").strip()
            try:
                self._json(start_download(repo, logs=pool.logs))
            except PermissionError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error"}}, 403)
            except ValueError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error"}}, 400)
            except Exception as extra:  # noqa: BLE001
                self._json({"error": {"message": str(extra), "type": "server_error"}}, 500)

        def _hub_pause(self) -> None:
            from mlx_edge.hub import pause_download

            self._json(pause_download())

        def _hub_resume(self) -> None:
            from mlx_edge.hub import resume_download

            self._json(resume_download())

        def _hub_cancel(self) -> None:
            from mlx_edge.hub import cancel_download

            self._json(cancel_download())

        def _hub_delete(self) -> None:
            from mlx_edge.hub import delete_hub_repo

            try:
                body = _read_json(self)
            except ValueError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error"}}, 400)
                return
            raw = str(body.get("repo") or body.get("path") or "").strip()
            try:
                self._json(delete_hub_repo(raw, pool=pool))
            except FileNotFoundError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error", "code": "model_not_found"}}, 404)
            except PermissionError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error"}}, 403)
            except ValueError as extra:
                self._json({"error": {"message": str(extra), "type": "invalid_request_error"}}, 400)
            except Exception as extra:  # noqa: BLE001
                self._json({"error": {"message": str(extra), "type": "server_error"}}, 500)

        def _load(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            engine = str(body.get("engine") or "lm")
            model = str(body.get("model") or "").strip()
            extra = body.get("args") or []
            if engine not in {"lm", "vlm", "embed", "tts", "stt", "rerank", "image"}:
                self._json({"error": {"message": "engine must be lm, vlm, embed, tts, stt, rerank, or image", "type": "invalid_request_error"}}, 400)
                return
            if not model:
                self._json({"error": {"message": "model is required", "type": "invalid_request_error"}}, 400)
                return
            if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
                self._json({"error": {"message": "args must be a list of strings", "type": "invalid_request_error"}}, 400)
                return
            try:
                item = pool.load(engine, model, extra)
            except Exception as exc:  # noqa: BLE001 — surface engine spawn errors
                from mlx_edge.pool import annotate_load_error

                self._json({"error": {"message": annotate_load_error(str(exc)), "type": "server_error"}}, 500)
                return
            self._json({"ok": True, "model": item.as_openai(), "models": [m.public_id for m in pool.list()]})

        def _unload(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            name = str(body.get("model") or body.get("id") or "").strip()
            if not name:
                self._json({"error": {"message": "model is required", "type": "invalid_request_error"}}, 400)
                return
            item = pool.unload(name)
            if not item:
                self._json({"error": {"message": f"{name} is not loaded", "type": "invalid_request_error"}}, 404)
                return
            self._json({"ok": True, "model": item.public_id, "models": [m.public_id for m in pool.list()]})

        def _scan(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            dirs = body.get("dirs") or []
            if isinstance(body.get("dir"), str) and body.get("dir").strip():
                dirs = [str(body["dir"]), *(dirs if isinstance(dirs, list) else [])]
            if not isinstance(dirs, list) or not all(isinstance(x, str) for x in dirs):
                self._json({"error": {"message": "dirs must be a list of strings", "type": "invalid_request_error"}}, 400)
                return
            from mlx_edge.scan import scan_dirs

            self._json(scan_dirs(dirs))

        def _prefs_save(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            from mlx_edge.prefs import save_prefs

            self._json(save_prefs(body))

        def _playground_get(self) -> None:
            self._json({"turns": playground.get()})

        def _playground_put(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            turns = body.get("turns")
            if not isinstance(turns, list):
                self._json({"error": {"message": "turns must be a list", "type": "invalid_request_error"}}, 400)
                return
            saved = playground.put(turns)
            self._json({"ok": True, "turns": saved})

        def _playground_clear(self) -> None:
            if int(self.headers.get("Content-Length") or 0) > 0:
                try:
                    _read_json(self)
                except ValueError as exc:
                    self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                    return
            playground.clear()
            self._json({"ok": True})

        def _progress(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            needle = (qs.get("model") or [None])[0]
            self._json(pool.progress.snapshot(needle))

        def _progress_stream(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            needle = (qs.get("model") or [None])[0]
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            last = ""
            try:
                while True:
                    snap = pool.progress.snapshot(needle)
                    blob = json.dumps(snap)
                    if blob != last:
                        self.wfile.write(f"data: {blob}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last = blob
                    elif snap.get("active"):
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    pool.progress.wait(pool.progress.seq(), 1.0)
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                return

        def _logs(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            needle = (qs.get("model") or [None])[0]
            after = 0
            try:
                after = int((qs.get("after") or ["0"])[0] or 0)
            except ValueError:
                after = 0
            limit = 500
            try:
                limit = int((qs.get("limit") or ["500"])[0] or 500)
            except ValueError:
                limit = 500
            self._json(pool.logs.snapshot(needle, after=after, limit=limit))

        def _logs_stream(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            needle = (qs.get("model") or [None])[0]
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            last = -1
            try:
                while True:
                    snap = pool.logs.snapshot(needle, after=last, limit=200)
                    lines = snap.get("lines") or []
                    if lines:
                        self.wfile.write(f"data: {json.dumps(snap)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last = int(snap.get("seq") or last)
                    else:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    pool.logs.wait(pool.logs.seq(), 1.0)
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                return

        def _template_inspect(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            model = (qs.get("model") or [""])[0]
            repo = (qs.get("repo") or [None])[0]
            if not model:
                self._json({"error": {"message": "model is required", "type": "invalid_request_error"}}, 400)
                return
            from mlx_edge.templates import inspect_template

            item = pool.resolve(model)
            path = item.model if item else model
            self._json(inspect_template(path, repo or (item.model if item else None)))

        def _template_fetch(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            model = str(body.get("model") or "").strip()
            repo = str(body.get("repo") or "").strip() or None
            if not model:
                self._json({"error": {"message": "model is required", "type": "invalid_request_error"}}, 400)
                return
            from mlx_edge.templates import fetch_template

            item = pool.resolve(model)
            path = item.model if item else model
            self._json(fetch_template(path, repo))

        def _resolve(self, requested: object, *, kind: str) -> LoadedModel | None:
            loaded = pool.list()
            special = kind in SPECIAL_KINDS
            if not loaded:
                hint = kind if special else "lm"
                self._json(
                    {
                        "error": {
                            "message": f"No models loaded. mlx-edge load --engine {hint} --model …",
                            "type": "server_error",
                        }
                    },
                    503,
                )
                return None
            needle = str(requested).strip() if requested else ""
            if special and not needle:
                item = next((m for m in loaded if m.engine == kind), None)
                if not item:
                    self._json(
                        {
                            "error": {
                                "message": f"No {kind} model loaded. mlx-edge load --engine {kind} --model …",
                                "type": "server_error",
                            }
                        },
                        503,
                    )
                    return None
            else:
                item = pool.resolve(needle or None)
            if needle and not item:
                available = ", ".join(m.public_id for m in loaded)
                self._json(
                    {
                        "error": {
                            "message": f"Model {requested!r} is not loaded. Loaded: {available}",
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    },
                    404,
                )
                return None
            if item is None:
                self._json(
                    {"error": {"message": "No models loaded. mlx-edge load --engine lm --model …", "type": "server_error"}},
                    503,
                )
                return None
            if special and item.engine != kind:
                label = SPECIAL_KINDS[kind][0]
                self._json(
                    {
                        "error": {
                            "message": (
                                f"{item.public_id} does not serve {label}. "
                                f"Serve a model tagged {kind}."
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                    400,
                )
                return None
            if kind == "chat" and item.engine in SPECIAL_KINDS:
                dest = SPECIAL_KINDS[item.engine][1]
                self._json(
                    {
                        "error": {
                            "message": f"{item.public_id} is a {item.engine} model. {dest} instead.",
                            "type": "invalid_request_error",
                        }
                    },
                    400,
                )
                return None
            return item

        def _proxy(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            item = self._resolve(body.get("model"), kind="chat")
            if item is None:
                return
            # mlx-lm/mlx-vlm treat a different `model` string as a new Hub
            # id and snapshot_download it. Pin to the path this child was
            # started with so a short repo name cannot trigger a re-download.
            body = prepare_chat_body(body, item)
            stream = wants_stream(body)
            pool.progress.begin(item.public_id, item.engine, stream=stream)
            job = pool.track_request(item.public_id)
            try:
                _proxy_to(
                    self,
                    item,
                    json.dumps(body).encode("utf-8"),
                    stream=stream,
                    progress=pool.progress,
                    strip_channels=item.engine != "embed",
                    assume_analysis=assume_think_start(item.model, item.public_id),
                    parse_tools=request_has_tools(body),
                    job=job,
                    logs=pool.logs,
                )
            finally:
                pool.untrack_request(job)

        def _kind_json(self, kind: str) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            item = self._resolve(body.get("model"), kind=kind)
            if item is None:
                return
            body["model"] = item.model
            pool.progress.begin(item.public_id, item.engine, stream=False)
            job = pool.track_request(item.public_id)
            try:
                _proxy_to(
                    self,
                    item,
                    json.dumps(body).encode("utf-8"),
                    stream=False,
                    progress=pool.progress,
                    job=job,
                    logs=pool.logs,
                )
            finally:
                pool.untrack_request(job)

        def _audio(self, kind: str) -> None:
            ctype = (self.headers.get("Content-Type") or "").lower()
            multipart = "multipart/" in ctype
            raw: bytes
            requested: object = None
            if multipart:
                raw = _read_raw(self)
                requested = _multipart_field(raw, "model")
            else:
                try:
                    body = _read_json(self)
                except ValueError as exc:
                    self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                    return
                requested = body.get("model")
            item = self._resolve(requested, kind=kind)
            if item is None:
                return
            if multipart:
                needle = str(requested or "").strip()
                if needle and needle != item.model:
                    raw = raw.replace(needle.encode("utf-8"), item.model.encode("utf-8"), 1)
            else:
                body["model"] = item.model
                raw = json.dumps(body).encode("utf-8")
            pool.progress.begin(item.public_id, item.engine, stream=False)
            job = pool.track_request(item.public_id)
            try:
                _proxy_to(
                    self,
                    item,
                    raw,
                    stream=False,
                    progress=pool.progress,
                    job=job,
                    logs=pool.logs,
                )
            finally:
                pool.untrack_request(job)

        def _stop(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            model = str(body.get("model") or "").strip()
            stopped = pool.stop_generation(model or None)
            for mid in stopped:
                engine = "lm"
                item = pool.resolve(mid)
                if item is not None:
                    engine = item.engine
                pool.logs.append(mid, engine, "Stopped generation")
            self._json({"ok": True, "stopped": stopped, "models": stopped})

    return GatewayHandler


def _client_gone(handler: BaseHTTPRequestHandler) -> bool:
    sock = getattr(handler, "connection", None)
    if sock is None:
        return False
    try:
        ready, _, _ = select.select([sock], [], [], 0)
        if not ready:
            return False
        data = sock.recv(1, socket.MSG_PEEK)
        return data == b""
    except (BlockingIOError, InterruptedError):
        return False
    except OSError:
        return True


def _wait_io(child_sock: socket.socket | None, handler: BaseHTTPRequestHandler, job: Inflight | None, idle: float = 0.2) -> str:
    """'data' | 'abort' | 'disconnect' | 'timeout'."""
    if job is not None and job.abort.is_set():
        return "abort"
    if _client_gone(handler):
        if job is not None:
            job.trigger()
        return "disconnect"
    client = getattr(handler, "connection", None)
    watch = [s for s in (child_sock, client) if s is not None]
    if not watch:
        return "timeout"
    try:
        ready, _, _ = select.select(watch, [], [], idle)
    except (OSError, ValueError):
        return "disconnect"
    if job is not None and job.abort.is_set():
        return "abort"
    if client is not None and client in ready and _client_gone(handler):
        if job is not None:
            job.trigger()
        return "disconnect"
    if child_sock is not None and child_sock in ready:
        return "data"
    return "timeout"


def _write_aborted_done(handler: BaseHTTPRequestHandler) -> None:
    try:
        payload = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        return


def _proxy_to(
    handler: BaseHTTPRequestHandler,
    item: LoadedModel,
    body: bytes,
    stream: bool = False,
    progress: ProgressTracker | None = None,
    strip_channels: bool = False,
    assume_analysis: bool = False,
    parse_tools: bool = False,
    job: Inflight | None = None,
    logs: Any = None,
) -> None:
    headers = {
        key: value
        for key, value in handler.headers.items()
        if key.lower() not in {"host", "content-length", "accept-encoding"}
    }
    headers["Content-Type"] = handler.headers.get("Content-Type") or "application/json"
    headers["Accept-Encoding"] = "identity"
    headers["Connection"] = "close"
    tracker = progress
    model_id = item.public_id
    conn = http.client.HTTPConnection("127.0.0.1", item.port, timeout=600)
    try:
        conn.connect()
        if job is not None:
            job.set_close(conn.close)
        conn.request("POST", handler.path, body=body, headers=headers)
        deadline = time.time() + 600
        while True:
            state = _wait_io(conn.sock, handler, job)
            if state == "data":
                break
            if state in {"abort", "disconnect"}:
                if tracker:
                    tracker.cancel(model_id)
                if logs is not None and state == "disconnect":
                    logs.append(model_id, item.engine, "Client disconnected — stopping generation")
                return
            if time.time() > deadline:
                raise TimeoutError("engine timed out")
        resp = conn.getresponse()
        content_type = resp.getheader("Content-Type") or "application/json"
        is_stream = stream or "text/event-stream" in content_type.lower()
        if is_stream:
            _pipe_sse(
                handler,
                resp,
                tracker,
                model_id,
                strip_channels=strip_channels,
                assume_analysis=assume_analysis,
                parse_tools=parse_tools,
                job=job,
                logs=logs,
                engine=item.engine,
                child_sock=conn.sock,
            )
            return
        payload = _read_body(resp, handler, job, conn.sock)
        if payload is None:
            if tracker:
                tracker.cancel(model_id)
            if logs is not None and _client_gone(handler):
                logs.append(model_id, item.engine, "Client disconnected — stopping generation")
            return
        if strip_channels:
            try:
                data = json.loads(payload.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                payload = json.dumps(
                    rewrite_completion_payload(
                        data, assume_analysis=assume_analysis, parse_tools=parse_tools
                    )
                ).encode("utf-8")
        handler.send_response(resp.status)
        for key, value in CORS.items():
            handler.send_header(key, value)
        for key, value in resp.getheaders():
            if key.lower() in {"transfer-encoding", "connection", "content-encoding", "content-length"}:
                continue
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
        if tracker:
            if resp.status >= 400:
                tracker.fail(model_id, f"engine HTTP {resp.status}")
            else:
                tracker.complete(model_id)
    except (BrokenPipeError, ConnectionResetError):
        if tracker:
            tracker.cancel(model_id)
        if logs is not None:
            logs.append(model_id, item.engine, "Client disconnected — stopping generation")
    except (TimeoutError, OSError, http.client.HTTPException) as exc:
        if job is not None and job.abort.is_set():
            if tracker:
                tracker.cancel(model_id)
            return
        try:
            handler.send_response(502)
            for key, value in CORS.items():
                handler.send_header(key, value)
            msg = json.dumps({"error": {"message": str(exc), "type": "server_error"}}).encode()
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(msg)))
            handler.end_headers()
            handler.wfile.write(msg)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        if tracker:
            tracker.fail(model_id, str(exc))
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _arm_timeout(sock: socket.socket | None, seconds: float) -> None:
    if sock is None:
        return
    try:
        sock.settimeout(seconds)
    except OSError:
        return


def _read_body(resp: http.client.HTTPResponse, handler: BaseHTTPRequestHandler, job: Inflight | None, child_sock: socket.socket | None) -> bytes | None:
    _arm_timeout(child_sock, 0.2)
    chunks: list[bytes] = []
    deadline = time.time() + 600
    while True:
        if job is not None and job.abort.is_set():
            return None
        if _client_gone(handler):
            if job is not None:
                job.trigger()
            return None
        try:
            chunk = resp.read(65536)
        except TimeoutError:
            if time.time() > deadline:
                raise TimeoutError("engine timed out")
            continue
        except (OSError, http.client.HTTPException):
            if job is not None and job.abort.is_set():
                return None
            raise
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _is_done_frame(frame: str) -> bool:
    for line in frame.splitlines():
        if line.startswith("data:") and line[5:].strip() == "[DONE]":
            return True
    return False


def _pipe_sse(
    handler: BaseHTTPRequestHandler,
    resp: Any,
    tracker: ProgressTracker | None,
    model_id: str,
    strip_channels: bool = False,
    assume_analysis: bool = False,
    parse_tools: bool = False,
    job: Inflight | None = None,
    logs: Any = None,
    engine: str = "lm",
    child_sock: socket.socket | None = None,
) -> None:
    handler.send_response(200)
    for key, value in CORS.items():
        handler.send_header(key, value)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()
    leftover = b""
    sse_buf = b""
    pending_done: str | None = None
    filt = (
        HarmonyFilter(assume_analysis=assume_analysis, parse_tools=parse_tools) if strip_channels else None
    )
    read1 = getattr(resp, "read1", None)
    stopped = False
    disconnected = False
    _arm_timeout(child_sock, 0.2)
    try:
        while True:
            if job is not None and job.abort.is_set():
                stopped = True
                break
            if _client_gone(handler):
                stopped = True
                disconnected = True
                if job is not None:
                    job.trigger()
                break
            try:
                chunk = read1(512) if read1 else resp.read(512)
            except TimeoutError:
                continue
            except (OSError, http.client.HTTPException):
                if job is not None and job.abort.is_set():
                    stopped = True
                break
            if not chunk:
                break
            if filt is None:
                handler.wfile.write(chunk)
                handler.wfile.flush()
                if tracker:
                    leftover = tracker.ingest_sse(model_id, leftover + chunk)
                continue
            sse_buf += chunk
            frames = sse_buf.split(b"\n\n")
            sse_buf = frames.pop() if frames else b""
            for raw in frames:
                frame = raw.decode("utf-8", "replace")
                if tracker:
                    leftover = tracker.ingest_sse(model_id, leftover + raw + b"\n\n")
                rewritten = _rewrite_sse_frame(frame, filt)
                if rewritten is None:
                    continue
                # OpenAI clients stop at [DONE]. Hold it until the filter
                # can flush a MiniMax answer that was buffered as thinking.
                if _is_done_frame(rewritten):
                    pending_done = rewritten
                    continue
                handler.wfile.write(rewritten.encode("utf-8") + b"\n\n")
                handler.wfile.flush()
        if stopped:
            if logs is not None and disconnected:
                logs.append(model_id, engine, "Client disconnected — stopping generation")
            if tracker:
                tracker.cancel(model_id)
            if not disconnected:
                _write_aborted_done(handler)
            return
        if filt is not None and sse_buf.strip():
            frame = sse_buf.decode("utf-8", "replace")
            rewritten = _rewrite_sse_frame(frame, filt)
            if rewritten:
                if _is_done_frame(rewritten):
                    pending_done = rewritten
                else:
                    handler.wfile.write(rewritten.encode("utf-8") + b"\n\n")
                    handler.wfile.flush()
        if filt is not None:
            extra_c, extra_r = filt.flush()
            extra_tools = filt.take_tool_calls()
            if extra_c or extra_r or extra_tools:
                tail = {"choices": [{"index": 0, "delta": {}}]}
                if extra_c:
                    tail["choices"][0]["delta"]["content"] = extra_c
                if extra_r:
                    tail["choices"][0]["delta"]["reasoning_content"] = extra_r
                if extra_tools:
                    tail["choices"][0]["delta"]["tool_calls"] = extra_tools
                    tail["choices"][0]["finish_reason"] = "tool_calls"
                handler.wfile.write(f"data: {json.dumps(tail)}\n\n".encode("utf-8"))
                handler.wfile.flush()
            elif filt.saw_tools:
                tail = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                handler.wfile.write(f"data: {json.dumps(tail)}\n\n".encode("utf-8"))
                handler.wfile.flush()
        if pending_done:
            handler.wfile.write(pending_done.encode("utf-8") + b"\n\n")
            handler.wfile.flush()
        if tracker:
            tracker.complete(model_id)
    except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
        if job is not None:
            job.trigger()
        if tracker:
            tracker.cancel(model_id)
        if logs is not None:
            logs.append(model_id, engine, "Client disconnected — stopping generation")


def _rewrite_sse_frame(frame: str, filt: HarmonyFilter) -> str | None:
    lines = frame.splitlines()
    if not lines:
        return None
    data_lines = [line[5:].strip() for line in lines if line.startswith("data:")]
    if not data_lines:
        return frame
    payload = "\n".join(data_lines)
    if payload == "[DONE]":
        return frame
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return frame
    if not isinstance(obj, dict):
        return frame
    rewritten = rewrite_completion_payload(obj, filt)
    comments = [line for line in lines if not line.startswith("data:")]
    if not rewritten.get("choices"):
        # Usage-only SSE (stream_options.include_usage) has empty choices —
        # Cline's context bar reads that chunk. Do not drop it.
        if rewritten.get("usage"):
            return "\n".join(comments + [f"data: {json.dumps(rewritten)}"])
        comments_only = [line for line in lines if not line.startswith("data:")]
        return "\n".join(comments_only) if comments_only else None
    out = comments + [f"data: {json.dumps(rewritten)}"]
    return "\n".join(out)


def serve_forever(
    pool: ModelPool,
    host: str,
    port: int,
    static_dir: Path | str | None = None,
) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(pool, static_dir=static_dir))
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        pool.unload_all()
