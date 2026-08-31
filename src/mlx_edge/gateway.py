"""OpenAI-compatible gateway in front of hot-loaded mlx engines."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from mlx_edge.pool import LoadedModel, ModelPool

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, content-type",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
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


def make_handler(pool: ModelPool) -> type[BaseHTTPRequestHandler]:
    class GatewayHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            sys_stderr = __import__("sys").stderr
            sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

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
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/health":
                models = [item.model for item in pool.list()]
                self._json({"status": "ok", "models": models, "model": models[0] if models else None})
                return
            if path == "/v1/models":
                data = [item.as_openai() for item in pool.list()]
                self._json({"object": "list", "data": data})
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
            if path in {"/v1/chat/completions", "/v1/completions", "/chat/completions"}:
                self._proxy()
                return
            self._json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)

        def _load(self) -> None:
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            engine = str(body.get("engine") or "lm")
            model = str(body.get("model") or "").strip()
            extra = body.get("args") or []
            if not model:
                self._json({"error": {"message": "model is required", "type": "invalid_request_error"}}, 400)
                return
            if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
                self._json({"error": {"message": "args must be a list of strings", "type": "invalid_request_error"}}, 400)
                return
            try:
                item = pool.load(engine, model, extra)
            except Exception as exc:  # noqa: BLE001 — surface engine spawn errors
                self._json({"error": {"message": str(exc), "type": "server_error"}}, 500)
                return
            self._json({"ok": True, "model": item.as_openai(), "models": [m.model for m in pool.list()]})

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
            self._json({"ok": True, "model": item.model, "models": [m.model for m in pool.list()]})

        def _proxy(self) -> None:
            loaded = pool.list()
            if not loaded:
                self._json(
                    {"error": {"message": "No models loaded. mlx-edge load --engine lm --model …", "type": "server_error"}},
                    503,
                )
                return
            try:
                body = _read_json(self)
            except ValueError as exc:
                self._json({"error": {"message": str(exc), "type": "invalid_request_error"}}, 400)
                return
            requested = body.get("model")
            item = pool.resolve(str(requested) if requested else None)
            if requested and not item:
                available = ", ".join(m.model for m in loaded)
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
                return
            assert item is not None
            _proxy_to(self, item, json.dumps(body).encode("utf-8"))

    return GatewayHandler


def _proxy_to(handler: BaseHTTPRequestHandler, item: LoadedModel, body: bytes) -> None:
    headers = {
        key: value
        for key, value in handler.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    headers["Content-Type"] = handler.headers.get("Content-Type") or "application/json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{item.port}{handler.path}",
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = resp.read()
            handler.send_response(resp.status)
            for key, value in CORS.items():
                handler.send_header(key, value)
            for key, value in resp.headers.items():
                if key.lower() in {"transfer-encoding", "connection", "content-encoding"}:
                    continue
                handler.send_header(key, value)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        handler.send_response(exc.code)
        for key, value in CORS.items():
            handler.send_header(key, value)
        handler.send_header("Content-Type", exc.headers.get("Content-Type") or "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        handler.send_response(502)
        for key, value in CORS.items():
            handler.send_header(key, value)
        msg = json.dumps({"error": {"message": str(exc), "type": "server_error"}}).encode()
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(msg)))
        handler.end_headers()
        handler.wfile.write(msg)


def serve_forever(pool: ModelPool, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(pool))
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        pool.unload_all()
