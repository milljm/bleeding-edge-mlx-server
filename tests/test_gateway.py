import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from mlx_edge.gateway import make_handler
from mlx_edge.pool import ModelPool, free_port, strip_bind_args


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.pool = ModelPool(spawn=lambda *_a, **_k: None, wait=lambda _port: None)
        self.port = free_port()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), make_handler(self.pool))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _json(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode() or "{}")
                return resp.status, body if isinstance(body, dict) else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode()
            try:
                body = json.loads(raw or "{}")
            except json.JSONDecodeError:
                body = {"error": {"message": raw}}
            return exc.code, body if isinstance(body, dict) else {}

    def test_empty_models_and_chat_503(self):
        status, body = self._json("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("data"), [])
        status, body = self._json("POST", "/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 503)

    def test_hot_load_two_then_unload_one(self):
        status, body = self._json(
            "POST",
            "/v1/load",
            {"engine": "lm", "model": "mlx-community/Qwen3-8B-4bit"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body.get("ok"))
        status, body = self._json(
            "POST",
            "/v1/load",
            {"engine": "vlm", "model": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body.get("models") or []), 2)

        status, listed = self._json("GET", "/v1/models")
        self.assertEqual(status, 200)
        ids = [row["id"] for row in listed["data"]]
        self.assertEqual(
            ids,
            [
                "Qwen3-8B-4bit",
                "Qwen2.5-VL-7B-Instruct-4bit",
            ],
        )

        status, body = self._json(
            "POST",
            "/v1/chat/completions",
            {"model": "missing-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 404)
        self.assertIn("not loaded", (body.get("error") or {}).get("message", ""))

        status, body = self._json("POST", "/v1/unload", {"model": "mlx-community/Qwen3-8B-4bit"})
        self.assertEqual(status, 200)
        self.assertEqual(body.get("models"), ["Qwen2.5-VL-7B-Instruct-4bit"])

        status, listed = self._json("GET", "/v1/models")
        self.assertEqual([row["id"] for row in listed["data"]], ["Qwen2.5-VL-7B-Instruct-4bit"])

    def test_health_lists_pool(self):
        self._json("POST", "/v1/load", {"engine": "lm", "model": "a"})
        self._json("POST", "/v1/load", {"engine": "lm", "model": "b"})
        status, body = self._json("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("models"), ["a", "b"])
        self.assertEqual(body.get("model"), "a")
        self.assertEqual(body.get("url"), f"http://127.0.0.1:{self.port}/v1")
        self.assertEqual(body.get("bind"), f"127.0.0.1:{self.port}")

    def test_strip_bind_args(self):
        self.assertEqual(
            strip_bind_args(["--temp", "0.2", "--host", "0.0.0.0", "--port", "9", "--max-tokens", "64"]),
            ["--temp", "0.2", "--max-tokens", "64"],
        )

    def test_gui_static(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from mlx_edge.gateway import make_handler, public_base

        info = public_base("127.0.0.1", 8080)
        self.assertEqual(info["url"], "http://127.0.0.1:8080/v1")
        wild = public_base("0.0.0.0", 9000)
        self.assertEqual(wild["bind"], "0.0.0.0:9000")
        self.assertTrue(str(wild["url"]).endswith(":9000/v1"))
        self.assertNotIn("0.0.0.0", str(wild["url"]))

        from mlx_edge.gateway import bundled_web_dir

        bundled = bundled_web_dir()
        self.assertIsNotNone(bundled)
        self.assertTrue((bundled / "index.html").is_file())

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.html").write_text("<html>edge-gui</html>", encoding="utf-8")
            (root / "assets").mkdir()
            (root / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
            pool = ModelPool(spawn=lambda *_a, **_k: None, wait=lambda _port: None)
            port = free_port()
            httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(pool, static_dir=root))
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertIn(b"edge-gui", resp.read())
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/assets/app.js", timeout=5) as resp:
                    self.assertEqual(resp.headers.get_content_type(), "text/javascript")
                    self.assertIn(b"console.log", resp.read())
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_proxy_rewrites_hub_id_to_loaded_path(self):
        """A short Hub id must not be forwarded — mlx-lm would download it."""
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        recorded: dict = {}

        class RecHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                recorded["body"] = json.loads(raw.decode() or "{}")
                payload = b'{"id":"chatcmpl-x","choices":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), RecHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        path = "/Users/me/.lmstudio/models/thetom-ai/MiniMax-M2.7-ConfigI-MLX"
        try:
            item = LoadedModel(id=path, engine="lm", model=path, port=engine_port, started_at=0.0)
            self.pool._models[item.id] = item
            status, _body = self._json(
                "POST",
                "/v1/chat/completions",
                {
                    "model": "thetom-ai/MiniMax-M2.7-ConfigI-MLX",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(recorded.get("body", {}).get("model"), path)
            self.assertEqual(recorded["body"]["messages"][0]["content"], "hello")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_scan_endpoint(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "mlx-community" / "Demo-4bit"
            model.mkdir(parents=True)
            (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"abcd")
            status, body = self._json("POST", "/v1/scan", {"dirs": [tmp]})
            self.assertEqual(status, 200)
            self.assertEqual(len(body.get("models") or []), 1)
            self.assertEqual(body["models"][0]["repo"], "mlx-community/Demo-4bit")
            self.assertEqual(body["errors"], [])

    def test_prefs_roundtrip(self):
        from tempfile import TemporaryDirectory
        from unittest import mock
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "studio.json"
            with mock.patch("mlx_edge.prefs.PREFS_PATH", path):
                status, body = self._json("GET", "/v1/prefs")
                self.assertEqual(status, 200)
                self.assertEqual(body.get("watchDirs"), [])
                status, body = self._json(
                    "PUT",
                    "/v1/prefs",
                    {"watchDirs": ["~/.lmstudio/models"], "flagsByModel": {"MiniMax": {"temp": 0.4}}},
                )
                self.assertEqual(status, 200)
                self.assertEqual(body.get("watchDirs"), ["~/.lmstudio/models"])
                status, body = self._json("GET", "/v1/prefs")
                self.assertEqual(body["flagsByModel"]["MiniMax"]["temp"], 0.4)

    def test_chat_resolves_basename_case_insensitive(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        recorded: dict = {}

        class RecHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                recorded["body"] = json.loads(self.rfile.read(length).decode() or "{}")
                payload = b'{"id":"chatcmpl-x","choices":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), RecHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        path = "/Users/me/.lmstudio/models/thetom-ai/MiniMax-M2.7-ConfigI-MLX"
        try:
            item = LoadedModel(
                id="MiniMax-M2.7-ConfigI-MLX",
                engine="lm",
                model=path,
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-ConfigI-MLX",
            )
            self.pool._models[item.id] = item
            status, _body = self._json(
                "POST",
                "/v1/chat/completions",
                {"model": "minimax-m2.7-configi-mlx", "messages": [{"role": "user", "content": "hello"}]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(recorded["body"]["model"], path)
            listed = self._json("GET", "/v1/models")[1]
            self.assertEqual(listed["data"][0]["id"], "MiniMax-M2.7-ConfigI-MLX")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_progress_idle_then_prefill_from_keepalive(self):
        from http.server import BaseHTTPRequestHandler
        import time

        from mlx_edge.pool import LoadedModel

        started = threading.Event()

        class SlowStream(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b": keepalive 2048/6540\n\n")
                self.wfile.flush()
                started.set()
                time.sleep(0.35)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
                self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), SlowStream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        path = "/models/MiniMax-M2.7-ConfigI-MLX"
        try:
            item = LoadedModel(
                id="MiniMax-M2.7-ConfigI-MLX",
                engine="lm",
                model=path,
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-ConfigI-MLX",
            )
            self.pool._models[item.id] = item
            self.pool.progress.ensure(item.public_id, "lm")
            idle = self._json("GET", "/v1/progress")[1]
            self.assertEqual(idle.get("object"), "edge.progress")
            self.assertFalse(idle.get("active"))

            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "MiniMax-M2.7-ConfigI-MLX",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    }
                ).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
                first = resp.read1(64) if hasattr(resp, "read1") else resp.read(64)
                elapsed = time.time() - t0
                self.assertLess(elapsed, 0.3)
                self.assertIn(b"keepalive", first)
                started.wait(1)
                snap = self._json("GET", "/v1/progress?model=minimax-m2.7-configi-mlx")[1]
                row = snap["models"][0]
                self.assertEqual(row["phase"], "prefill")
                self.assertEqual(row["prompt"]["processed_tokens"], 2048)
                self.assertEqual(row["prompt"]["total_tokens"], 6540)
                rest = resp.read()
            self.assertIn(b"[DONE]", first + rest)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_non_stream_chat_still_json(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        class RecHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                payload = b'{"id":"chatcmpl-x","choices":[{"message":{"content":"ok"}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), RecHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(id="demo", engine="lm", model="demo", port=engine_port, started_at=0.0, public_id="demo")
            self.pool._models[item.id] = item
            status, body = self._json(
                "POST",
                "/v1/chat/completions",
                {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
