import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from mlx_edge.gateway import _quiet_access, make_handler
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

    def test_hub_search_and_download(self):
        from unittest import mock

        fake = {
            "query": "Qwen/Qwen3-8B",
            "repo": "Qwen/Qwen3-8B",
            "stem": "Qwen3-8B",
            "token": True,
            "results": [{"id": "mlx-community/Qwen3-8B-4bit", "quant": "4-bit", "downloads": 1}],
        }
        with mock.patch("mlx_edge.hub.token_set", return_value=True):
            with mock.patch("mlx_edge.hub.search_quants", return_value=fake):
                status, body = self._json("POST", "/v1/hub/search", {"query": "Qwen/Qwen3-8B"})
            self.assertEqual(status, 200)
            self.assertEqual(body["results"][0]["id"], "mlx-community/Qwen3-8B-4bit")
            status, body = self._json("GET", "/v1/hub")
            self.assertEqual(status, 200)
            self.assertTrue(body.get("token"))
            started = {
                "repo": "mlx-community/Qwen3-8B-4bit",
                "phase": "downloading",
                "bytes": 0,
                "total": 10,
                "ratio": 0,
                "error": "",
                "path": "",
                "token": True,
            }
            with mock.patch("mlx_edge.hub.start_download", return_value=started):
                status, body = self._json("POST", "/v1/hub/download", {"repo": "mlx-community/Qwen3-8B-4bit"})
            self.assertEqual(status, 200)
            self.assertEqual(body["phase"], "downloading")
            status, body = self._json("GET", "/v1/hub/progress")
            self.assertEqual(status, 200)
            self.assertIn("jobs", body)

    def test_hub_search_without_token(self):
        from unittest import mock

        with mock.patch("mlx_edge.hub.token_set", return_value=False):
            status, body = self._json("POST", "/v1/hub/search", {"query": "Qwen/Qwen3-8B"})
        self.assertEqual(status, 403)

    def test_playground_ram_roundtrip(self):
        status, body = self._json(
            "PUT",
            "/v1/playground",
            {
                "turns": [
                    {"role": "user", "text": "hi"},
                    {"role": "assistant", "text": "hello", "thinking": "plan"},
                ],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body.get("turns") or []), 2)
        status, body = self._json("GET", "/v1/playground")
        self.assertEqual(status, 200)
        self.assertEqual(body["turns"][1]["thinking"], "plan")
        status, body = self._json(
            "PUT",
            "/v1/playground",
            {"model": "ignored", "turns": [{"role": "user", "text": "still-shared"}]},
        )
        self.assertEqual(status, 200)
        status, body = self._json("GET", "/v1/playground")
        self.assertEqual(body["turns"][0]["text"], "still-shared")
        status, body = self._json("DELETE", "/v1/playground")
        self.assertEqual(status, 200)
        status, body = self._json("GET", "/v1/playground")
        self.assertEqual(body.get("turns"), [])

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
                first = b""
                while b"keepalive" not in first and time.time() - t0 < 1.0:
                    piece = resp.read1(256) if hasattr(resp, "read1") else resp.read(256)
                    if not piece:
                        break
                    first += piece
                elapsed = time.time() - t0
                self.assertLess(elapsed, 0.3)
                self.assertIn(b"assistant", first)
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

    def test_minimax_held_answer_emitted_before_done(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        class MiniMaxStream(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hello from MiniMax."}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), MiniMaxStream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(
                id="MiniMax-M2.7-8bit",
                engine="lm",
                model="/models/MiniMax-M2.7-8bit",
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-8bit",
            )
            self.pool._models[item.id] = item
            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "MiniMax-M2.7-8bit",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    }
                ).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
            self.assertIn("Hello from MiniMax.", body)
            done_at = body.rfind("data: [DONE]")
            hello_at = body.find("Hello from MiniMax.")
            self.assertGreaterEqual(hello_at, 0)
            self.assertGreater(done_at, hello_at)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_minimax_think_close_streams_answer(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        class ThinkStream(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"plan"}}]}\n\n')
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"</think>Apple is at $1."}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), ThinkStream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(
                id="MiniMax-M2.7-8bit",
                engine="lm",
                model="/models/MiniMax-M2.7-8bit",
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-8bit",
            )
            self.pool._models[item.id] = item
            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "MiniMax-M2.7-8bit",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    }
                ).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
            self.assertIn("Apple is at $1.", body)
            self.assertNotIn("</think>", body)
            self.assertIn("reasoning_content", body)
            self.assertGreater(body.rfind("data: [DONE]"), body.find("Apple is at $1."))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_configi_streams_content_on_each_delta(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        class ConfigIStream(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"Here"}}]}\n\n')
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"\'s a plot"}}]}\n\n')
                self.wfile.write(b'data: {"choices":[{"delta":{"finish_reason":"stop"}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), ConfigIStream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(
                id="MiniMax-M2.7-ConfigI-MLX",
                engine="lm",
                model="/Users/milljm/.lmstudio/models/thetom-ai/MiniMax-M2.7-ConfigI-MLX",
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-ConfigI-MLX",
            )
            self.pool._models[item.id] = item
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
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
            contents = []
            for block in body.split("\n\n"):
                if not block.startswith("data:") or "[DONE]" in block:
                    continue
                payload = json.loads(block.split("data:", 1)[1].strip())
                delta = (payload.get("choices") or [{}])[0].get("delta") or {}
                if "content" in delta:
                    contents.append(delta["content"])
            self.assertEqual(contents[0], "Here")
            self.assertIn("'s a plot", contents)
            self.assertNotIn("reasoning_content", body)
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

    def test_stream_asks_child_for_usage(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        seen = {}

        class RecHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                seen["body"] = json.loads(self.rfile.read(length).decode())
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
                self.wfile.write(
                    b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":1,"total_tokens":13}}\n\n'
                )
                self.wfile.write(b"data: [DONE]\n\n")

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), RecHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(id="demo", engine="lm", model="/m/demo", port=engine_port, started_at=0.0, public_id="demo")
            self.pool._models[item.id] = item
            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                data=json.dumps({"model": "demo", "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
            self.assertTrue(seen["body"]["stream_options"]["include_usage"])
            self.assertEqual(seen["body"]["model"], "/m/demo")
            self.assertIn("prompt_tokens", body)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_minimax_xml_becomes_openai_tool_calls(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        xml = (
            "<minimax:tool_call>\n<invoke name=\"read_file\">"
            "<parameter name=\"path\">a.py</parameter></invoke>\n</minimax:tool_call>"
        )

        class ToolStream(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                payload = json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": xml},
                            }
                        ]
                    }
                ).encode()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), ToolStream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(
                id="MiniMax-M2.7-8bit",
                engine="lm",
                model="/m/MiniMax-M2.7-8bit",
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-8bit",
            )
            self.pool._models[item.id] = item
            status, body = self._json(
                "POST",
                "/v1/chat/completions",
                {
                    "model": "MiniMax-M2.7-8bit",
                    "messages": [{"role": "user", "content": "read a.py"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                            },
                        }
                    ],
                },
            )
            self.assertEqual(status, 200)
            choice = body["choices"][0]
            self.assertEqual(choice["finish_reason"], "tool_calls")
            self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "read_file")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_embeddings_proxy_rewrites_model_and_rejects_chat_engine(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        recorded: dict = {}

        class RecHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                recorded["path"] = self.path
                recorded["body"] = json.loads(self.rfile.read(length).decode() or "{}")
                payload = b'{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), RecHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        path = "/models/Qwen3-Embedding-0.6B-4bit"
        try:
            item = LoadedModel(
                id="Qwen3-Embedding-0.6B-4bit",
                engine="embed",
                model=path,
                port=engine_port,
                started_at=0.0,
                public_id="Qwen3-Embedding-0.6B-4bit",
            )
            self.pool._models[item.id] = item
            status, body = self._json(
                "POST",
                "/v1/embeddings",
                {"model": "qwen3-embedding-0.6b-4bit", "input": "hello rag"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(recorded["path"], "/v1/embeddings")
            self.assertEqual(recorded["body"]["model"], path)
            self.assertEqual(body["data"][0]["embedding"], [0.1, 0.2])

            chat = LoadedModel(id="demo", engine="lm", model="demo", port=engine_port, started_at=1.0, public_id="demo")
            self.pool._models[chat.id] = chat
            status, body = self._json("POST", "/v1/embeddings", {"model": "demo", "input": "nope"})
            self.assertEqual(status, 400)
            self.assertIn("does not serve embeddings", (body.get("error") or {}).get("message", ""))

            status, body = self._json(
                "POST",
                "/v1/chat/completions",
                {"model": "Qwen3-Embedding-0.6B-4bit", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(status, 400)
            self.assertIn("embed model", (body.get("error") or {}).get("message", ""))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_tts_proxy_and_rejects_chat(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        recorded: dict = {}

        class RecHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                recorded["path"] = self.path
                recorded["body"] = json.loads(self.rfile.read(length).decode() or "{}")
                payload = b"ID3fake-mp3"
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), RecHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        path = "/models/Kokoro-82M"
        try:
            item = LoadedModel(
                id="Kokoro-82M",
                engine="tts",
                model=path,
                port=engine_port,
                started_at=0.0,
                public_id="Kokoro-82M",
            )
            self.pool._models[item.id] = item
            req = urllib.request.Request(
                self.base + "/v1/audio/speech",
                data=json.dumps({"model": "kokoro-82m", "input": "hello"}).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.read(), b"ID3fake-mp3")
            self.assertEqual(recorded["path"], "/v1/audio/speech")
            self.assertEqual(recorded["body"]["model"], path)

            status, body = self._json(
                "POST",
                "/v1/chat/completions",
                {"model": "Kokoro-82M", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(status, 400)
            self.assertIn("/v1/audio/speech", (body.get("error") or {}).get("message", ""))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_rerank_proxy_pins_model(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        recorded: dict = {}

        class RecHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                recorded["path"] = self.path
                recorded["body"] = json.loads(self.rfile.read(length).decode() or "{}")
                payload = b'{"results":[{"index":1,"relevance_score":0.9}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), RecHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        path = "/models/Qwen3-Reranker-0.6B-4bit"
        try:
            item = LoadedModel(
                id="Qwen3-Reranker-0.6B-4bit",
                engine="rerank",
                model=path,
                port=engine_port,
                started_at=0.0,
                public_id="Qwen3-Reranker-0.6B-4bit",
            )
            self.pool._models[item.id] = item
            status, body = self._json(
                "POST",
                "/v1/rerank",
                {"model": "qwen3-reranker-0.6b-4bit", "query": "capital", "documents": ["a", "b"]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(recorded["path"], "/v1/rerank")
            self.assertEqual(recorded["body"]["model"], path)
            self.assertEqual(body["results"][0]["index"], 1)

            status, body = self._json(
                "POST",
                "/v1/chat/completions",
                {"model": "Qwen3-Reranker-0.6B-4bit", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(status, 400)
            self.assertIn("/v1/rerank", (body.get("error") or {}).get("message", ""))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_logs_template_progress_float_and_channel_strip(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        self.pool.logs.append("demo", "lm", "Prompt processing progress: 1/2")
        status, body = self._json("GET", "/v1/logs")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("object"), "edge.logs")
        self.assertEqual(body["lines"][0]["level"], "progress")
        status, body = self._json("POST", "/v1/logs/clear")
        self.assertEqual(status, 200)
        self.assertEqual(self._json("GET", "/v1/logs")[1]["lines"], [])

        status, body = self._json("GET", "/v1/template?model=MiniMax-M2.7-ConfigI-MLX")
        self.assertEqual(status, 200)
        self.assertEqual(body.get("preset"), "harmony")

        idle = self._json("GET", "/v1/progress")[1]
        self.assertIsInstance(idle.get("progress"), (int, float))
        self.assertEqual(idle["progress"], 0.0)

        class RecHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                payload = (
                    b'{"choices":[{"message":{"role":"assistant","content":'
                    b'"<|channel|>analysis<|message|>plan<|end|>'
                    b'<|start|>assistant<|channel|>final<|message|>ok"}}]}'
                )
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
            item = LoadedModel(
                id="MiniMax-M2.7-ConfigI-MLX",
                engine="lm",
                model="/models/MiniMax-M2.7-ConfigI-MLX",
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-ConfigI-MLX",
            )
            self.pool._models[item.id] = item
            status, body = self._json(
                "POST",
                "/v1/chat/completions",
                {"model": "minimax-m2.7-configi-mlx", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(status, 200)
            msg = body["choices"][0]["message"]
            self.assertEqual(msg["content"], "ok")
            self.assertEqual(msg["reasoning_content"], "plan")
            self.assertNotIn("<|channel|>", msg["content"])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_quiet_progress_polls(self):
        self.assertTrue(_quiet_access('"GET /v1/progress HTTP/1.1" 200 -'))
        self.assertTrue(_quiet_access('"GET /v1/progress?model=MiniMax-M2.7-8bit HTTP/1.1" 200 -'))
        self.assertTrue(_quiet_access('"GET /v1/logs/stream HTTP/1.1" 200 -'))
        self.assertFalse(_quiet_access('"POST /v1/chat/completions HTTP/1.1" 200 -'))
        self.assertFalse(_quiet_access('"POST /v1/embeddings HTTP/1.1" 200 -'))

    def test_models_list_and_retrieve_include_context(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from mlx_edge.pool import LoadedModel

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "Qwen3-8B-4bit"
            path.mkdir()
            (path / "config.json").write_text(
                '{"model_type":"qwen3","max_position_embeddings":40960}',
                encoding="utf-8",
            )
            item = LoadedModel(
                id="Qwen3-8B-4bit",
                engine="lm",
                model=str(path),
                port=1,
                started_at=0.0,
                public_id="Qwen3-8B-4bit",
            )
            self.pool._models[item.id] = item
            status, body = self._json("GET", "/v1/models")
            self.assertEqual(status, 200)
            row = body["data"][0]
            self.assertEqual(row["id"], "Qwen3-8B-4bit")
            self.assertEqual(row["context_length"], 40960)
            self.assertEqual(row["max_model_len"], 40960)
            status, one = self._json("GET", "/v1/models/qwen3-8b-4bit")
            self.assertEqual(status, 200)
            self.assertEqual(one["max_context_length"], 40960)
            status, missing = self._json("GET", "/v1/models/not-loaded")
            self.assertEqual(status, 404)
            status, native = self._json("GET", "/api/v0/models")
            self.assertEqual(status, 200)
            self.assertEqual(native["data"][0]["type"], "llm")
            self.assertEqual(native["data"][0]["loaded_context_length"], 40960)
            self.assertTrue(native["data"][0]["capabilities"]["tool_use"])
            self.assertTrue(row["capabilities"]["function_calling"])

    def test_stop_aborts_stream_and_clears_progress(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        released = threading.Event()
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
                self.wfile.write(b": keepalive 1/8\n\n")
                self.wfile.flush()
                started.set()
                try:
                    for _ in range(40):
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":"."}}]}\n\n')
                        self.wfile.flush()
                        time.sleep(0.05)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    released.set()

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), SlowStream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(
                id="MiniMax-M2.7-ConfigI-MLX",
                engine="lm",
                model="/models/MiniMax-M2.7-ConfigI-MLX",
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-ConfigI-MLX",
            )
            self.pool._models[item.id] = item
            self.pool.progress.ensure(item.public_id, "lm")
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
            result: dict = {}

            def consume() -> None:
                try:
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        result["body"] = resp.read()
                except Exception as exc:  # noqa: BLE001
                    result["error"] = exc

            worker = threading.Thread(target=consume, daemon=True)
            worker.start()
            self.assertTrue(started.wait(2))
            snap = self._json("GET", "/v1/progress")[1]
            self.assertTrue(snap.get("active"))
            status, body = self._json("POST", "/v1/stop", {"model": "MiniMax-M2.7-ConfigI-MLX"})
            self.assertEqual(status, 200)
            self.assertEqual(body.get("stopped"), ["MiniMax-M2.7-ConfigI-MLX"])
            worker.join(3)
            self.assertTrue(released.wait(2))
            idle = self._json("GET", "/v1/progress")[1]
            self.assertFalse(idle.get("active"))
            self.assertEqual(idle["models"][0]["phase"], "idle")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_client_disconnect_stops_generation(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        released = threading.Event()
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
                self.wfile.write(b": keepalive 2/8\n\n")
                self.wfile.flush()
                started.set()
                try:
                    for _ in range(40):
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":"."}}]}\n\n')
                        self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    released.set()

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), SlowStream)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(
                id="MiniMax-M2.7-ConfigI-MLX",
                engine="lm",
                model="/models/MiniMax-M2.7-ConfigI-MLX",
                port=engine_port,
                started_at=0.0,
                public_id="MiniMax-M2.7-ConfigI-MLX",
            )
            self.pool._models[item.id] = item
            self.pool.progress.ensure(item.public_id, "lm")
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
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertTrue(started.wait(2))
                resp.read(32)
            self.assertTrue(released.wait(2.5))
            # give the gateway finally-block a tick to cancel progress
            deadline = time.time() + 1.5
            phase = None
            while time.time() < deadline:
                snap = self._json("GET", "/v1/progress")[1]
                phase = (snap.get("models") or [{}])[0].get("phase")
                if not snap.get("active") and phase == "idle":
                    break
                time.sleep(0.05)
            self.assertEqual(phase, "idle")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_stream_role_before_slow_engine(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        class SlowStart(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                time.sleep(0.55)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b": keepalive 1/8\n\n")
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), SlowStart)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(
                id="demo",
                engine="lm",
                model="demo",
                port=engine_port,
                started_at=0.0,
                public_id="demo",
            )
            self.pool._models[item.id] = item
            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "demo",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    }
                ).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=5) as resp:
                first = resp.read1(180) if hasattr(resp, "read1") else resp.read(180)
                self.assertLess(time.time() - t0, 0.35)
                self.assertIn(b"assistant", first)
                rest = resp.read()
            self.assertIn(b"hi", first + rest)
            self.assertIn(b"[DONE]", first + rest)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_heartbeat_fills_prefill_stall(self):
        from http.server import BaseHTTPRequestHandler

        from mlx_edge.pool import LoadedModel

        class Stall(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b": keepalive 1/8\n\n")
                self.wfile.flush()
                time.sleep(2.4)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        engine_port = free_port()
        httpd = ThreadingHTTPServer(("127.0.0.1", engine_port), Stall)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            item = LoadedModel(
                id="demo",
                engine="lm",
                model="demo",
                port=engine_port,
                started_at=0.0,
                public_id="demo",
            )
            self.pool._models[item.id] = item
            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "demo",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    }
                ).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read()
            self.assertIn(b": heartbeat", body)
            self.assertIn(b"hi", body)
            self.assertIn(b"[DONE]", body)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_stop_with_nothing_inflight(self):
        status, body = self._json("POST", "/v1/stop", {"model": "missing"})
        self.assertEqual(status, 200)
        self.assertEqual(body.get("stopped"), [])


if __name__ == "__main__":
    unittest.main()
