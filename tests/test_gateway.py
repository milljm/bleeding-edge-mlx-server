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
                "mlx-community/Qwen3-8B-4bit",
                "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
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
        self.assertEqual(body.get("models"), ["mlx-community/Qwen2.5-VL-7B-Instruct-4bit"])

        status, listed = self._json("GET", "/v1/models")
        self.assertEqual([row["id"] for row in listed["data"]], ["mlx-community/Qwen2.5-VL-7B-Instruct-4bit"])

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


if __name__ == "__main__":
    unittest.main()
