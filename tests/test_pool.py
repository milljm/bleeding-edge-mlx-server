import json
import unittest
from unittest import mock

from mlx_edge.pool import (
    ModelPool,
    names_match,
    server_argv,
    spawn_argv,
    unique_public_id,
    wait_healthy,
)


class PoolTests(unittest.TestCase):
    def _pool(self) -> ModelPool:
        return ModelPool(spawn=lambda *_a, **_k: None, wait=lambda *_a, **_k: None)

    def test_hot_load_two_models(self):
        pool = self._pool()
        pool.load("lm", "mlx-community/Qwen3-8B-4bit")
        pool.load("vlm", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        self.assertEqual(len(pool.list()), 2)
        self.assertEqual(pool.resolve("mlx-community/Qwen3-8B-4bit").engine, "lm")
        self.assertEqual(pool.resolve("Qwen2.5-VL-7B-Instruct-4bit").engine, "vlm")
        self.assertEqual(pool.list()[0].public_id, "Qwen3-8B-4bit")

    def test_resolve_basename_case_insensitive(self):
        pool = self._pool()
        pool.load("lm", "/Users/me/.lmstudio/models/thetom-ai/MiniMax-M2.7-ConfigI-MLX")
        item = pool.resolve("minimax-m2.7-configi-mlx")
        self.assertIsNotNone(item)
        self.assertEqual(item.public_id, "MiniMax-M2.7-ConfigI-MLX")
        self.assertTrue(names_match("MiniMax-M2.7-ConfigI-MLX", "minimax-m2.7-configi-mlx"))

    def test_openai_id_is_basename(self):
        pool = self._pool()
        item = pool.load("lm", "/models/thetom-ai/MiniMax-M2.7-ConfigI-MLX")
        self.assertEqual(item.as_openai()["id"], "MiniMax-M2.7-ConfigI-MLX")
        self.assertEqual(unique_public_id("/a/MiniMax", ["MiniMax"]), "a/MiniMax")

    def test_reload_replaces_same_id(self):
        pool = self._pool()
        first = pool.load("lm", "qwen")
        second = pool.load("lm", "qwen")
        self.assertEqual(len(pool.list()), 1)
        self.assertGreaterEqual(second.started_at, first.started_at)
        self.assertEqual(second.args, [])
        third = pool.load("lm", "qwen", ["--temp", "0.7"])
        self.assertEqual(len(pool.list()), 1)
        self.assertEqual(third.args, ["--temp", "0.7"])

    def test_unload_leaves_others(self):
        pool = self._pool()
        pool.load("lm", "a")
        pool.load("lm", "b")
        pool.unload("a")
        self.assertIsNone(pool.resolve("a"))
        self.assertEqual(pool.resolve("b").model, "b")
        self.assertEqual(len(pool.list()), 1)

    def test_resolve_default_is_first(self):
        pool = self._pool()
        pool.load("lm", "first")
        pool.load("lm", "second")
        self.assertEqual(pool.resolve(None).model, "first")

    def test_openai_list_shape(self):
        pool = self._pool()
        item = pool.load("vlm", "vision")
        row = item.as_openai()
        self.assertEqual(row["id"], "vision")
        self.assertEqual(row["owned_by"], "mlx-vlm")
        self.assertEqual(row["object"], "model")

    def test_lm_server_argv_not_deprecated_module(self):
        argv = server_argv("lm")
        self.assertEqual(argv[-2:], ["mlx_lm", "server"])
        self.assertNotIn("mlx_lm.server", argv)

    def test_embed_spawn_uses_embedding_model_flag(self):
        argv = spawn_argv("embed", "/models/Qwen3-Embedding-0.6B", 9, ["--host", "0.0.0.0", "--model", "nope"])
        self.assertIn("mlx_vlm.server", argv)
        self.assertIn("--embedding-model", argv)
        self.assertEqual(argv[argv.index("--embedding-model") + 1], "/models/Qwen3-Embedding-0.6B")
        self.assertNotIn("--model", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "9")

    def test_embed_openai_owned_by(self):
        pool = self._pool()
        item = pool.load("embed", "/models/Qwen3-Embedding-0.6B")
        self.assertEqual(item.as_openai()["owned_by"], "mlx-embed")
        self.assertEqual(item.as_openai()["id"], "Qwen3-Embedding-0.6B")

    def test_warmup_embed_posts_embeddings(self):
        recorded: dict = {}

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"data":[]}'

        def fake_urlopen(req, timeout=None):
            recorded["url"] = req.full_url
            recorded["body"] = json.loads(req.data.decode())
            recorded["timeout"] = timeout
            return FakeResp()

        class DummyProc:
            stdout = None

            def poll(self):
                return None

            def terminate(self):
                return None

            def wait(self, timeout=None):
                return 0

            def kill(self):
                return None

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            pool = ModelPool(spawn=lambda *_a, **_k: DummyProc(), wait=lambda *_a, **_k: None)
            item = pool.load("embed", "/models/Qwen3-Embedding-0.6B")
        self.assertIn("/v1/embeddings", recorded["url"])
        self.assertEqual(recorded["body"]["model"], "/models/Qwen3-Embedding-0.6B")
        self.assertEqual(item.engine, "embed")

    def test_warmup_skipped_when_spawn_returns_none(self):
        with mock.patch("mlx_edge.pool.warmup_engine") as warm:
            pool = self._pool()
            pool.load("lm", "qwen")
            warm.assert_not_called()

    def test_wait_aborts_when_process_exits(self):
        class Dead:
            returncode = 9

            def poll(self):
                return 9

            def terminate(self):
                return None

            def wait(self, timeout=None):
                return 9

            def kill(self):
                return None

        pool = ModelPool(spawn=lambda *_a, **_k: Dead(), wait=wait_healthy)
        with self.assertRaises(RuntimeError) as ctx:
            pool.load("lm", "broken")
        self.assertIn("exited with code 9", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
