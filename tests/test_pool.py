import unittest

from mlx_edge.pool import ModelPool


class PoolTests(unittest.TestCase):
    def _pool(self) -> ModelPool:
        return ModelPool(spawn=lambda *_a, **_k: None, wait=lambda _port: None)

    def test_hot_load_two_models(self):
        pool = self._pool()
        pool.load("lm", "mlx-community/Qwen3-8B-4bit")
        pool.load("vlm", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
        self.assertEqual(len(pool.list()), 2)
        self.assertEqual(pool.resolve("mlx-community/Qwen3-8B-4bit").engine, "lm")
        self.assertEqual(pool.resolve("Qwen2.5-VL-7B-Instruct-4bit").engine, "vlm")

    def test_reload_replaces_same_id(self):
        pool = self._pool()
        first = pool.load("lm", "qwen", ["--temp", "0"])
        second = pool.load("lm", "qwen", ["--temp", "0.7"])
        self.assertEqual(len(pool.list()), 1)
        self.assertGreaterEqual(second.started_at, first.started_at)
        self.assertEqual(pool.list()[0].args, ["--temp", "0.7"])

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


if __name__ == "__main__":
    unittest.main()
