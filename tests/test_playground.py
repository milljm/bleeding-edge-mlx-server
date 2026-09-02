import unittest

from mlx_edge.playground import PlaygroundStore, key_for


class PlaygroundStoreTests(unittest.TestCase):
    def test_key_uses_basename(self):
        self.assertEqual(key_for("/Users/x/MiniMax-M2.7-ConfigI-MLX"), "minimax-m2.7-configi-mlx")
        self.assertEqual(key_for("MiniMax-M2.7-ConfigI-MLX"), "minimax-m2.7-configi-mlx")

    def test_put_get_clear_stays_in_ram(self):
        store = PlaygroundStore()
        saved = store.put(
            "Qwen3-8B-4bit",
            [
                {"role": "user", "text": "hi"},
                {"role": "assistant", "text": "hello", "thinking": "hmm"},
                {"role": "system", "text": "drop me"},
            ],
        )
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[1]["thinking"], "hmm")
        self.assertEqual(store.get("qwen3-8b-4bit")[0]["text"], "hi")
        store.clear("Qwen3-8B-4bit")
        self.assertEqual(store.get("Qwen3-8B-4bit"), [])

    def test_models_are_isolated(self):
        store = PlaygroundStore()
        store.put("a", [{"role": "user", "text": "one"}])
        store.put("b", [{"role": "user", "text": "two"}])
        store.clear("a")
        self.assertEqual(store.get("b")[0]["text"], "two")
        self.assertEqual(store.get("a"), [])

    def test_empty_put_drops_key(self):
        store = PlaygroundStore()
        store.put("a", [{"role": "user", "text": "x"}])
        store.put("a", [])
        self.assertEqual(store.get("a"), [])
