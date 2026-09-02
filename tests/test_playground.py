import unittest

from mlx_edge.playground import PlaygroundStore


class PlaygroundStoreTests(unittest.TestCase):
    def test_put_get_clear_stays_in_ram(self):
        store = PlaygroundStore()
        saved = store.put(
            [
                {"role": "user", "text": "hi"},
                {"role": "assistant", "text": "hello", "thinking": "hmm"},
                {"role": "system", "text": "drop me"},
            ],
        )
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[1]["thinking"], "hmm")
        self.assertEqual(store.get()[0]["text"], "hi")
        store.clear()
        self.assertEqual(store.get(), [])

    def test_one_thread_shared_across_models(self):
        store = PlaygroundStore()
        store.put([{"role": "user", "text": "from-a"}])
        store.put([{"role": "user", "text": "from-a"}, {"role": "assistant", "text": "from-b"}])
        self.assertEqual([t["text"] for t in store.get()], ["from-a", "from-b"])

    def test_empty_put_clears(self):
        store = PlaygroundStore()
        store.put([{"role": "user", "text": "x"}])
        store.put([])
        self.assertEqual(store.get(), [])

    def test_keeps_metrics(self):
        store = PlaygroundStore()
        saved = store.put(
            [
                {"role": "user", "text": "hi"},
                {
                    "role": "assistant",
                    "text": "hello",
                    "metrics": {"ttft": 1.25, "gen": 2.5, "tokens": 40, "tps": 16.0, "model": "Qwen3-8B-4bit"},
                },
            ],
        )
        self.assertEqual(saved[1]["metrics"]["tokens"], 40)
        self.assertEqual(saved[1]["metrics"]["model"], "Qwen3-8B-4bit")
        self.assertAlmostEqual(saved[1]["metrics"]["ttft"], 1.25)
