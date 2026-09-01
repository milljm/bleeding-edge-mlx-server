import unittest

from mlx_edge.logs import LogBuffer, classify


class LogTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify("Prompt processing progress: 1/2"), "progress")
        self.assertEqual(classify("ERROR failed to load"), "error")
        self.assertEqual(classify("WARNING kv cache"), "warn")
        self.assertEqual(classify('127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200'), "http")
        self.assertEqual(classify("hello"), "info")

    def test_snapshot_filter_and_clear(self):
        buf = LogBuffer(maxlen=10)
        buf.append("MiniMax", "lm", "Prompt processing progress: 2048/6540")
        buf.append("Qwen3-Embedding", "embed", "POST /v1/embeddings HTTP/1.1")
        snap = buf.snapshot()
        self.assertEqual(snap["object"], "edge.logs")
        self.assertEqual(len(snap["lines"]), 2)
        self.assertEqual(snap["lines"][0]["level"], "progress")
        only = buf.snapshot(model="minimax")
        self.assertEqual(len(only["lines"]), 1)
        self.assertEqual(only["lines"][0]["model"], "MiniMax")
        after = buf.snapshot(after=snap["seq"])
        self.assertEqual(after["lines"], [])
        buf.clear()
        self.assertEqual(buf.snapshot()["lines"], [])
        self.assertGreater(buf.seq(), snap["seq"])
