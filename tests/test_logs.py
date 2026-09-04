import unittest

from mlx_edge.logs import LogBuffer, classify, is_noise


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

    def test_is_noise_keep_hot_and_warmup(self):
        noise = [
            "Prompt Cache: 2 sequences, 0.06 GB",
            "- assistant: 2 sequences, 0.06 GB",
            "- user: 0 sequences, 0.00 GB",
            "- system: 0 sequences, 0.00 GB",
            "Prompt processing progress: 1/1",
            '127.0.0.1 - - [31/Aug/2026 23:00:46] "POST /v1/chat/completions HTTP/1.1" 200 -',
            'INFO:     127.0.0.1:63161 - "POST /v1/embeddings HTTP/1.1" 200 OK',
            "Generation queued: request=129cbf360 prompt_tokens=9 max_tokens=1 images=0 audio=0 videos=0",
            "Prefill started: request=129cbf360 backend=continuous_batching prompt_tokens=9 images=0",
            "Prefill progress: request=129cbf360 tokens=8/9 (88.9%)",
            "Prefill completed: request=129cbf360 prompt_tokens=9 cached_tokens=0 elapsed=0.208s rate=43.4 tok/s",
            "Decode completed: request=129cbf360 generated_tokens=1 elapsed=0.000s rate=0.0 tok/s finish_reason=length",
            "Request completed: endpoint=/chat/completions model=/m/Qwen stream=False backend=continuous_batching prompt_tokens=9 generated_tokens=1 elapsed=1.043s prefill=20.5 tok/s decode=0.0 tok/s finish_reason=length in_flight=10",
            "Request completed: endpoint=/v1/embeddings model=/m/embed stream=False backend=mlx-embeddings-native prompt_tokens=3 generated_tokens=0 elapsed=0.018s",
            '192.168.142.4 - "GET /v1/progress HTTP/1.1" 200 -',
            '192.168.142.4 - "GET /v1/progress?model=MiniMax-M2.7-8bit HTTP/1.1" 200 -',
            '192.168.142.4 - "GET /v1/host HTTP/1.1" 200 -',
        ]
        for line in noise:
            self.assertTrue(is_noise(line), line)

        keep = [
            "Prompt processing progress: 1024/1620",
            "Prompt processing progress: 1620/1620",
            '192.168.142.4 - "POST /v1/chat/completions HTTP/1.1" 200 -',
            '192.168.142.4 - "POST /v1/embeddings HTTP/1.1" 200 -',
            "Request completed: endpoint=/v1/embeddings model=/m/embed stream=False backend=mlx-embeddings-native prompt_tokens=12 generated_tokens=0 elapsed=0.106s",
            "ERROR failed to load weights",
        ]
        for line in keep:
            self.assertFalse(is_noise(line), line)

    def test_append_drops_noise(self):
        buf = LogBuffer()
        buf.append("MiniMax", "lm", "Prompt Cache: 1 sequences, 0.00 GB")
        buf.append("MiniMax", "lm", "Prompt processing progress: 1/1")
        buf.append("MiniMax", "lm", '127.0.0.1 - - [31/Aug/2026 23:00:46] "POST /v1/chat/completions HTTP/1.1" 200 -')
        buf.append("MiniMax", "lm", "Prompt processing progress: 1024/1620")
        snap = buf.snapshot()
        self.assertEqual(len(snap["lines"]), 1)
        self.assertIn("1024/1620", snap["lines"][0]["text"])


if __name__ == "__main__":
    unittest.main()
