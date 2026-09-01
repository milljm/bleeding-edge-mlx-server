import unittest

from mlx_edge.progress import ProgressTracker, parse_progress_text, parse_sse_event


class ProgressTests(unittest.TestCase):
    def test_parse_mlx_lm_prefill_log(self):
        event = parse_progress_text("2026-02-07 22:01:02,124 - INFO - Prompt processing progress: 2048/6540")
        self.assertEqual(event, {"kind": "prefill", "processed": 2048, "total": 6540})

    def test_parse_keepalive_comment(self):
        event = parse_sse_event(": keepalive 4096/6540")
        self.assertEqual(event, {"kind": "prefill", "processed": 4096, "total": 6540})

    def test_parse_vlm_prefill(self):
        event = parse_progress_text("Prefill progress: request=abc tokens=1024/4096 (25.0%)")
        self.assertEqual(event, {"kind": "prefill", "processed": 1024, "total": 4096})
        start = parse_progress_text("Prefill started: request=abc backend=mlx prompt_tokens=4096 images=0 audio=0 videos=0")
        self.assertEqual(start, {"kind": "prefill_start", "total": 4096})

    def test_parse_sse_token_and_done(self):
        delta = parse_sse_event('data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}')
        self.assertEqual(delta["kind"], "decode_delta")
        self.assertEqual(delta["text"], "Hi")
        self.assertEqual(parse_sse_event("data: [DONE]"), {"kind": "done"})

    def test_snapshot_schema_and_prefill_ratio(self):
        tracker = ProgressTracker(linger=0)
        tracker.ensure("MiniMax-M2.7-ConfigI-MLX", "lm")
        tracker.begin("MiniMax-M2.7-ConfigI-MLX", "lm", stream=True)
        tracker.ingest_log("MiniMax-M2.7-ConfigI-MLX", "lm", "Prompt processing progress: 2048/6540")
        snap = tracker.snapshot("minimax-m2.7-configi-mlx")
        self.assertEqual(snap["object"], "edge.progress")
        self.assertEqual(snap["version"], 1)
        self.assertTrue(snap["active"])
        row = snap["models"][0]
        self.assertEqual(row["id"], "MiniMax-M2.7-ConfigI-MLX")
        self.assertEqual(row["phase"], "prefill")
        self.assertEqual(row["status"], "processing")
        self.assertEqual(row["prompt"]["processed_tokens"], 2048)
        self.assertEqual(row["prompt"]["total_tokens"], 6540)
        self.assertEqual(row["prompt"]["ratio"], 0.3131)
        self.assertIsInstance(row["progress"], float)
        self.assertEqual(row["progress"], 0.3131)
        self.assertIsInstance(snap["progress"], float)
        self.assertEqual(snap["progress"], 0.3131)
        tracker.apply("MiniMax-M2.7-ConfigI-MLX", "lm", {"kind": "decode_delta", "text": "hello"})
        snap = tracker.snapshot()
        self.assertEqual(snap["models"][0]["phase"], "decode")
        self.assertEqual(snap["models"][0]["generation"]["tokens"], 1)
        self.assertEqual(snap["models"][0]["progress"], 1.0)
        self.assertEqual(snap["progress"], 1.0)
        tracker.complete("MiniMax-M2.7-ConfigI-MLX")
        snap = tracker.snapshot()
        self.assertEqual(snap["models"][0]["phase"], "idle")
        self.assertEqual(snap["models"][0]["progress"], 0.0)
        self.assertEqual(snap["progress"], 0.0)
        self.assertFalse(snap["active"])

    def test_sse_buffer_split_across_chunks(self):
        tracker = ProgressTracker(linger=0)
        tracker.begin("m", "lm", stream=True)
        leftover = tracker.ingest_sse("m", b": keepalive 10")
        leftover = tracker.ingest_sse("m", leftover + b"/20\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
        leftover = tracker.ingest_sse("m", leftover + b"data: [DONE]\n\n")
        self.assertEqual(leftover, b"")
        snap = tracker.snapshot()
        self.assertEqual(snap["models"][0]["phase"], "idle")


if __name__ == "__main__":
    unittest.main()
