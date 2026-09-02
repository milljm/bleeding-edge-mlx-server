import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from mlx_edge.hub import JOB, download_repo, parse_repo, search_quants, search_stem, start_download, token_set
from mlx_edge.scan import list_models


class HubTests(unittest.TestCase):
    def setUp(self):
        JOB.phase = "idle"
        JOB.repo = ""
        JOB.error = ""
        JOB.path = ""
        JOB.n = 0
        JOB.total = 0
        JOB.bars = []
        JOB.cancel.clear()
        JOB.pause.set()

    def test_parse_url_and_repo(self):
        self.assertEqual(
            parse_repo("https://huggingface.co/mlx-community/Qwen3-8B-4bit"),
            "mlx-community/Qwen3-8B-4bit",
        )
        self.assertEqual(parse_repo("mlx-community/SmolLM2-135M-Instruct-4bit"), "mlx-community/SmolLM2-135M-Instruct-4bit")
        self.assertEqual(search_stem("mlx-community/Qwen3-8B-4bit"), "Qwen3-8B")
        with self.assertRaises(ValueError):
            parse_repo("")
        with self.assertRaises(ValueError):
            parse_repo("???")

    def test_search_prefers_mlx_community_quants(self):
        rows = [
            {"id": "mlx-community/Qwen3-8B-4bit", "library_name": "mlx", "downloads": 10, "tags": ["mlx"]},
            {"id": "mlx-community/Qwen3-8B-8bit", "library_name": "mlx", "downloads": 3, "tags": ["mlx"]},
            {"id": "mlx-community/Qwen3-8B-1bit", "library_name": "mlx", "downloads": 1, "tags": ["mlx"]},
            {"id": "Qwen/Qwen3-8B", "library_name": "transformers", "downloads": 99, "tags": []},
        ]

        def fake_search(search, author=None, filt=None, limit=20):
            return list(rows)

        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch("mlx_edge.hub.hf_search", fake_search):
                out = search_quants("https://huggingface.co/Qwen/Qwen3-8B")
        ids = [r["id"] for r in out["results"]]
        self.assertEqual(ids[0], "mlx-community/Qwen3-8B-4bit")
        self.assertIn("mlx-community/Qwen3-8B-8bit", ids)
        self.assertNotIn("mlx-community/Qwen3-8B-1bit", ids)
        self.assertNotIn("Qwen/Qwen3-8B", ids)
        self.assertEqual(out["stem"], "Qwen3-8B")

    def test_search_requires_token(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PermissionError):
                search_quants("mlx-community/Qwen3-8B-4bit")

    def test_token_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(token_set())
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_abc"}, clear=True):
            self.assertTrue(token_set())

    def test_download_rejects_1bit(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with self.assertRaises(ValueError):
                download_repo("mlx-community/Bonsai-4B-mlx-1bit")

    def test_download_uses_snapshot(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch("mlx_edge.hub.repo_nbytes", return_value=0):
                with mock.patch("mlx_edge.hub._snapshot", return_value="/cache/snap") as snap:
                    out = download_repo("mlx-community/SmolLM2-135M-Instruct-4bit")
        snap.assert_called_once()
        self.assertEqual(out["path"], "/cache/snap")
        self.assertEqual(out["repo"], "mlx-community/SmolLM2-135M-Instruct-4bit")
        self.assertEqual(JOB.phase, "done")

    def test_pause_resume_cancel(self):
        JOB.repo = "mlx-community/Qwen3-8B-4bit"
        JOB.phase = "downloading"
        JOB.request_pause()
        self.assertEqual(JOB.phase, "paused")
        JOB.request_resume()
        self.assertEqual(JOB.phase, "downloading")
        JOB.request_cancel()
        self.assertEqual(JOB.phase, "cancelled")

    def test_start_download_returns_before_finish(self):
        gate = threading.Event()

        def slow_snap(repo, tqdm_class=None):
            gate.wait(timeout=2)
            return "/cache/snap"

        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch("mlx_edge.hub.repo_nbytes", return_value=100):
                with mock.patch("mlx_edge.hub._snapshot", side_effect=slow_snap):
                    snap = start_download("mlx-community/SmolLM2-135M-Instruct-4bit")
                    self.assertEqual(snap["phase"], "downloading")
                    gate.set()
                    if JOB.thread:
                        JOB.thread.join(timeout=2)
        self.assertEqual(JOB.phase, "done")

    def test_incomplete_hub_snapshot_is_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub = root / "models--mlx-community--SmolLM2-135M-Instruct-4bit"
            snap = hub / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text(json.dumps({"model_type": "llama", "quantization": {"bits": 4}}), encoding="utf-8")
            (snap / "model.safetensors").write_bytes(b"w" * 64)
            (hub / "incomplete").mkdir()
            (hub / "incomplete" / "tmp").write_bytes(b"x")
            (hub / "blobs").mkdir()
            (hub / "refs").mkdir()
            self.assertEqual(list_models(str(root)), [])
            (hub / "incomplete" / "tmp").unlink()
            models = list_models(str(root))
            self.assertEqual([m["repo"] for m in models], ["mlx-community/SmolLM2-135M-Instruct-4bit"])


if __name__ == "__main__":
    unittest.main()
