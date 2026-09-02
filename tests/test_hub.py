import json
import os
import signal
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from mlx_edge.hub import (
    JOB,
    download_repo,
    hub_downloaded_bytes,
    human_bytes,
    parse_repo,
    search_quants,
    search_stem,
    start_download,
    token_set,
)
from mlx_edge.scan import list_models


class InstantProc:
    stdout = None

    def poll(self):
        return 0

    def send_signal(self, sig):
        return None

    def terminate(self):
        return None

    def kill(self):
        return None

    def wait(self, timeout=None):
        return 0


class LiveProc:
    def __init__(self):
        self.stdout = None
        self.signals: list[int] = []
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def send_signal(self, sig):
        self.signals.append(sig)
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            self.alive = False

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        self.alive = False
        return 0


class HubTests(unittest.TestCase):
    def setUp(self):
        JOB.cancel.set()
        if JOB.thread and JOB.thread.is_alive():
            JOB.thread.join(timeout=1)
        JOB.phase = "idle"
        JOB.repo = ""
        JOB.error = ""
        JOB.path = ""
        JOB.n = 0
        JOB.total = 0
        JOB.proc = None
        JOB.cancel.clear()

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
            with mock.patch("mlx_edge.hub.repo_nbytes", return_value=100):
                with mock.patch("mlx_edge.hub._spawn_download", return_value=InstantProc()):
                    out = download_repo("mlx-community/SmolLM2-135M-Instruct-4bit")
        self.assertEqual(out["repo"], "mlx-community/SmolLM2-135M-Instruct-4bit")
        self.assertEqual(JOB.phase, "done")

    def test_pause_sends_sigstop(self):
        proc = LiveProc()
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch("mlx_edge.hub.repo_nbytes", return_value=1000):
                with mock.patch("mlx_edge.hub._spawn_download", return_value=proc):
                    start_download("mlx-community/SmolLM2-135M-Instruct-4bit")
                    for _ in range(20):
                        if JOB.proc is proc:
                            break
                        threading.Event().wait(0.05)
                    snap = JOB.request_pause()
                    self.assertEqual(snap["phase"], "paused")
                    self.assertIn(signal.SIGSTOP, proc.signals)
                    snap = JOB.request_resume()
                    self.assertEqual(snap["phase"], "downloading")
                    self.assertIn(signal.SIGCONT, proc.signals)
                    JOB.request_cancel()
                    self.assertFalse(proc.alive)

    def test_start_download_returns_before_finish(self):
        proc = LiveProc()
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch("mlx_edge.hub.repo_nbytes", return_value=100):
                with mock.patch("mlx_edge.hub._spawn_download", return_value=proc):
                    snap = start_download("mlx-community/SmolLM2-135M-Instruct-4bit")
                    self.assertEqual(snap["phase"], "downloading")
                    proc.alive = False
                    if JOB.thread:
                        JOB.thread.join(timeout=2)
        self.assertEqual(JOB.phase, "done")

    def test_hub_downloaded_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp)
            (hub / "blobs").mkdir()
            (hub / "incomplete").mkdir()
            (hub / "blobs" / "aaa").write_bytes(b"x" * 50)
            (hub / "incomplete" / "bbb").write_bytes(b"y" * 20)
            (hub / "snapshots").mkdir()
            (hub / "snapshots" / "link").symlink_to(hub / "blobs" / "aaa")
            self.assertEqual(hub_downloaded_bytes(hub), 70)
            self.assertEqual(human_bytes(1500), "1.5 KB")

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
