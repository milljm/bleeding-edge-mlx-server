import json
import os
import signal
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from mlx_edge.hub import (
    QUEUE,
    delete_hub_repo,
    download_progress,
    download_repo,
    hub_downloaded_bytes,
    human_bytes,
    parse_repo,
    repo_nbytes,
    resolve_hub_delete_target,
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
        QUEUE.clear()

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
        jobs = download_progress()["jobs"]
        self.assertEqual(jobs[-1]["phase"], "done")

    def test_pause_sends_sigstop(self):
        proc = LiveProc()
        repo = "mlx-community/SmolLM2-135M-Instruct-4bit"
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch("mlx_edge.hub.repo_nbytes", return_value=1000):
                with mock.patch("mlx_edge.hub._spawn_download", return_value=proc):
                    start_download(repo)
                    job = QUEUE.jobs[repo]
                    for _ in range(20):
                        if job.proc is proc:
                            break
                        threading.Event().wait(0.05)
                    snap = QUEUE.pause(repo)
                    self.assertEqual(snap["phase"], "paused")
                    self.assertIn(signal.SIGSTOP, proc.signals)
                    snap = QUEUE.resume(repo)
                    self.assertEqual(snap["phase"], "downloading")
                    self.assertIn(signal.SIGCONT, proc.signals)
                    QUEUE.cancel(repo)
                    self.assertFalse(proc.alive)

    def test_start_download_returns_before_finish(self):
        proc = LiveProc()
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch("mlx_edge.hub.repo_nbytes", return_value=100):
                with mock.patch("mlx_edge.hub._spawn_download", return_value=proc):
                    snap = start_download("mlx-community/SmolLM2-135M-Instruct-4bit")
                    self.assertEqual(snap["phase"], "downloading")
                    proc.alive = False
                    job = QUEUE.jobs[snap["repo"]]
                    if job.thread:
                        job.thread.join(timeout=2)
        self.assertEqual(QUEUE.jobs[snap["repo"]].phase, "done")

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

    def test_delete_only_hf_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            hub = home / "hub"
            folder = hub / "models--mlx-community--Toy-4bit"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}", encoding="utf-8")
            other = Path(tmp) / "lmstudio" / "Toy"
            other.mkdir(parents=True)
            (other / "weights.safetensors").write_bytes(b"x")
            with mock.patch.dict(os.environ, {"HF_HOME": str(home)}, clear=False):
                with self.assertRaises(PermissionError):
                    resolve_hub_delete_target(str(other))
                out = delete_hub_repo("mlx-community/Toy-4bit")
            self.assertFalse(folder.exists())
            self.assertTrue(other.exists())
            self.assertEqual(out["repo"], "mlx-community/Toy-4bit")

    def test_repo_nbytes_prefers_lfs_size(self):
        payload = {"siblings": [{"rfilename": "w.safetensors", "size": 130, "lfs": {"size": 4_000_000}}]}

        class Resp:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with mock.patch("urllib.request.urlopen", return_value=Resp()):
            self.assertEqual(repo_nbytes("mlx-community/Toy-4bit"), 4_000_000)

    def test_repo_nbytes_keeps_slash_and_used_storage(self):
        seen: list[str] = []

        def fake_open(req, timeout=12):
            seen.append(req.full_url)

            class Resp:
                def read(self):
                    return json.dumps({"siblings": [{"size": 10}], "usedStorage": 99}).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Resp()

        with mock.patch("urllib.request.urlopen", fake_open):
            n = repo_nbytes("mlx-community/gemma-4-e2b-it-4bit")
        self.assertEqual(n, 10)
        self.assertTrue(any("mlx-community/gemma-4-e2b-it-4bit?blobs=true" in url for url in seen))
        self.assertFalse(any("%2F" in url for url in seen))

        def storage_only(req, timeout=12):
            class Resp:
                def read(self):
                    return json.dumps({"siblings": [], "usedStorage": 7163942076}).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Resp()

        with mock.patch("urllib.request.urlopen", storage_only):
            self.assertEqual(repo_nbytes("mlx-community/gemma-4-e2b-it-4bit"), 7163942076)

    def test_two_downloads_no_duplicate(self):
        a, b = LiveProc(), LiveProc()

        def spawn(repo):
            return a if "Smol" in repo else b

        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch("mlx_edge.hub.repo_nbytes", return_value=1000):
                with mock.patch("mlx_edge.hub._spawn_download", side_effect=spawn):
                    start_download("mlx-community/SmolLM2-135M-Instruct-4bit")
                    start_download("mlx-community/Qwen3-8B-4bit")
                    again = start_download("mlx-community/SmolLM2-135M-Instruct-4bit")
        jobs = [j for j in download_progress()["jobs"] if j["phase"] in {"downloading", "paused"}]
        self.assertEqual({j["repo"] for j in jobs}, {"mlx-community/SmolLM2-135M-Instruct-4bit", "mlx-community/Qwen3-8B-4bit"})
        self.assertEqual(again["repo"], "mlx-community/SmolLM2-135M-Instruct-4bit")
        self.assertEqual(len(QUEUE.jobs), 2)


if __name__ == "__main__":
    unittest.main()
