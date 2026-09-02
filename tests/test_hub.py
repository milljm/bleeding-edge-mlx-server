import os
import unittest
from unittest import mock

from mlx_edge.hub import download_repo, parse_repo, search_quants, search_stem, token_set


class HubTests(unittest.TestCase):
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

        with mock.patch("mlx_edge.hub.hf_search", fake_search):
            out = search_quants("https://huggingface.co/Qwen/Qwen3-8B")
        ids = [r["id"] for r in out["results"]]
        self.assertEqual(ids[0], "mlx-community/Qwen3-8B-4bit")
        self.assertIn("mlx-community/Qwen3-8B-8bit", ids)
        self.assertNotIn("mlx-community/Qwen3-8B-1bit", ids)
        self.assertNotIn("Qwen/Qwen3-8B", ids)
        self.assertEqual(out["stem"], "Qwen3-8B")

    def test_token_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(token_set())
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_abc"}, clear=True):
            self.assertTrue(token_set())

    def test_download_rejects_1bit(self):
        with self.assertRaises(ValueError):
            download_repo("mlx-community/Bonsai-4B-mlx-1bit")

    def test_download_uses_snapshot(self):
        with mock.patch("mlx_edge.hub._snapshot", return_value="/cache/snap") as snap:
            out = download_repo("mlx-community/SmolLM2-135M-Instruct-4bit")
        snap.assert_called_once()
        self.assertEqual(out["path"], "/cache/snap")
        self.assertEqual(out["repo"], "mlx-community/SmolLM2-135M-Instruct-4bit")


if __name__ == "__main__":
    unittest.main()
