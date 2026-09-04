import json
import tempfile
import unittest
from pathlib import Path

from mlx_edge.prefs import load_prefs, save_prefs


class PrefsTests(unittest.TestCase):
    def test_roundtrip_keeps_lmstudio_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "studio.json"
            out = save_prefs(
                {"watchDirs": ["~/.lmstudio/models"], "flagsByModel": {"MiniMax": {"temp": 0.2}}},
                path=path,
            )
            self.assertEqual(out["watchDirs"], ["~/.lmstudio/models"])
            loaded = load_prefs(path)
            self.assertEqual(loaded["watchDirs"], ["~/.lmstudio/models"])
            self.assertEqual(loaded["flagsByModel"]["MiniMax"]["temp"], 0.2)

    def test_engine_override_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "studio.json"
            save_prefs(
                {
                    "watchDirs": ["~/.lmstudio/models"],
                    "engineByModel": {"thetom-ai/MiniMax-M3-ConfigI-MLX": "lm", "bad": "nope"},
                },
                path=path,
            )
            loaded = load_prefs(path)
            self.assertEqual(loaded["engineByModel"]["thetom-ai/MiniMax-M3-ConfigI-MLX"], "lm")
            self.assertNotIn("bad", loaded["engineByModel"])

    def test_empty_payload_does_not_invent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "studio.json"
            save_prefs({"watchDirs": ["~/.lmstudio/models"]}, path=path)
            again = save_prefs({"watchDirs": [], "flagsByModel": {}}, path=path)
            self.assertEqual(again["watchDirs"], [])
            self.assertEqual(json.loads(path.read_text())["watchDirs"], [])

    def test_fold_reason_enabled_matches_basename(self):
        from mlx_edge.prefs import fold_reason_enabled

        prefs = {
            "flagsByModel": {
                "mlx-community/Weird-Reason-MLX": {"streamReasonToResponse": True, "temp": 0.2},
            }
        }
        self.assertTrue(fold_reason_enabled(["Weird-Reason-MLX"], prefs))
        self.assertTrue(fold_reason_enabled(["/models/mlx-community/Weird-Reason-MLX"], prefs))
        self.assertFalse(fold_reason_enabled(["Qwen3-8B-4bit"], prefs))
        self.assertFalse(fold_reason_enabled(["Weird-Reason-MLX"], {"flagsByModel": {}}))


if __name__ == "__main__":
    unittest.main()
