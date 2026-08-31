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

    def test_empty_payload_does_not_invent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "studio.json"
            save_prefs({"watchDirs": ["~/.lmstudio/models"]}, path=path)
            again = save_prefs({"watchDirs": [], "flagsByModel": {}}, path=path)
            self.assertEqual(again["watchDirs"], [])
            self.assertEqual(json.loads(path.read_text())["watchDirs"], [])


if __name__ == "__main__":
    unittest.main()
