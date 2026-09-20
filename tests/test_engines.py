import json
import subprocess
import unittest
from unittest import mock

import mlx_edge.engines as engines
from mlx_edge.engines import (
    ENGINES,
    PYTHON_ENGINES,
    engine_features,
    get_engine,
    resolve_targets,
)


class EngineFeatureTests(unittest.TestCase):
    def setUp(self):
        engines._features_cache.clear()

    def test_cached_between_calls(self):
        with mock.patch.object(engines.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=json.dumps(["apc_enabled", "kv_bits"]))
            first = engine_features("vlm")
            second = engine_features("vlm")
        self.assertEqual(first, frozenset({"apc"}))
        self.assertEqual(second, first)
        self.assertEqual(run.call_count, 1)

    def test_probe_without_apc_knob_reports_none(self):
        with mock.patch.object(engines.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=json.dumps(["kv_bits"]))
            self.assertEqual(engine_features("vlm"), frozenset())

    def test_missing_engine_degrades_to_empty(self):
        with mock.patch.object(engines.subprocess, "run", side_effect=OSError("no python")):
            self.assertEqual(engine_features("vlm"), frozenset())

    def test_failing_probe_degrades_to_empty(self):
        with mock.patch.object(engines.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=1, stdout="")
            self.assertEqual(engine_features("vlm"), frozenset())

    def test_engines_without_probe_have_no_features(self):
        self.assertEqual(engine_features("lm"), frozenset())
        self.assertEqual(engine_features("nonsense"), frozenset())

    def test_live_vlm_probe_reports_apc_when_present(self):
        """Smoke against the real env; skips silently when mlx-vlm is absent."""
        try:
            probe = subprocess.run(
                [engines.sys.executable, "-c", "import mlx_vlm"],
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            self.skipTest("mlx-vlm not installed")
        if probe.returncode != 0:
            # import mlx_vlm failed (e.g. not installed on CI): skip, don't fail.
            self.skipTest("mlx-vlm not installed")
        self.assertIn("apc", engine_features("vlm"))


class EngineCatalogTests(unittest.TestCase):
    def test_python_engines_are_not_compiled(self):
        for key in PYTHON_ENGINES:
            self.assertFalse(ENGINES[key].compiled)
        self.assertIsNotNone(ENGINES["lm"].server_module)
        self.assertIsNotNone(ENGINES["vlm"].server_module)
        self.assertIsNone(ENGINES["audio"].server_module)

    def test_mlx_is_compiled_and_has_no_server(self):
        mlx = ENGINES["mlx"]
        self.assertTrue(mlx.compiled)
        self.assertIsNone(mlx.server_module)

    def test_aliases(self):
        self.assertEqual(get_engine("mlx-lm").id, "lm")
        self.assertEqual(get_engine("mlx_vlm").id, "vlm")
        self.assertEqual(get_engine("mlx-audio").id, "audio")

    def test_unknown_engine_exits(self):
        with self.assertRaises(SystemExit):
            get_engine("llama.cpp")

    def test_resolve_all(self):
        targets = resolve_targets("all")
        self.assertEqual([t.id for t in targets], ["lm", "vlm", "audio"])


if __name__ == "__main__":
    unittest.main()
