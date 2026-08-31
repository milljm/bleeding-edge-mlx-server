import unittest

from mlx_edge.engines import ENGINES, PYTHON_ENGINES, get_engine, resolve_targets


class EngineCatalogTests(unittest.TestCase):
    def test_python_engines_are_not_compiled(self):
        for key in PYTHON_ENGINES:
            self.assertFalse(ENGINES[key].compiled)
            self.assertIsNotNone(ENGINES[key].server_module)

    def test_mlx_is_compiled_and_has_no_server(self):
        mlx = ENGINES["mlx"]
        self.assertTrue(mlx.compiled)
        self.assertIsNone(mlx.server_module)

    def test_aliases(self):
        self.assertEqual(get_engine("mlx-lm").id, "lm")
        self.assertEqual(get_engine("mlx_vlm").id, "vlm")

    def test_unknown_engine_exits(self):
        with self.assertRaises(SystemExit):
            get_engine("llama.cpp")

    def test_resolve_all(self):
        targets = resolve_targets("all")
        self.assertEqual([t.id for t in targets], ["lm", "vlm"])


if __name__ == "__main__":
    unittest.main()
