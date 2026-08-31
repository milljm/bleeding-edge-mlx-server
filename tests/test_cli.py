import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from mlx_edge.cli import build_parser, cmd_engines, cmd_pin, cmd_update, main


class CliTests(unittest.TestCase):
    def test_help(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("status", help_text)
        self.assertIn("update", help_text)
        self.assertIn("serve", help_text)
        self.assertIn("load", help_text)
        self.assertIn("unload", help_text)

    def test_serve_parses_gui_flag(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--gui", "--no-browser", "--host", "0.0.0.0"])
        self.assertTrue(args.gui)
        self.assertTrue(args.no_browser)
        self.assertEqual(args.host, "0.0.0.0")

    def test_gui_main_help(self):
        from mlx_edge.cli import gui_main

        with self.assertRaises(SystemExit) as ctx:
            gui_main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_gui_without_assets_fails(self):
        from mlx_edge.cli import cmd_serve

        args = mock.Mock(
            host="127.0.0.1",
            port=0,
            gui=True,
            no_browser=True,
            lm=[],
            vlm=[],
            engine=None,
            model=[],
        )
        buf = io.StringIO()
        with mock.patch("mlx_edge.gateway.bundled_web_dir", return_value=None), mock.patch("sys.stderr", buf):
            rc = cmd_serve(args, [])
        self.assertEqual(rc, 1)
        self.assertIn("GUI assets missing", buf.getvalue())

    def test_serve_parses_without_engine(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.lm, [])
        self.assertEqual(args.vlm, [])

    def test_serve_parses_preload_flags(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--lm", "qwen", "--vlm", "qwen-vl", "--lm", "llama"])
        self.assertEqual(args.lm, ["qwen", "llama"])
        self.assertEqual(args.vlm, ["qwen-vl"])

    def test_legacy_engine_model(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--engine", "lm", "--model", "qwen"])
        self.assertEqual(args.engine, "lm")
        self.assertEqual(args.model, ["qwen"])

    def test_engines_lists_catalog(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_engines(None)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("mlx-lm", out)
        self.assertIn("mlx-vlm", out)
        self.assertIn("compiled", out)

    def test_main_engines(self):
        rc = main(["engines"])
        self.assertEqual(rc, 0)

    def test_status_offline(self):
        rc = main(["status", "--offline"])
        self.assertEqual(rc, 0)

    def test_status_json_offline(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["status", "--offline", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        ids = {row["id"] for row in payload}
        self.assertEqual(ids, {"lm", "vlm", "mlx"})

    def test_update_refuses_compiled_mlx(self):
        args = mock.Mock(
            engine="mlx",
            pinned=False,
            force=False,
            ref=None,
            branch="main",
            with_deps=False,
        )
        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            rc = cmd_update(args)
        self.assertEqual(rc, 2)
        self.assertIn("refusing", buf.getvalue())

    def test_pin_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            pin = Path(tmp) / "pins.json"
            with mock.patch("mlx_edge.cli.PIN_PATH", pin):
                rc = cmd_pin(mock.Mock())
            self.assertEqual(rc, 0)
            data = json.loads(pin.read_text())
            self.assertIn("engines", data)
            self.assertIn("lm", data["engines"])

    def test_unknown_command(self):
        with self.assertRaises(SystemExit):
            main(["not-a-command"])


if __name__ == "__main__":
    unittest.main()
