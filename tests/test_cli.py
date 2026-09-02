import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from mlx_edge.cli import build_parser, cmd_engines, cmd_pin, cmd_update, main, parse_build_spec


class CliTests(unittest.TestCase):
    def test_help(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("status", help_text)
        self.assertIn("update", help_text)
        self.assertIn("build", help_text)
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
            embed=[],
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
        args = parser.parse_args(["serve", "--lm", "qwen", "--vlm", "qwen-vl", "--lm", "llama", "--embed", "bge", "--tts", "kokoro", "--stt", "whisper", "--rerank", "qwen-rerank", "--image", "flux"])
        self.assertEqual(args.lm, ["qwen", "llama"])
        self.assertEqual(args.vlm, ["qwen-vl"])
        self.assertEqual(args.embed, ["bge"])
        self.assertEqual(args.tts, ["kokoro"])
        self.assertEqual(args.stt, ["whisper"])
        self.assertEqual(args.rerank, ["qwen-rerank"])
        self.assertEqual(args.image, ["flux"])

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

    def test_build_parses_git_url_and_pr(self):
        engine, spec = parse_build_spec(
            "git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1398/head"
        )
        self.assertEqual(engine.id, "lm")
        self.assertIn("1398", spec)
        engine, spec = parse_build_spec("1398")
        self.assertEqual(engine.id, "lm")
        self.assertEqual(spec, "git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1398/head")
        engine, spec = parse_build_spec("mlx-vlm#42")
        self.assertEqual(engine.id, "vlm")
        self.assertIn("refs/pull/42/head", spec)
        engine, spec = parse_build_spec("42", "vlm")
        self.assertEqual(engine.id, "vlm")
        parser = build_parser()
        args = parser.parse_args(["build", "1398"])
        self.assertEqual(args.spec, "1398")
        empty = parser.parse_args(["build"])
        self.assertFalse(empty.spec)

    def test_build_help_lists_search_urls(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["build"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("https://github.com/ml-explore/mlx-lm/pulls", out)
        self.assertIn("https://github.com/Blaizzy/mlx-vlm/pulls", out)


if __name__ == "__main__":
    unittest.main()
