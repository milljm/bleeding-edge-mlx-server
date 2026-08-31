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
        self.assertIn("status", parser.format_help())
        self.assertIn("update", parser.format_help())
        self.assertIn("serve", parser.format_help())

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
