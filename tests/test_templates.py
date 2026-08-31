import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlx_edge.templates import (
    HARMONY_TEMPLATE,
    has_local_template,
    template_for_spawn,
)


class TemplateTests(unittest.TestCase):
    def test_skips_non_directory_hub_ids(self):
        self.assertEqual(template_for_spawn("qwen", []), [])
        self.assertEqual(template_for_spawn("mlx-community/Qwen3-8B-4bit", ["--temp", "0.2"]), ["--temp", "0.2"])

    def test_skips_when_flag_already_present(self):
        extra = ["--chat-template", "already"]
        self.assertEqual(template_for_spawn("/not/a/real/dir", extra), extra)

    def test_skips_when_checkpoint_has_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Qwen3-8B-4bit"
            path.mkdir()
            (path / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}),
                encoding="utf-8",
            )
            self.assertTrue(has_local_template(str(path)))
            self.assertEqual(template_for_spawn(str(path), []), [])

    def test_injects_harmony_for_minimax_without_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "MiniMax-M2.7-ConfigI-MLX"
            path.mkdir()
            (path / "config.json").write_text("{}", encoding="utf-8")
            with mock.patch("mlx_edge.templates._http_text", return_value=None):
                extra = template_for_spawn(str(path), [])
            self.assertEqual(extra[0], "--chat-template")
            self.assertIn("<|channel|>", extra[1])
            self.assertIn("<|start|>assistant", extra[1])
            self.assertIn(HARMONY_TEMPLATE.strip()[:40], extra[1])
