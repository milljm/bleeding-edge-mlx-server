import json
import unittest

from mlx_edge.channels import (
    HarmonyFilter,
    filter_text,
    harmony_model_name,
    rewrite_completion_payload,
)
from mlx_edge.gateway import _rewrite_sse_frame


class ChannelTests(unittest.TestCase):
    def test_user_minimax_leak(self):
        raw = (
            "<|channel|>analysis<|message|>We need to answer with current stock price for APPL (Apple). "
            "In THIS_TURN_ATTACHMENTS we have a WEB_SEARCH result giving price $316.85. "
            "Also respond casually.<|end|><|start|>assistant<|channel|>final<|message|>"
            "Apple is trading at $316.85."
        )
        content, reasoning = filter_text(raw)
        self.assertNotIn("<|channel|>", content)
        self.assertNotIn("<|message|>", content)
        self.assertNotIn("<|end|>", content)
        self.assertIn("316.85", content)
        self.assertIn("stock price", reasoning)
        self.assertNotIn("<|channel|>", reasoning)

    def test_template_applied_starts_in_analysis(self):
        raw = (
            "We need to answer with current stock price for APPL.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Apple is trading at $316.85."
        )
        content, reasoning = filter_text(raw, assume_analysis=True)
        self.assertEqual(content, "Apple is trading at $316.85.")
        self.assertIn("stock price", reasoning)
        self.assertNotIn("<|", content)

    def test_plain_text_untouched(self):
        content, reasoning = filter_text("Hello from Qwen.")
        self.assertEqual(content, "Hello from Qwen.")
        self.assertEqual(reasoning, "")

    def test_assume_analysis_does_not_eat_qwen_when_off(self):
        content, reasoning = filter_text("Hello from Qwen.", assume_analysis=False)
        self.assertEqual(content, "Hello from Qwen.")
        self.assertEqual(reasoning, "")

    def test_stream_split_token(self):
        filt = HarmonyFilter()
        c1, r1 = filt.push("<|chan")
        c2, r2 = filt.push("nel|>analysis<|message|>thinking")
        c3, r3 = filt.push(" hard<|end|><|start|>assistant<|channel|>final<|message|>done")
        c4, r4 = filt.flush()
        self.assertEqual((c1 + c2 + c3 + c4).strip(), "done")
        self.assertIn("thinking", r1 + r2 + r3 + r4)

    def test_rewrites_openai_message(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<|channel|>analysis<|message|>plan<|end|>"
                        "<|start|>assistant<|channel|>final<|message|>ok",
                    }
                }
            ]
        }
        out = rewrite_completion_payload(payload)
        msg = out["choices"][0]["message"]
        self.assertEqual(msg["content"], "ok")
        self.assertEqual(msg["reasoning_content"], "plan")

    def test_sse_frame_rewrites_delta(self):
        filt = HarmonyFilter()
        frame = (
            'data: {"choices":[{"delta":{"content":'
            '"<|channel|>analysis<|message|>plan<|end|>'
            '<|start|>assistant<|channel|>final<|message|>ok"}}]}'
        )
        out = _rewrite_sse_frame(frame, filt)
        self.assertIsNotNone(out)
        payload = json.loads(out.split("data:", 1)[1].strip())
        delta = payload["choices"][0]["delta"]
        self.assertEqual(delta.get("content"), "ok")
        self.assertEqual(delta.get("reasoning_content"), "plan")

    def test_sse_drops_token_only_delta(self):
        filt = HarmonyFilter()
        frame = 'data: {"choices":[{"delta":{"content":"<|end|>"}}]}'
        out = _rewrite_sse_frame(frame, filt)
        self.assertTrue(out is None or "data:" not in out)

    def test_harmony_model_name(self):
        self.assertTrue(harmony_model_name("MiniMax-M2.7-ConfigI-MLX"))
        self.assertTrue(harmony_model_name("openai/gpt-oss-20b"))
        self.assertFalse(harmony_model_name("Qwen3-8B-4bit"))
