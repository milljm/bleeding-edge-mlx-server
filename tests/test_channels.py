import json
import unittest

from mlx_edge.channels import (
    HarmonyFilter,
    assume_think_start,
    filter_text,
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

    def test_assume_think_start(self):
        self.assertFalse(assume_think_start("MiniMax-M2.7-ConfigI-MLX"))
        self.assertFalse(assume_think_start("/Users/milljm/.lmstudio/models/thetom-ai/MiniMax-M2.7-ConfigI-MLX"))
        self.assertFalse(assume_think_start("openai/gpt-oss-20b"))
        self.assertTrue(assume_think_start("MiniMax-M2.7-8bit"))
        self.assertTrue(assume_think_start("MiniMax-M2-8bit"))

    def test_configi_plain_tokens_are_content(self):
        filt = HarmonyFilter(assume_analysis=assume_think_start("MiniMax-M2.7-ConfigI-MLX"))
        c1, r1 = filt.push("Here's a **multi-panel")
        c2, r2 = filt.push(" scatter plot**")
        c3, r3 = filt.flush()
        self.assertEqual(c1 + c2 + c3, "Here's a **multi-panel scatter plot**")
        self.assertEqual(r1 + r2 + r3, "")

    def test_minimax_m27_think_tags(self):
        raw = "Need the current price.\n</think>\n\nApple is trading at $316.85."
        content, reasoning = filter_text(raw, assume_analysis=True)
        self.assertIn("316.85", content)
        self.assertNotIn("</think>", content)
        self.assertIn("price", reasoning)
        self.assertNotIn("316.85", reasoning)

    def test_minimax_m3_mm_think_tags(self):
        raw = "<mm:think>plan the answer</mm:think>\nDone."
        content, reasoning = filter_text(raw)
        self.assertEqual(content.strip(), "Done.")
        self.assertIn("plan the answer", reasoning)

    def test_minimax_plain_answer_not_eaten(self):
        content, reasoning = filter_text("Hello from MiniMax.", assume_analysis=True)
        self.assertEqual(content, "Hello from MiniMax.")
        self.assertEqual(reasoning, "")

    def test_stream_think_then_answer(self):
        filt = HarmonyFilter(assume_analysis=True)
        c1, r1 = filt.push("thinking about stocks")
        self.assertEqual(c1, "")
        self.assertIn("thinking", r1)
        c2, r2 = filt.push("</think>\nApple is at $1.")
        c3, r3 = filt.flush()
        self.assertEqual((c1 + c2 + c3).strip(), "Apple is at $1.")
        self.assertIn("thinking", r1 + r2 + r3)

    def test_minimax_think_still_streams_when_thinking_mentions_tools(self):
        filt = HarmonyFilter(assume_analysis=True, parse_tools=True)
        c1, r1 = filt.push("maybe <minimax:tool_call> later")
        self.assertEqual(c1, "")
        self.assertIn("maybe", r1)
        c2, r2 = filt.push("</think>\nHello from MiniMax.")
        c3, r3 = filt.flush()
        self.assertEqual((c2 + c3).strip(), "Hello from MiniMax.")

    def test_minimax_answer_streams_without_tool_hold(self):
        filt = HarmonyFilter(assume_analysis=True, parse_tools=False)
        pieces = []
        for token in ["plan", "</think>\n", "Hel", "lo world"]:
            c, r = filt.push(token)
            pieces.append(c)
        more, _ = filt.flush()
        self.assertEqual(("".join(pieces) + more).strip(), "Hello world")

    def test_split_think_close(self):
        filt = HarmonyFilter(assume_analysis=True)
        c1, r1 = filt.push("plan </th")
        self.assertEqual(c1, "")
        self.assertIn("plan", r1)
        c2, r2 = filt.push("ink>\nDone.")
        c3, r3 = filt.flush()
        self.assertEqual((c1 + c2 + c3).strip(), "Done.")
        self.assertIn("plan", r1 + r2 + r3)

    def test_qwen_streams_immediately(self):
        filt = HarmonyFilter(assume_analysis=False)
        content, reasoning = filt.push("Hello from Qwen.")
        self.assertEqual(content, "Hello from Qwen.")
        self.assertEqual(reasoning, "")

    def test_fold_reason_streams_assumed_analysis_as_content(self):
        filt = HarmonyFilter(assume_analysis=True, fold_reasoning=True)
        c1, r1 = filt.push("Hello from MiniMax.")
        c1, r1 = filt.fold_out(c1, r1)
        self.assertEqual(c1, "Hello from MiniMax.")
        self.assertEqual(r1, "")
        c2, r2 = filt.flush()
        c2, r2 = filt.fold_out(c2, r2)
        self.assertEqual(c2, "")
        self.assertEqual(r2, "")

    def test_fold_reason_drops_trailing_content_dump(self):
        filt = HarmonyFilter(assume_analysis=True, fold_reasoning=True)
        streamed, _ = filt.fold_out(*filt.push("Hello world"))
        self.assertEqual(streamed, "Hello world")
        dump, reason = filt.fold_out(*filt.push("Hello world"))
        self.assertEqual(dump, "")
        self.assertEqual(reason, "")
        extra, _ = filt.fold_out(*filt.flush())
        self.assertEqual(extra, "")

    def test_fold_reason_native_reasoning_delta(self):
        filt = HarmonyFilter(fold_reasoning=True)
        out = rewrite_completion_payload(
            {"choices": [{"delta": {"reasoning_content": "Hello"}}]},
            filt=filt,
        )
        delta = out["choices"][0]["delta"]
        self.assertEqual(delta.get("content"), "Hello")
        self.assertNotIn("reasoning_content", delta)
        dumped = rewrite_completion_payload(
            {"choices": [{"delta": {"content": "Hello"}}]},
            filt=filt,
        )
        # Replay dump is dropped entirely.
        self.assertEqual(dumped.get("choices"), [])

    def test_fold_reason_mlx_lm_reasoning_field(self):
        # mlx-lm emits `delta.reasoning`, not `reasoning_content`.
        filt = HarmonyFilter(fold_reasoning=True)
        first = rewrite_completion_payload(
            {"choices": [{"delta": {"role": "assistant", "reasoning": "Hel"}}]},
            filt=filt,
        )
        self.assertEqual(first["choices"][0]["delta"].get("content"), "Hel")
        self.assertNotIn("reasoning", first["choices"][0]["delta"])
        self.assertNotIn("reasoning_content", first["choices"][0]["delta"])
        second = rewrite_completion_payload(
            {"choices": [{"delta": {"reasoning": "lo"}}]},
            filt=filt,
        )
        self.assertEqual(second["choices"][0]["delta"].get("content"), "lo")
        dumped = rewrite_completion_payload(
            {"choices": [{"delta": {"content": "Hello"}}]},
            filt=filt,
        )
        self.assertEqual(dumped.get("choices"), [])

    def test_fold_reason_does_not_double_aliased_fields(self):
        filt = HarmonyFilter(fold_reasoning=True)
        out = rewrite_completion_payload(
            {"choices": [{"delta": {"reasoning": "Hel", "reasoning_content": "Hel"}}]},
            filt=filt,
        )
        self.assertEqual(out["choices"][0]["delta"].get("content"), "Hel")

    def test_fold_reason_drops_markdown_stripped_dump(self):
        filt = HarmonyFilter(fold_reasoning=True)
        streamed, _ = filt.fold_out("", "Use **Streamlit** for the GUI.")
        self.assertIn("Streamlit", streamed)
        dump, reason = filt.fold_out("Use Streamlit for the GUI.", "")
        self.assertEqual(dump, "")
        self.assertEqual(reason, "")

    def test_normalize_reasoning_field_without_fold(self):
        out = rewrite_completion_payload(
            {"choices": [{"delta": {"reasoning": "plan"}}]},
        )
        delta = out["choices"][0]["delta"]
        self.assertEqual(delta.get("reasoning_content"), "plan")
        self.assertEqual(delta.get("reasoning"), "plan")

    def test_filter_text_fold_reason(self):
        content, reasoning = filter_text("Hello from MiniMax.", assume_analysis=True, fold_reasoning=True)
        self.assertEqual(content, "Hello from MiniMax.")
        self.assertEqual(reasoning, "")

    def test_fold_reason_strips_think_tags_into_content(self):
        content, reasoning = filter_text(
            "plan the answer</think>\nDone.",
            assume_analysis=True,
            fold_reasoning=True,
        )
        self.assertIn("plan the answer", content)
        self.assertIn("Done.", content)
        self.assertNotIn("</think>", content)
        self.assertEqual(reasoning, "")

