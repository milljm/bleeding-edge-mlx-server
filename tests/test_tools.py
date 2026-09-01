import json
import unittest

from mlx_edge.channels import HarmonyFilter, rewrite_completion_payload
from mlx_edge.tools import extract_tool_markup, parse_minimax_xml


class ToolParseTests(unittest.TestCase):
    def test_minimax_xml(self):
        raw = """
<minimax:tool_call>
<invoke name="read_file">
<parameter name="path">src/app.py</parameter>
<parameter name="offset">1</parameter>
</invoke>
</minimax:tool_call>
"""
        cleaned, calls = extract_tool_markup(raw)
        self.assertEqual(cleaned.strip(), "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["path"], "src/app.py")
        self.assertEqual(args["offset"], 1)

    def test_qwen_tool_call(self):
        raw = '<tool_call>{"name": "write_to_file", "arguments": {"path": "a.py", "content": "x"}}</tool_call>'
        cleaned, calls = extract_tool_markup(raw)
        self.assertEqual(cleaned.strip(), "")
        self.assertEqual(calls[0]["function"]["name"], "write_to_file")
        self.assertIn("a.py", calls[0]["function"]["arguments"])

    def test_harmony_function_call(self):
        raw = (
            "<|channel|>commentary to=functions.read_file <|constrain|>json"
            '<|message|>{"path": "src/app.py"}<|call|>'
        )
        filt = HarmonyFilter()
        content, reasoning = filt.push(raw)
        more_c, more_r = filt.flush()
        self.assertFalse((content + more_c).strip())
        self.assertEqual(len(filt.tool_calls), 1)
        call = filt.tool_calls[0]
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(json.loads(call["function"]["arguments"])["path"], "src/app.py")
        self.assertEqual(call["index"], 0)

    def test_harmony_stream_split(self):
        filt = HarmonyFilter()
        filt.push("<|channel|>commentary to=functions.")
        filt.push('read_file <|constrain|>json<|message|>{"path":')
        filt.push('"x.py"}<|call|>')
        filt.flush()
        self.assertEqual(filt.tool_calls[0]["function"]["name"], "read_file")
        self.assertIn("x.py", filt.tool_calls[0]["function"]["arguments"])

    def test_minimax_xml_in_message_payload(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '<minimax:tool_call>\n<invoke name="execute_command">'
                        '<parameter name="command">ls</parameter></invoke>\n</minimax:tool_call>',
                    },
                }
            ]
        }
        out = rewrite_completion_payload(payload)
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        msg = choice["message"]
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "execute_command")
        self.assertIsNone(msg.get("content") or None)

    def test_parse_minimax_multiple_invokes(self):
        blob = (
            '<invoke name="a"><parameter name="x">1</parameter></invoke>'
            '<invoke name="b"><parameter name="y">two</parameter></invoke>'
        )
        calls = parse_minimax_xml(blob)
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])

    def test_tools_after_think_still_parse(self):
        filt = HarmonyFilter(assume_analysis=True, parse_tools=True)
        filt.push("planning the call")
        filt.push(
            "</think>\n<minimax:tool_call>\n<invoke name=\"read_file\">"
            "<parameter name=\"path\">a.py</parameter></invoke>\n</minimax:tool_call>"
        )
        content, _ = filt.flush()
        self.assertFalse(content.strip())
        self.assertEqual(filt.tool_calls[0]["function"]["name"], "read_file")

    def test_special_wrap_minimax_token(self):
        raw = (
            ']<]minimax[>[\n<invoke name="read_file">'
            '<parameter name="path">a.py</parameter></invoke>\n'
            ']<]minimax[>['
        )
        cleaned, calls = extract_tool_markup(raw)
        self.assertEqual(cleaned.strip(), "")
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["path"], "a.py")

    def test_special_wrap_tool_call_spelling(self):
        raw = (
            ']<]minimax:tool_call[>[\n<invoke name="ls">'
            '<parameter name="command">pwd</parameter></invoke>\n'
            ']<]/minimax:tool_call[>['
        )
        _, calls = extract_tool_markup(raw)
        self.assertEqual(calls[0]["function"]["name"], "ls")

    def test_missing_angle_brackets(self):
        # mlx-lm#1145: special tokens skipped, inner tags lose `<` / `</`.
        raw = (
            'invoke name="get_weather">\n'
            'parameter name="location">Parisparameter>\n'
            'invoke>'
        )
        cleaned, calls = extract_tool_markup(raw)
        self.assertEqual(cleaned.strip(), "")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["location"], "Paris")

    def test_control_tokens_stripped(self):
        raw = (
            ']~b]ai\n<minimax:tool_call>\n<invoke name="read_file">'
            '<parameter name="path">x.py</parameter></invoke>\n'
            '</minimax:tool_call>[e~['
        )
        cleaned, calls = extract_tool_markup(raw)
        self.assertNotIn("]~b]", cleaned)
        self.assertNotIn("[e~[", cleaned)
        self.assertEqual(calls[0]["function"]["name"], "read_file")

    def test_stream_special_wrap_split(self):
        filt = HarmonyFilter(assume_analysis=True, parse_tools=True)
        filt.push("plan")
        c1, _ = filt.push("</think>\n]<")
        self.assertFalse(c1.strip())
        filt.push(']minimax[>[\n<invoke name="read_file">')
        filt.push('<parameter name="path">a.py</parameter></invoke>\n]<]minimax[>[')
        content, _ = filt.flush()
        self.assertFalse(content.strip())
        self.assertEqual(filt.tool_calls[0]["function"]["name"], "read_file")

    def test_special_wrap_does_not_eat_plain_answer(self):
        filt = HarmonyFilter(assume_analysis=True, parse_tools=True)
        filt.push("just thinking")
        c, _ = filt.push("</think>\nHello from MiniMax.")
        more, _ = filt.flush()
        self.assertEqual((c + more).strip(), "Hello from MiniMax.")
        self.assertFalse(filt.tool_calls)
