"""Parse MiniMax XML, Qwen <tool_call>, and Harmony tool markup into OpenAI tool_calls.

MiniMax's tokenizer stores some tags as ``]<]name[>[`` (same family as
``]<]image[>[``) and mlx-lm often drops the ``<`` / ``</`` on ``invoke`` /
``parameter`` (ml-explore/mlx-lm#1145). Normalize those back to XML before
the parsers run.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

MINIMAX_BLOCK = re.compile(
    r"<minimax:tool_call>(.*?)</minimax:tool_call>",
    re.I | re.S,
)
QWEN_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
INVOKE = re.compile(
    r'<invoke\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</invoke>',
    re.I | re.S,
)
PARAM = re.compile(
    r'<parameter\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)</parameter>',
    re.I | re.S,
)
TO_FUNC = re.compile(r"to=(?:functions\.)?([A-Za-z0-9_][A-Za-z0-9_.-]*)")

# Tokenizer encoding of <name> / </name>. ``]<]minimax[>[`` is the tool-call
# wrapper; ``]<]minimax:tool_call[>[`` is the same thing spelled out.
SPECIAL_WRAP = re.compile(r"\]<\](/?)([^[\]]+?)\[>\[")
# Chat-template control tokens that mlx-lm sometimes leaves in the stream.
CONTROL_TOKEN = re.compile(
    r"\]~!b\[|\[e~\[|\]!d~\[|\]!p~\[|\]~b\](?:system|user|ai|assistant|tool)?",
    re.I,
)
INCOMPLETE_SPECIAL = re.compile(
    r"(?:"
    r"\]<(?:\](?:[^[\]]*(?:\[>(?:\[)?)?)?)?|"
    r"\]~!b|"
    r"\]~!|"
    r"\]~b|"
    r"\]~|"
    r"\[e~|"
    r"\[e|"
    r"\]![dp]~|"
    r"\]![dp]|"
    r"\]!"
    r")$",
    re.I,
)

OPEN_TAGS = (
    ("<minimax:tool_call>", "</minimax:tool_call>"),
    ("<tool_call>", "</tool_call>"),
)


def openai_tool_call(name: str, arguments: Any, index: int | None = None) -> dict[str, Any]:
    if isinstance(arguments, (dict, list)):
        args = json.dumps(arguments, ensure_ascii=False)
    else:
        args = str(arguments or "").strip() or "{}"
    row: dict[str, Any] = {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": args},
    }
    if index is not None:
        row["index"] = index
    return row


def _parse_param_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _wrap_to_xml(match: re.Match[str]) -> str:
    slash, name = match.group(1), match.group(2).strip()
    lower = name.lower()
    if lower.startswith("end of "):
        return f"</{name[7:].strip()}>"
    if lower.startswith("start of "):
        return f"<{name[9:].strip()}>"
    if lower in {"minimax", "minimax:tool_call"}:
        return "</minimax:tool_call>" if slash else "<minimax:tool_call>"
    return f"<{slash}{name}>"


def repair_missing_brackets(text: str) -> str:
    """Restore ``<`` / ``</`` mlx-lm drops on MiniMax XML (issue 1145)."""
    if not text:
        return text
    text = re.sub(r"(?<!<)/minimax:tool_call>", "</minimax:tool_call>", text, flags=re.I)
    text = re.sub(r"(?<![</])minimax:tool_call>", "<minimax:tool_call>", text, flags=re.I)
    text = re.sub(r"(?<!<)invoke\s+name\s*=", "<invoke name=", text, flags=re.I)
    text = re.sub(r"(?<!<)parameter\s+name\s*=", "<parameter name=", text, flags=re.I)
    text = re.sub(r"(?<![</])invoke>", "</invoke>", text, flags=re.I)
    text = re.sub(r"(?<![</])parameter>", "</parameter>", text, flags=re.I)
    return text


def _pair_minimax_wrappers(text: str) -> str:
    """``]<]minimax[>[`` is both opener and closer. Pair extras into ``</...>``.

    Two real ``<minimax:tool_call>`` blocks with proper closes are left alone.
    """
    open_tag = "<minimax:tool_call>"
    close_tag = "</minimax:tool_call>"
    lower = text.lower()
    events: list[tuple[int, str, int]] = []
    i = 0
    while True:
        nxt_open = lower.find(open_tag, i)
        nxt_close = lower.find(close_tag, i)
        if nxt_open < 0 and nxt_close < 0:
            break
        if nxt_close >= 0 and (nxt_open < 0 or nxt_close < nxt_open):
            events.append((nxt_close, "close", len(close_tag)))
            i = nxt_close + len(close_tag)
        else:
            events.append((nxt_open, "open", len(open_tag)))
            i = nxt_open + len(open_tag)
    replacements: list[tuple[int, int, str]] = []
    depth = 0
    for pos, kind, length in events:
        if kind == "close":
            depth = max(0, depth - 1)
            continue
        if depth > 0:
            replacements.append((pos, pos + length, close_tag))
            depth -= 1
        else:
            depth += 1
    if not replacements:
        return text
    out: list[str] = []
    last = 0
    for start, end, new in replacements:
        out.append(text[last:start])
        out.append(new)
        last = end
    out.append(text[last:])
    return "".join(out)


def normalize_minimax_text(text: str) -> str:
    """Map tokenizer glyphs onto the XML the parsers already understand."""
    if not text:
        return text
    text = SPECIAL_WRAP.sub(_wrap_to_xml, text)
    text = _pair_minimax_wrappers(text)
    hold = incomplete_special_start(text)
    if hold is None:
        text = CONTROL_TOKEN.sub("", text)
        return repair_missing_brackets(text)
    head, suffix = text[:hold], text[hold:]
    head = CONTROL_TOKEN.sub("", head)
    head = repair_missing_brackets(head)
    return head + suffix


def parse_minimax_xml(blob: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for match in INVOKE.finditer(blob):
        name = match.group(1).strip()
        args: dict[str, Any] = {}
        for param in PARAM.finditer(match.group(2)):
            args[param.group(1)] = _parse_param_value(param.group(2))
        if name:
            out.append(openai_tool_call(name, args))
    return out


def parse_qwen_json(blob: str) -> list[dict[str, Any]]:
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    name = obj.get("name") or obj.get("function")
    if isinstance(name, dict):
        name = name.get("name")
    if not isinstance(name, str) or not name.strip():
        return []
    args = obj.get("arguments", obj.get("parameters", {}))
    return [openai_tool_call(name.strip(), args)]


def extract_tool_markup(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Pull completed MiniMax / Qwen tool blocks out of assistant text."""
    if not text:
        return "", []
    text = normalize_minimax_text(text)
    calls: list[dict[str, Any]] = []

    def minimax_sub(match: re.Match[str]) -> str:
        calls.extend(parse_minimax_xml(match.group(1)))
        return ""

    def qwen_sub(match: re.Match[str]) -> str:
        calls.extend(parse_qwen_json(match.group(1)))
        return ""

    cleaned = MINIMAX_BLOCK.sub(minimax_sub, text)
    cleaned = QWEN_BLOCK.sub(qwen_sub, cleaned)
    # mlx-lm may skip the <minimax:tool_call> special token entirely, leaving
    # bare <invoke> blocks (or the same after repair_missing_brackets).
    # `]<]minimax[>[` is used as both opener and closer, so both become
    # <minimax:tool_call> and MINIMAX_BLOCK never fires.
    bare = parse_minimax_xml(cleaned)
    if bare:
        calls.extend(bare)
        cleaned = INVOKE.sub("", cleaned)
    if calls:
        cleaned = re.sub(r"</?minimax:tool_call>", "", cleaned, flags=re.I)
    return cleaned, calls


def incomplete_special_start(buf: str) -> int | None:
    """Hold a MiniMax tokenizer glyph split across SSE chunks."""
    if not buf:
        return None
    match = INCOMPLETE_SPECIAL.search(buf)
    if not match or match.end() != len(buf):
        return None
    return match.start()


def incomplete_tool_start(buf: str) -> int | None:
    """Index of an unclosed tool-call tag, or a partial opener at the end."""
    if not buf:
        return None
    special = incomplete_special_start(buf)
    lower = buf.lower()
    holds: list[int] = []
    if special is not None:
        holds.append(special)
    for start, end in OPEN_TAGS:
        i = lower.rfind(start)
        if i >= 0 and end not in lower[i:]:
            holds.append(i)
    i = lower.rfind("<invoke")
    if i >= 0 and "</invoke>" not in lower[i:]:
        holds.append(i)
    if holds:
        return min(holds)
    for prefix in (
        "<minimax:tool_call",
        "<minimax:tool_cal",
        "<minimax:tool_ca",
        "<minimax:tool_c",
        "<minimax:tool",
        "<minimax:",
        "<minimax",
        "<invoke",
        "<invo",
        "<inv",
        "<tool_call",
        "<tool_cal",
        "<tool_ca",
        "<tool_c",
    ):
        if lower.endswith(prefix):
            return len(buf) - len(prefix)
    return None


def parse_harmony_recipient(text: str) -> str | None:
    match = TO_FUNC.search(text or "")
    if not match:
        return None
    name = match.group(1).strip().strip(".")
    if not name or name in {"functions", "function"}:
        return None
    if name.startswith("functions."):
        name = name[len("functions.") :]
    return name or None


def parse_tool_args(text: str) -> Any:
    blob = (text or "").strip()
    if not blob:
        return {}
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return blob
