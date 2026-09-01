"""Parse MiniMax XML, Qwen <tool_call>, and Harmony tool markup into OpenAI tool_calls."""

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
INVOKE = re.compile(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.I | re.S)
PARAM = re.compile(r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>', re.I | re.S)
TO_FUNC = re.compile(r"to=(?:functions\.)?([A-Za-z0-9_][A-Za-z0-9_.-]*)")

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
    calls: list[dict[str, Any]] = []

    def minimax_sub(match: re.Match[str]) -> str:
        calls.extend(parse_minimax_xml(match.group(1)))
        return ""

    def qwen_sub(match: re.Match[str]) -> str:
        calls.extend(parse_qwen_json(match.group(1)))
        return ""

    cleaned = MINIMAX_BLOCK.sub(minimax_sub, text)
    cleaned = QWEN_BLOCK.sub(qwen_sub, cleaned)
    return cleaned, calls


def incomplete_tool_start(buf: str) -> int | None:
    """Index of an unclosed tool-call tag, or a partial opener at the end."""
    if not buf:
        return None
    lower = buf.lower()
    holds: list[int] = []
    for start, end in OPEN_TAGS:
        i = lower.rfind(start)
        if i >= 0 and end not in lower[i:]:
            holds.append(i)
    if holds:
        return min(holds)
    for prefix in ("<minimax:tool_call", "<minimax:tool_cal", "<minimax:tool_ca", "<minimax:tool_c", "<minimax:tool", "<minimax:", "<minimax", "<tool_call", "<tool_cal", "<tool_ca", "<tool_c"):
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
