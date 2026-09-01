"""Strip Harmony / MiniMax / gpt-oss channel wrappers from model output.

MiniMax-M2.7 / M3 (Hugging Face templates) wrap thinking in ``<think>`` /
``<mm:think>`` and put the opener in the *prompt*, so the first generated
tokens are thinking text and the closer ``</think>`` is the switch to the
answer. ConfigI / gpt-oss use Harmony ``<|channel|>`` tokens instead.

Edge keeps OpenAI ``content`` as the final answer and puts analysis in
``reasoning_content``. If a MiniMax generation never emits a think/Harmony
closer, the buffered analysis is promoted back to ``content`` so clients
do not see an empty reply.
"""

from __future__ import annotations

import re
from mlx_edge.tools import (
    extract_tool_markup,
    incomplete_tool_start,
    openai_tool_call,
    parse_harmony_recipient,
    parse_tool_args,
)

TOKEN = re.compile(r"<\|[a-zA-Z0-9_-]+\|>")
ROLE = re.compile(r"^(assistant|user|system|tool|developer)\b")
CHANNEL_NAME = re.compile(r"^(analysis|commentary|final|tool)\b")
THINK_OPEN = re.compile(r"<think>|<mm:think>", re.I)
THINK_CLOSE = re.compile(r"</think>|</mm:think>", re.I)

# Incomplete Harmony or think tag at a chunk boundary.
INCOMPLETE = re.compile(
    r"(?:<\|[a-zA-Z0-9_-]*|<\|?|</(?:mm:)?(?:think|thin|thi|th|t)?|<(?:mm:)?(?:think|thin|thi|th|t)?|</?mm:?)$",
    re.I,
)


def _next_tag(buf: str) -> tuple[int, int, str, str] | None:
    hits: list[tuple[int, int, str, str]] = []
    for rx, kind in ((TOKEN, "harmony"), (THINK_OPEN, "think_open"), (THINK_CLOSE, "think_close")):
        match = rx.search(buf)
        if match:
            hits.append((match.start(), match.end(), kind, match.group(0)))
    if not hits:
        return None
    hits.sort(key=lambda row: row[0])
    return hits[0]


def _incomplete_start(buf: str) -> int | None:
    match = INCOMPLETE.search(buf)
    if not match:
        return None
    # Only hold a suffix — a match in the middle that is not a real tag
    # still needs to be emitted (e.g. comparison `a < b`).
    if match.end() != len(buf):
        return None
    return match.start()


class HarmonyFilter:
    def __init__(self, assume_analysis: bool = False, parse_tools: bool = True) -> None:
        self.buf = ""
        # MiniMax-M2.7's HF template already wrote `<think>\n` in the prompt,
        # so generation starts *inside* a think block. Qwen/Llama stay in
        # content unless they emit tags themselves.
        self.assume = assume_analysis
        self.parse_tools = parse_tools
        self.mode = "analysis" if assume_analysis else "content"
        self.seen_channel = False
        self.seen_think = False
        self.held = ""
        self.tool_calls: list[dict[str, Any]] = []
        self._pending_tools: list[dict[str, Any]] = []
        self._tool_name: str | None = None
        self._tool_buf = ""
        self._channel_header = ""
        self.saw_tools = False

    def take_tool_calls(self) -> list[dict[str, Any]]:
        out = self._pending_tools
        self._pending_tools = []
        return out

    def _add_tools(self, calls: list[dict[str, Any]]) -> None:
        if not calls:
            return
        start = len(self.tool_calls)
        for i, call in enumerate(calls):
            row = dict(call)
            row["index"] = start + i
            self.tool_calls.append(row)
            self._pending_tools.append(row)
        self.saw_tools = True

    def push(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        self.buf += text
        content: list[str] = []
        reasoning: list[str] = []
        while self.buf:
            tag = _next_tag(self.buf)
            if tag:
                start, end, kind, raw = tag
                self._emit(self.buf[:start], content, reasoning)
                if kind == "harmony":
                    self._token(raw)
                elif kind == "think_open":
                    self.seen_think = True
                    self.mode = "analysis"
                else:
                    self.seen_think = True
                    self.mode = "content"
                self.buf = self.buf[end:]
                continue
            hold = _incomplete_start(self.buf)
            tool_hold = self._tool_hold_index()
            cuts = [i for i in (hold, tool_hold) if i is not None]
            if cuts:
                cut = min(cuts)
                if cut > 0:
                    self._emit(self.buf[:cut], content, reasoning)
                    self.buf = self.buf[cut:]
                break
            self._emit(self.buf, content, reasoning)
            self.buf = ""
            break
        content_s = "".join(content)
        reasoning_s = "".join(reasoning)
        if self.parse_tools:
            content_s, extra = extract_tool_markup(content_s)
            self._add_tools(extra)
        confirmed = self.seen_think or self.seen_channel
        if self.assume and not confirmed:
            # Stream thinking live as reasoning; remember it so flush can
            # promote to content if MiniMax never emits a closer.
            blob = content_s + reasoning_s
            self.held += blob
            return "", blob
        if self.held:
            # Already streamed as reasoning — do not re-emit on the switch.
            self.held = ""
        return content_s, reasoning_s

    def flush(self) -> tuple[str, str]:
        content: list[str] = []
        reasoning: list[str] = []
        if self.buf:
            self._emit(self.buf, content, reasoning)
            self.buf = ""
        self._finish_tool()
        content_s = "".join(content)
        reasoning_s = "".join(reasoning)
        if self.parse_tools:
            content_s, extra = extract_tool_markup(content_s)
            self._add_tools(extra)
        blob = self.held + content_s + reasoning_s
        self.held = ""
        if self.assume and not self.seen_think and not self.seen_channel:
            leftover = blob
            if self.parse_tools:
                leftover, more = extract_tool_markup(blob)
                self._add_tools(more)
            return leftover, ""
        return content_s, reasoning_s

    def _tool_hold_index(self) -> int | None:
        """Hold an unclosed tool block only in the *answer*, never while thinking.

        MiniMax-M2 drafts ``<minimax:tool_call>`` inside ``<think>``. Buffering
        that used to freeze the rest of the generation and dump it at EOS.
        """
        if not self.parse_tools:
            return None
        if self.mode == "analysis":
            return None
        return incomplete_tool_start(self.buf)

    def _finish_tool(self) -> None:
        if not self._tool_name:
            self._tool_buf = ""
            return
        self._add_tools([openai_tool_call(self._tool_name, parse_tool_args(self._tool_buf))])
        self._tool_name = None
        self._tool_buf = ""

    def _token(self, token: str) -> None:
        name = token[2:-2]
        if name == "channel":
            self.seen_channel = True
            self.mode = "skip_channel"
            self._channel_header = ""
            return
        if name == "start":
            self.mode = "skip_role"
            return
        if name == "message":
            self._apply_channel_header()
            if self._tool_name:
                self.mode = "tool_args"
            elif self.mode == "skip_role":
                self.mode = "content"
            elif self.mode in {"skip_channel", "skip_constrain"}:
                self.mode = "analysis"
            return
        if name == "constrain":
            self._apply_channel_header()
            self.mode = "skip_constrain"
            return
        if name == "call":
            self._finish_tool()
            self.mode = "skip_role"
            return
        if name in {"end", "return"}:
            if self._tool_name:
                self._finish_tool()
            self.mode = "skip_role"
            return

    def _apply_channel_header(self) -> None:
        header = self._channel_header
        self._channel_header = ""
        if not header:
            return
        recipient = parse_harmony_recipient(header)
        if recipient and recipient not in {"functions", "function"}:
            self._tool_name = recipient
            self._tool_buf = ""
            return
        match = CHANNEL_NAME.match(header.lstrip())
        channel = match.group(1) if match else ""
        if channel in {"analysis", "commentary"}:
            self.mode = "analysis"
        elif channel:
            self.mode = "content"

    def _emit(self, text: str, content: list[str], reasoning: list[str]) -> None:
        if not text:
            return
        if self.mode == "skip_channel":
            self._channel_header += text
            return
        if self.mode == "skip_constrain":
            # Drop the "json" type token; args follow <|message|>.
            return
        if self.mode == "tool_args":
            self._tool_buf += text
            return
        if self.mode == "skip_role":
            stripped = text.lstrip()
            match = ROLE.match(stripped)
            if match:
                rest = stripped[match.end() :]
                self.mode = "content"
                if rest:
                    self._emit(rest, content, reasoning)
                return
            self.mode = "content"
        if self.mode == "analysis":
            reasoning.append(text)
            return
        content.append(text)


def filter_text(text: str, assume_analysis: bool = False) -> tuple[str, str]:
    filt = HarmonyFilter(assume_analysis=assume_analysis)
    content, reasoning = filt.push(text)
    more_c, more_r = filt.flush()
    content, reasoning = content + more_c, reasoning + more_r
    if assume_analysis and not filt.seen_think and not filt.seen_channel:
        return (content or reasoning), ""
    return content, reasoning


def looks_like_harmony(text: str) -> bool:
    if not text:
        return False
    return "<|channel|>" in text or "<|message|>" in text or "<|start|>" in text


def looks_like_think(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return "<think>" in lower or "</think>" in lower or "<mm:think>" in lower or "</mm:think>" in lower


def harmony_model_name(*names: str) -> bool:
    """True when Edge should strip Harmony / MiniMax think wrappers."""
    blob = " ".join(n for n in names if n).lower()
    return "minimax" in blob or "gpt-oss" in blob or "gpt_oss" in blob or "harmony" in blob


def assume_think_start(*names: str) -> bool:
    """HF MiniMax-M2.7 / M3 templates put ``<think>`` in the prompt.

    Generation then starts *inside* thinking, so the filter must treat the
    first tokens as analysis. ConfigI / gpt-oss are Harmony and start in
    the final channel — those tokens are already the answer.
    """
    blob = " ".join(n for n in names if n).lower()
    if "configi" in blob or "gpt-oss" in blob or "gpt_oss" in blob or "harmony" in blob:
        return False
    return "minimax" in blob


def rewrite_choice_delta(delta: dict[str, Any], filt: HarmonyFilter) -> dict[str, Any] | None:
    """Rewrite an OpenAI delta in-place. Returns the delta, or None to drop the event."""
    raw = delta.get("content")
    if isinstance(raw, str):
        content, reasoning = filt.push(raw)
        out: dict[str, Any] = {k: v for k, v in delta.items() if k != "content"}
        if reasoning:
            prev = out.get("reasoning_content")
            out["reasoning_content"] = (prev if isinstance(prev, str) else "") + reasoning
        if content:
            out["content"] = content
        elif "content" in delta and not reasoning and delta.get("finish_reason"):
            out["content"] = ""
    else:
        out = dict(delta)
    tools = filt.take_tool_calls()
    if tools:
        prev_tools = out.get("tool_calls")
        merged = list(prev_tools) if isinstance(prev_tools, list) else []
        merged.extend(tools)
        out["tool_calls"] = merged
        if filt.saw_tools and out.get("finish_reason") == "stop":
            out["finish_reason"] = "tool_calls"
    if (
        "content" not in out
        and "reasoning_content" not in out
        and "tool_calls" not in out
        and not out.get("finish_reason")
    ):
        if not out:
            return None
        if set(out) <= {"role"}:
            return out
        if not out:
            return None
    return out


def rewrite_message(message: dict[str, Any], assume_analysis: bool = False, parse_tools: bool = True) -> dict[str, Any]:
    raw = message.get("content")
    if not isinstance(raw, str):
        return message
    if not looks_like_harmony(raw) and not looks_like_think(raw) and "<minimax:tool_call>" not in raw.lower() and "<tool_call>" not in raw and not assume_analysis:
        return message
    filt = HarmonyFilter(assume_analysis=assume_analysis, parse_tools=parse_tools)
    content, reasoning = filt.push(raw)
    more_c, more_r = filt.flush()
    content, reasoning = content + more_c, reasoning + more_r
    out = dict(message)
    out["content"] = content
    if reasoning:
        out["reasoning_content"] = reasoning
    tools = filt.take_tool_calls() or list(filt.tool_calls)
    if tools:
        existing = out.get("tool_calls")
        merged = list(existing) if isinstance(existing, list) else []
        merged.extend(tools)
        out["tool_calls"] = merged
        if not out.get("content"):
            out["content"] = None
    return out


def rewrite_completion_payload(
    payload: dict[str, Any],
    filt: HarmonyFilter | None = None,
    assume_analysis: bool = False,
    parse_tools: bool = True,
) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    filt = filt or HarmonyFilter(assume_analysis=assume_analysis, parse_tools=parse_tools)
    out_choices = []
    for choice in choices:
        if not isinstance(choice, dict):
            out_choices.append(choice)
            continue
        row = dict(choice)
        if isinstance(row.get("delta"), dict):
            delta = rewrite_choice_delta(dict(row["delta"]), filt)
            if delta is None:
                continue
            row["delta"] = delta
            if filt.saw_tools and row.get("finish_reason") == "stop":
                row["finish_reason"] = "tool_calls"
                if delta.get("finish_reason") == "stop":
                    delta["finish_reason"] = "tool_calls"
        if isinstance(row.get("message"), dict):
            row["message"] = rewrite_message(
                dict(row["message"]), assume_analysis=assume_analysis, parse_tools=filt.parse_tools
            )
            msg = row["message"]
            if isinstance(msg, dict) and msg.get("tool_calls") and row.get("finish_reason") in {None, "stop"}:
                row["finish_reason"] = "tool_calls"
        out_choices.append(row)
    payload = dict(payload)
    payload["choices"] = out_choices
    return payload
