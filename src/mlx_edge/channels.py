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
from typing import Any

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
    def __init__(self, assume_analysis: bool = False) -> None:
        self.buf = ""
        # MiniMax-M2.7's HF template already wrote `<think>\n` in the prompt,
        # so generation starts *inside* a think block. Qwen/Llama stay in
        # content unless they emit tags themselves.
        self.assume = assume_analysis
        self.mode = "analysis" if assume_analysis else "content"
        self.seen_channel = False
        self.seen_think = False
        self.held = ""

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
            if hold is not None and hold > 0:
                self._emit(self.buf[:hold], content, reasoning)
                self.buf = self.buf[hold:]
                break
            if hold == 0:
                break
            self._emit(self.buf, content, reasoning)
            self.buf = ""
            break
        content_s = "".join(content)
        reasoning_s = "".join(reasoning)
        confirmed = self.seen_think or self.seen_channel
        if self.assume and not confirmed:
            self.held += content_s + reasoning_s
            return "", ""
        if self.held:
            reasoning_s = self.held + reasoning_s
            self.held = ""
        return content_s, reasoning_s

    def flush(self) -> tuple[str, str]:
        content: list[str] = []
        reasoning: list[str] = []
        if self.buf:
            self._emit(self.buf, content, reasoning)
            self.buf = ""
        content_s = "".join(content)
        reasoning_s = "".join(reasoning)
        blob = self.held + content_s + reasoning_s
        self.held = ""
        if self.assume and not self.seen_think and not self.seen_channel:
            return blob, ""
        return content_s, reasoning_s

    def _token(self, token: str) -> None:
        name = token[2:-2]
        if name == "channel":
            self.seen_channel = True
            self.mode = "skip_channel"
            return
        if name == "start":
            self.mode = "skip_role"
            return
        if name == "message":
            if self.mode == "skip_channel":
                self.mode = "analysis"
            elif self.mode == "skip_role":
                self.mode = "content"
            return
        if name in {"end", "call", "constrain", "return"}:
            self.mode = "skip_role"
            return

    def _emit(self, text: str, content: list[str], reasoning: list[str]) -> None:
        if not text:
            return
        if self.mode == "skip_channel":
            match = CHANNEL_NAME.match(text.lstrip())
            if match:
                channel = match.group(1)
                rest = text.lstrip()[match.end() :]
                self.mode = "analysis" if channel in {"analysis", "commentary"} else "content"
                if rest:
                    self._emit(rest, content, reasoning)
                return
            text = CHANNEL_NAME.sub("", text.lstrip(), count=1)
            self.mode = "analysis"
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
    return content + more_c, reasoning + more_r


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
    blob = " ".join(n for n in names if n).lower()
    return "minimax" in blob or "gpt-oss" in blob or "gpt_oss" in blob or "harmony" in blob


def rewrite_choice_delta(delta: dict[str, Any], filt: HarmonyFilter) -> dict[str, Any] | None:
    """Rewrite an OpenAI delta in-place. Returns the delta, or None to drop the event."""
    raw = delta.get("content")
    if not isinstance(raw, str):
        return delta
    content, reasoning = filt.push(raw)
    out: dict[str, Any] = {k: v for k, v in delta.items() if k != "content"}
    if reasoning:
        prev = out.get("reasoning_content")
        out["reasoning_content"] = (prev if isinstance(prev, str) else "") + reasoning
    if content:
        out["content"] = content
    elif "content" in delta and not reasoning and delta.get("finish_reason"):
        out["content"] = ""
    if "content" not in out and "reasoning_content" not in out and not out.get("finish_reason"):
        if not out:
            return None
        if set(out) <= {"role"}:
            return out
        if not out:
            return None
    return out


def rewrite_message(message: dict[str, Any], assume_analysis: bool = False) -> dict[str, Any]:
    raw = message.get("content")
    if not isinstance(raw, str):
        return message
    if not looks_like_harmony(raw) and not looks_like_think(raw) and not assume_analysis:
        return message
    content, reasoning = filter_text(raw, assume_analysis=assume_analysis)
    out = dict(message)
    out["content"] = content
    if reasoning:
        out["reasoning_content"] = reasoning
    return out


def rewrite_completion_payload(
    payload: dict[str, Any],
    filt: HarmonyFilter | None = None,
    assume_analysis: bool = False,
) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    filt = filt or HarmonyFilter(assume_analysis=assume_analysis)
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
        if isinstance(row.get("message"), dict):
            row["message"] = rewrite_message(dict(row["message"]), assume_analysis=assume_analysis)
        out_choices.append(row)
    payload = dict(payload)
    payload["choices"] = out_choices
    return payload
