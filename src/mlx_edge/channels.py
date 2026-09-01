"""Strip Harmony / MiniMax / gpt-oss channel wrappers from model output.

Those checkpoints emit visible tokens like ``<|channel|>analysis<|message|>``
when the chat template was not applied, and they still emit an analysis
channel even when it was. Edge keeps OpenAI ``content`` as the final
answer and puts the analysis in ``reasoning_content``.
"""

from __future__ import annotations

import re
from typing import Any

TOKEN = re.compile(r"<\|[a-zA-Z0-9_-]+\|>")
ROLE = re.compile(r"^(assistant|user|system|tool|developer)\b")
CHANNEL_NAME = re.compile(r"^(analysis|commentary|final|tool)\b")

# Incomplete Harmony token at a chunk boundary: "<" or "<|" or "<|chan"
INCOMPLETE = re.compile(r"<\|[a-zA-Z0-9_-]*$|<\|$|<$")


class HarmonyFilter:
    def __init__(self, assume_analysis: bool = False) -> None:
        self.buf = ""
        # MiniMax / gpt-oss generation often starts *inside* the analysis
        # channel (the chat template already wrote the opener). Plain
        # Qwen/Llama stay in content — they never emit Harmony tokens.
        self.mode = "analysis" if assume_analysis else "content"
        self.seen_channel = False

    def push(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        self.buf += text
        content: list[str] = []
        reasoning: list[str] = []
        while self.buf:
            match = TOKEN.search(self.buf)
            if match:
                self._emit(self.buf[: match.start()], content, reasoning)
                self._token(match.group(0))
                self.buf = self.buf[match.end() :]
                continue
            hold = INCOMPLETE.search(self.buf)
            if hold and hold.start() > 0:
                self._emit(self.buf[: hold.start()], content, reasoning)
                self.buf = self.buf[hold.start() :]
                break
            if hold and hold.start() == 0:
                break
            self._emit(self.buf, content, reasoning)
            self.buf = ""
            break
        return "".join(content), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        content: list[str] = []
        reasoning: list[str] = []
        if self.buf:
            self._emit(self.buf, content, reasoning)
            self.buf = ""
        return "".join(content), "".join(reasoning)

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
    if not looks_like_harmony(raw) and not assume_analysis:
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
