"""Literal `{target : replace}` rewrites on streamed channel text.

Each rule is an exact substring match. Streaming holds a short tail so a
target split across SSE chunks still rewrites. Empty replace deletes.
"""

from __future__ import annotations


def parse_replace_rules(text: str | None) -> list[tuple[str, str]]:
    if not text or not str(text).strip():
        return []
    blob = str(text)
    rules: list[tuple[str, str]] = []
    i = 0
    while True:
        start = blob.find("{", i)
        if start < 0:
            break
        end = blob.find("}", start + 1)
        if end < 0:
            break
        inner = blob[start + 1 : end]
        if " : " in inner:
            target, replace = inner.split(" : ", 1)
        elif ":" in inner:
            target, replace = inner.split(":", 1)
            target, replace = target.strip(), replace.strip()
        else:
            i = start + 1
            continue
        if target:
            rules.append((target, replace))
        i = end + 1
    return rules


class LiteralReplacer:
    """Rewrite exact targets in a token stream. Hold a suffix that might start a target."""

    def __init__(self, rules: list[tuple[str, str]] | None = None) -> None:
        self.rules = [(t, r) for t, r in (rules or []) if t]
        self.hold = ""

    def push(self, chunk: str) -> str:
        if not self.rules:
            return chunk or ""
        if chunk:
            self.hold += chunk
        return self._drain(flush=False)

    def flush(self) -> str:
        if not self.rules:
            rest, self.hold = self.hold, ""
            return rest
        out = self._drain(flush=True)
        rest, self.hold = self.hold, ""
        return out + rest

    def _drain(self, flush: bool) -> str:
        out: list[str] = []
        while self.hold:
            hit: tuple[int, int, str] | None = None
            for target, replace in self.rules:
                at = self.hold.find(target)
                if at < 0:
                    continue
                n = len(target)
                if hit is None or at < hit[0] or (at == hit[0] and n > hit[1]):
                    hit = (at, n, replace)
            if hit is None:
                if flush:
                    break
                keep = self._partial_suffix()
                emit, self.hold = self.hold[: len(self.hold) - keep], self.hold[len(self.hold) - keep :]
                if emit:
                    out.append(emit)
                break
            at, n, replace = hit
            if at:
                out.append(self.hold[:at])
            out.append(replace)
            self.hold = self.hold[at + n :]
        return "".join(out)

    def _partial_suffix(self) -> int:
        buf = self.hold
        best = 0
        for target, _ in self.rules:
            limit = min(len(target) - 1, len(buf))
            for k in range(limit, 0, -1):
                if buf.endswith(target[:k]):
                    best = max(best, k)
                    break
        return best
