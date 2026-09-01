"""Inspect, fetch, and apply chat templates.

MiniMax / gpt-oss checkpoints often ship without ``chat_template`` in
``tokenizer_config.json``. mlx-lm then does not wrap messages, and MiniMax
buffers the whole think+answer until EOS (the playground dumps at the end).

LM Studio's working MiniMax-M2.7 prompt ends with::

    ]~b]ai
    <think>

Edge injects that generation prompt when the checkpoint has no think-capable
template. ConfigI / gpt-oss still get Harmony ``<|channel|>``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mlx_edge import __version__

USER_AGENT = f"mlx-edge/{__version__}"

HARMONY_TEMPLATE = """{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
<|start|>system<|message|>{{ message['content'] }}<|end|>
{%- elif message['role'] == 'user' -%}
<|start|>user<|message|>{{ message['content'] }}<|end|>
{%- elif message['role'] == 'assistant' -%}
<|start|>assistant<|channel|>final<|message|>{{ message['content'] }}<|end|>
{%- else -%}
<|start|>{{ message['role'] }}<|message|>{{ message['content'] }}<|end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start|>assistant<|channel|>final<|message|>
{%- endif -%}
"""

# Official MiniMax-M2 / M2.7 generation prompt (LM Studio / MiniMaxAI jinja).
# Special tokens `]~b]` / `[e~[` are literal pieces of the tokenizer vocab.
MINIMAX_THINK_TEMPLATE = r"""{%- set model_identity = "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax." -%}
{%- set system_message = none -%}
{%- set conversation_messages = messages -%}
{%- if messages and messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0] -%}
    {%- set conversation_messages = messages[1:] -%}
{%- endif -%}
{{- ']~!b[' ~ ']~b]system' ~ '\n' }}
{%- if system_message and system_message['content'] -%}
    {{- system_message['content'] }}
{%- else -%}
    {{- model_identity }}
{%- endif -%}
{{- '[e~[' ~ '\n' }}
{%- for message in conversation_messages -%}
    {%- if message['role'] == 'user' -%}
        {{- ']~b]user' ~ '\n' ~ message['content'] ~ '[e~[' ~ '\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{- ']~b]ai' ~ '\n' }}
        {%- if message['reasoning_content'] -%}
            {{- '<think>' ~ '\n' ~ message['reasoning_content'] ~ '\n</think>' ~ '\n\n' }}
        {%- endif -%}
        {{- message['content'] }}
        {{- '[e~[' ~ '\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- ']~b]ai' ~ '\n' ~ '<think>' ~ '\n' }}
{%- endif -%}
"""

HF_NAME_HINTS = (
    ("minimax", ("MiniMaxAI/MiniMax-M2.7", "MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-M3", "mlx-community/MiniMax-M2.7-4bit")),
    ("gpt-oss", ("openai/gpt-oss-20b",)),
    ("gpt_oss", ("openai/gpt-oss-20b",)),
)


def template_opens_generation(tmpl: str | None) -> bool:
    """True if the jinja already puts the model in a think / Harmony channel."""
    if not tmpl:
        return False
    lower = tmpl.lower()
    # `]~b]ai` is only the MiniMax role marker — not enough. LM Studio streams
    # because the generation prompt also opens <think>.
    return (
        "<think>" in lower
        or "<mm:think>" in lower
        or "<|channel|>" in lower
        or "<|message|>" in tmpl
    )


def has_local_template(model_path: str) -> bool:
    path = Path(os.path.expanduser(model_path))
    if not path.is_dir():
        return False
    if (path / "chat_template.jinja").is_file():
        return True
    cfg = _read_json(path / "tokenizer_config.json")
    tmpl = cfg.get("chat_template") if cfg else None
    return bool(tmpl)


def read_local_template(model_path: str) -> str | None:
    path = Path(os.path.expanduser(model_path))
    jinja = path / "chat_template.jinja"
    if jinja.is_file():
        try:
            text = jinja.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            return text
    cfg = _read_json(path / "tokenizer_config.json")
    return _coerce_template(cfg.get("chat_template") if cfg else None)


def inspect_template(model_path: str, repo: str | None = None) -> dict[str, Any]:
    local = read_local_template(model_path)
    return {
        "path": model_path,
        "repo": repo or "",
        "bundled": bool(local),
        "source": "checkpoint" if local else None,
        "chat_template": local,
        "preset": _preset_for(model_path, repo),
    }


def fetch_template(model_path: str, repo: str | None = None) -> dict[str, Any]:
    info = inspect_template(model_path, repo)
    local = info.get("chat_template")
    preset = info.get("preset")
    if isinstance(local, str) and local.strip():
        # MiniMax quants often ship a stub template that does *not* open
        # `<think>`. Treat that as missing so we can inject.
        if template_opens_generation(local) or preset != "minimax-think":
            return info
    tried: list[str] = []
    for candidate in _hf_candidates(model_path, repo):
        tried.append(candidate)
        tmpl = _hf_tokenizer_template(candidate)
        if tmpl:
            info["chat_template"] = tmpl
            info["source"] = f"huggingface:{candidate}"
            info["bundled"] = False
            return info
    if preset == "harmony":
        info["chat_template"] = HARMONY_TEMPLATE
        info["source"] = "preset:harmony"
        return info
    if preset == "minimax-think":
        info["chat_template"] = MINIMAX_THINK_TEMPLATE
        info["source"] = "preset:minimax-think"
        return info
    info["tried"] = tried
    return info


def template_for_spawn(model_path: str, extra: list[str]) -> list[str]:
    """If mlx-lm / mlx-vlm would have no think-capable template, inject --chat-template."""
    if any(arg == "--chat-template" or arg.startswith("--chat-template=") for arg in extra):
        return extra
    path = Path(os.path.expanduser(model_path))
    if not path.is_dir():
        return extra
    local = read_local_template(model_path)
    preset = _preset_for(model_path, None)
    if local and (template_opens_generation(local) or preset != "minimax-think"):
        return extra
    info = fetch_template(model_path)
    tmpl = info.get("chat_template")
    if not isinstance(tmpl, str) or not tmpl.strip():
        return extra
    return [*extra, "--chat-template", tmpl]


def _preset_for(model_path: str, repo: str | None) -> str | None:
    blob = f"{model_path} {repo or ''}".lower()
    # Official MiniMax-M2.7 / M3 templates use <think> / <mm:think>, not Harmony.
    # Harmony is gpt-oss and thetom-ai ConfigI conversions.
    if "gpt-oss" in blob or "gpt_oss" in blob or "harmony" in blob or "configi" in blob:
        return "harmony"
    if "minimax" in blob:
        return "minimax-think"
    return None


def _hf_candidates(model_path: str, repo: str | None) -> list[str]:
    out: list[str] = []
    name = Path(model_path.rstrip("/")).name
    if repo and "/" in repo and not repo.startswith("/") and " " not in repo:
        out.append(repo)
    for needle, aliases in HF_NAME_HINTS:
        if needle in name.lower() or needle in (repo or "").lower():
            out.extend(aliases)
    # mlx-community/<folder> is a common layout
    if name and not name.startswith("."):
        out.append(f"mlx-community/{name}")
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq


def _hf_tokenizer_template(repo: str) -> str | None:
    urls = (
        f"https://huggingface.co/{repo}/resolve/main/chat_template.jinja",
        f"https://huggingface.co/{repo}/resolve/main/tokenizer_config.json",
    )
    for url in urls:
        raw = _http_text(url)
        if not raw:
            continue
        if url.endswith(".jinja"):
            text = raw.strip()
            if "{%" in text or "<|" in text or "<think>" in text.lower() or "]~b]" in text:
                return text
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        tmpl = _coerce_template(data.get("chat_template") if isinstance(data, dict) else None)
        if tmpl:
            return tmpl
    return None


def _coerce_template(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
            if isinstance(item, dict):
                tmpl = item.get("template") or item.get("chat_template")
                if isinstance(tmpl, str) and tmpl.strip():
                    return tmpl
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _http_text(url: str, timeout: float = 8.0) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) >= 400:
                return None
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError):
        return None
