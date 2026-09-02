# HTTP API

The gateway defaults to `127.0.0.1:8080`. Each loaded model is its own
`mlx_lm.server` / `mlx_vlm.server` child. The gateway is the one OpenAI URL.

`GET /v1/models` lists each loaded engine by **basename**
(`MiniMax-M2.7-ConfigI-MLX`), not the full disk path. Chat `model` may be that
basename (any case), `org/name`, or the path.

## Endpoints

- `GET /` — Edge GUI (`edge-gui` / `mlx-edge serve --gui`)
- `GET /v1/models` — every hot-loaded model, listed by basename. Each row is
  OpenAI-shaped (`id`, `object`, `created`, `owned_by`) plus the checkpoint's
  context window when `config.json` has one: `context_length`, `max_model_len`,
  and `max_context_length` (same integer). Cline / Continue / Open WebUI read
  these. `GET /v1/models/{id}` is the same object for one loaded engine.
  `GET /api/v0/models` is the LM Studio-shaped list (`type`, `state`,
  `max_context_length`, `loaded_context_length`, `capabilities.tool_use`) for
  clients that probe that. Chat/VLM rows also advertise `function_calling` /
  `tool_use` so Cline knows native tools are on.
- `POST /v1/chat/completions` — routed by basename / Hub id / path. The gateway
  pins the request to the already-loaded engine so mlx-lm does not Hub-download
  a second copy. Pass `"stream": true` for OpenAI SSE (`data: …` then
  `data: [DONE]`). Streaming requests always get `stream_options.include_usage`
  so the final chunk carries `usage.prompt_tokens` (Cline's context bar is
  usage / context_length). Tokens are flushed as they generate; the gateway does
  not buffer the child. Harmony `<|channel|>` wrappers and MiniMax thinking
  blocks (`<think>` / `<mm:think>`) are stripped from `content`
  (`reasoning_content` holds analysis). MiniMax `<minimax:tool_call>` XML,
  Qwen `<tool_call>` JSON, and Harmony `to=functions.NAME` calls are rewritten
  into OpenAI `tool_calls` with `finish_reason: "tool_calls"` — that is what
  Cline / Continue / Open WebUI execute. Tool XML is only buffered after
  `</think>` and only when the request actually has `tools`, so a MiniMax-M2
  chat without tools still streams token-by-token. If MiniMax never emits a
  closer, the buffered text is promoted back to `content` so the reply is not
  empty. Embedding models return 400 here — use `/v1/embeddings`.
- `POST /v1/embeddings` — OpenAI embeddings. Routed to a loaded `embed` engine
  (`mlx_vlm.server --embedding-model`). Body `model` is pinned to the spawn
  path. Does not touch chat children.
- `POST /v1/audio/speech` — OpenAI TTS. Routed to a loaded `tts` engine
  (`mlx_vlm.server --tts-model`). JSON `{model, input, voice?}`. Returns audio
  bytes. Playground stays text-only.
- `POST /v1/audio/transcriptions` — OpenAI STT. Routed to a loaded `stt` engine
  (`mlx_vlm.server --stt-model`). Multipart `file` + `model` (JSON is also
  accepted). Alias: `/v1/audio/translations`.
- `POST /v1/rerank` — Cohere/OpenAI-style rerank. Routed to a loaded `rerank`
  engine (`mlx_vlm.server --reranker-model`). JSON `{model, query, documents,
  top_n?}`. Body `model` is pinned to the spawn path.
- `POST /v1/images/generations` — OpenAI image generation. Routed to a loaded
  `image` engine (`mlx_vlm.server --image-model`). JSON `{model, prompt, n?}`.
  Alias: `/v1/images/edits`. Playground stays text-only.
- `GET /v1/progress` — Edge-specific JSON snapshot of prompt processing
  (prefill) and decode. Does not change the OpenAI surface. Top-level
  `progress` and `models[].progress` are always floats in `[0.0, 1.0]` (idle
  `0.0`, prefill = prompt ratio, decode/done `1.0`). `?model=` filters by
  basename. Alias: `GET /edge/progress`.
- `GET /v1/progress/stream` — the same object as SSE whenever it changes.
  Alias: `GET /edge/progress/stream`.
- `GET /v1/logs` / `GET /v1/logs/stream` — engine stdout ring buffer (CLI-like).
  `POST /v1/logs/clear` empties it.
- `GET`/`POST /v1/template` — inspect or pull a Jinja chat template (local
  checkpoint → Hugging Face → Harmony preset for gpt-oss / ConfigI).
- `GET`/`PUT /v1/prefs` — watch dirs and per-model flags
  (`~/.config/mlx-edge/studio.json`)
- `GET`/`PUT`/`DELETE /v1/playground` — one rolling Playground transcript shared
  by every model (RAM only — a browser reload keeps it, quitting Edge drops it).
  `PUT {turns}`. `DELETE` / `POST /v1/playground/clear` empties it.
- `POST /v1/completions` — routed by `model`
- `POST /v1/load` — hot-load `{engine, model, args?}` (replaces the same id).
  `engine` is `lm` | `vlm` | `embed` | `tts` | `stt` | `rerank` | `image`. After the child is healthy, Edge sends a
  1-token warmup (or a tiny embed) so Metal graphs are compiled before the
  first real request. After that the child sits idle until a client hits it.
- `POST /v1/unload` — unload `{model}`
- `POST /v1/stop` — abort in-flight chat/embed for `{model}` (omit `model` to stop
  every busy engine). Closes the child so mlx-lm actually stops generating.
  A remote OpenAI client that drops the stream is observed the same way —
  Edge cancels the child and the generating pulse clears. Alias: `/v1/chat/stop`.
- `POST /v1/scan` — `{dirs: […]}` → local MLX checkpoints (`config.json` + weights).
  Hugging Face hub layout (`models--org--name/snapshots/…`) is understood, so
  `~/.cache/huggingface/hub` works as a watch dir. Snapshot files are often
  symlinks into `blobs/`; scan follows those and also `.pth` / `.bin` / nested
  diffusers weights (`model_index.json`). A snapshot still needs a typed config
  (`model_type`, `architectures`, or `_class_name`). hexgrad/Kokoro's
  `config.json` is an istftnet dump with none of those — skipped.
  mlx-community conversions are kept. Encoder-only checkpoints (`clip`, `siglip`,
  `vit`) are skipped — mlx-vlm has no Serve path (it looks for a speculative
  drafter). 1-bit quants are skipped (mlx supports 2/3/4/5/6/8).
  `datasets--` / `spaces--` / `blobs` / `refs` are skipped.
- `GET /v1/hub` — `{token}` whether `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` is set
  (the value is never returned). `POST /v1/hub/search` `{query}` (Hub URL or
  `org/name`) lists MLX quants (mlx-community / `library_name=mlx`, no 1-bit).
  `POST /v1/hub/download` `{repo}` runs `huggingface_hub.snapshot_download`
  into `~/.cache/huggingface/hub`.
- `GET /health` — `{status, models, host, port, bind, url}`

## Progress

Prefill (reading the prompt) is the slow part on long contexts. Poll it from
another process without touching OpenAI chat:

```bash
curl http://127.0.0.1:8080/v1/progress
curl http://127.0.0.1:8080/v1/progress?model=MiniMax-M2.7-ConfigI-MLX
curl -N http://127.0.0.1:8080/v1/progress/stream
```

```json
{
  "object": "edge.progress",
  "version": 1,
  "generated_at": 1756670123.45,
  "active": true,
  "progress": 0.3131,
  "models": [
    {
      "id": "MiniMax-M2.7-ConfigI-MLX",
      "engine": "lm",
      "phase": "prefill",
      "status": "processing",
      "stream": true,
      "progress": 0.3131,
      "prompt": {
        "processed_tokens": 2048,
        "total_tokens": 6540,
        "ratio": 0.3131,
        "cached_tokens": null,
        "started_at": 1756670120.1,
        "updated_at": 1756670122.4,
        "tokens_per_second": 820.1
      },
      "generation": {
        "tokens": 0,
        "started_at": null,
        "updated_at": null,
        "tokens_per_second": null
      },
      "error": null
    }
  ]
}
```

`phase` is `idle` | `loading` | `prefill` | `decode` | `done` | `error`.
`progress` is always a float `0.0`–`1.0` (never `null`). During Serve,
`phase` is `loading` and `progress` tracks whatever the child logs (tqdm
percent, `Fetching n/m`, download bytes) — mlx-lm does not always emit a
ratio, so the GUI also eases the card fill until the engine is healthy.
New keys can land later under the same object (`version` bumps if the
meaning of a field changes). Numbers come from mlx-lm keepalives
(`: keepalive 2048/6540`) and from child logs
(`Prompt processing progress: 2048/6540`, mlx-vlm `Prefill progress: …`).

Live updates from another app (browser or Node):

```js
const es = new EventSource("http://127.0.0.1:8080/v1/progress/stream");
es.onmessage = (ev) => {
  const snap = JSON.parse(ev.data);
  const row = snap.models[0];
  const p = row?.progress ?? snap.progress; // 0.0 .. 1.0
};
```

## Chat templates and thinking tags

MiniMax-M2.7 and MiniMax-M3 Hugging Face templates wrap thinking in
`<think>…</think>` / `<mm:think>…</mm:think>` and put the opener in the
**prompt**, so the first generated tokens are thinking. ConfigI and gpt-oss
use Harmony `<|channel|>` tokens instead.

On Serve, if the folder has no template, Edge pulls one from Hugging Face
and passes `--chat-template` to **mlx-lm.server**. mlx-vlm.server has no
`--chat-template` / `--temp` / `--top-p` / `--prompt-cache-size` — sampling
is on the request, thinking is `--enable-thinking`. TTS, STT, embed, rerank, and
image-gen are their own Serve engines (scan tags kokoro / whisper / bge /
Qwen3-Reranker / FLUX, or force Engine). Harmony is only injected as a fallback for gpt-oss / ConfigI names — not for
generic MiniMax-M2.7 / M3 (those would start in the wrong dialect and yield
empty `content`). Settings → Chat template lets you paste Jinja or **Pull from
Hugging Face**, then Reload.

The gateway splits thinking into `reasoning_content` and keeps OpenAI
`content` as the visible answer. Thinking tokens stream immediately (they are
not buffered until the closer). After `</think>` / `</mm:think>` the answer
streams as `content`. **Tool calls** (MiniMax XML — including tokenizer
glyphs like `]<]minimax[>[` and mlx-lm dropping `<` on invoke tags — Qwen
`<tool_call>`, Harmony `to=functions.NAME`) become OpenAI `tool_calls` so
agent clients can execute them. If MiniMax never emits a closer, the thinking is promoted
back to `content` so clients do not see an empty reply. `[DONE]` is held until
that flush. **ConfigI / gpt-oss** start in Harmony's final channel — those
tokens are `content` from the first delta (clients like LangChain that only
read `delta.content` would otherwise see empty chunks and one dump at EOS).
Plain Qwen / Llama output is untouched. Injected MiniMax / Harmony chat
templates include a `# Tools` block when the request has `tools`.

## Hot-load vs LM Studio

Edge **does** keep several models resident: each Serve starts its own
`mlx_lm.server` / `mlx_vlm.server` child and leaves the others running. Switching
`model` on `/v1` is a route to a process that is already up, not an unload.

LM Studio’s mlx-engine is **one process** with several ModelKits on one Metal
device, so flipping A → B is an in-process handle switch. mlx-lm’s server holds
one model at a time, so Edge’s way to keep two chat models loaded is two
processes. The extra seconds you see on the first token after a switch are:

1. Metal context switch between processes
2. Graph compile on a cold shape (Edge now warms a 1-token request after Serve)
3. Prompt prefill of *this* request (see `GET /v1/progress`) — that cost exists
   in LM Studio too on a long prompt

RAG should Serve the embedding model **and** the chat model. Embeddings run in
their own process. The RAG round-trip (embed → retrieve → chat with a longer
prompt) is extra work in your client, not an Edge unload.

## Examples

Stream chat:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"MiniMax-M2.7-ConfigI-MLX","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Embeddings (RAG):

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen3-Embedding-0.6B-4bit","input":"what is the capital of France?"}'
```
