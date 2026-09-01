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
  `max_context_length`, `loaded_context_length`) for clients that probe that.
- `POST /v1/chat/completions` — routed by basename / Hub id / path. The gateway
  pins the request to the already-loaded engine so mlx-lm does not Hub-download
  a second copy. Pass `"stream": true` for OpenAI SSE (`data: …` then
  `data: [DONE]`). Tokens are flushed as they generate; the gateway does not
  buffer the child. Harmony `<|channel|>` wrappers and MiniMax thinking blocks
  (`<think>` / `<mm:think>`) are stripped from `content`
  (`reasoning_content` holds analysis). If MiniMax never emits a closer, the
  buffered text is promoted back to `content` so the reply is not empty.
  Embedding models return 400 here — use `/v1/embeddings`.
- `POST /v1/embeddings` — OpenAI embeddings. Routed to a loaded `embed` engine
  (`mlx_vlm.server --embedding-model`). Body `model` is pinned to the spawn
  path. Does not touch chat children.
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
- `POST /v1/completions` — routed by `model`
- `POST /v1/load` — hot-load `{engine, model, args?}` (replaces the same id).
  `engine` is `lm` | `vlm` | `embed`. After the child is healthy, Edge sends a
  1-token warmup (or a tiny embed) so Metal graphs are compiled before the
  first real request.
- `POST /v1/unload` — unload `{model}`
- `POST /v1/stop` — abort in-flight chat/embed for `{model}` (omit `model` to stop
  every busy engine). Closes the child so mlx-lm actually stops generating.
  A remote OpenAI client that drops the stream is observed the same way —
  Edge cancels the child and the generating pulse clears. Alias: `/v1/chat/stop`.
- `POST /v1/scan` — `{dirs: […]}` → local MLX checkpoints (`config.json` + weights)
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

On Serve, if the folder has no template, Edge pulls one from Hugging Face.
Harmony is only injected as a fallback for gpt-oss / ConfigI names — not for
generic MiniMax-M2.7 / M3 (those would start in the wrong dialect and yield
empty `content`). Settings → Chat template lets you paste Jinja or **Pull from
Hugging Face**, then Reload.

The gateway splits thinking into `reasoning_content` and keeps OpenAI
`content` as the visible answer. Thinking tokens stream immediately (they are
not buffered until the closer). After `</think>` / `</mm:think>` the answer
streams as `content`. If MiniMax never emits a closer, the thinking is promoted
back to `content` so clients do not see an empty reply. `[DONE]` is held until
that flush. Plain Qwen / Llama output is untouched.

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
their own process and stay warm (12s heartbeat). After embed, Edge re-warms the
chat graphs in the background so the RAG hop is not a cold Metal switch. The
RAG round-trip (embed → retrieve → chat with a longer prompt) is extra work
in your client, not an Edge unload.

## Keep-hot embeddings

Embedding children get a 12s heartbeat so RAG does not hit a cold Metal
graph. After an embeddings request, Edge warms loaded chat models in the
background (overlaps retrieval). After chat, it re-warms the embedding
model. llm → llm was already fast after the post-load warmup.

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
