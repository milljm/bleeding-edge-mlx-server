# Bleeding Edge MLX Server

Edge is a local OpenAI-compatible gateway for Apple Silicon. It hot-loads
`mlx-lm`, `mlx-vlm`, and embedding models side by side on one host/port.
`edge-gui` is the studio: Serve talks to `mlx-edge` over `/v1`. Chat tools use
the same URL.

Compiled `mlx` comes from a wheel. The Python engines install from git HEAD so
new architectures land without waiting on a conda-forge rebuild.

## Getting Started

Apple Silicon. [Miniforge](https://github.com/conda-forge/miniforge) is the
usual conda.

```bash
conda create -n edge python=3.13 uv pip git
conda activate edge
git clone https://github.com/milljm/bleeding-edge-mlx-server.git
cd bleeding-edge-mlx-server
uv pip install -r requirements.txt
edge-gui
```

`git` is in the conda create because `requirements.txt` pulls `mlx-lm` and
`mlx-vlm` from git HEAD. `uv pip` targets the active conda env (`CONDA_PREFIX`).

`edge-gui` starts the gateway **and** the studio on the same host/port, then
opens a browser. The footer shows the OpenAI base URL:

```
Serving on http://127.0.0.1:8080/v1
```

Point any OpenAI-compatible chat client at that address. `GET /v1/models`
lists each loaded engine by **basename** (`MiniMax-M2.7-ConfigI-MLX`), not the
full disk path. Chat `model` may be that basename (any case), `org/name`, or
the path. Watch folders and per-model settings persist in
`~/.config/mlx-edge/studio.json` (including `~/.lmstudio/models` if that is the
folder you added).

Remote clients (another machine on the LAN):

```bash
edge-gui --host 0.0.0.0 --port 8080
```

The footer then shows your LAN address, e.g. `http://192.168.1.50:8080/v1`.

Headless (no GUI):

```bash
mlx-edge serve --host 127.0.0.1 --port 8080
mlx-edge load --engine lm --model mlx-community/Qwen3-8B-4bit
mlx-edge load --engine vlm --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
mlx-edge load --engine embed --model mlx-community/Qwen3-Embedding-0.6B-4bit
```

Or preload at start:

```bash
edge-gui --host 127.0.0.1 --port 8080 \
  --lm mlx-community/Qwen3-8B-4bit \
  --vlm mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --embed mlx-community/Qwen3-Embedding-0.6B-4bit
```

`GET /v1/models` lists everything currently loaded. `POST /v1/chat/completions`
routes on `model`. `POST /v1/embeddings` routes to a loaded `embed` engine.
Unload one without touching the others:

```bash
mlx-edge unload --model mlx-community/Qwen3-8B-4bit
```

You will need to `conda activate edge` in every new shell.

## Daily

```bash
conda activate edge
mlx-edge update          # refresh git HEAD of mlx-lm and mlx-vlm
mlx-edge build --help    # overlay a not-yet-merged PR (new model classes)
mlx-edge status          # local vs conda-forge vs PyPI vs git
edge-gui
```

`mlx-edge update` runs `pip install --upgrade --force-reinstall --no-deps git+…`
so pip cannot replace the compiled `mlx` wheel.

## Why this split

| Layer | Package | Source of truth | Why |
| --- | --- | --- | --- |
| Runtime | `mlx` | PyPI wheel (or conda-forge) | Compiled Metal. Do not rebuild from git unless you mean to. |
| Text engine | `mlx-lm` | git overlay | Pure Python. Tracks new LLM architectures. |
| Vision engine | `mlx-vlm` | git overlay | Pure Python. Tracks VLMs / omni models. Also serves embeddings. |
| CLI | `mlx-edge` | this repo | `serve`, `load`, `unload`, `update`, `build`, `status`. |
| GUI | `edge-gui` | this repo | Studio that drives the CLI over `/v1`. |

Dedicated embedding checkpoints (Qwen3-Embedding, bge, e5, gte, nomic, MiniLM)
scan as engine `embed` and spawn `mlx_vlm.server --embedding-model PATH` — they
do not go through mlx-lm, which has no embeddings endpoint.

Scan guesses `lm` / `vlm` / `embed` from the checkpoint. MiniMax-M3
(`minimax_m3_vl`) looks like a VLM but the working loader is patched mlx-lm
(`mlx-edge build`). Settings → Engine forces mlx-lm (or mlx-vlm) per model;
it sticks in `~/.config/mlx-edge/studio.json`. `mlx-edge load --engine lm`
does the same from the CLI.

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

## Safety

HEAD can be broken. Pin a SHA that works; roll back to PyPI/conda-forge if it is not.

```bash
mlx-edge pin                     # write ~/.config/mlx-edge/pins.json
mlx-edge update --pinned         # reinstall those SHAs
mlx-edge rollback                # conda-forge mlx-lm + mlx-vlm, mlx untouched
mlx-edge update lm --ref abc123  # one engine, one commit
```

`mlx-edge update mlx` is refused unless you pass `--force`.

New architectures often land in a GitHub PR days before they merge. Overlay that
exact ref without waiting:

```bash
mlx-edge build --help
mlx-edge build git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1398/head
mlx-edge build 1398              # mlx-lm pull request
mlx-edge build mlx-vlm#42
```

`mlx-edge build --help` prints the mlx-lm and mlx-vlm pulls URLs. Serve failures
that look like a missing model class hint at the same command.

## CLI

```
edge-gui [--host 127.0.0.1] [--port 8080] [--lm MODEL]... [--vlm MODEL]... [--embed MODEL]... [--no-browser]
mlx-edge serve [--host 127.0.0.1] [--port 8080] [--gui] [--lm MODEL]... [--vlm MODEL]... [--embed MODEL]...
mlx-edge load --engine lm|vlm|embed --model MODEL [engine flags…]
mlx-edge unload --model MODEL
mlx-edge models
mlx-edge status [--json] [--offline]
mlx-edge update [lm|vlm|all] [--ref SHA] [--branch main] [--pinned] [--force] [--with-deps]
mlx-edge build [SPEC] [--engine lm|vlm|mlx] [--force] [--with-deps]
mlx-edge pin
mlx-edge rollback [lm|vlm|all]
mlx-edge doctor
mlx-edge which
mlx-edge engines
```

`--with-deps` lets pip resolve dependencies. That can replace the `mlx` wheel. Do not use it casually.

Old single-engine form still works and now hot-loads onto the gateway:

```bash
mlx-edge serve --engine lm --model mlx-community/Qwen3-8B-4bit
```

## Server endpoints

Gateway (defaults `127.0.0.1:8080`):

- `GET /` — Edge GUI (`edge-gui` / `mlx-edge serve --gui`)
- `GET /v1/models` — every hot-loaded model, listed by basename
- `POST /v1/chat/completions` — routed by basename / Hub id / path. The gateway pins the request to the already-loaded engine so mlx-lm does not Hub-download a second copy. Pass `"stream": true` for OpenAI SSE (`data: …` then `data: [DONE]`). Tokens are flushed as they generate; the gateway does not buffer the child. Harmony `<|channel|>` wrappers and MiniMax `<think>` / `<mm:think>` blocks are stripped from `content` (`reasoning_content` holds analysis). If MiniMax never emits a closer, the buffered text is promoted back to `content` so the reply is not empty. Embedding models return 400 here — use `/v1/embeddings`.
- `POST /v1/embeddings` — OpenAI embeddings. Routed to a loaded `embed` engine (`mlx_vlm.server --embedding-model`). Body `model` is pinned to the spawn path. Does not touch chat children.
- `GET /v1/progress` — Edge-specific JSON snapshot of prompt processing (prefill) and decode. Does not change the OpenAI surface. Top-level `progress` and `models[].progress` are always floats in `[0.0, 1.0]` (idle `0.0`, prefill = prompt ratio, decode/done `1.0`). `?model=` filters by basename. Alias: `GET /edge/progress`.
- `GET /v1/progress/stream` — the same object as SSE whenever it changes. Alias: `GET /edge/progress/stream`.
- `GET /v1/logs` / `GET /v1/logs/stream` — engine stdout ring buffer (CLI-like). `POST /v1/logs/clear` empties it.
- `GET`/`POST /v1/template` — inspect or pull a Jinja chat template (local checkpoint → Hugging Face → Harmony preset for gpt-oss / ConfigI).
- `GET`/`PUT /v1/prefs` — watch dirs and per-model flags (`~/.config/mlx-edge/studio.json`)
- `POST /v1/completions` — routed by `model`
- `POST /v1/load` — hot-load `{engine, model, args?}` (replaces the same id). `engine` is `lm` | `vlm` | `embed`. After the child is healthy, Edge sends a 1-token warmup (or a tiny embed) so Metal graphs are compiled before the first real request.
- `POST /v1/unload` — unload `{model}`
- `POST /v1/scan` — `{dirs: […]}` → local MLX checkpoints (`config.json` + weights)
- `GET /health` — `{status, models, host, port, bind, url}`

Each loaded model is its own `mlx_lm.server` / `mlx_vlm.server` child. The
gateway is the one OpenAI URL.

### Processing progress

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

`phase` is `idle` | `prefill` | `decode` | `done` | `error`. `progress` is always
a float `0.0`–`1.0` (never `null`). New keys can land later under the same
object (`version` bumps if the meaning of a field changes). Numbers come from
mlx-lm keepalives (`: keepalive 2048/6540`) and from child logs
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

### Chat templates and thinking tags

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

### Studio

The Models sidebar sorts loaded (and currently loading) cards above the rest
and tints them orange in both light and dark themes. While a user chat or
embed request is in flight, that card pulses orange↔green and says
`generating`. Keep-hot embedding heartbeats and post-request graph warmups do
not count — idle loaded models stay orange, not generating.

### Logging

The Logging tab tails child stdout over `GET /v1/logs/stream` (SSE). Filter by
model or errors. `GET /v1/logs` is the JSON snapshot for other apps.

### Keep-hot embeddings

Embedding children get a 12s heartbeat so RAG does not hit a cold Metal
graph. After an embeddings request, Edge warms loaded chat models in the
background (overlaps retrieval). After chat, it re-warms the embedding
model. llm → llm was already fast after the post-load warmup.

Stream chat as usual:

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

## GUI source

`gui/` is the studio. Prebuilt files ship in `src/mlx_edge/web/` so `edge-gui`
does not need Node. To rebuild the bundle:

```bash
npm install
npm run build:gui
```

## Doctor

```bash
mlx-edge doctor
```

Checks Darwin/arm64, `CONDA_DEFAULT_ENV`, git, Metal via `mlx.core.metal`, and whether each engine imports.

## Conda-forge recipe

`conda-recipe/` is a noarch Python feedstock sketch for submitting `mlx-edge` to conda-forge. Until that is accepted, `uv pip install -r requirements.txt` inside the conda env as above.

## License

MIT. `mlx`, `mlx-lm`, and `mlx-vlm` keep their own licenses.
