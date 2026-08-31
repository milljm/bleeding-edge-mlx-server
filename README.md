# Bleeding Edge MLX Server

Edge is a local OpenAI-compatible gateway for Apple Silicon. It hot-loads
`mlx-lm` and `mlx-vlm` models side by side on one host/port. `edge-gui` is the
studio: Serve talks to `mlx-edge` over `/v1`. Chat tools use the same URL.

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
`~/.config/mlx-edge/studio.json`.

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
```

Or preload at start:

```bash
edge-gui --host 127.0.0.1 --port 8080 \
  --lm mlx-community/Qwen3-8B-4bit \
  --vlm mlx-community/Qwen2.5-VL-7B-Instruct-4bit
```

`GET /v1/models` lists everything currently loaded. `POST /v1/chat/completions`
routes on `model`. Unload one without touching the others:

```bash
mlx-edge unload --model mlx-community/Qwen3-8B-4bit
```

You will need to `conda activate edge` in every new shell.

## Daily

```bash
conda activate edge
mlx-edge update          # refresh git HEAD of mlx-lm and mlx-vlm
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
| Vision engine | `mlx-vlm` | git overlay | Pure Python. Tracks VLMs / omni models. |
| CLI | `mlx-edge` | this repo | `serve`, `load`, `unload`, `update`, `status`. |
| GUI | `edge-gui` | this repo | Studio that drives the CLI over `/v1`. |

## Safety

HEAD can be broken. Pin a SHA that works; roll back to PyPI/conda-forge if it is not.

```bash
mlx-edge pin                     # write ~/.config/mlx-edge/pins.json
mlx-edge update --pinned         # reinstall those SHAs
mlx-edge rollback                # conda-forge mlx-lm + mlx-vlm, mlx untouched
mlx-edge update lm --ref abc123  # one engine, one commit
```

`mlx-edge update mlx` is refused unless you pass `--force`.

## CLI

```
edge-gui [--host 127.0.0.1] [--port 8080] [--lm MODEL]... [--vlm MODEL]... [--no-browser]
mlx-edge serve [--host 127.0.0.1] [--port 8080] [--gui] [--lm MODEL]... [--vlm MODEL]...
mlx-edge load --engine lm|vlm --model MODEL [engine flags…]
mlx-edge unload --model MODEL
mlx-edge models
mlx-edge status [--json] [--offline]
mlx-edge update [lm|vlm|all] [--ref SHA] [--branch main] [--pinned] [--force] [--with-deps]
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
- `POST /v1/chat/completions` — routed by basename / Hub id / path. The gateway pins the request to the already-loaded engine so mlx-lm does not Hub-download a second copy. Pass `"stream": true` for OpenAI SSE (`data: …` then `data: [DONE]`). Tokens are flushed as they generate; the gateway does not buffer the child.
- `GET /v1/progress` — Edge-specific JSON snapshot of prompt processing (prefill) and decode. Does not change the OpenAI surface. `?model=` filters by basename. Alias: `GET /edge/progress`.
- `GET /v1/progress/stream` — the same object as SSE whenever it changes. Alias: `GET /edge/progress/stream`.
- `GET`/`PUT /v1/prefs` — watch dirs and per-model flags (`~/.config/mlx-edge/studio.json`)
- `POST /v1/completions` — routed by `model`
- `POST /v1/load` — hot-load `{engine, model, args?}` (replaces the same id)
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
  "models": [
    {
      "id": "MiniMax-M2.7-ConfigI-MLX",
      "engine": "lm",
      "phase": "prefill",
      "status": "processing",
      "stream": true,
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

`phase` is `idle` | `prefill` | `decode` | `done` | `error`. New keys can land
later under the same object (`version` bumps if the meaning of a field changes).
Numbers come from mlx-lm keepalives (`: keepalive 2048/6540`) and from child
logs (`Prompt processing progress: 2048/6540`, mlx-vlm `Prefill progress: …`).

Stream chat as usual:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"MiniMax-M2.7-ConfigI-MLX","messages":[{"role":"user","content":"hello"}],"stream":true}'
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
