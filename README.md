# Bleeding Edge MLX Server

Edge is a local OpenAI-compatible gateway for Apple Silicon. It hot-loads
`mlx-lm` and `mlx-vlm` models side by side on one host/port. Chat tools speak
`/v1/models` and `/v1/chat/completions` and pick an engine with the `model`
field.

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
mlx-edge serve
```

`git` is in the conda create because `requirements.txt` pulls `mlx-lm` and
`mlx-vlm` from git HEAD. `uv pip` targets the active conda env (`CONDA_PREFIX`).
`mlx-edge serve` is the gateway — there is no `--do-stuff`. It stays up with an
empty pool until you load models.

```bash
mlx-edge load --engine lm --model mlx-community/Qwen3-8B-4bit
mlx-edge load --engine vlm --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
```

Or preload at start:

```bash
mlx-edge serve --host 127.0.0.1 --port 8080 \
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
mlx-edge serve --host 127.0.0.1 --port 8080
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
mlx-edge serve [--host 127.0.0.1] [--port 8080] [--lm MODEL]... [--vlm MODEL]...
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

- `GET /v1/models` — every hot-loaded model
- `POST /v1/chat/completions` — routed by `model`
- `POST /v1/completions` — routed by `model`
- `POST /v1/load` — hot-load `{engine, model, args?}`
- `POST /v1/unload` — unload `{model}`
- `GET /health`

Each loaded model is its own `mlx_lm.server` / `mlx_vlm.server` child on a
loopback ephemeral port. The gateway is the one OpenAI URL.

## Doctor

```bash
mlx-edge doctor
```

Checks Darwin/arm64, `CONDA_DEFAULT_ENV`, git, Metal via `mlx.core.metal`, and whether each engine imports.

## Conda-forge recipe

`conda-recipe/` is a noarch Python feedstock sketch for submitting `mlx-edge` to conda-forge. Until that is accepted, `uv pip install -r requirements.txt` inside the conda env as above.

## License

MIT. `mlx`, `mlx-lm`, and `mlx-vlm` keep their own licenses.
