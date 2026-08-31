# Bleeding Edge MLX Server

A conda-forge environment for Apple Silicon that keeps the **compiled MLX
runtime** on conda-forge and overlays **git HEAD** of `mlx-lm` and `mlx-vlm`
with one command.

New model architectures land in those Python packages days (sometimes hours)
before a conda-forge rebuild. `conda activate mlx-edge` then `mlx-edge update`
is the loop.

## Why this split

| Layer | Package | Source of truth | Why |
| --- | --- | --- | --- |
| Runtime | `mlx` | conda-forge | Compiled Metal. Do not rebuild from git unless you mean to. |
| Text engine | `mlx-lm` | git overlay | Pure Python. Tracks new LLM architectures. |
| Vision engine | `mlx-vlm` | git overlay | Pure Python. Tracks VLMs / omni models. |
| CLI | `mlx-edge` | this repo | `update`, `status`, `serve`, `pin`, `rollback`. |

`mlx-edge update` runs `pip install --upgrade --force-reinstall --no-deps git+…`
so pip cannot replace the conda `mlx` build.

## Install

Apple Silicon. conda-forge as the default (or highest-priority) channel. [Miniforge](https://github.com/conda-forge/miniforge) is the usual choice.

```bash
conda env create -f https://raw.githubusercontent.com/milljm/bleeding-edge-mlx-server/main/environment.yml
conda activate mlx-edge
mlx-edge update
mlx-edge status
```

Or without the yaml:

```bash
conda create -n mlx-edge -c conda-forge python=3.12 mlx mlx-lm mlx-vlm git pip huggingface_hub
conda activate mlx-edge
pip install git+https://github.com/milljm/bleeding-edge-mlx-server.git
mlx-edge update
```

You will need to `conda activate mlx-edge` in every new shell.

## Daily

```bash
conda activate mlx-edge
mlx-edge update          # overlay git HEAD of mlx-lm and mlx-vlm
mlx-edge status          # local vs conda-forge vs PyPI vs git
mlx-edge serve --engine lm --model mlx-community/Qwen3-8B-4bit
```

Vision / omni:

```bash
mlx-edge serve --engine vlm --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
```

`serve` is `os.execvp` onto `python -m mlx_lm.server` or `python -m mlx_vlm.server`. Extra flags pass through.

## Safety

HEAD can be broken. Pin a SHA that works; roll back to conda-forge if it is not.

```bash
mlx-edge pin                     # write ~/.config/mlx-edge/pins.json
mlx-edge update --pinned         # reinstall those SHAs
mlx-edge rollback                # conda-forge mlx-lm + mlx-vlm, mlx untouched
mlx-edge update lm --ref abc123  # one engine, one commit
```

`mlx-edge update mlx` is refused unless you pass `--force`. The compiled runtime stays on conda-forge.

## CLI

```
mlx-edge status [--json] [--offline]
mlx-edge update [lm|vlm|all] [--ref SHA] [--branch main] [--pinned] [--force] [--with-deps]
mlx-edge pin
mlx-edge rollback [lm|vlm|all]
mlx-edge serve --engine lm|vlm [--model ...] [--host ...] [--port ...] [engine flags…]
mlx-edge doctor
mlx-edge which
mlx-edge engines
```

`--with-deps` lets pip resolve dependencies. That can replace conda `mlx`. Do not use it casually.

## Doctor

```bash
mlx-edge doctor
```

Checks Darwin/arm64, `CONDA_DEFAULT_ENV`, git, Metal via `mlx.core.metal`, and whether each engine imports.

## Conda-forge recipe

`conda-recipe/` is a noarch Python feedstock sketch for submitting `mlx-edge` to conda-forge. Until that is accepted, install the CLI with pip *inside* the conda env as above. Run dependencies of the recipe are `mlx`, `mlx-lm`, `mlx-vlm`, `git`, and `pip` from conda-forge.

Build locally:

```bash
conda install -c conda-forge conda-build
conda build conda-recipe -c conda-forge
```

## Server endpoints

`mlx_lm.server` (defaults `127.0.0.1:8080`):

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /health`

`mlx_vlm.server` (defaults `0.0.0.0:8080`) is FastAPI, same OpenAI shape plus image/audio/video parts.

## License

MIT. `mlx`, `mlx-lm`, and `mlx-vlm` keep their own licenses.
