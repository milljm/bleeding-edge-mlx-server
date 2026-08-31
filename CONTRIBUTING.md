# Contributing

This repo is a small stdlib-only CLI plus a conda-forge recipe. The engines
themselves live upstream:

- [ml-explore/mlx](https://github.com/ml-explore/mlx)
- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

## Dev

```bash
conda create -n edge python=3.13 uv pip git
conda activate edge
uv pip install -e .
python -m unittest discover -s tests -v
```

Keep `mlx_edge` free of runtime dependencies. If a check needs `mlx`, it must
degrade when the import fails (`doctor`). `serve` / `load` spawn
`mlx_lm.server` / `mlx_vlm.server` as children; unit tests inject a fake spawn.

## Recipe

`conda-recipe/meta.yaml` is the feedstock sketch. When submitting to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes):

1. Tag `v0.2.0` (or whatever `meta.yaml` version is).
2. Prefer a PyPI sdist over `git_url` once the package is on PyPI.
3. Keep `noarch: python`.
4. Maintainer: `milljm`.

Do not add `pip` overlay behaviour to the conda build itself. The build
installs the CLI; `uv pip install -r requirements.txt` (or `mlx-edge update`)
is how users reach git HEAD of the engines.
