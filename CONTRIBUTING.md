# Contributing

This repo is a stdlib-only CLI + a bundled Edge GUI + a conda-forge recipe.
The engines themselves live upstream:

- [ml-explore/mlx](https://github.com/ml-explore/mlx)
- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

## Dev

```bash
conda activate edge
uv pip install -e .
python -m unittest discover -s tests -v
```

Keep `mlx_edge` free of runtime dependencies. If a check needs `mlx`, it must
degrade when the import fails (`doctor`, `serve`).

## GUI

`gui/` is the studio source. `src/mlx_edge/web/` is the prebuilt bundle
`edge-gui` serves. After changing the studio:

```bash
npm install
npm run build:gui
```

Commit both the source and the rebuilt `web/` files.

## Docs

`README.md` is the product page. Long API notes (endpoints, progress JSON,
thinking tags, keep-hot) live in `docs/api.md` — do not pile them back into
the README.

## Recipe

`conda-recipe/meta.yaml` is the feedstock sketch. When submitting to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes):

1. Tag `v0.3.0` (or whatever `meta.yaml` version is).
2. Prefer a PyPI sdist over `git_url` once the package is on PyPI.
3. Keep `noarch: python`. `mlx` as a run dep does the platform selection.
4. Maintainer: `milljm`.

Do not add `pip` overlay behaviour to the conda build itself. The build
installs the CLI; `mlx-edge update` is how users reach git HEAD.
