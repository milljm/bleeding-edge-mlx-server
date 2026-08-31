"""mlx-edge — overlay git HEAD of mlx-lm / mlx-vlm onto a conda-forge MLX env."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from mlx_edge import __version__
from mlx_edge.engines import ENGINES, PYTHON_ENGINES, Engine, get_engine, resolve_targets

USER_AGENT = "mlx-edge/0.1.0"
PIN_PATH = Path.home() / ".config" / "mlx-edge" / "pins.json"


def _tty() -> bool:
    return sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _tty():
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(text: str) -> str:
    return _c("2", text)


def bold(text: str) -> str:
    return _c("1", text)


def red(text: str) -> str:
    return _c("31", text)


def green(text: str) -> str:
    return _c("32", text)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print(dim("$ " + " ".join(cmd)))
    return subprocess.run(cmd, check=check, text=True)


def conda_exe() -> str | None:
    return os.environ.get("CONDA_EXE") or shutil.which("conda")


def http_json(url: str, timeout: float = 8.0) -> dict[str, Any] | None:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def dist_info(dist_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "installed": False,
        "version": None,
        "location": None,
        "source": "missing",
        "ref": None,
        "url": None,
    }
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return out
    out["installed"] = True
    out["version"] = dist.version
    out["location"] = str(dist.locate_file(""))
    raw = dist.read_text("direct_url.json")
    if raw:
        try:
            direct = json.loads(raw)
        except json.JSONDecodeError:
            direct = {}
        vcs = direct.get("vcs_info") or {}
        out["url"] = direct.get("url")
        out["ref"] = vcs.get("commit_id") or vcs.get("requested_revision")
        if vcs:
            out["source"] = "git"
        elif (direct.get("url") or "").startswith("file:"):
            out["source"] = "local"
        else:
            out["source"] = "pypi"
    elif os.environ.get("CONDA_PREFIX") and out["location"] and os.environ["CONDA_PREFIX"] in out["location"]:
        out["source"] = "conda"
    else:
        out["source"] = "unknown"
    return out


def github_head(owner_repo: str, branch: str = "main") -> dict[str, Any] | None:
    data = http_json(f"https://api.github.com/repos/{owner_repo}/commits/{branch}")
    if not data:
        return None
    sha = data.get("sha")
    date = ((data.get("commit") or {}).get("committer") or {}).get("date")
    return {"sha": sha, "date": date, "url": data.get("html_url")}


def conda_latest(name: str) -> str | None:
    data = http_json(f"https://api.anaconda.org/package/conda-forge/{name}")
    if not data:
        return None
    return data.get("latest_version")


def pypi_latest(name: str) -> str | None:
    data = http_json(f"https://pypi.org/pypi/{name}/json")
    if not data:
        return None
    return (data.get("info") or {}).get("version")


def cmd_status(args: argparse.Namespace) -> int:
    rows = []
    for engine in ENGINES.values():
        local = dist_info(engine.dist)
        git = github_head(engine.owner_repo, engine.branch) if not args.offline else None
        row = {
            "id": engine.id,
            "dist": engine.dist,
            "compiled": engine.compiled,
            "local": local,
            "conda_forge": conda_latest(engine.conda) if not args.offline else None,
            "pypi": pypi_latest(engine.dist) if not args.offline else None,
            "git": git,
        }
        rows.append(row)

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    env = os.environ.get("CONDA_DEFAULT_ENV") or "(no conda env)"
    print(bold(f"mlx-edge {__version__}") + dim(f"  env={env}  python={platform.python_version()}"))
    print()
    header = f"{'engine':<8} {'local':<18} {'source':<8} {'conda-forge':<12} {'pypi':<12} {'git HEAD'}"
    print(dim(header))
    for row in rows:
        local = row["local"]
        ver = local["version"] or "—"
        src = local["source"]
        if local.get("ref"):
            ver = f"{ver}+{str(local['ref'])[:7]}"
        git = row["git"] or {}
        git_s = (git.get("sha") or "—")[:7]
        line = (
            f"{row['id']:<8} {ver:<18} {src:<8} "
            f"{str(row['conda_forge'] or '—'):<12} {str(row['pypi'] or '—'):<12} {git_s}"
        )
        if row["compiled"]:
            print(line)
        elif src == "git":
            print(green(line))
        elif src == "missing":
            print(red(line))
        else:
            print(line)
    print()
    print(dim("green = git overlay. compiled mlx should stay on conda-forge."))
    return 0


def pip_install_git(engine: Engine, ref: str | None, branch: str, with_deps: bool) -> int:
    spec = f"git+{engine.repo}@{ref or branch}"
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall"]
    if not with_deps:
        cmd.append("--no-deps")
    cmd.append(spec)
    proc = run(cmd, check=False)
    return proc.returncode


def cmd_update(args: argparse.Namespace) -> int:
    if args.pinned:
        pins = load_pins()
        if not pins:
            print(red("no pins file. run: mlx-edge pin"), file=sys.stderr)
            return 1
        rc = 0
        for engine_id, meta in pins.get("engines", {}).items():
            engine = get_engine(engine_id)
            ref = meta.get("ref")
            if not ref:
                continue
            print(bold(f"update {engine.dist} @ {ref[:12]}"))
            rc = pip_install_git(engine, ref, engine.branch, args.with_deps) or rc
        return rc

    targets = resolve_targets(args.engine)
    rc = 0
    for engine in targets:
        if engine.compiled and not args.force:
            print(
                red(
                    f"refusing to git-install {engine.dist} (compiled). "
                    "it belongs on conda-forge. pass --force to override."
                ),
                file=sys.stderr,
            )
            rc = 2
            continue
        print(bold(f"update {engine.dist} from {engine.owner_repo}@{args.ref or args.branch}"))
        code = pip_install_git(engine, args.ref, args.branch, args.with_deps)
        if code:
            rc = code
    if rc == 0:
        print(green("ok") + dim("  run mlx-edge status"))
    return rc


def load_pins() -> dict[str, Any] | None:
    if not PIN_PATH.is_file():
        return None
    return json.loads(PIN_PATH.read_text())


def cmd_pin(_args: argparse.Namespace) -> int:
    engines = {}
    for key in PYTHON_ENGINES:
        engine = ENGINES[key]
        info = dist_info(engine.dist)
        engines[key] = {
            "dist": engine.dist,
            "version": info["version"],
            "ref": info["ref"],
            "url": info["url"],
            "source": info["source"],
        }
    payload = {
        "pinned_at": datetime.now(timezone.utc).isoformat(),
        "engines": engines,
    }
    PIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIN_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {PIN_PATH}")
    print(json.dumps(payload, indent=2))
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    targets = resolve_targets(args.engine)
    names = [e.conda for e in targets if not e.compiled]
    exe = conda_exe()
    if not exe:
        print(dim("conda not found; falling back to pip PyPI builds"))
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", *names]
        return run(cmd, check=False).returncode
    cmd = [exe, "install", "-y", "-c", "conda-forge", "--force-reinstall", *names]
    return run(cmd, check=False).returncode


def cmd_serve(args: argparse.Namespace, rest: list[str]) -> int:
    engine = get_engine(args.engine)
    if not engine.server_module:
        print(red(f"{engine.dist} has no server"), file=sys.stderr)
        return 2
    try:
        import_module(engine.server_module.rsplit(".", 1)[0])
    except ImportError as exc:
        print(red(f"cannot import {engine.module}: {exc}"), file=sys.stderr)
        print(dim("install the env and run mlx-edge update"), file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", engine.server_module]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.host:
        cmd.extend(["--host", args.host])
    if args.port:
        cmd.extend(["--port", str(args.port)])
    cmd.extend(rest)
    print(dim("$ " + " ".join(cmd)))
    os.execvp(cmd[0], cmd)
    return 1


def cmd_which(_args: argparse.Namespace) -> int:
    for engine in ENGINES.values():
        info = dist_info(engine.dist)
        print(bold(engine.dist))
        for key in ("version", "source", "ref", "location", "url"):
            print(f"  {key:<10} {info.get(key) or '—'}")
        print()
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []
    sysname = platform.system()
    machine = platform.machine()
    apple = sysname == "Darwin" and machine in {"arm64", "aarch64"}
    checks.append(("Apple Silicon", apple, f"{sysname} {machine}"))

    env = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = os.environ.get("CONDA_PREFIX")
    checks.append(("conda env", bool(env and prefix), env or "CONDA_DEFAULT_ENV unset"))

    checks.append(("git", shutil.which("git") is not None, shutil.which("git") or "not on PATH"))
    checks.append(("pip", True, f"{sys.executable} -m pip"))

    metal_ok = False
    metal_msg = "mlx not importable"
    try:
        mlx = import_module("mlx.core")
        metal = getattr(mlx, "metal", None)
        if metal is not None and hasattr(metal, "is_available"):
            metal_ok = bool(metal.is_available())
            metal_msg = "mlx.core.metal.is_available()"
        else:
            metal_ok = sysname == "Darwin"
            metal_msg = "mlx imported (metal probe unavailable)"
    except Exception as exc:  # noqa: BLE001 — doctor must never crash
        metal_msg = str(exc)
    checks.append(("Metal", metal_ok, metal_msg))

    for engine in ENGINES.values():
        info = dist_info(engine.dist)
        checks.append((engine.dist, info["installed"], info["version"] or "missing"))

    rc = 0
    for name, ok, detail in checks:
        mark = green("ok") if ok else red("fail")
        print(f"{mark}  {name:<16} {dim(detail)}")
        if not ok and name in {"Apple Silicon", "mlx"}:
            rc = 1
    return rc


def cmd_engines(_args: argparse.Namespace) -> int:
    for engine in ENGINES.values():
        flag = "compiled" if engine.compiled else "python overlay"
        print(f"{engine.id:<6} {engine.dist:<10} {flag:<16} {engine.owner_repo}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-edge",
        description="Overlay git HEAD of mlx-lm / mlx-vlm onto a conda-forge MLX environment.",
    )
    parser.add_argument("--version", action="version", version=f"mlx-edge {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="local vs conda-forge vs PyPI vs git HEAD")
    p_status.add_argument("--json", action="store_true")
    p_status.add_argument("--offline", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_update = sub.add_parser("update", help="pip-install engines from git (no deps)")
    p_update.add_argument("engine", nargs="?", default="all", help="lm, vlm, mlx, or all")
    p_update.add_argument("--ref", help="commit SHA or tag")
    p_update.add_argument("--branch", default="main")
    p_update.add_argument("--pinned", action="store_true", help="install SHAs from mlx-edge pin")
    p_update.add_argument("--force", action="store_true", help="allow git-install of compiled mlx")
    p_update.add_argument(
        "--with-deps",
        action="store_true",
        help="let pip resolve deps (can replace conda mlx — avoid)",
    )
    p_update.set_defaults(func=cmd_update)

    p_pin = sub.add_parser("pin", help="write current git SHAs to ~/.config/mlx-edge/pins.json")
    p_pin.set_defaults(func=cmd_pin)

    p_rollback = sub.add_parser("rollback", help="reinstall conda-forge mlx-lm / mlx-vlm")
    p_rollback.add_argument("engine", nargs="?", default="all")
    p_rollback.set_defaults(func=cmd_rollback)

    p_serve = sub.add_parser("serve", help="exec mlx_lm.server or mlx_vlm.server")
    p_serve.add_argument("--engine", required=True, choices=("lm", "vlm"))
    p_serve.add_argument("--model")
    p_serve.add_argument("--host")
    p_serve.add_argument("--port", type=int)
    p_serve.set_defaults(func=None, _serve=True)

    p_which = sub.add_parser("which", help="print install paths and sources")
    p_which.set_defaults(func=cmd_which)

    p_doctor = sub.add_parser("doctor", help="check Silicon, conda, Metal, engines")
    p_doctor.set_defaults(func=cmd_doctor)

    p_engines = sub.add_parser("engines", help="list known engines")
    p_engines.set_defaults(func=cmd_engines)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # serve forwards unknown flags to the engine
    if argv and argv[0] == "serve":
        args, rest = parser.parse_known_args(argv)
        return cmd_serve(args, rest)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
