"""mlx-edge — overlay git HEAD of mlx-lm / mlx-vlm onto a conda-forge MLX env."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
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

USER_AGENT = f"mlx-edge/{__version__}"
PIN_PATH = Path.home() / ".config" / "mlx-edge" / "pins.json"
GATEWAY_PATH = Path.home() / ".config" / "mlx-edge" / "gateway.json"


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
    return pip_install_spec(spec, with_deps)


def pip_install_spec(spec: str, with_deps: bool) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall"]
    if not with_deps:
        cmd.append("--no-deps")
    cmd.append(spec)
    proc = run(cmd, check=False)
    return proc.returncode


BUILD_EPILOG = """search for a possible engine update at:
  mlx-lm     https://github.com/ml-explore/mlx-lm/pulls
  mlx-vlm    https://github.com/Blaizzy/mlx-vlm/pulls
  mlx-audio  https://github.com/Blaizzy/mlx-audio/pulls

examples:
  mlx-edge build git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1398/head
  mlx-edge build 1398
  mlx-edge build mlx-vlm#42
  mlx-edge build mlx-audio#12
"""

_PR_SPEC = re.compile(
    r"^(?:(?P<name>mlx-lm|mlx-vlm|mlx-audio|mlx|lm|vlm|audio)#)?(?P<pr>\d+)$",
    re.I,
)


def parse_build_spec(spec: str, engine_id: str | None = None) -> tuple[Engine, str]:
    """Turn a git+ URL, GitHub URL, or PR number into (engine, pip spec)."""
    raw = spec.strip()
    if not raw:
        raise ValueError("empty build spec")
    if raw.startswith("git+") or "github.com" in raw or raw.startswith("https://"):
        url = raw if raw.startswith("git+") else f"git+{raw.lstrip('/')}"
        if url.startswith("git+github.com"):
            url = "git+https://" + url[len("git+") :]
        engine = _engine_from_url(url, engine_id)
        return engine, url
    match = _PR_SPEC.fullmatch(raw)
    if match:
        name = (match.group("name") or engine_id or "lm").lower()
        aliases = {"mlx-lm": "lm", "mlx-vlm": "vlm", "mlx-audio": "audio"}
        engine = get_engine(aliases.get(name, name))
        url = f"git+{engine.repo}@refs/pull/{match.group('pr')}/head"
        return engine, url
    raise ValueError(
        f"unrecognized build spec {spec!r}. pass a git+ URL or PR number (mlx-edge build --help)"
    )


def _engine_from_url(url: str, engine_id: str | None) -> Engine:
    if engine_id:
        return get_engine(engine_id)
    lower = url.lower()
    if "mlx-audio" in lower:
        return get_engine("audio")
    if "mlx-vlm" in lower:
        return get_engine("vlm")
    if "mlx-lm" in lower or "mlx_lm" in lower:
        return get_engine("lm")
    if re.search(r"github\.com/[^/]+/mlx(?:\.git|@|/|$)", lower):
        return get_engine("mlx")
    return get_engine("lm")


def cmd_build(args: argparse.Namespace) -> int:
    spec = (getattr(args, "spec", None) or "").strip()
    if not spec:
        sys.stdout.write(BUILD_EPILOG)
        return 0
    try:
        engine, pip_spec = parse_build_spec(spec, getattr(args, "engine", None))
    except ValueError as exc:
        print(red(str(exc)), file=sys.stderr)
        sys.stderr.write(BUILD_EPILOG)
        return 2
    if engine.compiled and not args.force:
        print(
            red(
                f"refusing to git-install {engine.dist} (compiled). "
                "it belongs on conda-forge. pass --force to override."
            ),
            file=sys.stderr,
        )
        return 2
    print(bold(f"build {engine.dist} from {pip_spec}"))
    code = pip_install_spec(pip_spec, args.with_deps)
    if code == 0:
        print(green("ok") + dim("  run mlx-edge status"))
    return code


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
    import threading
    import webbrowser

    from mlx_edge.gateway import bundled_web_dir, public_base, serve_forever
    from mlx_edge.pool import ModelPool

    host = args.host or "127.0.0.1"
    port = int(args.port or 8080)
    gui = bool(getattr(args, "gui", False))
    static_dir = bundled_web_dir() if gui else None
    if gui and static_dir is None:
        print(red("GUI assets missing. Rebuild gui/ (npm run build:gui) and reinstall."), file=sys.stderr)
        return 1

    preloads: list[tuple[str, str]] = []
    for model in args.lm or []:
        preloads.append(("lm", model))
    for model in args.vlm or []:
        preloads.append(("vlm", model))
    for model in getattr(args, "embed", None) or []:
        preloads.append(("embed", model))
    for model in getattr(args, "tts", None) or []:
        preloads.append(("tts", model))
    for model in getattr(args, "stt", None) or []:
        preloads.append(("stt", model))
    for model in getattr(args, "rerank", None) or []:
        preloads.append(("rerank", model))
    for model in getattr(args, "image", None) or []:
        preloads.append(("image", model))
    if args.engine and args.model:
        for model in args.model:
            preloads.append((args.engine, model))
    elif args.model and not args.engine:
        print(red("--model requires --engine lm|vlm|embed|tts|stt|rerank|image (or pass --lm / --vlm / --embed / --tts / --stt / --rerank / --image)"), file=sys.stderr)
        return 2

    pool = ModelPool()
    for engine_id, model in preloads:
        print(bold(f"load {engine_id} {model}"))
        try:
            pool.load(engine_id, model, rest)
        except Exception as exc:  # noqa: BLE001
            from mlx_edge.pool import annotate_load_error

            print(red(annotate_load_error(str(exc))), file=sys.stderr)
            pool.unload_all()
            return 1

    info = public_base(host, port)
    url = str(info["url"])
    GATEWAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATEWAY_PATH.write_text(json.dumps({"host": host, "port": port}) + "\n")
    print(bold(f"Serving on {url}"))
    print(dim(f"  bind {info['bind']}"))
    if gui:
        page = url[: -len("/v1")] + "/"
        print(dim(f"  GUI  {page}"))
    if preloads:
        for item in pool.list():
            print(dim(f"  {item.engine:<4} {item.model}"))
    else:
        print(dim("  empty pool. Serve from the GUI, or mlx-edge load --engine lm --model …"))
    print(dim("  GET /v1/models  POST /v1/chat/completions  POST /v1/embeddings  GET /v1/progress  POST /v1/load"))
    if gui and not getattr(args, "no_browser", False):
        page = url[: -len("/v1")] + "/"
        threading.Timer(0.6, lambda: webbrowser.open(page)).start()
    try:
        serve_forever(pool, host, port, static_dir=static_dir)
    except KeyboardInterrupt:
        print()
        print(dim("stopping"))
        pool.unload_all()
    finally:
        if GATEWAY_PATH.is_file():
            GATEWAY_PATH.unlink(missing_ok=True)
    return 0


def _gateway_url(args: argparse.Namespace) -> str:
    host = args.host
    port = args.port
    if host is None or port is None:
        if GATEWAY_PATH.is_file():
            try:
                data = json.loads(GATEWAY_PATH.read_text())
            except json.JSONDecodeError:
                data = {}
            host = host or data.get("host") or "127.0.0.1"
            port = port or data.get("port") or 8080
        else:
            host = host or "127.0.0.1"
            port = port or 8080
    return f"http://{host}:{int(port)}"


def _post_gateway(url: str, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
            return resp.status, body if isinstance(body, dict) else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            body = json.loads(raw or "{}")
        except json.JSONDecodeError:
            body = {"error": {"message": raw or str(exc)}}
        return exc.code, body if isinstance(body, dict) else {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(red(f"gateway not reachable at {url}. start mlx-edge serve first ({exc})")) from exc


def cmd_load(args: argparse.Namespace, rest: list[str]) -> int:
    url = _gateway_url(args)
    status, body = _post_gateway(
        url,
        "/v1/load",
        {"engine": args.engine, "model": args.model, "args": rest},
    )
    if status >= 400:
        err = (body.get("error") or {}).get("message") or json.dumps(body)
        print(red(str(err)), file=sys.stderr)
        return 1
    print(green("loaded") + " " + str(args.model))
    models = body.get("models") or []
    if models:
        print(dim("pool: " + ", ".join(str(m) for m in models)))
    return 0


def cmd_unload(args: argparse.Namespace) -> int:
    url = _gateway_url(args)
    status, body = _post_gateway(url, "/v1/unload", {"model": args.model})
    if status >= 400:
        err = (body.get("error") or {}).get("message") or json.dumps(body)
        print(red(str(err)), file=sys.stderr)
        return 1
    print(green("unloaded") + " " + str(args.model))
    models = body.get("models") or []
    print(dim("pool: " + (", ".join(str(m) for m in models) if models else "(empty)")))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    url = _gateway_url(args)
    req = urllib.request.Request(url + "/v1/models", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(red(f"gateway not reachable at {url} ({exc})"), file=sys.stderr)
        return 1
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    data = payload.get("data") or []
    if not data:
        print(dim("no models loaded"))
        return 0
    for row in data:
        print(f"{row.get('owned_by', 'mlx'):<10} {row.get('id')}")
    return 0


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
        description="Hot-load mlx-lm / mlx-vlm / embeddings onto one OpenAI /v1 gateway.",
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

    p_build = sub.add_parser(
        "build",
        help="overlay an mlx-lm / mlx-vlm git URL or GitHub pull request",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Install a not-yet-merged engine so new model classes load.",
        epilog=BUILD_EPILOG,
    )
    p_build.add_argument(
        "spec",
        nargs="?",
        help="git+ URL, PR number (mlx-lm), or engine#PR (mlx-vlm#42)",
    )
    p_build.add_argument("--engine", choices=("lm", "vlm", "audio", "mlx"), help="when spec is a bare PR number")
    p_build.add_argument("--force", action="store_true", help="allow git-install of compiled mlx")
    p_build.add_argument(
        "--with-deps",
        action="store_true",
        help="let pip resolve deps (can replace conda mlx — avoid)",
    )
    p_build.set_defaults(func=cmd_build)

    p_pin = sub.add_parser("pin", help="write current git SHAs to ~/.config/mlx-edge/pins.json")
    p_pin.set_defaults(func=cmd_pin)

    p_rollback = sub.add_parser("rollback", help="reinstall conda-forge mlx-lm / mlx-vlm")
    p_rollback.add_argument("engine", nargs="?", default="all")
    p_rollback.set_defaults(func=cmd_rollback)

    p_serve = sub.add_parser("serve", help="OpenAI /v1 gateway; hot-load models beside each other")
    p_serve.add_argument("--engine", choices=("lm", "vlm", "embed", "tts", "stt", "rerank", "image"), help="used with --model (repeatable)")
    p_serve.add_argument("--model", action="append", default=[], help="preload with --engine (repeatable)")
    p_serve.add_argument("--lm", action="append", default=[], metavar="MODEL", help="preload an mlx-lm model (repeatable)")
    p_serve.add_argument("--vlm", action="append", default=[], metavar="MODEL", help="preload an mlx-vlm model (repeatable)")
    p_serve.add_argument("--embed", action="append", default=[], metavar="MODEL", help="preload an embedding model (repeatable)")
    p_serve.add_argument("--tts", action="append", default=[], metavar="MODEL", help="preload a TTS model (repeatable)")
    p_serve.add_argument("--stt", action="append", default=[], metavar="MODEL", help="preload an STT model (repeatable)")
    p_serve.add_argument("--rerank", action="append", default=[], metavar="MODEL", help="preload a reranker (repeatable)")
    p_serve.add_argument("--image", action="append", default=[], metavar="MODEL", help="preload an image-generation model (repeatable)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--gui", action="store_true", help="serve the Edge GUI on the same host/port")
    p_serve.add_argument("--no-browser", action="store_true", help="do not open a browser (with --gui)")
    p_serve.set_defaults(func=None, _serve=True)

    p_load = sub.add_parser("load", help="hot-load a model onto a running gateway")
    p_load.add_argument("--engine", required=True, choices=("lm", "vlm", "embed", "tts", "stt", "rerank", "image"))
    p_load.add_argument("--model", required=True)
    p_load.add_argument("--host")
    p_load.add_argument("--port", type=int)
    p_load.set_defaults(func=None, _load=True)

    p_unload = sub.add_parser("unload", help="unload one model; others stay up")
    p_unload.add_argument("--model", required=True)
    p_unload.add_argument("--host")
    p_unload.add_argument("--port", type=int)
    p_unload.set_defaults(func=cmd_unload)

    p_models = sub.add_parser("models", help="list models on the running gateway")
    p_models.add_argument("--json", action="store_true")
    p_models.add_argument("--host")
    p_models.add_argument("--port", type=int)
    p_models.set_defaults(func=cmd_models)

    p_which = sub.add_parser("which", help="print install paths and sources")
    p_which.set_defaults(func=cmd_which)

    p_doctor = sub.add_parser("doctor", help="check Silicon, conda, Metal, engines")
    p_doctor.set_defaults(func=cmd_doctor)

    p_engines = sub.add_parser("engines", help="list known engines")
    p_engines.set_defaults(func=cmd_engines)

    return parser


def gui_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="edge-gui",
        description="Start the Edge GUI. Controls mlx-edge on the same host/port.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Use 0.0.0.0 for remote clients.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--lm", action="append", default=[], metavar="MODEL", help="preload an mlx-lm model")
    parser.add_argument("--vlm", action="append", default=[], metavar="MODEL", help="preload an mlx-vlm model")
    parser.add_argument("--embed", action="append", default=[], metavar="MODEL", help="preload an embedding model")
    parser.add_argument("--tts", action="append", default=[], metavar="MODEL", help="preload a TTS model")
    parser.add_argument("--stt", action="append", default=[], metavar="MODEL", help="preload an STT model")
    parser.add_argument("--rerank", action="append", default=[], metavar="MODEL", help="preload a reranker")
    parser.add_argument("--image", action="append", default=[], metavar="MODEL", help="preload an image-generation model")
    parser.add_argument("--no-browser", action="store_true")
    args, rest = parser.parse_known_args(argv)
    args.engine = None
    args.model = []
    args.gui = True
    return cmd_serve(args, rest)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if argv and argv[0] == "serve":
        args, rest = parser.parse_known_args(argv)
        return cmd_serve(args, rest)
    if argv and argv[0] == "load":
        args, rest = parser.parse_known_args(argv)
        return cmd_load(args, rest)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
