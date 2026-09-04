"""Host memory and GPU utilization for the Edge sidebar."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

GB = 1024**3
CACHE_TTL = 0.85

_PAGE = re.compile(r"page size of (\d+) bytes", re.I)
_VM_ROW = re.compile(r"^Pages ([A-Za-z ]+):\s+(\d+)\.?$")
_DEVICE_UTIL = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_RENDER_UTIL = re.compile(r'"Renderer Utilization %"\s*=\s*(\d+)')

_cache: dict[str, Any] | None = None
_cache_at = 0.0


def snapshot() -> dict[str, Any]:
    global _cache, _cache_at
    now = time.time()
    if _cache is not None and now - _cache_at < CACHE_TTL:
        return _cache
    mem = memory_stats()
    gpu = gpu_stats()
    snap = {
        "object": "edge.host",
        "generated_at": now,
        "memory": mem,
        "gpu": gpu,
    }
    _cache = snap
    _cache_at = now
    return snap


def memory_stats() -> dict[str, Any]:
    if sys.platform == "darwin":
        return _darwin_memory()
    return _linux_memory()


def gpu_stats() -> dict[str, Any] | None:
    if sys.platform == "darwin":
        return _darwin_gpu()
    return _linux_gpu()


def parse_vm_stat(text: str, total_bytes: int) -> dict[str, Any]:
    page = 16384
    hit = _PAGE.search(text)
    if hit:
        page = int(hit.group(1))
    counts: dict[str, int] = {}
    for line in text.splitlines():
        row = line.strip().rstrip(".")
        match = _VM_ROW.match(row)
        if not match:
            continue
        counts[match.group(1).strip().lower()] = int(match.group(2))
    used_pages = (
        counts.get("active", 0)
        + counts.get("wired down", 0)
        + counts.get("occupied by compressor", 0)
    )
    used = used_pages * page
    if total_bytes <= 0:
        total_bytes = used
    used = max(0, min(used, total_bytes))
    return _memory_payload(used, total_bytes)


def parse_meminfo(text: str) -> dict[str, Any]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            kb = int(parts[0])
        except ValueError:
            continue
        values[key] = kb * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    used = total - available if total and available <= total else values.get("MemTotal", 0) - values.get("MemFree", 0)
    used = max(0, min(used, total))
    return _memory_payload(used, total)


def parse_ioreg_gpu(text: str) -> dict[str, Any] | None:
    hits = [int(n) for n in _DEVICE_UTIL.findall(text)]
    if not hits:
        hits = [int(n) for n in _RENDER_UTIL.findall(text)]
    if not hits:
        return None
    percent = max(0, min(100, max(hits)))
    return {"percent": percent, "source": "ioreg"}


def _memory_payload(used: int, total: int) -> dict[str, Any]:
    ratio = (used / total) if total else 0.0
    return {
        "used_bytes": int(used),
        "total_bytes": int(total),
        "ratio": round(ratio, 4),
    }


def _darwin_memory() -> dict[str, Any]:
    total = 0
    raw = _run(["sysctl", "-n", "hw.memsize"])
    if raw:
        try:
            total = int(raw.strip())
        except ValueError:
            total = 0
    vm = _run(["vm_stat"]) or ""
    if vm:
        return parse_vm_stat(vm, total)
    return _memory_payload(0, total)


def _linux_memory() -> dict[str, Any]:
    path = Path("/proc/meminfo")
    if path.is_file():
        try:
            return parse_meminfo(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    phys = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    total = max(0, page * phys)
    return _memory_payload(0, total)


def _darwin_gpu() -> dict[str, Any] | None:
    for klass in ("AGXAccelerator", "IOAccelerator"):
        raw = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", klass])
        if not raw:
            continue
        parsed = parse_ioreg_gpu(raw)
        if parsed:
            return parsed
    return None


def _linux_gpu() -> dict[str, Any] | None:
    root = Path("/sys/class/drm")
    if not root.is_dir():
        return None
    best: int | None = None
    for path in root.glob("card*/device/gpu_busy_percent"):
        try:
            n = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if best is None or n > best:
            best = n
    if best is None:
        return None
    return {"percent": max(0, min(100, best)), "source": "sysfs"}


def _run(cmd: list[str], timeout: float = 0.7) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or ""
