from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import ctypes
import functools
from pathlib import Path
import subprocess
import sys
import threading

from jasna import os_utils

_DRM_CLASS_PATH = Path("/sys/class/drm")


@dataclass(frozen=True)
class SystemStats:
    gpu_util: int | None
    vram_util: int | None
    ram_util: int
    cpu_util: int


def _clamp_pct(value: float) -> int:
    v = int(round(float(value)))
    if v < 0:
        return 0
    if v > 100:
        return 100
    return v


def _parse_nvidia_smi_csv_line(line: str) -> tuple[int, int]:
    parts = [p.strip() for p in (line or "").split(",")]
    if len(parts) < 3:
        raise ValueError(f"Unexpected nvidia-smi output: {line!r}")
    gpu_util = _clamp_pct(float(parts[0]))
    mem_used = float(parts[1])
    mem_total = float(parts[2])
    if mem_total <= 0:
        raise ValueError(f"Unexpected nvidia-smi total memory: {mem_total!r}")
    vram_util = _clamp_pct((mem_used / mem_total) * 100.0)
    return gpu_util, vram_util


def _read_amd_sysfs() -> tuple[int | None, int | None]:
    for card in sorted(_DRM_CLASS_PATH.glob("card[0-9]*")):
        device = card / "device"
        try:
            if (device / "vendor").read_text(encoding="utf-8").strip().lower() != "0x1002":
                continue
            gpu_path = device / "gpu_busy_percent"
            gpu_util = (
                _clamp_pct(float(gpu_path.read_text(encoding="utf-8").strip()))
                if gpu_path.is_file()
                else None
            )
            used = int((device / "mem_info_vram_used").read_text(encoding="utf-8"))
            total = int((device / "mem_info_vram_total").read_text(encoding="utf-8"))
            vram_util = _clamp_pct((used / total) * 100.0) if total > 0 else None
            return gpu_util, vram_util
        except (OSError, ValueError):
            continue
    return None, None


@functools.cache
def _apple_iokit_handles():
    """Load IOKit + CoreFoundation symbols and interned CFString keys once per process."""
    iokit = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/IOKit.framework/IOKit")
    cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cf.CFDictionaryGetValue.restype = ctypes.c_void_p
    cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFRelease.restype = None

    iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    iokit.IOServiceMatching.restype = ctypes.c_void_p
    iokit.IOServiceGetMatchingServices.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
    iokit.IOIteratorNext.argtypes = [ctypes.c_uint32]
    iokit.IOIteratorNext.restype = ctypes.c_uint32
    iokit.IORegistryEntryCreateCFProperty.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
    iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
    iokit.IOObjectRelease.restype = ctypes.c_int

    utf8 = 0x08000100  # kCFStringEncodingUTF8
    keys = {
        name: cf.CFStringCreateWithCString(None, name.encode("utf-8"), utf8)
        for name in (
            "PerformanceStatistics",
            "Device Utilization %",
            "Renderer Utilization %",
            "In use system memory",
        )
    }
    return iokit, cf, keys


def _read_apple_iokit_stats() -> tuple[int | None, int | None]:
    """Read live Apple Silicon GPU utilization (%) and active Metal memory (bytes) via IOKit.

    Uses ``IOAccelerator`` -> ``PerformanceStatistics`` in-process (~0.2 ms, no subprocess,
    no root privileges required).
    """
    try:
        iokit, cf, keys = _apple_iokit_handles()
    except OSError:
        return None, None

    matching = iokit.IOServiceMatching(b"IOAccelerator")
    if not matching:
        return None, None
    iterator = ctypes.c_uint32(0)
    # IOServiceGetMatchingServices consumes one reference to ``matching``.
    if iokit.IOServiceGetMatchingServices(0, matching, ctypes.byref(iterator)) != 0:
        return None, None

    k_cf_number_sint64 = 4
    try:
        while True:
            service = iokit.IOIteratorNext(iterator)
            if not service:
                break
            try:
                props = iokit.IORegistryEntryCreateCFProperty(service, keys["PerformanceStatistics"], None, 0)
                if not props:
                    continue
                try:
                    def _dict_int(name: str) -> int | None:
                        val = cf.CFDictionaryGetValue(props, keys[name])
                        if not val:
                            return None
                        out = ctypes.c_int64(0)
                        return int(out.value) if cf.CFNumberGetValue(val, k_cf_number_sint64, ctypes.byref(out)) else None

                    gpu_util = _dict_int("Device Utilization %")
                    if gpu_util is None:
                        gpu_util = _dict_int("Renderer Utilization %")
                    in_use_bytes = _dict_int("In use system memory")
                    if gpu_util is not None or in_use_bytes is not None:
                        return (
                            _clamp_pct(gpu_util) if gpu_util is not None else None,
                            in_use_bytes if in_use_bytes is not None and in_use_bytes >= 0 else None,
                        )
                finally:
                    cf.CFRelease(props)
            finally:
                iokit.IOObjectRelease(service)
    finally:
        iokit.IOObjectRelease(iterator)
    return None, None


def _apple_gpu_total_memory() -> int | None:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and hasattr(torch_mod, "mps"):
        try:
            if torch_mod.backends.mps.is_available():
                total = int(torch_mod.mps.recommended_max_memory())
                if total > 0:
                    return total
        except Exception:
            pass
    try:
        import psutil
        mem_total = int(psutil.virtual_memory().total)
        if mem_total > 0:
            # Metal recommendedMaxWorkingSetSize defaults to ~75% of Unified Memory (>=32 GiB)
            # or ~67% on smaller Macs when torch is not yet imported.
            return (mem_total * 3 // 4) if mem_total >= (32 << 30) else (mem_total * 2 // 3)
    except Exception:
        pass
    return None


def _read_apple_mps() -> tuple[int | None, int | None]:
    gpu_util, io_in_use = _read_apple_iokit_stats()
    total = _apple_gpu_total_memory()

    torch_used: int | None = None
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and hasattr(torch_mod, "mps"):
        try:
            if torch_mod.backends.mps.is_available():
                torch_used = max(0, int(torch_mod.mps.current_allocated_memory()))
        except Exception:
            torch_used = None

    vram_util: int | None = None
    if total is not None and total > 0:
        if io_in_use is not None and torch_used is not None:
            vram_util = _clamp_pct((max(io_in_use, torch_used) / total) * 100.0)
        elif io_in_use is not None:
            vram_util = _clamp_pct((io_in_use / total) * 100.0)
        elif torch_used is not None and torch_used > 0:
            vram_util = _clamp_pct((torch_used / total) * 100.0)
    return gpu_util, vram_util


def read_gpu_vram() -> tuple[int | None, int | None]:
    exe_path = os_utils.find_executable("nvidia-smi")
    if exe_path is None:
        gpu_util, vram_util = _read_amd_sysfs()
        if gpu_util is not None or vram_util is not None:
            return gpu_util, vram_util
        if sys.platform == "darwin":
            return _read_apple_mps()
        return None, None

    cmd = [
        exe_path,
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=0.5,
            **os_utils.subprocess_no_window_kwargs(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None, None

    if completed.returncode != 0:
        return None, None

    lines = (completed.stdout or "").splitlines()
    first = next((ln for ln in lines if ln.strip()), "")
    if first == "":
        return None, None

    try:
        return _parse_nvidia_smi_csv_line(first)
    except ValueError:
        return None, None


def read_cpu_ram() -> tuple[int, int]:
    import psutil
    cpu_util = _clamp_pct(psutil.cpu_percent(interval=None))
    ram_util = _clamp_pct(psutil.virtual_memory().percent)
    return cpu_util, ram_util


def read_system_stats() -> SystemStats:
    cpu_util, ram_util = read_cpu_ram()
    gpu_util, vram_util = read_gpu_vram()
    return SystemStats(
        gpu_util=gpu_util,
        vram_util=vram_util,
        ram_util=ram_util,
        cpu_util=cpu_util,
    )


def start_poller(stop: threading.Event, on_stats: Callable[[SystemStats], None]) -> threading.Thread:
    """Call ``on_stats(read_system_stats())`` on a daemon thread every 1.5 s until ``stop`` is set."""

    def loop():
        while not stop.is_set():
            on_stats(read_system_stats())
            stop.wait(1.5)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread
