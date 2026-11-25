"""Measure the physics of the device we are running on: free memory, how fast it
can stream weights through a matrix-vector product, and what it is."""

import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

from .backend import to_numpy
from .model import Linear, STORAGE_DTYPES, quantize


def available_memory_bytes():
    """Bytes of RAM the OS says can be allocated without swapping. Works on
    Linux, Android/Termux, macOS and Windows, with or without psutil."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:  # Linux, Android
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) * 1024
        if "MemAvailable" in info:
            return info["MemAvailable"]
        return info.get("MemFree", 0) + info.get("Cached", 0) + info.get("Buffers", 0)
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = 4096
            pages = 0
            for line in out.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
                for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                    if line.startswith(key):
                        pages += int(line.split(":")[1].strip().rstrip("."))
            return pages * page
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys)
        except Exception:
            pass
    return 512 * 1024 * 1024  # unknown: assume a small device


def gpu_free_bytes(xp):
    try:
        free, _total = xp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return None


def lan_ip():
    """The address other machines on the LAN should use to reach us."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the route
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def describe():
    kind = "android" if ("ANDROID_ROOT" in os.environ or "TERMUX_VERSION" in os.environ) else sys.platform
    return {
        "hostname": socket.gethostname(),
        "os": kind,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpus": os.cpu_count() or 1,
        "pointer_bits": 64 if sys.maxsize > 2 ** 32 else 32,
    }


def measure_speed(xp, budget_bytes, seconds=0.25):
    """Weights per second this device pushes through y = x @ W.T, per storage dtype.
    Token generation reads every weight once per token, so this number (not peak
    FLOPS) predicts how long a layer takes here."""
    # 16M weights (64 MB as float32): bigger than CPU caches, so we measure RAM
    # bandwidth like a real layer would see, not cache bandwidth.
    cols = 1024
    rows = 16384
    while rows > 64 and rows * cols * 4 * 3 > budget_bytes:
        rows //= 2
    rng = np.random.default_rng(0)
    w = rng.standard_normal((rows, cols), dtype=np.float32)
    x = xp.asarray(rng.standard_normal((1, cols), dtype=np.float32))
    speeds = {}
    for dtype in STORAGE_DTYPES:
        q = quantize(w, dtype)
        lin = Linear(xp, q[""], q.get(".s"))
        to_numpy(lin(x))  # warm up
        n, t0 = 0, time.perf_counter()
        while True:
            y = lin(x)
            n += 1
            elapsed = time.perf_counter() - t0
            if elapsed > seconds and n >= 3:
                break
        to_numpy(y)  # wait for GPU work to finish
        elapsed = time.perf_counter() - t0
        speeds[dtype] = rows * cols * n / elapsed
        del lin, q
    return speeds


def measure_layer_overhead(xp, seconds=0.2):
    """Seconds of fixed cost per decoder layer per token (norms, RoPE, attention
    bookkeeping, interpreter dispatch), measured on a layer too small for its
    weights to matter. On slow CPUs this, not memory bandwidth, can dominate."""
    from .model import DecoderLayer, KVCache, ModelConfig, rope_inv_freq, rope_tables
    cfg = ModelConfig(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                           num_attention_heads=4, num_key_value_heads=2, vocab_size=8))
    rng = np.random.default_rng(0)
    r = lambda *s: rng.standard_normal(s, dtype=np.float32) * 0.05
    tensors = {"in_norm": np.ones(64, np.float32), "post_norm": np.ones(64, np.float32),
               "q": r(64, 64), "q_b": r(64), "k": r(32, 64), "k_b": r(32), "v": r(32, 64),
               "v_b": r(32), "o": r(64, 64), "gate": r(128, 64), "up": r(128, 64), "down": r(64, 128)}
    layer = DecoderLayer(cfg, xp, tensors)
    inv = xp.asarray(rope_inv_freq(cfg))
    cache = KVCache(xp, cfg.n_kv, cfg.head_dim)
    x = xp.asarray(r(1, 64))
    pos = 0
    for _ in range(64):  # warm up and give the cache a realistic length
        cos, sin = rope_tables(xp, inv, pos, 1)
        layer.forward(x, cos, sin, cache, pos)
        pos += 1
    n, t0 = 0, time.perf_counter()
    while True:
        cos, sin = rope_tables(xp, inv, pos, 1)
        y = layer.forward(x, cos, sin, cache, pos - 1)  # rewind one step: cache stays the same size
        n += 1
        if time.perf_counter() - t0 > seconds and n >= 10:
            break
    to_numpy(y)
    return (time.perf_counter() - t0) / n


def probe(xp, memory_limit_bytes=None, backend="numpy"):
    mem = available_memory_bytes()
    if backend == "cupy":
        g = gpu_free_bytes(xp)
        if g is not None:
            mem = min(mem, g) if "tegra" in platform.release().lower() else g
    if memory_limit_bytes:
        mem = min(mem, memory_limit_bytes)
    info = describe()
    info.update({
        "backend": backend,
        "mem_available": int(mem),
        "params_per_s": measure_speed(xp, mem),
        "layer_overhead_s": measure_layer_overhead(xp),
    })
    return info


def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return ("%.0f %s" if unit in ("B", "KB") else "%.1f %s") % (n, unit)
        n /= 1024
