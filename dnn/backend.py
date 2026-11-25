"""Array backend: NumPy everywhere; CuPy on NVIDIA GPUs (laptop GPU, Jetson)."""

import numpy as np

BACKENDS = ("numpy", "cupy")


def get_xp(name="numpy"):
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy
        except ImportError as e:
            raise SystemExit("--backend cupy needs CuPy (e.g. pip install cupy-cuda12x): %s" % e)
        return cupy
    raise SystemExit("unknown backend %r (choose from %s)" % (name, ", ".join(BACKENDS)))


def to_numpy(a):
    """Bring an array back to host memory as a NumPy array."""
    if isinstance(a, np.ndarray):
        return a
    get = getattr(a, "get", None)
    return get() if get else np.asarray(a)
