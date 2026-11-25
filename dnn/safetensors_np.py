"""Minimal .safetensors reader/writer using only NumPy (memory-mapped, so the
coordinator reads one tensor at a time instead of loading the whole checkpoint)."""

import json
import os
import struct

import numpy as np

_NP = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
       "I32": np.int32, "I64": np.int64}
_ST = {np.dtype(v).name: k for k, v in _NP.items()}


class SafeTensors:
    """A checkpoint directory: model.safetensors, or shards listed in
    model.safetensors.index.json."""

    def __init__(self, model_dir):
        self.dir = str(model_dir)
        self._where = {}   # tensor name -> file name
        self._files = {}   # file name -> (memmap, header, data start)
        index = os.path.join(self.dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self._where = dict(json.load(f)["weight_map"])
        else:
            files = sorted(n for n in os.listdir(self.dir) if n.endswith(".safetensors"))
            if not files:
                raise FileNotFoundError("no .safetensors files in %s" % self.dir)
            for fn in files:
                for name in self._open(fn)[1]:
                    if name != "__metadata__":
                        self._where[name] = fn

    def _open(self, fn):
        if fn not in self._files:
            path = os.path.join(self.dir, fn)
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            self._files[fn] = (np.memmap(path, dtype=np.uint8, mode="r"), header, 8 + n)
        return self._files[fn]

    def keys(self):
        return list(self._where)

    def has(self, name):
        return name in self._where

    def get(self, name):
        """Return the tensor as a float32 (or integer) NumPy array in RAM."""
        mm, header, start = self._open(self._where[name])
        meta = header[name]
        b, e = meta["data_offsets"]
        raw = np.array(mm[start + b:start + e])  # copy out of the map
        dt = meta["dtype"]
        if dt == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt in _NP:
            arr = raw.view(_NP[dt])
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
        else:
            raise TypeError("tensor %s has unsupported dtype %s" % (name, dt))
        return arr.reshape(meta["shape"])


def save(path, tensors, metadata=None):
    """Write {name: array} as a single .safetensors file."""
    header, offset, arrays = {}, 0, []
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr)
        a = a.astype(a.dtype.newbyteorder("<"))
        header[name] = {"dtype": _ST[a.dtype.name], "shape": list(a.shape),
                        "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
        arrays.append(a)
    if metadata:
        header["__metadata__"] = metadata
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for a in arrays:
            f.write(a.tobytes())
