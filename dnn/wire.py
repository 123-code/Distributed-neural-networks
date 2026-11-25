"""Wire protocol: length-prefixed frames carrying a JSON header plus raw tensors.

Frame layout (all integers big-endian):

    4 bytes   magic  b"DNN1"
    4 bytes   header length H (uint32)
    8 bytes   payload length P (uint64)
    H bytes   UTF-8 JSON header
    P bytes   payload: tensors packed back to back, little-endian

The header's "tensors" list describes each tensor:
    {"name": str, "dtype": "float32"|"float16"|"int8"|..., "shape": [..],
     "offset": byte offset into payload, "nbytes": int}

Nothing here unpickles or executes data from the network: only JSON and raw
numbers cross the wire. See PROTOCOL.md for the message types.
"""

import json
import socket
import struct
import sys

import numpy as np

MAGIC = b"DNN1"
PROTOCOL_VERSION = 1
_FRAME = struct.Struct(">4sIQ")
_MAX_HEADER = 16 * 1024 * 1024
_DTYPES = ("float32", "float16", "int8", "uint8", "int32", "int64", "uint16")
_BIG_ENDIAN_HOST = sys.byteorder == "big"


class ProtocolError(Exception):
    pass


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("peer closed the connection")
        got += r
    return buf


def send_msg(sock, header, tensors=None):
    """Send one frame. `tensors` maps name -> numpy array."""
    metas, arrays, offset = [], [], 0
    for name, arr in (tensors or {}).items():
        a = np.ascontiguousarray(arr)
        if a.dtype.name not in _DTYPES:
            raise ProtocolError("unsupported dtype %s for tensor %s" % (a.dtype, name))
        if _BIG_ENDIAN_HOST and a.dtype.itemsize > 1:
            a = a.astype(a.dtype.newbyteorder("<"))
        metas.append({"name": name, "dtype": a.dtype.name, "shape": list(a.shape),
                      "offset": offset, "nbytes": int(a.nbytes)})
        arrays.append(a)
        offset += a.nbytes
    h = dict(header)
    h["tensors"] = metas
    hb = json.dumps(h, separators=(",", ":")).encode("utf-8")
    sock.sendall(_FRAME.pack(MAGIC, len(hb), offset) + hb)
    for a in arrays:
        if a.nbytes:
            sock.sendall(memoryview(a.reshape(-1)).cast("B"))


def recv_msg(sock):
    """Receive one frame. Returns (header dict, {name: numpy array})."""
    magic, hlen, plen = _FRAME.unpack(bytes(_recv_exact(sock, _FRAME.size)))
    if magic != MAGIC:
        raise ProtocolError("bad magic %r (not a DNN peer?)" % magic)
    if hlen > _MAX_HEADER:
        raise ProtocolError("header too large: %d bytes" % hlen)
    header = json.loads(bytes(_recv_exact(sock, hlen)).decode("utf-8"))
    payload = _recv_exact(sock, plen) if plen else bytearray()
    tensors = {}
    for m in header.pop("tensors", []):
        if m["dtype"] not in _DTYPES:
            raise ProtocolError("unsupported dtype %s" % m["dtype"])
        dt = np.dtype(m["dtype"]).newbyteorder("<")
        shape = tuple(m["shape"])
        count = int(np.prod(shape)) if shape else 1
        if count == 0:
            arr = np.zeros(shape, dtype=np.dtype(m["dtype"]))
        else:
            arr = np.frombuffer(payload, dtype=dt, count=count, offset=m["offset"]).reshape(shape)
            if _BIG_ENDIAN_HOST and dt.itemsize > 1:
                arr = arr.astype(dt.newbyteorder("="))
        tensors[m["name"]] = arr
    return header, tensors


def parse_addr(text, default_port=7070):
    """'host:port' or 'host' -> (host, port)."""
    text = text.strip()
    if text.startswith("["):  # [ipv6]:port
        host, _, rest = text[1:].partition("]")
        port = int(rest[1:]) if rest.startswith(":") else default_port
        return host, port
    if text.count(":") == 1:
        host, port = text.split(":")
        return host, int(port)
    return text, default_port


def connect(addr, timeout=10.0):
    """Open a TCP connection with Nagle's algorithm off (latency matters more than
    packet count: each token's activation is only a few KB)."""
    sock = socket.create_connection(tuple(addr), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)
    return sock
