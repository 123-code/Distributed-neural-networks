"""A pipeline node. Run this on every device that should hold layers:

    python -m dnn.node                 # listen on 0.0.0.0:7070
    python -m dnn.node --port 7071 --memory-limit 600 --name firetv
    python -m dnn.node --probe         # just print what this device can do

Needs only Python 3 and NumPy. It downloads nothing: the coordinator streams it
the layers it was assigned. It never executes code received from the network.
"""

import argparse
import gc
import socket
import socketserver
import threading
import time
import traceback

from . import __version__
from .backend import BACKENDS, get_xp, to_numpy
from .model import ModelConfig, Stage
from .sysinfo import fmt_bytes, lan_ip, probe
from .wire import PROTOCOL_VERSION, connect, recv_msg, send_msg


class Node:
    def __init__(self, name, backend="numpy", memory_limit=None):
        self.name = name
        self.backend = backend
        self.xp = get_xp(backend)
        self.memory_limit = memory_limit
        self.cfg = None
        self.stage = None
        self.layer_range = None
        self.dtype = None
        self._probe = None
        self._compute = threading.Lock()
        self._peers = {}  # (host, port) -> [socket, lock]
        self._peers_lock = threading.Lock()

    # ---- control plane (request -> reply) ---------------------------------
    def control(self, h, t):
        kind = h.get("type")
        if kind == "hello":
            return {"type": "hello", "name": self.name, "protocol": PROTOCOL_VERSION,
                    "version": __version__}, None
        if kind == "ping":
            return {"type": "pong", "t": h.get("t")}, None
        if kind == "probe":
            if self._probe is None or h.get("refresh"):
                with self._compute:
                    self._probe = probe(self.xp, self.memory_limit, self.backend)
            info = dict(self._probe)
            info["name"] = self.name
            return {"type": "probe", "info": info}, None
        if kind == "load":
            with self._compute:
                self.stage = None
                gc.collect()
                self.cfg = ModelConfig(h["config"])
                self.dtype = h["dtype"]
                self.layer_range = tuple(h["layers"])
                self.stage = Stage(self.cfg, self.xp)
            print("[%s] assigned layers %d..%d (%s)" % (self.name, self.layer_range[0],
                                                        self.layer_range[1] - 1, self.dtype), flush=True)
            return {"type": "ok"}, None
        if kind == "weights":
            i = int(h["layer"])
            if self.stage is None or not (self.layer_range[0] <= i < self.layer_range[1]):
                raise ValueError("layer %d is not assigned to this node" % i)
            with self._compute:
                self.stage.add_layer(i, t)
            if i == self.layer_range[1] - 1:
                print("[%s] all %d layers loaded" % (self.name, len(self.stage.layers)), flush=True)
            return {"type": "ok"}, None
        if kind == "end_session":
            if self.stage is not None:
                with self._compute:
                    self.stage.end_session(h["session"])
            return {"type": "ok"}, None
        if kind == "status":
            return {"type": "status", "name": self.name, "layers": self.layer_range,
                    "loaded": 0 if self.stage is None else len(self.stage.layers),
                    "sessions": 0 if self.stage is None else len(self.stage.sessions)}, None
        raise ValueError("unknown message type %r" % kind)

    # ---- data plane (fire and forget, passed along the ring) ---------------
    def forward(self, h, t):
        reply_to = tuple(h["reply_to"])
        try:
            if self.stage is None:
                raise RuntimeError("no layers loaded on node %s" % self.name)
            t0 = time.perf_counter()
            with self._compute:
                x = to_numpy(self.stage.forward(h["session"], t["x"], int(h["pos"])))
            ms = (time.perf_counter() - t0) * 1000
            timings = list(h.get("timings", [])) + [[self.name, round(ms, 3)]]
            route = [tuple(a) for a in h.get("route", [])]
            if route:
                nxt = route.pop(0)
                msg = dict(h, route=route, timings=timings)
                self.send_to(nxt, msg, {"x": x})
            else:
                self.send_to(reply_to, {"type": "result", "req": h["req"], "timings": timings}, {"x": x})
        except Exception as e:
            traceback.print_exc()
            try:
                self.send_to(reply_to, {"type": "error", "req": h.get("req"), "node": self.name,
                                        "message": "%s: %s" % (type(e).__name__, e)})
            except Exception:
                traceback.print_exc()

    def send_to(self, addr, header, tensors=None):
        with self._peers_lock:
            entry = self._peers.get(addr)
            if entry is None:
                entry = self._peers[addr] = [None, threading.Lock()]
        with entry[1]:
            for attempt in (0, 1):
                try:
                    if entry[0] is None:
                        entry[0] = connect(addr)
                    send_msg(entry[0], header, tensors)
                    return
                except OSError:
                    if entry[0] is not None:
                        entry[0].close()
                    entry[0] = None
                    if attempt:
                        raise


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        node = self.server.node
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                h, t = recv_msg(sock)
            except (ConnectionError, OSError):
                return
            except Exception as e:
                print("[%s] dropping connection: %s" % (node.name, e), flush=True)
                return
            if h.get("type") == "forward":
                node.forward(h, t)
                continue
            try:
                reply, tensors = node.control(h, t)
            except Exception as e:
                traceback.print_exc()
                reply, tensors = {"type": "error", "node": node.name,
                                  "message": "%s: %s" % (type(e).__name__, e)}, None
            try:
                send_msg(sock, reply, tensors)
            except OSError:
                return


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(host="0.0.0.0", port=7070, name=None, backend="numpy", memory_limit=None, ready=None):
    node = Node(name or socket.gethostname(), backend, memory_limit)
    server = _Server((host, port), _Handler)
    server.node = node
    if ready is not None:
        ready(server)
    return server


def main(argv=None):
    ap = argparse.ArgumentParser(description="Distributed inference node (holds a slice of layers).")
    ap.add_argument("--host", default="0.0.0.0", help="interface to listen on (default: all)")
    ap.add_argument("--port", type=int, default=7070)
    ap.add_argument("--name", default=None, help="label shown in plans (default: hostname)")
    ap.add_argument("--backend", default="numpy", choices=BACKENDS,
                    help="numpy (any CPU) or cupy (NVIDIA GPU, e.g. Jetson)")
    ap.add_argument("--memory-limit", type=float, default=None, metavar="MB",
                    help="cap the memory this node offers, e.g. to leave room for other apps")
    ap.add_argument("--probe", action="store_true", help="print this device's measurements and exit")
    args = ap.parse_args(argv)
    limit = int(args.memory_limit * 1024 * 1024) if args.memory_limit else None

    if args.probe:
        info = probe(get_xp(args.backend), limit, args.backend)
        for k, v in info.items():
            if k == "params_per_s":
                v = ", ".join("%s %.0fM/s" % (d, s / 1e6) for d, s in v.items())
            elif k == "mem_available":
                v = fmt_bytes(v)
            print("%-14s %s" % (k, v))
        return

    server = serve(args.host, args.port, args.name, args.backend, limit)
    print("node %r listening on %s:%d  ->  add it to the coordinator as  %s:%d"
          % (server.node.name, args.host, args.port, lan_ip(), args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
