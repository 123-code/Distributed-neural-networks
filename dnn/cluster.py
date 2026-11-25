"""Coordinator side: talk to nodes, measure them, deploy layers, run the ring."""

import json
import os
import queue
import socket
import socketserver
import statistics
import threading
import time
import uuid

import numpy as np

from .backend import get_xp, to_numpy
from .model import Head, ModelConfig, Stage, layer_tensors_from_checkpoint
from .planner import Device, make_plan
from .safetensors_np import SafeTensors
from .sysinfo import fmt_bytes, probe
from .wire import PROTOCOL_VERSION, connect, recv_msg, send_msg


class NodeError(RuntimeError):
    pass


class RemoteNode:
    def __init__(self, addr, timeout=10.0):
        self.addr = tuple(addr)
        try:
            self.ctrl = connect(self.addr, timeout)
        except OSError as e:
            raise NodeError("can't reach node %s:%d (%s). Is `python -m dnn.node` running there, "
                            "and is the port open on its firewall?" % (self.addr[0], self.addr[1], e))
        self.data = connect(self.addr, timeout)
        self.lock = threading.Lock()
        self.data_lock = threading.Lock()
        hello = self.call({"type": "hello"})
        if hello.get("protocol") != PROTOCOL_VERSION:
            raise NodeError("node %s speaks protocol %s, we speak %s; update the code on that device"
                            % (self.addr, hello.get("protocol"), PROTOCOL_VERSION))
        self.name = hello.get("name") or "%s:%d" % self.addr

    def call(self, header, tensors=None, want_tensors=False):
        with self.lock:
            send_msg(self.ctrl, header, tensors)
            h, t = recv_msg(self.ctrl)
        if h.get("type") == "error":
            raise NodeError("node %s: %s" % (h.get("node", self.addr), h.get("message")))
        return (h, t) if want_tensors else h

    def send_data(self, header, tensors):
        with self.data_lock:
            send_msg(self.data, header, tensors)

    def measure_rtt(self, n=7):
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.call({"type": "ping"})
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    def coordinator_ip(self):
        """Our address as this node sees it (what it must dial to send results back)."""
        return self.ctrl.getsockname()[0]

    def close(self):
        for s in (self.ctrl, self.data):
            try:
                s.close()
            except OSError:
                pass


class _ResultHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                h, t = recv_msg(self.request)
            except (ConnectionError, OSError):
                return
            self.server.deliver(h, t)


class ResultListener(socketserver.ThreadingTCPServer):
    """Where the last node of the ring sends each finished activation."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port=0):
        super().__init__(("0.0.0.0", port), _ResultHandler)
        self._waiting = {}
        self._lock = threading.Lock()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def port(self):
        return self.server_address[1]

    def expect(self, req):
        q = queue.Queue(maxsize=1)
        with self._lock:
            self._waiting[req] = q
        return q

    def deliver(self, h, t):
        with self._lock:
            q = self._waiting.pop(h.get("req"), None)
        if q is not None:
            q.put((h, t))


class Cluster:
    """The coordinator's view of the whole machine.

    Ring for every token:  coordinator (embed + its own layers)
                           -> node 1 -> node 2 -> ... -> coordinator (norm + logits)
    """

    def __init__(self, model_dir, node_addrs=(), dtype="auto", strategy="spread",
                 max_context=2048, use_local=True, backend="numpy", memory_limit=None,
                 listen_port=0, log=print, timeout=120.0):
        self.log = log
        self.timeout = timeout
        self.dtype = dtype
        self.strategy = strategy
        self.max_context = max_context
        self.use_local = use_local
        self.xp = get_xp(backend)
        self.backend = backend
        self.memory_limit = memory_limit
        self.st = SafeTensors(model_dir)
        with open(os.path.join(str(model_dir), "config.json")) as f:
            self.cfg = ModelConfig(json.load(f))
        self.cfg.check()
        self.nodes = [RemoteNode(a) for a in node_addrs]
        self.listener = ResultListener(listen_port) if self.nodes else None
        self.head = None
        self.local_stage = None
        self.plan = None
        self.devices = []
        self.last_timings = []

    # ---- measure and plan ---------------------------------------------------
    def probe_devices(self):
        self.devices = []
        self.log("measuring this machine (coordinator)...")
        info = probe(self.xp, self.memory_limit, self.backend)
        self.local_device = Device("coordinator", info, is_local=True, holds_layers=self.use_local)
        self.devices.append(self.local_device)
        for n in self.nodes:
            self.log("measuring node %s (%s:%d)..." % (n.name, n.addr[0], n.addr[1]))
            info = n.call({"type": "probe"})["info"]
            self.devices.append(Device(n.name, info, addr=n.addr, rtt=n.measure_rtt()))
        return self.devices

    def make_plan(self):
        if not self.devices:
            self.probe_devices()
        plan = make_plan(self.cfg, self.devices, self.dtype, self.strategy, self.max_context)
        self.plan = plan
        return plan

    # ---- deploy -------------------------------------------------------------
    def deploy(self):
        plan = self.plan or self.make_plan()
        local_dtype = plan.dtype_of(self.local_device)
        self.log("loading embedding + output head on the coordinator (%s)..." % local_dtype)
        self.head = Head(self.cfg, self.xp, self.st, local_dtype)
        self.local_stage = Stage(self.cfg, self.xp)
        local = plan.local_stage()
        if local:
            _, s, e = local
            for i in range(s, e):
                self.local_stage.add_layer(i, layer_tensors_from_checkpoint(self.st, i, local_dtype))
            self.log("coordinator holds layers %d-%d (%s)" % (s, e - 1, local_dtype))
        by_addr = {n.addr: n for n in self.nodes}
        for dev, s, e in plan.remote_stages():
            node = by_addr[dev.addr]
            dtype = plan.dtype_of(dev)
            node.call({"type": "load", "config": self.cfg.raw, "layers": [s, e], "dtype": dtype})
            sent, t0 = 0, time.perf_counter()
            for i in range(s, e):
                tensors = layer_tensors_from_checkpoint(self.st, i, dtype)
                node.call({"type": "weights", "layer": i}, tensors)
                sent += sum(a.nbytes for a in tensors.values())
                rate = sent / max(1e-9, time.perf_counter() - t0)
                self.log("  -> %s: layer %d/%d  %s at %s/s" % (node.name, i - s + 1, e - s,
                                                              fmt_bytes(sent), fmt_bytes(rate)))
        self.log("deployed.")

    # ---- run ----------------------------------------------------------------
    def forward(self, session, x, pos):
        """Push hidden states [T, hidden] for positions pos.. through every layer."""
        x = to_numpy(self.local_stage.forward(session, x, pos)).astype(np.float32, copy=False)
        timings = []
        remote = self.plan.remote_stages()
        if remote:
            by_addr = {n.addr: n for n in self.nodes}
            first = by_addr[remote[0][0].addr]
            last = by_addr[remote[-1][0].addr]
            req = uuid.uuid4().hex
            q = self.listener.expect(req)
            header = {"type": "forward", "req": req, "session": session, "pos": int(pos),
                      "route": [list(d.addr) for d, _, _ in remote[1:]],
                      "reply_to": [last.coordinator_ip(), self.listener.port], "timings": []}
            first.send_data(header, {"x": x})
            try:
                h, t = q.get(timeout=self.timeout)
            except queue.Empty:
                raise NodeError("no result within %.0fs; a node may have crashed or cannot reach "
                                "the next hop or the coordinator (check firewalls)" % self.timeout)
            if h.get("type") == "error":
                raise NodeError("node %s failed: %s" % (h.get("node"), h.get("message")))
            x = t["x"]
            timings = h.get("timings", [])
        self.last_timings = timings
        return x

    def end_session(self, session):
        if self.local_stage is not None:
            self.local_stage.end_session(session)
        for n in self.nodes:
            try:
                n.call({"type": "end_session", "session": session})
            except Exception:
                pass

    def close(self):
        for n in self.nodes:
            n.close()
        if self.listener is not None:
            self.listener.shutdown()
            self.listener.server_close()
