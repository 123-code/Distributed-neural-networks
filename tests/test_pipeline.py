"""Offline tests that need only NumPy:  python -m unittest discover -s tests -v"""

import os
import socket
import sys
import tempfile
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dnn.cluster import Cluster  # noqa: E402
from dnn.model import ModelConfig  # noqa: E402
from dnn.node import serve  # noqa: E402
from dnn.planner import Device, PlanError, make_plan  # noqa: E402
from dnn.wire import recv_msg, send_msg  # noqa: E402
from tiny import CONFIGS, make_model  # noqa: E402


def start_node(name, memory_limit=None):
    server = serve("127.0.0.1", 0, name=name, memory_limit=memory_limit)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address


def run_tokens(cluster, ids, chunks):
    """Feed `ids` in the given chunk sizes; return the final hidden state of each chunk."""
    session, pos, outs = "s", 0, []
    for n in chunks:
        x = cluster.head.embed(ids[pos:pos + n])
        outs.append(np.array(cluster.forward(session, x, pos)))
        pos += n
    return outs


class WireTest(unittest.TestCase):
    def test_roundtrip(self):
        a, b = socket.socketpair()
        tensors = {"f": np.arange(6, dtype=np.float32).reshape(2, 3),
                   "h": np.ones(4, dtype=np.float16), "q": np.array([-3, 7], dtype=np.int8),
                   "empty": np.zeros((0, 5), dtype=np.float32)}
        send_msg(a, {"type": "x", "n": 1}, tensors)
        h, t = recv_msg(b)
        self.assertEqual(h["type"], "x")
        for k, v in tensors.items():
            np.testing.assert_array_equal(t[k], v)
            self.assertEqual(t[k].dtype, v.dtype)
        a.close()
        b.close()


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = make_model(os.path.join(cls.tmp.name, "qwen2"), "qwen2")
        cls.ids = list(np.random.default_rng(1).integers(0, 300, size=12))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def local(self, dtype="float32"):
        c = Cluster(self.model, dtype=dtype, log=lambda m: None)
        c.make_plan()
        c.deploy()
        return c

    def test_kv_cache_matches_full_prefill(self):
        c = self.local()
        full = run_tokens(c, self.ids, [12])[0]
        stepwise = run_tokens(c, self.ids, [5, 1, 1, 3, 2])
        np.testing.assert_allclose(stepwise[-1][-1], full[-1], rtol=1e-4, atol=1e-5)
        c.close()

    def test_distributed_equals_local(self):
        ref = run_tokens(self.local(), self.ids, [7, 1, 1, 3])
        servers = [start_node("a"), start_node("b")]
        try:
            c = Cluster(self.model, node_addrs=[s[1] for s in servers], dtype="float32",
                        strategy="spread", log=lambda m: None)
            plan = c.make_plan()
            self.assertEqual(len(plan.stages), 3, plan.report())
            self.assertEqual(sum(e - s for _, s, e in plan.stages), 6)
            c.deploy()
            got = run_tokens(c, self.ids, [7, 1, 1, 3])
            for r, g in zip(ref, got):
                np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-5)
            self.assertEqual([n for n, _ in c.last_timings], ["a", "b"])
            c.end_session("s")
            c.close()
        finally:
            for s, _ in servers:
                s.shutdown()
                s.server_close()

    def test_remote_only(self):
        ref = run_tokens(self.local(), self.ids, [4, 1])
        server, addr = start_node("solo")
        try:
            c = Cluster(self.model, node_addrs=[addr], dtype="float32", use_local=False,
                        log=lambda m: None)
            plan = c.make_plan()
            self.assertIsNone(plan.local_stage())
            c.deploy()
            got = run_tokens(c, self.ids, [4, 1])
            np.testing.assert_allclose(got[-1], ref[-1], rtol=1e-4, atol=1e-5)
            c.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_quantized_close_to_float32(self):
        ref = run_tokens(self.local("float32"), self.ids, [12])[0]
        for dtype, tol in (("float16", 2e-3), ("int8", 5e-2)):
            got = run_tokens(self.local(dtype), self.ids, [12])[0]
            rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
            self.assertLess(rel, tol, dtype)

    def test_generation_logits_finite(self):
        c = self.local("int8")
        h = run_tokens(c, self.ids, [12])[0]
        logits = np.asarray(c.head.logits(h[-1:]))
        self.assertEqual(logits.shape, (1, 300))
        self.assertTrue(np.isfinite(logits).all())


def fake(name, mem_mb, mparams_s, rtt_ms=3.0, local=False, f32_mparams_s=None):
    speed = mparams_s * 1e6
    f32 = speed if f32_mparams_s is None else f32_mparams_s * 1e6
    info = {"mem_available": int(mem_mb * 2 ** 20), "os": "test",
            "params_per_s": {"float32": f32, "float16": speed, "int8": speed}}
    return Device(name, info, addr=None if local else (name, 7070), rtt=rtt_ms / 1000, is_local=local)


QWEN05 = ModelConfig(dict(model_type="qwen2", hidden_size=896, intermediate_size=4864,
                          num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
                          vocab_size=151936, tie_word_embeddings=True))


class PlannerTest(unittest.TestCase):
    def test_microwave_is_excluded_with_reason(self):
        devs = [fake("laptop", 8000, 3000, local=True), fake("microwave", 0.002, 0.001)]
        plan = make_plan(QWEN05, devs, "int8", "spread")
        self.assertEqual([d.name for d, _, _ in plan.stages], ["laptop"])
        self.assertIn("can't hold one layer", dict((d.name, r) for d, r in plan.excluded)["microwave"])

    def test_spread_uses_every_capable_device_proportionally(self):
        devs = [fake("laptop", 8000, 3000, local=True), fake("jetson", 6000, 3000),
                fake("firetv", 700, 300)]
        plan = make_plan(QWEN05, devs, "int8", "spread")
        counts = {d.name: e - s for d, s, e in plan.stages}
        self.assertEqual(sum(counts.values()), 24)
        self.assertEqual(set(counts), {"laptop", "jetson", "firetv"})
        self.assertLess(counts["firetv"], counts["jetson"])

    def test_fastest_prefers_one_fast_device(self):
        devs = [fake("laptop", 8000, 3000, local=True), fake("jetson", 6000, 3000),
                fake("firetv", 700, 300)]
        plan = make_plan(QWEN05, devs, "int8", "fastest")
        self.assertEqual([d.name for d, _, _ in plan.stages], ["laptop"])

    def test_memory_forces_distribution(self):
        devs = [fake("laptop", 500, 3000, local=True), fake("jetson", 400, 3000),
                fake("firetv", 400, 300)]
        plan = make_plan(QWEN05, devs, "int8", "fastest")
        self.assertGreater(len(plan.stages), 1)
        self.assertEqual(sum(e - s for _, s, e in plan.stages), 24)
        caps_ok = all((e - s) >= 1 for _, s, e in plan.stages)
        self.assertTrue(caps_ok)

    def test_not_enough_memory_explains_shortfall(self):
        devs = [fake("laptop", 1200, 3000, local=True), fake("firetv", 150, 300)]
        with self.assertRaises(PlanError) as ctx:
            make_plan(QWEN05, devs, "float32", "spread")
        self.assertIn("Short by about", str(ctx.exception))

    def test_auto_dtype_picks_fastest_format_that_fits(self):
        # x86-like laptop: float32 10x faster than int8 and plenty of RAM;
        # small device: only int8 fits its share.
        devs = [fake("laptop", 16000, 2500, local=True, f32_mparams_s=25000),
                fake("firetv", 180, 250, f32_mparams_s=400)]
        plan = make_plan(QWEN05, devs, "auto", "spread")
        self.assertEqual(plan.dtype_of(devs[0]), "float32")
        self.assertEqual(plan.dtype_of(devs[1]), "int8")
        self.assertEqual(sum(e - s for _, s, e in plan.stages), 24)
        fixed = make_plan(QWEN05, devs, "int8", "spread")
        self.assertLess(plan.est_s, fixed.est_s)
        self.assertEqual({d.name for d, _, _ in plan.stages}, {"laptop", "firetv"})

    def test_layers_are_contiguous_and_complete(self):
        cfg = ModelConfig(CONFIGS["qwen2"])
        devs = [fake("c", 300, 50, local=True)] + [fake("n%d" % i, 150, 10 * (i + 1)) for i in range(4)]
        plan = make_plan(cfg, devs, "int8", "spread")
        edges = [(s, e) for _, s, e in plan.stages]
        self.assertEqual(edges[0][0], 0)
        self.assertEqual(edges[-1][1], cfg.n_layers)
        for (_, e1), (s2, _) in zip(edges, edges[1:]):
            self.assertEqual(e1, s2)


if __name__ == "__main__":
    unittest.main()
