"""Check our NumPy math against Hugging Face transformers on randomly initialised
models. Skipped when torch/transformers aren't installed (they are never needed
to *run* the framework)."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import transformers
    HAVE_HF = True
except Exception:
    HAVE_HF = False

from dnn.cluster import Cluster  # noqa: E402
from tiny import CONFIGS  # noqa: E402


@unittest.skipUnless(HAVE_HF, "torch + transformers not installed")
class ReferenceTest(unittest.TestCase):
    def check(self, kind):
        cfg = dict(CONFIGS[kind])
        cfg.pop("model_type")
        cls = {"qwen2": "Qwen2", "qwen3": "Qwen3", "llama": "Llama"}[kind]
        if not hasattr(transformers, cls + "Config"):
            self.skipTest("transformers has no %s" % cls)
        torch.manual_seed(0)
        conf = getattr(transformers, cls + "Config")(**cfg)
        model = getattr(transformers, cls + "ForCausalLM")(conf).eval()
        with torch.no_grad():  # random init leaves norms at 1; perturb them to test them
            for name, p in model.named_parameters():
                if "norm" in name:
                    p.add_(torch.randn_like(p) * 0.1)
        ids = [5, 17, 99, 3, 250, 42, 7, 7, 128, 1]
        with torch.no_grad():
            ref = model(torch.tensor([ids])).logits[0].numpy()
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d, safe_serialization=True)
            c = Cluster(d, dtype="float32", log=lambda m: None)
            c.make_plan()
            c.deploy()
            # prefill 6 tokens, then decode the rest one by one through the KV cache
            h = np.asarray(c.forward("s", c.head.embed(ids[:6]), 0))
            got = [np.asarray(c.head.logits(h))]
            for i in range(6, len(ids)):
                h = np.asarray(c.forward("s", c.head.embed([ids[i]]), i))
                got.append(np.asarray(c.head.logits(h)))
            got = np.concatenate(got)
            c.close()
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)

    def test_qwen2(self):
        self.check("qwen2")

    def test_qwen3(self):
        self.check("qwen3")

    def test_llama3_rope_scaling(self):
        self.check("llama")


if __name__ == "__main__":
    unittest.main()
