"""End-to-end: real node processes + the coordinator CLI on a tiny model.
Needs the `tokenizers` package (only the coordinator needs it in real use)."""

import os
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tokenizers  # noqa: F401
    HAVE_TOK = True
except ImportError:
    HAVE_TOK = False

from tiny import make_model  # noqa: E402


def make_tokenizer(model_dir, vocab_size):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    text = ["the quick brown fox jumps over the lazy dog", "hello world, how are you today?",
            "system user assistant you are a helpful assistant"] * 50
    tok.train_from_iterator(text, trainer)
    tok.save(os.path.join(model_dir, "tokenizer.json"))


@unittest.skipUnless(HAVE_TOK, "tokenizers not installed")
class CliTest(unittest.TestCase):
    def test_two_nodes_chat(self):
        with tempfile.TemporaryDirectory() as d:
            model = make_model(os.path.join(d, "m"), "qwen2")
            make_tokenizer(model, 300)
            env = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1")
            nodes = []
            try:
                for port, name in ((17071, "alpha"), (17072, "beta")):
                    nodes.append(subprocess.Popen(
                        [sys.executable, "-m", "dnn.node", "--host", "127.0.0.1", "--port", str(port),
                         "--name", name], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
                time.sleep(1.5)
                cmd = [sys.executable, "-m", "dnn.run", "--model", model,
                       "--nodes", "127.0.0.1:17071,127.0.0.1:17072",
                       "--max-new-tokens", "8", "--seed", "0", "--max-context", "256"]
                one = subprocess.run(cmd + ["--prompt", "hello world"], env=env,
                                     capture_output=True, text=True, timeout=120)
                self.assertEqual(one.returncode, 0, one.stderr)
                self.assertIn("alpha", one.stderr)
                self.assertIn("beta", one.stderr)
                self.assertIn("tok/s", one.stderr)
                chat = subprocess.run(cmd, input="hello\nhow are you\n/reset\nthe fox\n/quit\n",
                                      env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(chat.returncode, 0, chat.stderr)
                self.assertEqual(chat.stderr.count("new tokens at"), 3, chat.stderr)
                plan = subprocess.run(cmd + ["--plan-only", "--strategy", "fastest"], env=env,
                                      capture_output=True, text=True, timeout=120)
                self.assertEqual(plan.returncode, 0, plan.stderr)
                self.assertIn("not needed", plan.stderr)
            finally:
                for p in nodes:
                    p.terminate()
                    p.wait(timeout=10)
                    p.stdout.close()


if __name__ == "__main__":
    unittest.main()
