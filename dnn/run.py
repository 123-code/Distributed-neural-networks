"""Coordinator CLI: measure devices, plan, deploy, chat.

    python -m dnn.run --model Qwen/Qwen2.5-0.5B-Instruct                       # one machine
    python -m dnn.run --model Qwen/Qwen2.5-0.5B-Instruct \
        --nodes 192.168.1.20:7070,192.168.1.31:7070                            # a cluster
    python -m dnn.run ... --plan-only                                          # just show the plan
    python -m dnn.run ... --prompt "Why is the sky blue?"                      # one-shot
"""

import argparse
import json
import os
import sys
import time
import uuid

import numpy as np

from .backend import BACKENDS, to_numpy
from .cluster import Cluster, NodeError
from .download import resolve_model
from .planner import DTYPE_CHOICES, PlanError
from .wire import parse_addr


class Tokenizer:
    def __init__(self, model_dir):
        self.impl = None
        path = os.path.join(model_dir, "tokenizer.json")
        try:
            from tokenizers import Tokenizer as T
            if os.path.exists(path):
                self.impl = ("tokenizers", T.from_file(path))
        except ImportError:
            pass
        if self.impl is None:
            try:
                from transformers import AutoTokenizer
                self.impl = ("transformers", AutoTokenizer.from_pretrained(model_dir))
            except Exception:
                raise SystemExit("the coordinator needs a tokenizer: pip install tokenizers "
                                 "(and make sure tokenizer.json is in %s)" % model_dir)

    def encode(self, text):
        kind, t = self.impl
        if kind == "tokenizers":
            return t.encode(text, add_special_tokens=False).ids
        return t.encode(text, add_special_tokens=False)

    def decode(self, ids):
        kind, t = self.impl
        if kind == "tokenizers":
            return t.decode(ids, skip_special_tokens=True)
        return t.decode(ids, skip_special_tokens=True)

    def token_id(self, text):
        kind, t = self.impl
        if kind == "tokenizers":
            return t.token_to_id(text)
        i = t.convert_tokens_to_ids(text)
        return None if i == t.unk_token_id else i


class ChatFormat:
    """Turns chat messages into tokens for ChatML (Qwen) and Llama-3 models;
    anything else is treated as plain text completion."""

    def __init__(self, tok, system):
        self.tok, self.system = tok, system
        if tok.token_id("<|im_start|>") is not None:
            self.kind = "chatml"
            self.end = tok.token_id("<|im_end|>")
        elif tok.token_id("<|start_header_id|>") is not None:
            self.kind = "llama3"
            self.end = tok.token_id("<|eot_id|>")
        else:
            self.kind, self.end = "plain", None

    def first_turn(self, msg):
        if self.kind == "chatml":
            text = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
                    "<|im_start|>assistant\n" % (self.system, msg))
        elif self.kind == "llama3":
            text = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n" % (self.system, msg))
        else:
            text = msg
        return self.tok.encode(text)

    def next_turn(self, msg, pending, ended):
        """Tokens for a follow-up turn. `pending` holds the last token of the previous
        reply, which was sampled but never fed through the model; `ended` says whether
        it was the end-of-turn token (if not, the reply was cut off and we close it)."""
        ids = list(pending)
        if not ended and self.end is not None:
            ids.append(self.end)
        if self.kind == "chatml":
            return ids + self.tok.encode("\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % msg)
        if self.kind == "llama3":
            return ids + self.tok.encode("<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
                                         "<|start_header_id|>assistant<|end_header_id|>\n\n" % msg)
        return ids + self.tok.encode(msg)


def stop_ids(model_dir, tok):
    ids = set()
    for fn in ("generation_config.json", "config.json"):
        try:
            with open(os.path.join(model_dir, fn)) as f:
                e = json.load(f).get("eos_token_id")
            ids.update(e if isinstance(e, list) else [e] if e is not None else [])
        except (OSError, ValueError):
            pass
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>"):
        i = tok.token_id(t)
        if i is not None:
            ids.add(i)
    return ids


def sample(logits, temperature, top_p, rng):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature <= 0:
        return int(logits.argmax())
    z = logits / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        keep = np.cumsum(p[order]) - p[order] < top_p
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep]] = True
        p = np.where(mask, p, 0)
        p /= p.sum()
    return int(rng.choice(len(p), p=p))


class Generator:
    def __init__(self, cluster, args, tok, stops):
        self.c, self.args, self.tok, self.stops = cluster, args, tok, stops
        self.rng = np.random.default_rng(args.seed)
        self.session = uuid.uuid4().hex
        self.pos = 0
        self.pending, self.ended = [], True

    def reset(self):
        self.c.end_session(self.session)
        self.session = uuid.uuid4().hex
        self.pos, self.pending, self.ended = 0, [], True

    def step(self, ids):
        h = self.c.forward(self.session, self.c.head.embed(ids), self.pos)
        self.pos += len(ids)
        return to_numpy(self.c.head.logits(self.c.xp.asarray(h[-1:])))[0]

    def reply(self, ids, out):
        a = self.args
        if self.pos + len(ids) + a.max_new_tokens > a.max_context:
            out("\n[context full: starting a new conversation]\n")
            return None
        t0 = time.perf_counter()
        logits = self.step(ids)
        t_prefill = time.perf_counter() - t0
        generated, shown, closing = [], "", None
        t1 = time.perf_counter()
        for _ in range(a.max_new_tokens):
            tok = sample(logits, a.temperature, a.top_p, self.rng)
            if tok in self.stops:
                closing = tok
                break
            generated.append(tok)
            text = self.tok.decode(generated)
            if not text.endswith("�"):  # wait for complete UTF-8 characters
                out(text[len(shown):])
                shown = text
            if len(generated) == a.max_new_tokens:
                break
            logits = self.step([tok])
        t_decode = time.perf_counter() - t1
        if closing is not None:
            self.pending, self.ended = [closing], True
        else:
            self.pending, self.ended = generated[-1:], False
        n = max(1, len(generated))
        stats = {"prompt_tokens": len(ids), "prefill_s": t_prefill, "new_tokens": len(generated),
                 "decode_tok_s": n / t_decode if t_decode else 0.0, "timings": self.c.last_timings}
        return stats


def print_stats(stats, cluster):
    if not stats:
        return
    line = ("[%d prompt tokens in %.2fs | %d new tokens at %.1f tok/s"
            % (stats["prompt_tokens"], stats["prefill_s"], stats["new_tokens"], stats["decode_tok_s"]))
    if cluster.plan is not None:
        line += " | plan estimate %.1f tok/s" % (1 / cluster.plan.est_s if cluster.plan.est_s else 0)
    if stats["timings"]:
        line += " | last token per node: " + ", ".join("%s %.1fms" % (n, ms) for n, ms in stats["timings"])
    print(line + "]", file=sys.stderr)


def build_parser():
    ap = argparse.ArgumentParser(description="Run a language model across many devices.")
    ap.add_argument("--model", required=True,
                    help="Hub repo id (e.g. Qwen/Qwen2.5-0.5B-Instruct) or a local directory")
    ap.add_argument("--nodes", default="",
                    help="comma-separated host:port of running `python -m dnn.node` processes")
    ap.add_argument("--dtype", default="auto", choices=DTYPE_CHOICES,
                    help="weight storage on devices: auto = per device, the fastest format that "
                         "fits its share; int8 fits the most; float32 is exact")
    ap.add_argument("--strategy", default="spread", choices=("spread", "fastest"),
                    help="spread: use every device that can hold a layer; fastest: lowest latency")
    ap.add_argument("--no-local", action="store_true",
                    help="coordinator holds only the embedding/head, no decoder layers")
    ap.add_argument("--backend", default="numpy", choices=BACKENDS, help="coordinator array backend")
    ap.add_argument("--memory-limit", type=float, default=None, metavar="MB",
                    help="cap the memory the coordinator offers")
    ap.add_argument("--max-context", type=int, default=2048, help="tokens of context to reserve KV memory for")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--prompt", default=None, help="answer one prompt and exit")
    ap.add_argument("--plan-only", action="store_true", help="measure devices, print the plan, exit")
    ap.add_argument("--listen-port", type=int, default=0,
                    help="port the last node sends results to (default: any free port)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for one ring pass")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_dir = resolve_model(args.model)
    nodes = [parse_addr(a) for a in args.nodes.split(",") if a.strip()]
    limit = int(args.memory_limit * 2 ** 20) if args.memory_limit else None
    try:
        cluster = Cluster(model_dir, nodes, dtype=args.dtype, strategy=args.strategy,
                          max_context=args.max_context, use_local=not args.no_local,
                          backend=args.backend, memory_limit=limit, listen_port=args.listen_port,
                          log=lambda m: print(m, file=sys.stderr), timeout=args.timeout)
        plan = cluster.make_plan()
    except (NodeError, PlanError) as e:
        raise SystemExit("error: %s" % e)
    print(plan.report(), file=sys.stderr)
    if args.plan_only:
        cluster.close()
        return
    try:
        cluster.deploy()
        tok = Tokenizer(model_dir)
        chat = ChatFormat(tok, args.system)
        gen = Generator(cluster, args, tok, stop_ids(model_dir, tok))
        out = lambda s: (sys.stdout.write(s), sys.stdout.flush())

        if args.prompt is not None:
            stats = gen.reply(chat.first_turn(args.prompt), out)
            print()
            print_stats(stats, cluster)
            return

        print("\nready. /reset starts a new conversation, /quit exits.", file=sys.stderr)
        first = True
        while True:
            try:
                msg = input("\n> ").strip()
            except EOFError:
                break
            if not msg:
                continue
            if msg in ("/quit", "/exit"):
                break
            if msg == "/reset":
                gen.reset()
                first = True
                continue
            ids = chat.first_turn(msg) if first else chat.next_turn(msg, gen.pending, gen.ended)
            stats = gen.reply(ids, out)
            if stats is None:
                gen.reset()
                stats = gen.reply(chat.first_turn(msg), out)
            first = False
            print()
            print_stats(stats, cluster)
    except NodeError as e:
        raise SystemExit("error: %s" % e)
    except KeyboardInterrupt:
        pass
    finally:
        cluster.close()


if __name__ == "__main__":
    main()
