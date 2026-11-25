"""Decoder-only transformer math (Qwen2 / Qwen2.5 / Qwen3 / Llama / Mistral family)
written against a NumPy-compatible array module `xp` (numpy, or cupy on GPUs).

Only what a pipeline stage needs lives here: one decoder layer, its KV cache,
and the head (embedding, final norm, output projection) that the coordinator keeps.
"""

import math
import time
import warnings

import numpy as np

F32 = np.float32
STORAGE_DTYPES = ("float32", "float16", "int8")
BYTES_PER_PARAM = {"float32": 4.0, "float16": 2.0, "int8": 1.0}
SUPPORTED_MODEL_TYPES = ("qwen2", "qwen3", "llama", "mistral")

# Short names used on the wire -> Hugging Face tensor names inside layer i.
LAYER_TENSORS = {
    "in_norm": "input_layernorm.weight",
    "q": "self_attn.q_proj.weight",
    "q_b": "self_attn.q_proj.bias",
    "k": "self_attn.k_proj.weight",
    "k_b": "self_attn.k_proj.bias",
    "v": "self_attn.v_proj.weight",
    "v_b": "self_attn.v_proj.bias",
    "o": "self_attn.o_proj.weight",
    "o_b": "self_attn.o_proj.bias",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "post_norm": "post_attention_layernorm.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}
MATRIX_TENSORS = ("q", "k", "v", "o", "gate", "up", "down")


class ModelConfig:
    def __init__(self, d):
        self.raw = dict(d)
        self.model_type = d.get("model_type", "")
        self.hidden = int(d["hidden_size"])
        self.n_heads = int(d["num_attention_heads"])
        self.n_kv = int(d.get("num_key_value_heads") or self.n_heads)
        self.head_dim = int(d.get("head_dim") or self.hidden // self.n_heads)
        self.inter = int(d["intermediate_size"])
        self.n_layers = int(d["num_hidden_layers"])
        self.eps = float(d.get("rms_norm_eps", 1e-6))
        # Older configs keep rope_theta / rope_scaling at the top level; newer ones
        # nest both inside "rope_parameters".
        rp = d.get("rope_parameters") or {}
        self.rope_theta = float(d.get("rope_theta") or rp.get("rope_theta") or 10000.0)
        scaling = d.get("rope_scaling") or rp
        kind = scaling.get("rope_type") or scaling.get("type") if scaling else None
        self.rope_scaling = scaling if kind not in (None, "default") else None
        self.vocab = int(d["vocab_size"])
        self.tie = bool(d.get("tie_word_embeddings", False))
        self.max_pos = int(d.get("max_position_embeddings", 32768))

    def check(self):
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            warnings.warn("model_type %r is untested; it will work only if it is a plain "
                          "Llama-style decoder" % self.model_type)
        if self.n_heads % self.n_kv:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

    def layer_params(self):
        """Number of weights in one decoder layer (what each stage stores and
        reads once per generated token)."""
        q = self.hidden * self.n_heads * self.head_dim
        kv = 2 * self.hidden * self.n_kv * self.head_dim
        o = self.n_heads * self.head_dim * self.hidden
        mlp = 3 * self.hidden * self.inter
        return q + kv + o + mlp

    def layer_bytes(self, dtype):
        return int(self.layer_params() * BYTES_PER_PARAM[dtype])

    def kv_bytes_per_token_per_layer(self):
        return 2 * self.n_kv * self.head_dim * 4  # K and V, float32

    def head_params(self):
        emb = self.vocab * self.hidden
        return emb if self.tie else 2 * emb


# ---------------------------------------------------------------------------
# Weight storage: float32, float16, or int8 with one float32 scale per output row.
# ---------------------------------------------------------------------------

def quantize(w, dtype):
    """float32 matrix -> dict of arrays in the storage format."""
    w = np.asarray(w, dtype=F32)
    if dtype == "float32":
        return {"": w}
    if dtype == "float16":
        return {"": w.astype(np.float16)}
    if dtype == "int8":
        scale = np.abs(w).max(axis=-1, keepdims=True) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
        return {"": q, ".s": scale.reshape(-1).astype(F32)}
    raise ValueError("unknown dtype %r (use one of %s)" % (dtype, ", ".join(STORAGE_DTYPES)))


class Linear:
    """y = x @ W.T (+ b), with W kept in its storage dtype and expanded to float32
    a slice at a time, so a small device never needs a float32 copy of the matrix."""

    CHUNK_ELEMS = 1 << 21

    def __init__(self, xp, w, scale=None, bias=None):
        self.xp = xp
        self.w = xp.asarray(w)
        self.scale = None if scale is None else xp.asarray(scale, dtype=F32)
        self.bias = None if bias is None else xp.asarray(bias, dtype=F32)
        self.out_features, self.in_features = self.w.shape

    def __call__(self, x):
        xp = self.xp
        if self.w.dtype == F32 and self.scale is None:
            y = x @ self.w.T
        else:
            y = xp.empty(x.shape[:-1] + (self.out_features,), dtype=F32)
            step = max(1, self.CHUNK_ELEMS // self.in_features)
            for r in range(0, self.out_features, step):
                wc = self.w[r:r + step].astype(F32)
                yc = x @ wc.T
                if self.scale is not None:
                    yc *= self.scale[r:r + step]
                y[..., r:r + step] = yc
        if self.bias is not None:
            y = y + self.bias
        return y


def _linear(xp, tensors, name):
    if name not in tensors:
        return None
    return Linear(xp, tensors[name], tensors.get(name + ".s"), tensors.get(name + "_b"))


def rms_norm(xp, x, w, eps):
    var = (x * x).mean(axis=-1, keepdims=True)
    return (x / xp.sqrt(var + eps)) * w


def rope_inv_freq(cfg):
    hd = cfg.head_dim
    inv = (1.0 / (cfg.rope_theta ** (np.arange(0, hd, 2, dtype=np.int64).astype(F32) / hd))).astype(F32)
    rs = cfg.rope_scaling or {}
    kind = rs.get("rope_type") or rs.get("type")
    if kind == "linear":
        inv = inv / F32(rs["factor"])
    elif kind == "llama3":
        factor = rs["factor"]
        low, high = rs.get("low_freq_factor", 1.0), rs.get("high_freq_factor", 4.0)
        old = rs.get("original_max_position_embeddings", 8192)
        low_wl, high_wl = old / low, old / high
        wl = 2 * math.pi / inv
        scaled = np.where(wl > low_wl, inv / factor, inv)
        smooth = (old / wl - low) / (high - low)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = (wl >= high_wl) & (wl <= low_wl)
        inv = np.where(medium, smoothed, scaled).astype(F32)
    elif kind not in (None, "default"):
        warnings.warn("rope_scaling type %r not implemented; using plain RoPE" % kind)
    return inv


def rope_tables(xp, inv_freq, pos, n):
    positions = xp.arange(pos, pos + n, dtype=F32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = xp.concatenate([freqs, freqs], axis=-1)
    return xp.cos(emb)[:, None, :], xp.sin(emb)[:, None, :]  # [T, 1, head_dim]


def apply_rope(xp, x, cos, sin):
    half = x.shape[-1] // 2
    rotated = xp.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class KVCache:
    def __init__(self, xp, n_kv, head_dim):
        self.xp, self.n_kv, self.head_dim = xp, n_kv, head_dim
        self.k = self.v = None
        self.length = 0

    def append(self, pos, k, v):
        if pos > self.length:
            raise ValueError("position %d skips past cached length %d" % (pos, self.length))
        self.length = pos  # pos < length rewinds (e.g. regenerate)
        need = pos + k.shape[0]
        cap = 0 if self.k is None else self.k.shape[0]
        if need > cap:
            new_cap = max(need, 2 * cap, 64)
            nk = self.xp.empty((new_cap, self.n_kv, self.head_dim), dtype=F32)
            nv = self.xp.empty_like(nk)
            if self.length:
                nk[:self.length] = self.k[:self.length]
                nv[:self.length] = self.v[:self.length]
            self.k, self.v = nk, nv
        self.k[pos:need] = k
        self.v[pos:need] = v
        self.length = need
        return self.k[:need], self.v[:need]

    def nbytes(self):
        return 0 if self.k is None else 2 * self.k.nbytes


class DecoderLayer:
    def __init__(self, cfg, xp, tensors):
        self.cfg, self.xp = cfg, xp
        self.in_norm = xp.asarray(tensors["in_norm"], dtype=F32)
        self.post_norm = xp.asarray(tensors["post_norm"], dtype=F32)
        self.q_norm = xp.asarray(tensors["q_norm"], dtype=F32) if "q_norm" in tensors else None
        self.k_norm = xp.asarray(tensors["k_norm"], dtype=F32) if "k_norm" in tensors else None
        self.q, self.k, self.v, self.o = (_linear(xp, tensors, n) for n in ("q", "k", "v", "o"))
        self.gate, self.up, self.down = (_linear(xp, tensors, n) for n in ("gate", "up", "down"))

    def forward(self, x, cos, sin, cache, pos):
        xp, c = self.xp, self.cfg
        t = x.shape[0]
        h = rms_norm(xp, x, self.in_norm, c.eps)
        q = self.q(h).reshape(t, c.n_heads, c.head_dim)
        k = self.k(h).reshape(t, c.n_kv, c.head_dim)
        v = self.v(h).reshape(t, c.n_kv, c.head_dim)
        if self.q_norm is not None:
            q = rms_norm(xp, q, self.q_norm, c.eps)
            k = rms_norm(xp, k, self.k_norm, c.eps)
        q = apply_rope(xp, q, cos, sin)
        k = apply_rope(xp, k, cos, sin)
        keys, values = cache.append(pos, k, v)  # [S, n_kv, hd]
        s = keys.shape[0]
        g = c.n_heads // c.n_kv
        qh = q.reshape(t, c.n_kv, g, c.head_dim).transpose(1, 2, 0, 3)       # [kv, g, T, hd]
        scores = (qh @ keys.transpose(1, 2, 0)[:, None]) * F32(1.0 / math.sqrt(c.head_dim))
        if t > 1:  # causal mask: query at absolute position pos+i sees keys <= pos+i
            future = xp.arange(s)[None, :] > (pos + xp.arange(t))[:, None]
            scores = xp.where(future, F32(-np.inf), scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = xp.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        attn = p @ values.transpose(1, 0, 2)[:, None]                         # [kv, g, T, hd]
        attn = attn.transpose(2, 0, 1, 3).reshape(t, c.n_heads * c.head_dim)
        x = x + self.o(attn)
        h = rms_norm(xp, x, self.post_norm, c.eps)
        gate = self.gate(h)
        x = x + self.down(gate / (1.0 + xp.exp(-gate)) * self.up(h))  # SiLU(gate) * up
        return x.astype(F32, copy=False)


class Stage:
    """A contiguous run of decoder layers plus their per-session KV caches."""

    SESSION_TTL_S = 3600.0

    def __init__(self, cfg, xp):
        self.cfg, self.xp = cfg, xp
        self.layers = {}
        self.sessions = {}  # session id -> ({layer index: KVCache}, last used time)
        self.inv_freq = xp.asarray(rope_inv_freq(cfg))

    def add_layer(self, index, tensors):
        self.layers[index] = DecoderLayer(self.cfg, self.xp, tensors)

    def forward(self, session, x, pos):
        if not self.layers:
            return x
        self._expire()
        caches, _ = self.sessions.get(session, ({}, 0))
        self.sessions[session] = (caches, time.time())
        x = self.xp.asarray(x, dtype=F32)
        cos, sin = rope_tables(self.xp, self.inv_freq, pos, x.shape[0])
        for i in sorted(self.layers):
            cache = caches.get(i)
            if cache is None:
                cache = caches[i] = KVCache(self.xp, self.cfg.n_kv, self.cfg.head_dim)
            x = self.layers[i].forward(x, cos, sin, cache, pos)
        return x

    def end_session(self, session):
        self.sessions.pop(session, None)

    def _expire(self):
        now = time.time()
        for sid in [s for s, (_, t) in self.sessions.items() if now - t > self.SESSION_TTL_S]:
            del self.sessions[sid]


def layer_tensors_from_checkpoint(st, index, dtype):
    """Read layer `index` from a checkpoint and convert it to the storage dtype.
    Returns {wire name: array}."""
    out = {}
    prefix = "model.layers.%d." % index
    for short, hf in LAYER_TENSORS.items():
        name = prefix + hf
        if not st.has(name):
            continue
        w = st.get(name)
        if short in MATRIX_TENSORS:
            for suffix, arr in quantize(w, dtype).items():
                out[short + suffix] = arr
        else:
            out[short] = w.astype(F32)
    missing = [n for n in ("in_norm", "post_norm") + MATRIX_TENSORS if n not in out]
    if missing:
        raise KeyError("layer %d is missing tensors %s" % (index, missing))
    return out


class Head:
    """Embedding lookup, final norm and output projection. Lives on the coordinator,
    so the token loop (embed -> layers -> logits -> sample -> embed) closes locally
    and a tied embedding matrix is stored once."""

    def __init__(self, cfg, xp, st, dtype):
        self.cfg, self.xp = cfg, xp
        emb = quantize(st.get("model.embed_tokens.weight"), dtype)
        self.emb = xp.asarray(emb[""])
        self.emb_scale = xp.asarray(emb[".s"]) if ".s" in emb else None
        self.norm = xp.asarray(st.get("model.norm.weight"), dtype=F32)
        if cfg.tie or not st.has("lm_head.weight"):
            self.lm_head = Linear(xp, self.emb, self.emb_scale)
        else:
            lm = quantize(st.get("lm_head.weight"), dtype)
            self.lm_head = Linear(xp, lm[""], lm.get(".s"))

    def nbytes(self):
        n = self.emb.nbytes + self.norm.nbytes
        if self.lm_head.w is not self.emb:
            n += self.lm_head.w.nbytes
        return n

    def embed(self, ids):
        ids = self.xp.asarray(np.asarray(ids, dtype=np.int64))
        x = self.emb[ids].astype(F32)
        if self.emb_scale is not None:
            x *= self.emb_scale[ids][:, None]
        return x

    def logits(self, h):
        return self.lm_head(rms_norm(self.xp, h, self.norm, self.cfg.eps))
