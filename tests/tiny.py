"""Build tiny random checkpoints so tests run offline, in seconds, with NumPy only."""

import json
import os

import numpy as np

from dnn.safetensors_np import save

CONFIGS = {
    "qwen2": dict(model_type="qwen2", hidden_size=64, intermediate_size=160, num_hidden_layers=6,
                  num_attention_heads=4, num_key_value_heads=2, vocab_size=300, rms_norm_eps=1e-6,
                  rope_theta=1000000.0, tie_word_embeddings=True, max_position_embeddings=512),
    "qwen3": dict(model_type="qwen3", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=300,
                  rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True,
                  max_position_embeddings=512),
    "llama": dict(model_type="llama", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=8, num_key_value_heads=2, vocab_size=300, rms_norm_eps=1e-5,
                  rope_theta=500000.0, tie_word_embeddings=False, max_position_embeddings=512,
                  rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                "high_freq_factor": 4.0, "original_max_position_embeddings": 64}),
}


def make_model(path, kind="qwen2", seed=0):
    cfg = dict(CONFIGS[kind])
    rng = np.random.default_rng(seed)
    h, inter, nh, nkv = (cfg["hidden_size"], cfg["intermediate_size"],
                         cfg["num_attention_heads"], cfg["num_key_value_heads"])
    hd = cfg.get("head_dim", h // nh)

    def w(*shape, scale=0.08):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    t = {"model.embed_tokens.weight": w(cfg["vocab_size"], h, scale=0.5),
         "model.norm.weight": 1 + w(h)}
    if not cfg["tie_word_embeddings"]:
        t["lm_head.weight"] = w(cfg["vocab_size"], h)
    for i in range(cfg["num_hidden_layers"]):
        p = "model.layers.%d." % i
        t[p + "input_layernorm.weight"] = 1 + w(h)
        t[p + "post_attention_layernorm.weight"] = 1 + w(h)
        t[p + "self_attn.q_proj.weight"] = w(nh * hd, h)
        t[p + "self_attn.k_proj.weight"] = w(nkv * hd, h)
        t[p + "self_attn.v_proj.weight"] = w(nkv * hd, h)
        t[p + "self_attn.o_proj.weight"] = w(h, nh * hd)
        if kind == "qwen2":
            t[p + "self_attn.q_proj.bias"] = w(nh * hd)
            t[p + "self_attn.k_proj.bias"] = w(nkv * hd)
            t[p + "self_attn.v_proj.bias"] = w(nkv * hd)
        if kind == "qwen3":
            t[p + "self_attn.q_norm.weight"] = 1 + w(hd)
            t[p + "self_attn.k_norm.weight"] = 1 + w(hd)
        t[p + "mlp.gate_proj.weight"] = w(inter, h)
        t[p + "mlp.up_proj.weight"] = w(inter, h)
        t[p + "mlp.down_proj.weight"] = w(h, inter)
    os.makedirs(path, exist_ok=True)
    save(os.path.join(path, "model.safetensors"), t)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    return path
