# Wire protocol (version 1)

Write a node in any language (C on a microcontroller, Rust, Kotlin, ...) by
implementing this. The Python reference is `dnn/wire.py` and `dnn/node.py`.

## Framing

Every message is one frame over TCP (set `TCP_NODELAY`):

| Bytes | Field |
|---|---|
| 4 | magic `DNN1` |
| 4 | header length H, uint32 **big-endian** |
| 8 | payload length P, uint64 **big-endian** |
| H | UTF-8 JSON header (an object with a `"type"` field) |
| P | payload: tensors back to back, **little-endian**, C order |

`header.tensors` lists the payload's tensors:
`{"name", "dtype": "float32"|"float16"|"int8"|"uint8"|"int32"|"int64"|"uint16", "shape": [...], "offset", "nbytes"}`.

## Connections

- **Control:** the coordinator sends a request and the node replies on the same connection.
- **Data:** `forward` frames get no reply on their connection. The node sends its
  output onward by opening (and reusing) a connection to the next address.

If a request fails, the node replies `{"type": "error", "node": name, "message": str}`.

## Messages a node must handle

| Request | Reply / action |
|---|---|
| `hello` | `{"type":"hello","name":str,"protocol":1,"version":str}` |
| `ping` | `{"type":"pong","t":<echo>}` (used to measure round-trip time) |
| `probe` | `{"type":"probe","info":{...}}`, where info has at least `mem_available` (bytes it can offer), `params_per_s` (`{"float32":w/s,"float16":w/s,"int8":w/s}`, weights per second through `y = x @ W.T`), `os`, `name` |
| `load` `{config, layers:[start,end), dtype}` | Free previous layers and sessions, store the Hugging Face `config` (JSON object). Reply `{"type":"ok"}` |
| `weights` `{layer}` + tensors | Store layer `layer`. Reply `{"type":"ok"}`. Tensors are listed below. |
| `forward` `{req, session, pos, route, reply_to, timings}` + tensor `x` | Run `x` (float32 `[T, hidden]`, positions `pos..pos+T-1`) through your layers using that session's KV cache. Append `[name, ms]` to `timings`. If `route` isn't empty, pop its first `[host,port]` and send the same frame there with the shorter route. Otherwise send `{"type":"result","req","timings"}` + tensor `x` to `reply_to`. On failure send `{"type":"error","req",...}` to `reply_to`. |
| `end_session` `{session}` | Drop that session's KV cache, reply `ok` |
| `status` | Optional: `{"type":"status", ...}` |

`pos` lower than the cached length means "rewind to `pos`" (truncate the cache first).

## Layer tensors (`weights` message)

Matrices are `[out_features, in_features]`. With dtype `int8` each matrix `M` comes
with `M.s` (float32, one scale per output row): `W = M.s[:, None] * M`. With
`float16`/`float32` the matrix comes as-is. Norm weights and biases are always float32.

| Name | Shape | Present for |
|---|---|---|
| `in_norm`, `post_norm` | `[hidden]` | all |
| `q`, `k`, `v` | `[heads*hd, hidden]`, `[kv*hd, hidden]` ×2 | all |
| `q_b`, `k_b`, `v_b` | bias vectors | Qwen2/2.5 |
| `q_norm`, `k_norm` | `[hd]` (RMSNorm per head, before RoPE) | Qwen3 |
| `o` | `[hidden, heads*hd]` | all |
| `gate`, `up` | `[inter, hidden]` | all |
| `down` | `[hidden, inter]` | all |

## Layer math (per token row, float32)

```
h   = rmsnorm(x, in_norm)                 rmsnorm(v,w) = v / sqrt(mean(v²) + eps) * w
q,k,v = h@q.T+q_b, h@k.T+k_b, h@v.T+v_b    reshape to heads (q: heads, k/v: kv heads)
q,k = rmsnorm per head (Qwen3 only)
q,k = rope(q), rope(k)                     rotate-half RoPE, theta = config rope_theta
                                           (or rope_parameters.rope_theta), llama3 scaling if set
append k,v to cache; query head i uses kv head i // (heads/kv)
a   = softmax(q·kᵀ / sqrt(hd), causal) · v
x   = x + a@o.T
h   = rmsnorm(x, post_norm)
x   = x + (silu(h@gate.T) * (h@up.T)) @ down.T
```
`eps = rms_norm_eps`, `hd = head_dim` (or `hidden/heads`). Check your
implementation against the Python node with `tests/test_pipeline.py`'s approach:
same tiny model, same inputs, outputs within 1e-4.
