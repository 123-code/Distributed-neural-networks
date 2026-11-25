# Distributed neural networks

Run one language model across every device you own that can physically hold
part of it: laptop, Jetson, Fire TV, old phones, Raspberry Pis.

- **Nodes need only Python 3 + NumPy** (a ~30 KB single-file build is available).
  They download nothing: the coordinator streams each node its own layers.
- **The coordinator measures each device** (free RAM, how fast it streams weights,
  network round trip) and assigns layers to fit, explaining any device it leaves out.
- **Weights stay put, activations move.** Each token crosses the network as a few KB.
- Supports Qwen2 / Qwen2.5 / Qwen3, Llama (incl. Llama 3 RoPE scaling) and Mistral
  style checkpoints in safetensors (BF16/F16/F32), stored on devices as int8, float16
  or float32 (by default each device gets the fastest format that fits its share).
  Checked against Hugging Face `transformers`.

## Quick start

```sh
# on every extra device (Linux/macOS/Windows/Termux on Android & Fire TV/Jetson)
pip install numpy            # Termux: pkg install python python-numpy
python -m dnn.node           # prints the host:port to use below

# on the coordinator (your laptop)
pip install numpy tokenizers
python -m dnn.run --model Qwen/Qwen2.5-0.5B-Instruct --nodes 192.168.1.20:7070,192.168.1.31:7070 --plan-only
python -m dnn.run --model Qwen/Qwen2.5-0.5B-Instruct --nodes 192.168.1.20:7070,192.168.1.31:7070
```

Planner output for four **simulated** devices running Qwen2.5-0.5B (the speeds are
made-up but realistic; on real hardware every number comes from measurements):

```
device             os           usable dtype     Mparam/s  rtt ms   layers est ms/tok
-------------------------------------------------------------------------------------
laptop *           linux        6.0 GB float32      20000     0.0     0-16       19.5
jetson             linux        4.9 GB float32       8000     2.0    17-22       11.2
firetv             android    414.0 MB float32        300     5.0    23-23       49.7
old-phone          android         0 B -              150     8.0        -        0.0

estimated time per token (one conversation): 86.4 ms (~11.6 tokens/s), incl. 6.8 ms output projection. Rough lower bound; real timings print after each reply.
not used: old-phone - can't hold one layer: needs 17.1 MB even as int8, has 0 B usable
```

The same devices with `--strategy fastest`:

```
device             os           usable dtype     Mparam/s  rtt ms   layers est ms/tok
-------------------------------------------------------------------------------------
laptop *           linux        6.0 GB float32      20000     0.0     0-23       24.7
jetson             linux        4.9 GB -             8000     2.0        -        0.0
firetv             android    414.0 MB -              300     5.0        -        0.0
old-phone          android         0 B -              150     8.0        -        0.0

estimated time per token (one conversation): 24.7 ms (~40.5 tokens/s), incl. 6.8 ms output projection. Rough lower bound; real timings print after each reply.
not used: jetson - not needed: the 'fastest' strategy is quicker without it (use --strategy spread to include it)
not used: firetv - not needed: the 'fastest' strategy is quicker without it (use --strategy spread to include it)
not used: old-phone - can't hold one layer: needs 17.1 MB even as int8, has 0 B usable
```

The single Fire TV layer costs more time than the 23 layers on the laptop and Jetson
combined. `spread` includes the Fire TV anyway; `fastest` leaves it out because it
doesn't need it.

## How it decides

For one conversation, time per token ≈ Σ (weights on device ÷ device speed) + Σ network
latency. A device joins if it can hold at least one layer (weights + KV cache).
`--strategy spread` (default) uses every such device, proportionally to speed.
`--strategy fastest` minimises latency, which often means one fast device when the
model fits there. Splitting helps when the model doesn't fit in any single device's RAM.

## Docs

- `AGENTS.md`: full operating guide (setup per device type, flags, troubleshooting).
  Written so an AI agent can run the framework end to end.
- `PROTOCOL.md`: wire format, to write nodes in other languages (e.g. microcontrollers).

## Tests

```sh
python -m unittest discover -s tests -v   # numpy (+ tokenizers, + torch/transformers for reference checks)
```
