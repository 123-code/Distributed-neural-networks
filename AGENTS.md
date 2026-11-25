# AGENTS.md — operating this framework (for AI agents)

You are operating **dnn**: it runs one language model (Qwen2/2.5/3, Llama, Mistral
style) split by layers across whatever devices a person owns: laptops, a Jetson,
a Fire TV, old phones, Raspberry Pis, and so on. Read this whole file before you act.
Human-oriented docs: `README.md`. Wire format for non-Python nodes: `PROTOCOL.md`.

## 1. Mental model (physics first)

Every generated token must pass through every layer in order. Per token, the
time is roughly **Σ_devices (weights held ÷ device speed) + Σ_hops (network latency)**.

- **Weights stay put; activations move.** Between layers a token is
  `hidden_size` floats (~3.5 KB for Qwen2.5-0.5B). The network moves KB per token;
  each device reads its own weights (MB–GB) from its own RAM every token.
- **A device qualifies when it can hold at least one layer** in free RAM (weights +
  KV cache for the context). The planner computes this and says why it leaves out
  any device. A device whose program is fixed at the factory (a microwave's
  controller, for example) or that has only KBs of RAM can't be a node. Don't try.
- **Splitting doesn't make one conversation faster** if the model fits on the
  fastest device. It pays off when (a) the model is bigger than any single device's
  free RAM, or (b) you want every device to take part (`--strategy spread`, the default).
  `--strategy fastest` minimises latency and may use only one device.
- Latency comes from Wi-Fi and OS stacks, not distance. Prefer Ethernet or 5 GHz
  Wi-Fi. A running microwave oven disturbs 2.4 GHz Wi-Fi.

## 2. Repository map

| Path | Role |
|---|---|
| `dnn/node.py` | Node server (`python -m dnn.node`). Needs only Python 3 + NumPy. Holds assigned layers and KV caches; passes activations along the ring. |
| `dnn/run.py` | Coordinator CLI (`python -m dnn.run`): download model, measure devices, plan, deploy, chat. |
| `dnn/cluster.py` | Coordinator engine: node connections, probing, deploying weights, ring forward, result listener. |
| `dnn/planner.py` | Layer assignment from measured free memory and speed. |
| `dnn/model.py` | Transformer math in NumPy/CuPy: RMSNorm, RoPE (incl. llama3 scaling), GQA attention, SwiGLU, KV cache, int8/float16 storage. |
| `dnn/sysinfo.py` | Device probe: free RAM (Linux/Android/macOS/Windows), weights/s per dtype. |
| `dnn/wire.py` | Framed TCP protocol (JSON header + raw tensors). No pickle: safe to parse. |
| `dnn/safetensors_np.py` | Memory-mapped safetensors reader (BF16/F16/F32) and writer. |
| `dnn/download.py` | Hugging Face Hub download with the standard library (`HF_TOKEN`, `HF_ENDPOINT`, proxies). |
| `scripts/bootstrap_node.sh` | Installs NumPy (Termux/apt/pip/macOS) and starts a node. |
| `scripts/build_node_pyz.py` | Builds `dist/dnn-node.pyz`, a single file that runs a node on any Python 3 + NumPy. |
| `tests/` | `test_pipeline.py` (NumPy only), `test_cli.py` (needs `tokenizers`), `test_reference.py` (needs `torch` + `transformers`; compares logits with Hugging Face). |

## 3. Verify the code before anything else

```sh
pip install numpy tokenizers            # minimum for the full offline suite
python -m unittest discover -s tests -v
```
Expected: all tests pass. `test_reference` is skipped without torch/transformers;
install them (`pip install torch transformers`) when you change `dnn/model.py`,
and don't ship math changes until it passes. Tests use tiny random models: no
network and no model download needed.

## 4. Run on one machine (sanity check)

```sh
pip install numpy tokenizers
python -m dnn.run --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Say hi in five words."
```
The first run downloads into `~/.cache/dnn/` (override with `DNN_CACHE`). If the Hub
can't be reached, download the repo's `config.json`, `tokenizer.json`,
`generation_config.json` and `*.safetensors` (+ index) elsewhere and pass the
directory to `--model`.

## 5. Run a cluster

Roles: one **coordinator** (the machine with the most RAM, usually the laptop; it
holds the embedding/output head and the tokenizer, and also takes layers unless
`--no-local`) plus any number of **nodes**.

### 5.1 Start a node on each device

Every node needs: Python ≥ 3.8, NumPy, the `dnn/` folder (or `dist/dnn-node.pyz`),
and an open TCP port (default 7070) reachable from the coordinator and from the
other nodes. Nodes download nothing; the coordinator streams each one its layers.

The easiest way to get the code onto any device is to serve it from the coordinator:
```sh
python scripts/build_node_pyz.py && python -m http.server 8000 -d dist   # on the coordinator
curl -O http://<coordinator-ip>:8000/dnn-node.pyz                          # on the device
python3 dnn-node.pyz --port 7070 --name <label>                            # on the device
```
Each node prints the exact `host:port` to give the coordinator.

Per device type:

- **Linux / macOS / Windows / Raspberry Pi**: `pip install numpy`, then
  `python -m dnn.node --name <label>` or `sh scripts/bootstrap_node.sh --name <label>`.
- **NVIDIA Jetson (e.g. Orin Nano)**: same as Linux and it works on the CPU. For the
  GPU, install CuPy for your JetPack's CUDA (e.g. `pip install cupy-cuda12x` on
  JetPack 6) and add `--backend cupy`. Compare `--probe` output with and without it
  and keep the faster one. Use a high power mode (`sudo nvpmodel -m 0`; check the
  modes your board supports).
- **Android phones/tablets, Fire TV, Android TV boxes**: use **Termux**.
  1. Fire TV only: Settings → My Fire TV → Developer options → turn on *ADB debugging*
     and *Apps from Unknown Sources* (on newer Fire OS, allow unknown apps per app).
     If Developer options is hidden: Settings → My Fire TV → About → select the device
     name 7 times.
  2. Install Termux from F-Droid or its GitHub releases (current builds need Android 7+;
     Fire OS 6+ qualifies; if unsure about the CPU architecture pick the universal APK).
     From the coordinator: `adb connect <device-ip>:5555 && adb install termux.apk`.
  3. Open Termux once on the device. Typing with a TV remote is slow: from the
     coordinator, `adb shell input text "pkg%sinstall%s-y%spython%spython-numpy%scurl"`
     followed by `adb shell input keyevent 66` (Enter) types into the focused app.
     Or set up SSH once (`pkg install openssh && passwd && sshd`, then
     `ssh -p 8022 <device-ip>` from the coordinator).
  4. In Termux: `pkg install python python-numpy` (don't `pip install numpy`; it
     would compile from source), fetch `dnn-node.pyz` as above, run
     `termux-wake-lock` so the node keeps running when the screen sleeps, then
     `python dnn-node.pyz --name firetv --memory-limit 600`.
  5. Leave Termux in the foreground. Android may kill background apps that use
     a lot of memory. If the node dies during loading, lower `--memory-limit`.
- **iOS**: possible only through apps that bundle Python + NumPy (e.g. a-Shell,
  Pythonista); slow and memory-limited. Treat as experimental.
- **Devices without Python** (microcontrollers like ESP32-S3 with PSRAM, custom
  firmware): implement `PROTOCOL.md` natively. Rule of thumb: a node must hold at
  least one layer (`layer_params × bytes per weight + KV cache`, about 15 MB at
  int8 for Qwen2.5-0.5B).

### 5.2 Plan, then run

```sh
# 1) Always look at the plan first. Measures every device; changes nothing.
python -m dnn.run --model Qwen/Qwen2.5-0.5B-Instruct \
    --nodes 192.168.1.20:7070,192.168.1.31:7070 --plan-only

# 2) Run (interactive chat; /reset, /quit). Add --prompt "..." for one-shot.
python -m dnn.run --model Qwen/Qwen2.5-0.5B-Instruct \
    --nodes 192.168.1.20:7070,192.168.1.31:7070
```
List nodes by **LAN IP**, never `localhost`, when several machines are involved:
nodes connect to each other using exactly these addresses. After every reply the
coordinator prints tokens/s, the plan's estimate, and per-node compute time for the
last token. Use them to judge whether the network or a device is the bottleneck.

### 5.3 Knobs

| Flag | Use |
|---|---|
| `--dtype auto` (default) / `int8` / `float16` / `float32` | How weights are stored. `auto` gives each device the fastest format (measured by its probe) that still fits its share. A fixed value applies to every device: int8 fits the most (1 byte/weight), float32 is exact but 4× larger. On x86 NumPy, float32 is usually several times faster than int8/float16 (no software widening), so `auto` picks it where RAM allows. |
| `--strategy spread` (default) / `fastest` | Include every capable device / minimise single-conversation latency. |
| `--max-context N` | Tokens of KV cache reserved per layer; lower it to fit smaller devices. |
| `--memory-limit MB` (node and coordinator) | Cap what a device offers (e.g. leave room for a TV's own apps). |
| `--no-local` | Coordinator keeps only the embedding/head (use when the coordinator is weak). |
| `--listen-port P` | Fixed port where the last node sends results back. Open it on the coordinator's firewall. |
| `--backend cupy` | GPU arrays on NVIDIA hardware (coordinator or node). |

### 5.4 Choosing a model

Pick by total free RAM (int8 weights ≈ parameter count in bytes, plus KV cache):
`Qwen/Qwen2.5-0.5B-Instruct` (~0.5 GB, any setup), `Qwen/Qwen2.5-1.5B-Instruct`
(~1.6 GB), `Qwen/Qwen2.5-3B-Instruct` (~3.2 GB), `Qwen/Qwen2.5-7B-Instruct`
(~7.7 GB; this is where splitting across a laptop + Jetson starts to be necessary).
Gated models (e.g. Llama) need `HF_TOKEN`. The coordinator needs RAM for the
embedding/head: `vocab × hidden` bytes at int8 (e.g. ~136 MB for Qwen2.5-0.5B).

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `can't reach node host:port` | Node not running, wrong IP, firewall, or the Wi-Fi network isolates clients (guest networks, "AP isolation"). Test with `nc -vz host port`. |
| `no result within Ns` | A node can't reach the next node or the coordinator's result port. Use `--listen-port 7099` and allow it inbound on the coordinator; make sure every node can reach every other node's port. |
| `can't hold one layer` | Device too small for this model even as int8 at this context length. Lower `--max-context`, use a smaller model, or leave it out. |
| `the devices can hold X of Y layers` | Not enough RAM in total. The message says how much is missing. Add a device, lower `--max-context`, or choose a smaller model. |
| Node process killed while loading | Its OS ran out of memory (common on Android). Lower that node's `--memory-limit`. |
| Slow tokens | Check the per-node times printed after each reply. If they're small but tokens/s is low, the network is the bottleneck: use Ethernet/5 GHz, fewer hops, or `--strategy fastest`. |
| Much slower than the plan's estimate | The estimate is a lower bound (weight streaming + network). If you run several nodes on **one** machine for testing, they compete for its cores: give each process `OPENBLAS_NUM_THREADS=<cores ÷ processes>` (`OMP_NUM_THREADS` for other BLAS builds). Real separate devices don't have this problem. |
| `node ... speaks protocol N` | Update the code on that device (rebuild and re-copy `dnn-node.pyz`). |

## 7. Rules for agents

- **Never expose nodes to the internet.** The protocol has no authentication.
  Run it only on a trusted LAN or VPN. It never executes received data, but anyone
  who can connect can load or replace layers.
- **Never tell a person to open mains-powered appliances** (microwaves in particular
  hold lethal high voltage after unplugging) to turn them into nodes.
- Don't commit model weights or `dist/`.
- Keep nodes dependency-free beyond NumPy. Anything heavier (tokenizers, torch)
  belongs on the coordinator or in tests only.
- After changing `dnn/model.py`, `dnn/wire.py` or the planner, run the full test
  suite, and `test_reference` with torch + transformers installed. If you change the
  wire format, bump `PROTOCOL_VERSION` in `dnn/wire.py` and update `PROTOCOL.md`.
- Report results with the real numbers the tools print (plan table, tok/s,
  per-node ms). Don't extrapolate performance you didn't measure.
