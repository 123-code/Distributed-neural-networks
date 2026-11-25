"""Decide which device holds which layers, and in what number format, from
measured physics.

For one conversation, every token passes through every layer in order, so:

    time/token ~= sum over devices (layers_d * seconds_per_layer_d)
                + sum over network hops (one-way latency)

Constraint: a device holds only as many layers as fit in its free memory
(weights + KV cache for the context length).

Number format ("dtype") per device: int8 takes 1 byte per weight, float16 2,
float32 4. Which is *fastest* depends on the hardware (NumPy on x86 must widen
int8/float16 to float32 in software, so float32 is often several times faster
there if it fits). With dtype "auto" each device gets the fastest format that
still fits its share; otherwise every device uses the format given.

Strategies:
  spread   use every device that can hold at least one layer, with shares
           proportional to speed (the point of a cluster; best throughput
           when several conversations run at once).
  fastest  minimise the estimate above: fill the fastest devices first and add
           a device only when that lowers time/token (usually only when
           memory forces it).
"""

import math

from .model import BYTES_PER_PARAM, STORAGE_DTYPES
from .sysinfo import fmt_bytes

RUNTIME_RESERVE = 96 * 1024 * 1024   # Python + NumPy + buffers on each device
USABLE_FRACTION = 0.85               # don't push the OS into swapping / killing us
DTYPE_CHOICES = ("auto",) + STORAGE_DTYPES


class PlanError(Exception):
    pass


class Device:
    def __init__(self, name, info, addr=None, rtt=0.0, is_local=False, holds_layers=True):
        self.name = name
        self.info = info
        self.addr = addr            # (host, port) for remote nodes, None for the coordinator
        self.rtt = rtt              # seconds, round trip from the coordinator
        self.is_local = is_local
        self.holds_layers = holds_layers

    @property
    def mem(self):
        return int(self.info["mem_available"])

    def speed(self, dtype):
        return float(self.info["params_per_s"][dtype])

    def layer_seconds(self, cfg, dtype):
        """Time for one layer, one token: stream its weights + fixed per-layer cost."""
        return cfg.layer_params() / self.speed(dtype) + float(self.info.get("layer_overhead_s", 0.0))


class Plan:
    def __init__(self, cfg, dtype, strategy, max_context):
        self.cfg, self.dtype, self.strategy, self.max_context = cfg, dtype, strategy, max_context
        self.stages = []      # [(device, start, end)] in ring order, local stage first
        self.dtypes = {}      # device -> storage dtype
        self.excluded = []    # [(device, reason)]
        self.rows = []        # per-device report rows
        self.est_s = 0.0
        self.head_s = 0.0

    def dtype_of(self, device):
        return self.dtypes[device]

    def remote_stages(self):
        return [s for s in self.stages if not s[0].is_local]

    def local_stage(self):
        for s in self.stages:
            if s[0].is_local:
                return s
        return None

    def report(self):
        lines = ["model: %s, %d layers of %s weights each, context %d tokens"
                 % (self.cfg.model_type, self.cfg.n_layers,
                    "{:,}".format(self.cfg.layer_params()), self.max_context),
                 "strategy: %s, dtype: %s" % (self.strategy, self.dtype), ""]
        hdr = "%-18s %-8s %10s %-8s %9s %7s %8s %10s" % ("device", "os", "usable", "dtype",
                                                          "Mparam/s", "rtt ms", "layers", "est ms/tok")
        lines += [hdr, "-" * len(hdr)]
        for r in self.rows:
            lines.append("%-18s %-8s %10s %-8s %9.0f %7.1f %8s %10s" % r)
        lines.append("")
        lines.append("estimated time per token (one conversation): %.1f ms (~%.1f tokens/s), incl. "
                     "%.1f ms output projection. Rough lower bound; real timings print after each reply."
                     % (self.est_s * 1000, 1.0 / self.est_s if self.est_s else float("inf"),
                        self.head_s * 1000))
        for dev, why in self.excluded:
            lines.append("not used: %s - %s" % (dev.name, why))
        return "\n".join(lines)


def layer_footprint(cfg, dtype, max_context):
    """Bytes one layer occupies on a device: weights + scales + KV cache + slack."""
    weights = cfg.layer_params() * BYTES_PER_PARAM[dtype]
    if dtype == "int8":
        weights += 4 * (cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv * cfg.head_dim
                        + cfg.hidden + 2 * cfg.inter + cfg.hidden)
    kv = cfg.kv_bytes_per_token_per_layer() * max_context
    return int((weights + kv) * 1.05)


def head_footprint(cfg, dtype):
    return int(cfg.head_params() * BYTES_PER_PARAM[dtype] * 1.05)


def usable_bytes(dev, cfg, dtype):
    usable = dev.mem * USABLE_FRACTION - RUNTIME_RESERVE
    if dev.is_local:
        usable -= head_footprint(cfg, dtype)
    return usable


def capacity(dev, cfg, dtype, max_context):
    """(layers that fit, usable bytes) for this device storing weights as `dtype`."""
    usable = usable_bytes(dev, cfg, dtype)
    if not dev.holds_layers or usable <= 0:
        return 0, max(0, usable)
    return int(usable // layer_footprint(cfg, dtype, max_context)), usable


def _layer_s(devices, cfg, dtypes):
    return {d: d.layer_seconds(cfg, dtypes[d]) for d in devices}


def _head_s(devices, cfg, dtypes):
    """The coordinator's output projection runs every token (vocab x hidden weights)."""
    local = [d for d in devices if d.is_local]
    return cfg.vocab * cfg.hidden / local[0].speed(dtypes[local[0]]) if local else 0.0


def _estimate(stages, layer_s, head_s=0.0):
    compute = sum((e - s) * layer_s[d] for d, s, e in stages)
    remote = [d for d, s, e in stages if not d.is_local and e > s]
    network = sum(d.rtt / 2 for d in remote) + (remote[-1].rtt / 2 if remote else 0.0)
    return compute + network + head_s


def _integerize(shares, caps, total):
    alloc = {d: min(caps[d], int(math.floor(v))) for d, v in shares.items()}
    left = total - sum(alloc.values())
    order = sorted(shares, key=lambda d: shares[d] - math.floor(shares[d]), reverse=True)
    while left > 0:
        progressed = False
        for d in order:
            if left and alloc[d] < caps[d]:
                alloc[d] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def _spread(devs, caps, layer_s, n_layers):
    speeds = {d: 1.0 / layer_s[d] for d in devs}  # layers per second
    alloc = {d: 0.0 for d in devs}
    active, remaining = set(devs), float(n_layers)
    while remaining > 1e-9 and active:
        total = sum(speeds[d] for d in active)
        share = {d: remaining * speeds[d] / total for d in active}
        saturated = [d for d in active if alloc[d] + share[d] >= caps[d]]
        if not saturated:
            for d in active:
                alloc[d] += share[d]
            remaining = 0.0
        else:
            for d in saturated:
                remaining -= caps[d] - alloc[d]
                alloc[d] = float(caps[d])
                active.discard(d)
    counts = _integerize(alloc, caps, n_layers)
    if n_layers >= len(devs):  # every capable device takes part
        for d in devs:
            if counts[d] == 0:
                donor = max((x for x in devs if counts[x] > 1), key=lambda x: counts[x] / speeds[x])
                counts[donor] -= 1
                counts[d] = 1
    return counts


def _fastest(devs, caps, layer_s, n_layers):
    ranked = sorted(devs, key=lambda d: (layer_s[d], not d.is_local))
    best = None
    for k in range(1, len(ranked) + 1):
        chosen = ranked[:k]
        if sum(caps[d] for d in chosen) < n_layers:
            continue
        counts, left = {}, n_layers
        for d in chosen:
            counts[d] = min(caps[d], left)
            left -= counts[d]
        est = _estimate(_order(devs, counts), layer_s)
        if best is None or est < best[0]:
            best = (est, counts)
    return best[1]


def _order(devs, counts):
    """Ring order: the coordinator's own stage first (no hop), then remote nodes
    in the order they were listed. Returns [(device, start, end)]."""
    ordered = [d for d in devs if d.is_local] + [d for d in devs if not d.is_local]
    stages, start = [], 0
    for d in ordered:
        n = counts.get(d, 0)
        if n:
            stages.append((d, start, start + n))
            start += n
    return stages


def _allocate(devices, dtypes, cfg, strategy, max_context):
    """Assign layer counts for a fixed dtype per device. Returns (counts, caps) or
    raises PlanError if everything together can't hold the model."""
    caps = {d: capacity(d, cfg, dtypes[d], max_context)[0] for d in devices}
    layer_s = _layer_s(devices, cfg, dtypes)
    capable = [d for d in devices if caps[d] >= 1]
    if sum(caps[d] for d in capable) < cfg.n_layers:
        raise PlanError("not enough memory")
    if strategy == "spread":
        return _spread(capable, caps, layer_s, cfg.n_layers), caps
    return _fastest(capable, caps, layer_s, cfg.n_layers), caps


def _fastest_fitting(dev, cfg, n, max_context):
    """Fastest dtype in which `dev` can hold n layers (and the head, if local)."""
    if not dev.holds_layers:
        ok = [t for t in STORAGE_DTYPES if usable_bytes(dev, cfg, t) > 0]
    else:
        ok = [t for t in STORAGE_DTYPES if capacity(dev, cfg, t, max_context)[0] >= max(n, 1)]
    return max(ok, key=lambda t: dev.speed(t)) if ok else "int8"


def make_plan(cfg, devices, dtype="auto", strategy="spread", max_context=2048):
    if strategy not in ("spread", "fastest"):
        raise PlanError("unknown strategy %r" % strategy)
    if dtype not in DTYPE_CHOICES:
        raise PlanError("unknown dtype %r" % dtype)
    plan = Plan(cfg, dtype, strategy, max_context)
    base = "int8" if dtype == "auto" else dtype   # int8 = most capacity

    local = [d for d in devices if d.is_local]
    if local and usable_bytes(local[0], cfg, base) <= 0:
        hint = "use --dtype int8 or " if base != "int8" else ""
        raise PlanError("the coordinator needs %s free for the embedding/output head (%s) "
                        "and has %s usable; %srun the coordinator on a machine with more RAM"
                        % (fmt_bytes(head_footprint(cfg, base) + RUNTIME_RESERVE),
                           base, fmt_bytes(local[0].mem * USABLE_FRACTION), hint))

    dtypes = {d: base for d in devices}
    try:
        counts, caps = _allocate(devices, dtypes, cfg, strategy, max_context)
    except PlanError:
        per_layer = layer_footprint(cfg, base, max_context)
        lines = ["the devices can hold %d of %d layers (%s per layer as %s with %d tokens of context)."
                 % (sum(capacity(d, cfg, base, max_context)[0] for d in devices), cfg.n_layers,
                    fmt_bytes(per_layer), base, max_context)]
        total = 0
        for d in devices:
            n, u = capacity(d, cfg, base, max_context)
            total += n
            lines.append("  %-18s usable %10s -> %d layers" % (d.name, fmt_bytes(u), n))
        lines.append("Short by about %s. Options: %sa smaller --max-context, a smaller model, "
                     "or another device." % (fmt_bytes((cfg.n_layers - total) * per_layer),
                                             "--dtype int8, " if base != "int8" else ""))
        raise PlanError("\n".join(lines))

    if dtype == "auto":
        # Upgrade each device to the fastest format that still fits its share, then
        # re-balance with the new speeds; keep the result only if it is faster.
        best_est = _estimate(_order(devices, counts), _layer_s(devices, cfg, dtypes),
                             _head_s(devices, cfg, dtypes))
        for _ in range(4):
            new = {d: _fastest_fitting(d, cfg, counts.get(d, 0), max_context) for d in devices}
            if new == dtypes:
                break
            try:
                c2, caps2 = _allocate(devices, new, cfg, strategy, max_context)
            except PlanError:
                break
            if strategy == "spread" and set(d for d in c2 if c2[d]) < set(d for d in counts if counts[d]):
                break  # never drop a device from a spread plan to gain speed
            est = _estimate(_order(devices, c2), _layer_s(devices, cfg, new), _head_s(devices, cfg, new))
            if est >= best_est:
                break
            counts, caps, dtypes, best_est = c2, caps2, new, est

    plan.dtypes = dtypes
    plan.stages = _order(devices, counts)
    speeds = {d: d.speed(dtypes[d]) for d in devices}
    layer_s = _layer_s(devices, cfg, dtypes)
    plan.head_s = _head_s(devices, cfg, dtypes)
    plan.est_s = _estimate(plan.stages, layer_s, plan.head_s)
    for d in devices:
        n = counts.get(d, 0)
        _, usable = capacity(d, cfg, dtypes[d], max_context)
        if not d.holds_layers:
            plan.excluded.append((d, "holds only the embedding/output head (--no-local)"))
        elif capacity(d, cfg, "int8", max_context)[0] < 1:
            plan.excluded.append((d, "can't hold one layer: needs %s even as int8, has %s usable"
                                  % (fmt_bytes(layer_footprint(cfg, "int8", max_context)),
                                     fmt_bytes(max(0, usable_bytes(d, cfg, "int8"))))))
        elif not n:
            plan.excluded.append((d, "not needed: the 'fastest' strategy is quicker without it "
                                  "(use --strategy spread to include it)"))
        r = next(((s, e) for dd, s, e in plan.stages if dd is d), None)
        plan.rows.append((d.name[:16] + (" *" if d.is_local else ""), d.info.get("os", "?")[:8],
                          fmt_bytes(usable), dtypes[d] if (n or d.is_local) else "-",
                          speeds[d] / 1e6, d.rtt * 1000, "%d-%d" % (r[0], r[1] - 1) if r else "-",
                          "%.1f" % ((n * layer_s[d] + (plan.head_s if d.is_local else 0)) * 1000)))
    return plan
