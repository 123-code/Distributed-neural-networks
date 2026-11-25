"""Fetch a model from the Hugging Face Hub with only the standard library
(honours HTTPS_PROXY, HF_ENDPOINT and HF_TOKEN)."""

import json
import os
import sys
import urllib.error
import urllib.request

REQUIRED = ["config.json"]
OPTIONAL = ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]


def cache_root():
    return os.environ.get("DNN_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "dnn"))


def _fetch(url, dest, quiet=False):
    req = urllib.request.Request(url, headers={"User-Agent": "dnn/0.2"})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and not quiet:
                sys.stderr.write("\r  %s %5.1f%% of %.0f MB" % (os.path.basename(dest), 100 * done / total,
                                                               total / 2 ** 20))
        if total and not quiet:
            sys.stderr.write("\n")
    os.replace(tmp, dest)


def resolve_model(spec, revision="main"):
    """A local directory, or a Hub repo id like 'Qwen/Qwen2.5-0.5B-Instruct'."""
    path = os.path.expanduser(spec)
    if os.path.isdir(path):
        return path
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    dest = os.path.join(cache_root(), spec.replace("/", "--"))
    os.makedirs(dest, exist_ok=True)

    def get(name, required):
        out = os.path.join(dest, name)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
        url = "%s/%s/resolve/%s/%s" % (endpoint, spec, revision, name)
        try:
            _fetch(url, out)
            return out
        except urllib.error.HTTPError as e:
            if required:
                raise SystemExit("download failed for %s: HTTP %d (private/gated model? set HF_TOKEN)"
                                 % (url, e.code))
            return None
        except urllib.error.URLError as e:
            raise SystemExit("can't reach %s (%s). Download the model elsewhere and pass "
                             "its directory to --model." % (endpoint, e.reason))

    print("fetching %s into %s" % (spec, dest), file=sys.stderr)
    for n in REQUIRED:
        get(n, True)
    for n in OPTIONAL:
        get(n, False)
    index = get("model.safetensors.index.json", False)
    if index:
        with open(index) as f:
            for shard in sorted(set(json.load(f)["weight_map"].values())):
                get(shard, True)
    else:
        get("model.safetensors", True)
    return dest
