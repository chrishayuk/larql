"""A stand-in for `larql vindex3 exec … --logit-dump` used by the V0 tests.

It writes what the real binary writes (a `[positions, vocab]` f32 logit
dump, `<dump>.meta.json` and, with `--capture-sites`, an E0a capture
directory) for a tiny synthetic geometry. What it gets wrong is chosen by a
sidecar `<this file>.behaviour.json`, because V0 runs the binary under a
fixed environment and a test cannot steer it through variables.

The test copies this file to a path of its own and writes the shebang, so
each copy has its own sha256 (a "historical" and an "experimental" build).
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HIDDEN, Q_HEADS, KV_HEADS, HEAD_DIM = 8, 2, 1, 4
LAYERLESS = ("embedding", "head_input")
LAYER_SITES = ("attention_input", "query_unpositioned", "key_unpositioned", "query_positioned",
               "key_positioned", "attention_output", "attention_delta", "ffn_input", "ffn_output",
               "ffn_delta", "residual")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def flag(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def width(site, b):
    head_dim = b.get("head_dim", HEAD_DIM)
    if site.startswith("query"):
        return Q_HEADS * head_dim
    if site.startswith("key"):
        return KV_HEADS * head_dim
    return b.get("hidden", HIDDEN)


def write_capture(cap: Path, argv, tokens, logits_path: Path, b: dict, me: Path):
    cap.mkdir()
    positions = [int(p) for p in flag(argv, "--capture-positions").split(",")]
    container = Path(argv[3])
    layers = b.get("layers", 2)
    rng = np.random.default_rng(7)
    files = []
    planes = [(s, None) for s in LAYERLESS] + [(s, L) for L in range(layers) for s in LAYER_SITES]
    if b.get("drop_plane"):
        planes = planes[:-1]
    if b.get("extra_plane"):
        planes.append(("residual", layers))
    for i, (site, layer) in enumerate(planes):
        name = f"{site}.f32" if layer is None else f"{site}.L{layer:03}.f32"
        w = width(site, b) + (1 if b.get("wrong_width") and i == 2 else 0)
        values = rng.standard_normal((len(positions), w)).astype("<f4")
        if b.get("nonfinite_plane") and i == 3:
            values[0, 0] = np.nan
        data = values.tobytes()
        if b.get("plane_escape") and i == 0:
            name = "../escaped.f32"
        (cap / name).write_bytes(data)
        if b.get("plane_tamper") and i == 1:
            (cap / name).write_bytes(data[:-4] + b"\0\0\0\0")
        files.append({"site": site, "layer": layer, "rows": len(positions), "width": w,
                      "file": name, "sha256": sha(data)})
    environment = [{"name": k, "value": v} for k, v in sorted(os.environ.items())
                   if k.startswith("LARQL_") or k == "RAYON_NUM_THREADS"]
    environment += [{"name": k, "value": v} for k, v in b.get("extra_env", {}).items()]
    identity = {
        "build_commit": None,
        "binary_sha256": sha(me.read_bytes()),
        "container": str(container),
        "container_index_sha256": sha((container / "index.json").read_bytes()),
        "component": "target",
        "engine": "fake",
        "backend": "Reference" if flag(argv, "--backend") == "reference" else "Production",
        "representation_source": flag(argv, "--representation-source", "auto"),
        "tokens_sha256": sha(np.asarray(tokens, dtype="<u4").tobytes()),
        "rope_angles": "larql",
        "draft_depth": None,
        "environment": environment,
        "platform": "macos-aarch64",
    }
    identity.update(b.get("identity_override", {}))
    manifest = {
        "schema": "larql.site-capture/1",
        "identity": identity,
        "logit_dump": str(logits_path),
        "logit_dump_sha256": sha(logits_path.read_bytes()),
        "positions": positions,
        "sequence_len": len(tokens),
        "hidden": b.get("hidden", HIDDEN),
        "layers": [{"layer": L, "num_q_heads": Q_HEADS, "num_kv_heads": KV_HEADS,
                    "head_dim": b.get("head_dim", HEAD_DIM)}
                   for L in range(layers)],
        "files": files,
    }
    raw = json.dumps(manifest).encode()
    if b.get("malformed_manifest"):
        raw = raw[: len(raw) // 2]
    (cap / "manifest.json").write_bytes(raw)
    digest = sha(raw)
    if b.get("manifest_tamper"):
        digest = "0" * 64
    result = {"status": b.get("capture_status", "complete"),
              "detail": {"manifest_sha256": digest}, "identity": identity}
    (cap / "capture.result.json").write_text(json.dumps(result))


def main() -> int:
    me = Path(sys.argv[0]).resolve()
    sidecar = me.with_name(me.name + ".behaviour.json")
    b = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    argv = sys.argv
    if b.get("sleep"):
        time.sleep(b["sleep"])
    if b.get("exit"):
        return b["exit"]
    tokens = [int(t) for t in flag(argv, "--tokens").split(",")]
    vocab = b.get("vocab", 16)
    out = Path(flag(argv, "--logit-dump"))
    logits = (np.arange(len(tokens) * vocab, dtype="<f4").reshape(len(tokens), vocab) / 7.0)
    if flag(argv, "--backend") == "production":
        logits = logits + np.float32(0.25)
    if "differ_at" in b:
        p, v = b["differ_at"]
        logits[p, v] = np.nextafter(logits[p, v], np.float32(np.inf))
    if b.get("capture_perturbs") and "--capture-sites" in argv:
        logits[-1, -1] += np.float32(1.0)
    data = logits.astype("<f4").tobytes()
    if b.get("short_logits"):
        data = data[:-4]
    out.write_bytes(data)
    if not b.get("no_meta"):
        meta = {"engine": "fake", "positions": len(tokens), "vocab": vocab}
        if b.get("rope_angles", "larql") is not None:
            meta["rope_angles"] = b.get("rope_angles", "larql")
        out.with_name(out.stem + ".meta.json").write_text(json.dumps(meta))
    if b.get("allocate_bytes"):
        bytearray(b["allocate_bytes"])  # touched, so it counts toward peak RSS
    if "--capture-sites" in argv:
        write_capture(Path(flag(argv, "--capture-sites")), argv, tokens, out, b, me)
    return 0


if __name__ == "__main__":
    sys.exit(main())
