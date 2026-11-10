"""PREC-DIAG-1 V0: LARQL-only binary-continuity bridge and capture validation.

Spec: `PREC-DIAG-1-V0-bridge.draft.md`. V0 is a binary-continuity bridge,
NOT a reproduction of the 2.1.0 run: matching the historical executable
under today's environment does not show the 2.1.0 environment was
reproduced.

  freeze    write V0.freeze.json (before any run): every identity V0 will
            check, the exact runs, environment and limits.
  run       preflight, then the six runs in order, each under the lifecycle
            in `evidence.py`. A run directory that exists is never re-run.
  readings  byte-level readings only, appended to V0.results.jsonl.
  replace-interrupted
            the one documented replacement of a run left `in_progress` by an
            infrastructure interruption.

Nothing here reads a transformers output, the 2.1.0 per-position numbers,
or an `I_nr*` input. The only numbers computed from logits are byte
equality, digests, sizes and the count of non-finite values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prec_diag import evidence as ev  # noqa: E402

FREEZE_SCHEMA = "larql.prec-diag-1.v0-freeze/1"
BRIDGE_CAVEAT = ("V0 is a binary-continuity bridge, not a reproduction of the 2.1.0 run: matching the "
                 "historical executable under today's environment does not show that the 2.1.0 "
                 "environment was reproduced.")

ARM_FLAGS = {
    "C": ["--backend", "reference"],
    "D": ["--backend", "production", "--representation-source", "stored"],
}
#: The capture identity's backend name and representation source per arm.
ARM_CAPTURE_IDENTITY = {
    "C": {"backend": "Reference", "representation_source": "auto"},
    "D": {"backend": "Production", "representation_source": "stored"},
}
#: (run id suffix, binary key, arm, capture) in execution order.
#: Run ids follow the V0 document: the arm, then the binary's commit.
RUNS = (
    ("V0-C-66ac840e", "historical", "C", False),
    ("V0-C-5c375d43", "experimental", "C", False),
    ("V0-C-5c375d43-cap", "experimental", "C", True),
    ("V0-D-66ac840e", "historical", "D", False),
    ("V0-D-5c375d43", "experimental", "D", False),
    ("V0-D-5c375d43-cap", "experimental", "D", True),
)
CAPTURE_RUNS = ("V0-C-5c375d43-cap", "V0-D-5c375d43-cap")
READINGS = (
    ("V0-R1 bridge C", "V0-C-66ac840e", "V0-C-5c375d43"),
    ("V0-R1 bridge D", "V0-D-66ac840e", "V0-D-5c375d43"),
    ("V0-R2 capture neutrality C", "V0-C-5c375d43", "V0-C-5c375d43-cap"),
    ("V0-R2 capture neutrality D", "V0-D-5c375d43", "V0-D-5c375d43-cap"),
)
READING_PREFIX = "V0-R"

LAYERLESS_SITES = ("embedding", "head_input")
LAYER_SITES = ("attention_input", "query_unpositioned", "key_unpositioned", "query_positioned",
               "key_positioned", "attention_output", "attention_delta", "ffn_input", "ffn_output",
               "ffn_delta", "residual")
CAPTURE_SCHEMA = "larql.site-capture/1"
LOGITS = "logits.f32"
LOGITS_META = "logits.meta.json"
SITES_DIR = "sites"
F32 = 4
COMPARE_ROWS = 64
#: How peak memory is measured: the binary's own rusage, taken by wait4 on
#: the process the runner started (macOS reports ru_maxrss in bytes). This
#: is the figure `/usr/bin/time -l` prints, without a wrapper process
#: interleaving its report into the binary's stderr.
RSS_METHOD = "wait4 ru_maxrss of the binary process (bytes on macOS)"
EXPERIMENTAL_ROPE_ANGLES = "larql"


def ids_json_sha256(ids: list[int]) -> str:
    """The inputs manifest's rule: sha256 of `json.dumps(ids)`."""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def ids_u32le_sha256(ids: list[int]) -> str:
    """LARQL's capture identity rule: sha256 over the u32 little-endian ids."""
    return hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()


def load_ids(path: Path) -> list[int]:
    value = json.loads(Path(path).read_text())
    ids = value["ids"] if isinstance(value, dict) else value
    if not all(isinstance(i, int) and i >= 0 for i in ids):
        raise ValueError(f"{path}: ids must be non-negative integers")
    return ids


def segment_digests(container: Path) -> dict:
    index = json.loads((container / "index.json").read_text())
    return {name: {"segment": r["segment"], "segment_sha256": r["segment_sha256"]}
            for name, r in sorted(index["representations"].items())}


def frozen_geometry(container: Path) -> dict:
    """The capture geometry the frozen model declares (its system graph):
    hidden width, and per layer the q heads, kv heads and head width.
    A capture is checked against this, not only against its own manifest."""
    graph = json.loads((Path(container) / "system_graph.json").read_text())
    primary = [c for c in graph["components"] if c.get("role") == "primary_text"]
    if len(primary) != 1:
        raise ValueError(f"{container}: expected one primary_text component, found {len(primary)}")
    c = primary[0]
    q_heads = c["execution"]["attention"]["num_q_heads"]
    layers = [{"layer": L, "num_q_heads": q_heads, "num_kv_heads": a["geometry"]["num_kv_heads"],
               "head_dim": a["geometry"]["head_dim"]} for L, a in enumerate(c["attention"])]
    if len(layers) != c["num_layers"]:
        raise ValueError(f"{container}: {len(layers)} attention entries for {c['num_layers']} layers")
    return {"hidden": c["hidden_size"], "layers": layers}


def module_digests() -> dict:
    here = Path(__file__).resolve().parent
    return {name: ev.sha256_file(here / name) for name in ("v0.py", "evidence.py")}


# ---------------------------------------------------------------- freeze

def build_freeze(*, doc: Path, container: Path, ids_path: Path, expected_ids_sha256: str,
                 historical: Path, historical_sha256: str, experimental: Path,
                 experimental_commit: str, positions: list[int], layers: int, vocab: int,
                 home: str, limits: dict, registered_representations: dict) -> dict:
    """V0.freeze.json's content. Deterministic for identical inputs."""
    ids = load_ids(ids_path)
    if ids_json_sha256(ids) != expected_ids_sha256:
        raise ValueError(f"{ids_path}: ids digest {ids_json_sha256(ids)} != {expected_ids_sha256}")
    if ev.sha256_file(historical) != historical_sha256:
        raise ValueError(f"{historical}: sha256 differs from the registered {historical_sha256}")
    if positions != sorted(set(positions)) or not positions or positions[-1] >= len(ids):
        raise ValueError("capture positions must be strictly increasing and inside the sequence")
    segments = segment_digests(container)
    for name, rep_ in registered_representations.items():
        if segments.get(name, {}).get("segment_sha256") != rep_.get("segment_sha256"):
            raise ValueError(f"container representation {name} differs from the registered record")
    if set(segments) != set(registered_representations):
        raise ValueError("container representations differ from the registered record")
    geometry = frozen_geometry(container)
    if len(geometry["layers"]) != layers:
        raise ValueError(f"--layers {layers} != the frozen graph's {len(geometry['layers'])}")
    env = {"HOME": home, "PATH": "/usr/bin:/bin", "RAYON_NUM_THREADS": "16"}
    return {
        "schema": FREEZE_SCHEMA,
        "caveat": BRIDGE_CAVEAT,
        "document": {"path": str(doc), "sha256": ev.sha256_file(doc)},
        "runner": module_digests(),
        "container": {
            "path": str(container),
            "index_json_sha256": ev.sha256_file(container / "index.json"),
            "system_graph_sha256": ev.sha256_file(container / "system_graph.json"),
            "segments": segments,
            "registered_representations": registered_representations,
        },
        "input": {"path": str(ids_path), "length": len(ids), "ids_json_sha256": ids_json_sha256(ids),
                  "ids_u32le_sha256": ids_u32le_sha256(ids)},
        "binaries": {
            "historical": {"path": str(historical), "sha256": historical_sha256, "commit": "66ac840e"},
            "experimental": {"path": str(experimental), "sha256": ev.sha256_file(experimental),
                             "commit": experimental_commit},
        },
        "arms": ARM_FLAGS,
        "runs": [{"run_id": r, "binary": b, "arm": a, "capture": c} for r, b, a, c in RUNS],
        "capture": {"positions": positions, "budget_bytes": limits["capture_bytes_per_run"],
                    "layers": layers, "geometry": geometry, "layerless_sites": list(LAYERLESS_SITES),
                    "layer_sites": list(LAYER_SITES)},
        "vocab": vocab,
        "environment": env,
        "resource_measure": RSS_METHOD,
        "limits": limits,
    }


def registered_representations(record_path: Path) -> dict:
    """The container digests the 2.1.0 record measured (`runs[3]`)."""
    run = json.loads(Path(record_path).read_text())["runs"][3]
    if run.get("harness_version") != "2.1.0":
        raise ValueError(f"{record_path}: runs[3] is harness {run.get('harness_version')!r}, not 2.1.0")
    return {name: {"segment_sha256": r["segment_sha256"]}
            for name, r in sorted(run["container"]["representations"].items())}


def write_freeze(path: Path, freeze: dict) -> None:
    if path.exists():
        raise ev.EvidenceError(f"{path} exists: a freeze record is written once")
    ev.write_atomic(path, json.dumps(freeze, indent=1, sort_keys=True).encode() + b"\n")


# ------------------------------------------------------------- preflight

def preflight(freeze: dict, root: Path) -> list[str]:
    """Every identity V0 depends on, checked before anything executes."""
    problems = []
    if module_digests() != freeze["runner"]:
        problems.append("runner modules differ from the freeze record")
    doc = Path(freeze["document"]["path"])
    if not doc.is_file():
        problems.append(f"protocol document missing: {doc}")
    elif ev.sha256_file(doc) != freeze["document"]["sha256"]:
        problems.append("protocol document sha256 differs from the freeze record")
    for key, b in freeze["binaries"].items():
        p = Path(b["path"])
        if not p.is_file():
            problems.append(f"{key} binary missing: {p}")
        elif ev.sha256_file(p) != b["sha256"]:
            problems.append(f"{key} binary sha256 differs from the freeze record")
    c = freeze["container"]
    cdir = Path(c["path"])
    try:
        if ev.sha256_file(cdir / "index.json") != c["index_json_sha256"]:
            problems.append("container index.json sha256 differs")
        if ev.sha256_file(cdir / "system_graph.json") != c["system_graph_sha256"]:
            problems.append("container system_graph.json sha256 differs")
        if segment_digests(cdir) != c["segments"]:
            problems.append("container segment table differs")
        for name, rep_ in c["registered_representations"].items():
            if c["segments"].get(name, {}).get("segment_sha256") != rep_.get("segment_sha256"):
                problems.append(f"representation {name} differs from the registered record")
        for name, s in c["segments"].items():
            seg = ev.contained(cdir, s["segment"])
            if seg is None:
                problems.append(f"segment {name} names a path outside the container")
            elif ev.sha256_file(seg) != s["segment_sha256"]:
                problems.append(f"segment {name} sha256 differs")
    except (OSError, KeyError, ValueError) as e:
        problems.append(f"container unreadable: {e}")
    try:
        ids = load_ids(Path(freeze["input"]["path"]))
        if ids_json_sha256(ids) != freeze["input"]["ids_json_sha256"]:
            problems.append("input ids digest differs")
    except (OSError, ValueError, KeyError) as e:
        problems.append(f"input unreadable: {e}")
    run_root = root / "runs"
    if run_root.exists():
        problems.append(f"{run_root} exists: V0's runs are made once")
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    if free < freeze["limits"]["min_free_bytes"]:
        problems.append(f"free disk {free} < {freeze['limits']['min_free_bytes']}")
    return problems


# ------------------------------------------------------------------ runs

def run_identity(freeze: dict, run: dict) -> dict:
    b = freeze["binaries"][run["binary"]]
    return {
        "experiment": "V0",
        "run_id": run["run_id"],
        "arm": run["arm"],
        "binary": {"key": run["binary"], "sha256": b["sha256"], "commit": b["commit"]},
        "container_index_sha256": freeze["container"]["index_json_sha256"],
        "input_ids_json_sha256": freeze["input"]["ids_json_sha256"],
        "argv": command(freeze, run, Path("<run_dir>")),
        "environment": freeze["environment"],
        "machine": os.uname().machine,
    }


def os_build() -> str:
    """The OS build, recorded per run; V0 requires it equal across all six."""
    try:
        done = subprocess.run(["/usr/bin/sw_vers", "-buildVersion"], capture_output=True, text=True,
                              timeout=10)
        build = done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        build = ""
    return f"{os.uname().sysname} {os.uname().release} {build}".strip()


def command(freeze: dict, run: dict, run_dir: Path) -> list[str]:
    ids = load_ids(Path(freeze["input"]["path"])) if run_dir != Path("<run_dir>") else None
    tokens = ",".join(map(str, ids)) if ids is not None else "<I_rep>"
    argv = [freeze["binaries"][run["binary"]]["path"], "vindex3", "exec", freeze["container"]["path"],
            *ARM_FLAGS[run["arm"]], "--tokens", tokens, "--logit-dump", str(run_dir / LOGITS)]
    if run["capture"]:
        argv += ["--capture-sites", str(run_dir / SITES_DIR),
                 "--capture-positions", ",".join(map(str, freeze["capture"]["positions"])),
                 "--capture-budget-bytes", str(freeze["capture"]["budget_bytes"])]
    return argv


def count_nonfinite(path: Path, vocab: int) -> int:
    data = np.memmap(path, dtype="<f4", mode="r").reshape(-1, vocab)
    return int(sum(int((~np.isfinite(data[i:i + COMPARE_ROWS])).sum())
                   for i in range(0, data.shape[0], COMPARE_ROWS)))


def verify_logits(freeze: dict, run: dict, run_dir: Path,
                  rope_angles: str = EXPERIMENTAL_ROPE_ANGLES) -> list[str]:
    problems = []
    n, vocab = freeze["input"]["length"], freeze["vocab"]
    logits, meta = run_dir / LOGITS, run_dir / LOGITS_META
    if not logits.is_file():
        return [f"{LOGITS} missing"]
    if logits.stat().st_size != n * vocab * F32:
        problems.append(f"{LOGITS} is {logits.stat().st_size} bytes, expected {n * vocab * F32}")
    if not meta.is_file():
        problems.append(f"{LOGITS_META} missing")
    elif run["binary"] == "experimental":
        try:
            angles = json.loads(meta.read_text()).get("rope_angles")
        except ValueError:
            angles = None
        if angles != rope_angles:
            problems.append(f"meta rope_angles {angles!r}, expected {rope_angles!r}")
    return problems


def verify_capture(freeze: dict, run: dict, run_dir: Path) -> list[str]:
    """E0a's capture must be `complete`, its manifest digest verified, every
    declared plane present and correct, and exactly the requested sites,
    layers and positions, under this run's identity."""
    sites = run_dir / SITES_DIR
    cap = freeze["capture"]
    try:
        result = json.loads((sites / "capture.result.json").read_text())
    except (OSError, ValueError) as e:
        return [f"capture.result.json unreadable: {e}"]
    if result.get("status") != ev.COMPLETE:
        return [f"capture status {result.get('status')!r}"]
    problems = []
    manifest_path = sites / "manifest.json"
    if not manifest_path.is_file():
        return ["capture manifest.json missing"]
    if ev.sha256_file(manifest_path) != (result.get("detail") or {}).get("manifest_sha256"):
        problems.append("capture manifest sha256 differs from capture.result.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CAPTURE_SCHEMA:
        problems.append(f"capture schema {manifest.get('schema')!r}")
    if manifest.get("positions") != cap["positions"]:
        problems.append("captured positions differ from the request")
    if manifest.get("sequence_len") != freeze["input"]["length"]:
        problems.append("captured sequence length differs")
    expected = {(s, None) for s in LAYERLESS_SITES}
    expected |= {(s, layer) for s in LAYER_SITES for layer in range(cap["layers"])}
    seen = set()
    # Two layers of geometry: the frozen model's (authoritative), then the
    # manifest's own declarations for consistency.
    frozen = cap["geometry"]
    if manifest.get("hidden") != frozen["hidden"]:
        problems.append(f"capture hidden {manifest.get('hidden')} != the frozen model's {frozen['hidden']}")
    if manifest.get("layers") != frozen["layers"]:
        problems.append("capture layer geometry differs from the frozen model's")
    widths = expected_widths(frozen)
    declared = expected_widths(manifest)
    for entry in manifest.get("files", []):
        key = (entry.get("site"), entry.get("layer"))
        if key in seen:
            problems.append(f"plane {key} declared twice")
        seen.add(key)
        p = ev.contained(sites, entry.get("file"))
        if p is None:
            problems.append(f"declared plane {entry.get('file')!r} is not inside the capture directory")
            continue
        if not p.is_file():
            problems.append(f"declared plane {entry.get('file')} missing")
            continue
        rows, width = entry.get("rows"), entry.get("width")
        if rows != len(cap["positions"]):
            problems.append(f"{entry['file']}: rows {rows}")
        if key in widths and width != widths[key]:
            problems.append(f"{entry['file']}: width {width}, expected {widths[key]} (frozen geometry)")
        elif key in declared and width != declared[key]:
            problems.append(f"{entry['file']}: width {width} disagrees with its manifest's geometry")
        if not isinstance(rows, int) or not isinstance(width, int) or p.stat().st_size != rows * width * F32:
            problems.append(f"{entry['file']}: size {p.stat().st_size}")
            continue
        if ev.sha256_file(p) != entry.get("sha256"):
            problems.append(f"{entry['file']}: sha256 differs from the manifest")
        else:
            plane = np.fromfile(p, dtype="<f4")
            if not np.isfinite(plane).all():
                problems.append(f"{entry['file']}: non-finite values")
    if seen != expected:
        missing, extra = sorted(expected - seen, key=str), sorted(seen - expected, key=str)
        problems.append(f"capture sites differ from the request: missing {missing[:5]}, extra {extra[:5]}")
    ident = manifest.get("identity") or {}
    b = freeze["binaries"][run["binary"]]
    want = {
        "binary_sha256": b["sha256"],
        "container_index_sha256": freeze["container"]["index_json_sha256"],
        "tokens_sha256": freeze["input"]["ids_u32le_sha256"],
        "rope_angles": EXPERIMENTAL_ROPE_ANGLES,
        **ARM_CAPTURE_IDENTITY[run["arm"]],
    }
    for k, v in want.items():
        if ident.get(k) != v:
            problems.append(f"capture identity {k} = {ident.get(k)!r}, expected {v!r}")
    env_pairs = {e["name"]: e["value"] for e in ident.get("environment", [])}
    if env_pairs != ev.numeric_environment(freeze["environment"]):
        problems.append(f"capture environment {env_pairs} differs from the run's")
    logits = run_dir / LOGITS
    if logits.is_file() and manifest.get("logit_dump_sha256") != ev.sha256_file(logits):
        problems.append("capture manifest's logit_dump_sha256 differs from the dump")
    return problems


def expected_widths(manifest: dict) -> dict:
    """Each plane's width from the manifest's own geometry: q and k are per
    head, everything else is the hidden width."""
    hidden = manifest.get("hidden")
    out = {("embedding", None): hidden, ("head_input", None): hidden}
    for layer in manifest.get("layers", []):
        L = layer.get("layer")
        q = layer.get("num_q_heads", 0) * layer.get("head_dim", 0)
        k = layer.get("num_kv_heads", 0) * layer.get("head_dim", 0)
        for site in LAYER_SITES:
            out[(site, L)] = q if site.startswith("query") else k if site.startswith("key") else hidden
    return out


def capture_outputs(run_dir: Path) -> list[str]:
    sites = run_dir / SITES_DIR
    return sorted(str(p.relative_to(run_dir)) for p in sites.iterdir() if p.is_file())


def spawn(argv: list[str], env: dict, run_dir: Path, limit_s: float) -> dict:
    """Start the binary directly in its own process group, wait with wait4,
    and kill the group at the wall-clock limit."""
    timed_out = threading.Event()
    started = time.monotonic()
    with open(run_dir / "stdout.txt", "wb") as out, open(run_dir / "stderr.txt", "wb") as err:
        proc = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)

        def kill() -> None:
            timed_out.set()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(limit_s, kill)
        timer.start()
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        except BaseException:
            kill()  # never leave the binary running behind an interrupted runner
            raise
        finally:
            timer.cancel()
        proc.returncode = os.waitstatus_to_exitcode(status)
    return {"exit": proc.returncode, "timed_out": timed_out.is_set(),
            "wall_s": round(time.monotonic() - started, 3), "peak_rss_bytes": int(usage.ru_maxrss)}


def execute(freeze: dict, run: dict, run_dir: Path) -> dict:
    """One run under the lifecycle. Returns the final record.

    Anything that goes wrong after the record is `in_progress` ends as a
    final status; only an interruption of the runner itself (a signal, a
    lost machine) leaves `in_progress`."""
    identity = {**run_identity(freeze, run), "os_build": os_build()}
    rr = ev.RunRecord.begin(run_dir, run["run_id"], "v0-larql", identity)
    limits = freeze["limits"]
    try:
        argv = command(freeze, run, run_dir)
        done = spawn(argv, dict(freeze["environment"]), run_dir, limits["wall_s_per_run"])
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return rr.fail(ev.FAULT, {"spawn": f"{type(e).__name__}: {e}"})
    measured = {"wall_s": done["wall_s"], "peak_rss_bytes": done["peak_rss_bytes"],
                "resource_measure": RSS_METHOD}
    if done["timed_out"]:
        return rr.fail(ev.RESOURCE_LIMIT, {"limit": "wall_s_per_run", "value": limits["wall_s_per_run"]},
                       **measured)
    if done["exit"] != 0:
        return rr.fail(ev.FAULT, {"exit": done["exit"]}, **measured)
    if done["peak_rss_bytes"] > limits["rss_bytes"]:
        return rr.fail(ev.RESOURCE_LIMIT, {"limit": "rss_bytes", "value": done["peak_rss_bytes"]},
                       **measured)
    try:
        problems = verify_logits(freeze, run, run_dir)
        if run["capture"] and not problems:
            problems += verify_capture(freeze, run, run_dir)
        if problems:
            return rr.fail(ev.FAULT, {"verification": problems}, **measured)
        nonfinite = count_nonfinite(run_dir / LOGITS, freeze["vocab"])
        outputs = [LOGITS, LOGITS_META, "stdout.txt", "stderr.txt"]
        if run["capture"]:
            outputs += capture_outputs(run_dir)
        return rr.complete(outputs, nonfinite_logits=nonfinite, **measured)
    except ev.EvidenceError:
        raise
    except Exception as e:  # a malformed artifact is a fault, never a stuck in_progress
        return rr.fail(ev.FAULT, {"verification_error": f"{type(e).__name__}: {e}"}, **measured)


def load_freeze(path: Path) -> dict:
    freeze = json.loads(Path(path).read_text())
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise ev.EvidenceError(f"{path}: not a V0 freeze record")
    return freeze


def run_all(freeze_path: Path, root: Path) -> list[dict]:
    """Preflight, then every run in order. Stops at nothing: each run's
    outcome is recorded whatever it is (a later run doesn't depend on an
    earlier one's success), except the total wall-clock limit."""
    freeze = load_freeze(freeze_path)
    root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(root.glob("preflight-*"))
    pre = ev.RunRecord.begin(root / f"preflight-{len(attempts) + 1}", "V0-preflight", "v0-preflight",
                             {"freeze_sha256": ev.sha256_file(freeze_path)})
    problems = preflight(freeze, root)
    if problems:
        pre.fail(ev.ABORTED_PREFLIGHT, problems)
        return [pre.record]
    (pre.run_dir / "checks.json").write_text(json.dumps({"checked": "all", "problems": []}))
    pre.complete(["checks.json"])
    records, total_start = [], time.monotonic()
    for run in freeze["runs"]:
        run_dir = root / "runs" / run["run_id"]
        if time.monotonic() - total_start > freeze["limits"]["wall_s_total"]:
            rr = ev.RunRecord.begin(run_dir, run["run_id"], "v0-larql", run_identity(freeze, run))
            records.append(rr.fail(ev.RESOURCE_LIMIT, {"limit": "wall_s_total"}))
            continue
        records.append(execute(freeze, run, run_dir))
    return records


def replace_interrupted(freeze_path: Path, root: Path, run_id: str, note: str) -> dict:
    """The one allowed replacement: a run left `in_progress` by an
    infrastructure interruption. The original is kept beside it."""
    freeze = load_freeze(freeze_path)
    run_dir = root / "runs" / run_id
    record = json.loads((run_dir / ev.RESULT_FILE).read_text())
    if record.get("status") != ev.IN_PROGRESS:
        raise ev.EvidenceError(f"{run_id} is {record.get('status')!r}: only an interrupted run is replaced")
    kept = run_dir.with_name(f"{run_id}.interrupted-1")
    if kept.exists():
        raise ev.EvidenceError(f"{run_id} was already replaced once; a further replacement needs an amendment")
    if not note.strip():
        raise ev.EvidenceError("a replacement needs a written note citing the cause")
    run_dir.rename(kept)
    (kept / "REPLACEMENT_NOTE.txt").write_text(note)
    run = next(r for r in freeze["runs"] if r["run_id"] == run_id)
    return execute(freeze, run, run_dir)


# -------------------------------------------------------------- readings

def first_difference(a: Path, b: Path, vocab: int) -> dict:
    """Byte equality of two logit dumps, with where they first differ."""
    if a.stat().st_size != b.stat().st_size:
        return {"identical": False, "size_differs": [a.stat().st_size, b.stat().st_size]}
    x = np.memmap(a, dtype="<u4", mode="r").reshape(-1, vocab)
    y = np.memmap(b, dtype="<u4", mode="r").reshape(-1, vocab)
    differing_positions, differing_values, first = 0, 0, None
    for i in range(0, x.shape[0], COMPARE_ROWS):
        neq = x[i:i + COMPARE_ROWS] != y[i:i + COMPARE_ROWS]
        rows = neq.any(axis=1)
        differing_positions += int(rows.sum())
        differing_values += int(neq.sum())
        if first is None and rows.any():
            r = int(np.argmax(rows))
            first = {"position": i + r, "vocab_index": int(np.argmax(neq[r]))}
    return {"identical": differing_values == 0, "differing_positions": differing_positions,
            "differing_values": differing_values, "first_difference": first}


def readings(freeze_path: Path, root: Path) -> list[dict]:
    freeze = load_freeze(freeze_path)
    log = ev.ResultsLog(root / "V0.results.jsonl")
    if any(str(e.get("reading", "")).startswith(READING_PREFIX) for e in log.verify()):
        raise ev.EvidenceError(f"{log.path} already holds V0 readings: they are recorded once")
    runs = {r["run_id"]: r for r in freeze["runs"]}
    adm = {rid: ev.admit(root / "runs" / rid, run_identity(freeze, run), [LOGITS])
           for rid, run in runs.items()}
    builds = {a.record.get("identity", {}).get("os_build") for a in adm.values() if a.record}
    if len(builds) > 1:
        for a in adm.values():
            a.admitted = False
            a.reasons.append(f"OS build differs across runs: {sorted(map(str, builds))}")
    written = []
    for name, left, right in READINGS:
        pair = [adm[left], adm[right]]
        if all(a.admitted for a in pair):
            values = first_difference(root / "runs" / left / LOGITS, root / "runs" / right / LOGITS,
                                      freeze["vocab"])
            entry = ev.reading(name, ev.READING_COMPLETE, pair, values, evidence_class="E1",
                               compared_bytes="raw f32 logits, all positions", caveat=BRIDGE_CAVEAT)
        else:
            entry = ev.reading(name, ev.NOT_ADMITTED, pair, evidence_class="E1")
        written.append(log.append(entry))
    cap = [adm[r] for r in CAPTURE_RUNS]
    written.append(log.append(ev.reading(
        "V0-R3 capture completeness", ev.READING_COMPLETE, cap,
        {"admitted": [a.run_dir.name for a in cap if a.admitted]}, evidence_class="validation")))
    resources = {}
    for rid in runs:
        rec = adm[rid].record or {}
        resources[rid] = {"status": rec.get("status", "missing"), "wall_s": rec.get("wall_s"),
                          "peak_rss_bytes": rec.get("peak_rss_bytes"),
                          "nonfinite_logits": rec.get("nonfinite_logits")}
    written.append(log.append({"reading": "V0-R4/R5 resources and sanity", "status": ev.READING_COMPLETE,
                               "evidence_class": "measurement", "values": resources,
                               "consumed": [a.as_json() for a in adm.values()]}))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("doc", "container", "ids", "historical", "experimental", "out"):
        f.add_argument(f"--{name}", required=True, type=Path)
    f.add_argument("--expected-ids-sha256", required=True)
    f.add_argument("--historical-sha256", required=True)
    f.add_argument("--experimental-commit", required=True)
    f.add_argument("--record", required=True, type=Path,
                   help="the registered equivalence record (runs[3].container is the reference)")
    f.add_argument("--positions", required=True, help="comma-separated, strictly increasing")
    f.add_argument("--layers", required=True, type=int)
    f.add_argument("--vocab", required=True, type=int)
    for name in ("run", "readings"):
        p = sub.add_parser(name)
        p.add_argument("--freeze", required=True, type=Path)
        p.add_argument("--root", required=True, type=Path)
    r = sub.add_parser("replace-interrupted")
    r.add_argument("--freeze", required=True, type=Path)
    r.add_argument("--root", required=True, type=Path)
    r.add_argument("--run-id", required=True)
    r.add_argument("--note", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        gib = 1 << 30
        limits = {"wall_s_per_run": 1800, "wall_s_total": 10800, "rss_bytes": 16 * 10**9,
                  "capture_bytes_per_run": 2 * gib, "min_free_bytes": 16 * gib}
        freeze = build_freeze(doc=a.doc, container=a.container, ids_path=a.ids,
                              expected_ids_sha256=a.expected_ids_sha256, historical=a.historical,
                              historical_sha256=a.historical_sha256, experimental=a.experimental,
                              experimental_commit=a.experimental_commit,
                              positions=[int(p) for p in a.positions.split(",")], layers=a.layers,
                              vocab=a.vocab, home=os.environ["HOME"], limits=limits,
                              registered_representations=registered_representations(a.record))
        write_freeze(a.out, freeze)
    elif a.cmd == "run":
        for rec in run_all(a.freeze, a.root):
            print(rec["run_id"], rec["status"])
    elif a.cmd == "readings":
        for entry in readings(a.freeze, a.root):
            print(entry["reading"], entry["status"])
    else:
        rec = replace_interrupted(a.freeze, a.root, a.run_id, a.note.read_text())
        print(rec["run_id"], rec["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
