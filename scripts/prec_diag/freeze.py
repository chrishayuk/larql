"""PREC-DIAG-1 freeze manifest: deterministic `FREEZE.json` and its preflight.

rev 3 "Freeze manifest" lists what the manifest holds. A spec JSON names
each of those things:
- `files`: label -> path, each recorded with its sha256;
- `container`: path; records the `index.json` sha256 and every segment's
  digest, both from the index and re-hashed from disk;
- `checkpoint`: revision and directory; records every safetensors digest;
- `values`: label -> literal, for the LARQL commit, package versions,
  thread settings, capture positions and resource limits.

The readings' thresholds come from `analysis.thresholds()`, never from the
spec, so the frozen numbers are the ones the analysis runs with.

The manifest is canonical JSON, so the same spec over the same bytes gives
the same file, byte for byte. It is written once. `preflight()` recomputes
every digest and lists each mismatch; an experiment with any mismatch is
`aborted_preflight`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prec_diag import analysis  # noqa: E402
from prec_diag import evidence as ev  # noqa: E402

SCHEMA = "larql.prec-diag-1.freeze/1"
#: Labels the manifest must carry (rev 3 "Freeze manifest"); a spec missing
#: one is refused rather than frozen incomplete.
REQUIRED_FILES = ("plan", "evidence_classes", "e0a_doc", "e0b_doc", "inputs_manifest", "ids.I_rep",
                  "ids.I_nr1", "ids.I_nr2", "text.I_nr1", "text.I_nr2", "tokenizer_json", "larql_binary",
                  "rope_angle_tables_example", "script.hf_oracle", "script.v0", "script.analysis",
                  "script.evidence", "script.freeze", "script.oracle_loading", "script.equivalence_metrics",
                  "script.experiment", "loading_boundary")
REQUIRED_VALUES = ("larql_commit", "checkpoint_revision", "packages", "threads", "capture_positions",
                   "resource_limits")


class FreezeError(Exception):
    pass


def container_digests(path: Path) -> dict:
    index_path = Path(path) / "index.json"
    index = json.loads(index_path.read_text())
    segments = {}
    for name, rep in sorted(index["representations"].items()):
        seg = ev.contained(Path(path), rep["segment"])
        if seg is None:
            raise FreezeError(f"representation {name} names a segment outside the container")
        actual = ev.sha256_file(seg)
        if actual != rep["segment_sha256"]:
            raise FreezeError(f"segment {name} on disk differs from its index digest")
        segments[name] = {"segment": rep["segment"], "segment_sha256": actual,
                          "payload_sha256": rep.get("payload_sha256")}
    return {"path": str(path), "index_json_sha256": ev.sha256_file(index_path),
            "system_graph_sha256": ev.sha256_file(Path(path) / "system_graph.json"), "segments": segments}


def checkpoint_digests(directory: Path, revision: str) -> dict:
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise FreezeError(f"{directory}: no safetensors")
    return {"revision": revision, "directory": str(directory),
            "safetensors": {p.name: ev.sha256_file(p) for p in files},
            "config_json_sha256": ev.sha256_file(Path(directory) / "config.json")}


def build(spec: dict) -> dict:
    files = spec.get("files", {})
    values = spec.get("values", {})
    missing = [k for k in REQUIRED_FILES if k not in files] + [k for k in REQUIRED_VALUES if k not in values]
    if missing:
        raise FreezeError(f"the spec lacks {missing}")
    for label, path in files.items():
        if not Path(path).is_file():
            raise FreezeError(f"{label}: {path} is not a file")
    return {
        "schema": SCHEMA,
        "files": {label: {"path": str(path), "sha256": ev.sha256_file(Path(path))}
                  for label, path in sorted(files.items())},
        "container": container_digests(Path(spec["container"])),
        "checkpoint": checkpoint_digests(Path(spec["checkpoint"]["directory"]), spec["checkpoint"]["revision"]),
        "values": values,
        "thresholds": analysis.thresholds(),
    }


def write(path: Path, manifest: dict) -> str:
    """Write once; return the manifest's sha256."""
    path = Path(path)
    if path.exists():
        raise FreezeError(f"{path} exists: the freeze manifest is written once")
    data = ev.canonical_json(manifest) + b"\n"
    ev.write_atomic(path, data)
    return ev.sha256_bytes(data)


def preflight(manifest_path: Path, expected_sha256: str | None = None) -> list[str]:
    """Every mismatch between the frozen manifest and what is on disk now."""
    raw = Path(manifest_path).read_bytes()
    problems = []
    if expected_sha256 is not None and ev.sha256_bytes(raw) != expected_sha256:
        problems.append("FREEZE.json differs from the digest in the review record")
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA:
        return problems + [f"schema {manifest.get('schema')!r}"]
    for label, entry in manifest["files"].items():
        p = Path(entry["path"])
        if not p.is_file():
            problems.append(f"{label}: {p} missing")
        elif ev.sha256_file(p) != entry["sha256"]:
            problems.append(f"{label}: sha256 differs")
    try:
        if container_digests(Path(manifest["container"]["path"])) != manifest["container"]:
            problems.append("container digests differ")
    except (FreezeError, OSError, KeyError, ValueError) as e:
        problems.append(f"container: {e}")
    try:
        c = manifest["checkpoint"]
        if checkpoint_digests(Path(c["directory"]), c["revision"]) != c:
            problems.append("checkpoint digests differ")
    except (FreezeError, OSError, KeyError) as e:
        problems.append(f"checkpoint: {e}")
    if manifest["thresholds"] != json.loads(json.dumps(analysis.thresholds())):
        problems.append("analysis thresholds differ from the frozen ones")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--spec", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("preflight")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--sha256")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(write(a.out, build(json.loads(a.spec.read_text()))))
        return 0
    problems = preflight(a.manifest, a.sha256)
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
