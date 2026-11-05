#!/usr/bin/env python3
"""Executable CAR-1B capital protocol, checked artifacts, source build and adjudication."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from gwcar1b_analysis import PAIRS, adjudicate, effects, summarize
from gwcar1b_preregister import ROOT, excluded_subjects
from gwcar1b_sources import FEATURES, GRID, Row, construct, fit, grid_document, oracle, top_support
from gwv2_population import TOKENIZER_SHA256, canonical_hash, read_jsonl, sha256, short_id

BASE = ROOT / "bench/gw-car-1/gemma3-4b-it-phase1"
RUN = ROOT / "output/gwcar1b-capital-v1"
CAPTURE = ROOT / "output/gwv2-gemma3-4b-it-phase1-capture/capture.json"
CODE = (
    "scripts/gwcar1b_runtime.py", "scripts/gwcar1b_sources.py", "scripts/gwcar1b_analysis.py",
    "scripts/gwcar1b_preregister.py", "scripts/gwcar1b_population.py", "scripts/gwv2_population.py",
    "scripts/test_gwcar1b_sources.py", "scripts/test_gwcar1b_analysis.py",
    "scripts/test_gwcar1b_runtime.py", "crates/larql-demos/examples/inference/gwcar1b_record.rs",
    "crates/larql-demos/Cargo.toml", "docs/gw-car-1b-execution.md", "rust-toolchain.toml", "Cargo.lock",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_new(path: Path, document: dict, field: str) -> dict:
    document = dict(document)
    document[field] = canonical_hash(document, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return document


def artifact(path: Path, value: np.ndarray) -> dict:
    value = np.asarray(value, dtype="<f4")
    if not np.isfinite(value).all():
        raise ValueError("nonfinite artifact")
    with path.open("xb") as handle:
        handle.write(value.tobytes())
    return {"path": path.name, "shape": list(value.shape), "dtype": "f32-le",
            "bytes": value.nbytes, "sha256": sha256(path)}


def array(manifest_path: Path, name: str, shape: tuple) -> np.ndarray:
    manifest = read(manifest_path)
    entries = [e for e in manifest["artifacts"] if e["path"] == name]
    path = manifest_path.parent / name
    if (len(entries) != 1 or entries[0]["shape"] != list(shape)
        or entries[0]["dtype"] != "f32-le" or entries[0]["bytes"] != int(np.prod(shape)) * 4
        or path.stat().st_size != entries[0]["bytes"] or sha256(path) != entries[0]["sha256"]):
        raise ValueError(f"artifact identity/geometry mismatch: {path}")
    values = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite artifact")
    return values


def population() -> tuple[list[dict], list[dict], dict]:
    from tokenizers import Tokenizer

    legacy = read(ROOT / "bench/gw-v2/gemma3-4b-it-phase1/execution.json")
    container = Path(legacy["container"])
    tokenizer_path = container / "tokenizer.json"
    if sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("tokenizer changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    training = []
    for i, item in enumerate(legacy["rows"]):
        row = item["row"]
        if row["split"] == "train":
            semantic = row["semantic_edge"]
            training.append({"row_id": row["edge_id"], "subject_id": row["subject_id"],
                             "relation": semantic["relation"], "template": semantic["prompt_semantic_family"],
                             "split": "train", "token_ids": row["prompt"]["token_ids"],
                             "subject_token_ids": tokenizer.encode(semantic["subject"], add_special_tokens=False).ids,
                             "historical_row_index": i})
    proposal = read(BASE / "gwcar1b-capital-population.proposal.json")
    if proposal["proposal_sha256"] != canonical_hash(proposal, "proposal_sha256"):
        raise ValueError("population proposal identity mismatch")
    confirmation = []
    excluded = excluded_subjects()
    for row in proposal["rows"]:
        if row["subject_id"] in excluded:
            raise ValueError("confirmation subject previously studied")
        confirmation.append({"row_id": short_id("car1b", [row["subject_id"], row["prompt_family"]]),
                             "subject_id": row["subject_id"], "relation": "capital",
                             "template": row["prompt_family"], "split": row["split"],
                             "token_ids": row["prompt_token_ids"],
                             "subject_token_ids": row["subject_embedding_token_ids"]})
    for split in ("validation", "test"):
        for template in ("canonical", "question", "alternate"):
            indices = sorted((i for i, r in enumerate(confirmation)
                              if r["split"] == split and r["template"] == template),
                             key=lambda i: confirmation[i]["row_id"])
            for i, index in enumerate(indices):
                confirmation[index]["donor_index"] = indices[(i + 1) % len(indices)]
    if len(training) != 396 or len(confirmation) != 201:
        raise ValueError("capital population size changed")
    if sorted(r["donor_index"] for r in confirmation) != list(range(201)):
        raise ValueError("donor map is not a derangement")
    return training, confirmation, proposal


def expected() -> dict:
    training, confirmation, proposal = population()
    legacy = read(ROOT / "bench/gw-v2/gemma3-4b-it-phase1/execution.json")
    old = read(BASE / "gwcar1a-protocol.json")
    if sha256(CAPTURE) != old["authorities"]["v2_capture"]["sha256"]:
        raise ValueError("historical capture changed")
    authorities = list(CODE) + [
        "bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-population.proposal.json",
        "bench/gw-car-1/gemma3-4b-it-phase1/gwcar1a-protocol.json",
        "bench/gw-v2/gemma3-4b-it-phase1/execution.json",
        "bench/gw-v2/gemma3-4b-it-phase1/input.jsonl",
        "bench/gw-v2/gemma3-4b-it-phase1/population-manifest.json",
        str(CAPTURE.relative_to(ROOT)),
    ]
    return {
        "schema": "larql.gwcar1b.protocol.v1", "status": "draft_before_capture",
        "protocol_sha256": None, "executable_sha256": None,
        "authorities": {name: sha256(ROOT / name) for name in authorities},
        "container": legacy["container"], "model_metadata": old["model_metadata"],
        "model_segments": old["model_segments"], "plan_sha256": old["plan_sha256"],
        "historical_capture": str(CAPTURE), "output_root": str(RUN),
        "train_rows": training, "confirmation_rows": confirmation,
        "candidate_token_ids": proposal["candidate_token_ids"],
        "source_grid": grid_document(), "reference_contexts": [0, 128, 255],
        "h1_arms": ["donor", "target_exact_natural"],
        "analysis": {"seed": 27022034, "replicates": 10000, "family_size": 104,
                     "difference_pairs": [list(p) for p in PAIRS],
                     "degenerate_se_at_most": 1e-15,
                     "degenerate_rule": "standardized statistic zero; lower bound min(point,0)",
                     "critical_floor": 0, "scope": "capital only",
                     "validation_subjects": 34, "test_subjects": 33,
                     "proximal_point_min": 0.8, "proximal_lower_min": 0.5,
                     "terminal_lower_strictly_positive": True},
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "efficiency_claim": False,
    }


def validate(path: Path, frozen: bool = True) -> dict:
    document = read(path)
    match = expected()
    if document.get("status") == "frozen_before_capture":
        exe = document.get("executable_sha256")
        if not isinstance(exe, str) or len(exe) != 71 or not exe.startswith("sha256:"):
            raise ValueError("missing executable seal")
        match["status"], match["executable_sha256"] = document["status"], exe
    elif frozen:
        raise ValueError("model operations require a frozen CAR-1B protocol")
    match["protocol_sha256"] = canonical_hash(match, "protocol_sha256")
    if document != match:
        raise ValueError("protocol differs from declared population, code or authorities")
    for name, digest in document["model_metadata"].items():
        if sha256(Path(document["container"]) / name) != digest:
            raise ValueError("model metadata changed")
    return document


def seal(draft: Path, executable: Path, output: Path) -> dict:
    doc = validate(draft, frozen=False)
    if doc["status"] != "draft_before_capture" or RUN.exists():
        raise ValueError("seal requires draft and no CAR-1B capture/fitting/replay directory")
    for name, digest in doc["model_segments"].items():
        if sha256(Path(doc["container"]) / name) != digest:
            raise ValueError("model segment differs from historical execution image")
    doc.update(status="frozen_before_capture", executable_sha256=sha256(executable))
    write_new(output, doc, "protocol_sha256")
    return validate(output)


def checked_capture(protocol: dict, path: Path) -> dict[str, np.ndarray]:
    manifest = read(path)
    if (manifest.get("schema") != "larql.gwcar1b.capture.v1"
        or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or manifest.get("executable_sha256") != protocol["executable_sha256"]
        or manifest.get("rows") != 597 or manifest.get("identity_bit_mismatches") != 0):
        raise ValueError("unbound or incomplete CAR-1B capture")
    shapes = {f"{name}.f32": (597, 2560) for name in (*FEATURES, "carrier-before", "carrier-after")}
    shapes.update({"head-values.f32": (597, 8, 256), "natural-terminal.f32": (597, 57)})
    if {a["path"] for a in manifest["artifacts"]} != set(shapes):
        raise ValueError("capture inventory changed")
    result = {name: array(path, name, shape) for name, shape in shapes.items()}
    old = read(CAPTURE)
    # Legacy captures predate a dtype field; validate their sealed entry directly.
    entry = next(a for a in old["artifacts"] if a["path"] == "carrier-before.f32")
    source = CAPTURE.parent / entry["path"]
    if sha256(source) != entry["sha256"] or source.stat().st_size != 666 * 2560 * 4:
        raise ValueError("historical carrier tensor changed")
    historical = np.fromfile(source, dtype="<f4").reshape(666, 2560)
    indices = [r["historical_row_index"] for r in protocol["train_rows"]]
    if not np.array_equal(result["carrier-before.f32"][:396].view("<u4"), historical[indices].view("<u4")):
        raise ValueError("new train carriers do not reproduce historical carriers bit for bit")
    return result


def row_object(row: dict) -> Row:
    return Row(**{key: row[key] for key in ("row_id", "subject_id", "relation", "template", "split")})


def model_components(bank, candidate):
    c = candidate
    numeric = {}
    templates = np.stack([bank.templates[k] for k in sorted(bank.templates)])
    if c.family == "global":
        numeric["mean"] = bank.global_mean
    if c.family == "relation":
        numeric["means"] = np.stack([bank.relations[k] for k in sorted(bank.relations)])
    if c.family in ("template", "ridge") or c.candidate_id in (4, 5):
        numeric["templates"] = templates
    if c.family == "nearest":
        numeric.update(keys=bank.features[c.feature], values=bank.carriers)
    if c.candidate_id == 3:
        numeric["support"] = bank.carrier_support
    if c.family == "ridge" or c.candidate_id == 5:
        numeric["support"] = bank.delta_support
    if c.family == "ridge":
        fitted = bank.ridge[c.feature, c.sham]
        numeric.update(mean=fitted.mean, target_mean=fitted.target_mean, weights=fitted.weights)
    metadata = {"row_ids": [r.row_id for r in bank.rows] if c.family == "nearest" else [],
                "template_keys": [list(k) for k in sorted(bank.templates)] if "templates" in numeric else [],
                "relation_keys": sorted(bank.relations) if c.family == "relation" else [],
                "row_groups": [list(r.group) for r in bank.rows] if c.family == "nearest" else [],
                "feature": c.feature, "candidate_id": c.candidate_id}
    return numeric, metadata


def build_candidates(protocol_path: Path, capture_path: Path, output: Path) -> dict:
    protocol = validate(protocol_path)
    capture = checked_capture(protocol, capture_path)
    bank = fit([row_object(r) for r in protocol["train_rows"]], capture["carrier-before.f32"][:396],
               {name: capture[f"{name}.f32"][:396] for name in FEATURES})
    output.mkdir(parents=True, exist_ok=False)
    # Store each candidate's exact numeric model independently for a real byte ledger.
    models = []
    for c in GRID:
        numeric, metadata = model_components(bank, c)
        model = output / f"model-{c.candidate_id:02}.npz"
        with model.open("xb") as handle:
            np.savez(handle, **numeric)
        metadata_path = output / f"model-{c.candidate_id:02}.json"
        write_new(metadata_path, metadata, "metadata_sha256")
        models.append({"candidate_id": c.candidate_id, "numeric_bytes": sum(v.nbytes for v in numeric.values()),
                       "model_path": model.name, "model_file_bytes": model.stat().st_size,
                       "model_sha256": sha256(model), "metadata_path": metadata_path.name,
                       "metadata_bytes": metadata_path.stat().st_size, "metadata_sha256": sha256(metadata_path)})
    decoded = np.empty((201, 20, 2560), dtype="<f4")
    receipts, diagnostics = [], []
    for i, row in enumerate(protocol["confirmation_rows"]):
        query = row_object(row)
        natural = capture["carrier-before.f32"][396 + i]
        donor = capture["carrier-before.f32"][396 + row["donor_index"]]
        natural_support = top_support(np.abs(natural).astype(float))
        for c in GRID:
            if c.family == "oracle":
                value = oracle(bank, c.candidate_id, query, natural, donor)
                receipt = {"candidate_id": c.candidate_id, "query_row_id": query.row_id,
                           "reads_target_l24_carrier": True,
                           "uses_target_l24_values": c.candidate_id != 1,
                           "reads_donor_l24_carrier": c.candidate_id == 1, "efficiency_claim": False}
            else:
                feature = capture[f"{c.feature}.f32"][396 + i] if c.feature else None
                value, receipt = construct(bank, c.candidate_id, query, feature)
                if receipt["numeric_shared_bytes"] != models[c.candidate_id]["numeric_bytes"]:
                    raise ValueError("constructed numeric model bytes differ from serialized model")
            decoded[i, c.candidate_id] = value
            receipts.append(receipt)
            diagnostics.append({"row": i, "candidate": c.candidate_id,
                                "norm": float(np.linalg.norm(value.astype(float))),
                                "top32_support_overlap": int(np.intersect1d(
                                    natural_support, top_support(np.abs(value).astype(float))).size),
                                "mse": float(np.mean((value.astype(float) - natural) ** 2))})
    entries = [artifact(output / "carriers.f32", decoded)]
    receipt_path = output / "receipts.json"
    write_new(receipt_path, {"receipts": receipts, "diagnostics": diagnostics}, "receipts_sha256")
    return write_new(output / "candidates.json", {
        "schema": "larql.gwcar1b.candidates.v1", "protocol_sha256": protocol["protocol_sha256"],
        "capture_path": str(capture_path.resolve()), "capture_sha256": sha256(capture_path),
        "train_row_ids": [r.row_id for r in bank.rows], "candidate_ids": list(range(20)),
        "artifacts": entries, "models": models, "receipts_sha256": sha256(receipt_path),
        "heldout_used_for_fit": False, "efficiency_claim": False,
    }, "candidates_sha256")


def check_candidates(protocol_path: Path, path: Path) -> dict:
    protocol = validate(protocol_path)
    doc = read(path)
    if (doc.get("schema") != "larql.gwcar1b.candidates.v1" or doc.get("protocol_sha256") != protocol["protocol_sha256"]
        or doc.get("candidates_sha256") != canonical_hash(doc, "candidates_sha256")
        or doc.get("candidate_ids") != list(range(20)) or doc.get("heldout_used_for_fit") is not False
        or doc.get("train_row_ids") != sorted(r["row_id"] for r in protocol["train_rows"])
        or sha256(Path(doc["capture_path"])) != doc["capture_sha256"]
        or sha256(path.parent / "receipts.json") != doc["receipts_sha256"]):
        raise ValueError("candidate provenance mismatch")
    capture = checked_capture(protocol, Path(doc["capture_path"]))
    decoded = array(path, "carriers.f32", (201, 20, 2560))
    bank = fit([row_object(r) for r in protocol["train_rows"]], capture["carrier-before.f32"][:396],
               {name: capture[f"{name}.f32"][:396] for name in FEATURES})
    if len(doc["models"]) != 20:
        raise ValueError("missing source models")
    for i, model in enumerate(doc["models"]):
        if (model["candidate_id"] != i or model["model_path"] != f"model-{i:02}.npz"
            or model["metadata_path"] != f"model-{i:02}.json"):
            raise ValueError("source model order changed")
        for key, digest, size in (("model_path", "model_sha256", "model_file_bytes"),
                                  ("metadata_path", "metadata_sha256", "metadata_bytes")):
            file = path.parent / model[key]
            if file.stat().st_size != model[size] or sha256(file) != model[digest]:
                raise ValueError("source model identity mismatch")
        numeric, metadata = model_components(bank, GRID[i])
        metadata["metadata_sha256"] = canonical_hash(metadata, "metadata_sha256")
        if read(path.parent / model["metadata_path"]) != metadata:
            raise ValueError("source metadata is not the train-only lookup")
        if model["numeric_bytes"] != sum(v.nbytes for v in numeric.values()):
            raise ValueError("source model numeric cost mismatch")
        with np.load(path.parent / model["model_path"], allow_pickle=False) as stored:
            if set(stored.files) != set(numeric) or any(
                stored[k].shape != v.shape or stored[k].dtype != v.dtype or stored[k].tobytes() != v.tobytes()
                for k, v in numeric.items()):
                raise ValueError("source model does not reproduce the train-only fit")
    for i, row in enumerate(protocol["confirmation_rows"]):
        query = row_object(row)
        natural = capture["carrier-before.f32"][396 + i]
        donor = capture["carrier-before.f32"][396 + row["donor_index"]]
        for c in GRID:
            if c.family == "oracle":
                expected_carrier = oracle(bank, c.candidate_id, query, natural, donor)
            else:
                feature = capture[f"{c.feature}.f32"][396 + i] if c.feature else None
                expected_carrier, _ = construct(bank, c.candidate_id, query, feature)
            if not np.array_equal(decoded[i, c.candidate_id].view("<u4"), expected_carrier.view("<u4")):
                raise ValueError("decoded candidate does not reproduce its declared source")
    return doc


def replay_arrays(protocol: dict, path: Path, kind: str, candidates: Path | None = None):
    doc = read(path)
    n = 3 if kind == "reference" else 20
    if (doc.get("schema") != f"larql.gwcar1b.{kind}.v1"
        or doc.get("protocol_sha256") != protocol["protocol_sha256"]
        or doc.get("executable_sha256") != protocol["executable_sha256"]
        or doc.get("identity_bit_mismatches") != 0 or doc.get("intervention_firings") != 201 * n * 2
        or doc.get("h1_arms") != ["donor", "target_exact_natural"]
        or doc.get("ids") != ([0, 128, 255] if kind == "reference" else list(range(20)))):
        raise ValueError("unbound/incomplete replay")
    if candidates is not None and doc.get("candidate_manifest_sha256") != sha256(candidates):
        raise ValueError("replay candidate binding mismatch")
    return (array(path, "proximal.f32", (201, n, 2, 57)), array(path, "terminal.f32", (201, n, 2, 57)))


def adjudicate_run(protocol_path: Path, candidates_path: Path, reference_path: Path,
                   replay_path: Path, output: Path) -> dict:
    protocol = validate(protocol_path)
    candidates = check_candidates(protocol_path, candidates_path)
    reference = replay_arrays(protocol, reference_path, "reference")
    candidate_values = replay_arrays(protocol, replay_path, "replay", candidates_path)
    ref_doc, rep_doc = read(reference_path), read(replay_path)
    if (rep_doc["reference_manifest_sha256"] != sha256(reference_path)
        or ref_doc["capture_manifest_sha256"] != candidates["capture_sha256"]
        or rep_doc["capture_manifest_sha256"] != candidates["capture_sha256"]):
        raise ValueError("replay capture/reference binding mismatch")
    for values, refs in zip(candidate_values, reference, strict=True):
        for candidate, context in ((0, 1), (1, 0)):
            if not np.array_equal(values[:, candidate].view("<u4"), refs[:, context].view("<u4")):
                raise ValueError("oracle candidate does not equal independent reference bit for bit")
    results = {}
    for split, count in (("validation", 34), ("test", 33)):
        subjects, values = effects(protocol["confirmation_rows"], *candidate_values, split)
        ref_subjects, ref_values = effects(protocol["confirmation_rows"], *reference, split)
        if subjects != ref_subjects or len(subjects) != count:
            raise ValueError("subject pairing changed")
        results[split] = {"subjects": subjects, "analysis": adjudicate(values, ref_values[:, 2])}
    return write_new(output, {
        "schema": "larql.gwcar1b.adjudication.v1", "protocol_sha256": protocol["protocol_sha256"],
        "authorities": {str(p): sha256(p) for p in (candidates_path, reference_path, replay_path)},
        "results": results, "summary": summarize(results["validation"]["analysis"], results["test"]["analysis"]),
        "models": candidates["models"], "efficiency_claim": False,
    }, "adjudication_sha256")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draft", "validate", "seal", "check-capture", "candidates", "check-candidates", "adjudicate"])
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    paths = args.paths
    if args.command == "draft" and len(paths) == 1:
        result = write_new(paths[0], expected(), "protocol_sha256")
    elif args.command == "validate" and len(paths) == 1:
        result = validate(paths[0], frozen=False)
    elif args.command == "seal" and len(paths) == 3:
        result = seal(*paths)
    elif args.command == "check-capture" and len(paths) == 2:
        checked_capture(validate(paths[0]), paths[1]); result = {"status": "capture_valid"}
    elif args.command == "candidates" and len(paths) == 3:
        result = build_candidates(*paths)
    elif args.command == "check-candidates" and len(paths) == 2:
        result = check_candidates(*paths)
    elif args.command == "adjudicate" and len(paths) == 5:
        result = adjudicate_run(*paths)
    else:
        parser.error("incorrect path count for command")
    print(json.dumps({k: v for k, v in result.items() if k in (
        "schema", "status", "protocol_sha256", "candidates_sha256", "adjudication_sha256", "summary")}, indent=2))
