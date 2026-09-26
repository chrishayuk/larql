#!/usr/bin/env python3
"""Freeze train-only PCA and build checked CAR-1A carrier replay inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gwcar1a_codecs import (
    GRID, PCA_LEVELS, WIDTH, decode_ledger, encode, fit_pca, model_bytes_for_rank,
    pca_model_bytes,
)
from gwcar1a_preregister import FROZEN_STATUS, validate as validate_protocol
from gwstate1_preflight import matched_donors
from gwv2_io import array
from gwv2_population import canonical_hash, sha256


def source(protocol_path: Path) -> tuple[dict, list[dict], np.ndarray, list[int]]:
    status = validate_protocol(protocol_path)
    if status["status"] != FROZEN_STATUS:
        raise ValueError("CAR-1A carrier inputs require a frozen protocol")
    protocol = json.loads(protocol_path.read_text())
    execution = json.loads(Path(protocol["authorities"]["v2_execution"]["path"]).read_text())
    rows = execution["rows"]
    capture_path = Path(protocol["authorities"]["v2_capture"]["path"])
    capture = json.loads(capture_path.read_text())
    carriers = array(capture_path.parent, capture, "carrier-before.f32")
    if carriers.shape != (666, WIDTH):
        raise ValueError("frozen entering-carrier geometry changed")
    return protocol, rows, carriers, matched_donors(rows)


def write_json_new(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def fit(protocol_path: Path, output: Path) -> dict:
    protocol, rows, carriers, _ = source(protocol_path)
    train = [i for i, item in enumerate(rows) if item["row"]["split"] == "train"]
    if len(train) != 396:
        raise ValueError("CAR-1A train carrier coverage changed")
    mean, basis = fit_pca(carriers[train])
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "pca-model.f32"
    with model_path.open("xb") as handle:
        handle.write(pca_model_bytes(mean, basis))
    model_bytes = model_path.read_bytes()
    document = {
        "schema": "larql.gwcar1a.pca-fit.v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "capture_file_sha256": protocol["authorities"]["v2_capture"]["sha256"],
        "train_row_indices": train,
        "model": {"path": "pca-model.f32", "shape": [385, WIDTH],
                  "dtype": "f32-le", "bytes": len(model_bytes), "sha256": sha256(model_path)},
        "rank_prefixes": {str(rank): {"bytes": model_bytes_for_rank(rank),
                                        "sha256": "sha256:" + hashlib.sha256(
                                            model_bytes[:model_bytes_for_rank(rank)]).hexdigest()}
                          for rank in PCA_LEVELS},
        "outcomes_examined": False,
    }
    document["fit_sha256"] = canonical_hash(document, "fit_sha256")
    write_json_new(output / "fit.json", document)
    return {"fit_sha256": document["fit_sha256"]}


def validate_fit(protocol_path: Path, path: Path) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    protocol, rows, _, _ = source(protocol_path)
    document = json.loads(path.read_text())
    train = [i for i, item in enumerate(rows) if item["row"]["split"] == "train"]
    model_path = path.parent / "pca-model.f32"
    model_bytes = model_path.read_bytes()
    if (document.get("schema") != "larql.gwcar1a.pca-fit.v1"
        or document.get("fit_sha256") != canonical_hash(document, "fit_sha256")
        or document.get("protocol_sha256") != protocol["protocol_sha256"]
        or document.get("capture_file_sha256") != protocol["authorities"]["v2_capture"]["sha256"]
        or document.get("train_row_indices") != train
        or document.get("model") != {"path": "pca-model.f32", "shape": [385, WIDTH],
                                     "dtype": "f32-le", "bytes": 385 * WIDTH * 4,
                                     "sha256": sha256(model_path)}
        or document.get("rank_prefixes") !=
           {str(rank): {"bytes": model_bytes_for_rank(rank),
                        "sha256": "sha256:" + hashlib.sha256(
                            model_bytes[:model_bytes_for_rank(rank)]).hexdigest()}
            for rank in PCA_LEVELS}
        or document.get("outcomes_examined") is not False):
        raise ValueError("CAR-1A PCA fit authority mismatch")
    model = np.frombuffer(model_bytes, dtype="<f4").reshape(385, WIDTH)
    if not np.isfinite(model).all():
        raise ValueError("nonfinite CAR-1A PCA model")
    return document, (model[0], model[1:])


def stage_indices(rows: list[dict], stage: str) -> list[int]:
    if stage not in ("train", "heldout"):
        raise ValueError("invalid CAR-1A stage")
    indices = [i for i, item in enumerate(rows)
               if (item["row"]["split"] == "train" if stage == "train"
                   else item["row"]["split"] in ("validation", "test"))]
    if len(indices) != (396 if stage == "train" else 270):
        raise ValueError("CAR-1A stage row coverage changed")
    return indices


def build(protocol_path: Path, fit_path: Path, stage: str, output: Path) -> dict:
    protocol, rows, carriers, donors = source(protocol_path)
    fitted, model = validate_fit(protocol_path, fit_path)
    indices = stage_indices(rows, stage)
    output.mkdir(parents=True, exist_ok=False)
    decoded_path = output / "carriers.f32"
    payload_paths = [output / f"payload-{candidate.candidate_id:02d}.bin" for candidate in GRID]
    payloads = [path.open("xb") for path in payload_paths]
    try:
        with decoded_path.open("xb") as decoded:
            for row in indices:
                for candidate, payload in zip(GRID, payloads, strict=True):
                    encoded, reconstruction, _ = encode(candidate, carriers[row],
                                                          carriers[donors[row]], model)
                    payload.write(encoded)
                    decoded.write(reconstruction.astype("<f4", copy=False).tobytes())
    finally:
        for payload in payloads:
            payload.close()
    cost = []
    for candidate, path in zip(GRID, payload_paths, strict=True):
        size = path.stat().st_size
        if size % len(indices):
            raise ValueError("nonuniform CAR-1A payload size")
        cost.append({"candidate_id": candidate.candidate_id, "family": candidate.family,
                     "level": candidate.level, "payload_path": path.name,
                     "per_row_bytes": size // len(indices), "stage_payload_bytes": size,
                     "shared_model_bytes": model_bytes_for_rank(candidate.level)
                     if candidate.family == "pca" else 0,
                     "decode_operation_ledger": decode_ledger(candidate),
                     "source_requires_natural_prefix": True})
    artifacts = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in [decoded_path, *payload_paths]]
    artifacts[0].update({"dtype": "f32-le", "shape": [len(indices), len(GRID), WIDTH]})
    document = {
        "schema": "larql.gwcar1a.candidates.v1", "stage": stage,
        "protocol_sha256": protocol["protocol_sha256"],
        "fit_manifest_path": str(fit_path.resolve()),
        "fit_manifest_file_sha256": sha256(fit_path),
        "fit_sha256": fitted["fit_sha256"],
        "capture_file_sha256": protocol["authorities"]["v2_capture"]["sha256"],
        "row_indices": indices, "donor_row_indices": donors,
        "candidate_ids": list(range(len(GRID))), "h1_arms": ["donor", "target_exact_natural"],
        "cost": cost, "artifacts": artifacts,
        "model_outcomes_examined": False,
    }
    document["candidates_sha256"] = canonical_hash(document, "candidates_sha256")
    write_json_new(output / "candidates.json", document)
    return {"candidates_sha256": document["candidates_sha256"], "stage": stage}


def validate_candidates(protocol_path: Path, path: Path) -> dict:
    protocol, rows, carriers, donors = source(protocol_path)
    document = json.loads(path.read_text())
    fit_path = Path(document["fit_manifest_path"])
    fitted, model = validate_fit(protocol_path, fit_path)
    indices = stage_indices(rows, document["stage"])
    if (document.get("schema") != "larql.gwcar1a.candidates.v1"
        or document.get("candidates_sha256") != canonical_hash(document, "candidates_sha256")
        or document.get("protocol_sha256") != protocol["protocol_sha256"]
        or document.get("fit_manifest_file_sha256") != sha256(fit_path)
        or document.get("fit_sha256") != fitted["fit_sha256"]
        or document.get("capture_file_sha256") != protocol["authorities"]["v2_capture"]["sha256"]
        or document.get("row_indices") != indices
        or document.get("donor_row_indices") != donors
        or document.get("candidate_ids") != list(range(len(GRID)))
        or document.get("h1_arms") != ["donor", "target_exact_natural"]
        or document.get("model_outcomes_examined") is not False):
        raise ValueError("CAR-1A candidate authority mismatch")
    expected_cost = []
    for candidate in GRID:
        encoded, _, shared = encode(candidate, carriers[indices[0]],
                                    carriers[donors[indices[0]]], model)
        expected_cost.append({"candidate_id": candidate.candidate_id, "family": candidate.family,
                              "level": candidate.level,
                              "payload_path": f"payload-{candidate.candidate_id:02d}.bin",
                              "per_row_bytes": len(encoded),
                              "stage_payload_bytes": len(encoded) * len(indices),
                              "shared_model_bytes": shared,
                              "decode_operation_ledger": decode_ledger(candidate),
                              "source_requires_natural_prefix": True})
    if document.get("cost") != expected_cost:
        raise ValueError("CAR-1A cost ledger differs from registered encodings")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1 + len(GRID):
        raise ValueError("CAR-1A candidate artifacts missing")
    for position, entry in enumerate(artifacts):
        name = "carriers.f32" if position == 0 else f"payload-{position-1:02d}.bin"
        file = path.parent / name
        if (entry.get("path") != name or entry.get("bytes") != file.stat().st_size
            or entry.get("sha256") != sha256(file)):
            raise ValueError(f"CAR-1A candidate artifact mismatch: {name}")
    if (artifacts[0].get("shape") != [len(indices), len(GRID), WIDTH]
        or artifacts[0].get("dtype") != "f32-le"
        or artifacts[0]["bytes"] != len(indices) * len(GRID) * WIDTH * 4):
        raise ValueError("CAR-1A decoded-carrier tensor geometry changed")
    decoded = np.memmap(path.parent / "carriers.f32", dtype="<f4", mode="r",
                        shape=(len(indices), len(GRID), WIDTH))
    encoded_files = [(path.parent / f"payload-{candidate.candidate_id:02d}.bin").read_bytes()
                     for candidate in GRID]
    for position, row in enumerate(indices):
        for candidate in GRID:
            payload, reconstruction, _ = encode(candidate, carriers[row],
                                                 carriers[donors[row]], model)
            start = position * len(payload)
            stored = encoded_files[candidate.candidate_id][start:start + len(payload)]
            if (stored != payload or
                not np.array_equal(reconstruction.view("<u4"),
                                   decoded[position, candidate.candidate_id].view("<u4"))):
                raise ValueError(f"CAR-1A encoded/decoded mismatch at row {row}, candidate {candidate.candidate_id}")
    return {"candidates_sha256": document["candidates_sha256"], "stage": document["stage"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "fit-validate", "build", "validate"])
    parser.add_argument("protocol", type=Path)
    parser.add_argument("args", nargs="*", help="stage-specific paths")
    args = parser.parse_args()
    if args.command == "fit" and len(args.args) == 1:
        result = fit(args.protocol, Path(args.args[0]))
    elif args.command == "fit-validate" and len(args.args) == 1:
        result = {"fit_sha256": validate_fit(args.protocol, Path(args.args[0]))[0]["fit_sha256"]}
    elif args.command == "build" and len(args.args) == 3:
        result = build(args.protocol, Path(args.args[0]), args.args[1], Path(args.args[2]))
    elif args.command == "validate" and len(args.args) == 1:
        result = validate_candidates(args.protocol, Path(args.args[0]))
    else:
        parser.error("invalid CAR-1A candidate command arguments")
    print(json.dumps(result, indent=2, sort_keys=True))
