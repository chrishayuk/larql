#!/usr/bin/env python3
"""Fit frozen GW-READ-1 Q/K/V candidate tensors using train rows only."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Hashable

import numpy as np


LAYERS = [0, 4, 8, 12, 16, 20, 23, 24]
SOURCE_LAYERS = LAYERS[:-1]
RANKS = [8, 16, 32, 64, 128]
HEAD_DIM = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def semantic_edge_identity(input_row: dict[str, Any]) -> str:
    """Identify the transition without reading its held-out destination."""
    semantic = input_row["semantic_edge"]
    return "\x1f".join([semantic["subject"], semantic["relation"]])


def artifact(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in manifest["artifacts"] if item["path"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one artifact {name}")
    return matches[0]


def checked_array(root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    descriptor = artifact(manifest, name)
    path = root / name
    if sha256(path) != descriptor["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    shape = tuple(descriptor["shape"])
    values = np.fromfile(path, dtype="<f4")
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"invalid tensor: {path}")
    return values.reshape(shape)


def write_array(path: Path, values: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise ValueError(f"refusing to replace {path}")
    values = np.asarray(values, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite output: {path}")
    values.tofile(path)
    return {
        "path": path.name,
        "dtype": "f32-le",
        "shape": list(values.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_u8(path: Path, values: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise ValueError(f"refusing to replace {path}")
    values = np.asarray(values, dtype=np.uint8)
    values.tofile(path)
    return {
        "path": path.name,
        "dtype": "u8",
        "shape": list(values.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def fit_ridge(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc
    lam = 1e-3 * float(np.trace(gram)) / x.shape[1]
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("degenerate frozen ridge fit")
    weight = np.linalg.solve(gram + lam * np.eye(x.shape[1]), xc.T @ yc)
    u, singular, vt = np.linalg.svd(weight, full_matrices=False)
    for column in range(u.shape[1]):
        pivot = int(np.argmax(np.abs(u[:, column])))
        if u[pivot, column] < 0:
            u[:, column] *= -1
            vt[column, :] *= -1
    return x_mean, y_mean, u, singular, vt


def rank_predictions(
    x: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[int, np.ndarray]:
    x_mean, y_mean, u, singular, vt = model
    centered = x.astype(np.float64, copy=False) - x_mean
    result: dict[int, np.ndarray] = {}
    for rank in RANKS:
        weight = (u[:, :rank] * singular[:rank]) @ vt[:rank, :]
        result[rank] = (centered @ weight + y_mean).astype(np.float32)
    return result


def loo_ridge(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[str],
    edges: list[str],
) -> dict[int, np.ndarray]:
    train = np.array([split == "train" for split in splits])
    outputs = {rank: np.empty_like(y, dtype=np.float32) for rank in RANKS}
    full = fit_ridge(x[train], y[train])
    heldout = ~train
    full_predictions = rank_predictions(x[heldout], full)
    for rank in RANKS:
        outputs[rank][heldout] = full_predictions[rank]
    for edge in sorted({edges[index] for index in np.flatnonzero(train)}):
        target = np.array([train[index] and edges[index] == edge for index in range(len(edges))])
        model = fit_ridge(x[train & ~target], y[train & ~target])
        predictions = rank_predictions(x[target], model)
        for rank in RANKS:
            outputs[rank][target] = predictions[rank]
    return outputs


def loo_mean(
    values: np.ndarray,
    rows: list[dict[str, Any]],
    grouping: Callable[[dict[str, Any]], Hashable],
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros_like(values)
    available = np.zeros(len(rows), dtype=np.uint8)
    sums: dict[Hashable, np.ndarray] = defaultdict(lambda: np.zeros(values.shape[1], dtype=np.float64))
    counts: dict[Hashable, int] = defaultdict(int)
    edge_sums: dict[tuple[Hashable, str], np.ndarray] = defaultdict(lambda: np.zeros(values.shape[1], dtype=np.float64))
    edge_counts: dict[tuple[Hashable, str], int] = defaultdict(int)
    for index, row in enumerate(rows):
        if row["split"] != "train":
            continue
        key = grouping(row)
        edge_key = (key, row["semantic_edge"])
        sums[key] += values[index]
        counts[key] += 1
        edge_sums[edge_key] += values[index]
        edge_counts[edge_key] += 1
    for index, row in enumerate(rows):
        key = grouping(row)
        total = sums[key].copy()
        count = counts[key]
        if row["split"] == "train":
            edge_key = (key, row["semantic_edge"])
            total -= edge_sums[edge_key]
            count -= edge_counts[edge_key]
        if count > 0:
            result[index] = total / count
            available[index] = 1
    return result, available


def scaled_cell_rows(
    natural: np.ndarray,
    position_rows: list[dict[str, Any]],
    grouping: Callable[[dict[str, Any]], Hashable],
    loo: bool,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros_like(natural)
    available = np.zeros(len(position_rows), dtype=np.uint8)
    sums: dict[Hashable, np.ndarray] = defaultdict(lambda: np.zeros(HEAD_DIM, dtype=np.float64))
    norm_sums: dict[Hashable, float] = defaultdict(float)
    counts: dict[Hashable, int] = defaultdict(int)
    edge_sums: dict[tuple[Hashable, str], np.ndarray] = defaultdict(lambda: np.zeros(HEAD_DIM, dtype=np.float64))
    edge_norm_sums: dict[tuple[Hashable, str], float] = defaultdict(float)
    edge_counts: dict[tuple[Hashable, str], int] = defaultdict(int)
    for index, row in enumerate(position_rows):
        if row["split"] != "train":
            continue
        key = grouping(row)
        edge_key = (key, row["semantic_edge"])
        vector = natural[index].astype(np.float64)
        vector_norm = float(np.linalg.norm(vector))
        sums[key] += vector
        norm_sums[key] += vector_norm
        counts[key] += 1
        edge_sums[edge_key] += vector
        edge_norm_sums[edge_key] += vector_norm
        edge_counts[edge_key] += 1
    for index, row in enumerate(position_rows):
        key = grouping(row)
        total = sums[key].copy()
        norm_total = norm_sums[key]
        count = counts[key]
        if loo and row["split"] == "train":
            edge_key = (key, row["semantic_edge"])
            total -= edge_sums[edge_key]
            norm_total -= edge_norm_sums[edge_key]
            count -= edge_counts[edge_key]
        if count <= 0:
            continue
        direction = total / count
        direction_norm = float(np.linalg.norm(direction))
        target_norm = norm_total / count
        if direction_norm <= 0 or target_norm <= 0:
            continue
        result[index] = direction * (target_norm / direction_norm)
        available[index] = 1
    return result, available


def dynamic_reference_rows(
    natural: np.ndarray,
    position_rows: list[dict[str, Any]],
) -> np.ndarray:
    static_direction, available = scaled_cell_rows(
        natural,
        position_rows,
        lambda row: (row["template_id"], row["role"]),
        loo=True,
    )
    if not available.all():
        raise ValueError("GW-KEY template/role reference cell is incomplete")
    direction_norms = np.linalg.norm(static_direction.astype(np.float64), axis=1)
    target_norms = np.linalg.norm(natural.astype(np.float64), axis=1)
    return (static_direction * (target_norms / direction_norms)[:, None]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "candidate-artifact-manifest.json"
    if manifest_path.exists():
        raise ValueError("GW-READ-1 candidate artifact is already sealed")

    protocol = json.loads(args.protocol.read_text())
    capture = json.loads(args.capture.read_text())
    if protocol["schema"] != "larql.gwread1.protocol.v2" or capture["schema"] != "larql.gwread1.candidate-input-capture.v1":
        raise ValueError("inadmissible GW-READ-1 authority")
    if capture["protocol"]["sha256"] != protocol["protocol_sha256"] or capture["outcomes_captured"]:
        raise ValueError("capture does not bind the frozen outcome-free protocol")
    protocol_root = args.protocol.parent
    input_path = (protocol_root / protocol["authorities"]["input_rows"]["path"]).resolve()
    role_path = (protocol_root / protocol["authorities"]["source_roles"]["path"]).resolve()
    source_manifest_path = (protocol_root / protocol["authorities"]["gwkey1_source_capture"]["path"]).resolve()
    if sha256(input_path) != protocol["authorities"]["input_rows"]["sha256"] or sha256(role_path) != protocol["authorities"]["source_roles"]["sha256"]:
        raise ValueError("frozen input authority changed")

    capture_root = args.capture.parent
    original_rows = load_jsonl(capture_root / "original-rows.jsonl")
    entity_rows = load_jsonl(capture_root / "entity-only-rows.jsonl")
    input_rows = load_jsonl(input_path)
    role_rows = load_jsonl(role_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    source_root = source_manifest_path.parent
    source_rows = load_jsonl(source_root / "source-rows.jsonl")
    if not all(len(rows) == 426 for rows in (original_rows, input_rows, role_rows, source_rows)):
        raise ValueError("GW-READ-1 row count changed")
    q = checked_array(capture_root, capture, "original-q.f32")
    subject_v = checked_array(capture_root, capture, "original-subject-v.f32")
    entity_v = checked_array(capture_root, capture, "entity-only-subject-v.f32")
    source_k = checked_array(source_root, source_manifest, "source-k.f32")
    source_v = checked_array(source_root, source_manifest, "source-v.f32")

    row_meta: list[dict[str, Any]] = []
    for index, (original, source) in enumerate(zip(original_rows, source_rows, strict=True)):
        if original["row"] != index or original["edge_id"] != source["edge_id"]:
            raise ValueError("capture row order changed")
        row_meta.append({
            **original,
            "semantic_edge": semantic_edge_identity(input_rows[index]),
        })

    position_rows: list[dict[str, Any]] = []
    role_names = ["bos_system", "subject_entity", "relation_query", "instruction_template", "answer_cue", "punctuation_separator"]
    for index, (row, roles, source) in enumerate(zip(row_meta, role_rows, source_rows, strict=True)):
        count = source["source_count"]
        role_at = [None] * count
        ordinal_at = [None] * count
        for role in role_names:
            for ordinal, position in enumerate(roles["roles"][role]):
                if role_at[position] is not None:
                    raise ValueError("source roles overlap")
                role_at[position] = role
                ordinal_at[position] = ordinal
        if any(role is None for role in role_at):
            raise ValueError("source roles are not exhaustive")
        for position in range(count):
            position_rows.append({
                "original_row": index,
                "split": row["split"],
                "semantic_edge": row["semantic_edge"],
                "template_id": row["template_id"],
                "prompt_family": row["prompt_semantic_family"],
                "relation": row["relation"],
                "role": role_at[position],
                "ordinal": ordinal_at[position],
                "position": position,
            })
    if len(position_rows) != source_k.shape[0]:
        raise ValueError("source position expansion changed")

    q_candidates: list[np.ndarray] = []
    q_ids: list[str] = []
    q_available: list[np.ndarray] = []
    for name, grouping in [
        ("template-relation", lambda row: (row["template_id"], row["relation"])),
        ("relation", lambda row: row["relation"]),
        ("global", lambda row: "global"),
    ]:
        values, available = loo_mean(q[:, -1, :], row_meta, grouping)
        q_ids.append(f"q/train-mean/{name}")
        q_candidates.append(values)
        q_available.append(available)
    row_splits = [row["split"] for row in row_meta]
    row_edges = [row["semantic_edge"] for row in row_meta]
    for layer_index, layer in enumerate(SOURCE_LAYERS):
        q_ids.append(f"q/layer{layer}-native")
        q_candidates.append(q[:, layer_index, :].copy())
        q_available.append(np.ones(426, dtype=np.uint8))
        predictions = loo_ridge(q[:, layer_index, :], q[:, -1, :], row_splits, row_edges)
        for rank in RANKS:
            q_ids.append(f"q/layer{layer}-ridge-r{rank}")
            q_candidates.append(predictions[rank])
            q_available.append(np.ones(426, dtype=np.uint8))
        print(f"fit Q layer {layer}", flush=True)

    k_candidates: list[np.ndarray] = []
    k_ids: list[str] = []
    k_available: list[np.ndarray] = []
    for name, grouping in [
        ("template-role-ordinal", lambda row: (row["template_id"], row["role"], row["ordinal"])),
        ("prompt-family-role-ordinal", lambda row: (row["prompt_family"], row["role"], row["ordinal"])),
        ("relation-role-ordinal", lambda row: (row["relation"], row["role"], row["ordinal"])),
        ("global-role-ordinal", lambda row: (row["role"], row["ordinal"])),
    ]:
        values, available = scaled_cell_rows(source_k, position_rows, grouping, loo=True)
        k_ids.append(f"k/{name}-mean")
        k_candidates.append(values)
        k_available.append(available)

    fixed_k, fixed_k_available = scaled_cell_rows(
        source_k,
        position_rows,
        lambda row: (row["template_id"], row["role"]),
        loo=True,
    )
    fixed_v, fixed_v_available = scaled_cell_rows(
        source_v,
        position_rows,
        lambda row: (row["template_id"], row["role"]),
        loo=True,
    )
    exact_reference_k = dynamic_reference_rows(source_k, position_rows)
    exact_reference_v = dynamic_reference_rows(source_v, position_rows)

    subject_position_rows: list[dict[str, Any]] = []
    source_subject_indices: list[int] = []
    for row in row_meta:
        original_row = row["row"]
        source_offset = source_rows[original_row]["source_offset"]
        for ordinal, position in enumerate(row["subject_positions"]):
            subject_position_rows.append({
                **row,
                "subject_ordinal": ordinal,
                "subject_tokens_key": tuple(row["subject_token_ids"]),
            })
            source_subject_indices.append(source_offset + position)
    if len(subject_position_rows) != subject_v.shape[0]:
        raise ValueError("subject position expansion changed")

    v_candidates: list[np.ndarray] = []
    v_ids: list[str] = []
    v_available: list[np.ndarray] = []
    cross_prompt = np.zeros_like(subject_v[:, -1, :])
    cross_available = np.zeros(subject_v.shape[0], dtype=np.uint8)
    by_edge: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(row_meta):
        by_edge[row["semantic_edge"]].append(row_index)
    for members in by_edge.values():
        members.sort(key=lambda index: row_meta[index]["prompt_semantic_family"])
        for source_row, donor_row in zip(members, members[1:] + members[:1], strict=True):
            source_meta = row_meta[source_row]
            donor_meta = row_meta[donor_row]
            if source_meta["subject_count"] != donor_meta["subject_count"]:
                continue
            source_slice = slice(source_meta["subject_offset"], source_meta["subject_offset"] + source_meta["subject_count"])
            donor_slice = slice(donor_meta["subject_offset"], donor_meta["subject_offset"] + donor_meta["subject_count"])
            cross_prompt[source_slice] = subject_v[donor_slice, -1, :]
            cross_available[source_slice] = 1
    v_ids.append("v/same-fact-cross-prompt-cycle")
    v_candidates.append(cross_prompt)
    v_available.append(cross_available)

    for name, grouping in [
        ("cache-entity-mean", lambda row: (row["subject_tokens_key"], row["subject_ordinal"])),
        ("cache-entity-relation-mean", lambda row: (row["subject_tokens_key"], row["relation"], row["subject_ordinal"])),
    ]:
        values, available = loo_mean(subject_v[:, -1, :], subject_position_rows, grouping)
        v_ids.append(f"v/{name}")
        v_candidates.append(values)
        v_available.append(available)

    entity_by_tokens = {tuple(row["subject_token_ids"]): row for row in entity_rows}
    entity_only_l24 = np.zeros_like(subject_v[:, -1, :])
    entity_only_available = np.zeros(subject_v.shape[0], dtype=np.uint8)
    for index, row in enumerate(subject_position_rows):
        entity = entity_by_tokens.get(row["subject_tokens_key"])
        if entity is None or row["subject_ordinal"] >= entity["subject_count"]:
            continue
        source_index = entity["subject_offset"] + row["subject_ordinal"]
        entity_only_l24[index] = entity_v[source_index, -1, :]
        entity_only_available[index] = 1
    for name in ("entity-only-L24", "cache-entity-only-L24"):
        v_ids.append(f"v/{name}")
        v_candidates.append(entity_only_l24.copy())
        v_available.append(entity_only_available.copy())

    subject_splits = [row["split"] for row in subject_position_rows]
    subject_edges = [row["semantic_edge"] for row in subject_position_rows]
    for layer_index, layer in enumerate(SOURCE_LAYERS):
        v_ids.append(f"v/layer{layer}-native")
        v_candidates.append(subject_v[:, layer_index, :].copy())
        v_available.append(np.ones(subject_v.shape[0], dtype=np.uint8))
        predictions = loo_ridge(
            subject_v[:, layer_index, :],
            subject_v[:, -1, :],
            subject_splits,
            subject_edges,
        )
        for rank in RANKS:
            v_ids.append(f"v/layer{layer}-ridge-r{rank}")
            v_candidates.append(predictions[rank])
            v_available.append(np.ones(subject_v.shape[0], dtype=np.uint8))
        print(f"fit V layer {layer}", flush=True)

    artifacts = [
        write_array(args.output / "q-candidates.f32", np.stack(q_candidates)),
        write_u8(args.output / "q-availability.u8", np.stack(q_available)),
        write_array(args.output / "k-candidates.f32", np.stack(k_candidates)),
        write_u8(args.output / "k-availability.u8", np.stack(k_available)),
        write_array(args.output / "v-candidates.f32", np.stack(v_candidates)),
        write_u8(args.output / "v-availability.u8", np.stack(v_available)),
        write_array(args.output / "fixed-static-k.f32", fixed_k),
        write_u8(args.output / "fixed-static-k-availability.u8", fixed_k_available),
        write_array(args.output / "fixed-static-v.f32", fixed_v),
        write_u8(args.output / "fixed-static-v-availability.u8", fixed_v_available),
        write_array(args.output / "exact-reference-k.f32", exact_reference_k),
        write_array(args.output / "exact-reference-v.f32", exact_reference_v),
    ]
    candidate_manifest = {
        "schema": "larql.gwread1.candidate-artifact.v1",
        "status": "train_only_candidates_fitted_pre_effect_execution",
        "protocol_sha256": protocol["protocol_sha256"],
        "capture_file_sha256": sha256(args.capture),
        "source_capture_file_sha256": sha256(source_manifest_path),
        "fit": {
            "selection_rows": "train only",
            "train_predictions": "leave-one-semantic-edge-out",
            "heldout_predictions": "single all-train fit",
            "heldout_outcomes_read": False,
            "causal_effects_read": False,
            "semantic_targets_read": False,
        },
        "candidates": {
            "Q": q_ids,
            "K": k_ids,
            "V": v_ids,
        },
        "counts": {
            "Q": len(q_ids),
            "K": len(k_ids),
            "V": len(v_ids),
            "source_rows": len(position_rows),
            "subject_rows": len(subject_position_rows),
        },
        "fixed_treatments": {
            "READ-1V_K": "train template/role direction and cell-mean norm",
            "READ-1V_non_entity_V": "train template/role direction and cell-mean norm",
            "GWKEY_exact_reference": "train template/role direction scaled to target natural norm",
        },
        "availability": {
            "unit": "Q per prompt row; K per source row; V per subject-token row",
            "zero": "hard refusal, never fallback or imputation",
        },
        "artifacts": artifacts,
    }
    encoded = json.dumps(candidate_manifest, indent=2, sort_keys=True).encode() + b"\n"
    with manifest_path.open("xb") as handle:
        handle.write(encoded)
    print(json.dumps(candidate_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
