#!/usr/bin/env python3
"""Build and validate the frozen GW-0 input manifest without executing a model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROW_SCHEMA = "larql.gw0.input-row.v1"
MANIFEST_SCHEMA = "larql.gw0.input-manifest.v1"
SPLIT_SEED = "larql-gw0-gemma3-4b-phase1-v1"
PROMPT_FAMILIES = ("canonical", "question", "alternate")
SPLITS = ("train", "validation", "test")


FACTUAL_TEMPLATES = {
    "capital": {
        "canonical": ("The capital of {subject} is", "E20/e20_banked_highway.py:GOLD+RELATIONS"),
        "question": (
            "Question: What is the capital of {subject}?\nAnswer:",
            "MI13/run_mi13.py:TEMPLATES.capital-of.question",
        ),
        "alternate": (
            "Name the capital city of {subject}:",
            "MI13/run_mi13.py:TEMPLATES.capital-of.direct",
        ),
    },
    "currency": {
        "canonical": ("The currency of {subject} is", "E20/e20_banked_highway.py:GOLD+RELATIONS"),
        "question": (
            "Question: what is the currency of {subject}? Answer:",
            "semantic_graph/sg_rep1.py:TEMPLATES_B.currency",
        ),
        "alternate": (
            "Currency of {subject}:",
            "MI11/discover_templates_from_kg.py:surface_forms_currency",
        ),
    },
    "language": {
        "canonical": (
            "The official language of {subject} is",
            "E20/e20_banked_highway.py:GOLD+RELATIONS",
        ),
        "question": (
            "Question: What language is spoken officially in {subject}?\nAnswer:",
            "MI13/run_mi13.py:TEMPLATES.language-of.question",
        ),
        "alternate": (
            "Official language of {subject}:",
            "MI13/run_mi13.py:TEMPLATES.language-of.direct",
        ),
    },
}

HYPERNYM_TEMPLATE_IDS = {
    "canonical": "hyp-1",
    "question": "hyp-3",
    "alternate": "hyp-5",
}


class ManifestError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def short_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_bytes(value)).hexdigest()[:16]}"


def ast_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise ManifestError(f"{path}: no literal assignment named {name}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ManifestError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def tokenized_prompt_and_continuation(tokenizer: Any, prompt: str, target: str) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True).ids
    completed_ids = tokenizer.encode(prompt + " " + target, add_special_tokens=True).ids
    if completed_ids[: len(prompt_ids)] != prompt_ids:
        raise ManifestError(f"prompt/target boundary retokenized for target {target!r}")
    continuation = completed_ids[len(prompt_ids) :]
    if not prompt_ids or not continuation:
        raise ManifestError(f"target {target!r} has no continuation tokens")
    return prompt_ids, continuation


def select_hypernyms(path: Path, count: int) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in read_jsonl(path):
        if row.get("arm") != "core" or row.get("relation") != "hypernym":
            continue
        if row.get("entity_split") != "test" or row.get("template_id") not in HYPERNYM_TEMPLATE_IDS.values():
            continue
        subject = row["subject"]
        if subject not in by_subject:
            order.append(subject)
        by_subject[subject].append(row)

    selected = []
    for subject in order:
        rows = by_subject[subject]
        targets = {target for row in rows for target in row.get("targets", [])}
        templates = {row["template_id"] for row in rows}
        if len(targets) != 1 or templates != set(HYPERNYM_TEMPLATE_IDS.values()):
            continue
        selected.append({"subject": subject, "target": next(iter(targets)), "rows": rows})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ManifestError(f"wanted {count} sealed test hypernyms, found {len(selected)}")
    return selected


def assign_splits(triples: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], str]:
    by_relation: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for triple in triples:
        by_relation[triple[1]].append(triple)
    result = {}
    for relation, values in sorted(by_relation.items()):
        values.sort(key=lambda value: hashlib.sha256(canonical_bytes([SPLIT_SEED, *value])).hexdigest())
        count = len(values)
        train_end = round(count * 0.60)
        validation_end = train_end + round(count * 0.20)
        for index, value in enumerate(values):
            result[value] = "train" if index < train_end else "validation" if index < validation_end else "test"
    return result


def make_rows(
    gold: dict[str, dict[str, str]],
    hypernyms: list[dict[str, Any]],
    tokenizer: Any,
    factual_source: Path,
    lexical_source: Path,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    triples: list[tuple[str, str, str]] = []
    for relation in ("capital", "currency", "language"):
        for subject, target in gold[relation].items():
            triple = (subject, relation, target)
            triples.append(triple)
            for family in PROMPT_FAMILIES:
                template, template_source = FACTUAL_TEMPLATES[relation][family]
                items.append(
                    {
                        "triple": triple,
                        "family": family,
                        "prompt": template.format(subject=subject),
                        "template_id": f"{relation}-{family}",
                        "template_source": template_source,
                        "source": {"path": str(factual_source), "kind": "sealed-gold"},
                    }
                )

    for item in hypernyms:
        triple = (item["subject"], "hypernym", item["target"])
        triples.append(triple)
        rows_by_template = {row["template_id"]: row for row in item["rows"]}
        for family, template_id in HYPERNYM_TEMPLATE_IDS.items():
            source_row = rows_by_template[template_id]
            items.append(
                {
                    "triple": triple,
                    "family": family,
                    "prompt": source_row["prompt"],
                    "template_id": template_id,
                    "template_source": "semantic_graph/captures/sg1_manifest_full.jsonl",
                    "source": {
                        "path": str(lexical_source),
                        "kind": "sealed-wordnet-capture-manifest",
                        "source_row_idx": source_row["idx"],
                    },
                }
            )

    split_by_triple = assign_splits(list(dict.fromkeys(triples)))
    rows = []
    lookup: dict[tuple[tuple[str, str, str], str], str] = {}
    for item in items:
        triple = item["triple"]
        family_key = [*triple, item["family"]]
        edge_family_id = short_id("gw0f", family_key)
        edge_id = short_id("gw0e", [*family_key, item["prompt"]])
        lookup[(triple, item["family"])] = edge_id
        prompt_ids, full_target_ids = tokenized_prompt_and_continuation(tokenizer, item["prompt"], triple[2])
        rows.append(
            {
                "schema": ROW_SCHEMA,
                "edge_id": edge_id,
                "edge_family_id": edge_family_id,
                "split": split_by_triple[triple],
                "cohort": "primary",
                "semantic_edge": {
                    "status": "positive",
                    "subject": triple[0],
                    "relation": triple[1],
                    "target": triple[2],
                    "target_token_ids": [full_target_ids[0]],
                    "target_continuation_token_ids": full_target_ids,
                    "prompt_semantic_family": item["family"],
                },
                "prompt": {
                    "protocol": "raw_text_completion_container_tokenizer_with_special_tokens",
                    "template_id": item["template_id"],
                    "text": item["prompt"],
                    "token_ids": prompt_ids,
                    "template_source": item["template_source"],
                },
                "source": item["source"],
                "capture_request": {
                    "position": len(prompt_ids) - 1,
                    "position_rule": "last_prompt_token",
                    "surfaces": [
                        "exact_v3_walk",
                        "relation_readable_band",
                        "emergence_transition",
                        "candidate_read_operators",
                        "carrier_writes",
                        "gate_feature_ranking",
                    ],
                },
                "control_ids": [],
                "execution_status": "unexecuted",
            }
        )

    rows_by_relation_family_split: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        semantic = row["semantic_edge"]
        key = (semantic["relation"], semantic["prompt_semantic_family"], row["split"])
        rows_by_relation_family_split[key].append(row)
    for values in rows_by_relation_family_split.values():
        values.sort(key=lambda row: row["edge_id"])

    family_index = {family: index for index, family in enumerate(PROMPT_FAMILIES)}
    for row in rows:
        semantic = row["semantic_edge"]
        triple = (semantic["subject"], semantic["relation"], semantic["target"])
        family = semantic["prompt_semantic_family"]
        alternate_family = PROMPT_FAMILIES[(family_index[family] + 1) % len(PROMPT_FAMILIES)]
        same_relation = rows_by_relation_family_split[(semantic["relation"], family, row["split"])]
        position = next(index for index, candidate in enumerate(same_relation) if candidate is row)
        other_subject = same_relation[(position + 1) % len(same_relation)]
        controls = [
            {
                "control_id": short_id("gw0c", [row["edge_id"], "prompt-family"]),
                "kind": "same_fact_different_prompt_family",
                "paired_edge_id": lookup[(triple, alternate_family)],
            },
            {
                "control_id": short_id("gw0c", [row["edge_id"], "subject"]),
                "kind": "same_relation_different_subject",
                "paired_edge_id": other_subject["edge_id"],
            },
        ]
        row["control_ids"] = controls

    rows.sort(key=lambda row: (row["semantic_edge"]["relation"], row["edge_id"]))
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ManifestError("input manifest has no rows")
    edge_ids = {row.get("edge_id") for row in rows}
    split_by_edge = {row.get("edge_id"): row.get("split") for row in rows}
    if None in edge_ids or len(edge_ids) != len(rows):
        raise ManifestError("edge IDs must be present and unique")
    split_by_triple: dict[tuple[str, str, str], str] = {}
    control_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        if row.get("schema") != ROW_SCHEMA or row.get("execution_status") != "unexecuted":
            raise ManifestError(f"row {index}: invalid schema or execution status")
        semantic = row.get("semantic_edge", {})
        triple = tuple(semantic.get(key) for key in ("subject", "relation", "target"))
        if not all(isinstance(value, str) and value for value in triple):
            raise ManifestError(f"row {index}: incomplete semantic triple")
        split = row.get("split")
        if split not in SPLITS:
            raise ManifestError(f"row {index}: invalid split {split!r}")
        prior = split_by_triple.setdefault(triple, split)
        if prior != split:
            raise ManifestError(f"row {index}: semantic triple crosses splits")
        if semantic.get("prompt_semantic_family") not in PROMPT_FAMILIES:
            raise ManifestError(f"row {index}: invalid prompt family")
        target_ids = semantic.get("target_token_ids")
        continuation = semantic.get("target_continuation_token_ids")
        if not isinstance(target_ids, list) or len(target_ids) != 1:
            raise ManifestError(f"row {index}: target must identify exactly one next-token target")
        if not isinstance(continuation, list) or not continuation or target_ids[0] != continuation[0]:
            raise ManifestError(f"row {index}: invalid full target continuation")
        controls = row.get("control_ids")
        if not isinstance(controls, list) or len(controls) != 2:
            raise ManifestError(f"row {index}: expected two matched controls")
        for control in controls:
            control_id = control.get("control_id")
            if not isinstance(control_id, str) or control_id in control_ids:
                raise ManifestError(f"row {index}: control IDs must be present and unique")
            control_ids.add(control_id)
            if control.get("paired_edge_id") not in edge_ids or control["paired_edge_id"] == row["edge_id"]:
                raise ManifestError(f"row {index}: invalid paired control edge")
            if split_by_edge[control["paired_edge_id"]] != split:
                raise ManifestError(f"row {index}: paired control edge crosses splits")
        prompt = row.get("prompt", {})
        prompt_ids = prompt.get("token_ids")
        capture = row.get("capture_request", {})
        if not isinstance(prompt_ids, list) or not prompt_ids or not all(isinstance(value, int) for value in prompt_ids):
            raise ManifestError(f"row {index}: prompt token IDs are missing")
        if capture.get("position") != len(prompt_ids) - 1 or capture.get("position_rule") != "last_prompt_token":
            raise ManifestError(f"row {index}: capture position disagrees with prompt tokens")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ManifestError("building requires the Python 'tokenizers' package") from error

    container_index = json.loads(args.container_index.read_text(encoding="utf-8"))
    system_graph = json.loads(args.system_graph.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    gold = ast_assignment(args.factual_source, "GOLD")
    hypernyms = select_hypernyms(args.lexical_source, args.hypernyms)
    rows = make_rows(gold, hypernyms, tokenizer, args.factual_source, args.lexical_source)
    validate_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "input.jsonl"
    write_jsonl(rows_path, rows)
    source_paths = [
        args.factual_source,
        args.lexical_source,
        args.capital_language_templates,
        args.currency_question_templates,
        args.currency_direct_templates,
    ]
    split_counts = Counter(row["split"] for row in rows)
    relation_counts = Counter(row["semantic_edge"]["relation"] for row in rows)
    relation_edges = Counter(
        (row["semantic_edge"]["relation"], row["semantic_edge"]["subject"], row["semantic_edge"]["target"])
        for row in rows
    )
    profile = container_index.get("profiles", {})
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen_input_unexecuted",
        "frozen_date": "2026-09-20",
        "rows": {"path": rows_path.name, "sha256": digest_file(rows_path), "count": len(rows)},
        "model": {
            "name": "Gemma 3 4B IT",
            "model_identity": container_index["model"],
            "family": container_index["family"],
            "num_layers": container_index["num_layers"],
            "hidden_size": container_index["hidden_size"],
            "container_root": str(args.container_index.parent),
            "container_identity": digest_file(args.container_index),
            "index_sha256": digest_file(args.container_index),
            "system_graph_sha256": digest_file(args.system_graph),
            "tokenizer_sha256": digest_file(args.tokenizer),
            "profiles": sorted(profile) if isinstance(profile, dict) else profile,
            "text_component": next(
                component for component in system_graph["components"] if component.get("role") == "primary_text"
            )["id"],
        },
        "prompt_protocol": "raw_text_completion_container_tokenizer_with_special_tokens",
        "split_policy": {
            "unit": "semantic triple; all three prompt families remain on one side",
            "method": "relation-stratified deterministic SHA-256 ordering",
            "fractions": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "seed": SPLIT_SEED,
        },
        "selection": {
            "factual": "all E20 verified GOLD entries for capital/currency/language",
            "hypernym": f"first {args.hypernyms} single-target test subjects in sealed SG-1 source order",
            "prompt_families": list(PROMPT_FAMILIES),
            "target_token_rule": "encode(prompt + ' ' + target) minus the exact encode(prompt) prefix; target_token_ids contains the first next-token target and target_continuation_token_ids preserves the full suffix",
        },
        "controls": {
            "per_execution": 2,
            "kinds": ["same_fact_different_prompt_family", "same_relation_different_subject"],
            "execution_policy": "paired rows are already in the positive input manifest; controls add no executions",
        },
        "counts": {
            "executions": len(rows),
            "controls": len(rows) * 2,
            "semantic_edges": len(relation_edges),
            "by_relation_executions": dict(sorted(relation_counts.items())),
            "by_relation_semantic_edges": dict(sorted(Counter(key[0] for key in relation_edges).items())),
            "by_split_executions": dict(sorted(split_counts.items())),
        },
        "sources": [
            {"path": str(path), "sha256": digest_file(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
    }
    manifest["manifest_sha256"] = digest_bytes(canonical_bytes(manifest))
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def validate(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError("invalid input manifest schema")
    expected = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if expected != digest_bytes(canonical_bytes(body)):
        raise ManifestError("input manifest hash mismatch")
    rows_path = manifest_path.parent / manifest["rows"]["path"]
    if digest_file(rows_path) != manifest["rows"]["sha256"]:
        raise ManifestError("input rows hash mismatch")
    rows = read_jsonl(rows_path)
    if len(rows) != manifest["rows"]["count"]:
        raise ManifestError("input row count mismatch")
    validate_rows(rows)
    for source in manifest["sources"]:
        path = Path(source["path"])
        if not path.is_file() or path.stat().st_size != source["bytes"] or digest_file(path) != source["sha256"]:
            raise ManifestError(f"source changed: {path}")
    for field in ("index_sha256", "system_graph_sha256", "tokenizer_sha256"):
        if not isinstance(manifest["model"].get(field), str):
            raise ManifestError(f"missing model {field}")
    container_root = Path(manifest["model"]["container_root"])
    for filename, field in (
        ("index.json", "index_sha256"),
        ("system_graph.json", "system_graph_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
    ):
        path = container_root / filename
        if not path.is_file() or digest_file(path) != manifest["model"][field]:
            raise ManifestError(f"container input changed: {path}")
    return manifest


def parser() -> argparse.ArgumentParser:
    root = Path("/Users/christopherhay/chris-source/chris-experiments")
    models = Path("/Users/christopherhay/chris-models/gemma3-4b-it.vindex3")
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--container-index", type=Path, default=models / "index.json")
    build_parser.add_argument("--system-graph", type=Path, default=models / "system_graph.json")
    build_parser.add_argument("--tokenizer", type=Path, default=models / "tokenizer.json")
    build_parser.add_argument(
        "--factual-source", type=Path, default=root / "fleet/E20_banked_highway/e20_banked_highway.py"
    )
    build_parser.add_argument(
        "--lexical-source", type=Path, default=root / "semantic_graph/captures/sg1_manifest_full.jsonl"
    )
    build_parser.add_argument(
        "--capital-language-templates",
        type=Path,
        default=root / "mechinterp/MI13_attention_graph_walk/run_mi13.py",
    )
    build_parser.add_argument(
        "--currency-question-templates", type=Path, default=root / "semantic_graph/sg_rep1.py"
    )
    build_parser.add_argument(
        "--currency-direct-templates",
        type=Path,
        default=root / "mechinterp/MI11_cascade_trie/discover_templates_from_kg.py",
    )
    build_parser.add_argument("--hypernyms", type=int, default=40)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    manifest = build(args) if args.command == "build" else validate(args.manifest)
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ManifestError, OSError, json.JSONDecodeError, KeyError) as error:
        raise SystemExit(f"error: {error}") from error
