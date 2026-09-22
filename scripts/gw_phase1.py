#!/usr/bin/env python3
"""Seal and analyse the first VINDEX3 semantic-WALK research phase.

The tool has three intentionally separate jobs:

* ``seal`` validates a GW-0 census, verifies every referenced artifact, and
  writes a content-addressed manifest;
* ``gw1`` learns relation-only site rankings on one split and emits the full
  transition-candidate-retention versus eligible-site curve on another;
* ``gw3a`` scores externally produced postings lookups against semantic,
  exact-WALK, transition-candidate, and supported-causal surfaces.

It never edits a census. Derived results bind to the sealed bundle hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA = "larql.gw.phase1.manifest.v1"
CENSUS_SCHEMA = "larql.gw0.census-row.v1"
GW1_SCHEMA = "larql.gw1.relation-site-curve.v1"
GW3A_INPUT_SCHEMA = "larql.gw3a.lookup.v1"
GW3A_REPORT_SCHEMA = "larql.gw3a.report.v1"
CAUSAL_STATUSES = {"untested", "supported", "refuted"}
SPLITS = {"train", "validation", "test"}


class GwError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GwError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise GwError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise GwError(f"{path}: census/result file is empty")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def required_object(row: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise GwError(f"{where}: {key} must be an object")
    return value


def required_string(row: dict[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise GwError(f"{where}: {key} must be a non-empty string")
    return value


def site_key(site: dict[str, Any], where: str) -> str:
    layer = site.get("layer")
    role = site.get("site")
    if not isinstance(layer, int) or layer < 0 or not isinstance(role, str) or not role:
        raise GwError(f"{where}: site requires a non-negative layer and non-empty site")
    return f"{layer}:{role}"


def feature_key(value: Any, where: str) -> tuple[int, int]:
    if isinstance(value, dict):
        layer, feature = value.get("layer"), value.get("feature")
    elif isinstance(value, list) and len(value) == 2:
        layer, feature = value
    else:
        raise GwError(f"{where}: feature address must be {{layer,feature}} or [layer,feature]")
    if not isinstance(layer, int) or layer < 0 or not isinstance(feature, int) or feature < 0:
        raise GwError(f"{where}: feature address values must be non-negative integers")
    return layer, feature


def artifact_refs(row: dict[str, Any], where: str) -> list[dict[str, Any]]:
    refs = row.get("artifacts", [])
    if not isinstance(refs, list):
        raise GwError(f"{where}: artifacts must be an array")
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            raise GwError(f"{where}: artifacts[{index}] must be an object")
        required_string(ref, "path", f"{where}.artifacts[{index}]")
        required_string(ref, "sha256", f"{where}.artifacts[{index}]")
    return refs


def validate_rows(rows: list[dict[str, Any]], site_universe: set[str] | None = None) -> None:
    edge_ids: set[str] = set()
    family_splits: dict[str, str] = {}
    for index, row in enumerate(rows):
        where = f"row {index + 1}"
        if row.get("schema") != CENSUS_SCHEMA:
            raise GwError(f"{where}: schema must be {CENSUS_SCHEMA}")
        edge_id = required_string(row, "edge_id", where)
        if edge_id in edge_ids:
            raise GwError(f"{where}: duplicate edge_id {edge_id!r}")
        edge_ids.add(edge_id)
        family = required_string(row, "edge_family_id", where)
        split = required_string(row, "split", where)
        if split not in SPLITS:
            raise GwError(f"{where}: split must be one of {sorted(SPLITS)}")
        prior = family_splits.setdefault(family, split)
        if prior != split:
            raise GwError(f"{where}: edge family {family!r} crosses {prior!r}/{split!r}")

        semantic = required_object(row, "semantic_edge", where)
        for key in ("subject", "relation", "target", "prompt_semantic_family"):
            required_string(semantic, key, f"{where}.semantic_edge")
        if semantic.get("status") not in {"positive", "negative", "discovery"}:
            raise GwError(f"{where}: semantic_edge.status must be positive, negative, or discovery")
        cohort = required_string(row, "cohort", where)
        expected_cohort = {
            "positive": "primary",
            "negative": "negative",
            "discovery": "discovery",
        }[semantic["status"]]
        if cohort != expected_cohort:
            raise GwError(
                f"{where}: semantic status {semantic['status']!r} belongs in cohort {expected_cohort!r}"
            )
        target_ids = semantic.get("target_token_ids", [])
        if not isinstance(target_ids, list) or not all(
            isinstance(token_id, int) and token_id >= 0 for token_id in target_ids
        ):
            raise GwError(f"{where}: semantic_edge.target_token_ids must be non-negative integers")

        exact = required_object(row, "exact_walk_result", where)
        if not isinstance(exact.get("hits"), list):
            raise GwError(f"{where}: exact_walk_result.hits must be an array")
        for hit_index, hit in enumerate(exact["hits"]):
            feature_key(hit, f"{where}.exact_walk_result.hits[{hit_index}]")

        transition = row.get("transition_candidate")
        if transition is not None:
            if not isinstance(transition, dict) or not isinstance(transition.get("sites"), list):
                raise GwError(f"{where}: transition_candidate must be null or carry sites[]")
            for site_index, site in enumerate(transition["sites"]):
                if not isinstance(site, dict):
                    raise GwError(f"{where}: transition_candidate.sites[{site_index}] must be an object")
                key = site_key(site, f"{where}.transition_candidate.sites[{site_index}]")
                if site_universe is not None and key not in site_universe:
                    raise GwError(f"{where}: transition site {key!r} is outside manifest universe")
            features = transition.get("feature_addresses", [])
            if not isinstance(features, list):
                raise GwError(f"{where}: transition_candidate.feature_addresses must be an array")
            for feature_index, feature in enumerate(features):
                feature_key(feature, f"{where}.transition_candidate.feature_addresses[{feature_index}]")

        status = row.get("causal_status")
        evidence = row.get("causal_evidence_ids")
        if status not in CAUSAL_STATUSES or not isinstance(evidence, list) or not all(
            isinstance(item, str) and item for item in evidence
        ):
            raise GwError(f"{where}: invalid causal_status/causal_evidence_ids")
        if status == "untested" and evidence:
            raise GwError(f"{where}: untested rows cannot carry causal evidence")
        if status != "untested" and not evidence:
            raise GwError(f"{where}: {status} requires causal evidence")

        provenance = required_object(row, "provenance", where)
        for key in (
            "model_identity",
            "container_identity",
            "execution_fingerprint",
            "basis_identity",
            "run_record_hash",
        ):
            required_string(provenance, key, f"{where}.provenance")
        if not is_sha256(provenance["run_record_hash"]):
            raise GwError(f"{where}: provenance.run_record_hash must be a sha256 digest")
        if not isinstance(provenance.get("position"), int) or provenance["position"] < 0:
            raise GwError(f"{where}: provenance.position must be a non-negative integer")
        for ref in artifact_refs(row, where):
            if not is_sha256(ref["sha256"]):
                raise GwError(f"{where}: artifact sha256 must be a lowercase sha256 digest")


def resolve_under(root: Path, relative: str, where: str) -> Path:
    candidate = (root / relative).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise GwError(f"{where}: artifact path escapes root: {relative!r}")
    return candidate


def load_site_universe(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise GwError("site universe must be a non-empty JSON string array")
    if len(set(value)) != len(value):
        raise GwError("site universe contains duplicates")
    for item in value:
        layer, separator, site = item.partition(":")
        if not separator or not layer.isdigit() or not site:
            raise GwError(f"invalid site-universe entry {item!r}; expected <layer>:<site>")
    return value


def seal(census_path: Path, artifact_root: Path, site_universe_path: Path, output: Path) -> dict[str, Any]:
    rows = read_jsonl(census_path)
    universe = load_site_universe(site_universe_path)
    validate_rows(rows, set(universe))
    artifacts: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        for ref in artifact_refs(row, f"row {row_index + 1}"):
            relative = ref["path"]
            path = resolve_under(artifact_root, relative, f"row {row_index + 1}")
            if not path.is_file():
                raise GwError(f"missing artifact {path}")
            digest = sha256_file(path)
            if digest != ref["sha256"]:
                raise GwError(f"artifact digest mismatch for {relative}: {digest} != {ref['sha256']}")
            entry = {"path": relative, "sha256": digest, "bytes": path.stat().st_size}
            if relative in artifacts and artifacts[relative] != entry:
                raise GwError(f"artifact {relative!r} has inconsistent references")
            artifacts[relative] = entry

    split_counts = Counter(row["split"] for row in rows)
    cohort_counts = Counter(row["cohort"] for row in rows)
    causal_counts = Counter(row["causal_status"] for row in rows)
    manifest_dir = output.parent.resolve()
    body = {
        "schema": MANIFEST_SCHEMA,
        "census": {
            "path": os.path.relpath(census_path.resolve(), manifest_dir),
            "sha256": sha256_file(census_path),
            "rows": len(rows),
        },
        "artifact_root": os.path.relpath(artifact_root.resolve(), manifest_dir),
        "artifacts": sorted(artifacts.values(), key=lambda entry: entry["path"]),
        "site_universe": universe,
        "counts": {
            "splits": dict(sorted(split_counts.items())),
            "cohorts": dict(sorted(cohort_counts.items())),
            "causal_status": dict(sorted(causal_counts.items())),
        },
        "split_unit": "edge_family_id=(subject canonical identity, relation, target canonical identity, prompt semantic family)",
    }
    body["bundle_sha256"] = sha256_bytes(canonical_bytes(body))
    write_json(output, body)
    return body


def load_sealed(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise GwError(f"{manifest_path}: invalid manifest schema")
    expected_bundle = manifest.get("bundle_sha256")
    body = dict(manifest)
    body.pop("bundle_sha256", None)
    if expected_bundle != sha256_bytes(canonical_bytes(body)):
        raise GwError(f"{manifest_path}: bundle hash mismatch")
    base = manifest_path.parent
    census_path = (base / manifest["census"]["path"]).resolve()
    if sha256_file(census_path) != manifest["census"]["sha256"]:
        raise GwError("sealed census hash mismatch")
    rows = read_jsonl(census_path)
    if len(rows) != manifest["census"]["rows"]:
        raise GwError("sealed census row count mismatch")
    validate_rows(rows, set(manifest["site_universe"]))
    artifact_root = (base / manifest["artifact_root"]).resolve()
    for entry in manifest["artifacts"]:
        path = resolve_under(artifact_root, entry["path"], "manifest")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise GwError(f"sealed artifact changed: {entry['path']}")
    return manifest, rows


def transition_sites(row: dict[str, Any]) -> set[str]:
    transition = row.get("transition_candidate")
    if transition is None:
        return set()
    return {site_key(site, row["edge_id"]) for site in transition["sites"]}


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def gw1(manifest_path: Path, train_split: str, test_split: str, output: Path) -> dict[str, Any]:
    manifest, rows = load_sealed(manifest_path)
    universe: list[str] = manifest["site_universe"]
    order = {site: index for index, site in enumerate(universe)}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["split"] == train_split and row["cohort"] == "primary":
            counts[row["semantic_edge"]["relation"]].update(transition_sites(row))
    # Include zero-count sites after the observed sites. The right edge of the
    # curve must mean "all sites" and therefore reach full retention even when
    # training never saw the held-out site's transition candidate.
    rankings = {
        relation: sorted(universe, key=lambda site: (-counter[site], order[site]))
        for relation, counter in counts.items()
    }

    test_rows = [
        row
        for row in rows
        if row["split"] == test_split
        and row["cohort"] == "primary"
        and transition_sites(row)
    ]
    relations = sorted({row["semantic_edge"]["relation"] for row in test_rows})
    max_budget = len(universe)
    curves: list[dict[str, Any]] = []
    per_relation: dict[str, list[dict[str, Any]]] = {relation: [] for relation in relations}
    for budget in range(max_budget + 1):
        retained = 0
        denominator = 0
        by_relation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in test_rows:
            relation = row["semantic_edge"]["relation"]
            mask = set(rankings.get(relation, [])[:budget])
            hit = bool(mask & transition_sites(row))
            denominator += 1
            retained += int(hit)
            by_relation[relation][1] += 1
            by_relation[relation][0] += int(hit)
        curves.append(
            {
                "site_budget": budget,
                "eligible_site_fraction": budget / max_budget,
                "retained": retained,
                "denominator": denominator,
                "transition_candidate_retention": ratio(retained, denominator),
            }
        )
        for relation in relations:
            rel_retained, rel_denominator = by_relation[relation]
            per_relation[relation].append(
                {
                    "site_budget": budget,
                    "retained": rel_retained,
                    "denominator": rel_denominator,
                    "transition_candidate_retention": ratio(rel_retained, rel_denominator),
                }
            )

    report = {
        "schema": GW1_SCHEMA,
        "bundle_sha256": manifest["bundle_sha256"],
        "train_split": train_split,
        "test_split": test_split,
        "predictors": ["semantic_edge.relation"],
        "site_universe": universe,
        "rankings": rankings,
        "curve": curves,
        "per_relation_curves": per_relation,
        "test_edges_with_transition_candidates": len(test_rows),
    }
    write_json(output, report)
    return report


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def gw3a(manifest_path: Path, lookup_path: Path, output: Path) -> dict[str, Any]:
    manifest, rows = load_sealed(manifest_path)
    census = {row["edge_id"]: row for row in rows}
    lookups = read_jsonl(lookup_path)
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for index, lookup in enumerate(lookups):
        where = f"lookup row {index + 1}"
        if lookup.get("schema") != GW3A_INPUT_SCHEMA:
            raise GwError(f"{where}: schema must be {GW3A_INPUT_SCHEMA}")
        edge_id = required_string(lookup, "edge_id", where)
        operating_point = required_string(lookup, "operating_point", where)
        if edge_id not in census:
            raise GwError(f"{where}: unknown edge_id {edge_id!r}")
        if (edge_id, operating_point) in seen:
            raise GwError(f"{where}: duplicate edge/operating point")
        seen.add((edge_id, operating_point))
        if not isinstance(lookup.get("candidate_features"), list):
            raise GwError(f"{where}: candidate_features must be an array")
        for feature_index, feature in enumerate(lookup["candidate_features"]):
            feature_key(feature, f"{where}.candidate_features[{feature_index}]")
        if not isinstance(lookup.get("promoted_token_ids"), list) or not all(
            isinstance(token_id, int) and token_id >= 0 for token_id in lookup["promoted_token_ids"]
        ):
            raise GwError(f"{where}: promoted_token_ids must be non-negative integers")
        costs = required_object(lookup, "costs", where)
        for key in ("candidate_features", "total_features", "gate_bytes", "other_bytes", "dot_products", "index_bytes"):
            if not isinstance(costs.get(key), int) or costs[key] < 0:
                raise GwError(f"{where}: costs.{key} must be a non-negative integer")
        wall = costs.get("wall_time_ns")
        if wall is not None and (not isinstance(wall, int) or wall < 0):
            raise GwError(f"{where}: costs.wall_time_ns must be null or non-negative")
        if costs["candidate_features"] != len(set(feature_key(value, where) for value in lookup["candidate_features"])):
            raise GwError(f"{where}: candidate feature count disagrees with payload")
        grouped[operating_point].append((census[edge_id], lookup))

    reports = []
    for operating_point, pairs in sorted(grouped.items()):
        exact_n = exact_d = exact_edge_n = exact_edge_d = 0
        transition_n = transition_d = transition_edge_n = transition_edge_d = 0
        causal_n = causal_d = causal_edge_n = causal_edge_d = 0
        semantic_n = semantic_d = 0
        fractions: list[float] = []
        cost_values: dict[str, list[float]] = defaultdict(list)
        for row, lookup in pairs:
            candidates = {feature_key(value, row["edge_id"]) for value in lookup["candidate_features"]}
            exact = {feature_key(value, row["edge_id"]) for value in row["exact_walk_result"]["hits"]}
            if exact:
                overlap = candidates & exact
                exact_d += len(exact)
                exact_n += len(overlap)
                exact_edge_d += 1
                exact_edge_n += int(bool(overlap))
            transition = row.get("transition_candidate") or {}
            transition_features = {
                feature_key(value, row["edge_id"])
                for value in transition.get("feature_addresses", [])
            }
            if transition_features:
                hit = bool(candidates & transition_features)
                overlap = candidates & transition_features
                transition_d += len(transition_features)
                transition_n += len(overlap)
                transition_edge_d += 1
                transition_edge_n += int(hit)
                if row["causal_status"] == "supported":
                    causal_d += len(transition_features)
                    causal_n += len(overlap)
                    causal_edge_d += 1
                    causal_edge_n += int(hit)
            target_ids = row["semantic_edge"].get("target_token_ids", [])
            if row["cohort"] == "primary" and target_ids:
                semantic_d += 1
                semantic_n += int(bool(set(target_ids) & set(lookup["promoted_token_ids"])))
            costs = lookup["costs"]
            if costs["total_features"]:
                fractions.append(costs["candidate_features"] / costs["total_features"])
            for key, value in costs.items():
                if value is not None:
                    cost_values[key].append(float(value))
        reports.append(
            {
                "operating_point": operating_point,
                "rows": len(pairs),
                "exact_walk_recall": {"value": ratio(exact_n, exact_d), "hits": exact_n, "denominator": exact_d},
                "exact_walk_edge_retention": {"value": ratio(exact_edge_n, exact_edge_d), "hits": exact_edge_n, "denominator": exact_edge_d},
                "transition_candidate_recall": {"value": ratio(transition_n, transition_d), "hits": transition_n, "denominator": transition_d},
                "transition_candidate_edge_retention": {"value": ratio(transition_edge_n, transition_edge_d), "hits": transition_edge_n, "denominator": transition_edge_d},
                "supported_causal_recall": {"value": ratio(causal_n, causal_d), "hits": causal_n, "denominator": causal_d},
                "supported_causal_edge_retention": {"value": ratio(causal_edge_n, causal_edge_d), "hits": causal_edge_n, "denominator": causal_edge_d},
                "semantic_target_recall": {"value": ratio(semantic_n, semantic_d), "hits": semantic_n, "denominator": semantic_d},
                "mean_candidate_fraction": mean(fractions),
                "mean_costs": {key: mean(values) for key, values in sorted(cost_values.items())},
            }
        )
    report = {
        "schema": GW3A_REPORT_SCHEMA,
        "bundle_sha256": manifest["bundle_sha256"],
        "lookup_sha256": sha256_file(lookup_path),
        "operating_points": reports,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal_parser = commands.add_parser("seal", help="validate and seal a GW-0 census")
    seal_parser.add_argument("--census", type=Path, required=True)
    seal_parser.add_argument("--artifacts", type=Path, required=True)
    seal_parser.add_argument("--site-universe", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate", help="revalidate a sealed bundle")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    gw1_parser = commands.add_parser("gw1", help="evaluate relation-only site pruning")
    gw1_parser.add_argument("--manifest", type=Path, required=True)
    gw1_parser.add_argument("--train-split", default="train", choices=sorted(SPLITS))
    gw1_parser.add_argument("--test-split", default="test", choices=sorted(SPLITS))
    gw1_parser.add_argument("--output", type=Path, required=True)
    gw3a_parser = commands.add_parser("gw3a", help="evaluate postings lookup results")
    gw3a_parser.add_argument("--manifest", type=Path, required=True)
    gw3a_parser.add_argument("--lookups", type=Path, required=True)
    gw3a_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "seal":
            result = seal(args.census, args.artifacts, args.site_universe, args.output)
            print(result["bundle_sha256"])
        elif args.command == "validate":
            manifest, rows = load_sealed(args.manifest)
            print(f"{manifest['bundle_sha256']} {len(rows)} rows")
        elif args.command == "gw1":
            result = gw1(args.manifest, args.train_split, args.test_split, args.output)
            print(f"{len(result['curve'])} operating points")
        else:
            result = gw3a(args.manifest, args.lookups, args.output)
            print(f"{len(result['operating_points'])} operating points")
    except (GwError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
