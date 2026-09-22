#!/usr/bin/env python3
"""Evaluate frozen GW-3A-F postings against GW-0B FFN execution support."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent))
from gw3af_preregister import validate as validate_preregistration

WIDTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
TOP_NS = [20, 50, 100]
NULL_TRIALS = 1000
SEED = 0x4757334146


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def quantiles(values: Iterable[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": at(0.5),
        "p05": at(0.05),
        "p95": at(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def entropy(weights: list[float]) -> float | None:
    total = sum(weights)
    if total <= 0:
        return None
    probabilities = [value / total for value in weights if value > 0]
    return -sum(value * math.log(value) for value in probabilities)


def breakdowns(rows: list[dict]) -> dict:
    dimensions = {
        "site_layer": lambda row: str(row["site_layer"]),
        "relation": lambda row: row["relation"],
        "prompt_family": lambda row: row["prompt_family"],
        "split": lambda row: row["split"],
        "edge_family": lambda row: row["edge_family"],
    }
    output = {}
    for name, key_fn in dimensions.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        output[name] = {
            key: {
                "rows": len(group),
                "address_recall": ratio(
                    sum(row["hits"] for row in group),
                    sum(row["denominator"] for row in group),
                ),
                "contribution_l2_mass_recall": ratio(
                    sum(row["mass_hit"] for row in group),
                    sum(row["mass_denominator"] for row in group),
                ),
                "address_weighted_candidate_coverage": ratio(
                    sum(row["candidates"] for row in group),
                    sum(row["eligible_features"] for row in group),
                ),
            }
            for key, group in sorted(groups.items())
        }
    return output


def address_set(layer_rows: list[dict], field: str) -> set[tuple[int, int]]:
    return {
        (layer_row["layer"], feature)
        for layer_row in layer_rows
        for feature in layer_row[field]
    }


def validate_input_manifest(path: Path, census: list[dict]) -> list[dict]:
    manifest = json.loads(path.read_text())
    expected = census[0]["provenance"]["input_manifest_sha256"]
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("frozen input manifest does not match sealed census provenance")
    rows_path = path.parent / manifest["rows"]["path"]
    if sha(rows_path) != manifest["rows"]["sha256"]:
        raise ValueError("frozen input rows hash mismatch")
    rows = jsonl(rows_path)
    if [row["edge_id"] for row in rows] != [row["edge_id"] for row in census]:
        raise ValueError("frozen input row order differs from sealed census")
    return rows


def semantic_null(
    census: list[dict],
    input_rows: list[dict],
    promotions: dict[tuple[int, int], set[int]],
    vocab_size: int,
) -> dict:
    token_frequency = Counter()
    for row in input_rows:
        token_frequency.update(row["prompt"]["token_ids"])
        token_frequency.update(row["semantic_edge"]["target_continuation_token_ids"])
    pools: dict[int, list[int]] = defaultdict(list)
    for token_id in range(vocab_size):
        pools[token_frequency[token_id]].append(token_id)
    true_targets = {
        row["semantic_edge"]["target_token_ids"][0] for row in census
    }
    for frequency in list(pools):
        pools[frequency] = [token for token in pools[frequency] if token not in true_targets]

    triples: dict[tuple[str, str, str], tuple[int, set[tuple[int, int]]]] = {}
    for row in census:
        semantic = row["semantic_edge"]
        key = (semantic["subject"], semantic["relation"], semantic["target"])
        target = semantic["target_token_ids"][0]
        walk = {(hit["layer"], hit["feature"]) for hit in row["exact_walk_result"]["hits"]}
        previous = triples.setdefault(key, (target, walk))
        if previous != (target, walk):
            raise ValueError(f"prompt-family WALK instability for {key}")

    eligible = []
    exclusions = []
    for triple, (target, walk) in sorted(triples.items()):
        frequency = token_frequency[target]
        pool = pools[frequency]
        if not pool:
            exclusions.append({"triple": triple, "target_token_id": target, "frequency": frequency})
        else:
            eligible.append((triple, target, walk, pool, frequency))
    observed = sum(
        any(target in promotions.get(address, set()) for address in walk)
        for _, target, walk, _, _ in eligible
    )
    rng = random.Random(SEED)
    null_hits = []
    for _ in range(NULL_TRIALS):
        hits = 0
        for _, _, walk, pool, _ in eligible:
            token = pool[rng.randrange(len(pool))]
            hits += any(token in promotions.get(address, set()) for address in walk)
        null_hits.append(hits)
    return {
        "name": "same-WALK-layer-profile/frozen-corpus-exact-token-frequency-null",
        "interpretation": (
            "The address set, including its layer profile and search surface, is fixed. "
            "Target IDs are replaced by non-target vocabulary IDs with exactly the same "
            "frequency in the sealed prompts plus target continuations."
        ),
        "trials": NULL_TRIALS,
        "seed": SEED,
        "eligible_unique_edges": len(eligible),
        "excluded_unique_edges": exclusions,
        "observed_hits": observed,
        "observed_rate": ratio(observed, len(eligible)),
        "null_hits": quantiles(null_hits),
        "one_sided_empirical_p": ratio(
            1 + sum(value >= observed for value in null_hits), NULL_TRIALS + 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--attributions", type=Path, required=True)
    parser.add_argument("--promotions", type=Path, required=True)
    parser.add_argument("--postings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError(f"GW-3A-F output already exists: {args.output}")
    preregistration_result = validate_preregistration(args.preregistration)
    preregistration = json.loads(args.preregistration.read_text())
    preregistration_sha256 = preregistration_result["preregistration_sha256"]
    if preregistration["posting_rule"]["widths"] != WIDTHS:
        raise ValueError("evaluator width ladder differs from preregistration")
    if preregistration["metrics"]["descriptive_execution_top_ns"] != TOP_NS:
        raise ValueError("evaluator support depths differ from preregistration")
    null_contract = preregistration["controls"]["semantic_null"]
    if null_contract["trials"] != NULL_TRIALS or null_contract["seed"] != SEED:
        raise ValueError("semantic null differs from preregistration")

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "larql.gw.phase1.manifest.v1":
        raise ValueError("not a sealed phase-one manifest")
    census_path = args.manifest.parent / manifest["census"]["path"]
    if sha(census_path) != manifest["census"]["sha256"]:
        raise ValueError("sealed census hash mismatch")
    census = jsonl(census_path)
    bundle = manifest["bundle_sha256"]
    authorities = preregistration["authorities"]
    if sha(args.manifest) != authorities["sealed_gw0"]["file_sha256"]:
        raise ValueError("sealed manifest argument differs from preregistration")
    if sha(args.input_manifest) != authorities["input_manifest"]["file_sha256"]:
        raise ValueError("input manifest argument differs from preregistration")
    if sha(args.promotions) != authorities["gw0b_promotions"]["sha256"]:
        raise ValueError("promotion argument differs from preregistration")
    input_rows = validate_input_manifest(args.input_manifest, census)
    input_manifest = json.loads(args.input_manifest.read_text())
    hidden_size = input_manifest["model"]["hidden_size"]

    promotions = {}
    for row in jsonl(args.promotions):
        if row["bundle_sha256"] != bundle:
            raise ValueError("promotion bundle binding mismatch")
        address = (row["address"]["layer"], row["address"]["feature"])
        if address in promotions:
            raise ValueError(f"duplicate promotion address {address}")
        promotions[address] = {item["token_id"] for item in row["candidates"]}

    attributions = {}
    attribution_files = []
    for path in sorted(args.attributions.glob("layer-*.jsonl")):
        rows = jsonl(path)
        attribution_files.append({"path": path.name, "sha256": sha(path), "rows": len(rows)})
        for row in rows:
            if row["bundle_sha256"] != bundle or row["status"] != "reconstructed":
                raise ValueError(f"unbound or refused attribution in {path}")
            if len(row["contributions"]) != 100:
                raise ValueError(f"{row['edge_id']}: attribution is not top-100")
            if row["edge_id"] in attributions:
                raise ValueError(f"duplicate FFN attribution row {row['edge_id']}")
            attributions[row["edge_id"]] = row
    expected = {
        row["edge_id"]
        for row in census
        if any(site["site"] == "ffn" for site in row["transition_candidate"]["sites"])
    }
    if set(attributions) != expected:
        raise ValueError("attribution rows do not equal the FFN-eligible census")
    reconciliation = json.loads(
        (
            args.preregistration.parent
            / authorities["gw0b_reconciliation"]["path"]
        ).resolve().read_text()
    )
    if attribution_files != reconciliation["input_artifacts"]["attributions"]:
        raise ValueError("attribution directory differs from preregistration")

    lookups = {}
    vocab_sizes = set()
    plan_ids = set()
    container_ids = set()
    for row in jsonl(args.postings):
        if (
            row.get("schema") != "larql.gw3af.postings.v1"
            or row["bundle_sha256"] != bundle
            or row.get("preregistration_sha256") != preregistration_sha256
        ):
            raise ValueError("invalid or unbound postings row")
        width = row["width_per_source_token_per_layer"]
        if width not in WIDTHS:
            raise ValueError(f"undeclared postings width {width}")
        key = (row["subject"], width)
        if key in lookups:
            raise ValueError(f"duplicate postings row {key}")
        if row["costs"]["candidate_features"] != len(address_set(row["layers"], "postings")):
            raise ValueError(f"postings candidate count mismatch for {key}")
        for layer_row in row["layers"]:
            if len(layer_row["postings"]) != len(layer_row["exact_walk_matched"]):
                raise ValueError(f"matched exact-WALK count mismatch for {key}")
        lookups[key] = row
        vocab_sizes.add(row["vocab_size"])
        plan_ids.add(row["plan_sha256"])
        container_ids.add(row["container_identity"])
    subjects = {row["semantic_edge"]["subject"] for row in census}
    if set(lookups) != {(subject, width) for subject in subjects for width in WIDTHS}:
        raise ValueError("postings subject/width grid is incomplete")
    if len(vocab_sizes) != 1 or len(plan_ids) != 1 or len(container_ids) != 1:
        raise ValueError("postings physical identity is inconsistent")

    records = []
    random_control = random.Random(SEED)
    for census_row in census:
        edge_id = census_row["edge_id"]
        if edge_id not in attributions:
            continue
        attribution = attributions[edge_id]
        layer = attribution["site"]["layer"]
        contributions = attribution["contributions"]
        walk = {
            (item["layer"], item["feature"])
            for item in census_row["exact_walk_result"]["hits"]
        }
        target = census_row["semantic_edge"]["target_token_ids"][0]
        for width in WIDTHS:
            lookup = lookups[(census_row["semantic_edge"]["subject"], width)]
            postings = address_set(lookup["layers"], "postings")
            exact = address_set(lookup["layers"], "exact_walk_matched")
            layer_row = next(item for item in lookup["layers"] if item["layer"] == layer)
            oracle = {(layer, feature) for feature in layer_row["postings"]}
            eligible = layer_row["eligible_features"]
            random_features = random_control.sample(range(eligible), len(oracle))
            random_addresses = {(layer, feature) for feature in random_features}
            methods = {
                "postings_unmasked": postings,
                "exact_walk_matched_candidates": exact,
                "layer_matched_random": random_addresses,
                "oracle_execution_layer_postings": oracle,
            }
            for method, candidates in methods.items():
                if method == "oracle_execution_layer_postings":
                    denominator = eligible
                elif method == "layer_matched_random":
                    denominator = eligible
                else:
                    denominator = lookup["costs"]["eligible_features"]
                if method == "postings_unmasked":
                    accounting = {
                        "gate_bytes": lookup["costs"]["gate_bytes_at_lookup"],
                        "other_bytes": lookup["costs"]["logical_bytes_touched"],
                        "dot_products": lookup["costs"]["dot_products_at_lookup"],
                        "index_bytes": lookup["costs"]["logical_index_bytes"],
                    }
                elif method == "oracle_execution_layer_postings":
                    source_entries = sum(
                        len(item["features"])
                        for item in layer_row["source_token_postings"]
                    )
                    accounting = {
                        "gate_bytes": 0,
                        "other_bytes": source_entries * 12,
                        "dot_products": 0,
                        "index_bytes": lookup["costs"]["logical_index_bytes"],
                    }
                elif method == "exact_walk_matched_candidates":
                    accounting = {
                        "gate_bytes": denominator * hidden_size * 4,
                        "other_bytes": len(candidates) * 8,
                        "dot_products": denominator,
                        "index_bytes": 0,
                    }
                else:
                    accounting = {
                        "gate_bytes": None,
                        "other_bytes": None,
                        "dot_products": None,
                        "index_bytes": None,
                    }
                for top_n in TOP_NS:
                    top = contributions[:top_n]
                    addresses = {(layer, item["feature"]) for item in top}
                    masses = { (layer, item["feature"]): item["contribution_l2"] for item in top }
                    records.append(
                        {
                            "edge_id": edge_id,
                            "relation": census_row["semantic_edge"]["relation"],
                            "site_layer": layer,
                            "prompt_family": census_row["semantic_edge"]["prompt_semantic_family"],
                            "split": census_row["split"],
                            "edge_family": census_row["edge_family_id"],
                            "width": width,
                            "top_n": top_n,
                            "method": method,
                            "hits": len(candidates & addresses),
                            "denominator": len(addresses),
                            "mass_hit": sum(masses[address] for address in candidates & addresses),
                            "mass_denominator": sum(masses.values()),
                            "candidates": len(candidates),
                            "eligible_features": denominator,
                            "exact_walk_hits": len(candidates & walk),
                            "exact_walk_denominator": len(walk),
                            "semantic_hit_known": any(
                                target in promotions.get(address, set()) for address in candidates
                            ),
                            "annotated_candidates": sum(address in promotions for address in candidates),
                            "accounting": accounting,
                        }
                    )

    curves = []
    for method in (
        "postings_unmasked",
        "exact_walk_matched_candidates",
        "layer_matched_random",
        "oracle_execution_layer_postings",
    ):
        for width in WIDTHS:
            for top_n in TOP_NS:
                selected = [
                    row for row in records
                    if row["method"] == method and row["width"] == width and row["top_n"] == top_n
                ]
                hits = sum(row["hits"] for row in selected)
                address_denominator = sum(row["denominator"] for row in selected)
                mass_hit = sum(row["mass_hit"] for row in selected)
                mass_denominator = sum(row["mass_denominator"] for row in selected)
                coverage = [row["candidates"] / row["eligible_features"] for row in selected]
                address_weighted_coverage = ratio(
                    sum(row["candidates"] for row in selected),
                    sum(row["eligible_features"] for row in selected),
                )
                address_per_row = [row["hits"] / row["denominator"] for row in selected]
                mass_per_row = [row["mass_hit"] / row["mass_denominator"] for row in selected]
                curves.append(
                    {
                        "method": method,
                        "width": width,
                        "execution_top_n": top_n,
                        "rows": len(selected),
                        "address_recall": {
                            "value": ratio(hits, address_denominator),
                            "hits": hits,
                            "denominator": address_denominator,
                            "per_row": quantiles(address_per_row),
                        },
                        "contribution_l2_mass_recall": {
                            "value": ratio(mass_hit, mass_denominator),
                            "hit_mass": mass_hit,
                            "denominator_mass": mass_denominator,
                            "per_row": quantiles(mass_per_row),
                        },
                        "candidate_fraction": {
                            "address_weighted_value": address_weighted_coverage,
                            "per_row": quantiles(coverage),
                        },
                        "candidate_count": {
                            "sum": sum(row["candidates"] for row in selected),
                            "per_row": quantiles(row["candidates"] for row in selected),
                        },
                        "eligible_address_count": {
                            "sum": sum(row["eligible_features"] for row in selected),
                            "per_row": quantiles(row["eligible_features"] for row in selected),
                        },
                        "accounting": {
                            key: {
                                (
                                    "shared_index_bytes"
                                    if key == "index_bytes"
                                    else "total_across_site_queries"
                                ): max(values) if key == "index_bytes" else sum(values),
                                "per_row": quantiles(values),
                            }
                            for key in ("gate_bytes", "other_bytes", "dot_products", "index_bytes")
                            if (
                                values := [
                                    row["accounting"][key]
                                    for row in selected
                                    if row["accounting"][key] is not None
                                ]
                            )
                        },
                        "exact_walk_top20_recall": ratio(
                            sum(row["exact_walk_hits"] for row in selected),
                            sum(row["exact_walk_denominator"] for row in selected),
                        ),
                        "semantic_target_recovery_lower_bound": {
                            "value": ratio(sum(row["semantic_hit_known"] for row in selected), len(selected)),
                            "hits": sum(row["semantic_hit_known"] for row in selected),
                            "denominator": len(selected),
                            "annotation_coverage": ratio(
                                sum(row["annotated_candidates"] for row in selected),
                                sum(row["candidates"] for row in selected),
                            ),
                            "qualification": (
                                "Only addresses already annotated by GW-0B are scored; an unannotated "
                                "candidate cannot be called a semantic miss. The annotated universe was "
                                "selected from exact-WALK and execution addresses, so this is a descriptive "
                                "lower bound and is not comparable semantic recall across retrieval arms."
                            ),
                        },
                        "breakdowns": breakdowns(selected),
                    }
                )

    primary = [
        row for row in curves
        if row["method"] == "postings_unmasked" and row["execution_top_n"] == 100
    ]
    qualifying = [
        row for row in primary
        if row["address_recall"]["value"] >= 0.95
        and row["candidate_fraction"]["address_weighted_value"] <= 0.10
    ]
    audits = []
    for census_row in census:
        edge_id = census_row["edge_id"]
        attribution = attributions.get(edge_id)
        if attribution is None:
            continue
        layer = attribution["site"]["layer"]
        execution = {(layer, item["feature"]) for item in attribution["contributions"]}
        walk = {(item["layer"], item["feature"]) for item in census_row["exact_walk_result"]["hits"]}
        target = census_row["semantic_edge"]["target_token_ids"][0]
        semantic_walk = any(target in promotions[address] for address in walk)
        semantic_execution = any(target in promotions[address] for address in execution)
        if not (semantic_walk and semantic_execution and walk & execution):
            continue
        contribution_weights = [item["contribution_l2"] for item in attribution["contributions"]]
        walk_weights = [
            abs(item["score"])
            for item in census_row["exact_walk_result"]["hits"]
            if item["layer"] == layer
        ]
        contribution_total = sum(contribution_weights)
        contribution_probabilities = [
            value / contribution_total for value in contribution_weights
        ]
        independent_hits = {}
        for width in WIDTHS:
            lookup = lookups[(census_row["semantic_edge"]["subject"], width)]
            token_addresses: dict[int, set[tuple[int, int]]] = defaultdict(set)
            for layer_row in lookup["layers"]:
                for token_posting in layer_row["source_token_postings"]:
                    token_addresses[token_posting["token_id"]].update(
                        (layer_row["layer"], feature)
                        for feature in token_posting["features"]
                    )
            independent_hits[str(width)] = sum(
                bool(addresses & execution) for addresses in token_addresses.values()
            )
        audits.append(
            {
                "edge_id": edge_id,
                "relation": census_row["semantic_edge"]["relation"],
                "emergence": census_row["emergence"],
                "target_confidence": census_row["target_readout"]["final"],
                "prompt_family": census_row["semantic_edge"]["prompt_semantic_family"],
                "execution_layer": layer,
                "contributor_concentration": {
                    "herfindahl": sum(value * value for value in contribution_probabilities),
                    "top1_l2_share": ratio(contribution_weights[0], contribution_total),
                    "top10_l2_share": ratio(sum(contribution_weights[:10]), contribution_total),
                    "entropy_nats": entropy(contribution_weights),
                },
                "walk_top20_absolute_score_entropy_nats": entropy(walk_weights),
                "walk_top20_absolute_score_entropy_normalized": ratio(
                    entropy(walk_weights), math.log(len(walk_weights))
                ),
                "largest_contribution_dominance": ratio(contribution_weights[0], contribution_total),
                "independent_source_token_postings_hitting_execution": independent_hits,
                "distinct_source_tokens": len(set(lookups[(census_row["semantic_edge"]["subject"], WIDTHS[0])]["subject_token_ids"])),
            }
        )

    expected_aligned = preregistration["aligned_audit"]["expected_rows"]
    if len(audits) != expected_aligned:
        raise ValueError(
            f"fully agreeing audit changed: expected {expected_aligned}, got {len(audits)}"
        )

    report = {
        "schema": "larql.gw3af.execution-postings-report.v1",
        "bundle_sha256": bundle,
        "claim_boundary": (
            "Observational top-N FFN contribution support is the retrieval target. "
            "No causal-support claim is made; attention-only rows are excluded."
        ),
        "preregistration": {
            "path": str(args.preregistration),
            "identity_sha256": preregistration_sha256,
            "file_sha256": sha(args.preregistration),
            "widths": WIDTHS,
            "execution_top_ns": TOP_NS,
            "primary_method": "postings_unmasked",
            "primary_gate": "top-100 micro address recall >= 0.95 at address-weighted eligible-universe coverage <= 0.10",
            "candidate_fraction_denominator": (
                "the lookup's eligible address universe: all declared dense FFN layers for the "
                "unmasked primary and matched exact control; the observed execution layer only "
                "for the explicitly non-progressive oracle and layer-matched random control"
            ),
            "interpretation_contract": {
                "high_address_high_mass": "stable support set recovered",
                "low_address_high_mass": "dominant computation recovered but support identity unstable",
                "high_address_low_mass": "many addresses recovered but important computation missed",
                "low_address_low_mass": "postings fail as execution support",
            },
        },
        "inputs": {
            "sealed_manifest": {"path": str(args.manifest), "sha256": sha(args.manifest)},
            "frozen_input_manifest": {"path": str(args.input_manifest), "sha256": sha(args.input_manifest)},
            "promotions": {"path": str(args.promotions), "sha256": sha(args.promotions)},
            "postings": {"path": str(args.postings), "sha256": sha(args.postings)},
            "attributions": attribution_files,
            "container_identity": next(iter(container_ids)),
            "plan_sha256": next(iter(plan_ids)),
        },
        "denominators": {
            "sealed_rows": len(census),
            "ffn_eligible_rows": len(attributions),
            "attention_only_excluded_rows": len(census) - len(attributions),
            "supported_causal_rows": sum(row["causal_status"] == "supported" for row in census),
        },
        "curves": curves,
        "primary_site_values": [
            row for row in records if row["method"] == "postings_unmasked"
        ],
        "lookup_costs": [
            {
                "width": width,
                "subjects": len(subjects),
                "logical_index_bytes": sorted({
                    lookups[(subject, width)]["costs"]["logical_index_bytes"]
                    for subject in subjects
                }),
                "candidate_features": quantiles(
                    lookups[(subject, width)]["costs"]["candidate_features"]
                    for subject in subjects
                ),
                "candidate_fraction": quantiles(
                    lookups[(subject, width)]["costs"]["candidate_fraction"]
                    for subject in subjects
                ),
                "logical_bytes_touched": quantiles(
                    lookups[(subject, width)]["costs"]["logical_bytes_touched"]
                    for subject in subjects
                ),
                "source_postings_touched": quantiles(
                    lookups[(subject, width)]["costs"]["source_postings_touched"]
                    for subject in subjects
                ),
                "gate_bytes_at_lookup": 0,
                "dot_products_at_lookup": 0,
                "wall_time_ns": None,
                "wall_time_status": "not measured; no performance claim",
            }
            for width in WIDTHS
        ],
        "progression": {
            "passed": bool(qualifying),
            "qualifying_widths": [row["width"] for row in qualifying],
            "decision_rule": "curve only; no interpolation or post-hoc operating point",
        },
        "supported_causal_recall": {
            "value": None,
            "status": "N/A",
            "reason": "the sealed GW-0 corpus contains no supported causal rows",
        },
        "semantic_walk_null": semantic_null(
            census, input_rows, promotions, next(iter(vocab_sizes))
        ),
        "fully_agreeing_rows_descriptive_audit": {
            "progressive": False,
            "rows": audits,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "progression": report["progression"],
        "semantic_walk_null": report["semantic_walk_null"],
        "fully_agreeing_rows": len(audits),
    }, indent=2))


if __name__ == "__main__":
    main()
