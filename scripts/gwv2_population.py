#!/usr/bin/env python3
"""Build and validate the frozen, outcome-free GW-V2 population."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROW_SCHEMA = "larql.gwv2.input-row.v1"
MANIFEST_SCHEMA = "larql.gwv2.population-manifest.v1"
SOURCE_COMMIT = "c8015eebdd94c533358406b0d709f441389e1f2e"
SOURCE_URL = (
    "https://raw.githubusercontent.com/mledoze/countries/"
    f"{SOURCE_COMMIT}/countries.json"
)
SOURCE_SHA256 = "sha256:99b85dda36895c79caf72e191d035e4b9d82e811ee34f9ca15dfa67d7c561ba8"
OLD_ROWS_SHA256 = "sha256:faada97d311f206dae635f23fc4bdb1bd93085ac823e6e16cc2f009d6d954f43"
TOKENIZER_SHA256 = "sha256:4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"
SPLIT_SEED = "larql-gwv2-fresh-subject-split-v1"
PROMPT_FAMILIES = ("canonical", "question", "alternate")
RELATIONS = ("capital", "currency", "language")
SPLITS = ("train", "validation", "test")
OLD_NAME_OVERRIDES = {"Turkey": "TUR"}

TEMPLATES = {
    "capital": {
        "canonical": "The capital of {subject} is",
        "question": "Question: What is the capital of {subject}?\nAnswer:",
        "alternate": "Name the capital city of {subject}:",
    },
    "currency": {
        "canonical": "The currency of {subject} is",
        "question": "Question: what is the currency of {subject}? Answer:",
        "alternate": "Currency of {subject}:",
    },
    "language": {
        "canonical": "The official language of {subject} is",
        "question": "Question: What language is spoken officially in {subject}?\nAnswer:",
        "alternate": "Official language of {subject}:",
    },
}


class PopulationError(ValueError):
    """Raised when a frozen GW-V2 population invariant is violated."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def canonical_hash(value: dict[str, Any], identity_field: str) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    return "sha256:" + hashlib.sha256(canonical_bytes(body)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def short_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_bytes(value)).hexdigest()[:16]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def aliases(country: dict[str, Any]) -> set[str]:
    names = [country["name"]["common"], country["name"]["official"], *country.get("altSpellings", [])]
    return {name.casefold() for name in names}


def old_subject_ids(old_rows: list[dict[str, Any]], countries: list[dict[str, Any]]) -> dict[str, str]:
    old_subjects = sorted(
        {
            row["semantic_edge"]["subject"]
            for row in old_rows
            if row["semantic_edge"]["relation"] in RELATIONS
        }
    )
    by_alias: dict[str, list[str]] = defaultdict(list)
    valid_ids = {country["cca3"] for country in countries}
    for country in countries:
        for name in aliases(country):
            by_alias[name].append(country["cca3"])
    resolved: dict[str, str] = {}
    for subject in old_subjects:
        if subject in OLD_NAME_OVERRIDES:
            country_id = OLD_NAME_OVERRIDES[subject]
            if country_id not in valid_ids:
                raise PopulationError(f"old-subject override is not a source country: {subject}")
            resolved[subject] = country_id
            continue
        matches = sorted(set(by_alias.get(subject.casefold(), [])))
        if len(matches) != 1:
            raise PopulationError(f"old subject does not resolve uniquely: {subject!r} -> {matches}")
        resolved[subject] = matches[0]
    if len(set(resolved.values())) != len(resolved):
        raise PopulationError("multiple old subjects resolve to one country identity")
    return resolved


def eligible_countries(
    countries: list[dict[str, Any]], excluded_ids: set[str]
) -> list[dict[str, Any]]:
    result = []
    for country in countries:
        if not country.get("independent") or not country.get("unMember"):
            continue
        if country["cca3"] in excluded_ids:
            continue
        if len(country.get("capital") or []) != 1:
            continue
        if len(country.get("currencies") or {}) != 1:
            continue
        if len(country.get("languages") or {}) != 1:
            continue
        result.append(country)
    result.sort(
        key=lambda country: hashlib.sha256(
            canonical_bytes([SPLIT_SEED, country["cca3"]])
        ).hexdigest()
    )
    return result


def assign_splits(countries: list[dict[str, Any]]) -> dict[str, str]:
    count = len(countries)
    train_end = round(count * 0.60)
    validation_end = train_end + round(count * 0.20)
    return {
        country["cca3"]: (
            "train" if index < train_end else "validation" if index < validation_end else "test"
        )
        for index, country in enumerate(countries)
    }


def targets(country: dict[str, Any]) -> dict[str, str]:
    currency = next(iter(country["currencies"].values()))["name"]
    language = next(iter(country["languages"].values()))
    return {"capital": country["capital"][0], "currency": currency, "language": language}


def tokenize(tokenizer: Any, prompt: str, target: str) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True).ids
    completed_ids = tokenizer.encode(prompt + " " + target, add_special_tokens=True).ids
    if completed_ids[: len(prompt_ids)] != prompt_ids:
        raise PopulationError(f"prompt/target boundary retokenized for {prompt!r} -> {target!r}")
    continuation = completed_ids[len(prompt_ids) :]
    if not prompt_ids or not continuation:
        raise PopulationError(f"empty prompt or continuation for {prompt!r} -> {target!r}")
    return prompt_ids, continuation


def make_rows(countries: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    split_by_country = assign_splits(countries)
    rows: list[dict[str, Any]] = []
    edge_lookup: dict[tuple[str, str, str], str] = {}
    for country in countries:
        subject = country["name"]["common"]
        for relation, target in targets(country).items():
            semantic_key = [country["cca3"], relation, target]
            for family in PROMPT_FAMILIES:
                prompt = TEMPLATES[relation][family].format(subject=subject)
                prompt_ids, continuation = tokenize(tokenizer, prompt, target)
                edge_id = short_id("gwv2e", [*semantic_key, family, prompt])
                edge_lookup[(country["cca3"], relation, family)] = edge_id
                rows.append(
                    {
                        "schema": ROW_SCHEMA,
                        "edge_id": edge_id,
                        "edge_family_id": short_id("gwv2f", [*semantic_key, family]),
                        "subject_id": country["cca3"],
                        "split": split_by_country[country["cca3"]],
                        "cohort": "fresh_factual_primary",
                        "semantic_edge": {
                            "status": "positive",
                            "subject": subject,
                            "relation": relation,
                            "target": target,
                            "target_token_ids": [continuation[0]],
                            "target_continuation_token_ids": continuation,
                            "prompt_semantic_family": family,
                        },
                        "prompt": {
                            "protocol": "raw_text_completion_container_tokenizer_with_special_tokens",
                            "template_id": f"{relation}-{family}",
                            "text": prompt,
                            "token_ids": prompt_ids,
                            "template_source": "GW-V2 frozen template table v1",
                        },
                        "source": {
                            "kind": "pinned-mledoze-countries-single-valued",
                            "country_id": country["cca3"],
                            "commit": SOURCE_COMMIT,
                        },
                        "capture_request": {
                            "position": len(prompt_ids) - 1,
                            "position_rule": "last_prompt_token",
                            "surfaces": [
                                "natural_l24h1_q",
                                "natural_l24h1_subject_v",
                                "natural_l24_carrier_before",
                                "natural_l24_non_h1_attention",
                                "subject_v_depth_grid",
                                "candidate_readout_before",
                            ],
                        },
                        "control_ids": [],
                        "execution_status": "unexecuted",
                    }
                )

    rows_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        semantic = row["semantic_edge"]
        rows_by_cell[(row["split"], semantic["relation"], semantic["prompt_semantic_family"])].append(row)
    for values in rows_by_cell.values():
        values.sort(key=lambda row: row["edge_id"])

    family_index = {family: index for index, family in enumerate(PROMPT_FAMILIES)}
    for row in rows:
        semantic = row["semantic_edge"]
        family = semantic["prompt_semantic_family"]
        alternate = PROMPT_FAMILIES[(family_index[family] + 1) % len(PROMPT_FAMILIES)]
        same_cell = rows_by_cell[(row["split"], semantic["relation"], family)]
        position = next(index for index, candidate in enumerate(same_cell) if candidate is row)
        different_subject = same_cell[(position + 1) % len(same_cell)]
        row["control_ids"] = [
            {
                "control_id": short_id("gwv2c", [row["edge_id"], "prompt-family"]),
                "kind": "same_fact_different_prompt_family",
                "paired_edge_id": edge_lookup[(row["subject_id"], semantic["relation"], alternate)],
            },
            {
                "control_id": short_id("gwv2c", [row["edge_id"], "subject"]),
                "kind": "same_relation_different_subject",
                "paired_edge_id": different_subject["edge_id"],
            },
        ]
    rows.sort(key=lambda row: (row["split"], row["subject_id"], row["semantic_edge"]["relation"], row["edge_id"]))
    return rows


def validate_rows(rows: list[dict[str, Any]], old_ids: set[str]) -> None:
    if not rows:
        raise PopulationError("population has no rows")
    edge_ids = {row.get("edge_id") for row in rows}
    by_edge = {row["edge_id"]: row for row in rows}
    if None in edge_ids or len(edge_ids) != len(rows):
        raise PopulationError("edge IDs must be unique and present")
    if {row["subject_id"] for row in rows} & old_ids:
        raise PopulationError("fresh population overlaps a prior GW-0 subject identity")
    split_by_subject: dict[str, str] = {}
    groups: Counter[tuple[str, str, str]] = Counter()
    target_by_edge: dict[tuple[str, str], str] = {}
    control_ids: set[str] = set()
    for row in rows:
        if row.get("schema") != ROW_SCHEMA or row.get("execution_status") != "unexecuted":
            raise PopulationError("row is not an unexecuted GW-V2 input")
        split = row.get("split")
        if split not in SPLITS:
            raise PopulationError(f"invalid split: {split}")
        prior = split_by_subject.setdefault(row["subject_id"], split)
        if prior != split:
            raise PopulationError("a subject identity crosses splits")
        semantic = row["semantic_edge"]
        if semantic["relation"] not in RELATIONS or semantic["prompt_semantic_family"] not in PROMPT_FAMILIES:
            raise PopulationError("row leaves the frozen factual relation/prompt-family scope")
        groups[(row["subject_id"], semantic["relation"], semantic["prompt_semantic_family"])] += 1
        target_key = (row["subject_id"], semantic["relation"])
        prior_target = target_by_edge.setdefault(target_key, semantic["target"])
        if prior_target != semantic["target"]:
            raise PopulationError("prompt families disagree on a semantic target")
        prompt_ids = row["prompt"]["token_ids"]
        continuation = semantic["target_continuation_token_ids"]
        if not prompt_ids or not continuation or semantic["target_token_ids"] != [continuation[0]]:
            raise PopulationError("invalid prompt or target tokenization")
        capture = row["capture_request"]
        if capture["position"] != len(prompt_ids) - 1 or capture["position_rule"] != "last_prompt_token":
            raise PopulationError("capture position does not identify the final prompt token")
        controls = row.get("control_ids")
        if not isinstance(controls, list) or len(controls) != 2:
            raise PopulationError("each row must have two matched controls")
        if {control["kind"] for control in controls} != {
            "same_fact_different_prompt_family",
            "same_relation_different_subject",
        }:
            raise PopulationError("matched-control kinds changed")
        for control in controls:
            if control["control_id"] in control_ids:
                raise PopulationError("duplicate control ID")
            control_ids.add(control["control_id"])
            paired = by_edge.get(control["paired_edge_id"])
            if paired is None or paired["split"] != split or paired["edge_id"] == row["edge_id"]:
                raise PopulationError("invalid or cross-split control")
            paired_semantic = paired["semantic_edge"]
            if control["kind"] == "same_fact_different_prompt_family":
                if (
                    paired["subject_id"] != row["subject_id"]
                    or paired_semantic["relation"] != semantic["relation"]
                    or paired_semantic["target"] != semantic["target"]
                    or paired_semantic["prompt_semantic_family"]
                    == semantic["prompt_semantic_family"]
                ):
                    raise PopulationError("same-fact control changed fact or retained prompt family")
            elif (
                paired["subject_id"] == row["subject_id"]
                or paired_semantic["relation"] != semantic["relation"]
                or paired_semantic["prompt_semantic_family"]
                != semantic["prompt_semantic_family"]
            ):
                raise PopulationError("different-fact control is not relation/family matched")
    if any(count != 1 for count in groups.values()):
        raise PopulationError("subject/relation/prompt-family cells are not unique")
    expected = len(split_by_subject) * len(RELATIONS) * len(PROMPT_FAMILIES)
    if len(rows) != expected:
        raise PopulationError("population is not a complete subject x relation x prompt-family product")


def build(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise PopulationError("building requires the Python tokenizers package") from error

    if sha256(args.country_source) != SOURCE_SHA256:
        raise PopulationError("pinned country source hash mismatch")
    if sha256(args.old_rows) != OLD_ROWS_SHA256:
        raise PopulationError("GW-0 exclusion authority changed")
    if sha256(args.tokenizer) != TOKENIZER_SHA256:
        raise PopulationError("tokenizer authority changed")
    countries = json.loads(args.country_source.read_text(encoding="utf-8"))
    old_rows = read_jsonl(args.old_rows)
    resolved_old = old_subject_ids(old_rows, countries)
    selected = eligible_countries(countries, set(resolved_old.values()))
    if len(selected) != 74:
        raise PopulationError(f"frozen selection expected 74 subjects, found {len(selected)}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = make_rows(selected, tokenizer)
    validate_rows(rows, set(resolved_old.values()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "input.jsonl"
    manifest_path = args.output_dir / "population-manifest.json"
    if rows_path.exists() or manifest_path.exists():
        raise PopulationError("refusing to replace an existing frozen GW-V2 population")
    write_jsonl(rows_path, rows)

    subjects = {row["subject_id"] for row in rows}
    semantic_edges = {
        (row["subject_id"], row["semantic_edge"]["relation"], row["semantic_edge"]["target"])
        for row in rows
    }
    split_subjects = Counter()
    split_rows = Counter()
    split_edges = Counter()
    for split in SPLITS:
        split_subjects[split] = len({row["subject_id"] for row in rows if row["split"] == split})
        split_rows[split] = sum(row["split"] == split for row in rows)
        split_edges[split] = len(
            {
                (row["subject_id"], row["semantic_edge"]["relation"])
                for row in rows
                if row["split"] == split
            }
        )
    candidate_tokens = sorted({row["semantic_edge"]["target_token_ids"][0] for row in rows})
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen_fresh_population_unexecuted",
        "frozen_date": "2026-09-21",
        "population_sha256": None,
        "rows": {"path": "input.jsonl", "sha256": sha256(rows_path), "count": len(rows)},
        "model": {
            "name": "Gemma 3 4B IT",
            "model_identity": "093f9f388b31de276ce2de164bdc2081324b9767",
            "component": "target",
            "layers": 34,
            "hidden_size": 2560,
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        "source": {
            "repository": "mledoze/countries",
            "commit": SOURCE_COMMIT,
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "license": "Open Database License (ODbL) 1.0",
            "fields": ["cca3", "independent", "unMember", "name.common", "name.official", "altSpellings", "capital", "currencies", "languages"],
        },
        "freshness": {
            "exclusion_authority": {"path": "../../gw0/gemma3-4b-it-phase1/input.jsonl", "sha256": OLD_ROWS_SHA256},
            "unit": "resolved ISO-3166 alpha-3 country identity",
            "old_subjects": len(resolved_old),
            "resolved_old_subjects": dict(sorted(resolved_old.items())),
            "selected_old_identity_intersection": [],
            "prior_outcomes_read": False,
            "rule": "exclude every country identity used by any factual GW-0 row before eligibility and splitting",
        },
        "selection": {
            "scope": "all source rows satisfying the frozen predicate; no sample or manual adjudication",
            "predicate": [
                "independent is true",
                "unMember is true",
                "country identity absent from factual GW-0",
                "exactly one capital",
                "exactly one currency",
                "exactly one official language",
            ],
            "relations": list(RELATIONS),
            "prompt_families": list(PROMPT_FAMILIES),
            "manual_inclusions": [],
            "manual_exclusions": [],
        },
        "split": {
            "unit": "subject country identity; every relation and prompt family for a subject remains on one side",
            "method": "deterministic SHA-256 order of [seed, cca3]",
            "seed": SPLIT_SEED,
            "fractions": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "subjects": dict(split_subjects),
            "semantic_edges": dict(split_edges),
            "executions": dict(split_rows),
        },
        "counts": {
            "subjects": len(subjects),
            "semantic_edges": len(semantic_edges),
            "executions": len(rows),
            "matched_controls": len(rows) * 2,
            "candidate_first_tokens": len(candidate_tokens),
        },
        "candidate_readout": {
            "construction": "sorted unique first continuation-token IDs from every frozen target annotation",
            "token_ids": candidate_tokens,
            "outcome_independent": True,
        },
        "controls": {
            "same_fact": "cyclic next prompt family for the identical subject/relation/target",
            "different_fact": "cyclic next edge-ID within split x relation x prompt-family",
            "cross_split": False,
        },
        "execution": {"model_prompts_executed": 0, "candidate_effects_observed": 0, "fitting_performed": False},
    }
    manifest["population_sha256"] = canonical_hash(manifest, "population_sha256")
    write_json(manifest_path, manifest)
    return manifest


def validate(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PopulationError("invalid population schema")
    if manifest.get("status") != "frozen_fresh_population_unexecuted":
        raise PopulationError("population is not frozen and unexecuted")
    if canonical_hash(manifest, "population_sha256") != manifest.get("population_sha256"):
        raise PopulationError("population identity mismatch")
    rows_path = manifest_path.parent / manifest["rows"]["path"]
    if sha256(rows_path) != manifest["rows"]["sha256"]:
        raise PopulationError("population rows hash mismatch")
    rows = read_jsonl(rows_path)
    if len(rows) != manifest["rows"]["count"]:
        raise PopulationError("population row count mismatch")
    old_ids = set(manifest["freshness"]["resolved_old_subjects"].values())
    if len(old_ids) != 53 or manifest["freshness"]["selected_old_identity_intersection"]:
        raise PopulationError("fresh-subject exclusion audit changed")
    validate_rows(rows, old_ids)
    if manifest["counts"] != {
        "subjects": 74,
        "semantic_edges": 222,
        "executions": 666,
        "matched_controls": 1332,
        "candidate_first_tokens": len(manifest["candidate_readout"]["token_ids"]),
    }:
        raise PopulationError("frozen population counts changed")
    if manifest["split"]["subjects"] != {"train": 44, "validation": 15, "test": 15}:
        raise PopulationError("frozen subject split changed")
    candidate_tokens = sorted(
        {row["semantic_edge"]["target_token_ids"][0] for row in rows}
    )
    if manifest["candidate_readout"] != {
        "construction": "sorted unique first continuation-token IDs from every frozen target annotation",
        "token_ids": candidate_tokens,
        "outcome_independent": True,
    }:
        raise PopulationError("candidate-readout inventory changed")
    if manifest["execution"] != {
        "model_prompts_executed": 0,
        "candidate_effects_observed": 0,
        "fitting_performed": False,
    }:
        raise PopulationError("population manifest records premature execution or fitting")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--country-source", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument(
        "--old-rows",
        type=Path,
        default=Path("bench/gw0/gemma3-4b-it-phase1/input.jsonl"),
    )
    build_parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/Users/christopherhay/chris-models/gemma3-4b-it.vindex3/tokenizer.json"),
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        document = build(args)
    else:
        document = validate(args.manifest)
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
