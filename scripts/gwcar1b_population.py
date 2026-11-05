#!/usr/bin/env python3
"""Build/verify an outcome-free fresh-capital population proposal, never a seal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gwcar1b_preregister import excluded_subjects
from gwv2_population import (
    PROMPT_FAMILIES, SOURCE_COMMIT, SOURCE_SHA256, SOURCE_URL, TEMPLATES,
    TOKENIZER_SHA256, canonical_bytes, canonical_hash, sha256, tokenize,
)

SEED = "larql-gwcar1b-capital-fresh-subject-proposal-v1"


def select(countries: list[dict], excluded: set[str]) -> list[dict]:
    selected = [c for c in countries if c.get("independent") and c.get("unMember")
                and c["cca3"] not in excluded and len(c.get("capital") or []) == 1]
    if len({c["cca3"] for c in selected}) != len(selected) or len(selected) < 4:
        raise ValueError("duplicate country identity or insufficient fresh capital population")
    return sorted(selected, key=lambda c: hashlib.sha256(canonical_bytes([SEED, c["cca3"]])).hexdigest())


def build(source: Path, tokenizer_path: Path) -> dict:
    if sha256(source) != SOURCE_SHA256 or sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("pinned factual source or tokenizer changed")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    excluded = excluded_subjects()
    selected = select(json.loads(source.read_text()), excluded)
    rows = []
    for index, country in enumerate(selected):
        for family in PROMPT_FAMILIES:
            subject, target = country["name"]["common"], country["capital"][0]
            prompt = TEMPLATES["capital"][family].format(subject=subject)
            prompt_ids, target_ids = tokenize(tokenizer, prompt, target)
            subject_ids = tokenizer.encode(subject, add_special_tokens=False).ids
            if not subject_ids:
                raise ValueError("empty standalone subject tokenization")
            rows.append({"subject_id": country["cca3"], "subject": subject, "target": target,
                         "split": "validation" if index < (len(selected) + 1) // 2 else "test",
                         "prompt_family": family, "prompt": prompt,
                         "prompt_token_ids": prompt_ids, "target_continuation_token_ids": target_ids,
                         "subject_embedding_token_ids": subject_ids})
    document = {
        "schema": "larql.gwcar1b.population-proposal.v1",
        "status": "outcome_free_proposal_not_an_experiment_seal",
        "source": {"url": SOURCE_URL, "commit": SOURCE_COMMIT, "sha256": SOURCE_SHA256},
        "tokenizer_sha256": TOKENIZER_SHA256,
        "predicate": ["source independent flag true", "source unMember flag true",
                      "country ID absent from GW-0 and all GW-V2 splits", "exactly one source capital"],
        "scope": "capital only; no currency/language or global-relation generalization claim",
        "exclusion_subject_ids": sorted(excluded), "split_seed": SEED,
        "split_rule": "SHA256 canonical [seed,cca3] ascending; ceil(n/2) validation, remaining test; subject clustered",
        "counts": {"subjects": len(selected), "rows": len(rows),
                   "validation_subjects": (len(selected) + 1) // 2, "test_subjects": len(selected) // 2},
        "candidate_token_ids": sorted({r["target_continuation_token_ids"][0] for r in rows}),
        "rows": rows, "model_executions": 0, "model_execution_authorized": False,
        "limitations": [
            "source annotations are frozen historical labels, not independently re-audited modern country classifications",
            "no causal controls, capture, fitting, or replay executed",
            "proposal does not settle statistical power or full experimental protocol",
        ],
    }
    document["proposal_sha256"] = canonical_hash(document, "proposal_sha256")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument("source", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    document = build(args.source, args.tokenizer)
    if args.command == "verify":
        if json.loads(args.output.read_text()) != document:
            raise ValueError("population proposal differs from pinned source and selection")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"proposal_sha256": document["proposal_sha256"], "counts": document["counts"],
                      "model_execution_authorized": False}, indent=2, sort_keys=True))
