"""The transformers oracle's weight-loading facts, and the gate on them.

An equivalence verdict is only as good as the oracle it is measured
against: a transformers model that silently initialised a missing weight
(or dropped a language-model tensor as `unexpected`) computes a different
function from the checkpoint. So every equivalence record carries what the
load did — the class, the expected parameter keys, the checkpoint's tensor
count, the full missing and unexpected lists grouped by prefix, and
`tie_word_embeddings` — and the run is gated on it before any forward pass.

Gate (fixed before any run):
  * no missing key, except `lm_head.weight` when `tie_word_embeddings` is
    true (the head is the embedding, by declaration);
  * every unexpected key lies under an ALLOWLISTED prefix. The allowlist
    is declared by LARQL itself, never written here: the tensor prefixes
    the plan's system graph binds to components outside the
    text-generation closure (`capability_boundary`). Any other unexpected
    key fails, whether or not it looks like language-model weight.

Cross-check, recorded beside the gate: LARQL's declared boundary against
what transformers actually loaded — a checkpoint tensor LARQL calls out of
closure that the oracle loads, an in-closure tensor the oracle leaves
unloaded (also a gate failure), and a checkpoint tensor LARQL binds to no
component at all.

Pure (no torch), so the gate is unit tested in `test_oracle_loading.py`.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path

#: The tied output head's key, the one missing key a tied model may have.
TIED_HEAD_KEY = "lm_head.weight"
#: Dotted segments kept when grouping keys for the record.
GROUP_DEPTH = 2
#: A safetensors header's length field.
HEADER_LENGTH_BYTES = 8


def group_by_prefix(keys, depth: int = GROUP_DEPTH) -> dict[str, int]:
    """Count keys per leading `depth` dotted segments."""
    return dict(sorted(Counter(".".join(k.split(".")[:depth]) for k in keys).items()))


def under(key: str, prefixes) -> bool:
    """Whether `key` lies under one of `prefixes`, on whole dotted segments."""
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def capability_boundary(plan: dict, primary_role: str = "primary_text") -> dict:
    """LARQL's declared text-generation boundary, from a plan's system graph:
    the tensor prefixes bound to components of `primary_role` (in closure)
    and to every other component (out of closure, the allowlist)."""
    roles = {c["id"]: c.get("role") for c in plan["graph"]["components"]}
    inside, outside = {}, {}
    for obj in plan["graph"]["objects"]:
        bucket = inside if roles.get(obj["component"]) == primary_role else outside
        for binding in obj.get("source_bindings", []):
            bucket[binding["tensor_prefix"]] = obj["id"]
    return {
        "source": "larql vindex3 plan --capability text-generation: graph objects' source_bindings, "
                  f"split by component role ({primary_role} = in closure)",
        "in_closure_prefixes": dict(sorted(inside.items())),
        "allowlist": dict(sorted(outside.items())),
    }


def safetensors_header_keys(path: Path) -> list[str]:
    """Tensor names in one safetensors file, from its header alone."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(HEADER_LENGTH_BYTES))
        header = json.loads(f.read(length))
    return [k for k in header if k != "__metadata__"]


def index_keys(index: dict) -> list[str]:
    """Tensor names a `model.safetensors.index.json` maps."""
    return list(index["weight_map"])


def loading_facts(
    *,
    model_class: str,
    expected_keys,
    checkpoint_keys,
    missing,
    unexpected,
    mismatched,
    tie_word_embeddings: bool,
    key_mapping: dict | None,
    boundary: dict,
) -> dict:
    """The record of one oracle load, with its gate verdict and the
    boundary cross-check."""
    missing = sorted(missing)
    unexpected = sorted(unexpected)
    checkpoint_keys = list(checkpoint_keys)
    allow, closure = boundary["allowlist"], boundary["in_closure_prefixes"]
    allowed_missing = {TIED_HEAD_KEY} if tie_word_embeddings else set()
    bad_missing = [k for k in missing if k not in allowed_missing]
    not_allowlisted = [k for k in unexpected if not under(k, allow)]
    unexpected_set = set(unexpected)
    problems = []
    if bad_missing:
        problems.append(f"{len(bad_missing)} missing key(s) not explained by a tied head")
    if not_allowlisted:
        problems.append(f"{len(not_allowlisted)} unexpected key(s) outside the declared allowlist")
    if mismatched:
        problems.append(f"{len(mismatched)} shape-mismatched key(s)")
    return {
        "transformers_class": model_class,
        "key_mapping": key_mapping,
        "expected_parameter_keys": len(list(expected_keys)),
        "checkpoint_tensors": len(list(checkpoint_keys)),
        "tie_word_embeddings": tie_word_embeddings,
        "missing": missing,
        "missing_by_prefix": group_by_prefix(missing),
        "unexpected": unexpected,
        "unexpected_by_prefix": group_by_prefix(unexpected),
        "unexpected_not_allowlisted": not_allowlisted,
        "unexpected_inside_language_model": [k for k in unexpected if under(k, closure)],
        "mismatched": sorted(str(m) for m in mismatched),
        "capability_boundary": boundary,
        "boundary_cross_check": {
            "out_of_closure_but_loaded": sorted(
                k for k in checkpoint_keys if under(k, allow) and k not in unexpected_set),
            "in_closure_but_unloaded": sorted(k for k in unexpected if under(k, closure)),
            "bound_to_no_component": sorted(
                k for k in checkpoint_keys if not under(k, allow) and not under(k, closure)),
        },
        "gate": "pass" if not problems else "fail",
        "gate_problems": problems,
    }
