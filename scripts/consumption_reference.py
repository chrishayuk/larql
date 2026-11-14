#!/usr/bin/env python3
"""An independent reference for the runtime-consumption record (`larql-consumption-record/1`).

The engine's harness (`opplan/exec/tests/consumption_real.rs`) records, for a base container and an edit of one FFN
`down_proj`, the vector the production backend fed that tensor at each position and the residual carrier on both sides
of the FFN write. This script recomputes the FFN write from each CONTAINER'S OWN tensor bytes and that recorded input,
and asks of each run: which container's weights is this write the product of?

    write(W, h) = post_ffn_norm(W · h)          W: the layer's stored down_proj, BF16 widened exactly
    post_ffn_norm(y) = y / sqrt(mean(y²) + EPS) × (1 + g)      g: the layer's stored post_feedforward_layernorm

It shares no code with the engine: its own segment reader, binary64 arithmetic, no LARQL import.

The verdict is a 2 × 2 table per position, run × weights, of the relative error between the recomputed write and the
recorded one. Consumption is SHOWN when, at every position of every prompt:

  * the base run matches the base weights (this also calibrates the reference: if it fails, nothing is judged),
  * the edited run matches the edited weights,
  * the edited run does NOT match the base weights, and the base run does NOT match the edited weights.

The thresholds below were fixed before any record was read. Between them a position is inconclusive, and one
inconclusive or contradicting position makes the whole answer other than `shown`.

What this does not establish: that the edit is the intended one (the ledger contract), that any other tensor is
consumed correctly, or anything about behaviour. The record is the engine's own report of its intermediates.

    python3 -I scripts/consumption_reference.py <record.json> [--out <verdict.json>]
"""

import argparse
import hashlib
import json
import struct
import sys

import numpy as np

RECORD_SCHEMA = "larql-consumption-record/1"
VERDICT_SCHEMA = "larql-consumption-verdict/1"
REPRESENTATION = "target.decoder_stack@BF16"
DOWN_PROJ = "{layer}.mlp.down_proj.weight"
POST_FFN_NORM = "{layer}.post_feedforward_layernorm.weight"
#: Gemma's RMS norm: epsilon inside the root, and a learned gain stored as an offset from one.
NORM_EPSILON = 1e-6
#: A recomputed write within this relative error of the recorded one is that write. Binary32 kernels summing 10,240
#: products against a binary64 reference differ by about 1e-6; this leaves two orders of headroom.
MATCH_AT_MOST = 1e-4
#: A recomputed write at least this far from the recorded one is not that write. Fixed before any record was read.
MISMATCH_AT_LEAST = 1e-2
HEADER_LENGTH_BYTES = 8
BF16_BYTES = 2
F32_BYTES = 4
#: The loader setting under which a production CPU run keeps every BF16 operand exact.
EXACT_POLICY_VAR, EXACT_POLICY_VALUE = "LARQL_CPU_MAX_FORMAT", "bf16"
MATCH, MISMATCH, INCONCLUSIVE = "match", "mismatch", "inconclusive"
SHOWN, NOT_SHOWN, NOT_CALIBRATED = "shown", "not_shown", "reference_not_calibrated"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tensor(root, name):
    """One BF16 tensor of the canonical decoder representation, as binary64, with its shape and the SHA-256 of its
    stored bytes. A segment is an 8-byte little-endian header length, a JSON header, then the payload."""
    with open(f"{root}/index.json") as f:
        entry = json.load(f)["representations"][REPRESENTATION]
    with open(f"{root}/{entry['segment']}", "rb") as f:
        (header_length,) = struct.unpack("<Q", f.read(HEADER_LENGTH_BYTES))
        header = json.loads(f.read(header_length))
        tensor = next(t for t in header["tensors"] if t["name"] == name)
        if tensor["dtype"] != "BF16":
            raise ValueError(f"{name} is {tensor['dtype']}, not BF16")
        f.seek(HEADER_LENGTH_BYTES + header_length + tensor["offset"])
        raw = f.read(tensor["len"])
    if len(raw) != tensor["len"] or len(raw) != BF16_BYTES * int(np.prod(tensor["shape"])):
        raise ValueError(f"{name}: short or mis-sized tensor")
    # A BF16 pattern is the top half of a binary32: widen exactly, then to binary64 exactly.
    values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return values.reshape(tensor["shape"]), hashlib.sha256(raw).hexdigest()


def f32(hex_text):
    raw = bytes.fromhex(hex_text)
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def write(weights, gain, down_input):
    y = weights @ down_input
    return y / np.sqrt(np.mean(y * y) + NORM_EPSILON) * (1.0 + gain)


def relative_error(recomputed, recorded):
    return float(np.linalg.norm(recomputed - recorded) / np.linalg.norm(recorded))


def classify(error):
    if error <= MATCH_AT_MOST:
        return MATCH
    if error >= MISMATCH_AT_LEAST:
        return MISMATCH
    return INCONCLUSIVE


def judge(record, require_exact=True):
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"not a {RECORD_SCHEMA} record")
    # The reference recomputes with the stored BF16 weights. A run whose loader was free to manufacture a lossy
    # residency did not necessarily compute with them, so it is not judged as if it had.
    policy = (record.get("environment") or {}).get(EXACT_POLICY_VAR)
    if require_exact and policy != EXACT_POLICY_VALUE:
        raise ValueError(f"the record does not show {EXACT_POLICY_VAR}={EXACT_POLICY_VALUE} (it shows {policy!r}): "
                         "the run is not known to have executed the stored BF16 weights")
    layer = record["edited_layer"]
    sides = {}
    for side in ("base", "edited"):
        root = record[side]["path"]
        if sha256_file(f"{root}/index.json") != record[side]["index_sha256"]:
            raise ValueError(f"the {side} container's index.json is not the one the record was made from")
        weights, weights_sha = read_tensor(root, DOWN_PROJ.format(layer=layer))
        gain, gain_sha = read_tensor(root, POST_FFN_NORM.format(layer=layer))
        sides[side] = {"weights": weights, "gain": gain, "down_proj_sha256": weights_sha, "norm_sha256": gain_sha}
    if sides["base"]["down_proj_sha256"] == sides["edited"]["down_proj_sha256"]:
        raise ValueError("the two containers hold the same down_proj: there is no edit to consume")
    #: What each cell must be for consumption to be shown.
    wanted = {("base", "base"): MATCH, ("edited", "edited"): MATCH,
              ("base", "edited"): MISMATCH, ("edited", "base"): MISMATCH}
    positions, calibrated, shown = [], True, True
    for prompt in record["prompts"]:
        for run in ("base", "edited"):
            for p in prompt[run]["positions"]:
                down_input = f32(p["down_input_f32le_hex"])
                recorded = f32(p["carrier_after_ffn_f32le_hex"]) - f32(p["carrier_before_ffn_f32le_hex"])
                cells = {}
                for weights_of in ("base", "edited"):
                    s = sides[weights_of]
                    error = relative_error(write(s["weights"], s["gain"], down_input), recorded)
                    cells[weights_of] = {"relative_error": error, "class": classify(error)}
                    if cells[weights_of]["class"] != wanted[(run, weights_of)]:
                        shown = False
                        if (run, weights_of) == ("base", "base"):
                            calibrated = False
                positions.append({"prompt": prompt["name"], "run": run, "position": p["position"],
                                  "against_weights": cells})
    errors = lambda run, w: [x["against_weights"][w]["relative_error"] for x in positions if x["run"] == run]  # noqa: E731
    summary = {f"{run}_run_against_{w}_weights": {"min": min(errors(run, w)), "max": max(errors(run, w))}
               for run in ("base", "edited") for w in ("base", "edited")}
    return {
        "schema": VERDICT_SCHEMA,
        "status": NOT_CALIBRATED if not calibrated else SHOWN if shown else NOT_SHOWN,
        "claim": "at every recorded position, each run's FFN write at the edited layer is the product of its own "
                 "container's stored down_proj and the recorded input, and is not the product of the other "
                 "container's",
        "thresholds": {"match_at_most": MATCH_AT_MOST, "mismatch_at_least": MISMATCH_AT_LEAST,
                       "fixed": "before any record was read"},
        "edited_layer": layer,
        "tensors": {side: {k: s[k] for k in ("down_proj_sha256", "norm_sha256")} for side, s in sides.items()},
        "positions_judged": len(positions),
        "relative_error_ranges": summary,
        "engine_checks_reported": record.get("checks"),
        "execution_policy_reported": record.get("environment"),
        "prompts_file_sha256": record.get("prompts_file_sha256"),
        "final_logits_reported": [{"prompt": p["name"],
                                   "base_vs_edited": p.get("final_logits_base_vs_edited"),
                                   "base_vs_control": p.get("final_logits_base_vs_control")}
                                  for p in record["prompts"]],
        "not_established": ["that the edit is the intended one (the ledger contract's claim)",
                            "that any tensor other than the edited down_proj is consumed correctly",
                            "anything about behaviour",
                            "the record is the engine's own report of its intermediates, not an independent capture"],
        "positions": positions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--out")
    ap.add_argument("--any-policy", action="store_true",
                    help="judge a record that does not show the exact-BF16 loader setting (exploratory records)")
    args = ap.parse_args()
    with open(args.record) as f:
        record = json.load(f)
    verdict = judge(record, require_exact=not args.any_policy)
    verdict["record_sha256"] = sha256_file(args.record)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f, indent=1)
    brief = {k: verdict[k] for k in ("status", "positions_judged", "relative_error_ranges", "tensors")}
    json.dump(brief, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0 if verdict["status"] == SHOWN else 1


if __name__ == "__main__":
    sys.exit(main())
