#!/usr/bin/env python3
"""PATCH-TRAIN-1 L1 step 2: validate the model-evaluation path. Applies emulated patches to the declared backend and checks,
by bits and by digest, that (a) unpatched evaluation reproduces the committed baseline, (b) the patched weight is exactly the
emulator's BF16 result widened, (c) nothing else changes and the base is restored and never written, (d) a zero-update patch
is end-to-end identical, (e) a compiled container produced by the engine loads through the same path.
    python3 -I validate_eval_path.py        writes eval_path.json
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import emulate as E  # noqa: E402
import evaluation as V  # noqa: E402

FOOTPRINT_TENSOR, FOOTPRINT_COLUMNS, SEED = "9.mlp.down_proj.weight", 256, 20261113
SCALE_RELATIVE = 0.03          # patch element RMS as a fraction of the footprint's weight RMS (large enough to change outputs)
REAL2_CONTAINER = os.path.join(V.REPO, "bench", "structured-real-2", "work", "out1.vindex3")
REAL2_TENSOR, REAL2_ROW = "5.mlp.down_proj.weight", 7
SHARED = ("logp_target", "top1_full_vocab", "top1_in_answer_set", "margin_in_set", "rank_in_set")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root):
    out = {}
    for d, _, names in os.walk(root):
        for n in sorted(names):
            out[os.path.relpath(os.path.join(d, n), root)] = sha_file(os.path.join(d, n))
    return out


def checkpoint_file(tasks):
    from huggingface_hub import hf_hub_download
    m = tasks["model"]
    return hf_hub_download(m["repo"], "model.safetensors", revision=m["revision"], local_files_only=True)


def random_patch(base_bits, rng):
    rows, cols = base_bits.shape
    columns = np.sort(rng.choice(cols, FOOTPRINT_COLUMNS, replace=False))
    w = E.bf16_to_f64(base_bits[:, columns])
    rms = float(np.sqrt((w ** 2).mean()))
    a = rng.standard_normal((rows, 1))
    b = rng.standard_normal((1, len(columns)))
    t = SCALE_RELATIVE * rms / np.sqrt((a ** 2).mean() * (b ** 2).mean())
    return columns, a * t, b


def main():
    tasks = V.load_tasks()
    model, tok = V.load_model(tasks)
    ckpt = checkpoint_file(tasks)
    before = {"checkpoint_sha256": sha_file(ckpt), "container": tree_digest(V.CONTAINER), "model": V.model_digest(model)}
    rec = {"protocol": "PATCH-TRAIN-1 L1 evaluation-path validation", "backend": V.backend_record(tasks), "checks": {}}

    # (a) unpatched evaluation reproduces the committed baseline, on the fields the baseline recorded
    base_eval = V.evaluate_suite(model, tok, tasks)
    with open(os.path.join(V.COMPETENCE, "baseline.json")) as f:
        baseline = json.load(f)["rows"]
    mismatches = []
    for key in ("A", "B"):
        for old, new in zip(baseline["tasks"][key], base_eval["tasks"][key]):
            assert (old["split"], old["entity"], old["template"]) == (new["split"], new["entity"], new["template"])
            mismatches += [(key, new["entity"], f) for f in SHARED if old[f] != new[f]]
    mismatches += [("real", o["country"], f) for o, n in zip(baseline["real_facts"], base_eval["real_facts"])
                   for f in ("logp_target", "top1") if o[f] != n[f]]
    mismatches += [("neutral",)] if baseline["neutral_nll"] != base_eval["neutral_nll"] else []
    rec["checks"]["unpatched_reproduces_committed_baseline"] = {
        "compared_fields_per_prompt": list(SHARED), "mismatches": len(mismatches), "pass": not mismatches}

    # (b)+(c) an emulated rank-1 patch on the declared footprint
    key = V.hf_key(FOOTPRINT_TENSOR)
    base_bits = V.weight_bits(model, key)
    from_container = V.container_tensor_bits(V.CONTAINER, key)
    rec["checks"]["checkpoint_tensor_equals_container_tensor"] = {"tensor": FOOTPRINT_TENSOR, "pass": bool((base_bits == from_container).all())}
    columns, a, b = random_patch(base_bits, np.random.default_rng(SEED))
    new_bits, written = E.apply_columns(base_bits, columns, a, b, E.PROFILE_RNE1)
    others = V.model_digest(model, skip={key})
    with V.patched_weight(model, key, new_bits):
        patched_ok = bool((V.weight_bits(model, key) == new_bits).all())
        others_ok = V.model_digest(model, skip={key}) == others
        sample = V.evaluate_suite(model, tok, tasks, which=("A",))
    changed = int((new_bits != base_bits).sum())
    outside = np.ones(base_bits.shape[1], bool)
    outside[columns] = False
    differs = any(o["logp_target"] != n["logp_target"] for o, n in zip(base_eval["tasks"]["A"], sample["tasks"]["A"]))
    rec["checks"]["emulated_patch_applied_exactly"] = {
        "tensor": FOOTPRINT_TENSOR, "footprint_columns": len(columns), "rank": 1, "profile": E.PROFILE_RNE1,
        "elements_changed": changed, "elements_in_footprint": int(base_bits.shape[0] * len(columns)),
        "patched_weight_equals_emulator_bits": patched_ok, "all_other_parameters_unchanged_while_patched": others_ok,
        "no_change_outside_footprint_columns": bool((new_bits[:, outside] == base_bits[:, outside]).all()),
        "outputs_differ_from_unpatched": differs,
        "pass": patched_ok and others_ok and bool((new_bits[:, outside] == base_bits[:, outside]).all()) and differs and changed > 0}
    rec["checks"]["base_restored_after_patch"] = {"model_digest_equal": V.model_digest(model) == before["model"]}
    rec["checks"]["base_restored_after_patch"]["pass"] = rec["checks"]["base_restored_after_patch"]["model_digest_equal"]

    # (d) a zero-update patch is an identity end to end
    zero_bits, zero_written = E.apply_columns(base_bits, columns, np.zeros_like(a), b, E.PROFILE_RNE1)
    with V.patched_weight(model, key, zero_bits):
        zero_eval = V.evaluate_suite(model, tok, tasks)
    rec["checks"]["zero_update_is_identity_end_to_end"] = {
        "weight_bits_equal_base": bool((zero_bits == base_bits).all()), "nothing_written": not bool(zero_written.any()),
        "all_metrics_bitwise_equal": zero_eval == base_eval}
    rec["checks"]["zero_update_is_identity_end_to_end"]["pass"] = all(rec["checks"]["zero_update_is_identity_end_to_end"].values())

    # (e) a compiled container produced by the engine loads through the same path (REAL-2: 8 known elements changed)
    if os.path.isdir(REAL2_CONTAINER):
        key5 = V.hf_key(REAL2_TENSOR)
        bits5 = V.container_tensor_bits(REAL2_CONTAINER, key5)
        base5 = V.weight_bits(model, key5)
        diff = np.argwhere(bits5 != base5)
        old, new = E.bf16_to_f64(base5[tuple(diff.T)]), E.bf16_to_f64(bits5[tuple(diff.T)])
        others5 = V.model_digest(model, skip={key5})
        with V.patched_weight(model, key5, bits5):
            loaded = bool((V.weight_bits(model, key5) == bits5).all()) and V.model_digest(model, skip={key5}) == others5
        rec["checks"]["engine_compiled_container_loads_through_the_same_path"] = {
            "container": "bench/structured-real-2/work/out1.vindex3 (REAL-2, compiled by the engine from a structured op)",
            "elements_changed_vs_registry": int(len(diff)), "all_in_row": bool((diff[:, 0] == REAL2_ROW).all()),
            "values_equal_minus_2^-30_times_old": bool(np.array_equal(new, -old * 2.0 ** -30)), "loaded_exactly": loaded,
            "pass": len(diff) == 8 and bool((diff[:, 0] == REAL2_ROW).all()) and bool(np.array_equal(new, -old * 2.0 ** -30)) and loaded}
    else:
        rec["checks"]["engine_compiled_container_loads_through_the_same_path"] = {"not_run": "REAL-2 work directory absent", "pass": None}

    # the base was never written
    after = {"checkpoint_sha256": sha_file(ckpt), "container": tree_digest(V.CONTAINER), "model": V.model_digest(model)}
    rec["checks"]["immutable_base_never_written"] = {"checkpoint_unchanged": before["checkpoint_sha256"] == after["checkpoint_sha256"],
                                                     "registry_container_unchanged": before["container"] == after["container"],
                                                     "in_memory_model_restored": before["model"] == after["model"]}
    rec["checks"]["immutable_base_never_written"]["pass"] = all(rec["checks"]["immutable_base_never_written"].values())
    rec["all_pass"] = all(c["pass"] is not False for c in rec["checks"].values()) and all(
        c["pass"] for k, c in rec["checks"].items() if c["pass"] is not None)
    with open(os.path.join(HERE, "eval_path.json"), "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v.get("pass") for k, v in rec["checks"].items()}, indent=1), "\nall_pass:", rec["all_pass"])
    for k in ("emulated_patch_applied_exactly", "engine_compiled_container_loads_through_the_same_path"):
        print(k, json.dumps(rec["checks"][k]))


if __name__ == "__main__":
    main()
