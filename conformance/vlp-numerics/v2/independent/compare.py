#!/usr/bin/env python3
"""Compare an independent evaluator's blind results with the golden vectors, and prove the comparison can fail.

The evaluator (evaluate.py) never sees expected values. This script is not part of it: it is the referee.

    python3 -I compare.py <results.json> <mapping.json> <inputs.json>   writes RECORD.json next to this file
"""
import copy
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
V1 = os.path.join(os.path.dirname(V2), "v1")
DOCUMENTED_DIFFERENCE = {"eq-lr-negzero-plus-poszero"}
HEX_WORD = 4


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def outcome(entry):
    """A case's outcome in one comparable form."""
    if "refuse" in entry:
        return ("refuse", entry["refuse"])
    return ("bf16", tuple(entry["bf16"]))


def compare(results, expected):
    """ids whose outcome differs (missing counts as different)."""
    return sorted(i for i, e in expected.items() if i not in results or outcome(results[i]) != outcome(e))


def mutate(entry, flavour):
    """A deliberately wrong outcome built from a right one."""
    m = copy.deepcopy(entry)
    if "refuse" in m:
        return {"bf16": ["0000"]} if flavour == "kind" else {"refuse": m["refuse"] + "_x"}
    words = m["bf16"]
    if flavour == "kind":
        return {"refuse": "overflow"}
    if flavour == "length":
        return {"bf16": words[:-1]}
    i = len(words) // 2
    words[i] = f"{int(words[i], 16) ^ 1:0{HEX_WORD}x}"  # the lowest mantissa bit of one element
    return m


def main():
    results_path, mapping_path, inputs_path = sys.argv[1:4]
    results, mapping = json.load(open(results_path)), json.load(open(mapping_path))
    vectors = json.load(open(os.path.join(V2, "vectors.json")))
    by_name = {c["id"]: c for c in vectors["cases"]}
    v1 = {c["id"]: c for c in json.load(open(os.path.join(V1, "vectors.json")))["cases"]}
    expected = {cid: by_name[name]["expect"] for cid, name in mapping.items()}
    assert set(results) == set(expected), "the evaluator did not answer exactly the cases it was given"

    wrong = compare(results, expected)

    # the v1 equivalence cases, against v1's own expected values
    eq, same, differ = 0, 0, []
    for cid, name in mapping.items():
        v1_id = by_name[name].get("equivalent_to_v1")
        if not v1_id:
            continue
        eq += 1
        if outcome(results[cid]) == outcome(v1[v1_id]["expect"]):
            same += 1
        else:
            differ.append(name)

    # can the comparison fail? every case, three kinds of corruption of the EXPECTED value, then of the RESULT
    undetected = []
    for cid in expected:
        for flavour in ("bit", "kind", "length"):
            bad = {**expected, cid: mutate(expected[cid], flavour)}
            if outcome(bad[cid]) != outcome(expected[cid]) and not compare(results, bad):
                undetected.append(("expected", cid, flavour))
            corrupted = {**results, cid: mutate(results[cid], flavour)}
            if outcome(corrupted[cid]) != outcome(results[cid]) and not compare(corrupted, expected):
                undetected.append(("result", cid, flavour))

    record = {
        "schema": "vlp-numerics-v2-independent-reproduction/1",
        "evaluator": {"path": "independent/evaluate.py", "sha256": sha(os.path.join(HERE, "evaluate.py")),
                      "python": platform.python_version()},
        "documents_sha256": {"v2/SPEC.md": sha(os.path.join(V2, "SPEC.md")), "v1/SPEC.md": sha(os.path.join(V1, "SPEC.md"))},
        "inputs_sha256": sha(inputs_path), "results_sha256": sha(results_path),
        "vectors_sha256": sha(os.path.join(V2, "vectors.json")),
        "cases": len(expected), "reproduced": len(expected) - len(wrong), "not_reproduced": wrong,
        "v1_equivalence": {"cases": eq, "identical_to_v1": same, "differing": differ,
                           "differing_is_exactly_the_documented_one": set(differ) == DOCUMENTED_DIFFERENCE},
        "comparison_self_test": {"corruptions_tried": 2 * 3 * len(expected), "undetected": undetected},
    }
    with open(os.path.join(HERE, "RECORD.json"), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(json.dumps({k: record[k] for k in ("cases", "reproduced", "not_reproduced", "v1_equivalence",
                                              "comparison_self_test")}, indent=1))
    return 0 if not wrong and not undetected and set(differ) == DOCUMENTED_DIFFERENCE else 1


if __name__ == "__main__":
    sys.exit(main())
