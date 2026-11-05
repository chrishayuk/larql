#!/usr/bin/env python3
"""Render an already sealed CAR-1B adjudication; no fitting or decision logic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gwcar1b_sources import GRID
from gwv2_population import canonical_hash, sha256


def report(path: Path) -> str:
    doc = json.loads(path.read_text())
    if (doc.get("schema") != "larql.gwcar1b.adjudication.v1"
        or doc.get("adjudication_sha256") != canonical_hash(doc, "adjudication_sha256")):
        raise ValueError("adjudication identity mismatch")
    summary = doc["summary"]
    valid = summary["status"] == "valid"
    for name, digest in doc["authorities"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("adjudicated artifact manifest changed")
    lines = ["# GW-CAR-1B — fresh-capital carrier-source result", "",
             f"**Status:** {'CLOSED — ' + summary['verdict'] if valid else 'INVALID — required control/denominator did not clear'}.", "",
             "Population: 67 fresh capital subjects, 201 prompts; 34 validation subjects and",
             "33 test subjects, with three templates per subject. Source fitting used only",
             "the 44-subject / 396-row historical training bank. No source was selected on",
             "validation results. The simultaneous family contains 104 readouts per split.", "",
             f"Protocol: `{doc['protocol_sha256']}`.",
             f"Adjudication: `{doc['adjudication_sha256']}`.",
             f"Adjudication file: `{sha256(path)}`.", ""]
    if valid:
        sources = summary["clearing_sources"]
        informative = summary["informative_delta_sources"]
        lines += [f"**{len(sources)}/10 registered construction sources clear both splits.**",
                  "Clearing source IDs: " + (", ".join(map(str, sources)) if sources else "none") + ".",
                  "Informative-delta IDs (also clearing template and shuffle comparisons): "
                  + (", ".join(map(str, informative)) if informative else "none") + ".", ""]
    else:
        lines += ["The mandatory validity gate failed. Candidate measurements below are",
                  "descriptive only and cannot establish a positive source result or a",
                  "negative construction frontier.", "",
                  "Split status: `" + json.dumps(summary["split_status"], sort_keys=True) + "`.", ""]
    if valid:
        analyses = [doc["results"][split]["analysis"] for split in ("validation", "test")]
        zero_passes = all(r["candidates"]["16"]["gate_pass"] for r in analyses)
        top32_passes = all(r["candidates"]["2"]["gate_pass"] for r in analyses)
        lines += ["## Interpretation and control results", ""]
        if zero_passes:
            ratios = [[a / b for a, b in zip(r["candidates"]["16"]["terminal_effect"],
                                            r["candidates"]["0"]["terminal_effect"], strict=True)]
                      for r in analyses]
            lines += ["**The zero-carrier diagnostic passes both splits.** It was excluded from",
                      "eligible source IDs before execution, so it does not change the registered",
                      "`no_registered_carrier_source` verdict. It is nevertheless a positive",
                      "absolute-gate result and prevents interpreting that verdict as evidence",
                      "that every cheap carrier substitute fails or that natural carrier",
                      "construction is intrinsically necessary.", "",
                      "The zero substitution strongly amplifies the proximal contrast. Its",
                      "terminal effect remains positive but is substantially smaller than exact",
                      "context 128: descriptive zero/exact terminal-effect point ratios are",
                      f"`{ratios[0][0]:.6f} / {ratios[0][1]:.6f}` on validation and",
                      f"`{ratios[1][0]:.6f} / {ratios[1][1]:.6f}` on test (raw / z).",
                      "These post-adjudication descriptive ratios introduce no new gate. The",
                      "registered terminal requirement was positivity, not retention or answer",
                      "fidelity. Consequently, passing this gate does not certify a generally",
                      "valid execution state. Exact target H1 and natural target KV also remain",
                      "available to the zero-control intervention.", ""]
        if not top32_passes:
            lines += ["**The target-derived top-32 oracle does not clear fresh confirmation.**",
                      "CAR-1A's 192-byte compactness result therefore does not automatically",
                      "transfer to this new capital population/readout/control surface. This",
                      "does not retrospectively invalidate CAR-1A on its original surface.", ""]
        if all(not r["candidates"]["5"]["gate_pass"] for r in analyses):
            lines += ["The template-plus-fixed-support *oracle* delta also fails both splits:",
                      "even supplying the exact target delta on those 32 train-selected",
                      "coordinates does not meet the gate. Failure of the corresponding learned",
                      "delta sources thus cannot be attributed solely to regression quality or",
                      "missing early-state information. The tested representation itself is",
                      "insufficient under this criterion.", ""]
        lines += ["Any next claim about executable state needs a separately prospective test",
                  "that distinguishes retention of the registered relational contrast from",
                  "natural continuation or answer fidelity. These outcomes do not authorize",
                  "changing this experiment's gate, promoting its diagnostic control to a",
                  "registered source, or extending its grid on the same confirmation rows.", ""]
    lines += ["## Execution integrity", "",
              "All 396 freshly captured train carriers reproduced the historical carriers",
              "bit for bit; all 597 natural head mixtures reconstructed their observed",
              "attention writes. Fresh context 255 reproduced ordinary natural continuation",
              "on all 201 confirmation rows. The 1,206 independent reference interventions",
              "and 8,040 candidate interventions completed with zero identity mismatches;",
              "candidate 0/1 proximal and terminal tensors matched contexts 128/0. Source",
              "models and every decoded candidate were independently rebuilt and compared",
              "before the frozen analysis. No latency experiment was performed.", "",
              "The [authoritative adjudication](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-adjudication.json)",
              "retains the full precision values and decision flags. The",
              "[post-run audit](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-postrun-audit.json)",
              "independently checks the stored gate flags, candidate numbering, zero-valued",
              "payload, executable identity, and intervention counts; it introduces no new",
              "statistical gate or model run.", ""]
    lines += ["## Full registered surface", "",
              "Pairs are raw / z. Lower bounds are simultaneous, not pointwise. Each split",
              "passes only if its proximal points are ≥0.80, proximal lower bounds ≥0.50,",
              "and terminal-effect lower bounds >0. Values above one indicate contrast",
              "amplification, not natural-state reconstruction fidelity.", "",
              "| ID / source | Validation retention | Validation proximal lower | Validation terminal lower | Test retention | Test proximal lower | Test terminal lower | Absolute split gates |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    def pair(value):
        return " / ".join(f"{x:.6f}" for x in value)
    for c in GRID:
        cells, passes = [], []
        for split in ("validation", "test"):
            result = doc["results"][split]["analysis"]["candidates"].get(str(c.candidate_id))
            if result:
                cells += [pair(result[k]) for k in ("proximal_retention", "proximal_lower95", "terminal_lower95")]
                passes.append("pass" if result["gate_pass"] else "fail")
            else:
                cells += ["not estimable"] * 3
                passes.append("invalid")
        lines.append(f"| {c.candidate_id} / {c.name} | " + " | ".join(cells) + " | " + " / ".join(passes) + " |")
    lines += ["", "IDs 0–5 are oracle controls, 16 is zero, and 17–19 are shuffled-target",
              "controls. Only IDs 6–15 are eligible construction sources.", "",
              "## Learned-correction comparisons", "",
              "Each cell lists raw/z proximal and terminal lower bounds. The informative",
              "label requires both the template and shuffle comparisons to pass on both",
              "splits, as well as the source's absolute gate.", "",
              "| Difference | Validation proximal; terminal | Test proximal; terminal |",
              "|---|---:|---:|"]
    for key in ("11-8", "12-8", "13-8", "11-17", "12-18", "13-19"):
        cells = []
        for split in ("validation", "test"):
            result = doc["results"][split]["analysis"]["comparisons"].get(key)
            cells.append("; ".join(pair(v) for v in result["lower95"]) if result else "not estimable")
        lines.append("| " + key + " | " + " | ".join(cells) + " |")
    lines += ["", "## Source storage and remaining work", "",
              "These are per-candidate source-model bytes, not replay-tensor size, process",
              "memory, total offline cost, or latency. The dense candidate replay tensor is",
              "a materialization artifact. Feature acquisition and ordinary prefix costs",
              "are not included in these storage columns.", "",
              "| ID / source | Numeric model bytes | Serialized NPZ bytes | Metadata bytes |",
              "|---|---:|---:|---:|"]
    for model in doc["models"]:
        i = model["candidate_id"]
        lines.append(f"| {i} / {GRID[i].name} | {model['numeric_bytes']} | {model['model_file_bytes']} | {model['metadata_bytes']} |")
    lines += ["", "All cases retain exact target H1, seven donor non-H1 heads, target-natural",
              "KV, and ordinary continuation. A passing source supports carrier substitution",
              "at that interface; it does not establish end-to-end prefix avoidance, answer",
              "accuracy, a universal semantic head, or economical graph-walk execution.", "",
              "The [executable protocol](gw-car-1b-execution.md) defines the frozen scope",
              "and analysis. The [pre-execution review](gw-car-1b-preseal-review.md) records",
              "the synthetic checks and seal. Captures, serialized source models, receipts,",
              "and replay tensors remain under the ignored `output/gwcar1b-capital-v1/` tree;",
              "their hashes are bound by the adjudication's artifact manifests.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adjudication", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    content = report(args.adjudication)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    print(args.output)
