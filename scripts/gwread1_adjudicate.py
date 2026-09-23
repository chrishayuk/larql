#!/usr/bin/env python3
"""Adjudicate frozen GW-READ-1 component fidelity and composition."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gwconv1_adjudicate import mean_pairwise_js, softmax
from gwsup1_preregister import canonical_hash, sha


ROWS = 426
HELDOUT = 171
CANDIDATES = 126
BOOTSTRAPS = 10_000


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def descriptor(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in manifest["artifacts"] if item["path"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one artifact {name}")
    return matches[0]


def checked(root: Path, manifest: dict[str, Any], name: str, shape: tuple[int, ...] | None = None) -> np.memmap:
    item = descriptor(manifest, name)
    path = root / name
    actual_shape = shape or tuple(item["shape"])
    if sha(path) != item["sha256"] or path.stat().st_size != math.prod(actual_shape) * 4:
        raise ValueError(f"artifact changed: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=actual_shape)


def zdist(logits: np.ndarray) -> np.ndarray:
    std = np.std(logits, axis=-1, keepdims=True)
    if np.any(std == 0) or np.any(~np.isfinite(std)):
        raise ValueError("invalid z-scored logits")
    return softmax((logits - np.mean(logits, axis=-1, keepdims=True)) / std)


def per_edge_effect(before: np.ndarray, after: np.ndarray, facts: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (mean_pairwise_js(before[fact]) - mean_pairwise_js(after[fact]))
            - (mean_pairwise_js(before[control]) - mean_pairwise_js(after[control]))
            for fact, control in zip(facts, controls, strict=True)
        ]
    )


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def summarize(
    exact: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    relations: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    factual = relations != "hypernym"
    result: dict[str, Any] = {}
    strata = [np.flatnonzero((relations == relation) & factual) for relation in ("capital", "currency", "language")]
    for surface in ("raw", "zscored"):
        exact_mean = float(np.mean(exact[surface][factual]))
        candidate_mean = float(np.mean(candidate[surface][factual]))
        retention = candidate_mean / exact_mean if exact_mean > 0 else float("nan")
        boot_effect = np.empty(BOOTSTRAPS)
        boot_retention = np.empty(BOOTSTRAPS)
        for draw in range(BOOTSTRAPS):
            sample = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
            exact_draw = float(np.mean(exact[surface][sample]))
            candidate_draw = float(np.mean(candidate[surface][sample]))
            boot_effect[draw] = candidate_draw
            boot_retention[draw] = candidate_draw / exact_draw if exact_draw > 0 else np.nan
        finite = boot_retention[np.isfinite(boot_retention)]
        if len(finite) != BOOTSTRAPS:
            raise ValueError("bootstrap produced an unstable exact denominator")
        result[surface] = {
            "exact_E": exact_mean,
            "candidate_E": candidate_mean,
            "candidate_E_ci95": interval(boot_effect),
            "retention": retention,
            "retention_ci95": interval(finite),
            "E_by_relation": {
                relation: float(np.mean(candidate[surface][relations == relation]))
                for relation in ("capital", "currency", "language", "hypernym")
            },
        }
    return result


def passes_component(proximal: dict[str, Any], terminal: dict[str, Any]) -> bool:
    return all(
        proximal[surface]["retention"] >= 0.8
        and proximal[surface]["retention_ci95"][0] >= 0.5
        and terminal[surface]["candidate_E_ci95"][0] > 0
        for surface in ("raw", "zscored")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--head-capture", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("GW-READ-1 adjudication already exists")
    protocol = json.loads(args.protocol.read_text())
    selection = json.loads(args.selection.read_text())
    heldout = json.loads(args.heldout.read_text())
    head = json.loads(args.head_capture.read_text())
    groups = json.loads(args.groups.read_text())
    if protocol["schema"] != "larql.gwread1.protocol.v2" or selection["schema"] != "larql.gwread1.selection.v1" or heldout["schema"] != "larql.gwread1.heldout-replay.v1":
        raise ValueError("inadmissible GW-READ-1 adjudication authority")
    if heldout["selection_sha256"] != selection["selection_sha256"] or heldout["selection_file_sha256"] != sha(args.selection):
        raise ValueError("held-out replay binds another selection")
    if heldout["parity"] != {"gwkey_exact_proximal_bit_mismatches": 0, "gwkey_exact_terminal_bit_mismatches": 0}:
        raise ValueError("held-out exact parity failed")

    root = args.heldout.parent
    head_root = args.head_capture.parent
    heldout_rows = jsonl(root / "heldout-rows.jsonl")
    natural_rows = jsonl(head_root / "natural-rows.jsonl")
    if len(heldout_rows) != HELDOUT or len(natural_rows) != ROWS:
        raise ValueError("held-out row count changed")
    original_rows = np.asarray([row["original_row"] for row in heldout_rows], dtype=int)
    before_all = checked(head_root, head, "candidate-before-logits.f32", (ROWS, CANDIDATES))
    before_logits = np.asarray(before_all[original_rows], dtype=float)
    before = {"raw": softmax(before_logits), "zscored": zdist(before_logits)}
    lookup = {row["edge_id"]: index for index, row in enumerate(heldout_rows)}
    controls = {item["edge_id"]: item["control_edge_id"] for item in groups["different_destination_controls"]}
    split_groups: dict[str, list[dict[str, Any]]] = {}
    for split in ("validation", "test"):
        selected_groups = [
            group
            for group in groups["prompt_family_groups"]
            if group["edge_ids"][0] in lookup
            and heldout_rows[lookup[group["edge_ids"][0]]]["split"] == split
        ]
        expected = 29 if split == "validation" else 28
        if len(selected_groups) != expected:
            raise ValueError(f"{split} semantic-edge group count changed")
        split_groups[split] = selected_groups

    proximal_arrays = {
        "Q": checked(root, heldout, "heldout-q-proximal.f32"),
        "K": checked(root, heldout, "heldout-k-proximal.f32"),
        "V": checked(root, heldout, "heldout-v-proximal.f32"),
        "E": checked(root, heldout, "heldout-e-proximal.f32")[:, [0, 7], :],
    }
    terminal_all = checked(root, heldout, "heldout-terminal.f32")
    terminal_arrays = {
        "Q": terminal_all[:, [0, 1], :],
        "K": terminal_all[:, [2, 3], :],
        "V": terminal_all[:, [4, 5], :],
        "E": terminal_all[:, [6, 7], :],
    }
    results: dict[str, Any] = {split: {} for split in ("validation", "test")}
    seed = 18012034
    for split_number, split in enumerate(("validation", "test")):
        selected_groups = split_groups[split]
        facts = np.asarray([[lookup[edge] for edge in group["edge_ids"]] for group in selected_groups])
        control_rows = np.asarray([[lookup[controls[edge]] for edge in group["edge_ids"]] for group in selected_groups])
        relations = np.asarray([group["relation"] for group in selected_groups])
        for arm_number, arm in enumerate(("Q", "K", "V", "E")):
            arm_result: dict[str, Any] = {}
            for surface, transform in (("raw", softmax), ("zscored", zdist)):
                proximal = transform(np.asarray(proximal_arrays[arm], dtype=float))
                terminal = transform(np.asarray(terminal_arrays[arm], dtype=float))
                arm_result.setdefault("proximal_edges", {})[surface] = [
                    per_edge_effect(before[surface], proximal[:, index], facts, control_rows)
                    for index in (0, 1)
                ]
                arm_result.setdefault("terminal_edges", {})[surface] = [
                    per_edge_effect(before[surface], terminal[:, index], facts, control_rows)
                    for index in (0, 1)
                ]
            rng = np.random.default_rng(seed + split_number * 100 + arm_number)
            proximal_summary = summarize(
                {surface: arm_result["proximal_edges"][surface][0] for surface in ("raw", "zscored")},
                {surface: arm_result["proximal_edges"][surface][1] for surface in ("raw", "zscored")},
                relations,
                rng,
            )
            terminal_summary = summarize(
                {surface: arm_result["terminal_edges"][surface][0] for surface in ("raw", "zscored")},
                {surface: arm_result["terminal_edges"][surface][1] for surface in ("raw", "zscored")},
                relations,
                rng,
            )
            results[split][arm] = {
                "proximal": proximal_summary,
                "terminal": terminal_summary,
                "quality_gate_pass": passes_component(proximal_summary, terminal_summary),
            }

    component_pass = {
        arm: all(results[split][arm]["quality_gate_pass"] for split in ("validation", "test"))
        for arm in ("Q", "K", "V")
    }
    e_quality = all(results[split]["E"]["quality_gate_pass"] for split in ("validation", "test"))
    no_hidden = heldout["natural_context_inputs"]["READ_1E_no_hidden_full_execution_gate"] is True
    e_systems_pass = e_quality and no_hidden
    verdict = (
        "CREDIBLE MODEL-NATIVE READ PRIMITIVE"
        if e_systems_pass
        else "mechanistically_compact_but_not_economical_read_path"
    )
    document: dict[str, Any] = {
        "schema": "larql.gwread1.adjudication.v1",
        "status": "complete",
        "adjudication_sha256": None,
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "authorities": {
            "protocol": {"path": str(args.protocol), "sha256": sha(args.protocol)},
            "selection": {"path": str(args.selection), "sha256": sha(args.selection)},
            "heldout": {"path": str(args.heldout), "sha256": sha(args.heldout)},
            "head_capture": {"path": str(args.head_capture), "sha256": sha(args.head_capture)},
            "groups": {"path": str(args.groups), "sha256": sha(args.groups)},
        },
        "selected": selection["selected"],
        "results": results,
        "component_quality_pass": component_pass,
        "READ-1E": {
            "quality_gate_pass": e_quality,
            "no_hidden_full_execution_inputs_pass": no_hidden,
            "charged_context": heldout["natural_context_inputs"],
            "physical_cost_gate_pass": False,
            "reason": "natural carrier_before and seven non-H1 L24 heads require full pre-L24 execution; measured candidate-path latency cannot count that work as avoided",
            "systems_gate_pass": e_systems_pass,
        },
        "verdict": verdict,
        "interpretation": "Q/K/V fidelity and efficient execution are separate; component quality cannot rescue an end-to-end hidden-input or cost failure.",
    }
    document["adjudication_sha256"] = canonical_hash(document, "adjudication_sha256")
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "adjudication_sha256": document["adjudication_sha256"],
        "selected": document["selected"],
        "component_quality_pass": component_pass,
        "READ-1E": document["READ-1E"],
        "verdict": verdict,
        "results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
