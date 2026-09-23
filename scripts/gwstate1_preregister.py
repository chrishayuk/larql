#!/usr/bin/env python3
"""Construct and validate the outcome-blind GW-STATE-1 machine protocol.

`draft` writes an unsealed, non-executable review artifact. `seal` is a
separate explicit operation after joint runner/protocol review; it refuses
to overwrite the draft or an existing seal.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from gwstate1_preflight import AUTHORITIES, ROOT, audit
from gwv2_population import canonical_hash, sha256

SCHEMA = "larql.gwstate1.protocol.v1"
DRAFT_STATUS = "design_draft_no_state1_outcomes"
FROZEN_STATUS = "frozen_before_state1_outcomes"
RUNNER = ROOT / "crates/larql-demos/examples/inference/gwstate1.rs"
ENTRYPOINT = ROOT / "crates/larql-demos/examples/inference/observatory_record.rs"
ANALYSIS = ROOT / "scripts/gwstate1_analysis.py"
SELECTION = ROOT / "scripts/gwstate1_select.py"
ADJUDICATOR = ROOT / "scripts/gwstate1_adjudicate.py"
PREFLIGHT = ROOT / "scripts/gwstate1_preflight.py"
HUMAN_SPEC = ROOT / "docs/gw-state-1.md"
IMPORTED_CODE = (
    "gwv2_adjudicate.py", "gwv2_io.py", "gwv2_population.py",
    "gwv2_prepare.py", "gwv2_amend.py", "gwv2_preregister.py",
    "gwv2_preflight.py", "gwkey1_preregister.py",
    "gwsup1_preregister.py",
)


def authority(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def expected_document() -> dict:
    checked = audit(verify_segments=True)
    return {
        "schema": SCHEMA,
        "status": DRAFT_STATUS,
        "protocol_sha256": None,
        "human_spec": authority(HUMAN_SPEC),
        "authorities": {name: authority(path) for name, (path, _) in sorted(AUTHORITIES.items())},
        "v2_lineage": checked["v2_lineage"],
        "v2_adjudication_identity": checked["v2_adjudication_identity"],
        "execution_identity": checked["execution_identity"],
        "plan_sha256": checked["plan_sha256"],
        "model_metadata": checked["model_metadata"],
        "model_segments": checked["model_segments"],
        "matched_control_donor_indices_sha256": checked["matched_control_donor_indices_sha256"],
        "population": {"rows": 666, "split_rows": checked["split_rows"],
                       "donor_source": "each frozen input row's same_relation_different_subject matched control",
                       "donor_cardinality": "one-to-one derangement within split, relation and prompt family",
                       "fit_sham_donor_map_role": "none"},
        "interface": {"component": "target", "layer": 24, "head": 1,
                      "head_numbering": "zero_based", "position": "last_prompt_token",
                      "h1_baseline": "frozen matched donor exact pre-W_O weighted-V head value",
                      "h1_treatment": "target exact natural pre-W_O weighted-V head value",
                      "context_id": "carrier_bit_times_128_plus_seven_head_mask",
                      "carrier_bit": {"0": "matched donor entering L24 attention carrier",
                                      "1": "target natural entering L24 attention carrier"},
                      "head_mask_order": [0, 2, 3, 4, 5, 6, 7],
                      "head_bit": {"0": "matched donor head", "1": "target natural head"},
                      "h1_arm_order": ["donor", "target_exact_natural"],
                      "train_context_ids": list(range(256)),
                      "heldout_context_ids": "sorted union of 0,127,128,255 and non-null frozen branch selections",
                      "composition": "effective prepared W_O then declared post-attention norm, residual scale and write; intervene at L24 attention write; execute L24 FFN and all later layers",
                      "source_prefix_and_kv": "target row's frozen natural prefix and KV state",
                      "zero_carrier_or_head_arms": "Zero-input arms are excluded because STATE-1 tests dependence on naturally realized carrier/head context under matched substitution, not ablation robustness.",
                      "replay_output_shape": "[stage rows, sorted context IDs, donor/target H1, 142 candidate tokens]",
                      "predictor_search": False},
        "controls": {"natural_composition_raw_relative_l2_max": 1e-5,
                     "natural_composition_carrier_bit_exact": True,
                     "v2_natural_proximal_and_terminal_bit_exact": True,
                     "natural_noop_terminal_bit_exact": True,
                     "full_exact_identity_terminal_bit_exact": True,
                     "all_natural_context_target_h1_proximal_terminal_bit_exact": True,
                     "intervention_firings_per_replacement": 1,
                     "missing_nonfinite_or_cross_split_donor": "invalidate_run"},
        "train_selection": {"effect": "control-adjusted target-H1 minus donor-H1 same-fact JS change minus matched-control JS change",
                            "surfaces": ["proximal", "terminal"],
                            "metrics": ["raw", "independently_row_zscored"],
                            "unit": "subject country; mean over capital, currency and language, each with three prompt families",
                            "reference_context": 255,
                            "retention": "proximal H1 contrast divided by context-255 proximal H1 contrast within metric and split",
                            "minimum_point_retention": 0.80,
                            "positive_proximal_and_terminal": True,
                            "one_selection_per_carrier_branch": True,
                            "tie_break": ["fewest natural non-H1 heads", "highest minimum raw/z proximal retention",
                                          "lowest sum of selected heads' train-row mean natural effective-W_O contribution L2 norms",
                                          "lexicographically ascending head IDs"],
                            "none_if_no_eligible_context": True,
                            "heldout_tuning": False},
        "heldout": {"splits": ["validation", "test"], "independent_split_gates": True,
                    "bootstrap_seed_reset_per_split": 27022033, "bootstrap_replicates": 10000,
                    "bootstrap_unit": "subject country, carrying three relations and nine executions",
                    "bootstrap_subject_order": "sorted subject IDs within each split",
                    "bootstrap_rng": "numpy.default_rng PCG64; reset with seed for each split",
                    "bootstrap_draw": "sample 15 subject indices with replacement per replicate",
                    "bootstrap_standard_error": "draw standard deviation with ddof=1",
                    "bootstrap_quantile": "numpy default linear interpolation",
                    "simultaneous_method": "one-sided 95% studentized max-t lower band",
                    "simultaneous_statistic": "maximum over (point estimate minus bootstrap draw) divided by bootstrap standard error",
                    "simultaneous_family": "distinct selected carrier-branch contexts plus all four fixed sentinels; raw/z proximal retention and raw/z terminal effect in one family per split",
                    "exact_all_natural_retention": "deterministic one; exclude zero-SE identity from studentization",
                    "exact_denominator": "positive point and positive 2.5th-percentile bootstrap lower bound on both metrics; every bootstrap denominator positive",
                    "proximal_point_retention_min": 0.80,
                    "proximal_simultaneous_lower_min": 0.50,
                    "terminal_simultaneous_lower_min": 0.0,
                    "terminal_lower_strict": True,
                    "invalidity_precedes_verdict": "failed exact or mandatory controls gives no_stable_context",
                    "verdict_order_after_validity": ["edge_only", "carrier_plus_edge",
                                                     "local_head_circuit", "broad_context"],
                    "heldout_contexts_outside_selected_plus_sentinels": "forbidden"},
        "runner": {"source": authority(RUNNER), "entrypoint": authority(ENTRYPOINT),
                   "executable_sha256": None, "train_stage": "--gwstate1 train",
                   "heldout_stage": "--gwstate1 heldout", "validation_stage": "--gwstate1 validate"},
        "analysis_implementation": {"source": authority(ANALYSIS),
                                    "selection_source": authority(SELECTION),
                                    "adjudicator_source": authority(ADJUDICATOR),
                                    "preflight_source": authority(PREFLIGHT),
                                    "validator_source": authority(Path(__file__)),
                                    "imported_code": {name: authority(ROOT / "scripts" / name)
                                                      for name in IMPORTED_CODE},
                                    "python_version": platform.python_version(),
                                    "numpy_version": np.__version__},
        "seal_order": ["joint protocol and runner review", "protocol/executable seal",
                       "train full-context replay", "train-only selection seal",
                       "heldout sentinel-and-selected replay", "adjudication"],
        "state1_outcomes_before_seal": 0,
    }


def validate(path: Path) -> dict:
    document = json.loads(path.read_text())
    expected = expected_document()
    if document.get("status") == FROZEN_STATUS:
        executable = document.get("runner", {}).get("executable_sha256")
        if not isinstance(executable, str) or not executable.startswith("sha256:"):
            raise ValueError("frozen protocol lacks executable identity")
        expected["status"] = FROZEN_STATUS
        expected["runner"]["executable_sha256"] = executable
        expected["protocol_sha256"] = canonical_hash(expected, "protocol_sha256")
    elif document.get("status") != DRAFT_STATUS:
        raise ValueError("invalid STATE-1 protocol status")
    if document != expected:
        raise ValueError("STATE-1 protocol differs from checked authorities or declared experiment")
    return {"schema": SCHEMA, "status": document["status"],
            "protocol_sha256": document["protocol_sha256"],
            "matched_control_donor_indices_sha256": document["matched_control_donor_indices_sha256"]}


def write_new(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def seal(draft_path: Path, output: Path, executable: Path) -> dict:
    details = validate(draft_path)
    if details["status"] != DRAFT_STATUS:
        raise ValueError("only an unsealed reviewed draft can be sealed")
    document = json.loads(draft_path.read_text())
    document["status"] = FROZEN_STATUS
    document["runner"]["executable_sha256"] = sha256(executable)
    document["protocol_sha256"] = canonical_hash(document, "protocol_sha256")
    write_new(output, document)
    validate(output)
    return {"protocol_sha256": document["protocol_sha256"],
            "executable_sha256": document["runner"]["executable_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draft", "validate", "seal"])
    parser.add_argument("path", type=Path, help="draft path for draft/validate, frozen output for seal")
    parser.add_argument("extra", nargs="*", type=Path, help="seal: reviewed draft path, executable path")
    args = parser.parse_args()
    if args.command == "draft":
        if args.extra:
            parser.error("draft takes only its new output path")
        write_new(args.path, expected_document())
        result = validate(args.path)
    elif args.command == "validate":
        if args.extra:
            parser.error("validate takes only one protocol path")
        result = validate(args.path)
    else:
        if len(args.extra) != 2:
            parser.error("seal takes FROZEN_OUTPUT REVIEWED_DRAFT EXECUTABLE")
        result = seal(args.extra[0], args.path, args.extra[1])
    print(json.dumps(result, indent=2, sort_keys=True))
