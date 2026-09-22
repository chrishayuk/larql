#!/usr/bin/env python3
"""Validate ADDR-1 seams against a completed PAGE-1 authority capture."""

import argparse
import json
from pathlib import Path

import numpy as np

import audit_mad_addresses as addresses
import audit_mad_pages as pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("seams", type=Path)
    args = parser.parse_args()
    reference = json.loads((args.reference / "manifest.json").read_text())
    manifest = json.loads((args.seams / "manifest.json").read_text())
    reference_rows = {sample["id"]: index for index, sample in enumerate(reference["samples"])}
    reference_indices = []
    for sample in manifest["samples"]:
        index = reference_rows.get(sample["id"])
        if index is None:
            raise ValueError(f"reference lacks sample {sample['id']}")
        reference_sample = reference["samples"][index]
        for field in (
            "split",
            "group_id",
            "operation_id",
            "wording_id",
            "alias_id",
            "answer_exact",
            "generated_token_ids",
        ):
            if sample.get(field) != reference_sample.get(field):
                raise ValueError(f"sample {sample['id']} differs at {field}")
        reference_indices.append(index)

    reference_residuals = pages.open_plane(
        args.reference,
        reference,
        "residuals_file",
        (
            len(reference["samples"]),
            len(reference["residual_layers"]),
            reference["hidden_size"],
        ),
    )
    seam_residuals = pages.open_plane(
        args.seams,
        manifest,
        "residuals_file",
        (
            len(manifest["samples"]),
            len(manifest["residual_layers"]),
            manifest["hidden_size"],
        ),
    )
    residual_error = 0.0
    for layer in sorted(set(reference["residual_layers"]) & set(manifest["residual_layers"])):
        expected = np.asarray(
            reference_residuals[
                reference_indices, reference["residual_layers"].index(layer), :
            ]
        )
        actual = np.asarray(
            seam_residuals[:, manifest["residual_layers"].index(layer), :]
        )
        residual_error = max(residual_error, float(np.max(np.abs(expected - actual))))

    reference_contributions = pages.open_plane(
        args.reference,
        reference,
        "contributions_file",
        (len(reference["samples"]), len(reference["objects"])),
    )
    seam_contributions = pages.open_plane(
        args.seams,
        manifest,
        "contributions_file",
        (len(manifest["samples"]), len(manifest["objects"])),
    )
    reference_objects = {obj["id"]: index for index, obj in enumerate(reference["objects"])}
    contribution_error = 0.0
    for actual_column, obj in enumerate(manifest["objects"]):
        expected_column = reference_objects.get(obj["id"])
        if expected_column is None:
            raise ValueError(f"reference lacks object {obj['id']}")
        expected = np.asarray(
            reference_contributions[np.ix_(reference_indices, [expected_column])]
        )
        actual = np.asarray(seam_contributions[:, [actual_column]])
        contribution_error = max(
            contribution_error, float(np.max(np.abs(expected - actual)))
        )

    seams = addresses.open_seams(args.seams, manifest)
    seam_layers = manifest["seam_layers"]
    seam_kinds = manifest["seam_kinds"]
    residual_slots = {layer: manifest["residual_layers"].index(layer) for layer in seam_layers}
    seam_slots = {layer: seam_layers.index(layer) for layer in seam_layers}
    kind_slots = {kind: seam_kinds.index(kind) for kind in seam_kinds}
    attention_join_error = 0.0
    ffn_join_error = 0.0
    for layer in seam_layers:
        layer_input = np.asarray(seam_residuals[:, residual_slots[layer], :])
        attention_output = np.asarray(
            seams[:, seam_slots[layer], kind_slots["attention_output"], :]
        )
        post_attention = np.asarray(
            seams[:, seam_slots[layer], kind_slots["post_attention"], :]
        )
        attention_join_error = max(
            attention_join_error,
            float(np.max(np.abs((layer_input + attention_output) - post_attention))),
        )
        if layer + 1 in residual_slots:
            ffn_output = np.asarray(
                seams[:, seam_slots[layer], kind_slots["ffn_output"], :]
            )
            next_input = np.asarray(seam_residuals[:, residual_slots[layer + 1], :])
            ffn_join_error = max(
                ffn_join_error,
                float(np.max(np.abs((post_attention + ffn_output) - next_input))),
            )
    if not np.isfinite(seams).all():
        raise ValueError("seam plane contains non-finite values")
    if residual_error != 0.0 or contribution_error != 0.0:
        raise ValueError(
            f"capture changed authority planes: residual={residual_error}, contribution={contribution_error}"
        )
    print(
        json.dumps(
            {
                "samples": len(manifest["samples"]),
                "residual_max_abs_error": residual_error,
                "contribution_max_abs_error": contribution_error,
                "attention_join_max_abs_error": attention_join_error,
                "ffn_join_max_abs_error": ffn_join_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
