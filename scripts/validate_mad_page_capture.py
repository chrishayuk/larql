#!/usr/bin/env python3
"""Validate multi-schema MAD capture against independent single-schema arms."""

import argparse
import json
from pathlib import Path

import numpy as np


def manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("multi", type=Path)
    parser.add_argument(
        "--single",
        action="append",
        required=True,
        help="SCHEMA=CAPTURE directory; repeat for every schema",
    )
    args = parser.parse_args()
    multi_manifest = manifest(args.multi)
    multi_samples = multi_manifest["samples"]
    multi_shape = (len(multi_samples), len(multi_manifest["objects"]))
    multi_values = np.memmap(
        args.multi / multi_manifest["contributions_file"],
        dtype="<f4",
        mode="r",
        shape=multi_shape,
    )
    result = {"multi": str(args.multi), "schemas": {}}
    for item in args.single:
        schema_text, separator, path_text = item.partition("=")
        if not separator:
            raise ValueError(f"invalid --single value {item!r}")
        schema = int(schema_text)
        path = Path(path_text)
        single_manifest = manifest(path)
        if single_manifest["samples"] != multi_samples:
            raise ValueError(f"schema {schema}: sample manifests differ")
        multi_residual = args.multi / multi_manifest["residuals_file"]
        single_residual = path / single_manifest["residuals_file"]
        residual_identical = multi_residual.read_bytes() == single_residual.read_bytes()
        columns = [
            index
            for index, obj in enumerate(multi_manifest["objects"])
            if obj.get("block_channels") == schema
        ]
        single_columns = [
            index
            for index, obj in enumerate(single_manifest["objects"])
            if obj.get("block_channels") == schema
        ]
        if len(columns) != len(single_columns) or not columns:
            raise ValueError(f"schema {schema}: object counts differ or are empty")
        multi_objects = [multi_manifest["objects"][index] for index in columns]
        single_objects = [single_manifest["objects"][index] for index in single_columns]
        for multi_obj, single_obj in zip(multi_objects, single_objects):
            for key in (
                "layer",
                "kind",
                "byte_count",
                "operand",
                "channel_start",
                "channel_end",
                "block_channels",
            ):
                if multi_obj.get(key) != single_obj.get(key):
                    raise ValueError(f"schema {schema}: object field {key} differs")
        single_shape = (len(multi_samples), len(single_manifest["objects"]))
        single_values = np.memmap(
            path / single_manifest["contributions_file"],
            dtype="<f4",
            mode="r",
            shape=single_shape,
        )
        left = np.asarray(multi_values[:, columns])
        right = np.asarray(single_values[:, single_columns])
        max_abs = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        result["schemas"][str(schema)] = {
            "objects": len(columns),
            "residual_plane_identical": residual_identical,
            "contribution_plane_identical": bool(np.array_equal(left, right)),
            "max_abs_contribution_error": max_abs,
        }
    if not all(
        row["residual_plane_identical"] and row["contribution_plane_identical"]
        for row in result["schemas"].values()
    ):
        raise ValueError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
