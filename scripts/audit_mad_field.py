#!/usr/bin/env python3
"""MAD-V3-FIELD-1 continuous future-computation audit.

The evaluator reuses a sealed MAD capture and predicts held-out future
contribution profiles from exact residual cosine neighbourhoods.  It does not
change model execution and supplies no operation, answer, wording or basin
labels to the predictors.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "larql.mad-v3-field-1.audit.v1"
METHODS = (
    "popularity",
    "uniform_knn",
    "distance_knn",
    "local_linear",
    "local_low_rank",
    "oracle",
)
CANDIDATES = ("distance_knn", "local_linear", "local_low_rank")


def parse_layers(spec: str, available: list[int]) -> list[int]:
    selected = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))
    selected = list(dict.fromkeys(selected))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"capture has no residual layers {missing}")
    return selected


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= np.finfo(np.float64).eps).any():
        raise ValueError("capture contains zero or non-finite residuals")
    return values / norms


def safe_profile(candidate: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    candidate = np.maximum(np.asarray(candidate, dtype=np.float64), 0.0)
    if not np.isfinite(candidate).all() or candidate.sum() <= np.finfo(np.float64).eps:
        return np.asarray(fallback, dtype=np.float64).copy(), True
    return candidate, False


def distance_weighted_profile(
    neighbor_scores: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    distances = np.maximum(1.0 - np.asarray(neighbor_scores, dtype=np.float64), 0.0)
    bandwidth = max(float(distances.max()), np.finfo(np.float64).eps)
    weights = np.exp(-0.5 * np.square(distances / bandwidth))
    weights /= weights.sum()
    return weights @ targets


def local_linear_profile(
    query: np.ndarray, neighbors: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    delta = np.asarray(neighbors, dtype=np.float64) - np.asarray(query, dtype=np.float64)
    gram = delta @ delta.T
    ridge = 1e-3 * float(np.trace(gram)) / len(neighbors)
    ridge = max(ridge, np.finfo(np.float64).eps)
    regularized = gram + ridge * np.eye(len(neighbors), dtype=np.float64)
    try:
        weights = np.linalg.solve(regularized, np.ones(len(neighbors), dtype=np.float64))
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(
            regularized, np.ones(len(neighbors), dtype=np.float64), rcond=None
        )[0]
    total = float(weights.sum())
    if not math.isfinite(total) or abs(total) <= np.finfo(np.float64).eps:
        return np.asarray(targets, dtype=np.float64).mean(axis=0)
    weights /= total
    return weights @ targets


def local_low_rank_profile(
    query: np.ndarray, neighbors: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, int]:
    neighbors = np.asarray(neighbors, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    x_mean = neighbors.mean(axis=0)
    y_mean = targets.mean(axis=0)
    x_centered = neighbors - x_mean
    gram = x_centered @ x_centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > np.finfo(np.float64).eps * max(1.0, eigenvalues[0])
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    if not len(eigenvalues):
        return y_mean, 0

    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    variance_rank = int(np.searchsorted(cumulative, 0.95) + 1)
    rank = min(variance_rank, 8, max(1, len(neighbors) - 2), len(eigenvalues))
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]
    scales = np.sqrt(eigenvalues)
    train_scores = eigenvectors * scales
    query_cross = (np.asarray(query, dtype=np.float64) - x_mean) @ x_centered.T
    query_scores = (query_cross @ eigenvectors) / scales
    y_centered = targets - y_mean
    ridge = max(1e-3 * float(eigenvalues.mean()), np.finfo(np.float64).eps)
    coefficients = np.linalg.solve(
        train_scores.T @ train_scores + ridge * np.eye(rank),
        train_scores.T @ y_centered,
    )
    return y_mean + query_scores @ coefficients, rank


def rank_density(profile: np.ndarray, byte_counts: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(len(profile)), -(profile / byte_counts)))


def coverage(
    truth: np.ndarray, ranking: np.ndarray, byte_counts: np.ndarray, budget: float
) -> dict:
    limit = budget * float(byte_counts.sum())
    used = 0
    covered = 0.0
    for column in ranking:
        cost = int(byte_counts[column])
        if used + cost > limit:
            continue
        used += cost
        covered += float(truth[column])
    total = float(truth.sum())
    return {
        "candidate_byte_fraction": used / float(byte_counts.sum()),
        "contribution_coverage": covered / total if total else 0.0,
    }


def vector_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    truth_total = float(truth.sum())
    prediction_total = float(prediction.sum())
    if truth_total <= 0 or prediction_total <= 0:
        return {"cosine": 0.0, "js_divergence": math.log(2.0)}
    p = truth / truth_total
    q = prediction / prediction_total
    denominator = float(np.linalg.norm(p) * np.linalg.norm(q))
    cosine = float(p @ q / denominator) if denominator else 0.0
    midpoint = 0.5 * (p + q)

    def kl(left, right):
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return {
        "cosine": cosine,
        "js_divergence": 0.5 * (kl(p, midpoint) + kl(q, midpoint)),
    }


def paired_sign_flip(
    differences: np.ndarray, permutations: int, seed: int
) -> dict:
    differences = np.asarray(differences, dtype=np.float64)
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = permutations
    while remaining:
        batch = min(1024, remaining)
        signs = rng.integers(0, 2, size=(batch, len(differences)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        exceedances += int(np.count_nonzero((signs * differences).mean(axis=1) >= observed))
        remaining -= batch
    return {
        "mean_delta": observed,
        "win_fraction": float(np.mean(differences > 0)),
        "ties_fraction": float(np.mean(differences == 0)),
        "permutations": permutations,
        "exceedances": exceedances,
        "p_value_plus_one": (exceedances + 1) / (permutations + 1),
    }


def mean_rows(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def audit_layer(
    samples,
    residuals,
    layer_slot,
    contributions,
    object_columns,
    byte_counts,
    history,
    queries,
    neighbors,
    budgets,
    permutations,
    comparison_count,
    seed,
):
    history_matrix = normalized(np.asarray(residuals[history, layer_slot, :]))
    query_matrix = normalized(np.asarray(residuals[queries, layer_slot, :]))
    history_row = {sample: row for row, sample in enumerate(history)}
    history_targets = np.asarray(
        contributions[np.ix_(history, object_columns)], dtype=np.float64
    )
    popularity = history_targets.mean(axis=0)
    coverage_rows = {
        method: {str(budget): [] for budget in budgets} for method in METHODS
    }
    vector_rows = {method: [] for method in METHODS}
    fallbacks = {method: 0 for method in CANDIDATES}
    ranks = []

    for query_row, query in enumerate(queries):
        query_vector = query_matrix[query_row]
        scores = history_matrix @ query_vector
        eligible = [
            sample
            for sample in history
            if samples[sample].get("group_id") != samples[query].get("group_id")
        ]
        if len(eligible) < neighbors:
            raise ValueError(
                f"query {samples[query]['id']} has only {len(eligible)} eligible history rows"
            )
        chosen = sorted(
            eligible, key=lambda sample: (-scores[history_row[sample]], sample)
        )[:neighbors]
        rows = np.array([history_row[sample] for sample in chosen], dtype=np.int64)
        chosen_scores = scores[rows]
        chosen_residuals = history_matrix[rows]
        chosen_targets = history_targets[rows]
        uniform = chosen_targets.mean(axis=0)
        low_rank, rank = local_low_rank_profile(
            query_vector, chosen_residuals, chosen_targets
        )
        ranks.append(rank)
        raw_profiles = {
            "popularity": popularity,
            "uniform_knn": uniform,
            "distance_knn": distance_weighted_profile(chosen_scores, chosen_targets),
            "local_linear": local_linear_profile(
                query_vector, chosen_residuals, chosen_targets
            ),
            "local_low_rank": low_rank,
            "oracle": np.asarray(contributions[query, object_columns], dtype=np.float64),
        }
        profiles = {}
        for method, candidate in raw_profiles.items():
            if method in CANDIDATES:
                profiles[method], used_fallback = safe_profile(candidate, uniform)
                fallbacks[method] += int(used_fallback)
            else:
                profiles[method] = np.asarray(candidate, dtype=np.float64)
        truth = profiles["oracle"]
        for method, profile in profiles.items():
            vector_rows[method].append(vector_metrics(truth, profile))
            ranking = rank_density(profile, byte_counts)
            for budget in budgets:
                coverage_rows[method][str(budget)].append(
                    coverage(truth, ranking, byte_counts, budget)
                )

    methods = {}
    for method in METHODS:
        methods[method] = {
            "vector": mean_rows(vector_rows[method]),
            "budgets": {
                key: mean_rows(rows) for key, rows in coverage_rows[method].items()
            },
        }
    popularity_coverage = methods["popularity"]["budgets"]
    oracle_coverage = methods["oracle"]["budgets"]
    for method in METHODS:
        recovered = {}
        for budget in budgets:
            key = str(budget)
            base = popularity_coverage[key]["contribution_coverage"]
            ceiling = oracle_coverage[key]["contribution_coverage"]
            value = methods[method]["budgets"][key]["contribution_coverage"]
            recovered[key] = (value - base) / (ceiling - base) if ceiling > base else None
        methods[method]["oracle_headroom_recovered"] = recovered

    comparisons = {}
    for method_index, method in enumerate(CANDIDATES):
        comparisons[method] = {
            "vector_delta_vs_uniform": {
                key: methods[method]["vector"][key]
                - methods["uniform_knn"]["vector"][key]
                for key in ("cosine", "js_divergence")
            },
            "budgets": {},
        }
        for budget_index, budget in enumerate(budgets):
            key = str(budget)
            candidate = np.array(
                [row["contribution_coverage"] for row in coverage_rows[method][key]]
            )
            baseline = np.array(
                [
                    row["contribution_coverage"]
                    for row in coverage_rows["uniform_knn"][key]
                ]
            )
            test = paired_sign_flip(
                candidate - baseline,
                permutations,
                seed + method_index * 100 + budget_index,
            )
            test["bonferroni_p_60"] = min(
                1.0, test["p_value_plus_one"] * comparison_count
            )
            comparisons[method]["budgets"][key] = test
    return {
        "queries": len(queries),
        "neighbors": neighbors,
        "methods": methods,
        "comparisons_vs_uniform": comparisons,
        "fallbacks": fallbacks,
        "local_low_rank": {
            "mean_rank": float(np.mean(ranks)),
            "min_rank": int(min(ranks)),
            "max_rank": int(max(ranks)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frontier-layers", default="36-45")
    parser.add_argument("--target-layer", type=int, default=49)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--byte-budgets", default="0.05,0.2")
    parser.add_argument("--seed", type=int, default=0x4649454C4431)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.neighbors, args.permutations) <= 0:
        raise ValueError("count arguments must be positive")

    manifest_path = args.capture / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("capture is incomplete: manifest.json is absent")
    manifest = json.loads(manifest_path.read_text())
    samples = manifest["samples"]
    layers = manifest["residual_layers"]
    frontier = parse_layers(args.frontier_layers, layers)
    budgets = [float(value) for value in args.byte_budgets.split(",")]
    if any(not 0 < budget <= 1 for budget in budgets):
        raise ValueError("byte budgets must be in (0, 1]")
    history = [
        index
        for index, sample in enumerate(samples)
        if sample["split"] == "history" and sample.get("answer_exact") is True
    ]
    queries = [
        index
        for index, sample in enumerate(samples)
        if sample["split"] == "query" and sample.get("answer_exact") is True
    ]
    if not history or not queries:
        raise ValueError("FIELD-1 requires exact-success history and query executions")
    object_columns = [
        index
        for index, obj in enumerate(manifest["objects"])
        if obj["layer"] == args.target_layer
    ]
    if not object_columns:
        raise ValueError(f"capture has no contribution objects at layer {args.target_layer}")
    byte_counts = np.array(
        [manifest["objects"][column]["byte_count"] for column in object_columns],
        dtype=np.float64,
    )
    residual_shape = (len(samples), len(layers), manifest["hidden_size"])
    contribution_shape = (len(samples), len(manifest["objects"]))
    expected_residual_bytes = int(np.prod(residual_shape)) * 4
    expected_contribution_bytes = int(np.prod(contribution_shape)) * 4
    residual_path = args.capture / manifest["residuals_file"]
    contribution_path = args.capture / manifest["contributions_file"]
    if residual_path.stat().st_size != expected_residual_bytes:
        raise ValueError("residual plane size does not match manifest")
    if contribution_path.stat().st_size != expected_contribution_bytes:
        raise ValueError("contribution plane size does not match manifest")
    residuals = np.memmap(residual_path, dtype="<f4", mode="r", shape=residual_shape)
    contributions = np.memmap(
        contribution_path, dtype="<f4", mode="r", shape=contribution_shape
    )

    reports = []
    comparison_count = len(CANDIDATES) * len(frontier) * len(budgets)
    for layer_index, layer in enumerate(frontier):
        reports.append(
            {
                "layer": layer,
                **audit_layer(
                    samples,
                    residuals,
                    layers.index(layer),
                    contributions,
                    object_columns,
                    byte_counts,
                    history,
                    queries,
                    args.neighbors,
                    budgets,
                    args.permutations,
                    comparison_count,
                    args.seed + layer_index * 10_000,
                ),
            }
        )
    result = {
        "schema": SCHEMA,
        "capture": str(args.capture),
        "frontier_layers": frontier,
        "target_layer": args.target_layer,
        "neighbors": args.neighbors,
        "byte_budgets": budgets,
        "permutations": args.permutations,
        "multiplicity_comparisons": comparison_count,
        "history_successes": len(history),
        "query_successes": len(queries),
        "layers": reports,
        "interpretation_guard": (
            "This is a discovery fold because uniform KNN on the same held-out capture "
            "was known before the continuous estimators were preregistered."
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
