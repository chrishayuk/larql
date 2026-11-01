"""Prospective CAR-1B source constructors; no model replay or outcome selection.

The online constructor accepts metadata and one declared feature vector, never
the target L24 carrier. Oracle controls deliberately have a separate entrypoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

WIDTH = 2560
SUPPORT = 32
RIDGE_SCALE = 0.01
FEATURES = ("subject_embedding", "enter_l12", "enter_l16")


@dataclass(frozen=True)
class Row:
    row_id: str
    subject_id: str
    relation: str
    template: str
    split: str

    @property
    def group(self) -> tuple[str, str]:
        return self.relation, self.template


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    name: str
    family: str
    feature: str | None = None
    neighbors: int = 0
    sham: bool = False


GRID = (
    Candidate(0, "exact", "oracle"),
    Candidate(1, "matched_donor", "oracle"),
    Candidate(2, "target_top32", "oracle"),
    Candidate(3, "train_support_target_values32", "oracle"),
    Candidate(4, "template_target_top_delta32", "oracle"),
    Candidate(5, "template_train_support_target_delta32", "oracle"),
    Candidate(6, "global_mean", "global"),
    Candidate(7, "relation_mean", "relation"),
    Candidate(8, "template_mean", "template"),
    Candidate(9, "template_embedding_nn1", "nearest", "subject_embedding", 1),
    Candidate(10, "template_embedding_nn4", "nearest", "subject_embedding", 4),
    Candidate(11, "template_entity_delta32", "ridge", "subject_embedding"),
    Candidate(12, "template_l12_delta32", "ridge", "enter_l12"),
    Candidate(13, "template_l16_delta32", "ridge", "enter_l16"),
    Candidate(14, "template_l12_nn1", "nearest", "enter_l12", 1),
    Candidate(15, "template_l16_nn1", "nearest", "enter_l16", 1),
    Candidate(16, "zero", "zero"),
    Candidate(17, "entity_sham", "ridge", "subject_embedding", sham=True),
    Candidate(18, "l12_sham", "ridge", "enter_l12", sham=True),
    Candidate(19, "l16_sham", "ridge", "enter_l16", sham=True),
)
CONSTRUCTION_IDS = tuple(range(6, 16))


def grid_document() -> list[dict]:
    return [asdict(candidate) for candidate in GRID]


def matrix(value: np.ndarray, rows: int) -> np.ndarray:
    result = np.asarray(value, dtype="<f4")
    if result.shape != (rows, WIDTH) or not np.isfinite(result).all():
        raise ValueError("expected finite [rows, 2560] source array")
    return result


def vector(value: np.ndarray) -> np.ndarray:
    return matrix(np.asarray(value)[None, :], 1)[0]


def top_support(scores: np.ndarray) -> np.ndarray:
    # Descending score, increasing coordinate breaks ties; serialization sorted.
    return np.sort(np.lexsort((np.arange(WIDTH), -scores))[:SUPPORT]).astype("<u2")


def sparsify(value: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
    value = vector(value)
    if support is None:
        support = top_support(np.abs(value).astype(np.float64))
    result = np.zeros(WIDTH, dtype="<f4")
    result[support] = value[support]
    return result


@dataclass
class Ridge:
    mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    alpha: float

    @property
    def nbytes(self) -> int:
        return self.mean.nbytes + self.target_mean.nbytes + self.weights.nbytes


@dataclass
class Bank:
    rows: tuple[Row, ...]
    carriers: np.ndarray
    features: dict[str, np.ndarray]
    global_mean: np.ndarray
    relations: dict[str, np.ndarray]
    templates: dict[tuple[str, str], np.ndarray]
    carrier_support: np.ndarray
    delta_support: np.ndarray
    ridge: dict[tuple[str, bool], Ridge]


def fit(rows: list[Row], carriers: np.ndarray, features: dict[str, np.ndarray]) -> Bank:
    """Fit a complete, balanced train-only bank; the caller binds input hashes."""
    if not rows or any(row.split != "train" for row in rows):
        raise ValueError("source fitting accepts train rows only")
    if any(not all((r.row_id, r.subject_id, r.relation, r.template)) for r in rows):
        raise ValueError("empty training row identity")
    if len({r.row_id for r in rows}) != len(rows):
        raise ValueError("duplicate training row ID")
    subjects = sorted({r.subject_id for r in rows})
    groups = sorted({r.group for r in rows})
    if len(subjects) < 4 or len(rows) != len(subjects) * len(groups):
        raise ValueError("need at least four subjects and one row per subject/group")
    lookup = {(r.subject_id, r.group): i for i, r in enumerate(rows)}
    if len(lookup) != len(rows) or any((s, g) not in lookup for s in subjects for g in groups):
        raise ValueError("incomplete subject/template crossing")
    if set(features) != set(FEATURES):
        raise ValueError("declared source feature inventory changed")
    order = sorted(range(len(rows)), key=lambda i: rows[i].row_id)
    rows = [rows[i] for i in order]
    carriers = matrix(carriers, len(rows))[order].copy()
    feature_arrays = {name: matrix(features[name], len(rows))[order].copy() for name in FEATURES}
    # L0 subject pooling is explicitly independent of relation and template.
    for subject in subjects:
        indices = [i for i, r in enumerate(rows) if r.subject_id == subject]
        values = feature_arrays["subject_embedding"][indices]
        if not np.all(values.view("<u4") == values[0].view("<u4")):
            raise ValueError("subject embedding varies with template or relation")
    mean = lambda x: x.astype(np.float64).mean(axis=0).astype("<f4")
    templates = {g: mean(carriers[[i for i, r in enumerate(rows) if r.group == g]]) for g in groups}
    relations = {rel: mean(carriers[[i for i, r in enumerate(rows) if r.relation == rel]])
                 for rel in sorted({r.relation for r in rows})}
    deltas = (carriers.astype(np.float64) - np.stack([templates[r.group] for r in rows])).astype("<f4")
    support = top_support(np.abs(deltas.astype(np.float64)).mean(axis=0))
    carrier_support = top_support(np.abs(carriers.astype(np.float64)).mean(axis=0))
    lookup = {(r.subject_id, r.group): i for i, r in enumerate(rows)}
    next_subject = {s: subjects[(i + 1) % len(subjects)] for i, s in enumerate(subjects)}
    permuted = [lookup[next_subject[r.subject_id], r.group] for r in rows]
    fitted = {}
    for name, values in feature_arrays.items():
        feature_mean = mean(values)
        x = values.astype(np.float64) - feature_mean.astype(np.float64)
        gram = x @ x.T
        alpha = max(RIDGE_SCALE * float(np.trace(gram)) / len(rows), 1e-12)
        regularized = gram + alpha * np.eye(len(rows))
        for sham in (False, True):
            y = deltas[permuted if sham else np.arange(len(rows))][:, support].astype(np.float64)
            target_mean = y.mean(axis=0).astype("<f4")
            weights = (x.T @ np.linalg.solve(regularized, y - target_mean)).astype("<f4")
            if not np.isfinite(weights).all():
                raise ValueError("nonfinite fitted source weights")
            fitted[name, sham] = Ridge(feature_mean, target_mean, weights, alpha)
    return Bank(tuple(rows), carriers, feature_arrays, mean(carriers), relations,
                templates, carrier_support, support, fitted)


def construct(bank: Bank, candidate_id: int, query: Row,
              feature: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Construct for a fresh subject without access to its natural L24 carrier.

    Feature extraction is outside this function and must be accounted for by
    the replay manifest. This API cannot itself attest feature provenance.
    """
    if not 0 <= candidate_id < len(GRID):
        raise ValueError("unknown candidate")
    c = GRID[candidate_id]
    if c.family == "oracle":
        raise ValueError("oracle candidates require the separate oracle entrypoint")
    if query.split not in ("validation", "test"):
        raise ValueError("construction query must declare validation or test")
    if query.subject_id in {r.subject_id for r in bank.rows} or query.row_id in {r.row_id for r in bank.rows}:
        raise ValueError("confirmation query overlaps the training bank")
    if query.group not in bank.templates:
        raise ValueError("unregistered relation/template; no fallback")
    if (feature is None) != (c.feature is None):
        raise ValueError("supply exactly the declared source feature")
    x = vector(feature) if feature is not None else None
    neighbors: list[str] = []
    template = bank.templates[query.group]
    if c.family == "global":
        value = bank.global_mean.copy()
        shared = value.nbytes
    elif c.family == "relation":
        value = bank.relations[query.relation].copy()
        shared = sum(v.nbytes for v in bank.relations.values())
    elif c.family == "template":
        value = template.copy()
        shared = sum(v.nbytes for v in bank.templates.values())
    elif c.family == "zero":
        value = np.zeros(WIDTH, dtype="<f4")
        shared = 0
    elif c.family == "nearest":
        eligible = [i for i, r in enumerate(bank.rows) if r.group == query.group]
        distances = ((bank.features[c.feature][eligible].astype(np.float64) - x) ** 2).sum(axis=1)
        # Bank order is lexical row ID, the frozen tie-break for equal distance.
        chosen = np.asarray(eligible)[np.argsort(distances, kind="stable")[:c.neighbors]]
        neighbors = [bank.rows[i].row_id for i in chosen]
        value = bank.carriers[chosen].astype(np.float64).mean(axis=0).astype("<f4")
        shared = bank.carriers.nbytes + bank.features[c.feature].nbytes
    elif c.family == "ridge":
        fitted = bank.ridge[c.feature, c.sham]
        delta = ((x.astype(np.float64) - fitted.mean) @ fitted.weights.astype(np.float64)
                 + fitted.target_mean).astype("<f4")
        value = template.copy()
        value[bank.delta_support] = (value[bank.delta_support].astype(np.float64) + delta).astype("<f4")
        shared = (sum(v.nbytes for v in bank.templates.values()) + bank.delta_support.nbytes
                  + fitted.nbytes)
    else:
        raise ValueError("unimplemented source family")
    value = vector(value)
    receipt = {
        "candidate_id": candidate_id, "query_row_id": query.row_id,
        "feature": c.feature, "retrieved_row_ids": neighbors,
        "reads_target_l24_carrier": False,
        "retained_prefix_blocks": {None: 0, "subject_embedding": 0, "enter_l12": 12, "enter_l16": 16}[c.feature],
        "numeric_shared_bytes": shared,
        "online_feature_bytes": 0 if x is None else x.nbytes,
        "dense_output_bytes": value.nbytes,
        "ridge_multiply_add_pairs": WIDTH * SUPPORT if c.family == "ridge" else 0,
        "distance_coordinates": len(eligible) * WIDTH if c.family == "nearest" else 0,
        "metadata_bytes_included": False,
        "feature_extraction_cost_included": False,
        "end_to_end_prefix_avoided": False,
        "efficiency_claim": False,
    }
    return value, receipt


def oracle(bank: Bank, candidate_id: int, query: Row, natural: np.ndarray,
           donor: np.ndarray | None = None) -> np.ndarray:
    """Controls that explicitly consume unavailable target/donor carrier state."""
    natural = vector(natural)
    if candidate_id == 0:
        return natural.copy()
    if candidate_id == 1:
        if donor is None:
            raise ValueError("matched donor carrier required")
        return vector(donor).copy()
    if candidate_id == 2:
        return sparsify(natural)
    if candidate_id == 3:
        return sparsify(natural, bank.carrier_support)
    if candidate_id not in (4, 5):
        raise ValueError("not an oracle candidate")
    template = bank.templates[query.group]
    delta = (natural.astype(np.float64) - template).astype("<f4")
    selected = None if candidate_id == 4 else bank.delta_support
    return (template.astype(np.float64) + sparsify(delta, selected)).astype("<f4")
