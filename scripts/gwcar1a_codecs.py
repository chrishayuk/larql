#!/usr/bin/env python3
"""Deterministic encoded carrier candidates for the outcome-blind CAR-1A grid."""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

WIDTH = 2560
PCA_MAX_RANK = 384
SPARSE_LEVELS = (32, 64, 128, 256, 512, 1024)
PCA_LEVELS = (8, 16, 32, 64, 128, 256, 384)


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    family: str
    level: int


GRID = (
    Candidate(0, "exact", 32),
    Candidate(1, "donor", 32),
    Candidate(2, "f16", 16),
    Candidate(3, "symmetric_quant", 8),
    Candidate(4, "symmetric_quant", 4),
    Candidate(5, "symmetric_quant", 2),
    *(Candidate(6 + i, "sparse_top_abs", k) for i, k in enumerate(SPARSE_LEVELS)),
    *(Candidate(12 + i, "pca", k) for i, k in enumerate(PCA_LEVELS)),
)
assert len(GRID) == 19 and [candidate.candidate_id for candidate in GRID] == list(range(19))


def checked_carrier(carrier: np.ndarray) -> np.ndarray:
    value = np.asarray(carrier, dtype="<f4")
    if value.shape != (WIDTH,) or not np.isfinite(value).all():
        raise ValueError("CAR-1A requires one finite 2560-wide carrier")
    return value


def fit_pca(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen centered train basis with an explicit sign convention."""
    train = np.asarray(train, dtype="<f4")
    if train.shape != (396, WIDTH) or not np.isfinite(train).all():
        raise ValueError("PCA requires all 396 finite frozen train carriers")
    mean = train.astype(np.float64).mean(axis=0).astype("<f4")
    centered = train.astype(np.float64) - mean.astype(np.float64)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    basis = right[:PCA_MAX_RANK].astype("<f4")
    for component in basis:
        if component[int(np.argmax(np.abs(component)))] < 0:
            component *= -1
    if not np.isfinite(basis).all():
        raise ValueError("nonfinite PCA basis")
    return mean, basis


def pca_model_bytes(mean: np.ndarray, basis: np.ndarray) -> bytes:
    if mean.shape != (WIDTH,) or basis.shape != (PCA_MAX_RANK, WIDTH):
        raise ValueError("wrong PCA model geometry")
    return mean.astype("<f4", copy=False).tobytes() + basis.astype("<f4", copy=False).tobytes()


def model_bytes_for_rank(rank: int) -> int:
    if rank not in PCA_LEVELS:
        raise ValueError("unregistered PCA rank")
    return (1 + rank) * WIDTH * 4


def decode_ledger(candidate: Candidate) -> dict[str, int]:
    """Count specified scalar decode work; these are not measured FLOPs."""
    family = candidate.family
    return {
        "multiply_add_pairs": candidate.level * WIDTH if family == "pca" else 0,
        "scalar_additions": WIDTH if family == "pca" else 0,
        "scalar_multiplications": WIDTH if family == "symmetric_quant" else 0,
        "scalar_conversions": WIDTH if family == "f16" else 0,
        "bit_unpack_coordinates": WIDTH if family == "symmetric_quant" else 0,
        "coordinate_writes": candidate.level if family == "sparse_top_abs" else 0,
    }


def _pack_codes(codes: np.ndarray, bits: int) -> bytes:
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    packed = bytearray((WIDTH + per_byte - 1) // per_byte)
    for index, code in enumerate(codes):
        packed[index // per_byte] |= (int(code) & mask) << ((index % per_byte) * bits)
    return bytes(packed)


def _unpack_codes(data: bytes, bits: int) -> np.ndarray:
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    codes = np.empty(WIDTH, dtype=np.int16)
    for index in range(WIDTH):
        raw = (data[index // per_byte] >> ((index % per_byte) * bits)) & mask
        codes[index] = raw - (1 << bits) if raw & sign else raw
    return codes


def encode(candidate: Candidate, natural: np.ndarray, donor: np.ndarray,
           model: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[bytes, np.ndarray, int]:
    """Return actual row payload, decoded carrier, and shared model bytes."""
    natural = checked_carrier(natural)
    donor = checked_carrier(donor)
    family, level = candidate.family, candidate.level
    if family in ("exact", "donor"):
        source = natural if family == "exact" else donor
        payload = source.tobytes()
        decoded = np.frombuffer(payload, dtype="<f4").copy()
        shared = 0
    elif family == "f16":
        payload = natural.astype("<f2").tobytes()
        decoded = np.frombuffer(payload, dtype="<f2").astype("<f4")
        shared = 0
    elif family == "symmetric_quant":
        if level not in (2, 4, 8):
            raise ValueError("unregistered quantization width")
        max_abs = float(np.max(np.abs(natural)))
        qmax = (1 << (level - 1)) - 1
        scale = np.float32(max_abs / qmax)
        codes = (np.zeros(WIDTH, dtype=np.int16) if scale == 0 else
                 np.rint(natural.astype(np.float64) / float(scale))
                 .clip(-qmax, qmax).astype(np.int16))
        payload = struct.pack("<f", float(scale)) + _pack_codes(codes, level)
        decoded = (_unpack_codes(payload[4:], level).astype("<f4") * scale).astype("<f4")
        shared = 0
    elif family == "sparse_top_abs":
        if level not in SPARSE_LEVELS:
            raise ValueError("unregistered sparse level")
        rank = np.lexsort((np.arange(WIDTH), -np.abs(natural)))[:level]
        indices = np.sort(rank)
        payload = b"".join(struct.pack("<Hf", int(i), float(natural[i])) for i in indices)
        decoded = np.zeros(WIDTH, dtype="<f4")
        for index, value in struct.iter_unpack("<Hf", payload):
            decoded[index] = value
        shared = 0
    elif family == "pca":
        if level not in PCA_LEVELS or model is None:
            raise ValueError("unregistered PCA level or missing frozen model")
        mean, basis = model
        if mean.shape != (WIDTH,) or basis.shape != (PCA_MAX_RANK, WIDTH):
            raise ValueError("wrong frozen PCA model geometry")
        coefficients = ((natural.astype(np.float64) - mean.astype(np.float64))
                        @ basis[:level].astype(np.float64).T).astype("<f4")
        payload = coefficients.tobytes()
        decoded = (mean.astype(np.float64) +
                   coefficients.astype(np.float64) @ basis[:level].astype(np.float64)).astype("<f4")
        shared = model_bytes_for_rank(level)
    else:
        raise ValueError("unregistered CAR-1A candidate")
    if not np.isfinite(decoded).all():
        raise ValueError("nonfinite decoded carrier")
    return payload, decoded, shared
