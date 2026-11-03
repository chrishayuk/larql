"""E1 machinery: per-tensor weight reconstruction metrics and the T2 bound (PROTOCOL.md A1, A2, rule T2).

E1 compares decoded container weights with the upstream checkpoint's weights. Nothing here runs a model.

T2, for an element x in a block whose stored scale is d16:

    |x - q*d16| <= d16 * (1/2 + 2**-12 + 2**-16 + 127 * 2**-11) + 127 * 2**-25

The constant factor is 36849 * 2**-16. It is applied to d16 in float64: d16 has 11 significant bits and the factor 16, so
the product is exact, and the added 127 * 2**-25 term is exact too (the sum spans at most 41 bits). The error
|x - q*d16| is also formed in float64; x and q*d16 are both f32 values, so the difference is exact unless their exponents
differ by more than 29 bits (not the case for a block of weights within one scale). The tests check both facts against
exact rational arithmetic.

T2's domain (decision revision D1, OQ1). Every Q8_0 block is in exactly one class:

  * invalid under the format: the container reader rejects it, so it never reaches this module;
  * valid, outside T2's domain: the stored scale is negative and finite, +inf, -inf or NaN. T2 makes no statement about
    these blocks. They stay in the inventory, are never counted as within T2 and never as exceeding T2;
  * valid, inside T2's domain: a finite d16 >= 0, +0 and -0 included. T2 is evaluated and each element is within or
    exceeds. Blocks with a +/-0 scale keep their own count.

Element level (D5 AMB-019): inside a domain block, an element whose upstream value is NaN, +inf or -inf is outside T2's
finite-input domain. It is not a bound violation: it is counted per tensor by class (nan, inf, -inf) and is never within
T2 nor exceeding it. (D5 put its error into the floating-point statistics as non-finite; for such a tensor D6 replaces
that with null full-tensor metrics, below.) That is not a claim that the encoder or decoder behaved correctly on it.

Every other statistic includes every block, and a non-finite value is reported as non-finite (inf or NaN), not dropped.
All counts are Python ints, so a scorer can compare them with a control. `canonical_json` writes a report
deterministically.

Non-finite upstream tensors (decision revision D6, every representation). When any upstream element of a tensor is
NaN, +inf or -inf, its full-tensor metrics (`max_abs_error`, `absolute_frobenius_error` and its alias `err_frobenius`,
`ref_frobenius`, `rel_frobenius`) are None (JSON null), each with the reason `nonfinite_upstream` in
`null_metric_reasons`, and `numerical_metric_status` is `non_comparable_nonfinite`. `upstream_nonfinite` counts the
upstream elements by class (`nan`, `+inf`, `-inf`). Bit identity (`bit_identical`, `n_bit_mismatches`) is still
reported, NaN payloads and infinity signs compared bit for bit. A finite-subset diagnostic is reported only under a
`finite_subset_*` name. D5's element-level T2 counts are unchanged. A tensor whose upstream values are all finite has
`numerical_metric_status` `comparable` and its statistics are as before, including a non-finite *decoded* value (D1
OQ1), which stays non-finite. All statistics are computed in binary64 (D6, question 22).

Conventions where the protocol is silent are AMB-010 to AMB-012 and AMB-030 in AMBIGUITIES.md.
"""

from __future__ import annotations

import json
import math
from typing import Callable, Mapping

import numpy as np

from . import q8_0

# T2 constants, spelled as the protocol writes them.
T2_FACTOR = 0.5 + 2.0**-12 + 2.0**-16 + 127 * 2.0**-11
T2_ADDEND = 127 * 2.0**-25

QUANTILES = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
# Upper edges of the error/bound histogram bins; a final bin counts ratio > 1 (T2 exceeded).
RATIO_EDGES = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


# Scale classes outside T2's domain (D1 OQ1), in the order the report lists them.
OUT_OF_DOMAIN_CLASSES = ("negative", "pos_inf", "neg_inf", "nan")
# Element-level classes of a non-finite upstream value inside a domain block (D5 AMB-019).
NONFINITE_UPSTREAM_CLASSES = ("nan", "inf", "-inf")

# D6: tensor-level numerical-metric status, null reasons and the whole-tensor upstream non-finite classes.
METRIC_STATUS_COMPARABLE = "comparable"
METRIC_STATUS_NONFINITE = "non_comparable_nonfinite"
NULL_REASON_NONFINITE_UPSTREAM = "nonfinite_upstream"
NULL_REASON_ZERO_REFERENCE_NORM = "zero_reference_norm"
UPSTREAM_NONFINITE_CLASSES = ("nan", "+inf", "-inf")
# The full-tensor finite-real metrics D6 makes null for a non-finite upstream tensor. `err_frobenius` is the older name
# of `absolute_frobenius_error` (D6 question 22), kept as an alias with the same value.
FULL_TENSOR_METRICS = ("max_abs_error", "absolute_frobenius_error", "err_frobenius", "ref_frobenius", "rel_frobenius")


def upstream_nonfinite_counts(x) -> dict:
    """Count of upstream elements in each non-finite class over the whole tensor (D6): `nan`, `+inf`, `-inf`."""
    x = np.asarray(x)
    return {"nan": int(np.count_nonzero(np.isnan(x))),
            "+inf": int(np.count_nonzero(np.isposinf(x))),
            "-inf": int(np.count_nonzero(np.isneginf(x)))}


def _full_tensor_fields(up_counts: dict, max_err: float, err_fro: float, ref_fro: float, rel, zero_ref: bool,
                        err: np.ndarray, fin: np.ndarray) -> dict:
    """The full-tensor metrics and their D6 status for one tensor. `err` is the element error array (binary64) and
    `fin` the elements where upstream and decoded are both finite (the finite subset)."""
    n_nf = sum(up_counts.values())
    out = {"upstream_nonfinite": dict(up_counts), "n_upstream_nonfinite": n_nf}
    if n_nf:
        # D6: not comparable. Null, never 0 and never inf/NaN, each with its reason.
        out["numerical_metric_status"] = METRIC_STATUS_NONFINITE
        out.update({f: None for f in FULL_TENSOR_METRICS})
        out["zero_reference_norm"] = False  # the upstream holds a non-finite element, so it is not the zero tensor
        out["null_metric_reasons"] = {f: NULL_REASON_NONFINITE_UPSTREAM for f in FULL_TENSOR_METRICS}
        sub = err[fin]
        out["finite_subset_n_elements"] = int(sub.size)
        out["finite_subset_max_abs_error"] = float(sub.max()) if sub.size else None
        return out
    out["numerical_metric_status"] = METRIC_STATUS_COMPARABLE
    out.update(max_abs_error=max_err, absolute_frobenius_error=err_fro, err_frobenius=err_fro, ref_frobenius=ref_fro,
               rel_frobenius=rel, zero_reference_norm=zero_ref)
    out["null_metric_reasons"] = {"rel_frobenius": NULL_REASON_ZERO_REFERENCE_NORM} if zero_ref else {}
    return out


def _bit_compare(upstream, decoded_f32: np.ndarray) -> tuple[int, bool]:
    """(n_bit_mismatches, bit_identical) of a decoded float32 tensor against the upstream values, as bit patterns (NaN
    payloads, signs of zero and of infinity all count). Upstream float64 values are compared as binary64 patterns
    against the decoded values widened exactly to binary64; any other upstream dtype (float32, float16, bfloat16) is
    widened exactly to float32 and compared as binary32 patterns (AMB-030)."""
    u = np.asarray(upstream).reshape(-1)
    d = np.asarray(decoded_f32, dtype=np.float32).reshape(-1)
    if u.shape != d.shape:
        raise ValueError(f"shape mismatch {u.shape} vs {d.shape}")
    if u.dtype == np.float64:
        neq = u.view(np.uint64) != d.astype(np.float64).view(np.uint64)
    else:
        neq = q8_0.f32_bits(u.astype(np.float32)) != q8_0.f32_bits(d)
    n = int(np.count_nonzero(neq))
    return n, n == 0


def in_t2_domain(d) -> np.ndarray:
    """True where the stored scale is finite and >= 0 (-0 included). False for negative, +/-inf and NaN."""
    d = np.asarray(d, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(d) & (d >= 0.0)


def scale_class(d) -> np.ndarray:
    """Per scale: 0 inside T2's domain, else 1 + the index into OUT_OF_DOMAIN_CLASSES."""
    d = np.asarray(d, dtype=np.float64)
    out = np.zeros(d.shape, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        out[np.isfinite(d) & (d < 0.0)] = 1
    out[np.isposinf(d)] = 2
    out[np.isneginf(d)] = 3
    out[np.isnan(d)] = 4
    return out


def t2_bound_from_scale(d: np.ndarray) -> np.ndarray:
    """T2 bound for each stored scale, given as float64 values of the f16 scales. Defined for a finite d >= 0 (+/-0
    included); NaN outside T2's domain (negative, +/-inf, NaN), where T2 makes no statement."""
    d = np.asarray(d, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out = d * T2_FACTOR + T2_ADDEND
    return np.where(in_t2_domain(d), out, np.nan)


def block_scales_f64(buf) -> np.ndarray:
    """Stored f16 scale of each block as float64 (exact)."""
    return q8_0.f16_bits_to_f32_bits(q8_0.block_scale_bits(buf)).view(np.float32).astype(np.float64)


def t2_bounds(buf) -> np.ndarray:
    """T2 bound of every block of a Q8_0 run, shape (n_blocks,)."""
    return t2_bound_from_scale(block_scales_f64(buf))


def _summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {f"q{q:g}": None for q in QUANTILES}
    qs = np.quantile(values, QUANTILES, method="inverted_cdf")  # an observed value, never interpolated (inf-safe)
    return {f"q{q:g}": float(v) for q, v in zip(QUANTILES, qs)}


def _histogram(ratio: np.ndarray) -> dict:
    labels, lo = [], 0.0
    counts = []
    for hi in RATIO_EDGES:
        if hi == RATIO_EDGES[-1]:
            counts.append(int(np.count_nonzero((ratio >= lo) & (ratio <= hi))))
            labels.append(f"[{lo:g},{hi:g}]")
        else:
            counts.append(int(np.count_nonzero((ratio >= lo) & (ratio < hi))))
            labels.append(f"[{lo:g},{hi:g})")
        lo = hi
    counts.append(int(np.count_nonzero(ratio > 1.0)))
    labels.append(f">{RATIO_EDGES[-1]:g}")
    return dict(zip(labels, counts))


def e1_q8_tensor(upstream, q8_bytes, shape: tuple[int, ...] | None = None) -> dict:
    """E1 for one Q8_0 tensor. `upstream` is the upstream values (any float dtype that is exact in float64);
    `q8_bytes` the container's blocks. Blocks run along the last dimension (AMB-009), so the flat order of both
    is the block order. Returns a JSON-serialisable dict.

    Reported (A1): max |x - decoded|, relative Frobenius error, the block-level distribution of error and of
    error/bound, the number of elements and blocks exceeding T2, the number of blocks with a +/-0 scale; and (D6) bit
    identity of the decode against upstream, the numerical-metric status and the upstream non-finite counts.
    """
    with np.errstate(invalid="ignore"):  # a signalling-NaN upstream value is quietened here; it stays NaN
        x = np.asarray(upstream, dtype=np.float64)
    shape = tuple(x.shape) if shape is None else tuple(shape)
    if x.size % q8_0.QK8_0:
        raise ValueError(f"{x.size} elements is not a multiple of {q8_0.QK8_0}")
    nb = x.size // q8_0.QK8_0
    if len(q8_bytes) != nb * q8_0.BLOCK_BYTES:
        raise ValueError(f"{len(q8_bytes)} Q8_0 bytes do not hold {nb} blocks of {q8_0.BLOCK_BYTES}")

    dec32 = q8_0.dequantize_q8_0(q8_bytes)
    n_bit_mismatches, bit_identical = _bit_compare(upstream, dec32)
    dec = dec32.astype(np.float64).reshape(nb, q8_0.QK8_0)
    xb = x.reshape(nb, q8_0.QK8_0)
    d = block_scales_f64(q8_bytes)
    cls = scale_class(d)
    dom = cls == 0  # block is inside T2's domain (D1 OQ1)
    bound = t2_bound_from_scale(d)  # NaN outside the domain
    dom_el = np.broadcast_to(dom[:, None], xb.shape)

    with np.errstate(invalid="ignore", over="ignore"):
        err = np.abs(xb - dec)  # NaN where either side is NaN, or inf - inf; inf where one side is inf
        # D5 AMB-019: inside a domain block, an element whose upstream value is NaN, +inf or -inf is outside T2's
        # finite-input domain at element level. It is neither within nor exceeding; it is counted by class.
        up_nan = dom_el & np.isnan(xb)
        up_pinf = dom_el & np.isposinf(xb)
        up_ninf = dom_el & np.isneginf(xb)
        nonfin_up = up_nan | up_pinf | up_ninf
        t2_el = dom_el & ~nonfin_up  # elements T2 is evaluated on
        # Outside the domain nothing is within and nothing exceeds. The scale is finite here, so the decoded value is
        # finite and err is finite for every element of t2_el.
        within = t2_el & (err <= bound[:, None])
        exceed = t2_el & ~within
        ratio = np.where(t2_el, err / bound[:, None], np.nan)  # NaN = no ratio (outside the domain or element)

    fin = np.isfinite(xb) & np.isfinite(dec)
    # Every block and element is included. A non-finite error stays non-finite (NaN beats inf in the maxima).
    max_err = _max_reported(err)
    block_max_err = np.where(np.isnan(err), np.inf, err).max(axis=1) if nb else np.zeros(0)
    err_fro = _norm(err)
    ref_fro = _norm(xb)
    rel, zero_ref = _rel_frobenius(err_fro, ref_fro)

    if t2_el.any():
        ratio_for_max = np.where(np.isnan(ratio), -1.0, ratio)
        worst_ratio = float(ratio_for_max.max())
        worst_idx = int(np.argmax(ratio_for_max))
        block_max_ratio = ratio_for_max.max(axis=1)
        block_ratio = block_max_ratio[block_max_ratio >= 0.0]  # blocks with at least one element T2 is evaluated on
    else:
        worst_ratio, worst_idx, block_ratio = None, None, np.zeros(0)

    outside = {}
    for i, name in enumerate(OUT_OF_DOMAIN_CLASSES, start=1):
        outside[name] = {"blocks": int(np.count_nonzero(cls == i)),
                         "elements": int(np.count_nonzero(cls == i)) * q8_0.QK8_0}
    n_out_blocks = int(np.count_nonzero(~dom))
    hist = _histogram(ratio)
    hist["outside_t2_domain"] = n_out_blocks * q8_0.QK8_0
    n_nan_up, n_pinf_up, n_ninf_up = (int(np.count_nonzero(m)) for m in (up_nan, up_pinf, up_ninf))
    hist["nonfinite_upstream_in_t2_domain"] = n_nan_up + n_pinf_up + n_ninf_up
    full = _full_tensor_fields(upstream_nonfinite_counts(xb), max_err, err_fro, ref_fro, rel, zero_ref, err, fin)
    if full["numerical_metric_status"] == METRIC_STATUS_COMPARABLE:
        # Pre-D6 finite-element maximum, kept only where the upstream is finite. For a non-finite upstream tensor the
        # finite-subset diagnostic is `finite_subset_max_abs_error` instead (D6: distinct `finite_subset_*` names).
        full["max_abs_error_finite_elements"] = float(np.where(fin, err, 0.0).max()) if err.size else 0.0
    return {
        "shape": list(shape),
        "n_elements": int(x.size),
        "n_blocks": int(nb),
        # D6: bit identity of the decode against the upstream values, every tensor (AMB-030 for Q8_0).
        "n_bit_mismatches": n_bit_mismatches,
        "bit_identical": bit_identical,
        # Full-tensor metrics, `numerical_metric_status`, `upstream_nonfinite`, `null_metric_reasons` (D4, D6).
        # rel_frobenius is None (JSON null) when the reference norm is zero (D4); all are None for a non-finite
        # upstream tensor (D6).
        **full,
        "n_nonfinite_elements": int(np.count_nonzero(~fin)),
        # T2 domain classes (D1 OQ1)
        "n_blocks_in_t2_domain": int(np.count_nonzero(dom)),
        "n_elements_in_t2_domain": int(np.count_nonzero(dom)) * q8_0.QK8_0,
        "n_within_t2": int(np.count_nonzero(within)),
        "n_exceed_t2": int(np.count_nonzero(exceed)),
        # D5 AMB-019: elements of in-domain blocks with a non-finite upstream value, by class; never within or exceeding.
        # n_elements_in_t2_domain = n_within_t2 + n_exceed_t2 + n_nonfinite_upstream_in_t2_domain.
        "nonfinite_upstream_in_t2_domain": {"nan": n_nan_up, "inf": n_pinf_up, "-inf": n_ninf_up},
        "n_nonfinite_upstream_in_t2_domain": n_nan_up + n_pinf_up + n_ninf_up,
        "n_blocks_exceed_t2": int(np.count_nonzero(exceed.any(axis=1))),
        "outside_t2_domain": outside,
        "n_blocks_outside_t2_domain": n_out_blocks,
        "n_elements_outside_t2_domain": n_out_blocks * q8_0.QK8_0,
        "zero_scale_blocks": int(np.count_nonzero(d == 0.0)),
        "nonfinite_scale_blocks": int(np.count_nonzero(~np.isfinite(d))),
        "worst_ratio": worst_ratio,
        "worst_flat_index": worst_idx,
        "block_max_abs_error": _summary(block_max_err),  # every block; a block with a non-finite error counts as inf
        "block_error_over_bound": _summary(block_ratio),  # blocks inside T2's domain only (no ratio elsewhere)
        "error_over_bound_histogram": hist,
    }


def _max_reported(a: np.ndarray) -> float:
    """Maximum of an array of non-negative errors; NaN if any element is NaN, else inf if any is inf."""
    if a.size == 0:
        return 0.0
    if np.isnan(a).any():
        return math.nan
    return float(a.max())


def _norm(a: np.ndarray) -> float:
    """Euclidean norm in float64; inf or NaN if any element is non-finite."""
    if a.size == 0:
        return 0.0
    if np.isnan(a).any():
        return math.nan
    if np.isinf(a).any():
        return math.inf
    return math.sqrt(float(np.sum(a * a)))


def _rel_frobenius(err_fro: float, ref_fro: float) -> tuple[float | None, bool]:
    """Relative Frobenius error and the zero-reference-norm flag (D4). With ||upstream|| = 0 the ratio is undefined:
    it is None (JSON null), never 0, 0/0 or x/0, and err_frobenius is reported alongside. Otherwise the ratio is
    computed (NaN if either norm is NaN or both are infinite)."""
    if ref_fro == 0.0:
        return None, True
    return err_fro / ref_fro, False


def _json_safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def canonical_json(report: dict) -> str:
    """Deterministic text of a report: sorted keys, fixed separators, non-finite floats written as the strings
    "inf", "-inf", "nan" (strict JSON has no such numbers). Equal reports give equal text."""
    return json.dumps(_json_safe(report), sort_keys=True, separators=(",", ":"), allow_nan=False)


def e1_exact_tensor(upstream, decoded) -> dict:
    """E1 for a tensor the container stores unquantised: the decoded float32 values against the upstream values,
    as bit patterns (so +0 and -0 differ), plus (D4) the error statistics every tensor gets: max |x - decoded| and
    the relative Frobenius error, computed in float64 from the values. An element with equal bit patterns has
    error 0 (so a bit-identical tensor reports 0 and 0, even if it holds inf or NaN); a differing element has error
    |x - decoded| (NaN/inf as for Q8_0). Zero reference norm: `rel_frobenius` is None, `zero_reference_norm` true.

    D6: when any upstream element is NaN or +/-inf, the full-tensor metrics are None with the reason
    `nonfinite_upstream` and `numerical_metric_status` is `non_comparable_nonfinite`, whether or not the tensor is
    bit-identical; bit identity is still reported."""
    ua = np.asarray(upstream, dtype=np.float32)
    ub = np.asarray(decoded, dtype=np.float32)
    a = q8_0.f32_bits(ua)
    b = q8_0.f32_bits(ub)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    neq = a != b
    x = ua.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        err = np.where(neq, np.abs(x - ub.astype(np.float64)), 0.0)
    err_fro, ref_fro = _norm(err), _norm(x)
    rel, zero_ref = _rel_frobenius(err_fro, ref_fro)
    if not zero_ref and not neq.any():
        rel = 0.0  # bit-identical with a non-zero reference norm (finite upstream only; D6 nulls the rest)
    fin = np.isfinite(ua) & np.isfinite(ub)
    full = _full_tensor_fields(upstream_nonfinite_counts(ua), _max_reported(err), err_fro, ref_fro, rel, zero_ref,
                               err, fin)
    return {
        "shape": list(a.shape),
        "n_elements": int(a.size),
        "n_bit_mismatches": int(np.count_nonzero(neq)),
        "bit_identical": not bool(neq.any()),
        **full,
    }


def coverage(container: Mapping[str, tuple[int, ...]], upstream: Mapping[str, tuple[int, ...]]) -> dict:
    """AMB-005: do the container's HF keys and shapes match the upstream checkpoint's? Both arguments map
    HF key -> shape. `ok` is true only when nothing is missing, extra or differently shaped."""
    ck, uk = set(container), set(upstream)
    shape_mismatch = {
        k: {"container": list(container[k]), "upstream": list(upstream[k])}
        for k in sorted(ck & uk)
        if tuple(container[k]) != tuple(upstream[k])
    }
    missing, extra = sorted(uk - ck), sorted(ck - uk)
    return {
        "ok": not (missing or extra or shape_mismatch),
        "missing_from_container": missing,
        "extra_in_container": extra,
        "shape_mismatch": shape_mismatch,
    }


def gguf_identical(q8_bytes, rows: int, k: int) -> bool:
    """A1 checker-correctness item: this checker's decode equals gguf-py 0.18.0's, bit for bit (f32 patterns)."""
    import gguf
    import numpy as _np

    raw = _np.frombuffer(q8_bytes, dtype=_np.uint8).reshape(rows, (k // q8_0.QK8_0) * q8_0.BLOCK_BYTES)
    theirs = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q8_0)
    ours = q8_0.dequantize_q8_0_rows(q8_bytes, rows, k)
    a, b = q8_0.f32_bits(theirs), q8_0.f32_bits(ours)
    # NaN payload/sign from 0 x inf is host-dependent (AMB-001): NaN positions only need to agree.
    both_nan = np.isnan(theirs) & np.isnan(ours)
    return bool(np.all((a == b) | both_nan))


_ADDITIVE = (
    "n_elements",
    "n_blocks",
    "n_nonfinite_elements",
    "n_blocks_in_t2_domain",
    "n_elements_in_t2_domain",
    "n_within_t2",
    "n_nonfinite_upstream_in_t2_domain",
    "n_blocks_outside_t2_domain",
    "n_elements_outside_t2_domain",
    "n_exceed_t2",
    "n_blocks_exceed_t2",
    "zero_scale_blocks",
    "nonfinite_scale_blocks",
)


def e1_report(container, upstream: Callable[[str], np.ndarray], upstream_shapes: Mapping[str, tuple[int, ...]],
              crosscheck_gguf: bool = True) -> dict:
    """E1 over a whole container: every tensor of the representation production-q8 executes (section 7), in table order.

    `container` is a `refcheck.container.Container`; `upstream(hf_key)` returns the upstream tensor and
    `upstream_shapes` maps every upstream HF key to its shape (for the coverage check). Per-tensor entries are
    keyed by HF key; `total` sums the additive counts and takes the maxima of the others.
    """
    refs = container.hf_key_map()
    cov = coverage({k: r.shape for k, r in refs.items()}, upstream_shapes)
    tensors: dict[str, dict] = {}
    for key, ref in refs.items():
        if key not in upstream_shapes:
            continue  # reported by coverage()
        if tuple(upstream_shapes[key]) != tuple(ref.shape):
            continue  # reported by coverage(); the values cannot be compared
        up = upstream(key)
        if ref.dtype == "Q8_0":
            raw = container.read_bytes(ref)
            entry = e1_q8_tensor(up, raw, ref.shape)
            entry["dtype"] = "Q8_0"
            if crosscheck_gguf:
                rows = int(np.prod(ref.shape[:-1], dtype=np.int64)) if len(ref.shape) > 1 else 1
                entry["gguf_bit_identical"] = gguf_identical(raw, rows, ref.shape[-1])
        else:
            entry = e1_exact_tensor(up, container.decode(ref))
            entry["dtype"] = ref.dtype
        tensors[key] = entry

    q = [t for t in tensors.values() if t["dtype"] == "Q8_0"]
    total = {k: sum(t[k] for t in q) for k in _ADDITIVE}
    total["outside_t2_domain"] = {
        c: {f: sum(t["outside_t2_domain"][c][f] for t in q) for f in ("blocks", "elements")}
        for c in OUT_OF_DOMAIN_CLASSES
    }
    total["nonfinite_upstream_in_t2_domain"] = {
        c: sum(t["nonfinite_upstream_in_t2_domain"][c] for t in q) for c in NONFINITE_UPSTREAM_CLASSES
    }
    # D6 totals: the upstream non-finite counts over every tensor (Q8_0 and unquantised), and the number of tensors whose
    # full-tensor metrics are not comparable. The Q8_0 maximum error is null if any Q8_0 tensor's is (AMB-030).
    total["upstream_nonfinite"] = {c: sum(t["upstream_nonfinite"][c] for t in tensors.values())
                                   for c in UPSTREAM_NONFINITE_CLASSES}
    total["n_upstream_nonfinite"] = sum(total["upstream_nonfinite"].values())
    total["n_tensors_non_comparable_nonfinite"] = sum(
        t["numerical_metric_status"] == METRIC_STATUS_NONFINITE for t in tensors.values())
    errs = [t["max_abs_error"] for t in q]
    if any(e is None for e in errs):
        total["max_abs_error"] = None
        total["null_metric_reasons"] = {"max_abs_error": NULL_REASON_NONFINITE_UPSTREAM}
    else:
        total["max_abs_error"] = math.nan if any(math.isnan(e) for e in errs) else max(errs, default=0.0)
        total["null_metric_reasons"] = {}
    total["worst_ratio"] = max((t["worst_ratio"] for t in q if t["worst_ratio"] is not None), default=None)
    total["n_q8_tensors"] = len(q)
    total["n_unquantised_tensors"] = len(tensors) - len(q)
    total["unquantised_all_bit_identical"] = all(
        t["bit_identical"] for t in tensors.values() if t["dtype"] != "Q8_0"
    )
    if crosscheck_gguf:
        total["gguf_all_bit_identical"] = all(t["gguf_bit_identical"] for t in q)
    return {"coverage": cov, "tensors": tensors, "total": total}
