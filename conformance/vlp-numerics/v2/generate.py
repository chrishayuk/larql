#!/usr/bin/env python3
"""VLP numerics v2 golden vectors: `add_low_rank_columns`. DRAFT, not frozen.

Standard library only; imports nothing from LARQL. It reuses v1's exact-rational arithmetic (the one rounder in the
repository) by loading v1/generate.py read-only, and never writes into v1. Every expected value is computed here, none is
typed in. See SPEC.md.

    python3 -I generate.py            write vectors.json
    python3 -I generate.py --check    regenerate and compare byte for byte
"""
import hashlib
import importlib.util
import json
import math
import os
import random
import struct
import sys
from fractions import Fraction

sys.dont_write_bytecode = True  # never leave a __pycache__ in the frozen v1 directory

HERE = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(os.path.dirname(HERE), "v1")
OUT = os.path.join(HERE, "vectors.json")
SCHEMA = "vlp-numerics-golden/v2"
KIND = "add_low_rank_columns"
SEED = 20261112


def _load_v1():
    spec = importlib.util.spec_from_file_location("vlp_numerics_v1_generate", os.path.join(V1_DIR, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V1 = _load_v1()
Refuse, PROFILES, Fraction_ = V1.Refuse, V1.PROFILES, Fraction
R1, R2, F32 = "f64-seq-rne1/v1", "f64-seq-rne2/v1", "f32-seq-rne1/v1"

# the wrong semantics this version adds, for discrimination only
VARIANTS = {
    "zero_padded_full": "v1 add_low_rank on a b zero-padded to every column: the whole tensor is read and written",
    "zero_update_not_kept": "an element whose terms are all zero is still computed, so (-0) + (+0) = +0",
    "low_rank_from_zero": "s = +0 accumulated over k first, then w = add(widen(W), s)",
}


def nonzero(v):
    return v[0] != 0


def written(terms):
    """An element is written iff some term has two nonzero factors (operand level; rounding plays no part)."""
    return any(nonzero(x) and nonzero(y) for x, y in terms)


def run(case, variant=None):
    """The whole op on one case. Returns ("bf16", [bits...]) or ("refuse", code)."""
    variant = variant or {}
    try:
        op = case["op"]
        prof = op.get("numerics")
        if prof not in PROFILES:
            raise Refuse("numerics_invalid")
        return ("bf16", _run(case, op, prof, variant))
    except Refuse as e:
        return ("refuse", e.code)


def _run(case, op, prof, variant):
    work = PROFILES[prof]["work"]
    rows, cols, fixups = V1.validate_common(case, op)
    J = op.get("columns")
    V1.check_columns(J, cols)
    K = op.get("rank")
    if type(K) is not int or K < 1:
        raise Refuse("rank_invalid")
    n = len(J)
    if len(op["a"]) != rows * K or len(op["b"]) != K * n:
        raise Refuse("shape_invalid")
    a = [V1.operand(int(h, 16), work) for h in op["a"]]
    b = [V1.operand(int(h, 16), work) for h in op["b"]]
    w_all = V1.header_phase(case, rows, cols, J)
    inside = set(J)
    V1.check_fixups(fixups, rows, cols, lambda r, c: c in inside)
    if variant.get("zero_padded_full"):
        return _zero_padded_full(case, op, prof, rows, cols, K, J, a, b, fixups)
    V1.check_touched_finite(w_all, [r * cols + j for r in range(rows) for j in J])
    ar = V1.Arith(work, variant.get("ftz", False))
    out = [int(h, 16) for h in case["base"]["bf16"]]
    for r in range(rows):
        for p, j in enumerate(J):
            terms = [(a[r * K + k], b[k * n + p]) for k in range(K)]
            if not written(terms) and not variant.get("zero_update_not_kept"):
                continue  # keeps its exact base bits
            if variant.get("start_from_zero"):
                s = (Fraction(0), False)
                for x, y in terms:
                    s = ar.add(s, ar.mul(x, y))
                w = ar.add(w_all[r * cols + j], s)
            else:
                w = w_all[r * cols + j]
                for x, y in terms:
                    w = ar.add(w, ar.mul(x, y))
            out[r * cols + j] = V1.convert(w, prof, variant)
    computed = list(out)
    V1.apply_fixups(out, fixups, rows, cols, computed)
    return out


def _zero_padded_full(case, op, prof, rows, cols, K, J, a, b, fixups):
    """The wrong semantics an implementer gets by reusing v1 add_low_rank with b padded by zeros outside J."""
    work = PROFILES[prof]["work"]
    zero = "0" * (V1.FMT[work][3] // 4)
    pad = [zero] * (K * cols)
    for k in range(K):
        for p, j in enumerate(J):
            pad[k * cols + j] = op["b"][k * len(J) + p]
    v1 = {"base": case["base"], "op": {"kind": "add_low_rank", "numerics": prof, "shape": [rows, cols], "rank": K,
                                       "a": op["a"], "b": pad, "fixups": op.get("fixups", [])}}
    kind, val = V1.run(v1)
    if kind == "refuse":
        raise Refuse(val)
    return val


# ---------------------------------------------------------------- the independent native route (f64 profiles)

def native_run(case):
    op = case["op"]
    prof = op.get("numerics")
    f64 = lambda h: struct.unpack("<d", int(h, 16).to_bytes(8, "little"))[0]  # noqa: E731
    bf = lambda h: struct.unpack("<f", (int(h, 16) << 16).to_bytes(4, "little"))[0]  # noqa: E731
    rows, cols = op["shape"]
    J, K = op["columns"], op["rank"]
    n = len(J)
    out = [int(h, 16) for h in case["base"]["bf16"]]

    def fin(x):
        if not math.isfinite(x):
            raise Refuse("non_finite_intermediate")
        return x

    def conv(x):
        try:
            if prof == R2:
                x = V1.native_f32(x)
            return V1.native_bf16_from_f64(x)
        except OverflowError:
            raise Refuse("overflow")

    a, b = [f64(h) for h in op["a"]], [f64(h) for h in op["b"]]
    try:
        for r in range(rows):
            for p, j in enumerate(J):
                if not any(a[r * K + k] != 0.0 and b[k * n + p] != 0.0 for k in range(K)):
                    continue
                w = bf(case["base"]["bf16"][r * cols + j])
                for k in range(K):
                    w = fin(w + fin(a[r * K + k] * b[k * n + p]))
                out[r * cols + j] = conv(w)
    except Refuse as e:
        return ("refuse", e.code)
    for r, c, h in op.get("fixups", []):
        out[r * cols + c] = int(h, 16)
    return ("bf16", out)


# ---------------------------------------------------------------- cases

def case(cid, purpose, prof, base, shape, rank, columns, a, b, fixups=(), op_extra=None, **extra):
    work = PROFILES[prof]["work"] if prof in PROFILES else "f64"
    op = {"kind": KIND, "numerics": prof, "shape": list(shape), "rank": rank, "columns": list(columns),
          "a": [x if isinstance(x, str) else V1.wv(x, work) for x in a],
          "b": [x if isinstance(x, str) else V1.wv(x, work) for x in b],
          "fixups": [list(f) for f in fixups]}
    op.update(op_extra or {})
    return {"id": cid, "purpose": purpose, "base": {"shape": list(shape), "bf16": V1.bf16_list(base)}, "op": op, **extra}


def v1_vectors():
    with open(os.path.join(V1_DIR, "vectors.json")) as f:
        return json.load(f)


def equivalence_cases():
    """Every v1 add_low_rank case, re-expressed with J = every column. Same operands, same base."""
    out = []
    for c in v1_vectors()["cases"]:
        op = c["op"]
        if op["kind"] != "add_low_rank":
            continue
        rows, cols = op["shape"] if isinstance(op.get("shape"), list) and len(op["shape"]) == 2 else (0, 0)
        v2 = {"id": "eq-" + c["id"], "purpose": "v1 case " + c["id"] + " with columns = every column: " + c["purpose"],
              "base": {"shape": c["base"]["shape"], "bf16": list(c["base"]["bf16"])},
              "op": {**{k: v for k, v in op.items() if k != "kind"}, "kind": KIND,
                     "columns": list(range(cols)) if cols else [0]},
              "equivalent_to_v1": c["id"]}
        if c.get("tensor_role"):
            v2["tensor_role"] = c["tensor_role"]
        out.append(v2)
    return out


def build_cases():
    rng = random.Random(SEED)
    rb = lambda: V1.random_bf16(rng)  # noqa: E731
    cases = []
    T = cases.append

    def rnd_ops(count, scale=1.0):
        return [rng.uniform(-scale, scale) for _ in range(count)]

    # subset cases
    for cid, prof, shape, K, J in [
        ("sub-k1-single-column", R1, (4, 5), 1, [2]),
        ("sub-k1-first-column", R1, (4, 5), 1, [0]),
        ("sub-k1-last-column", R1, (4, 5), 1, [4]),
        ("sub-k2-two-columns", R1, (4, 6), 2, [1, 4]),
        ("sub-k3-three-columns", R1, (5, 6), 3, [0, 3, 5]),
        ("sub-all-but-one", R1, (3, 5), 2, [0, 1, 3, 4]),
        ("sub-one-row", R1, (1, 6), 2, [1, 2, 5]),
        ("sub-rne2", R2, (4, 6), 2, [1, 4]),
        ("sub-f32-profile", F32, (4, 6), 2, [1, 4]),
    ]:
        rows, cols = shape
        T(case(cid, f"only columns {J} change; K={K}; every other column keeps its exact base bits", prof,
               [rb() for _ in range(rows * cols)], shape, K, J, rnd_ops(rows * K), rnd_ops(K * len(J))))

    # outside J: bits kept exactly, including -0, a subnormal, the largest finite, and non-finite values (not read)
    base = [rb() for _ in range(4 * 6)]
    special = {0: "8000", 5: "0001", 6: "7f7f", 11: "ff7f", 12: "7fc0", 17: "7f80", 18: "ff80", 23: "8000"}
    for i, h in special.items():
        base[i] = h
    T(case("out-keeps-special-values", "outside J: -0, a subnormal, +/-max finite, NaN and +/-inf are kept bit for bit "
           "and are not read (v1 would refuse on the non-finite ones)", R1, base, (4, 6), 2, [1, 3],
           rnd_ops(8), rnd_ops(4), discriminates=["zero_padded_full"]))

    # signed zero inside J
    z = lambda: "8000"  # noqa: E731
    base = [rb() for _ in range(3 * 4)]
    base[1], base[5], base[6] = z(), z(), z()
    T(case("zero-update-keeps-negzero", "a = +0 (a sham): every element of J whose terms are all zero keeps its bits, "
           "so -0 stays -0", R1, base, (3, 4), 1, [1, 2], [0.0, 0.0, 0.0], [1.0, 2.0],
           discriminates=["zero_update_not_kept"]))
    T(case("zero-b-keeps-negzero", "b = +0 with nonzero a: all terms are zero, so the bytes are unchanged", R1, base,
           (3, 4), 1, [1, 2], [1.5, -2.0, 0.5], [0.0, 0.0], discriminates=["zero_update_not_kept"]))
    T(case("zero-update-k2-keeps-negzero", "K = 2 with both terms zero (one zero a, one zero b): unchanged", R1, base,
           (3, 4), 2, [1, 2], [0.0, 1.0, 0.0, -1.0, 0.0, 2.0], [3.0, 4.0, 0.0, 0.0],
           discriminates=["zero_update_not_kept"]))
    T(case("sham-whole-tensor-unchanged", "the sham law: a = 0 over a tensor with -0 elements inside J leaves every "
           "byte of the tensor unchanged", R1, ["8000"] * 12, (3, 4), 1, [0, 1, 2, 3], [0.0, 0.0, 0.0],
           [1.0, -1.0, 2.0, 0.5], discriminates=["zero_update_not_kept"]))
    # a rank-1 edit with a zero in a or in b leaves that row's / column's elements untouched (SG-EDIT-1's request)
    base = [rb() for _ in range(3 * 4)]
    base[4], base[6] = "8000", "8000"     # row 1, columns 0 and 2
    T(case("k1-zero-row-of-a-untouched", "K = 1, a[1] = 0: every element of row 1 inside J keeps its bits, -0 included; "
           "rows 0 and 2 are written", R1, base, (3, 4), 1, [0, 2], [0.5, 0.0, -0.25], [1.0, 2.0],
           discriminates=["zero_update_not_kept"]))
    base = [rb() for _ in range(3 * 4)]
    base[1], base[9] = "8000", "8000"     # column 1 in rows 0 and 2
    T(case("k1-zero-entry-of-b-untouched", "K = 1, b[0] = 0 (column J[0] = 1): every element of that column keeps its "
           "bits, -0 included; column 3 is written", R1, base, (3, 4), 1, [1, 3], [0.5, 1.0, -0.25], [0.0, 2.0],
           discriminates=["zero_update_not_kept"]))
    T(case("mixed-terms-write", "K = 2: term 0 is zero and term 1 is not, so the element is written; the zero term "
           "adds +0 and the result is the ordinary one", R1, base, (3, 4), 2, [1, 2], [0.0, 0.5, 0.0, 0.25, 0.0, -0.5],
           [9.0, 9.0, 2.0, -1.0]))
    T(case("nonzero-update-underflows-positive", "from -0: a nonzero update 2^-200 is written; w = -0 + 2^-200 rounds to "
           "+0 in BF16 (the sign of the value)", R1, ["8000"], (1, 1), 1, [0], [2.0 ** -100], [2.0 ** -100]))
    T(case("nonzero-update-underflows-negative", "from -0: a nonzero update -2^-200 is written and rounds to -0",
           R1, ["8000"], (1, 1), 1, [0], [-(2.0 ** -100)], [2.0 ** -100]))
    T(case("nonzero-operands-underflowing-product-f32", "f32 profile: the product of two nonzero operands underflows "
           "to +0, so the element is written (written is decided on operands, not on rounded products): -0 becomes +0",
           F32, ["8000"], (1, 1), 1, [0], [2.0 ** -100], [2.0 ** -100]))
    T(case("exact-cancellation-from-negzero", "terms +x and -x are nonzero operands, so the element is written: "
           "-0 + x = x, x + (-x) = +0, and the result is +0", R1, ["8000"], (1, 1), 2, [0], [0.5, -0.5], [1.0, 1.0]))
    T(case("poszero-and-zero-update", "+0 with a zero update stays +0", R1, ["0000"], (1, 1), 1, [0], [0.0], [1.0]))

    # k order (inherited from v1 by equivalence; one direct case where starting from the widened base matters)
    T(case("k-order-start-from-base", "K = 2: accumulation starts from widen(W) and rounds each step: 1 + 2^53 ties to 2^53, "
           "then + -(2^53 - 1) = 1, so BF16 1.0; starting from +0 gives 2.0", R1, ["3f80"], (1, 1), 2, [0],
           [2.0 ** 53, -(2.0 ** 53 - 1)], [1.0, 1.0], discriminates=["low_rank_from_zero"]))


    # BF16 rounding boundaries inside J (K = 1): ties to even in both directions and just either side of a tie
    for cid, base_h, upd in [
        ("tie-to-even-down", "3f80", 2.0 ** -8),            # 1 + 2^-8 is halfway between 3f80 and 3f81: even is 3f80
        ("tie-to-even-up", "3f81", 2.0 ** -8),              # halfway between 3f81 and 3f82: even is 3f82
        ("tie-negative", "bf80", -(2.0 ** -8)),
        ("just-above-tie", "3f80", 2.0 ** -8 + 2.0 ** -30),
        ("just-below-tie", "3f81", 2.0 ** -8 - 2.0 ** -30),
    ]:
        T(case(cid, f"BF16 rounding boundary inside J: base {base_h}, update {upd!r}", R1, [base_h, "3f80"], (1, 2), 1,
               [0], [upd], [1.0]))
    T(case("tie-double-rounding-rne1", "a value that is a BF16 tie only after the f32 step differs between the chains",
           R1, ["3f80"], (1, 1), 1, [0], [2.0 ** -8 + 2.0 ** -33], [1.0], discriminates=[]))
    T(case("tie-double-rounding-rne2", "the same operands under rne2", R2, ["3f80"], (1, 1), 1, [0],
           [2.0 ** -8 + 2.0 ** -33], [1.0]))

    # near-cancellation inside J: the tiny nonzero result no f32 delta can write (REAL-2's construction)
    T(case("near-cancellation", "w + a*b leaves a tiny nonzero: 1.0 + (-(1 + 2^-30)) * 1 = -2^-30", R1, ["3f80", "3f80"],
           (1, 2), 1, [0], [-(1.0 + 2.0 ** -30)], [1.0]))
    T(case("near-cancellation-rne2", "the same under rne2: 2^-30 is representable in f32", R2, ["3f80"], (1, 1), 1, [0],
           [-(1.0 + 2.0 ** -30)], [1.0]))
    T(case("exact-cancellation", "w + a*b = 0 exactly from a positive base: +0", R1, ["3f80"], (1, 1), 1, [0], [-1.0],
           [1.0]))

    # refusals, one defect each
    mk = lambda **kw: dict(prof=R1, base=[rb() for _ in range(2 * 4)], shape=(2, 4), K=1, J=[1, 2],  # noqa: E731
                           a=[1.0, 2.0], b=[0.5, 0.25], **kw)

    def refuse(cid, purpose, **over):
        p = mk()
        p.update(over)
        T(case(cid, purpose, p["prof"], p["base"], p["shape"], p["K"], p["J"], p["a"], p["b"],
               fixups=p.get("fixups", ()), op_extra=p.get("op_extra"), **p.get("extra", {})))

    refuse("ref-columns-empty", "columns empty", J=[], b=[])
    refuse("ref-columns-unsorted", "columns not increasing", J=[2, 1])
    refuse("ref-columns-duplicate", "columns repeat a column", J=[1, 1])
    refuse("ref-column-out-of-range", "a column >= cols", J=[1, 4])
    refuse("ref-b-full-width", "b has K x cols values although |J| < cols: v1's b layout is not accepted here",
           b=[0.5, 0.25, 0.125, 0.0625])
    refuse("ref-a-length", "len(a) != rows x K", a=[1.0])
    refuse("ref-rank-zero", "rank < 1", K=0, a=[], b=[])
    refuse("ref-numerics-missing", "numerics absent", prof=None)
    refuse("ref-fixup-outside-j", "a fixup in a column outside J (inside the tensor)", fixups=[(0, 0, "3f80")])
    refuse("ref-fixup-outside-tensor", "a fixup outside the tensor", fixups=[(5, 1, "3f80")])
    refuse("ref-tensor-embed", "the token embedding is refused", extra={"tensor_role": "embed_tokens"})
    b2 = [rb() for _ in range(8)]
    b2[1] = "7fc0"
    refuse("ref-nan-base-inside-j", "a NaN base element inside J is read", base=b2)
    b3 = [rb() for _ in range(8)]
    b3[0] = "7fc0"
    T(case("out-nan-base-outside-j-is-kept", "a NaN base element outside J is not read: it is kept", R1, b3, (2, 4), 1,
           [1, 2], [1.0, 2.0], [0.5, 0.25]))
    refuse("ref-nan-operand-b", "a non-finite b", b=["7ff8000000000000", 0.25])
    refuse("ref-overflow", "the result overflows BF16", base=["7f7f"] * 8, a=[1.0, 1.0], b=[2.0 ** 126, 2.0 ** 126])
    refuse("ref-intermediate-overflow", "an f64 intermediate overflows", a=[1e300, 1.0], b=[1e300, 1.0])
    T(case("ref-fixup-no-op", "a fixup equal to the computed bits", R1, [rb() for _ in range(8)], (2, 4), 1, [1, 2],
           [1.0, 2.0], [0.5, 0.25], fixups=[(0, 1, "0000")]))

    # fixups that apply
    base = [rb() for _ in range(2 * 4)]
    T(case("fixup-inside-j", "a fixup replaces one computed element of J", R1, base, (2, 4), 1, [1, 2], [1.0, 2.0],
           [0.5, 0.25], fixups=[(1, 2, "3f80")]))
    return cases


def sha_bf16(hexes):
    return hashlib.sha256(b"".join(int(h, 16).to_bytes(2, "little") for h in hexes)).hexdigest()


def finish(cases):
    ids, v1 = set(), {c["id"]: c for c in v1_vectors()["cases"]}
    for c in cases:
        assert c["id"] not in ids, c["id"]
        ids.add(c["id"])
        kind, val = run(c)
        c["base"]["sha256"] = sha_bf16(c["base"]["bf16"])
        c["expect"] = ({"bf16": [V1.hx(x, "bf16") for x in val], "sha256": sha_bf16([V1.hx(x, "bf16") for x in val])}
                       if kind == "bf16" else {"refuse": val})
        alts = {}
        for name in VARIANTS:
            var = {"zero_padded_full": {"zero_padded_full": True}, "zero_update_not_kept": {"zero_update_not_kept": True},
                   "low_rank_from_zero": {"start_from_zero": True}}[name]
            k2, v2 = run(c, var)
            r = {"bf16": [V1.hx(x, "bf16") for x in v2]} if k2 == "bf16" else {"refuse": v2}
            if r != {k: v for k, v in c["expect"].items() if k != "sha256"}:
                alts[name] = r
        for d in c.get("discriminates", []):
            assert d in alts, f"{c['id']}: variant {d} does not change the result"
        c["under_wrong_semantics"] = alts
        c.setdefault("discriminates", [])
        if "equivalent_to_v1" in c:  # the equivalence law, made explicit and checked
            old = v1[c["equivalent_to_v1"]]["expect"]
            same = {k: v for k, v in c["expect"].items() if k != "sha256"} == {k: v for k, v in old.items() if k != "sha256"}
            c["same_as_v1"] = same
            if not same:
                base_bits, new = c["base"]["bf16"], c["expect"].get("bf16")
                diff = [i for i in range(len(base_bits)) if new is None or new[i] != old.get("bf16", [None] * len(base_bits))[i]]
                assert new is not None and old.get("bf16") and all(base_bits[i] == "8000" and new[i] == "8000" for i in diff), \
                    f"{c['id']}: differs from v1 other than by a kept -0: {diff}"
                c["differs_from_v1_because"] = "elements whose terms are all zero keep a base -0 (v1 writes +0)"
    return cases


def cross_check(cases):
    n = 0
    for c in cases:
        op = c["op"]
        if op.get("numerics") not in (R1, R2):
            continue
        exp = c["expect"]
        if "refuse" in exp and exp["refuse"] not in ("overflow", "non_finite_intermediate"):
            continue
        if c.get("tensor_role"):
            continue
        got = native_run(c)
        want = ("refuse", exp["refuse"]) if "refuse" in exp else ("bf16", [int(h, 16) for h in exp["bf16"]])
        assert got == want, f"{c['id']}: native route {got} != exact route {want}"
        n += 1
    return n


def document():
    cases = finish(equivalence_cases() + build_cases())
    return {"schema": SCHEMA, "spec": "SPEC.md", "generator": "generate.py", "extends": "vlp-numerics/v1 (unchanged)",
            "profiles": PROFILES, "refusals": V1.REFUSALS, "variants": VARIANTS,
            "native_cross_checked_cases": cross_check(cases), "cases": cases}


def dumps(doc):
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


if __name__ == "__main__":
    text = dumps(document())
    if "--check" in sys.argv[1:]:
        with open(OUT) as f:
            if f.read() != text:
                raise SystemExit("vectors.json differs from what generate.py produces")
        print("vectors.json reproduces;", json.loads(text)["native_cross_checked_cases"], "cases cross-checked")
    else:
        with open(OUT, "w") as f:
            f.write(text)
        print("wrote", OUT, hashlib.sha256(text.encode()).hexdigest())
