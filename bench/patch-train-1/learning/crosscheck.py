"""Random cross-check of the emulator against the frozen independent evaluator (conformance/vlp-numerics/v2/independent/),
and the comparison that the mutation tests prove can fail. The evaluator is run as a separate process on blind inputs."""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EVALUATOR = os.path.join(REPO, "conformance", "vlp-numerics", "v2", "independent", "evaluate.py")
PROFILES = ("f64-seq-rne1/v1", "f64-seq-rne2/v1", "f32-seq-rne1/v1")
MAX_ROWS, MAX_COLS, MAX_RANK = 5, 6, 4
# structural defects that are well-formed documents (the evaluator is defined only for those)
DEFECTS = ("none",) * 6 + ("columns_unsorted", "columns_duplicate", "columns_negative", "columns_empty", "column_out_of_range",
                           "a_length", "b_length", "rank_zero", "nan_operand", "nan_base_inside", "nan_base_outside",
                           "embed", "fixup_inside", "fixup_outside_j", "fixup_unsorted", "fixup_nan", "fixup_random")


def hexw(bits, width):
    return f"{bits:0{width}x}"


def f64_hex(x):
    return hexw(struct.unpack("<Q", struct.pack("<d", x))[0], 16)


def f32_hex(x):
    return hexw(struct.unpack("<I", struct.pack("<f", x))[0], 8)


def bf16_value(word):
    return struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]


def random_bf16_word(rng):
    kind = rng.random()
    if kind < 0.08:
        return rng.choice([0x0000, 0x8000])
    if kind < 0.16:
        return rng.choice([0x0001, 0x0040, 0x007F, 0x8001, 0x807F])               # subnormals
    if kind < 0.22:
        return rng.choice([0x7F7F, 0xFF7F, 0x7F00, 0x0080, 0x8080])               # extremes, smallest normal
    word = rng.getrandbits(16)
    return word if (word & 0x7F80) != 0x7F80 else word & 0x7F7F                  # finite


def operand_value(rng, profile, near=None):
    f32 = profile == "f32-seq-rne1/v1"
    lo, hi = (-149, 127) if f32 else (-1074, 1023)
    kind = rng.random()
    if kind < 0.14:
        return rng.choice([0.0, -0.0])
    if kind < 0.30:
        return rng.choice([1.0, -1.0, 0.5, -0.5, 2.0 ** rng.randint(-12, 12), -(2.0 ** rng.randint(-12, 12))])
    if kind < 0.40:
        e = rng.randint(lo, hi) if rng.random() < 0.3 else rng.choice([lo, lo + 5, hi - 1, hi])
        return rng.choice([1, -1]) * (2.0 ** e if e > -1070 or not f32 else 2.0 ** -140)
    if kind < 0.45 and not f32:
        return rng.choice([1, -1]) * rng.choice([1e300, 1e154, 2.0 ** 1000, 5e-324])
    return rng.uniform(-3.0, 3.0) * (2.0 ** rng.randint(-10, 6))


def to_work(x, profile):
    """A Python float as the working format's bit pattern (rounded for f32; the generator only needs well-formed bits)."""
    if profile == "f32-seq-rne1/v1":
        try:
            return f32_hex(x)
        except OverflowError:
            return f32_hex(3.4028234663852886e38 * (1 if x > 0 else -1))
    return f64_hex(x)


def one_case(rng, idx):
    profile = rng.choice(PROFILES)
    rows, cols, rank = rng.randint(1, MAX_ROWS), rng.randint(1, MAX_COLS), rng.randint(1, MAX_RANK)
    ncols = rng.randint(1, cols)
    columns = sorted(rng.sample(range(cols), ncols))
    base = [random_bf16_word(rng) for _ in range(rows * cols)]
    a = [operand_value(rng, profile) for _ in range(rows * rank)]
    b = [operand_value(rng, profile) for _ in range(rank * ncols)]
    shape_trick = rng.random()
    if shape_trick < 0.18:    # ties: update = +-half an ulp of the base element, optionally split over two terms
        for r in range(rows):
            w = bf16_value(base[r * cols + columns[0]])
            if w and abs(w) < 1e30:
                half_ulp = 2.0 ** (int(struct.unpack("<I", struct.pack("<f", abs(w)))[0] >> 23) - 127 - 8)
                a[r * rank], b[0] = rng.choice([1, -1]) * half_ulp * rng.choice([1, 1 + 2.0 ** -30, 1 - 2.0 ** -30]), 1.0
    elif shape_trick < 0.30:  # near-cancellation: one row, update ~ -(base) (1 + 2^-30)
        w = bf16_value(base[columns[0]])
        if w:
            a[0], b[0] = -w * (1 + 2.0 ** -30), 1.0
    defect = rng.choice(DEFECTS)
    shape, fixups, role = [rows, cols], [], None
    if defect == "columns_unsorted" and ncols >= 2:
        columns = list(reversed(columns))
    elif defect == "columns_duplicate":
        columns = columns + [columns[-1]]
        b = b + [operand_value(rng, profile) for _ in range(rank)]
    elif defect == "columns_negative":
        columns = [-1] + columns
        b = [operand_value(rng, profile) for _ in range(rank)] + b
    elif defect == "columns_empty":
        columns, b = [], []
    elif defect == "column_out_of_range":
        columns = columns + [cols + rng.randint(0, 2)]
        b = b + [operand_value(rng, profile) for _ in range(rank)]
    elif defect == "a_length":
        a = a[:-1] or a + [1.0]
    elif defect == "b_length":
        b = b + [1.0]
    elif defect == "rank_zero":
        rank, a, b = 0, [], []
    elif defect == "nan_operand":
        (a if rng.random() < 0.5 else b)[0] = float("nan") if rng.random() < 0.5 else float("inf")
    elif defect == "nan_base_inside":
        base[rng.randrange(rows) * cols + rng.choice(columns)] = rng.choice([0x7FC0, 0x7F80, 0xFF80])
    elif defect == "nan_base_outside" and ncols < cols:
        outside = [c for c in range(cols) if c not in columns]
        base[rng.randrange(rows) * cols + rng.choice(outside)] = rng.choice([0x7FC0, 0x7F80, 0xFF80])
    elif defect == "embed":
        role = "embed_tokens"
    elif defect == "fixup_inside":
        fixups = [[rng.randrange(rows), rng.choice(columns), hexw(random_bf16_word(rng), 4)]]
    elif defect == "fixup_outside_j" and ncols < cols:
        fixups = [[rng.randrange(rows), rng.choice([c for c in range(cols) if c not in columns]), hexw(random_bf16_word(rng), 4)]]
    elif defect == "fixup_unsorted":
        fixups = [[1, 1, "3f80"], [0, 0, "3f80"]]
    elif defect == "fixup_nan":
        fixups = [[0, columns[0] if columns and columns[0] >= 0 else 0, rng.choice(["7fc0", "7f80", "ff80"])]]
    elif defect == "fixup_random":
        fixups = [[rng.randrange(rows + 1), rng.randrange(cols + 1), hexw(random_bf16_word(rng), 4)]]
    work_width = 8 if profile == "f32-seq-rne1/v1" else 16

    def enc(x):
        if x != x or x in (float("inf"), float("-inf")):   # a non-finite operand, as bits
            return ("7fc00000" if work_width == 8 else "7ff8000000000000") if x != x else \
                   ("7f800000" if work_width == 8 else "7ff0000000000000")
        return to_work(x, profile)
    case = {"id": f"x-{idx:05d}", "base": {"shape": [rows, cols], "bf16": [hexw(w, 4) for w in base]},
            "op": {"kind": "add_low_rank_columns", "numerics": profile, "shape": shape, "rank": rank, "columns": columns,
                   "a": [enc(x) for x in a], "b": [enc(x) for x in b], "fixups": fixups}}
    if role:
        case["tensor_role"] = role
    return case


def random_cases(count, seed):
    rng = random.Random(seed)
    return [one_case(rng, i) for i in range(count)]


def outcome(entry):
    """One comparable form for a result: ('refuse', code) or ('bf16', tuple of ints)."""
    if isinstance(entry, dict):
        return ("refuse", entry["refuse"]) if "refuse" in entry else ("bf16", tuple(int(h, 16) for h in entry["bf16"]))
    kind, val = entry
    return (kind, val if kind == "refuse" else tuple(val))


def differing(left, right):
    """ids on which two result maps differ (a missing id counts as different)."""
    return sorted(i for i in set(left) | set(right) if i not in left or i not in right or outcome(left[i]) != outcome(right[i]))


def run_independent(cases):
    """The frozen independent evaluator, in its own process, on blind inputs."""
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        with open(inp, "w") as f:
            json.dump({"cases": cases}, f)
        subprocess.run([sys.executable, "-I", EVALUATOR, inp, out], check=True, capture_output=True)
        with open(out) as f:
            return json.load(f)


def mutate(entry, flavour):
    kind, val = outcome(entry)
    if kind == "refuse":
        return ("bf16", (0,)) if flavour == "kind" else ("refuse", val + "_x")
    if flavour == "kind":
        return ("refuse", "overflow")
    if flavour == "length":
        return ("bf16", val[:-1])
    words = list(val)
    words[len(words) // 2] ^= 1
    return ("bf16", tuple(words))
