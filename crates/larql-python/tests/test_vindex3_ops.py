"""
The typed VINDEX3 operations on a Session, held to the LQL statements they share a Rust operation with.

Each test runs a Python method and the matching LQL statement on the same session state and compares them:
arrays bit for bit against the .npy the statement wrote, provenance against its sidecar, and structured
results against what the statement prints. The container is the binding's synthetic test container
(larql._native._write_test_container): two BF16 layers, hidden 64, and a word-level tokenizer.

Run: pytest crates/larql-python/tests/test_vindex3_ops.py -v
"""

import hashlib
import json
import os
import re
import shutil
import tempfile

import numpy as np
import pytest

import larql
from larql import _native

PROMPT = "The capital of France is"
CONTINUATION = " Paris"
TENSOR = "1.mlp.down_proj.weight"
F64_PROFILE = "f64-seq-rne1/v1"
F32_PROFILE = "f32-seq-rne1/v1"
COLUMNS = [8, 9, 33]
SILENT_ROW = 2


@pytest.fixture(scope="module")
def container():
    work = tempfile.mkdtemp(prefix="larql_v3_ops_")
    path = _native._write_test_container(work)
    yield path
    shutil.rmtree(work, ignore_errors=True)


@pytest.fixture
def s(container):
    return larql.session(container)


def q(path):
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def load_npy(path):
    with open(f"{path}.json") as f:
        sidecar = json.load(f)
    return np.load(path), sidecar


def same_bits(a, b):
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    width = {2: np.uint16, 4: np.uint32}[a.dtype.itemsize]
    return np.array_equal(a.view(width), b.view(width))


def assert_matches_sidecar(readout, sidecar):
    """The provenance is the sidecar, except the statement that wrote the file."""
    provenance = dict(readout.provenance)
    assert provenance.pop("statement") is None
    written = dict(sidecar)
    written.pop("statement")
    assert provenance == written


def a_factor(rows, scale, dtype=np.float64):
    a = np.full((rows, 1), scale, dtype=dtype)
    a[SILENT_ROW, 0] = 0.0
    return a


def tensor_rows(s, name=TENSOR):
    return next(t for t in s.tensors() if t["tensor"] == name and t["canonical"])["shape"][0]


# ─── SHOW TENSORS ───


class TestTensors:
    @pytest.mark.parametrize("filter", [None, "decoder_stack"])
    def test_tensors_equal_show_tensors(self, s, filter):
        listed = s.tensors(filter)
        stmt = "SHOW TENSORS" if filter is None else f'SHOW TENSORS "{filter}"'
        lines = s.query(stmt)
        rows = lines[2:-1]
        assert len(rows) == len(listed) > 0
        for line, t in zip(rows, listed):
            cells = line.split()
            assert cells[0] == t["name"]
            assert cells[1] == t["dtype"]
            assert str(t["shape"]) in line
            assert line.rstrip().endswith(f"  {t['bare']}")
        assert lines[-1].startswith(f"{len(listed)} tensor(s)")

    def test_a_listing_names_its_parts(self, s):
        t = next(t for t in s.tensors() if t["tensor"] == TENSOR and t["canonical"])
        assert t["name"].endswith(f"::{TENSOR}")
        assert t["dtype"] == "BF16"
        assert len(t["shape"]) == 2


# ─── EXPORT WEIGHT ───


class TestTensor:
    @pytest.mark.parametrize(
        "rows,cols,bits",
        [(None, None, False), ([1, 3], [0, 5, 7], False), ([4], None, True), (None, [2], True)],
    )
    def test_tensor_equals_export_weight(self, s, tmp_path, rows, cols, bits):
        out = tmp_path / "w.npy"
        stmt = f'EXPORT WEIGHT "{TENSOR}"'
        if rows is not None:
            stmt += f" ROWS ({', '.join(map(str, rows))})"
        if cols is not None:
            stmt += f" COLUMNS ({', '.join(map(str, cols))})"
        stmt += f' TO "{q(out)}"' + (" AS BITS" if bits else "")
        s.query(stmt)
        written, sidecar = load_npy(out)

        readout = s.tensor(TENSOR, rows=rows, cols=cols, bits=bits)
        assert isinstance(readout.array, np.ndarray)
        assert readout.array.dtype == (np.uint16 if bits else np.float32)
        assert same_bits(readout.array, written)
        assert_matches_sidecar(readout, sidecar)

    def test_the_array_is_an_owned_copy(self, s):
        readout = s.tensor(TENSOR, rows=[0])
        array = readout.array
        assert array.flags["OWNDATA"] or array.base is not None
        array[...] = 0
        assert not np.all(s.tensor(TENSOR, rows=[0]).array == 0)

    def test_an_unknown_tensor_is_refused(self, s):
        with pytest.raises(RuntimeError, match="EXPORT WEIGHT refused"):
            s.tensor("no.such.tensor")


# ─── Tokens ───


class TestTokens:
    def test_tokenize_is_the_prompt_encoding_capture_records(self, s, tmp_path):
        out = tmp_path / "c.npy"
        s.query(f'CAPTURE logits FOR "{PROMPT}" CONTINUE "{CONTINUATION}" TO "{q(out)}"')
        _, sidecar = load_npy(out)
        sequence = sidecar["sequence"]
        prompt_ids = s.tokenize(PROMPT)
        assert prompt_ids == sequence["token_ids"][: sequence["prompt_tokens"]]
        assert s.tokenize(CONTINUATION, raw=True) == sequence["continuation"]["token_ids"]

    def test_decode_is_the_token_capture_records(self, s, tmp_path):
        out = tmp_path / "c.npy"
        s.query(f'CAPTURE logits FOR "{PROMPT}" POSITIONS ALL TO "{q(out)}"')
        _, sidecar = load_npy(out)
        for position in sidecar["positions"]:
            assert s.decode([position["token_id"]]) == position["token"]
        assert s.decode(s.tokenize(PROMPT, raw=True)) == PROMPT


# ─── CAPTURE ───

CAPTURES = [
    ("logits", {}, ""),
    ("final", {"positions": "ALL"}, "POSITIONS ALL"),
    ("residual", {"layer": 0, "positions": "PROMPT_END"}, "AT LAYER 0 POSITIONS PROMPT_END"),
    (
        "post_attention",
        {"layers": [1, 0], "continue_text": CONTINUATION, "positions": "REPLY"},
        f'AT LAYERS (1, 0) CONTINUE "{CONTINUATION}" POSITIONS REPLY',
    ),
    (
        "ffn.input",
        {"layer": 1, "continue_tokens": [3, 7], "positions": [0, 6]},
        "AT LAYER 1 CONTINUE TOKENS (3, 7) POSITIONS (0, 6)",
    ),
    (
        "ffn.down_output",
        {"layers": [0], "generate": 2, "positions": "all"},
        "AT LAYERS (0) GENERATE 2 POSITIONS ALL",
    ),
    ("ffn.down_input", {"layer": 1}, "AT LAYER 1"),
]


class TestCapture:
    @pytest.mark.parametrize("site,kwargs,clauses", CAPTURES, ids=[c[0] for c in CAPTURES])
    def test_capture_equals_the_statement(self, s, tmp_path, site, kwargs, clauses):
        out = tmp_path / "c.npy"
        at, sequence = split_clauses(clauses)
        s.query(f'CAPTURE {site} {at} FOR "{PROMPT}" {sequence} TO "{q(out)}"')
        written, sidecar = load_npy(out)

        readout = s.capture(site, PROMPT, **kwargs)
        assert readout.array.dtype == np.float32
        assert same_bits(readout.array, written)
        assert_matches_sidecar(readout, sidecar)

    def test_logits_last_are_the_logits_infer_ranks(self, s):
        logits = s.capture("logits", PROMPT).array[0]
        top = s.infer(PROMPT, top=3)["predictions"]
        assert [p["token_id"] for p in top] == list(np.argsort(-logits, kind="stable")[:3])

    def test_bad_arguments_are_refused_before_running(self, s):
        with pytest.raises(ValueError, match="unknown site"):
            s.capture("nowhere", PROMPT)
        with pytest.raises(ValueError, match="not both"):
            s.capture("residual", PROMPT, layer=0, layers=[0])
        with pytest.raises(ValueError, match="at most one"):
            s.capture("logits", PROMPT, continue_text="x", generate=1)
        with pytest.raises(ValueError, match="positions"):
            s.capture("logits", PROMPT, positions="FIRST")
        with pytest.raises(RuntimeError, match="CAPTURE refused"):
            s.capture("residual", PROMPT)


def split_clauses(clauses):
    """`AT LAYER…` goes before FOR; continuation and POSITIONS after it."""
    if not clauses.startswith("AT"):
        return "", clauses
    words = clauses.split(" ")
    for i, word in enumerate(words):
        if word in ("CONTINUE", "GENERATE", "POSITIONS"):
            return " ".join(words[:i]), " ".join(words[i:])
    return clauses, ""


# ─── INFER ───


class TestInfer:
    def test_infer_equals_infer_top(self, s):
        result = s.infer(PROMPT, top=4)
        lines = s.query(f'INFER "{PROMPT}" TOP 4')
        printed = [re.match(r"\s*\d+\. (.*?)\s+\(([\d.]+)%\)\s+\[id (\d+)\]", l) for l in lines[1:5]]
        assert result["prompt_ids"] == s.tokenize(PROMPT)
        assert result["knn_override"] is None
        assert len(result["predictions"]) == 4
        for p, m in zip(result["predictions"], printed):
            assert m is not None
            assert p["token"] == m.group(1)
            assert f"{p['probability'] * 100:.2f}" == m.group(2)
            assert p["token_id"] == int(m.group(3))

    def test_generate_equals_infer_generate(self, s):
        result = s.generate(PROMPT, 3)
        lines = s.query(f'INFER "{PROMPT}" GENERATE 3')
        ids = next(l for l in lines if l.strip().startswith("ids:")).split("ids:")[1].strip()
        text = next(l for l in lines if l.strip().startswith("text:")).split("text: ", 1)[1]
        assert ",".join(map(str, result["token_ids"])) == ids
        assert result["text"] == text
        assert result["prompt_ids"] == s.tokenize(PROMPT)


# ─── Weight edits and patches ───


def operand_file(path, a, b, origin):
    hexes = lambda x: [format(int(v), "016x") for v in np.ascontiguousarray(x, dtype="<f8").view("<u8").ravel()]
    doc = {"rank": int(a.shape[1]), "a": hexes(a), "b": hexes(b), "origin": origin}
    with open(path, "w") as f:
        json.dump(doc, f)


class TestLowRank:
    def test_numpy_operands_save_the_same_vlp_as_the_operand_file(self, container, tmp_path):
        s = larql.session(container)
        a = a_factor(tensor_rows(s), 0.125)
        b = np.ones((1, len(COLUMNS)))
        from_arrays = tmp_path / "arrays.vlp"
        s.begin_patch(str(from_arrays))
        staged = s.add_low_rank(TENSOR, a, b, COLUMNS, numerics=F64_PROFILE)
        saved = s.save_patch()
        assert saved["path"] == str(from_arrays)
        assert saved["weight_edits"] == 1

        a_sha = hashlib.sha256(a.astype("<f8").tobytes()).hexdigest()
        b_sha = hashlib.sha256(b.astype("<f8").tobytes()).hexdigest()
        assert staged["origin"] == (
            f"python arrays: a {a.shape[0]}x1 f64 sha256 {a_sha}; b 1x{len(COLUMNS)} f64 sha256 {b_sha}"
        )

        t = larql.session(container)
        operands = tmp_path / "operands.json"
        operand_file(operands, a, b, staged["origin"])
        from_file = tmp_path / "file.vlp"
        t.query(f'BEGIN PATCH "{q(from_file)}"')
        lines = t.query(
            f'ADD LOW RANK "{q(operands)}" TO "{TENSOR}" COLUMNS ({", ".join(map(str, COLUMNS))}) '
            f'NUMERICS "{F64_PROFILE}"'
        )
        printed_save = t.query("SAVE PATCH")
        assert f"op digest {staged['digest'][:16]}" in "\n".join(lines)
        assert f"{staged['changed_elements']} element(s) would change" in lines[0]
        assert f"{staged['pending_structured_ops']} structured op(s) pending" in "\n".join(lines)
        assert printed_save[0].startswith(f"Saved: {from_file} (")
        assert f"{saved['weight_edits']} weight edit(s)" in printed_save[0]
        assert from_arrays.read_bytes() == from_file.read_bytes()

    def test_f32_operands_under_the_f32_profile(self, s):
        a = a_factor(tensor_rows(s), 0.125, np.float32)
        b = np.ones((1, len(COLUMNS)), dtype=np.float32)
        staged = s.add_low_rank(TENSOR, a, b, COLUMNS, numerics=F32_PROFILE)
        assert " f32 sha256 " in staged["origin"]
        assert staged["changed_elements"] > 0

    def test_narrowing_and_misshapen_operands_are_refused(self, s):
        rows = tensor_rows(s)
        with pytest.raises(RuntimeError, match="narrowing"):
            s.add_low_rank(TENSOR, a_factor(rows, 0.5), np.ones((1, 3)), COLUMNS, numerics=F32_PROFILE)
        with pytest.raises(ValueError, match="must be 1x3"):
            s.add_low_rank(TENSOR, a_factor(rows, 0.5), np.ones((2, 3)), COLUMNS, numerics=F64_PROFILE)
        with pytest.raises(TypeError, match="float64 or float32"):
            s.add_low_rank(TENSOR, np.ones((rows, 1), dtype=np.int32), np.ones((1, 3)), COLUMNS, F64_PROFILE)


class TestUpdateWeights:
    def test_update_weights_equals_the_statement(self, container):
        s = larql.session(container)
        t = larql.session(container)
        cols = [0, 1, 2]
        staged = s.update_weights(TENSOR, 5, cols, 2.0)
        lines = t.query(
            f'UPDATE WEIGHTS SET DELTA = 2.0 WHERE TENSOR = "{TENSOR}" AND ROW = 5 AND COL IN (0, 1, 2)'
        )
        bf16 = lambda bits: np.array([bits << 16], dtype=np.uint32).view(np.float32)[0]
        for e, line in zip(staged["elements"], lines[1:4]):
            assert line.strip() == f"[5, {e['col']}]: {bf16(e['old_bits']):.7f} -> {bf16(e['new_bits']):.7f}"
        assert f"  op digest {staged['digest'][:16]}" in lines
        assert f"  {staged['pending_elements']} weight element(s) pending in this session" in lines


class TestPatchLifecycle:
    def test_apply_changes_capture_and_remove_restores_it_bit_for_bit(self, container, tmp_path):
        s = larql.session(container)
        base = s.capture("logits", PROMPT, positions="ALL").array

        vlp = tmp_path / "edit.vlp"
        with larql.recording(s, str(vlp)) as rec:
            s.add_low_rank(TENSOR, a_factor(tensor_rows(s), 4.0), np.ones((1, 3)), COLUMNS, F64_PROFILE)
        assert rec["saved"]["path"] == str(vlp)

        fresh = larql.session(container)
        assert same_bits(fresh.capture("logits", PROMPT, positions="ALL").array, base)
        applied = fresh.apply_patch(str(vlp))
        assert applied["sha256"] == hashlib.sha256(vlp.read_bytes()).hexdigest()
        assert applied["structured_ops"] == 1
        edited = fresh.capture("logits", PROMPT, positions="ALL")
        assert not same_bits(edited.array, base)
        assert edited.provenance["executes"] == "patched"
        assert edited.provenance["patches"][0]["sha256"] == applied["sha256"]

        listed = fresh.patches()
        assert [p["path"] for p in listed["applied"]] == [str(vlp)]
        assert listed["applied"][0]["sha256"] == applied["sha256"]
        assert listed["weight_ops"] == 1
        assert listed["recording"] is None

        removed = fresh.remove_patch(str(vlp))
        assert removed == {"number": 1, "carried_weight_ops": True}
        restored = fresh.capture("logits", PROMPT, positions="ALL")
        assert same_bits(restored.array, base)
        assert restored.provenance["executes"] == "stored"
        assert fresh.patches()["applied"] == []

    def test_patch_results_equal_the_statements(self, container, tmp_path):
        vlp = tmp_path / "edit.vlp"
        s = larql.session(container)
        s.begin_patch(str(vlp))
        s.update_weights(TENSOR, 5, [0, 1], 64.0)
        assert s.patches()["recording"] == {"path": str(vlp), "operations": 1}
        assert f"  Recording: {vlp} (1 ops pending)" in s.query("SHOW PATCHES")
        s.save_patch()

        py, lql = larql.session(container), larql.session(container)
        applied = py.apply_patch(str(vlp))
        printed = lql.query(f'APPLY PATCH "{q(vlp)}"')[0]
        assert printed == (
            f"Applied: {vlp} ({applied['operations']} operations: {applied['inserts']} inserts, "
            f"{applied['updates']} updates, {applied['deletes']} deletes; {applied['weight_edits']} "
            "stored-weight edit(s), bound to this container and pending COMPILE)"
        )
        listed = py.patches()
        shown = lql.query("SHOW PATCHES")
        p = listed["applied"][0]
        assert shown[0] == f"  1. {p['name']:<40} {p['operations']} ops ({p['inserts']} ins, {p['updates']} upd, {p['deletes']} del)"
        assert (
            f"  Stored-weight edits: {listed['weight_elements']} element(s) in {listed['weight_ops']} op(s), "
            "pending COMPILE"
        ) in shown
        removed = py.remove_patch(str(vlp))
        assert lql.query(f'REMOVE PATCH "{q(vlp)}"')[0] == f"Removed patch #{removed['number']}"

    def test_missing_recording_and_patch_are_refused(self, s):
        with pytest.raises(RuntimeError, match="no active patch session"):
            s.save_patch()
        with pytest.raises(RuntimeError, match="patch not found"):
            s.remove_patch("absent.vlp")
        with pytest.raises(RuntimeError, match="patch not found"):
            s.apply_patch("absent.vlp")


class TestDiscardWeightEdits:
    def test_discard_equals_the_statement(self, container):
        s = larql.session(container)
        t = larql.session(container)
        for session in (s, t):
            session.update_weights(TENSOR, 5, [0, 1], 2.0)
        discarded = s.discard_weight_edits()
        printed = t.query("DISCARD WEIGHT EDITS")
        assert discarded == {"discarded": 1, "unrecorded": 0, "from_patches": 0}
        assert printed[0] == f"Discarded: {discarded['discarded']} stored-weight op(s) staged in this session"
        assert s.patches()["weight_ops"] == 0
        assert s.discard_weight_edits()["discarded"] == 0
