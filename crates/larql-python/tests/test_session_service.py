"""
`larql.connect`: the typed session operations through a `larql serve-session` process, held to the same
operations run in-process (`larql.session`) on the same container: arrays bit for bit, provenance and
structured results value for value. Also the service's refusals as the client sees them: operations outside
the policy, paths outside the workdir, the budget, the limits, and a second client.

The service is the `larql` binary: $LARQL_BIN, or target/release/larql or target/debug/larql in this checkout
(the newer one). Build it with `cargo build -p larql-cli`. The tests are skipped only when no binary exists.

Run: pytest crates/larql-python/tests/test_session_service.py -v
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

import larql
from larql import _native
from larql.session_client import Readout, ServiceError

PROMPT = "The capital of France is"
CONTINUATION = " Paris"
TENSOR = "1.mlp.down_proj.weight"
F64_PROFILE = "f64-seq-rne1/v1"
COLUMNS = [8, 9, 33]
ALL_OPS = [
    "tensors", "tensor", "tokenize", "decode", "capture", "infer", "generate", "begin_patch", "add_low_rank",
    "update_weights", "save_patch", "apply_patch", "remove_patch", "patches", "discard_weight_edits",
]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STARTUP_SECONDS = 60


def larql_binary():
    if os.environ.get("LARQL_BIN"):
        return os.environ["LARQL_BIN"]
    found = [p for p in (os.path.join(REPO, "target", d, "larql") for d in ("release", "debug")) if os.path.exists(p)]
    return max(found, key=os.path.getmtime) if found else None


BINARY = larql_binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="no larql binary; run `cargo build -p larql-cli`")


@pytest.fixture(scope="module")
def container():
    work = tempfile.mkdtemp(prefix="lss_model_")
    path = _native._write_test_container(work)
    yield path
    shutil.rmtree(work, ignore_errors=True)


class Service:
    """A running service, its workdir, its audit log, and a client of it."""

    def __init__(self, container, budget=5, operations=ALL_OPS, limits=None, resume=False, root=None):
        self.root = root or tempfile.mkdtemp(prefix="lss_")
        self.workdir = os.path.join(self.root, "work")
        os.makedirs(self.workdir, exist_ok=True)
        self.log = os.path.join(self.root, "audit.jsonl")
        self.socket = os.path.join(self.root, "s.sock")
        self.staging = os.path.join(self.root, "staging")
        os.makedirs(self.staging, exist_ok=True)
        policy = {
            "policy": "larql-session-policy/v1", "container": container, "workdir": self.workdir,
            "audit_log": self.log, "operations": operations, "budget": {"edited_executions": budget},
            "resume": resume, "staging_dir": self.staging,
        }
        if limits:
            policy["limits"] = limits
        self.policy = os.path.join(self.root, "policy.json")
        with open(self.policy, "w") as f:
            json.dump(policy, f)
        self.proc = subprocess.Popen(
            [BINARY, "serve-session", "--config", self.policy, "--socket", self.socket],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        deadline = time.time() + STARTUP_SECONDS
        while not os.path.exists(self.socket):
            if self.proc.poll() is not None:
                raise RuntimeError(f"the service exited: {self.proc.stderr.read().decode()}")
            assert time.time() < deadline, "the service did not start"
            time.sleep(0.05)
        self.client = larql.connect(self.socket)

    def records(self):
        with open(self.log) as f:
            return [json.loads(line) for line in f]

    def stop(self):
        self.client.close()
        self.proc.send_signal(signal.SIGTERM)
        self.proc.wait(timeout=30)
        self.proc.stderr.close()

    def path(self, name):
        return os.path.join(self.workdir, name)


@pytest.fixture
def svc(container):
    service = Service(container)
    yield service
    if service.proc.poll() is None:
        service.stop()
    shutil.rmtree(service.root, ignore_errors=True)


CONTAINER_LABEL = "<bound container>"


def redacted(provenance):
    """The provenance as the service sends it: the container's path replaced by its label."""
    out = dict(provenance)
    if "container" in out:
        out["container"] = CONTAINER_LABEL
    return out


def same_bits(a, b):
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    width = {2: np.uint16, 4: np.uint32}[a.dtype.itemsize]
    return np.array_equal(a.view(width), b.view(width))


def tensor_rows(s):
    return next(t for t in s.tensors() if t["tensor"] == TENSOR and t["canonical"])["shape"][0]


def a_factor(rows, scale):
    a = np.full((rows, 1), scale)
    a[2, 0] = 0.0
    return a


# ─── Every operation equals the in-process session ───


class TestEquality:
    def test_tensors_and_tensor(self, svc, container):
        s, r = larql.session(container), svc.client
        assert r.tensors() == s.tensors()
        assert r.tensors("decoder_stack") == s.tensors("decoder_stack")
        for rows, cols, bits in [(None, None, False), ([1, 3], [0, 5, 7], False), ([4], None, True)]:
            got, want = r.tensor(TENSOR, rows=rows, cols=cols, bits=bits), s.tensor(TENSOR, rows=rows, cols=cols, bits=bits)
            assert isinstance(got, Readout)
            assert same_bits(got.array, want.array)
            assert got.provenance == redacted(want.provenance)

    def test_update_weights(self, svc, container):
        s = larql.session(container)
        assert svc.client.update_weights(TENSOR, 5, [0, 1], 2.0) == s.update_weights(TENSOR, 5, [0, 1], 2.0)

    def test_tokens(self, svc, container):
        s, r = larql.session(container), svc.client
        assert r.tokenize(PROMPT) == s.tokenize(PROMPT)
        assert r.tokenize(CONTINUATION, raw=True) == s.tokenize(CONTINUATION, raw=True)
        ids = s.tokenize(PROMPT, raw=True)
        assert r.decode(ids) == s.decode(ids) == PROMPT

    @pytest.mark.parametrize("site,kwargs", [
        ("logits", {}),
        ("final", {"positions": "ALL"}),
        ("residual", {"layer": 0, "positions": "PROMPT_END"}),
        ("post_attention", {"layers": [1, 0], "continue_text": CONTINUATION, "positions": "REPLY"}),
        ("ffn.input", {"layer": 1, "continue_tokens": [3, 7], "positions": [0, 6]}),
        ("ffn.down_output", {"layers": [0], "generate": 2, "positions": "all"}),
        ("ffn.down_input", {"layer": 1}),
    ])
    def test_capture(self, svc, container, site, kwargs):
        got = svc.client.capture(site, PROMPT, **kwargs)
        want = larql.session(container).capture(site, PROMPT, **kwargs)
        assert same_bits(got.array, want.array)
        assert got.provenance == redacted(want.provenance)

    def test_infer_and_generate(self, svc, container):
        s, r = larql.session(container), svc.client
        assert r.infer(PROMPT, top=4) == s.infer(PROMPT, top=4)
        assert r.infer(PROMPT) == s.infer(PROMPT)
        assert r.generate(PROMPT, 3) == s.generate(PROMPT, 3)

    def test_low_rank_update_save_apply_remove(self, svc, container, tmp_path):
        s, r = larql.session(container), svc.client
        a, b = a_factor(tensor_rows(s), 0.125), np.ones((1, 3))

        r.begin_patch("edit.vlp")
        s.begin_patch(str(tmp_path / "edit.vlp"))
        assert r.add_low_rank(TENSOR, a, b, COLUMNS, F64_PROFILE) == s.add_low_rank(TENSOR, a, b, COLUMNS, F64_PROFILE)
        saved_r, saved_s = r.save_patch(), s.save_patch()
        assert saved_r.pop("path") == "edit.vlp"
        saved_s.pop("path")
        assert saved_r == saved_s
        with open(svc.path("edit.vlp"), "rb") as f, open(tmp_path / "edit.vlp", "rb") as g:
            assert f.read() == g.read()
        assert r.discard_weight_edits() == s.discard_weight_edits()

        fresh = larql.session(container)
        base = fresh.capture("logits", PROMPT, positions="ALL").array
        applied_r, applied_s = r.apply_patch("edit.vlp"), fresh.apply_patch(str(tmp_path / "edit.vlp"))
        assert applied_r.pop("path") == "edit.vlp"
        applied_s.pop("path")
        assert applied_r == applied_s
        edited = r.capture("logits", PROMPT, positions="ALL")
        assert same_bits(edited.array, fresh.capture("logits", PROMPT, positions="ALL").array)
        assert not same_bits(edited.array, base)
        listed = r.patches()
        assert listed["applied"][0]["path"] == "edit.vlp"
        assert listed["applied"][0]["sha256"] == fresh.patches()["applied"][0]["sha256"]
        assert r.remove_patch("edit.vlp") == fresh.remove_patch(str(tmp_path / "edit.vlp"))
        assert same_bits(r.capture("logits", PROMPT, positions="ALL").array, base)


# ─── Refusals ───


class TestRefusals:
    def test_errors_carry_the_service_reason(self, svc):
        with pytest.raises(ServiceError) as e:
            svc.client.capture("nowhere", PROMPT)
        assert e.value.kind == "invalid_request"
        assert "unknown site" in e.value.reason
        with pytest.raises(ServiceError, match="EXPORT WEIGHT refused"):
            svc.client.tensor("no.such.tensor")
        with pytest.raises(TypeError, match="float64 or float32"):
            svc.client.add_low_rank(TENSOR, np.ones((4, 1), dtype=np.int32), np.ones((1, 3)), COLUMNS, F64_PROFILE)
        # The service keeps answering.
        assert svc.client.tokenize(PROMPT)

    def test_a_disallowed_operation_is_refused(self, container):
        service = Service(container, operations=["tokenize", "capture"])
        try:
            with pytest.raises(ServiceError) as e:
                service.client.update_weights(TENSOR, 5, [0], 1.0)
            assert e.value.kind == "refused"
            assert "the policy does not allow `update_weights`" in e.value.reason
            assert service.client.info["operations"] == ["tokenize", "capture"]
        finally:
            service.stop()

    def test_paths_outside_the_workdir_are_refused(self, svc):
        outside = os.path.join(svc.root, "outside.vlp")
        with open(outside, "w") as f:
            f.write("{}")
        os.symlink(outside, svc.path("link.vlp"))
        for path in ["../outside.vlp", outside, "/etc/hosts", "link.vlp"]:
            with pytest.raises(ServiceError) as e:
                svc.client.apply_patch(path)
            assert e.value.kind == "refused", (path, e.value.reason)
            with pytest.raises(ServiceError) as e:
                svc.client.begin_patch(path)
            assert e.value.kind == "refused", (path, e.value.reason)

    def test_limits(self, container):
        service = Service(container, limits={"max_generate_tokens": 2, "max_response_bytes": 1024})
        try:
            with pytest.raises(ServiceError, match="max_generate_tokens"):
                service.client.generate(PROMPT, 3)
            with pytest.raises(ServiceError, match="max_response_bytes"):
                service.client.tensor(TENSOR)
        finally:
            service.stop()

    def test_a_second_client_is_turned_away(self, svc):
        with pytest.raises(ServiceError) as e:
            larql.connect(svc.socket)
        assert e.value.kind == "busy"
        assert svc.client.budget() == 5


# ─── Budget ───


class TestBudget:
    def test_base_free_edited_counted_refused_then_free_after_remove(self, container):
        service = Service(container, budget=2)
        try:
            r = service.client
            base = r.capture("logits", PROMPT, positions="ALL").array
            r.infer(PROMPT)
            assert r.budget() == 2
            r.begin_patch("edit.vlp")
            r.update_weights(TENSOR, 5, list(range(8)), 64.0)
            r.save_patch()
            r.discard_weight_edits()
            assert r.budget() == 2
            r.apply_patch("edit.vlp")
            r.capture("logits", PROMPT)
            assert r.last_response["counted"] is True
            assert r.budget() == 1
            r.generate(PROMPT, 2)
            assert r.budget() == 0
            with pytest.raises(ServiceError) as e:
                r.infer(PROMPT)
            assert e.value.kind == "budget_exhausted"
            r.remove_patch("edit.vlp")
            after = r.capture("logits", PROMPT, positions="ALL")
            assert r.last_response["counted"] is False
            assert same_bits(after.array, base)
            assert r.budget_status() == {"limit": 2, "used": 2, "remaining": 0}
        finally:
            service.stop()
        records = service.records()
        assert [rec["seq"] for rec in records] == list(range(1, len(records) + 1))
        refused = [rec for rec in records if rec.get("error") and rec["error"]["kind"] == "budget_exhausted"]
        assert len(refused) == 1 and refused[0]["op"] == "infer"
        assert records[-1]["event"] == "stop"
        assert not os.path.exists(service.socket)


# ─── The client without the native extension ───


def test_the_client_imports_without_the_native_extension():
    code = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def guarded(name, *a, **k):\n"
        "    if name.startswith('larql._native'):\n"
        "        raise ImportError('blocked for the test')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guarded\n"
        "import larql\n"
        "assert callable(larql.connect)\n"
        "try:\n"
        "    larql.session('x')\n"
        "except ImportError as e:\n"
        "    assert 'native extension' in str(e)\n"
        "else:\n"
        "    raise SystemExit('larql.session worked without the extension')\n"
        "assert 'larql._native' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_an_answer_to_another_request_is_refused(svc):
    real = svc.client._read_line

    def answer_with_another_id():
        response = json.loads(real())
        response["id"] = -1
        return json.dumps(response).encode()

    svc.client._read_line = answer_with_another_id
    with pytest.raises(ServiceError, match="different request"):
        svc.client.budget()
