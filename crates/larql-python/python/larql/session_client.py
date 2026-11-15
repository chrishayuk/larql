"""
A client of `larql serve-session`: the typed VINDEX3 session operations, over a Unix socket.

`larql.connect(socket_path)` returns a `RemoteSession` whose methods have the names, arguments and return
types of the in-process `larql.session(...)` methods (`tensors`, `tensor`, `tokenize`, `decode`, `capture`,
`infer`, `generate`, `begin_patch`, `add_low_rank`, `update_weights`, `save_patch`, `apply_patch`,
`remove_patch`, `patches`, `discard_weight_edits`), plus `budget()`. The model runs in the service process;
this process only sends requests and reads answers.

This module is pure Python (the standard library and NumPy). It does not import the native extension, so it
works where `larql._native` is not installed.

Paths name files in the service's workdir, relative to it (or absolutely under it). Errors come back as
`ServiceError`, carrying the service's `kind` and `reason`.

Protocol: `larql-session/v1`, one JSON object per line each way (see the service's documentation in
crates/larql-cli/README.md). Arrays travel as whole `.npy` files, base64 in the JSON.
"""

import base64
import hashlib
import io
import json
import socket

import numpy as np

PROTOCOL = "larql-session/v1"
_READ_CHUNK = 1 << 20


class ServiceError(RuntimeError):
    """A request the service did not answer with a result. `kind` is one of invalid_request,
    unsupported_protocol, refused, budget_exhausted, failed or busy; `reason` is the service's."""

    def __init__(self, kind, reason, response=None):
        super().__init__(f"{kind}: {reason}")
        self.kind = kind
        self.reason = reason
        self.response = response


class Readout:
    """An owned array and the provenance record of the values in it, as the in-process `Readout`."""

    __slots__ = ("array", "provenance")

    def __init__(self, array, provenance):
        self.array = array
        self.provenance = provenance

    def __repr__(self):
        return f"Readout(shape={self.array.shape!r}, dtype={self.array.dtype})"


def _decode_array(encoded):
    raw = base64.b64decode(encoded["npy_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != encoded["sha256"]:
        raise ServiceError("failed", "an array's bytes do not match the digest the service sent")
    return np.load(io.BytesIO(raw), allow_pickle=False)


def _encode_operand(array, which):
    if not isinstance(array, np.ndarray) or array.ndim != 2 or array.dtype not in (np.float64, np.float32):
        raise TypeError(f"`{which}` must be a 2-D numpy array of float64 or float32")
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")), allow_pickle=False)
    return {"npy_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}


class RemoteSession:
    """A connection to one `larql serve-session`. One request at a time; not shared between threads."""

    def __init__(self, socket_path, timeout=None):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(str(socket_path))
        self._buffer = b""
        self._next_id = 0
        self.last_response = None
        self.info = self._call("hello")

    # ── transport ──

    def _read_line(self):
        while b"\n" not in self._buffer:
            chunk = self._sock.recv(_READ_CHUNK)
            if not chunk:
                raise ServiceError("failed", "the service closed the connection")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line

    def _call(self, op, **args):
        self._next_id += 1
        request = {"protocol": PROTOCOL, "id": self._next_id, "op": op, "args": args}
        self._sock.sendall(json.dumps(request).encode("utf-8") + b"\n")
        response = json.loads(self._read_line())
        self.last_response = response
        if not response.get("ok"):
            error = response.get("error") or {}
            raise ServiceError(error.get("kind", "failed"), error.get("reason", "no reason given"), response)
        if response.get("id") != self._next_id:
            raise ServiceError("failed", "the service answered a different request")
        return response["result"]

    def _readout(self, result):
        return Readout(_decode_array(result["array"]), result["provenance"])

    def close(self):
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── the session operations ──

    def tensors(self, filter=None):
        return self._call("tensors", filter=filter)

    def tensor(self, name, rows=None, cols=None, bits=False):
        return self._readout(self._call("tensor", name=name, rows=rows, cols=cols, bits=bits))

    def tokenize(self, text, raw=False):
        return self._call("tokenize", text=text, raw=raw)

    def decode(self, ids):
        return self._call("decode", ids=[int(i) for i in ids])

    def capture(self, site, prompt, layer=None, layers=None, continue_text=None, continue_tokens=None,
                generate=None, positions=None):
        args = {"site": site, "prompt": prompt}
        for key, value in (("layer", layer), ("layers", layers), ("continue_text", continue_text),
                           ("continue_tokens", continue_tokens), ("generate", generate),
                           ("positions", positions)):
            if value is not None:
                args[key] = [int(v) for v in value] if isinstance(value, (list, tuple)) else value
        return self._readout(self._call("capture", **args))

    def infer(self, prompt, top=5):
        return self._call("infer", prompt=prompt, top=top)

    def generate(self, prompt, n):
        return self._call("generate", prompt=prompt, n=n)

    def begin_patch(self, path):
        return self._call("begin_patch", path=str(path))

    def add_low_rank(self, tensor, a, b, columns, numerics):
        return self._call("add_low_rank", tensor=tensor, a=_encode_operand(a, "a"), b=_encode_operand(b, "b"),
                          columns=[int(c) for c in columns], numerics=numerics)

    def update_weights(self, tensor, row, cols, delta):
        return self._call("update_weights", tensor=tensor, row=int(row), cols=[int(c) for c in cols],
                          delta=float(delta))

    def save_patch(self):
        return self._call("save_patch")

    def apply_patch(self, path):
        return self._call("apply_patch", path=str(path))

    def remove_patch(self, path):
        return self._call("remove_patch", path=str(path))

    def patches(self):
        return self._call("patches")

    def discard_weight_edits(self):
        return self._call("discard_weight_edits")

    def budget(self):
        """The edited-model executions left."""
        return self._call("budget")["remaining"]

    def budget_status(self):
        """The budget as the service reports it: `limit`, `used` and `remaining`."""
        return self._call("budget")


def connect(socket_path, timeout=None):
    """Connect to a `larql serve-session` socket. Raises `ServiceError` if the service turns the connection
    away (another client is connected) or speaks another protocol."""
    return RemoteSession(socket_path, timeout=timeout)
