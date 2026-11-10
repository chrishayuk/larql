"""PREC-DIAG-1 evidence lifecycle and admission (rev 3 "Evidence admission").

One rule for every artifact: transformers outputs, LARQL logit dumps and
captures, rotary tables, derived metrics, results.

* A run (or derivation) owns a directory with `run.result.json`, written by
  temp file + rename. It is `in_progress` from creation and then exactly one
  of the final statuses. Only `complete` admits evidence.
* `complete` carries the sha256 and size of every output file and the run's
  identity.
* `admit()` re-verifies all of that against the bytes on disk and against
  the identity the protocol froze. Every failure is a refusal with a reason:
  it fails closed and never raises past the caller.
* `ResultsLog` is an append-only, hash-chained JSON-lines file.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RESULT_FILE = "run.result.json"
RECORD_SCHEMA = "larql.prec-diag-1.run-result/1"
LOG_SCHEMA = "larql.prec-diag-1.results-line/1"

IN_PROGRESS = "in_progress"
COMPLETE = "complete"
RESOURCE_LIMIT = "resource_limit"
FAULT = "fault"
REFUSED = "refused"
ABORTED_PREFLIGHT = "aborted_preflight"
FINAL_STATUSES = (COMPLETE, RESOURCE_LIMIT, FAULT, REFUSED, ABORTED_PREFLIGHT)

#: Reading statuses in the results log.
READING_COMPLETE = "complete"
NOT_ADMITTED = "not_admitted"
INVALID = "invalid"

GENESIS = "0" * 64
HASH_CHUNK = 1 << 22

#: Environment variables that can change numerics or threading, recorded
#: (and compared) whenever an identity carries an environment block.
NUMERIC_ENV_PREFIXES = ("LARQL_", "OMP_", "VECLIB_", "RAYON_", "MKL_", "OPENBLAS_", "PYTORCH_", "TORCH_")


class EvidenceError(Exception):
    """A lifecycle rule was broken by the caller (not an admission refusal)."""


def contained(base: Path, rel: str) -> Path | None:
    """`base / rel` if `rel` is a plain relative name that stays inside
    `base` (lexically and after resolving symlinks); otherwise None.

    Every name a record or manifest supplies goes through this before it
    is opened: an absolute name or a `..` would make `Path.__truediv__`
    leave the run directory."""
    if not isinstance(rel, str) or not rel or "\\" in rel or "\0" in rel:
        return None
    # On the raw string: pathlib drops "." and empty components, which would
    # let "./x" or "a//b" through as if they were plain names.
    if rel.startswith("/") or any(part in ("", ".", "..") for part in rel.split("/")):
        return None
    path = Path(base) / rel
    try:
        root = Path(base).resolve(strict=True)
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    if resolved != root and root not in resolved.parents:
        return None
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def numeric_environment(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in sorted(env.items()) if k.startswith(NUMERIC_ENV_PREFIXES)}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory, fsync, rename."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def write_json_atomic(path: Path, value) -> None:
    write_atomic(path, (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode())


class RunRecord:
    """The lifecycle of one run directory."""

    def __init__(self, run_dir: Path, record: dict):
        self.run_dir = run_dir
        self.record = record

    @property
    def path(self) -> Path:
        return self.run_dir / RESULT_FILE

    @classmethod
    def begin(cls, run_dir: Path, run_id: str, kind: str, identity: dict) -> "RunRecord":
        """Create `run_dir` (it must not exist) and record `in_progress`.

        A run directory is never reused: an existing one is refused, so a
        run can't be silently repeated over its own record."""
        run_dir = Path(run_dir)
        if run_dir.exists():
            raise EvidenceError(f"{run_dir} exists: a run directory is never reused")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
        record = {
            "schema": RECORD_SCHEMA,
            "run_id": run_id,
            "kind": kind,
            "status": IN_PROGRESS,
            "identity": identity,
            "started": now(),
        }
        rr = cls(run_dir, record)
        write_json_atomic(rr.path, record)
        return rr

    def _final(self, status: str, **fields) -> dict:
        if status not in FINAL_STATUSES:
            raise EvidenceError(f"{status!r} is not a final status")
        on_disk = json.loads(self.path.read_text())
        if on_disk.get("status") != IN_PROGRESS:
            raise EvidenceError(f"{self.path}: already final ({on_disk.get('status')})")
        record = {**self.record, **fields, "status": status, "finished": now()}
        write_json_atomic(self.path, record)
        self.record = record
        return record

    def complete(self, outputs: list[str], **fields) -> dict:
        """Hash every output (paths relative to the run directory) and record
        `complete`. An output that is missing is a caller error, recorded as a
        fault rather than a completion."""
        files = {}
        for rel in sorted(outputs):
            p = contained(self.run_dir, rel)
            if p is None:
                self._final(FAULT, detail={"output_outside_run_dir": rel})
                raise EvidenceError(f"output {rel!r} is not inside the run directory; recorded fault")
            if not p.is_file():
                self._final(FAULT, detail={"missing_output": rel})
                raise EvidenceError(f"output {rel} missing; recorded fault")
            files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        return self._final(COMPLETE, outputs=files, **fields)

    def fail(self, status: str, detail, **fields) -> dict:
        if status == COMPLETE:
            raise EvidenceError("use complete() to record completion")
        return self._final(status, detail=detail, **fields)


def record_digest(run_dir: Path) -> str:
    return sha256_file(Path(run_dir) / RESULT_FILE)


@dataclass
class Admission:
    run_dir: Path
    admitted: bool
    reasons: list[str] = field(default_factory=list)
    record: dict | None = None
    record_sha256: str | None = None

    def as_json(self) -> dict:
        return {"run_dir": str(self.run_dir), "admitted": self.admitted, "reasons": self.reasons,
                "record_sha256": self.record_sha256}


def _identity_mismatches(expected: dict, actual: dict, prefix: str = "") -> list[str]:
    """Every expected key must be present and equal (recursively for dicts).
    Keys the identity has beyond `expected` are allowed (recorded, not gated)."""
    out = []
    for key, want in expected.items():
        name = f"{prefix}{key}"
        if key not in actual:
            out.append(f"identity.{name} missing")
        elif isinstance(want, dict) and isinstance(actual[key], dict):
            out += _identity_mismatches(want, actual[key], f"{name}.")
        elif actual[key] != want:
            out.append(f"identity.{name} = {actual[key]!r}, expected {want!r}")
    return out


def admit(run_dir: Path, expected_identity: dict | None = None,
          required_outputs: list[str] | None = None) -> Admission:
    """Whether `run_dir` admits evidence. Never raises: any failure is a refusal."""
    run_dir = Path(run_dir)
    adm = Admission(run_dir, False)
    try:
        path = run_dir / RESULT_FILE
        if not path.is_file():
            adm.reasons.append(f"{RESULT_FILE} missing")
            return adm
        raw = path.read_bytes()
        adm.record_sha256 = sha256_bytes(raw)
        record = json.loads(raw)
        adm.record = record
        if record.get("schema") != RECORD_SCHEMA:
            adm.reasons.append(f"schema {record.get('schema')!r}")
        if record.get("status") != COMPLETE:
            adm.reasons.append(f"status {record.get('status')!r} (only {COMPLETE!r} admits)")
            return adm
        outputs = record.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            adm.reasons.append("complete record lists no outputs")
            return adm
        for rel in required_outputs or []:
            if rel not in outputs:
                adm.reasons.append(f"required output {rel} not in record")
        for rel, meta in outputs.items():
            p = contained(run_dir, rel)
            if p is None:
                adm.reasons.append(f"output {rel!r} is not inside the run directory")
                continue
            if not isinstance(meta, dict):
                adm.reasons.append(f"output {rel} has no digest record")
                continue
            if not p.is_file():
                adm.reasons.append(f"output {rel} missing on disk")
                continue
            if p.stat().st_size != meta.get("bytes"):
                adm.reasons.append(f"output {rel} size {p.stat().st_size} != recorded {meta.get('bytes')}")
                continue
            if sha256_file(p) != meta.get("sha256"):
                adm.reasons.append(f"output {rel} sha256 differs from the record")
        if expected_identity is not None:
            adm.reasons += _identity_mismatches(expected_identity, record.get("identity") or {})
    except Exception as e:  # fail closed
        adm.reasons.append(f"unreadable: {type(e).__name__}: {e}")
        return adm
    adm.admitted = not adm.reasons
    return adm


class ResultsLog:
    """Append-only, hash-chained JSON lines. Each line names the sha256 of
    the line before it, so an edit or deletion anywhere breaks `verify()`."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _lines(self) -> list[bytes]:
        if not self.path.exists():
            return []
        return [ln for ln in self.path.read_bytes().split(b"\n") if ln]

    def verify(self) -> list[dict]:
        prev, entries = GENESIS, []
        for i, line in enumerate(self._lines()):
            entry = json.loads(line)
            if entry.get("schema") != LOG_SCHEMA:
                raise EvidenceError(f"line {i}: schema {entry.get('schema')!r}")
            if entry.get("seq") != i:
                raise EvidenceError(f"line {i}: seq {entry.get('seq')}")
            if entry.get("prev_line_sha256") != prev:
                raise EvidenceError(f"line {i}: chain broken")
            prev = sha256_bytes(line)
            entries.append(entry)
        return entries

    def append(self, entry: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                lines = self._lines()
                self.verify()  # never extend a broken chain
                prev = sha256_bytes(lines[-1]) if lines else GENESIS
                line_entry = {**entry, "schema": LOG_SCHEMA, "seq": len(lines),
                              "prev_line_sha256": prev, "written_at": now()}
                f.write(canonical_json(line_entry) + b"\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return line_entry


def reading(name: str, status: str, admissions: list[Admission], values: dict | None = None,
            **fields) -> dict:
    """A results-log entry. A reading over any unadmitted input is
    `not_admitted` and carries no values."""
    if any(not a.admitted for a in admissions) and status == READING_COMPLETE:
        status = NOT_ADMITTED
    return {
        "reading": name,
        "status": status,
        "consumed": [a.as_json() for a in admissions],
        "values": values if status == READING_COMPLETE else None,
        **fields,
    }
