"""Container reader for the Q8_0 Qwen3-0.6B VINDEX3 profile.

Written from specs/vindex3-container/SPEC.md revision 2 (sha256 3c4feb5e...), which is
admissible for container framing only. Section numbers below refer to that document.
Numerical semantics (how a Q8_0 block decodes) come from `refcheck.q8_0`, never from the
container specification.

Two kinds of finding:
  * `ContainerError`: the spec says a reader MUST reject (SPEC sections 4.3, 5, 7, 8, 13.2, 14).
  * `Container.warnings`: SHOULD-level or informative discrepancies. They never reject.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from . import q8_0

MAX_HEADER_BYTES = 268435456  # section 4.1: 256 MiB
SEGMENT_SCHEMA = 1
GRAPH_SCHEMA = 6
INDEX_VERSION = 4
BF16 = "BF16"
Q8_0 = "Q8_0"

# section 8.3
CODEC_Q8_0 = {
    "family": "Q8_0",
    "revision": 1,
    "group_elems": 32,
    "element": "ggml-q8_0",
    "group_scale": "in-block",
    "tensor_scale": "in-block",
    "layout": "ggml-block-34",
}

_REP_REQUIRED = {
    "object": str,
    "encoding": str,
    "segment": str,
    "tensor_count": int,
    "payload_bytes": int,
    "payload_sha256": str,
    "segment_sha256": str,
}


class ContainerError(Exception):
    """The container violates a MUST-reject rule of the specification."""


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int  # payload-relative
    len: int


@dataclass
class Segment:
    rep_id: str
    path: str  # absolute path of the file
    header_len: int  # H
    payload_start: int  # P = 8 + H
    payload_bytes: int
    file_size: int
    tensors: list[TensorEntry]
    by_name: dict[str, TensorEntry]


@dataclass(frozen=True)
class TensorRef:
    object_id: str
    name: str
    hf_key: str
    rep_id: str
    segment_path: str
    dtype: str
    shape: tuple[int, ...]
    abs_start: int
    abs_end: int


@dataclass
class ObjectInfo:
    id: str
    kind: str
    tensor_prefix: str
    graph_encodings: list[str]  # element 0 is canonical
    binding_tensors: int | None
    binding_bytes: int | None


@dataclass
class Container:
    root: str
    index: dict
    graph: dict
    objects: dict[str, ObjectInfo]
    segments: dict[str, Segment]
    reps: dict[str, dict]
    warnings: list[str] = field(default_factory=list)

    # -- section 7 ---------------------------------------------------------------
    def representation_for(self, object_id: str, want: str = Q8_0) -> str:
        """Section 7 step 3: `<object>@<want>` if the index has it, else the canonical one."""
        obj = self.objects[object_id]
        if f"{object_id}@{want}" in self.reps:
            return f"{object_id}@{want}"
        return f"{object_id}@{obj.graph_encodings[0]}"

    def locate(self, object_id: str, name: str, representation: str | None = None) -> TensorRef:
        """Find a tensor. `representation=None` follows section 7 (what production-q8 stored executes);
        otherwise pass an encoding label such as "BF16" to read that representation. Never falls back."""
        if object_id not in self.objects:
            raise ContainerError(f"unknown object {object_id!r}")
        if representation is None:
            rep_id = self.representation_for(object_id)
        else:
            rep_id = f"{object_id}@{representation}"
            if rep_id not in self.reps:
                raise ContainerError(f"representation {rep_id!r} does not exist")
        seg = self.segments[rep_id]
        entry = seg.by_name.get(name)
        if entry is None:
            raise ContainerError(f"tensor {name!r} absent from representation {rep_id!r}")
        start = seg.payload_start + entry.offset
        return TensorRef(
            object_id=object_id,
            name=name,
            hf_key=self.objects[object_id].tensor_prefix + "." + name,
            rep_id=rep_id,
            segment_path=seg.path,
            dtype=entry.dtype,
            shape=entry.shape,
            abs_start=start,
            abs_end=start + entry.len,
        )

    def tensors(self, representation: str | None = None) -> list[TensorRef]:
        """Every logical tensor in the selected representation of every object (object-id order, then table order)."""
        out = []
        for oid in sorted(self.objects):
            if representation is None:
                rep_id = self.representation_for(oid)
            else:
                rep_id = f"{oid}@{representation}"
                if rep_id not in self.reps:
                    continue
            out.extend(self.locate(oid, e.name, rep_id.split("@", 1)[1]) for e in self.segments[rep_id].tensors)
        return out

    def hf_key_map(self, representation: str | None = None) -> dict[str, TensorRef]:
        m: dict[str, TensorRef] = {}
        for ref in self.tensors(representation):
            if ref.hf_key in m:
                other = m[ref.hf_key]
                raise ContainerError(
                    f"duplicate ownership of HF key {ref.hf_key!r}: {other.object_id}:{other.name} and "
                    f"{ref.object_id}:{ref.name} (SPEC 10.1, D1 OQ12)"
                )
            m[ref.hf_key] = ref
        return m

    def check_unique_ownership(self) -> None:
        """D5 AMB-018: each upstream HF key has exactly one owning logical object, across every representation the
        container declares in index.json. Rejects (a) a representation of one object and a representation of a
        different object that yield the same HF key under the section 10.1 forward rule, and (b) two stored tensors
        within one representation that yield the same key. It does not matter whether the objects are ever executed
        together, or which representation is selected for execution.

        Not rejected: several representations of one object's tensor sharing a key, physically shared storage,
        identical content under distinct keys, and reuse the graph declares (head_reuses_embedding)."""
        owner: dict[str, tuple[str, str]] = {}  # hf_key -> (object_id, first representation that yielded it)
        for rep_id in sorted(self.reps):
            oid = rep_id.split("@", 1)[0]
            if oid not in self.objects:
                continue
            prefix = self.objects[oid].tensor_prefix
            seen: dict[str, str] = {}
            for e in self.segments[rep_id].tensors:
                key = prefix + "." + e.name
                if key in seen:
                    raise ContainerError(
                        f"duplicate ownership of HF key {key!r}: tensors {seen[key]!r} and {e.name!r} in "
                        f"representation {rep_id!r} (SPEC 10.1, D5 AMB-018)"
                    )
                seen[key] = e.name
                held = owner.get(key)
                if held is None:
                    owner[key] = (oid, rep_id)
                elif held[0] != oid:
                    raise ContainerError(
                        f"duplicate ownership of HF key {key!r}: objects {held[0]!r} ({held[1]}) and {oid!r} "
                        f"({rep_id}) both yield it (SPEC 10.1, D5 AMB-018)"
                    )

    def read_bytes(self, ref: TensorRef) -> bytes:
        with open(ref.segment_path, "rb") as f:
            f.seek(ref.abs_start)
            data = f.read(ref.abs_end - ref.abs_start)
        if len(data) != ref.abs_end - ref.abs_start:
            raise ContainerError(f"short read for {ref.hf_key}")
        return data

    def decode(self, ref: TensorRef) -> np.ndarray:
        """Decode a tensor to float32, in its declared shape. BF16 is widened exactly; Q8_0 uses refcheck.q8_0."""
        data = self.read_bytes(ref)
        if ref.dtype == BF16:
            return decode_bf16(data).reshape(ref.shape)
        if ref.dtype == Q8_0:
            k = ref.shape[-1]
            rows = int(np.prod(ref.shape[:-1], dtype=np.int64)) if len(ref.shape) > 1 else 1
            return q8_0.dequantize_q8_0_rows(data, rows, k).reshape(ref.shape)
        raise ContainerError(f"cannot decode dtype {ref.dtype!r}")

    # -- section 12: tied embedding facts ----------------------------------------------
    def tied_report(self) -> dict:
        """Facts about the tied embedding/head, read from the container only (never a verdict)."""
        out: dict = {}
        emb, head = "target.embedding@BF16", "target.output_head@BF16"
        if emb in self.reps and head in self.reps:
            out["payload_sha256_equal"] = self.reps[emb]["payload_sha256"] == self.reps[head]["payload_sha256"]
            out["separate_objects"] = True
        else:
            out["separate_objects"] = False
        for c in self.graph.get("components", []):
            surf = (c.get("execution") or {}).get("head") or {}
            if "head_reuses_embedding" in surf:
                out["head_reuses_embedding"] = surf["head_reuses_embedding"]
        for oid in ("target.embedding", "target.output_head"):
            if oid in self.objects:
                out[f"{oid}_executed_representation"] = self.representation_for(oid)
        return out


def decode_bf16(data: bytes) -> np.ndarray:
    """BF16 -> float32: the 16 bits are the high half of the f32 pattern (exact). Little-endian bytes."""
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size % 2:
        raise ContainerError("BF16 byte length is odd")
    u = raw[0::2].astype(np.uint32) | (raw[1::2].astype(np.uint32) << 8)
    return (u << 16).astype(np.uint32).view(np.float32)


def expected_len(dtype: str, shape: tuple[int, ...]) -> int:
    """Section 8.1 byte length, raising ContainerError for layouts the spec refuses."""
    if dtype == BF16:
        return 2 * int(np.prod(shape, dtype=object)) if shape else 2
    if dtype == Q8_0:
        if len(shape) < 1:
            raise ContainerError("Q8_0 tensor needs at least one dimension")
        k = shape[-1]
        if k % q8_0.QK8_0:
            raise ContainerError(f"Q8_0 row length {k} is not a multiple of 32")
        rows = 1
        for d in shape[:-1]:
            rows *= d
        return rows * (k // q8_0.QK8_0) * q8_0.BLOCK_BYTES
    raise ContainerError(f"dtype {dtype!r} is not BF16 or Q8_0")


# ---------------------------------------------------------------------------------------
# parsing helpers


def _strict_json(data: bytes, what: str):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ContainerError(f"{what}: duplicate JSON key {k!r}")
            d[k] = v
        return d

    def const(name):
        raise ContainerError(f"{what}: non-finite JSON constant {name}")

    try:
        text = data.decode("utf-8")
        return json.loads(text, object_pairs_hook=pairs, parse_constant=const)
    except ContainerError:
        raise
    except (UnicodeDecodeError, ValueError) as e:
        raise ContainerError(f"{what}: not valid JSON ({e})") from e


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _safe_relpath(root: str, rel, what: str) -> str:
    if not isinstance(rel, str) or not rel:
        raise ContainerError(f"{what}: path must be a non-empty string")
    p = PurePosixPath(rel)
    if p.is_absolute() or rel.startswith("\\") or (len(rel) > 1 and rel[1] == ":") or ".." in p.parts:
        raise ContainerError(f"{what}: path {rel!r} is absolute or contains '..'")
    return os.path.join(root, *p.parts)


def _read_json_file(path: str, what: str):
    try:
        with open(path, "rb") as f:
            return _strict_json(f.read(), what)
    except FileNotFoundError as e:
        raise ContainerError(f"{what}: file is missing ({path})") from e
    except IsADirectoryError as e:
        raise ContainerError(f"{what}: not a file ({path})") from e


def _parse_segment(rep_id: str, path: str, entry: dict, warnings: list[str]) -> Segment:
    what = f"segment {rep_id}"
    try:
        size = os.path.getsize(path)
        f = open(path, "rb")
    except OSError as e:
        raise ContainerError(f"{what}: cannot open {path}: {e}") from e
    with f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ContainerError(f"{what}: shorter than the 8-byte length prefix")
        (hlen,) = struct.unpack("<Q", prefix)
        if hlen > MAX_HEADER_BYTES:
            raise ContainerError(f"{what}: header length {hlen} exceeds 256 MiB")
        if 8 + hlen > size:
            raise ContainerError(f"{what}: header length {hlen} runs past the end of the file")
        hdr_bytes = f.read(hlen)
    header = _strict_json(hdr_bytes, f"{what} header")
    if not isinstance(header, dict):
        raise ContainerError(f"{what}: header is not a JSON object")
    if header.get("schema") != SEGMENT_SCHEMA or not _is_int(header.get("schema")):
        raise ContainerError(f"{what}: header schema is {header.get('schema')!r}, expected 1")
    if header.get("representation") != rep_id:
        raise ContainerError(
            f"{what}: header.representation {header.get('representation')!r} differs from its key"
        )
    raw = header.get("tensors")
    if not isinstance(raw, list):
        raise ContainerError(f"{what}: header.tensors is not an array")

    p = 8 + hlen
    if p % 4:
        warnings.append(f"{what}: payload start {p} is not a multiple of 4")
    tensors: list[TensorEntry] = []
    by_name: dict[str, TensorEntry] = {}
    cursor = 0
    for i, t in enumerate(raw):
        if not isinstance(t, dict):
            raise ContainerError(f"{what}: tensor entry {i} is not an object")
        name, dtype, shape = t.get("name"), t.get("dtype"), t.get("shape")
        off, ln = t.get("offset"), t.get("len")
        if not isinstance(name, str):
            raise ContainerError(f"{what}: tensor entry {i} has no string name")
        if not isinstance(dtype, str):
            raise ContainerError(f"{what}: tensor {name!r} has no string dtype")
        if not isinstance(shape, list) or not all(_is_int(d) and d >= 0 for d in shape):
            raise ContainerError(f"{what}: tensor {name!r} has an invalid shape")
        if not (_is_int(off) and _is_int(ln) and 0 <= off < 2**64 and 0 <= ln < 2**64):
            raise ContainerError(f"{what}: tensor {name!r} has an invalid offset or len")
        if name in by_name:
            raise ContainerError(f"{what}: duplicate tensor name {name!r}")
        if off != cursor:
            raise ContainerError(f"{what}: tensor {name!r} offset {off} breaks contiguity (expected {cursor})")
        try:
            want = expected_len(dtype, tuple(shape))
        except ContainerError as e:
            raise ContainerError(f"{what}: tensor {name!r}: {e}") from e
        if ln != want:
            raise ContainerError(f"{what}: tensor {name!r} len {ln} disagrees with dtype/shape ({want})")
        e = TensorEntry(name, dtype, tuple(shape), off, ln)
        tensors.append(e)
        by_name[name] = e
        cursor += ln

    names = [t.name.encode("utf-8") for t in tensors]
    if names != sorted(names):
        warnings.append(f"{what}: tensor table is not sorted by name bytes")

    payload_bytes = entry["payload_bytes"]
    if entry["tensor_count"] != len(tensors):
        raise ContainerError(f"{what}: tensor_count {entry['tensor_count']} != table length {len(tensors)}")
    if cursor != payload_bytes:
        raise ContainerError(f"{what}: sum of tensor lengths {cursor} != payload_bytes {payload_bytes}")
    if size != p + payload_bytes:
        raise ContainerError(f"{what}: file length {size} != P + payload_bytes ({p + payload_bytes})")
    return Segment(rep_id, path, hlen, p, payload_bytes, size, tensors, by_name)


def _verify_digests(seg: Segment, entry: dict) -> None:
    whole, payload = hashlib.sha256(), hashlib.sha256()
    pos = 0
    with open(seg.path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            whole.update(chunk)
            if pos + len(chunk) > seg.payload_start:
                payload.update(chunk[max(0, seg.payload_start - pos) :])
            pos += len(chunk)
    if whole.hexdigest() != entry["segment_sha256"]:
        raise ContainerError(f"segment {seg.rep_id}: segment_sha256 mismatch")
    if payload.hexdigest() != entry["payload_sha256"]:
        raise ContainerError(f"segment {seg.rep_id}: payload_sha256 mismatch")


# ---------------------------------------------------------------------------------------


def open_container(root: str, verify_digests: bool = True) -> Container:
    """Open and fully validate a container directory. Raises ContainerError on any MUST-reject condition."""
    root = os.path.abspath(root)
    warnings: list[str] = []

    # section 14 rule 1, 2, 3, 9
    index = _read_json_file(os.path.join(root, "index.json"), "index.json")
    if not isinstance(index, dict):
        raise ContainerError("index.json is not a JSON object")
    if "version" not in index:
        raise ContainerError("index.json has no version")
    if not _is_int(index["version"]) or index["version"] != INDEX_VERSION:
        raise ContainerError(f"index.json version is {index['version']!r}, this profile specifies 4")
    if "system_graph" not in index:
        raise ContainerError("index.json has no system_graph (bank shape is out of scope)")
    if "moe_manifest" in index:
        raise ContainerError("index.json has moe_manifest (bank shape is out of scope)")
    if "auxiliary_references" in index:
        raise ContainerError("index.json has auxiliary_references (outside this profile)")
    if index.get("authority", "canonical") != "canonical":
        raise ContainerError(f"index.json authority is {index['authority']!r}, this profile requires canonical")

    graph_path = _safe_relpath(root, index["system_graph"], "index.system_graph")
    graph = _read_json_file(graph_path, "system_graph.json")
    if not isinstance(graph, dict):
        raise ContainerError("system_graph.json is not a JSON object")
    if not _is_int(graph.get("schema")) or graph["schema"] != GRAPH_SCHEMA:
        raise ContainerError(f"system_graph.json schema is {graph.get('schema')!r}, expected 6")

    # graph objects (sections 6, 10.1)
    objects: dict[str, ObjectInfo] = {}
    g_objects = graph.get("objects")
    if not isinstance(g_objects, list):
        raise ContainerError("system_graph.json has no objects array")
    for o in g_objects:
        if not isinstance(o, dict) or not isinstance(o.get("id"), str):
            raise ContainerError("graph object without a string id")
        oid = o["id"]
        if oid in objects:
            raise ContainerError(f"duplicate graph object {oid!r}")
        bindings = o.get("source_bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ContainerError(f"graph object {oid!r} must have exactly one source binding")
        b = bindings[0]
        if not isinstance(b, dict) or not isinstance(b.get("tensor_prefix"), str) or not b["tensor_prefix"]:
            raise ContainerError(f"graph object {oid!r} binding has no tensor_prefix")
        reps = o.get("representations")
        if (
            not isinstance(reps, list)
            or not reps
            or not all(isinstance(r, dict) and isinstance(r.get("encoding"), str) for r in reps)
        ):
            raise ContainerError(f"graph object {oid!r} has no valid representations list")
        objects[oid] = ObjectInfo(
            id=oid,
            kind=str(o.get("kind")),
            tensor_prefix=b["tensor_prefix"],
            graph_encodings=[r["encoding"] for r in reps],
            binding_tensors=b.get("tensors") if _is_int(b.get("tensors")) else None,
            binding_bytes=b.get("bytes") if _is_int(b.get("bytes")) else None,
        )

    # index representations (section 5.3)
    reps = index.get("representations")
    if not isinstance(reps, dict) or not reps:
        raise ContainerError("index.json has no representations object")
    for rid, e in reps.items():
        if not isinstance(e, dict):
            raise ContainerError(f"representation {rid!r} is not an object")
        for k, ty in _REP_REQUIRED.items():
            v = e.get(k)
            if not isinstance(v, ty) or (ty is int and not _is_int(v)):
                raise ContainerError(f"representation {rid!r}: field {k!r} missing or of the wrong type")
        # section 14 rule 5
        if e["object"] not in objects:
            raise ContainerError(f"representation {rid!r} names object {e['object']!r} absent from the graph")
        if rid != f"{e['object']}@{e['encoding']}":
            raise ContainerError(f"representation key {rid!r} is not '<object>@<encoding>' of its own fields")
        if e["encoding"] not in (BF16, Q8_0):
            raise ContainerError(f"representation {rid!r}: unsupported encoding {e['encoding']!r}")
        # section 8.3
        if "codec" in e or e["encoding"] == Q8_0:
            if e.get("codec") != CODEC_Q8_0 or not _codec_types_ok(e.get("codec")):
                raise ContainerError(f"representation {rid!r}: codec is not exactly the Q8_0 identity")
    for oid, obj in objects.items():
        for enc in obj.graph_encodings:
            if f"{oid}@{enc}" not in reps:
                raise ContainerError(f"graph object {oid!r} lists {enc} but index.json has no {oid}@{enc}")
        for rid in reps:
            if reps[rid]["object"] == oid and reps[rid]["encoding"] not in obj.graph_encodings:
                warnings.append(f"index representation {rid} is not listed in the graph for {oid}")

    # segments (section 4)
    segments: dict[str, Segment] = {}
    seen_paths: dict[str, str] = {}
    for rid, e in reps.items():
        path = _safe_relpath(root, e["segment"], f"representation {rid} segment")
        if path in seen_paths:
            warnings.append(f"representations {seen_paths[path]} and {rid} share a segment file")
        seen_paths[path] = rid
        seg = _parse_segment(rid, path, e, warnings)
        if e["encoding"] == BF16 and any(t.dtype != BF16 for t in seg.tensors):
            raise ContainerError(f"segment {rid}: canonical BF16 representation holds a non-BF16 tensor")
        if verify_digests:
            _verify_digests(seg, e)
        segments[rid] = seg

    # SHOULD-level and informative checks
    for rid, e in reps.items():
        src = e.get("compiled_from")
        if src is not None:
            if src not in reps:
                warnings.append(f"{rid}: compiled_from {src!r} is not in the index")
            elif e.get("source_representation_digest") != reps[src]["payload_sha256"]:
                warnings.append(f"{rid}: source_representation_digest != payload_sha256 of {src}")
        elif e["encoding"] != BF16:
            warnings.append(f"{rid}: compiled representation without compiled_from")
    for oid, obj in objects.items():
        canon = f"{oid}@{obj.graph_encodings[0]}"
        if obj.binding_tensors is not None and obj.binding_tensors != reps[canon]["tensor_count"]:
            warnings.append(f"{oid}: binding tensors {obj.binding_tensors} != canonical tensor_count")
        if obj.binding_bytes is not None and obj.binding_bytes != reps[canon]["payload_bytes"]:
            warnings.append(f"{oid}: binding bytes {obj.binding_bytes} != canonical payload_bytes")
    seg_keys = index.get("segments")
    if isinstance(seg_keys, dict):
        have = {k + ".bin" for k in seg_keys}
        want = {e["segment"] for e in reps.values()}
        if have != want:
            warnings.append("index.segments keys do not match the representation segment paths")
    else:
        warnings.append("index.json has no segments object")

    c = Container(root, index, graph, objects, segments, reps, warnings)
    c.check_unique_ownership()
    return c


def _codec_types_ok(codec) -> bool:
    """dict equality treats 1 == True and 32 == 32.0; the identity must match field by field in type too."""
    return isinstance(codec, dict) and all(type(codec.get(k)) is type(v) for k, v in CODEC_Q8_0.items())
