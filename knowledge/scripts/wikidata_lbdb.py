#!/usr/bin/env python3
"""Ladybug-backed Wikidata extraction (fast local path).

Queries the local ``wikidata.lbdb`` columnar graph with the ladybug Python
bindings instead of hitting the Wikidata API / SPARQL endpoint or streaming
the multi-hundred-GB N-Triples dump. The SPARQL/API/dump code paths in the
sibling ``*wikidata*.py`` scripts are untouched and remain the default; this
module powers the opt-in ``--from-lbdb`` / ``--export-lbdb`` flags.

Source schema (``wikidata.lbdb``)::

    wikidata_node(csr_index INT64 PK, qid INT64, name STRING)  -- 91M nodes
    edge_meta(id INT64 PK, name STRING)                        -- 12k property labels
    wikidata_rel(FROM wikidata_node TO wikidata_node, property INT64)
                                                              -- 766M edges

``wikidata_rel.property`` is the numeric Wikidata PID (``36`` == P36) whose
English label lives in ``edge_meta``. There is an ART index on
``wikidata_node.name`` and ``wikidata_node`` internal row offsets equal
``csr_index``, so ``id(o).offset`` from a scan can be resolved with a
batched PK lookup instead of a 766M-row hash join.

Query-plan notes (from EXPLAIN on this database):

* ``MATCH (s)-[r]->(o) WHERE r.property = $pid RETURN s.name, o.name``
  expands **all** 766M edges into a hash-join build side before the LIMIT
  applies, so it OOMs the buffer pool. Do NOT fetch names for both ends in
  one query over the full edge table.
* ``MATCH ()-[r]->() RETURN r.property, count(*)`` (pure columnar scan +
  small GROUP BY) runs in ~5s over all 766M edges.
* ``MATCH (s)-[r]->(o) WHERE r.property = $pid RETURN s.csr_index,
  s.name, id(o) LIMIT ...`` streams (``s.*`` is free from the node scan,
  ``id(o)`` needs no join) and short-circuits, so per-property sampling is
  seconds, not minutes.
* A ``((s.csr_index * 2654435761) % 1000) < $t`` predicate gives a
  deterministic spread sample; calibrate ``$t`` from per-property counts so
  one full scan returns ~``limit`` rows from across the whole ID range.

Derived subset schema (``--export-lbdb`` output)::

    PropInfo(pid INT64 PK, name STRING)
    Entity(id INT64 PK, name STRING)
    Triple(FROM Entity TO Entity, pid INT64, relation STRING)

Requires the ``ladybug`` package (0.22.3+)::

    uv pip install ladybug==0.22.3
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Deterministic spread-sampling buckets (see module docstring).
HASH_BUCKETS = 1000
HASH_MULTIPLIER = 2654435761  # Knuth multiplicative constant

# Chunk size for batched PK lookups (IN-list) and UNWIND inserts.
BATCH_SIZE = 1000

# Default buffer pool cap for Ladybug connections opened here. The engine
# default (~80% of system RAM) plus full-edge-table scans can exhaust a
# small machine, so cap it; override with LBUG_BUFFER_POOL_MB.
DEFAULT_BUFFER_POOL_MB = 2048

# How many extra candidate rows to pull per requested pair (quality
# filtering + dedupe + per-subject capping discard some rows).
OVERSAMPLE = 4

# Max pairs from one subject entity (keeps example sets diverse).
PER_SUBJECT_CAP = 2

# Label rules mirror ingest_wikidata_dump.is_good_label.
MAX_LABEL_WORDS = 6
MAX_LABEL_LEN = 60


def _default_source_candidates() -> list[Path]:
    """Candidate locations for the readonly source ``wikidata.lbdb``."""
    here = Path(__file__).resolve()
    return [
        Path(os.environ["LBUG_WIKIDATA_PATH"]) if os.environ.get("LBUG_WIKIDATA_PATH") else None,
        here.parent.parent.parent / "wikidata.lbdb",  # repo root symlink
        Path("/home/ubuntu/src/ladybug-python/wikidata.lbdb"),
    ]


def find_source_db(explicit: str | Path | None = None) -> Path:
    """Return the first existing source db path, or raise FileNotFoundError."""
    candidates = ([Path(explicit)] if explicit else []) + [
        c for c in _default_source_candidates() if c is not None
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No wikidata.lbdb found (searched: {searched}). "
        "Set LBUG_WIKIDATA_PATH or pass an explicit path."
    )


def require_ladybug():
    """Import ladybug or raise a helpful error."""
    try:
        import ladybug as lb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The ladybug package is required for the --from-lbdb path. Install it with:\n"
            "  uv pip install ladybug==0.22.3"
        ) from exc
    import ladybug

    return ladybug


def _buffer_pool_size() -> int | None:
    """Buffer pool cap in bytes (env override, conservative default)."""
    raw = os.environ.get("LBUG_BUFFER_POOL_MB")
    if raw:
        try:
            return int(raw) * 1024 * 1024
        except ValueError:
            pass
    return DEFAULT_BUFFER_POOL_MB * 1024 * 1024


def connect_readonly(path: str | Path):
    """Open the source database readonly (never mutates the 766M-edge base)."""
    lb = require_ladybug()
    return lb.Connection(
        lb.Database(str(path), read_only=True, buffer_pool_size=_buffer_pool_size())
    )


def connect_writable(path: str | Path):
    """Open (creating if needed) a writable database for subset export."""
    lb = require_ladybug()
    return lb.Connection(lb.Database(str(path), buffer_pool_size=_buffer_pool_size()))


# ---------------------------------------------------------------------------
# Catalog + counts
# ---------------------------------------------------------------------------

def list_properties(conn) -> list[tuple[int, str]]:
    """Return ``[(pid, english_label)]`` for every property in ``edge_meta``."""
    result = conn.execute(
        "MATCH (p:edge_meta) RETURN p.id, p.name ORDER BY p.id"
    )
    props = []
    while result.has_next():
        pid, name = result.get_next()
        props.append((int(pid), name))
    result.close()
    return props


def property_counts(conn, pids: list[int]) -> dict[int, int]:
    """Return ``{pid: edge_count}`` with a single grouped columnar scan."""
    if not pids:
        return {}
    result = conn.execute(
        "MATCH ()-[r:wikidata_rel]->() WHERE r.property IN $pids "
        "RETURN r.property, count(*) AS c",
        {"pids": [int(p) for p in pids]},
    )
    counts: dict[int, int] = {}
    while result.has_next():
        pid, count = result.get_next()
        counts[int(pid)] = int(count)
    result.close()
    return counts


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def is_good_label(label: str) -> bool:
    """Mirror ``ingest_wikidata_dump.is_good_label`` so both paths agree."""
    if not label or not isinstance(label, str):
        return False
    if len(label) > MAX_LABEL_LEN:
        return False
    if len(label.split()) > MAX_LABEL_WORDS:
        return False
    stripped = label.strip()
    if stripped.isdigit():
        return False
    if stripped.startswith("Q") and stripped[1:].isdigit():
        return False
    if stripped.startswith(("Category:", "Template:", "Module:")):
        return False
    if "http" in stripped:
        return False
    return True


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------

def _calibrate_threshold(count: int | None, limit: int) -> int:
    """Pick the hash-bucket threshold so one scan returns ~limit*OVERSAMPLE rows."""
    want = max(limit * OVERSAMPLE, 50)
    if count is None or count <= 0:
        return 5  # blind guess; caller escalates on shortfall
    return max(1, min(HASH_BUCKETS, (want * HASH_BUCKETS) // count))


def _sample_rows(conn, pid: int, threshold: int, row_limit: int) -> list[tuple]:
    """Stream ``(src_csr, src_name, dst_offset)`` rows for one property."""
    result = conn.execute(
        "MATCH (s:wikidata_node)-[r:wikidata_rel]->(o:wikidata_node) "
        "WHERE r.property = $pid "
        "AND ((s.csr_index * $mult) % $buckets) < $t "
        "RETURN s.csr_index, s.name, id(o) LIMIT $lim",
        {
            "pid": int(pid),
            "mult": HASH_MULTIPLIER,
            "buckets": HASH_BUCKETS,
            "t": int(threshold),
            "lim": int(row_limit),
        },
    )
    rows = []
    while result.has_next():
        src_csr, src_name, dst_id = result.get_next()
        offset = dst_id["offset"] if isinstance(dst_id, dict) else dst_id
        rows.append((int(src_csr), src_name, int(offset)))
    result.close()
    return rows


def _resolve_names(conn, csrs: list[int]) -> dict[int, str]:
    """Batch-resolve ``csr_index -> name`` (internal offset == csr_index)."""
    names: dict[int, str] = {}
    unique = sorted(set(int(c) for c in csrs))
    for i in range(0, len(unique), BATCH_SIZE):
        chunk = unique[i:i + BATCH_SIZE]
        result = conn.execute(
            "MATCH (n:wikidata_node) WHERE n.csr_index IN $ids "
            "RETURN n.csr_index, n.name",
            {"ids": chunk},
        )
        while result.has_next():
            csr, name = result.get_next()
            names[int(csr)] = name
        result.close()
    return names


def fetch_pairs(
    conn,
    pid: int,
    limit: int = 30,
    count: int | None = None,
    per_subject_cap: int = PER_SUBJECT_CAP,
) -> list[list[str]]:
    """Fetch up to ``limit`` distinct ``[subject, object]`` pairs for a PID.

    Deterministic spread sample over the local graph; labels pass the same
    quality rules as the dump ingester. Properties with no entity-entity
    edges in the graph (dates, literal-valued like chemical symbols) return
    ``[]`` so callers can fall back to SPARQL/dump logic.
    """
    if limit <= 0:
        return []
    threshold = _calibrate_threshold(count, limit)
    tried = set()
    pairs: list[list[str]] = []
    for _ in range(3):
        if threshold in tried:
            threshold = min(HASH_BUCKETS, max(threshold * 4, 1))
        tried.add(threshold)
        rows = _sample_rows(conn, pid, threshold, limit * OVERSAMPLE * 2)
        if not rows:
            if threshold >= HASH_BUCKETS:
                return []
            threshold = HASH_BUCKETS  # keep-all full scan as last resort
            continue
        dst_names = _resolve_names(conn, [r[2] for r in rows])
        seen: set[tuple[str, str]] = set()
        per_subject: dict[str, int] = {}
        for src_csr, src_name, dst_offset in rows:
            dst_name = dst_names.get(dst_offset, "")
            if not is_good_label(src_name) or not is_good_label(dst_name):
                continue
            key = (src_name, dst_name)
            if key in seen:
                continue
            if per_subject.get(src_name, 0) >= per_subject_cap:
                continue
            seen.add(key)
            per_subject[src_name] = per_subject.get(src_name, 0) + 1
            pairs.append([src_name, dst_name])
            if len(pairs) >= limit:
                break
        if len(pairs) >= limit or threshold >= HASH_BUCKETS:
            break
        threshold = min(HASH_BUCKETS, threshold * 4)
    return pairs[:limit]


# ---------------------------------------------------------------------------
# Subset .lbdb export (the "another .lbdb file")
# ---------------------------------------------------------------------------

SUBSET_SCHEMA = """
CREATE NODE TABLE PropInfo(pid INT64, name STRING, PRIMARY KEY(pid));
CREATE NODE TABLE Entity(id INT64, name STRING, PRIMARY KEY(id));
CREATE REL TABLE Triple(FROM Entity TO Entity, pid INT64, relation STRING);
""".strip()


def export_subset(
    source_conn,
    dest_path: str | Path,
    pid_to_relation: dict[int, str],
    max_per_relation: int = 500,
    overwrite: bool = False,
) -> dict[str, int]:
    """Extract target relations from the source graph into a new small .lbdb.

    Returns ``{relation_name: pair_count}``. Properties with no edges are
    skipped (reported with a count of 0) so callers can fall back.
    """
    dest_path = Path(dest_path)
    if dest_path.exists():
        if not overwrite:
            raise FileExistsError(f"{dest_path} exists (pass overwrite=True)")
        if dest_path.is_dir():
            import shutil

            shutil.rmtree(dest_path)
        else:
            dest_path.unlink()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    counts = property_counts(source_conn, sorted(pid_to_relation))
    conn = connect_writable(dest_path)
    for statement in SUBSET_SCHEMA.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)

    prop_names = {pid: name for pid, name in list_properties(source_conn)}
    entity_ids: dict[str, int] = {}
    stats: dict[str, int] = {}

    for pid, relation in sorted(pid_to_relation.items(), key=lambda kv: kv[1]):
        pairs = fetch_pairs(source_conn, pid, max_per_relation, counts.get(pid, 0))
        stats[relation] = len(pairs)
        conn.execute(
            "UNWIND $rows AS row CREATE (p:PropInfo {pid: row.pid, name: row.name})",
            {"rows": [{"pid": int(pid), "name": prop_names.get(pid, relation)}]},
        )
        if not pairs:
            continue
        new_entities = []
        for subj, obj in pairs:
            for name in (subj, obj):
                if name not in entity_ids:
                    entity_ids[name] = len(entity_ids)
                    new_entities.append({"id": entity_ids[name], "name": name})
        for i in range(0, len(new_entities), BATCH_SIZE):
            conn.execute(
                "UNWIND $rows AS row CREATE (e:Entity {id: row.id, name: row.name})",
                {"rows": new_entities[i:i + BATCH_SIZE]},
            )
        edge_rows = [
            {
                "a": entity_ids[subj],
                "b": entity_ids[obj],
                "pid": int(pid),
                "rel": relation,
            }
            for subj, obj in pairs
        ]
        for i in range(0, len(edge_rows), BATCH_SIZE):
            conn.execute(
                "UNWIND $rows AS row "
                "MATCH (a:Entity), (b:Entity) WHERE a.id = row.a AND b.id = row.b "
                "CREATE (a)-[:Triple {pid: row.pid, relation: row.rel}]->(b)",
                {"rows": edge_rows[i:i + BATCH_SIZE]},
            )
    conn.close()
    return stats


def read_subset_pairs(dest_path: str | Path, relation: str | None = None) -> dict:
    """Read the derived subset db back as ``{relation: {pid, pairs}}`` JSON-shape."""
    conn = connect_readonly(dest_path)
    if relation is None:
        props = conn.execute("MATCH (p:PropInfo) RETURN p.pid, p.name")
        relations = []
        while props.has_next():
            pid, name = props.get_next()
            relations.append((int(pid), name))
        props.close()
    else:
        match = conn.execute(
            "MATCH (p:PropInfo) WHERE p.name = $name RETURN p.pid, p.name",
            {"name": relation},
        )
        relations = []
        while match.has_next():
            pid, name = match.get_next()
            relations.append((int(pid), name))
        match.close()
    out: dict = {}
    for pid, prop_name in relations:
        result = conn.execute(
            "MATCH (a:Entity)-[t:Triple]->(b:Entity) WHERE t.pid = $pid "
            "RETURN a.name, b.name",
            {"pid": int(pid)},
        )
        pairs = []
        while result.has_next():
            subj, obj = result.get_next()
            pairs.append([subj, obj])
        result.close()
        # Relation key: prefer the Triple.relation label actually stored.
        rel_result = conn.execute(
            "MATCH ()-[t:Triple]->() WHERE t.pid = $pid RETURN DISTINCT t.relation LIMIT 1",
            {"pid": int(pid)},
        )
        rel_name = prop_name
        if rel_result.has_next():
            rel_name = rel_result.get_next()[0]
        rel_result.close()
        out[rel_name] = {"pid": f"P{pid}", "pairs": pairs}
    conn.close()
    return out


def pairs_to_relation_dict(pid: int, relation: str, pairs: list[list[str]]) -> dict:
    """Wrap pairs in the ``wikidata_triples.json`` relation shape."""
    return {"pid": f"P{pid}", "relation": relation, "pairs": pairs}
