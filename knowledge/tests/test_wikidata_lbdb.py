"""Ladybug-backed Wikidata extraction: parity + performance tests.

Compares the local ``wikidata.lbdb`` columnar path (``scripts/wikidata_lbdb.py``,
opt-in ``--from-lbdb`` / ``--export-lbdb`` flags) against the pre-existing
JSON/network behaviour:

* same output shapes (``wikidata_triples.json`` relation format),
* same property catalog and label quality rules,
* subset ``.lbdb`` round-trips exactly,
* derived-db queries are orders of magnitude faster than full-graph scans,
* the SPARQL/API/dump code paths are untouched and still the default.

Skipped when ``ladybug`` is not installed or no source ``wikidata.lbdb`` is
found (set ``LBUG_WIKIDATA_PATH`` to point at one).
"""

import inspect
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

lb = pytest.importorskip("ladybug", reason="ladybug package not installed")

try:
    import wikidata_lbdb as WL

    SOURCE_DB = WL.find_source_db(None)
except Exception as exc:  # FileNotFoundError or native-lib issues
    pytest.skip(f"no wikidata.lbdb available: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def source_conn():
    conn = WL.connect_readonly(SOURCE_DB)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def counts(source_conn):
    return WL.property_counts(source_conn, [36, 30])


# ---------------------------------------------------------------------------
# Catalog parity
# ---------------------------------------------------------------------------

def test_property_catalog_covers_script_pids(source_conn):
    props = dict(WL.list_properties(source_conn))
    assert len(props) > 10_000
    assert props[36] == "capital"
    assert props[30] == "continent"
    assert props[37] == "official language"
    assert props[31] == "instance of"


def test_counts_are_positive_for_core_relations(counts):
    assert counts[36] > 1_000
    assert counts[30] > 1_000


# ---------------------------------------------------------------------------
# Pair sampling: shape + quality (same contract as the JSON path)
# ---------------------------------------------------------------------------

def test_fetch_pairs_shape_and_quality(source_conn, counts):
    pairs = WL.fetch_pairs(source_conn, 36, limit=10, count=counts[36])
    assert 0 < len(pairs) <= 10
    seen = set()
    for pair in pairs:
        assert isinstance(pair, list) and len(pair) == 2
        subj, obj = pair
        assert isinstance(subj, str) and isinstance(obj, str)
        assert WL.is_good_label(subj) and WL.is_good_label(obj)
        assert (subj, obj) not in seen
        seen.add((subj, obj))


def test_fetch_pairs_empty_for_unknown_pid(source_conn):
    assert WL.fetch_pairs(source_conn, 999999, limit=10, count=0) == []


def test_known_fact_france_paris(source_conn):
    """Entity-anchored query (ART index on name) — deterministic known answer."""
    result = source_conn.execute(
        "MATCH (s:wikidata_node)-[r:wikidata_rel]->(o:wikidata_node) "
        "WHERE s.name = 'France' AND r.property = 36 "
        "RETURN DISTINCT o.name"
    )
    objects = set()
    while result.has_next():
        objects.add(result.get_next()[0])
    result.close()
    assert "Paris" in objects


# ---------------------------------------------------------------------------
# Subset .lbdb round-trip
# ---------------------------------------------------------------------------

def test_export_roundtrip(source_conn, counts, tmp_path):
    dest = tmp_path / "subset.lbdb"
    stats = WL.export_subset(
        source_conn, dest, {36: "capital", 30: "continent"},
        max_per_relation=10, overwrite=True,
    )
    assert stats["capital"] == 10
    assert stats["continent"] == 10

    tables = WL.connect_readonly(dest)
    try:
        result = tables.execute("CALL show_tables() RETURN name")
        names = set()
        while result.has_next():
            names.add(result.get_next()[0])
        result.close()
    finally:
        tables.close()
    assert {"PropInfo", "Entity", "Triple"} <= names

    back = WL.read_subset_pairs(dest)
    assert set(back) == {"capital", "continent"}
    for relation, data in back.items():
        assert data["pid"].startswith("P")
        assert len(data["pairs"]) == 10
        for subj, obj in data["pairs"]:
            assert WL.is_good_label(subj) and WL.is_good_label(obj)


def test_subset_json_shape_matches_pipeline(tmp_path, source_conn):
    """Derived-db reads come back in the wikidata_triples.json relation shape."""
    dest = tmp_path / "shaped.lbdb"
    WL.export_subset(source_conn, dest, {36: "capital"},
                     max_per_relation=5, overwrite=True)
    back = WL.read_subset_pairs(dest, relation="capital")
    assert list(back) == ["capital"]
    assert back["capital"]["pid"] == "P36"
    assert len(back["capital"]["pairs"]) == 5


# ---------------------------------------------------------------------------
# Performance: columnar subset beats full-graph scans
# ---------------------------------------------------------------------------

def test_derived_faster_than_source(tmp_path, source_conn, counts):
    dest = tmp_path / "fast.lbdb"
    WL.export_subset(source_conn, dest, {36: "capital"},
                     max_per_relation=10, overwrite=True)

    t0 = time.time()
    source_pairs = WL.fetch_pairs(source_conn, 36, limit=10, count=counts[36])
    source_time = time.time() - t0
    assert len(source_pairs) == 10

    t0 = time.time()
    back = WL.read_subset_pairs(dest, relation="capital")
    derived_time = time.time() - t0
    assert len(back["capital"]["pairs"]) == 10

    assert derived_time < source_time, (
        f"derived {derived_time:.2f}s should beat source {source_time:.2f}s"
    )
    assert derived_time < 10.0


# ---------------------------------------------------------------------------
# The SPARQL/API/dump paths are untouched and still the default
# ---------------------------------------------------------------------------

def _script_source(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text()


def test_sparql_paths_intact():
    triples_src = _script_source("fetch_wikidata_triples.py")
    assert "query.wikidata.org" in triples_src
    assert "SERVICE wikibase:label" in triples_src
    assert "wbsearchentities" in triples_src
    assert "def _fetch_via_sparql" in triples_src
    assert "def _fetch_via_api" in triples_src
    assert len(WL.HASH_BUCKETS * [0]) == 1000  # sanity: helper constants sane

    dump_src = _script_source("ingest_wikidata_dump.py")
    assert "def run_pass_1" in dump_src
    assert "def run_pass_2" in dump_src
    assert "latest-truthy.nt.gz" in dump_src


def test_lbdb_is_opt_in_not_default():
    """No top-level ladybug import: default runs never need the package."""
    for name in (
        "fetch_wikidata_triples.py",
        "fetch_wikidata_properties.py",
        "ingest_wikidata_dump.py",
    ):
        src = _script_source(name)
        assert "import ladybug" not in src
        assert "from ladybug" not in src
        assert "--from-lbdb" in src
    helper_src = _script_source("wikidata_lbdb.py")
    assert "import ladybug" in helper_src


def test_top_properties_preserved():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import fetch_wikidata_triples as fwt

    pids = [pid for pid, _ in fwt.TOP_PROPERTIES]
    assert len(pids) > 60
    assert "P36" in pids and "P30" in pids
    assert inspect.signature(fwt.fetch_triples_for_property).parameters.keys() >= {
        "pid", "label", "limit",
    }
