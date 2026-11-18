#!/usr/bin/env python3
"""Ingest a Wikidata truthy N-Triples dump and extract (subject, object) pairs.

Parses the standard Wikidata truthy dump (latest-truthy.nt.gz) in streaming
fashion and extracts entity pairs for specified properties. Resolves entity
QIDs to English labels using rdfs:label triples found in the same dump.

Output format matches existing wikidata_triples.json:
  { "relation_name": { "pid": "P36", "pairs": [["France", "Paris"], ...] } }

One JSON file per property is written to the output directory.

Usage:
    # Extract capital and language relations, max 500 pairs each
    python3 scripts/ingest_wikidata_dump.py \\
        --dump latest-truthy.nt.gz \\
        --properties P36,P37 \\
        --max-per-relation 500 \\
        --output data/dump_triples/

    # Extract top 50 properties
    python3 scripts/ingest_wikidata_dump.py \\
        --dump latest-truthy.nt.gz \\
        --properties top500 \\
        --output data/dump_triples/

Download the dump from:
    https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Top ~50 most useful Wikidata properties for relation labeling
# ---------------------------------------------------------------------------
TOP_PROPERTIES = {
    # Geography / political
    "P36": "capital",
    "P37": "official_language",
    "P30": "continent",
    "P47": "borders",
    "P38": "currency",
    "P6": "head_of_government",
    "P35": "head_of_state",
    "P17": "country",
    "P131": "located_in",
    # People
    "P106": "occupation",
    "P19": "birthplace",
    "P20": "deathplace",
    "P27": "nationality",
    "P569": "birth_year",
    "P570": "death_year",
    "P26": "spouse",
    "P69": "alma_mater",
    "P140": "religion",
    "P166": "award",
    "P102": "party",
    "P39": "position",
    # Music
    "P136": "genre",
    "P1303": "instrument",
    "P264": "record_label",
    "P527": "band_member",
    "P175": "album_artist",
    "P86": "composer",
    # Film / TV
    "P57": "director",
    "P161": "starring",
    "P577": "film_year",
    "P272": "film_studio",
    # Literature
    "P50": "author",
    "P123": "publisher",
    # Sports
    "P54": "team",
    "P118": "league",
    "P641": "sport",
    # Companies / orgs
    "P112": "founder",
    "P159": "headquarters",
    "P169": "ceo",
    "P452": "industry",
    "P749": "parent_company",
    "P355": "subsidiary",
    # Products / science
    "P176": "manufacturer",
    "P61": "inventor_discoverer",
    "P246": "chemical_symbol",
    # Additional high-value
    "P103": "native_language",
    "P108": "employer",
    "P127": "owned_by",
    "P150": "contains",
    "P138": "named_after",
    "P495": "country_of_origin",
}

# N-Triples URI patterns
WIKIDATA_ENTITY_RE = re.compile(r"<http://www\.wikidata\.org/entity/(Q\d+)>")
WIKIDATA_PROP_RE = re.compile(
    r"<http://www\.wikidata\.org/prop/direct/(P\d+)>"
)
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
LABEL_LANG_RE = re.compile(r'"(.+)"@en\b')

# Heuristics for "clean" labels
MAX_LABEL_WORDS = 6
MAX_LABEL_LEN = 60
SKIP_PATTERNS = re.compile(
    r"Q\d{4,}|^[A-Z]{2,5}\d|^\d+$|^Category:|^Template:|^Module:"
)


def is_good_label(label: str) -> bool:
    """Return True if the label is suitable for relation triples."""
    if not label:
        return False
    if len(label) > MAX_LABEL_LEN:
        return False
    if len(label.split()) > MAX_LABEL_WORDS:
        return False
    if SKIP_PATTERNS.search(label):
        return False
    # Skip labels that are pure numbers or codes
    if label.strip().isdigit():
        return False
    return True


def parse_line(line: str):
    """Parse a single N-Triples line into (subject, predicate, object, rest).

    Returns None if the line is malformed or a comment.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # N-Triples format: <subject> <predicate> <object> .
    # Objects can be URIs or literals (with quotes and language tags)
    parts = line.split(" ", 2)
    if len(parts) < 3:
        return None

    subject = parts[0]
    predicate = parts[1]
    rest = parts[2]

    return subject, predicate, rest


def extract_entity_id(uri: str) -> str | None:
    """Extract QID from a Wikidata entity URI like <http://...entity/Q123>."""
    m = WIKIDATA_ENTITY_RE.match(uri)
    return m.group(1) if m else None


def extract_property_id(uri: str) -> str | None:
    """Extract PID from a Wikidata property URI."""
    m = WIKIDATA_PROP_RE.match(uri)
    return m.group(1) if m else None


def extract_english_label(obj_rest: str) -> str | None:
    """Extract English label from an N-Triples object+rest string."""
    m = LABEL_LANG_RE.search(obj_rest)
    return m.group(1) if m else None


def open_dump(path: str):
    """Open an .nt or .nt.gz file for line-by-line reading."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        return open(path, "r", encoding="utf-8", errors="replace")


def run_pass_1(dump_path: str, target_pids: set[str], max_per_relation: int):
    """Pass 1: collect entity-pair QIDs for target properties.

    Also collects rdfs:label mappings for all entities we encounter as
    subjects or objects in matching triples.

    Returns:
        relation_pairs: dict[pid] -> list of (subj_qid, obj_qid)
        needed_qids: set of all QIDs that need labels
    """
    relation_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    filled: set[str] = set()
    needed_qids: set[str] = set()

    t0 = time.time()
    lines_processed = 0
    pairs_found = 0

    print(f"[Pass 1] Scanning for property triples...")
    print(f"  Target properties: {len(target_pids)}")
    print(f"  Max per relation: {max_per_relation}")
    print()

    with open_dump(dump_path) as f:
        for line in f:
            lines_processed += 1

            if lines_processed % 10_000_000 == 0:
                elapsed = time.time() - t0
                rate = lines_processed / elapsed / 1e6
                print(
                    f"  {lines_processed / 1e6:.0f}M lines | "
                    f"{pairs_found} pairs | "
                    f"{rate:.1f}M lines/s | "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

            # Quick prefix check before full parse
            if "wikidata.org/prop/direct/" not in line:
                continue

            parsed = parse_line(line)
            if parsed is None:
                continue

            subject, predicate, rest = parsed

            pid = extract_property_id(predicate)
            if pid is None or pid not in target_pids or pid in filled:
                continue

            subj_qid = extract_entity_id(subject)
            if subj_qid is None:
                continue

            # Object: extract entity QID (skip literal values for now)
            # The rest might be like: <http://...entity/Q456> .
            obj_qid = None
            obj_m = WIKIDATA_ENTITY_RE.search(rest)
            if obj_m:
                obj_qid = obj_m.group(1)

            if obj_qid is None:
                # For date properties (P569, P570, P577), extract year from literal
                if pid in ("P569", "P570", "P577"):
                    year_m = re.search(r'"(\d{4})-', rest)
                    if year_m:
                        # Store year as a pseudo-QID we resolve later
                        year_str = year_m.group(1)
                        relation_pairs[pid].append((subj_qid, f"__YEAR_{year_str}"))
                        needed_qids.add(subj_qid)
                        pairs_found += 1
                        if len(relation_pairs[pid]) >= max_per_relation:
                            filled.add(pid)
                # For chemical_symbol (P246), extract literal string
                elif pid == "P246":
                    sym_m = re.search(r'"([A-Za-z]{1,3})"', rest)
                    if sym_m:
                        relation_pairs[pid].append(
                            (subj_qid, f"__LIT_{sym_m.group(1)}")
                        )
                        needed_qids.add(subj_qid)
                        pairs_found += 1
                        if len(relation_pairs[pid]) >= max_per_relation:
                            filled.add(pid)
                continue

            relation_pairs[pid].append((subj_qid, obj_qid))
            needed_qids.add(subj_qid)
            needed_qids.add(obj_qid)
            pairs_found += 1

            if len(relation_pairs[pid]) >= max_per_relation:
                filled.add(pid)

            # Early exit when all relations filled
            if len(filled) == len(target_pids):
                print(f"  All relations filled at line {lines_processed}")
                break

    elapsed = time.time() - t0
    print(f"\n[Pass 1] Done: {lines_processed / 1e6:.1f}M lines in {elapsed:.0f}s")
    print(f"  {pairs_found} total pairs across {len(relation_pairs)} properties")
    print(f"  {len(needed_qids)} unique entities need labels")

    return dict(relation_pairs), needed_qids


def run_pass_2(dump_path: str, needed_qids: set[str]) -> dict[str, str]:
    """Pass 2: resolve QID -> English label for all needed entities."""
    labels: dict[str, str] = {}
    remaining = len(needed_qids)

    t0 = time.time()
    lines_processed = 0

    print(f"\n[Pass 2] Resolving {remaining} entity labels...")

    with open_dump(dump_path) as f:
        for line in f:
            lines_processed += 1

            if lines_processed % 10_000_000 == 0:
                elapsed = time.time() - t0
                rate = lines_processed / elapsed / 1e6
                print(
                    f"  {lines_processed / 1e6:.0f}M lines | "
                    f"{len(labels)} labels resolved | "
                    f"{rate:.1f}M lines/s | "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

            # Quick check for rdfs:label
            if "rdf-schema#label" not in line:
                continue
            if "@en" not in line:
                continue

            parsed = parse_line(line)
            if parsed is None:
                continue

            subject, predicate, rest = parsed

            if RDFS_LABEL not in predicate:
                continue

            qid = extract_entity_id(subject)
            if qid is None or qid not in needed_qids or qid in labels:
                continue

            label = extract_english_label(rest)
            if label and is_good_label(label):
                labels[qid] = label
                remaining -= 1

            if remaining <= 0:
                print(f"  All labels resolved at line {lines_processed}")
                break

    elapsed = time.time() - t0
    print(f"\n[Pass 2] Done: {lines_processed / 1e6:.1f}M lines in {elapsed:.0f}s")
    print(f"  {len(labels)} labels resolved out of {len(needed_qids)} needed")

    return labels


def resolve_and_write(
    relation_pairs: dict[str, list[tuple[str, str]]],
    labels: dict[str, str],
    output_dir: str,
):
    """Resolve QIDs to labels and write output JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    summary: dict[str, int] = {}

    for pid, pairs in sorted(relation_pairs.items()):
        relation_name = TOP_PROPERTIES.get(pid, f"property_{pid}")

        resolved = []
        for subj_qid, obj_qid in pairs:
            subj_label = labels.get(subj_qid)
            if subj_label is None:
                continue

            # Handle literal pseudo-QIDs
            if obj_qid.startswith("__YEAR_"):
                obj_label = obj_qid.replace("__YEAR_", "")
            elif obj_qid.startswith("__LIT_"):
                obj_label = obj_qid.replace("__LIT_", "")
            else:
                obj_label = labels.get(obj_qid)
                if obj_label is None:
                    continue

            resolved.append([subj_label, obj_label])

        if not resolved:
            print(f"  {pid} ({relation_name}): 0 pairs (skipped)")
            continue

        out_data = {
            "pid": pid,
            "relation": relation_name,
            "count": len(resolved),
            "pairs": resolved,
        }

        out_path = Path(output_dir) / f"{relation_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

        summary[relation_name] = len(resolved)
        print(f"  {pid} ({relation_name}): {len(resolved)} pairs -> {out_path.name}")

    # Write combined file matching existing format
    combined = {}
    for pid, pairs in sorted(relation_pairs.items()):
        relation_name = TOP_PROPERTIES.get(pid, f"property_{pid}")
        resolved = []
        for subj_qid, obj_qid in pairs:
            subj_label = labels.get(subj_qid)
            if subj_label is None:
                continue
            if obj_qid.startswith("__YEAR_"):
                obj_label = obj_qid.replace("__YEAR_", "")
            elif obj_qid.startswith("__LIT_"):
                obj_label = obj_qid.replace("__LIT_", "")
            else:
                obj_label = labels.get(obj_qid)
                if obj_label is None:
                    continue
            resolved.append([subj_label, obj_label])
        if resolved:
            combined[relation_name] = {"pid": pid, "pairs": resolved}

    combined_path = Path(output_dir) / "_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\n  Combined file: {combined_path} ({len(combined)} relations)")

    return summary


def write_resolved(
    resolved: dict[str, list[list[str]]],
    target_pids: set[str],
    output_dir: str,
) -> dict[str, int]:
    """Write already-resolved ``{relation: pairs}`` in the standard output format.

    Shared by the ``--from-lbdb`` path (labels come straight from the graph,
    so the dump's Pass 1/Pass 2 + ``resolve_and_write`` are skipped). Output
    files match ``resolve_and_write`` exactly: one JSON per relation plus a
    ``_combined.json`` in the ``wikidata_triples.json`` shape.
    """
    os.makedirs(output_dir, exist_ok=True)
    pid_by_relation = {v: k for k, v in TOP_PROPERTIES.items()}
    summary: dict[str, int] = {}
    combined: dict = {}
    for relation_name in sorted(resolved):
        pairs = resolved[relation_name]
        if not pairs:
            continue
        pid = pid_by_relation.get(relation_name, "")
        if not pid:
            for cand in target_pids:
                if TOP_PROPERTIES.get(cand, f"property_{cand}") == relation_name:
                    pid = cand
                    break
        out_data = {
            "pid": pid,
            "relation": relation_name,
            "count": len(pairs),
            "pairs": pairs,
        }
        out_path = Path(output_dir) / f"{relation_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        summary[relation_name] = len(pairs)
        print(f"  {pid} ({relation_name}): {len(pairs)} pairs -> {out_path.name}")
        combined[relation_name] = {"pid": pid, "pairs": pairs}
    combined_path = Path(output_dir) / "_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\n  Combined file: {combined_path} ({len(combined)} relations)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Wikidata truthy N-Triples dump and extract relation pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specific properties
  python3 scripts/ingest_wikidata_dump.py \\
      --dump latest-truthy.nt.gz --properties P36,P37,P30

  # Top 50 properties
  python3 scripts/ingest_wikidata_dump.py \\
      --dump latest-truthy.nt.gz --properties top500
        """,
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="Path to Wikidata truthy dump (.nt or .nt.gz). Required unless --from-lbdb is given.",
    )
    parser.add_argument(
        "--from-lbdb",
        default=None,
        help="Read pairs from a local wikidata.lbdb via ladybug instead of "
        "streaming the N-Triples dump (same output format; no Pass 1/Pass 2).",
    )
    parser.add_argument(
        "--properties",
        required=True,
        help='Comma-separated PIDs (e.g. P36,P37) or "top500" for all built-in',
    )
    parser.add_argument(
        "--max-per-relation",
        type=int,
        default=1000,
        help="Maximum pairs per relation (default: 1000)",
    )
    parser.add_argument(
        "--output",
        default="data/dump_triples/",
        help="Output directory (default: data/dump_triples/)",
    )
    args = parser.parse_args()

    # Resolve properties
    if args.properties.lower() in ("top500", "top50", "all"):
        target_pids = set(TOP_PROPERTIES.keys())
        print(f"Using all {len(target_pids)} built-in properties")
    else:
        target_pids = set(p.strip() for p in args.properties.split(","))
        unknown = target_pids - set(TOP_PROPERTIES.keys())
        if unknown:
            print(f"Warning: unknown PIDs (will use generic names): {unknown}")
        print(f"Targeting {len(target_pids)} properties: {sorted(target_pids)}")

    # Verify dump exists (dump path only)
    if args.from_lbdb is None:
        if not args.dump:
            print("Error: --dump is required unless --from-lbdb is given.", file=sys.stderr)
            sys.exit(2)
        if not os.path.exists(args.dump):
            print(f"Error: dump file not found: {args.dump}", file=sys.stderr)
            sys.exit(1)

        file_size = os.path.getsize(args.dump)
        print(f"Dump file: {args.dump} ({file_size / 1e9:.1f} GB)")
        print()

        t_total = time.time()

        # Pass 1: collect QID pairs
        relation_pairs, needed_qids = run_pass_1(
            args.dump, target_pids, args.max_per_relation
        )

        if not relation_pairs:
            print("No matching triples found. Check your dump file and properties.")
            sys.exit(1)

        # Pass 2: resolve labels
        labels = run_pass_2(args.dump, needed_qids)

        # Write output
        print(f"\n[Output] Writing to {args.output}/")
        summary = resolve_and_write(relation_pairs, labels, args.output)
    else:
        import wikidata_lbdb

        t_total = time.time()
        source = wikidata_lbdb.find_source_db(args.from_lbdb)
        print(f"Using local Ladybug source: {source}")
        conn = wikidata_lbdb.connect_readonly(source)
        pid_nums = sorted(int(p[1:]) for p in target_pids if p[1:].isdigit())
        counts = wikidata_lbdb.property_counts(conn, pid_nums)
        resolved: dict[str, list[list[str]]] = {}
        for pid in sorted(target_pids):
            relation_name = TOP_PROPERTIES.get(pid, f"property_{pid}")
            pid_num = int(pid[1:]) if pid[1:].isdigit() else None
            pairs = (
                wikidata_lbdb.fetch_pairs(
                    conn, pid_num, args.max_per_relation, counts.get(pid_num, 0)
                )
                if pid_num is not None
                else []
            )
            if pairs:
                resolved[relation_name] = pairs
                print(f"  {pid} ({relation_name}): {len(pairs)} pairs")
            else:
                print(f"  {pid} ({relation_name}): 0 pairs (no entity-entity edges; skipping)")
        if not resolved:
            print("No matching triples found. Check your properties.")
            sys.exit(1)
        print(f"\n[Output] Writing to {args.output}/")
        summary = write_resolved(resolved, target_pids, args.output)

    elapsed = time.time() - t_total
    total_pairs = sum(summary.values())
    print(f"\n{'=' * 60}")
    print(f"Total: {total_pairs} pairs across {len(summary)} relations in {elapsed:.0f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
