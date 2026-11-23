#!/usr/bin/env python3
"""Fetch Wikidata triples for relation labeling.

For the top N most-used Wikidata properties, fetches example (subject_label, object_label)
pairs. These triples are used to label relation clusters by matching the cluster's
(gate_input_token, output_token) pairs against known Wikidata relations.

Output: data/wikidata_triples.json

Usage:
    python3 scripts/fetch_wikidata_triples.py
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Top Wikidata properties by usage (manually curated from most common)
TOP_PROPERTIES = [
    ("P17", "country"),
    ("P19", "place of birth"),
    ("P20", "place of death"),
    ("P21", "sex or gender"),
    ("P27", "country of citizenship"),
    ("P30", "continent"),
    ("P31", "instance of"),
    ("P36", "capital"),
    ("P37", "official language"),
    ("P38", "currency"),
    ("P39", "position held"),
    ("P47", "shares border with"),
    ("P50", "author"),
    ("P69", "educated at"),
    ("P86", "composer"),
    ("P101", "field of work"),
    ("P103", "native language"),
    ("P106", "occupation"),
    ("P108", "employer"),
    ("P112", "founded by"),
    ("P118", "league"),
    ("P127", "owned by"),
    ("P131", "located in"),
    ("P135", "movement"),
    ("P136", "genre"),
    ("P138", "named after"),
    ("P140", "religion"),
    ("P150", "contains"),
    ("P155", "follows"),
    ("P156", "followed by"),
    ("P159", "headquarters location"),
    ("P170", "creator"),
    ("P171", "parent taxon"),
    ("P175", "performer"),
    ("P176", "manufacturer"),
    ("P178", "developer"),
    ("P180", "depicts"),
    ("P184", "doctoral advisor"),
    ("P264", "record label"),
    ("P272", "production company"),
    ("P276", "location"),
    ("P279", "subclass of"),
    ("P287", "designed by"),
    ("P355", "subsidiary"),
    ("P361", "part of"),
    ("P364", "original language"),
    ("P407", "language of work"),
    ("P449", "original broadcaster"),
    ("P463", "member of"),
    ("P495", "country of origin"),
    ("P509", "cause of death"),
    ("P527", "has part"),
    ("P530", "diplomatic relation"),
    ("P551", "residence"),
    ("P569", "date of birth"),
    ("P570", "date of death"),
    ("P571", "inception"),
    ("P607", "conflict"),
    ("P674", "characters"),
    ("P706", "located on terrain feature"),
    ("P737", "influenced by"),
    ("P740", "location of formation"),
    ("P749", "parent organization"),
    ("P800", "notable work"),
    ("P840", "narrative location"),
    ("P921", "main subject"),
    ("P937", "work location"),
    ("P1001", "applies to jurisdiction"),
    ("P1303", "instrument"),
    ("P1376", "capital of"),
    ("P1412", "languages spoken"),
]


def fetch_triples_for_property(pid: str, label: str, limit: int = 30, max_retries: int = 3) -> list:
    """Fetch example (subject_label, object_label) pairs for a Wikidata property.

    Uses Wikidata API (wbsearchentities + claims) which is faster than SPARQL
    for getting example triples. Falls back to SPARQL if needed.
    """
    # Try the faster Wikidata REST API first — get entities that have this property
    pairs = _fetch_via_api(pid, label, limit)
    if pairs:
        return pairs

    # Fallback: SPARQL with tight limit and timeout hint
    return _fetch_via_sparql(pid, label, limit, max_retries)


def _fetch_via_api(pid: str, label: str, limit: int) -> list:
    """Fetch triples via Wikidata REST API — faster than SPARQL for common properties."""
    try:
        # Get a sample of entities that have this property
        url = (
            f"https://www.wikidata.org/w/api.php?"
            f"action=query&list=backlinks&bltitle=Property:{pid}"
            f"&blnamespace=0&bllimit={min(limit, 50)}&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "larql/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        entity_ids = []
        for bl in data.get("query", {}).get("backlinks", []):
            qid = f"Q{bl['pageid']}" if "pageid" in bl else bl.get("title", "")
            if qid.startswith("Q"):
                entity_ids.append(qid)

        if not entity_ids:
            return []

        # Fetch entities with their claims for this property
        pairs = []
        batch_size = 50
        for batch_start in range(0, len(entity_ids), batch_size):
            batch = entity_ids[batch_start:batch_start + batch_size]
            ids_str = "|".join(batch)
            url = (
                f"https://www.wikidata.org/w/api.php?"
                f"action=wbgetentities&ids={ids_str}"
                f"&props=labels|claims&languages=en&format=json"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "larql/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            for qid, entity in data.get("entities", {}).items():
                # Get subject label
                s_label = entity.get("labels", {}).get("en", {}).get("value", "")
                if not s_label or (s_label.startswith("Q") and s_label[1:].isdigit()):
                    continue

                # Get object from claims
                claims = entity.get("claims", {}).get(pid, [])
                for claim in claims[:1]:  # just first claim
                    mainsnak = claim.get("mainsnak", {})
                    datavalue = mainsnak.get("datavalue", {})

                    if datavalue.get("type") == "wikibase-entityid":
                        obj_id = "Q" + str(datavalue["value"]["numeric-id"])
                        # Need to resolve object label
                        obj_label = _resolve_label(obj_id)
                        if obj_label:
                            pairs.append((s_label, obj_label))
                    elif datavalue.get("type") == "string":
                        pairs.append((s_label, datavalue["value"]))
                    elif datavalue.get("type") == "time":
                        time_val = datavalue["value"]["time"]
                        # Extract year
                        if time_val.startswith("+"):
                            year = time_val[1:5]
                            pairs.append((s_label, year))

                if len(pairs) >= limit:
                    break
            if len(pairs) >= limit:
                break

        return pairs[:limit]

    except Exception:
        return []


def _resolve_label(qid: str) -> str:
    """Resolve a Wikidata Q-ID to its English label."""
    try:
        url = (
            f"https://www.wikidata.org/w/api.php?"
            f"action=wbgetentities&ids={qid}&props=labels&languages=en&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "larql/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["entities"][qid]["labels"]["en"]["value"]
    except Exception:
        return ""


# Cache for resolved labels
_label_cache = {}

def _resolve_labels_batch(qids: list) -> dict:
    """Resolve multiple Q-IDs to labels in one API call."""
    global _label_cache
    uncached = [q for q in qids if q not in _label_cache]
    if not uncached:
        return {q: _label_cache[q] for q in qids}

    try:
        ids_str = "|".join(uncached[:50])
        url = (
            f"https://www.wikidata.org/w/api.php?"
            f"action=wbgetentities&ids={ids_str}&props=labels&languages=en&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "larql/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            for qid, entity in data.get("entities", {}).items():
                label = entity.get("labels", {}).get("en", {}).get("value", "")
                _label_cache[qid] = label
    except Exception:
        pass

    return {q: _label_cache.get(q, "") for q in qids}


def _fetch_via_sparql(pid: str, label: str, limit: int, max_retries: int) -> list:
    """Fallback: fetch via SPARQL with tight limits."""
    query = f"""
    SELECT ?sLabel ?oLabel WHERE {{
      ?s wdt:{pid} ?o .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT {limit}
    """

    url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "larql-triple-fetcher/1.0 (https://github.com/chrishayuk/chuk-larql-rs)",
    }

    data = urllib.parse.urlencode({"query": query}).encode()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                result = json.loads(resp.read())
                pairs = []
                for b in result["results"]["bindings"]:
                    s = b.get("sLabel", {}).get("value", "")
                    o = b.get("oLabel", {}).get("value", "")
                    if s.startswith("Q") and s[1:].isdigit():
                        continue
                    if o.startswith("Q") and o[1:].isdigit():
                        continue
                    if s and o:
                        pairs.append((s, o))
                return pairs
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt+1}/{max_retries} for {pid} ({label}): {e} — waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  Failed {pid} ({label}) after {max_retries} attempts: {e}", file=sys.stderr)
                return []


def main() -> None:
    """Fetch Wikidata triples for top properties and save to JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Wikidata example triples (API/SPARQL, or fast local Ladybug path)."
    )
    parser.add_argument(
        "--from-lbdb",
        default=None,
        help="Read pairs from a local wikidata.lbdb via ladybug instead of "
        "the network API/SPARQL (falls back per-property on empty).",
    )
    parser.add_argument(
        "--export-lbdb",
        default=None,
        help="Also export the fetched relations into a new small .lbdb file.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Pairs per property.")
    args = parser.parse_args()

    lbdb_conn = None
    lbdb_counts: dict[int, int] = {}
    if args.from_lbdb is not None:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).parent))
        import wikidata_lbdb

        source = wikidata_lbdb.find_source_db(args.from_lbdb)
        print(f"Using local Ladybug source: {source}")
        lbdb_conn = wikidata_lbdb.connect_readonly(source)
        pids = [int(pid[1:]) for pid, _ in TOP_PROPERTIES]
        lbdb_counts = wikidata_lbdb.property_counts(lbdb_conn, pids)

    print(f"Fetching triples for {len(TOP_PROPERTIES)} properties...")

    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "wikidata_triples.json"

    # Load existing results (resume support)
    all_triples = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                all_triples = json.load(f)
            print(f"  Loaded {len(all_triples)} existing properties (resuming)")
        except Exception:
            pass

    for i, (pid, label) in enumerate(TOP_PROPERTIES):
        # Skip if already fetched
        if label in all_triples and len(all_triples[label].get("pairs", [])) > 0:
            print(f"  {pid:6s} {label:<30s} (cached: {len(all_triples[label]['pairs'])} pairs)")
            continue

        pairs: list = []
        used_network = lbdb_conn is None
        if lbdb_conn is not None:
            import wikidata_lbdb

            pid_num = int(pid[1:])
            lbdb_pairs = wikidata_lbdb.fetch_pairs(
                lbdb_conn, pid_num, args.limit, lbdb_counts.get(pid_num)
            )
            pairs = [(s, o) for s, o in lbdb_pairs]
            if pairs:
                print(f"  {pid:6s} {label:<30s} {len(pairs)} pairs (lbdb)")
            else:
                # No entity-entity edges locally (dates, literals) — fall back.
                print(f"  {pid:6s} {label:<30s} (lbdb empty, falling back to API/SPARQL)")
                pairs = fetch_triples_for_property(pid, label, limit=args.limit)
                used_network = True
        else:
            pairs = fetch_triples_for_property(pid, label, limit=args.limit)
        if pairs:
            all_triples[label] = {
                "pid": pid,
                "pairs": [[s, o] for s, o in pairs],
            }
            print(f"  {pid:6s} {label:<30s} {len(pairs)} pairs")

            # Save after each successful fetch
            with open(output_path, "w") as f:
                json.dump(all_triples, f, indent=2, ensure_ascii=False)
        else:
            print(f"  {pid:6s} {label:<30s} (no results)")

        # Rate limit — be gentle with the endpoint (network path only;
        # the local Ladybug path needs no throttling).
        if used_network:
            time.sleep(2)

    # Final save
    with open(output_path, "w") as f:
        json.dump(all_triples, f, indent=2, ensure_ascii=False)

    total_pairs = sum(len(v["pairs"]) for v in all_triples.values())
    print(f"\nSaved {len(all_triples)} properties, {total_pairs} total pairs to {output_path}")

    # Show examples
    print("\nExamples:")
    for label, data in list(all_triples.items())[:10]:
        pairs = data["pairs"][:3]
        pair_str = ", ".join(f"{s}→{o}" for s, o in pairs)
        print(f"  {label}: {pair_str}")

    if args.export_lbdb is not None:
        import wikidata_lbdb

        pid_to_relation = {int(pid[1:]): label for pid, label in TOP_PROPERTIES}
        if lbdb_conn is None:
            source = wikidata_lbdb.find_source_db(args.from_lbdb)
            lbdb_conn = wikidata_lbdb.connect_readonly(source)
        stats = wikidata_lbdb.export_subset(
            lbdb_conn, args.export_lbdb, pid_to_relation,
            max_per_relation=args.limit, overwrite=True,
        )
        exported = sum(1 for n in stats.values() if n > 0)
        print(f"\nExported {exported}/{len(stats)} relations to {args.export_lbdb}")


if __name__ == "__main__":
    main()
