#!/usr/bin/env python3
"""Fetch Wikidata property labels for use as relation type categories.

Queries the Wikidata API for property labels (P1-P2000) and filters to
semantic relation labels suitable for knowledge graph edge types.

Output: data/wikidata_categories.json — a sorted list of category strings.

Usage:
    python3 scripts/fetch_wikidata_properties.py
"""

import json
import sys
import urllib.request
from pathlib import Path

# Properties to skip (media, technical, non-semantic)
SKIP_PREFIXES = {
    "image", "video", "audio", "logo", "icon", "photo", "banner",
    "commons", "wikimedia", "wikidata", "wikipedia",
    "url", "uri", "api", "schema", "format",
    "route map", "traffic sign", "flag image", "coat of arms",
    "page banner", "locator map", "collage image",
    # IDs and codes
    " id", " code", " number", "identifier", " link",
    " rank", " index", " key",
    # Technical
    "bibcode", "digital library", "allmusic", "allmovie", "allociné",
    "discogs", "musicbrainz", "imdb", "orcid", "isni", "viaf",
    "register of", "catalogue", "classification",
}

def should_skip(label: str) -> bool:
    """Skip media references, technical properties, and non-semantic labels."""
    lower = label.lower()
    for prefix in SKIP_PREFIXES:
        if prefix in lower:
            return True
    # Skip very long labels and compound phrases with 5+ words
    if len(label) > 40:
        return True
    if len(label.split()) > 4:
        return True
    # Skip labels with parenthetical qualifiers
    if "(" in label:
        return True
    return False


def fetch_properties(max_id: int = 2000, batch_size: int = 50) -> dict:
    """Fetch English labels for Wikidata properties P1..P{max_id}."""
    all_props = {}
    for batch_start in range(1, max_id + 1, batch_size):
        batch_end = min(batch_start + batch_size, max_id + 1)
        ids = "|".join(f"P{i}" for i in range(batch_start, batch_end))
        url = (
            f"https://www.wikidata.org/w/api.php?"
            f"action=wbgetentities&ids={ids}&props=labels&languages=en&format=json"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "larql/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                for pid, entity in data.get("entities", {}).items():
                    if "labels" in entity and "en" in entity["labels"]:
                        label = entity["labels"]["en"]["value"]
                        if not should_skip(label):
                            all_props[pid] = label
        except Exception as e:
            print(f"  Warning: batch {batch_start}-{batch_end}: {e}", file=sys.stderr)
            continue

        if batch_start % 200 == 1:
            print(f"  Fetched P{batch_start}-P{batch_end}... ({len(all_props)} so far)")

    return all_props


def fetch_properties_from_lbdb(source: str | None = None) -> dict:
    """Fetch property labels from a local wikidata.lbdb (no network).

    Returns the same ``{pid: label}`` map as :func:`fetch_properties`, read
    from the ``edge_meta`` node table via the ladybug Python bindings.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    import wikidata_lbdb

    db_path = wikidata_lbdb.find_source_db(source)
    print(f"Using local Ladybug source: {db_path}")
    conn = wikidata_lbdb.connect_readonly(db_path)
    all_props = {f"P{pid}": name for pid, name in wikidata_lbdb.list_properties(conn)}
    return all_props


def main() -> None:
    """Fetch Wikidata property labels and save as category list."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Wikidata property labels (API, or fast local Ladybug path)."
    )
    parser.add_argument(
        "--from-lbdb",
        default=None,
        help="Read property labels from a local wikidata.lbdb via ladybug "
        "instead of the network API.",
    )
    args = parser.parse_args()

    if args.from_lbdb is not None:
        print("Fetching Wikidata property labels from local Ladybug db...")
        props = fetch_properties_from_lbdb(args.from_lbdb)
    else:
        print("Fetching Wikidata property labels...")
        props = fetch_properties(max_id=2000)
    print(f"Got {len(props)} semantic properties")

    # Extract unique labels, lowercased and deduplicated
    labels = sorted(set(v.lower() for v in props.values()))

    # Save as JSON list
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "wikidata_categories.json"
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Saved {len(labels)} categories to {output_path}")

    # Also save the full property map for reference
    props_path = output_dir / "wikidata_properties.json"
    with open(props_path, "w") as f:
        json.dump(props, f, indent=2, sort_keys=True)

    print(f"Saved full property map to {props_path}")

    # Show some examples
    print("\nExamples:")
    for label in labels[:30]:
        print(f"  {label}")
    print(f"  ... ({len(labels)} total)")


if __name__ == "__main__":
    main()
