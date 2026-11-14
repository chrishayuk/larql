#!/usr/bin/env python3
"""Hashes of the frozen protocol contracts (frozen-protocols.json).

A frozen protocol is its document and the files that hold its parameters
and decision rules. Drivers and trial binaries are implementation, not
contract: they may be corrected without a new revision, and every receipt
records the hashes of the ones that produced it.

  frozen_protocols.py --check    fail if a frozen file no longer matches
  frozen_protocols.py --write    record the current hashes (a freezing act)
"""
import hashlib
import json
from pathlib import Path
import sys

import block_time_model as bt
import link_block_frames_model as bm

RECORD = "frozen-protocols.json"
FROZEN = {
    bm.ADDENDUM: {
        "document": "LINK-2-block-frames-revision-2.md",
        "contract": ["link_block_frames_model.py", "link_pair_model.py", "link_model.py", "block_frames.json"],
    },
    bt.PROTOCOL: {
        "document": "BLOCK-TIME-1.md",
        "contract": ["block_time_model.py", "link_block_frames_model.py", "link_model.py", "profile_exact.py",
                     "block_frames.json"],
    },
}


def sha(name):
    return hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()


def current():
    return {protocol: {"document": files["document"],
                       "sha256": {name: sha(name) for name in [files["document"], *files["contract"]]}}
            for protocol, files in FROZEN.items()}


def recorded():
    return json.loads(Path(__file__).with_name(RECORD).read_text())


def changed():
    """(protocol, file) pairs whose content no longer matches what was frozen."""
    frozen, now = recorded()["protocols"], current()
    if set(frozen) != set(now):
        return [(protocol, "<protocol set>") for protocol in sorted(set(frozen) ^ set(now))]
    return [(protocol, name) for protocol in now for name in now[protocol]["sha256"]
            if frozen[protocol]["sha256"].get(name) != now[protocol]["sha256"][name]
            or set(frozen[protocol]["sha256"]) != set(now[protocol]["sha256"])]


def main():
    if sys.argv[1:] == ["--write"]:
        Path(__file__).with_name(RECORD).write_text(json.dumps({
            "frozen_on": "2026-11-14",
            "note": "A change to any file below is a new protocol revision, not an edit. Drivers and "
                    "trial binaries are not frozen; receipts record the hashes of those that ran.",
            "protocols": current()}, indent=2) + "\n")
        return
    if sys.argv[1:] != ["--check"]:
        raise SystemExit(__doc__)
    differences = changed()
    for protocol, name in differences:
        print(f"{protocol}: {name} differs from the frozen record")
    raise SystemExit(1 if differences else 0)


if __name__ == "__main__":
    main()
