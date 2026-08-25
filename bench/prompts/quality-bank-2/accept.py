#!/usr/bin/env python3
"""Score the ONE selected candidate against quality-bank-2. Exactly once.

    python3 accept.py <container> <bank2-outdir> --sha <freeze-sha>

Bank 2 is a one-shot instrument. Its value is entirely that no candidate
was ever selected using it, so every way of accidentally spending it —
running the wrong bank, running a modified candidate, running twice and
keeping the better number — has to be refused by the tool rather than
avoided by care.

REFUSES unless all of:

    working tree clean
    candidate SOURCE (crates/) identical to the freeze SHA
    bank identity == quality-bank-2, digests unchanged
    arithmetic == Q8[64] x asymmetric-Q8[16] -> I32 -> F32
    no existing bank-2 result for this bank

and stamps every one of those into the result, so the number can name
the immutable state it validated rather than merely resembling it.
"""
import json, os, subprocess, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
BANK1 = os.path.join(HERE, "..", "quality-bank-1")

# The candidate's arithmetic, spelled out. Anything else is a different
# experiment wearing this one's name.
CANDIDATE = {
    "LARQL_CPU_ARITHMETIC": "q8xq8b",
    "LARQL_CPU_ACT_BLOCK": "16",
    "LARQL_CPU_ACT_CODE": "asymmetric",
}
LABEL = "candidate"


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=os.path.join(HERE, "..", "..", "..")).stdout.strip()


def refuse(why):
    raise SystemExit(f"REFUSED: {why}\n\nBank 2 is one-shot; it is not spent on an "
                     f"ambiguous state.")


def check_source(sha):
    if git("status", "--porcelain"):
        refuse("the working tree is dirty — the candidate is not the frozen source")
    head = git("rev-parse", "HEAD")
    # Bench tooling may move; the CANDIDATE is what must not.
    drift = git("diff", "--name-only", sha, "HEAD", "--", "crates/")
    if drift:
        refuse(f"candidate source changed since {sha[:12]}:\n  " +
               "\n  ".join(drift.splitlines()))
    return {"freeze_sha": sha, "head": head, "crates_identical_to_freeze": True}


def check_bank():
    d = json.load(open(os.path.join(HERE, "prompts.json")))
    if d.get("bank") != "quality-bank-2":
        refuse(f"bank identity is {d.get('bank')!r}, not quality-bank-2")
    subprocess.run([sys.executable, os.path.join(HERE, "freeze.py"), "verify",
                    sys.argv[1]], check=True)
    return {"bank": d["bank"], "frozen": d["frozen"], "count": d["count"],
            "text_digest": d["text_digest"]["value"],
            "token_digest": d["token_digest"]["value"]}


def check_arithmetic():
    for k, v in CANDIDATE.items():
        got = os.environ.get(k)
        if got != v:
            refuse(f"{k} is {got!r}, candidate requires {v!r}")
    for k in ("LARQL_CPU_MAX_FORMAT", "LARQL_CPU_Q4_CLASSES", "LARQL_CPU_WEIGHT_INDEX",
              "LARQL_CPU_BIT_IDENTICAL"):
        if os.environ.get(k):
            refuse(f"{k} is set — the candidate runs the default policy")
    return dict(CANDIDATE)


def main():
    container, outdir = sys.argv[1], sys.argv[2]
    sha = sys.argv[sys.argv.index("--sha") + 1]

    # CHEAP STATE CHECKS FIRST. A misconfigured run should be refused
    # in a second, not after a reference has been built — and ordering
    # the reference check first once masked every other refusal, so the
    # gates could not be shown to work at all.
    source = check_source(sha)
    arithmetic = check_arithmetic()
    bank = check_bank()

    result = os.path.join(outdir, f"compare-{LABEL}.json")
    if os.path.exists(result):
        refuse(f"{result} already exists — bank 2 has been spent. A second run "
               f"against it would make it a discovery bank retroactively.")
    ref = os.path.join(outdir, "reference.json")
    if not os.path.exists(ref):
        refuse(f"no bank-2 reference at {ref}")
    meta = json.load(open(ref))
    if meta.get("bank") != "quality-bank-2":
        refuse(f"the reference at {ref} is for {meta.get('bank')!r}")

    provenance = {
        "source": source,
        "bank": bank,
        "arithmetic": arithmetic,
        "reference_entries": len(meta["entries"]),
        "reference_positions": sum(len(e["ids"]) for e in meta["entries"]),
    }
    print(json.dumps(provenance, indent=1))

    env = dict(os.environ, QBANK_DIR=HERE)
    for cmd in (["compare", container, outdir, "--backend", "production", "--source",
                 "auto", "--label", LABEL],
                ["report", outdir, "--label", LABEL]):
        subprocess.run([sys.executable, os.path.join(BANK1, "run_bank.py"), *cmd],
                       check=True, env=env)

    out = json.load(open(result))
    out["provenance"] = provenance
    json.dump(out, open(result, "w"))
    print(f"\nstamped provenance into {result}")


if __name__ == "__main__":
    main()
