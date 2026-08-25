#!/usr/bin/env python3
"""Run one CPU-6 bank's three arms, stamping provenance the adjudicator checks.

    python3 cpu6_run.py <container> <outdir> --bank 3a --sha <freeze-sha>

Arms, in order, so a bank's three runs sit close together in time:

    reference   BF16 exact          (LARQL_CPU_MAX_FORMAT=bf16, no arm)
    shipped     Q8 x F32 ANCHOR     (default policy, NO overrides)
    candidate   Q8[64] x asym-Q8[16]

The anchor is the repair CPU-5 needed: it is measured INSIDE the bank it
adjudicates, so nothing numerical travels between prompt sets.

Refuses a dirty tree or a candidate whose `crates/` has drifted from the
freeze, and refuses to overwrite an arm that already exists — each bank
is scored once.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BANK1 = os.path.join(HERE, "quality-bank-1")
CANDIDATE = {"LARQL_CPU_ARITHMETIC": "q8xq8b", "LARQL_CPU_ACT_BLOCK": "16",
             "LARQL_CPU_ACT_CODE": "asymmetric"}
POLICY_VARS = ["LARQL_CPU_ARITHMETIC", "LARQL_CPU_ACT_BLOCK", "LARQL_CPU_ACT_CODE",
               "LARQL_CPU_MAX_FORMAT", "LARQL_CPU_Q4_CLASSES",
               "LARQL_CPU_WEIGHT_INDEX", "LARQL_CPU_BIT_IDENTICAL"]


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True,
                          cwd=os.path.join(HERE, "..", "..")).stdout.strip()


def refuse(w):
    raise SystemExit(f"REFUSED: {w}")


def clean_env(bank_dir):
    e = {k: v for k, v in os.environ.items() if k not in POLICY_VARS}
    e["QBANK_DIR"] = bank_dir
    return e


def run(cmd, env):
    subprocess.run([sys.executable, os.path.join(BANK1, "run_bank.py"), *cmd],
                   check=True, env=env)


def main():
    container, outdir = sys.argv[1], sys.argv[2]
    bank = sys.argv[sys.argv.index("--bank") + 1]
    sha = sys.argv[sys.argv.index("--sha") + 1]
    bank_dir = os.path.join(HERE, f"quality-bank-{bank}")

    if git("status", "--porcelain"):
        refuse("working tree is dirty — the candidate is not the frozen source")
    drift = git("diff", "--name-only", sha, "HEAD", "--", "crates/")
    if drift:
        refuse(f"candidate source changed since {sha[:12]}:\n  " + "\n  ".join(drift.splitlines()))
    source = {"freeze_sha": sha, "head": git("rev-parse", "HEAD"),
              "crates_identical_to_freeze": True}

    tok = os.path.join(container, "tokenizer.json")
    ref = os.path.join(outdir, "reference.json")
    if not os.path.exists(ref):
        run(["reference", container, tok, outdir, "--backend", "production", "--limit", "128"],
            dict(clean_env(bank_dir), LARQL_CPU_MAX_FORMAT="bf16"))
    meta = json.load(open(ref))
    if meta.get("bank") != f"quality-bank-{bank}":
        refuse(f"reference is for {meta.get('bank')!r}, expected quality-bank-{bank}")

    for label, extra, prov in (
        ("shipped", {}, {"arm": "shipped"}),
        ("candidate", CANDIDATE, {"arm": "candidate", "arithmetic": CANDIDATE, "source": source}),
    ):
        out = os.path.join(outdir, f"compare-{label}.json")
        if os.path.exists(out):
            print(f"{label}: already scored, leaving it alone")
            continue
        env = clean_env(bank_dir)
        env.update(extra)
        run(["compare", container, outdir, "--backend", "production", "--source", "auto",
             "--label", label], env)
        d = json.load(open(out))
        d["provenance"] = prov
        d["bank"] = f"quality-bank-{bank}"
        json.dump(d, open(out, "w"))
        print(f"{label}: stamped provenance")


if __name__ == "__main__":
    main()
