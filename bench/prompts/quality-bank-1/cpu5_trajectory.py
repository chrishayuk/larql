#!/usr/bin/env python3
"""CPU-5 trajectory gate: greedy continuations, arm against reference.

    python3 cpu5_trajectory.py <container> <out.json> --tokens a,b,c -n 256 \
        [--arms shipped,q4xq8b,...]

Runs one greedy continuation per arm from the same prompt and reports
where each arm first departs from the reference's ids.

**This is a confirmatory gate, not the primary one.** A free-running
continuation lets the arms wander apart, so a divergence at step 30 does
not say the representation is bad at step 30 — it says the arms stopped
sharing a context somewhere at or before it, and every later token is
scored against a prefix the reference never saw. The bank is teacher
forced precisely so that it does not have this property.

`48/48 ids identical` is the WEAKEST evidence this programme collects and
must never be its headline.
"""
import subprocess, sys, os, json

LARQL = os.environ.get("LARQL", "./target/release/larql")


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def run(container, tokens, n, arm):
    env = dict(os.environ)
    env.pop("LARQL_CPU_ARITHMETIC", None)
    env.pop("LARQL_CPU_MAX_FORMAT", None)
    if arm == "reference":
        env["LARQL_CPU_MAX_FORMAT"] = "bf16"
    elif arm != "shipped":
        env["LARQL_CPU_ARITHMETIC"] = arm
    cmd = [LARQL, "vindex3", "exec", container, "--tokens", tokens,
           "--backend", "production", "--generate", str(n)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"arm {arm} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    for line in r.stdout.splitlines():
        if line.startswith("generated ids:"):
            return [int(t) for t in line.split(":", 1)[1].replace(" ", "").split(",") if t]
    raise SystemExit(f"arm {arm} printed no generated ids")


def main():
    container, out = sys.argv[1], sys.argv[2]
    tokens = arg("--tokens", "760,6511,314,9338,369")
    n = int(arg("-n", "256"))
    arms = arg("--arms", "reference,shipped,bf16xq8b,q8xq8b,q4xq8b").split(",")

    results = {}
    for a in arms:
        ids = run(container, tokens, n, a)
        results[a] = ids
        print(f"{a:<12} {len(ids)} ids")

    ref = results[arms[0]]
    print(f"\n{'arm':<12}{'agree':>8}{'first div':>11}{'prefix':>9}")
    rows = []
    for a in arms:
        ids = results[a]
        m = min(len(ids), len(ids))
        first = next((i for i in range(min(len(ref), len(ids))) if ref[i] != ids[i]), None)
        agree = sum(1 for i in range(min(len(ref), len(ids))) if ref[i] == ids[i])
        pref = first if first is not None else min(len(ref), len(ids))
        rows.append({"arm": a, "agree": agree, "first_divergence": first, "prefix": pref})
        fd = "none" if first is None else str(first)
        print(f"{a:<12}{agree:>8}{fd:>11}{pref:>9}")

    json.dump({"tokens": tokens, "n": n, "ids": results, "summary": rows},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
