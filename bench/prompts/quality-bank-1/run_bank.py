#!/usr/bin/env python3
"""Q-BANK-1 runner: a client of `larql vindex3 measure` (MEASURE-PLAN-1,
docs/measure-plan-1.md). It computes no numbers of its own.

    python3 run_bank.py reference <container> <tokenizer.json> <out-dir> [--backend B] [--limit N]
    python3 run_bank.py compare   <container> <out-dir> [--backend B] [--source S] [--label L]
    python3 run_bank.py report    <out-dir> [--label L]

`reference` exports this bank's prompts as a sealed token bank, tokenised
by the reference container's own tokenizer, and records which container and
backend are the reference. It no longer banks logits. The procedure re-runs
the reference on every comparison, twice for its null arm, so the canonical
container must stay on disk.

`compare` runs the procedure: the reference arm against the candidate over
every bank sample, with its proofs (null arm, changed variable, physical
attribution, byte identity, seals, corpus). Output goes to
`<out-dir>/measure-<label>/`. `report` prints the procedure's own aggregates,
in nats.

Refused here, by name, rather than measured without a check:
- **Output directories from the legacy runner.** Their `reference.json`
  banked logits. Use `run_bank_legacy.py` to reproduce those results.
- **K-quant candidates.** The procedure cannot yet attest which execution
  arm a stored K-quant ran (in place or widened). That arrives with
  execution-scoped projection accounting.
- **Arithmetic modes chosen by environment variables.** Both arms run in
  one process. Declare the mode as a registered backend instead.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LARQL = os.environ.get("LARQL", "./target/release/larql")
BANK_DIR = os.environ.get("QBANK_DIR", HERE)

CLIENT_SCHEMA = "qbank-client/v1"
REFERENCE_FILE = "reference.json"
TOKEN_BANK_DIR = "bank"
MEASURE_PREFIX = "measure-"
DEFAULT_REFERENCE_BACKEND = "metal"
DEFAULT_CANDIDATE_BACKEND = "metal-nvfp4-no-head"
DEFAULT_SOURCE = "stored"
DEFAULT_LABEL = "candidate"
DEFAULT_LIMIT = 128
KQUANT_ENCODINGS = ("Q8_0", "Q6_K", "Q4_K")
LEGACY_FLAGS = ("--keep", "--per-prompt")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_provenance():
    """The checkout that ran this, recorded in the procedure's report."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=HERE, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    status = git("status", "--porcelain")
    return {
        "source_commit": git("rev-parse", "HEAD") or "unknown",
        "source_dirty": "unknown" if status is None else str(bool(status)).lower(),
        "client": "bench/prompts/quality-bank-1/run_bank.py",
    }


def read_reference(outdir):
    path = os.path.join(outdir, REFERENCE_FILE)
    if not os.path.exists(path):
        refuse(f"no {REFERENCE_FILE} in {outdir}; run `reference` first")
    meta = load_json(path)
    if meta.get("schema") != CLIENT_SCHEMA:
        refuse(f"{path} was written by the legacy runner, which banked logits. "
               f"Reproduce those results with run_bank_legacy.py, or run `reference` "
               f"again into a new directory.")
    return meta


def declared_encoding(container):
    index = load_json(os.path.join(container, "index.json"))
    return (index.get("precision_map") or {}).get("encoding")


def cmd_reference(container, tokenizer, outdir, backend, limit):
    own = os.path.join(container, "tokenizer.json")
    if not os.path.exists(own):
        refuse(f"{container} carries no tokenizer.json")
    if sha256_file(tokenizer) != sha256_file(own):
        refuse(f"{tokenizer} is not {container}'s own tokenizer; a bank's ids must come "
               f"from the model that scores them")
    prompts = os.path.join(BANK_DIR, "prompts.json")
    bank = load_json(prompts)
    os.makedirs(outdir, exist_ok=True)
    subprocess.run([LARQL, "vindex3", "token-bank", "export", container,
                    "--prompts", prompts, "--max-tokens", str(limit),
                    "--output", os.path.join(outdir, TOKEN_BANK_DIR)], check=True)
    meta = {"schema": CLIENT_SCHEMA, "bank": bank.get("bank"), "bank_dir": BANK_DIR,
            "reference": os.path.abspath(container), "backend": backend,
            "token_bank": TOKEN_BANK_DIR}
    with open(os.path.join(outdir, REFERENCE_FILE), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"reference {container} via {backend}; bank {meta['bank']} -> {outdir}")


def measure_command(meta, container, outdir, backend, source, label):
    manifest = load_json(os.path.join(outdir, meta["token_bank"], "manifest.json"))
    cmd = [LARQL, "vindex3", "measure",
           "--reference", meta["reference"], "--reference-backend", meta["backend"],
           "--candidate", container, "--candidate-backend", backend,
           "--candidate-source", source,
           "--bank", os.path.join(outdir, meta["token_bank"]),
           "--sequences", str(len(manifest["samples"])),
           "--label", label,
           "--output", os.path.join(outdir, MEASURE_PREFIX + label)]
    for key, value in source_provenance().items():
        cmd += ["--provenance", f"{key}={value}"]
    return cmd


def cmd_compare(container, outdir, backend, source, label):
    meta = read_reference(outdir)
    encoding = declared_encoding(container)
    if encoding in KQUANT_ENCODINGS:
        refuse(f"{container} declares a {encoding} program. The procedure cannot yet attest "
               f"whether a stored K-quant ran in place or widened; measure it once "
               f"execution-scoped projection accounting lands, or use run_bank_legacy.py.")
    subprocess.run(measure_command(meta, container, outdir, backend, source, label), check=True)


def cmd_report(outdir, label):
    path = os.path.join(outdir, MEASURE_PREFIX + label, "report.json")
    if not os.path.exists(path):
        refuse(f"no procedure report at {path}")
    report = load_json(path)
    s = report["summary"]
    print(f"\nQ-BANK-1 (MEASURE-PLAN-1) — {report['label']}")
    print(f"  reference  {report['reference']['arm']['arm']}  {report['reference']['arm']['container']}")
    print(f"  candidate  {report['candidate']['arm']['arm']}  {report['candidate']['arm']['container']}")
    print(f"  changed    {', '.join(report['facts']['changed_representations']) or 'the arm only'}")
    print(f"  units      {report['units']}")
    rows = [("all", s["all"])] + s["by_category"] + [
        (f"margin {lo:.1f}-{hi:.1f}", a) for (lo, hi), a in s["by_margin_band"]]
    print(f"  {'':<18}{'n':>7}{'KL mean':>12}{'KL p99':>12}{'top-1':>9}{'max|dl| p99':>13}")
    for name, a in rows:
        print(f"  {name:<18}{a['positions']:>7}{a['kl_mean']:>12.3e}{a['kl_p99']:>12.3e}"
              f"{100 * a['top1_agreement']:>8.2f}%{a['max_abs_delta_p99']:>13.4f}")


def flag(args, name, default):
    return args[args.index(name) + 1] if name in args else default


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    legacy = [f for f in LEGACY_FLAGS if f in argv]
    if legacy:
        refuse(f"{', '.join(legacy)} belong to the legacy runner; use run_bank_legacy.py")
    command = argv[0]
    if command == "reference":
        cmd_reference(argv[1], argv[2], argv[3], flag(argv, "--backend", DEFAULT_REFERENCE_BACKEND),
                      int(flag(argv, "--limit", DEFAULT_LIMIT)))
    elif command == "compare":
        cmd_compare(argv[1], argv[2], flag(argv, "--backend", DEFAULT_CANDIDATE_BACKEND),
                    flag(argv, "--source", DEFAULT_SOURCE), flag(argv, "--label", DEFAULT_LABEL))
    elif command == "report":
        cmd_report(argv[1], flag(argv, "--label", DEFAULT_LABEL))
    else:
        refuse(f"unknown command {command!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
