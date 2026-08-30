#!/usr/bin/env python3
"""Q-BANK-1 runner: characterise a compiled representation against BF16.

    python3 run_bank.py reference <container> <tokenizer.json> <out-dir> [--backend metal]
    python3 run_bank.py compare   <container> <out-dir> [--backend ... --source stored] [--keep]
    python3 run_bank.py report    <out-dir>

`reference` runs the BF16 arm once and banks its logits with the model
identity and per-representation digests. Every later candidate compares
against that bank without re-running BF16 — which is what makes the
canonical container expendable afterwards.
"""
import json, os, subprocess, sys, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LARQL = os.environ.get("LARQL", "./target/release/larql")


# Which bank's prompts to run. Defaults to this directory's own, so
# existing invocations are unchanged.
#
# **This existed as a silent hazard.** `reference` writes wherever it is
# told but READ its prompts from HERE, so pointing the output at a
# quality-bank-2 directory produced a bank-2-shaped run containing
# BANK-1 prompts — destroying exactly the independence the second bank
# exists to provide, without any error. The bank a run scores is now
# stated rather than implied.
BANK_DIR = os.environ.get("QBANK_DIR", HERE)


def compiler_provenance():
    """Which build produced the representation under test.

    The container's digests pin the *artifact* exactly; they say nothing
    about the code that compiled it. Those are two different facts, and
    a bank that records only the first cannot answer "was this measured
    before or after the role-classifier fix?" — a question that changed
    Qwen3.8's decoder from 6.4138 to 4.5124 bits/weight without touching
    a single byte of the source checkpoint.

    `dirty` is reported rather than hidden. A result measured against an
    uncommitted tree is still a result; it is just not one anybody else
    can reproduce from a SHA, and saying so is the difference between
    provenance and decoration.
    """
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=HERE, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": head,
        "dirty": bool(status) if status is not None else None,
        "describe": git("describe", "--always", "--dirty"),
    }


def read_execution_facts(stdout):
    """What the run actually executed, from the executor's own report.

    `larql vindex3 exec` prints `representation: <enc>  source: <src>
    objects from a compiled pack: A/B` **only when the backend asked for
    a compiled encoding at all**. Its absence therefore means the run
    bound the container's canonical bytes — which is the fact that has
    to be recorded, because it is invisible in the numbers until they
    come out as an exact zero.
    """
    facts = {"executed": None, "source": None, "objects_from_pack": None,
             "objects_total": None, "runtime_compiled": 0}
    for line in stdout.splitlines():
        if line.startswith("runtime compile:"):
            facts["runtime_compiled"] = int(line.split(":")[1].strip().split()[0])
        elif line.startswith("representation:"):
            # representation: NVFP4  source: stored  objects from a compiled pack: 1/5
            body = line.split("representation:", 1)[1]
            facts["executed"] = body.split()[0]
            if "source:" in body:
                facts["source"] = body.split("source:", 1)[1].split()[0]
            if "pack:" in body:
                ratio = body.rsplit("pack:", 1)[1].strip()
                if "/" in ratio:
                    a, b = ratio.split("/", 1)
                    facts["objects_from_pack"], facts["objects_total"] = int(a), int(b)
    return facts


def assert_candidate_executed_its_representation(container_id, facts, label):
    """A bank must never be able to compare the reference against itself.

    This exists because it happened. A candidate compiled to NVFP4 was
    run on a backend that requests no compiled encoding, so it bound the
    canonical BF16 bytes and scored KL 0.00000 on all 1,740 positions —
    a result indistinguishable from perfect fidelity, and entirely an
    artefact of the invocation. `--representation-source stored` did not
    catch it: that flag forbids *manufacturing* a representation, and
    binding canonical bytes manufactures nothing.

    So the artifact's declared programme is checked against what the
    executor says it ran. An arm that cannot prove it executed the
    representation under test is not evidence.
    """
    declared = (container_id or {}).get("precision_map")
    if not declared:
        return
    want = declared.get("encoding")
    got = facts.get("executed")
    if got != want:
        raise SystemExit(
            f"{label}: the candidate's precision map declares `{want}`, but the run "
            f"executed {got or 'the canonical representation'}. A candidate that did "
            f"not execute its own representation cannot be compared against the "
            f"reference — that is a reference-against-reference run, and it will look "
            f"like perfect fidelity. Choose a backend that requests `{want}`."
        )
    if not facts.get("objects_from_pack"):
        raise SystemExit(
            f"{label}: reported `{want}` but bound no object from a compiled pack."
        )


def load_bank():
    path = os.path.join(BANK_DIR, "prompts.json")
    bank = json.load(open(path))
    # Say which bank ran. A result that cannot name its own prompt set is
    # not evidence about anything.
    print(f"bank: {bank.get('bank', '?')}  ({len(bank['prompts'])} prompts, {path})")
    return bank


def tokenize(tokenizer_path, prompts, limit):
    import tokenizers
    tk = tokenizers.Tokenizer.from_file(tokenizer_path)
    out = []
    for p in prompts:
        ids = tk.encode(p["text"]).ids[:limit]
        # Two positions is the minimum that yields one scored transition.
        if len(ids) >= 3:
            out.append({**p, "ids": ids})
    return out


def container_identity(container):
    idx = json.load(open(os.path.join(container, "index.json")))
    digests = {k: v.get("payload_sha256", "") for k, v in idx.get("representations", {}).items()}
    return {
        "model": idx.get("model"),
        "authority": idx.get("authority", "canonical"),
        "representations": digests,
        "payload_bytes": sum(v.get("payload_bytes", 0) for v in idx.get("representations", {}).values()),
        # The programme the compiler was asked to produce, as the
        # container itself records it: `{name, encoding, roles}`, where
        # `roles` are the ones compiled and every other role is
        # preserved. The SHA says which compiler ran; this says what it
        # was asked for, and a result is only independently intelligible
        # with both. Two candidates differing by one role name here is
        # exactly what makes a controlled comparison legible later.
        "precision_map": idx.get("precision_map"),
    }


def run_bank_arm(container, entries, backend, source, dump_dir):
    """One resident model, every entry. Q-BANK-2.

    Proven bitwise interchangeable with the process-per-prompt path
    (69/69 on Granite), so results from either are comparable — but a
    Glimmer sweep is only affordable this way.
    """
    manifest = os.path.join(dump_dir, "_entries.jsonl")
    os.makedirs(dump_dir, exist_ok=True)
    with open(manifest, "w") as f:
        for e in entries:
            f.write(json.dumps({"id": e["id"], "ids": e["ids"]}) + "\n")
    cmd = [LARQL, "vindex3", "exec", container, "--tokens", "1",
           "--backend", backend, "--bank", manifest, "--dump-dir", dump_dir]
    if source:
        cmd += ["--representation-source", source]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"bank run failed:\n{r.stdout}\n{r.stderr}")
    return read_execution_facts(r.stdout)


def run_arm(container, entry, backend, source, dump):
    cmd = [LARQL, "vindex3", "exec", container,
           "--tokens", ",".join(map(str, entry["ids"])),
           "--backend", backend, "--logit-dump", dump]
    if source:
        cmd += ["--representation-source", source]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{entry['id']}: {r.stdout}\n{r.stderr}")
    return read_execution_facts(r.stdout)


def softmax_rows(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def cmd_reference(container, tokenizer, outdir, backend, limit):
    os.makedirs(outdir, exist_ok=True)
    bank = load_bank()
    entries = tokenize(tokenizer, bank["prompts"], limit)
    meta = {"arm": "reference", "backend": backend, "container": container_identity(container),
            "bank": bank["bank"], "bank_dir": BANK_DIR, "entries": []}
    refdir = os.path.join(outdir, "ref")
    run_bank_arm(container, entries, backend, None, refdir)
    for e in entries:
        dump = os.path.join(refdir, f"{e['id']}.f32")
        meta["entries"].append({"id": e["id"], "category": e["category"],
                                "ids": e["ids"], "dump": os.path.relpath(dump, outdir)})
    json.dump(meta, open(os.path.join(outdir, "reference.json"), "w"), indent=1)
    print(f"banked {len(entries)} references -> {outdir}")


def cmd_compare(container, outdir, backend, source, label, keep=False):
    meta = json.load(open(os.path.join(outdir, "reference.json")))
    rows = []
    canddir = os.path.join(outdir, f"_cand-{label}")
    facts = run_bank_arm(container, meta["entries"], backend, source, canddir)
    cand_id = container_identity(container)
    assert_candidate_executed_its_representation(cand_id, facts, label)
    compiled_total = facts["runtime_compiled"]
    for i, e in enumerate(meta["entries"]):
        refpath = os.path.join(outdir, e["dump"])
        candpath = os.path.join(canddir, f"{e['id']}.f32")
        # **A truncated dump must never reach a KL number.** A run that
        # was interrupted, or that raced a second writer, leaves a short
        # file; reshaping it raises somewhere far from the cause, and a
        # dump that happened to be a whole number of positions short
        # would not raise at all — it would silently score fewer
        # positions and report a mean over them.
        for path, what in ((refpath, "reference"), (candpath, "candidate")):
            if not os.path.exists(path):
                raise SystemExit(f"REFUSED: {what} dump missing for `{e['id']}`: {path}")
        want, got = os.path.getsize(refpath), os.path.getsize(candpath)
        if want != got:
            raise SystemExit(
                f"REFUSED: candidate dump for `{e['id']}` is {got} bytes against the "
                f"reference's {want}. The run was interrupted or two writers shared the "
                f"dump directory; regenerate this entry rather than scoring it.")
        ref = np.fromfile(refpath, dtype=np.float32)
        n = len(e["ids"])
        vocab = ref.size // n
        ref = ref.reshape(n, vocab).astype(np.float64)
        cand = np.fromfile(candpath, dtype=np.float32).reshape(n, vocab).astype(np.float64)

        P, Q = softmax_rows(ref), softmax_rows(cand)
        eps = 1e-12
        kl = (P * (np.log(P + eps) - np.log(Q + eps))).sum(1) / np.log(2)
        ent = -(P * np.log(P + eps)).sum(1) / np.log(2)
        srt = np.sort(P, 1)
        margin = srt[:, -1] - srt[:, -2]
        a1, b1 = ref.argmax(1), cand.argmax(1)
        t5r = np.argsort(-ref, 1)[:, :5]
        t5c = np.argsort(-cand, 1)[:, :5]
        ov = np.array([len(set(t5r[j]) & set(t5c[j])) / 5 for j in range(n)])
        nxt = e["ids"][1:]
        m = len(nxt)
        dnll = np.array([-np.log2(Q[j, nxt[j]] + eps) + np.log2(P[j, nxt[j]] + eps)
                         for j in range(m)])
        for j in range(n):
            rows.append({
                "id": e["id"], "category": e["category"], "pos": j,
                "kl": float(kl[j]), "entropy": float(ent[j]), "margin": float(margin[j]),
                "flip": bool(a1[j] != b1[j]), "top5": float(ov[j]),
                "dmax": float(np.abs(ref[j] - cand[j]).max()),
                "dmean": float(np.abs(ref[j] - cand[j]).mean()),
                "dnll": float(dnll[j]) if j < m else None,
            })
    # Dumps are removed by default: a sweep that kept every arm's logits
    # would hold gigabytes per arm. `--keep` retains them for analyses
    # that need the ERROR VECTORS rather than the summary statistics —
    # notably testing whether two perturbations interact, which a
    # per-position KL cannot answer.
    if not keep:
        import shutil
        shutil.rmtree(canddir, ignore_errors=True)
    ref_bytes = meta["container"].get("payload_bytes", 0)
    cand_bytes = container_identity(container).get("payload_bytes", 0)
    out = {"label": label, "backend": backend, "source": source,
           "payload_bytes": cand_bytes,
           "runtime_compiled_total": compiled_total,
           "container": cand_id,
           "execution": facts,
           "compiler": compiler_provenance(),
           "reference": meta["container"], "rows": rows}
    path = os.path.join(outdir, f"compare-{label}.json")
    json.dump(out, open(path, "w"))
    print(f"wrote {path}  ({len(rows)} positions, runtime compile {compiled_total})")


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def cmd_report(outdir, label):
    d = json.load(open(os.path.join(outdir, f"compare-{label}.json")))
    rows = d["rows"]
    kl = np.array([r["kl"] for r in rows])
    ent = np.array([r["entropy"] for r in rows])
    mar = np.array([r["margin"] for r in rows])
    flips = np.array([r["flip"] for r in rows])
    top5 = np.array([r["top5"] for r in rows])
    dnll = np.array([r["dnll"] for r in rows if r["dnll"] is not None])
    dmax = np.array([r["dmax"] for r in rows])

    print(f"\nQ-BANK-1 — {d['label']}")
    print(f"  reference model  {d['reference']['model']}")
    print(f"  candidate        {d['container']['model']}  ({d['container']['authority']})")
    print(f"  backend/source   {d['backend']} / {d['source']}")
    print(f"  runtime compile  {d['runtime_compiled_total']} tensor(s)"
          + ("   <- INVARIANT VIOLATED" if d["source"] == "stored" and d["runtime_compiled_total"] else ""))
    print("=" * 66)
    print(f"  positions              {len(rows):,}   prompts {len({r['id'] for r in rows})}")
    print()
    print("  KL bits/token          mean {:.5f}  median {:.5f}".format(kl.mean(), q(kl, 50)))
    print("                         p95  {:.5f}  p99 {:.5f}  max {:.5f}".format(q(kl, 95), q(kl, 99), kl.max()))
    print("  dNLL bits              mean {:+.5f}  p95 {:+.5f}  max {:+.5f}".format(dnll.mean(), q(dnll, 95), dnll.max()))
    print("  max |dlogit|           mean {:.4f}  p99 {:.4f}".format(dmax.mean(), q(dmax, 99)))
    print()
    print("  top-1 agreement        {:.2f}%   ({} flips)".format(100 * (1 - flips.mean()), int(flips.sum())))
    print("  top-5 overlap          {:.2f}%".format(100 * top5.mean()))
    if flips.any():
        fm = mar[flips]
        low = int((fm < 0.01).sum())
        print("    flips where BF16 margin < 0.01   {}".format(low))
        print("    flips where BF16 margin >= 0.01  {}".format(int(flips.sum()) - low))
        print("    flip margin  median {:.5f}  max {:.5f}".format(q(fm, 50), fm.max()))
    print()
    print("  BF16 entropy bits      mean {:.3f}  median {:.3f}".format(ent.mean(), q(ent, 50)))
    print("  BF16 top-1 margin      mean {:.3f}  median {:.3f}".format(mar.mean(), q(mar, 50)))
    print()
    print("  by category" + " " * 12 + "positions      KL mean       KL p95    flips")
    cats = sorted({r["category"] for r in rows})
    for c in cats:
        sel = [r for r in rows if r["category"] == c]
        k = np.array([r["kl"] for r in sel])
        f = sum(r["flip"] for r in sel)
        print(f"    {c:<20} {len(sel):>9,}  {k.mean():>11.5f}  {q(k,95):>11.5f}  {f:>7}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "reference":
        backend = a[a.index("--backend") + 1] if "--backend" in a else "metal"
        limit = int(a[a.index("--limit") + 1]) if "--limit" in a else 128
        cmd_reference(a[1], a[2], a[3], backend, limit)
    elif a[0] == "compare":
        backend = a[a.index("--backend") + 1] if "--backend" in a else "metal-nvfp4-no-head"
        source = a[a.index("--source") + 1] if "--source" in a else "stored"
        label = a[a.index("--label") + 1] if "--label" in a else "candidate"
        cmd_compare(a[1], a[2], backend, source, label, keep="--keep" in a)
    elif a[0] == "report":
        label = a[a.index("--label") + 1] if "--label" in a else "candidate"
        cmd_report(a[1], label)
