#!/usr/bin/env python3
"""REAL-EVIDENCE-1: experimental identity and repeatability of a recorded arm.

Two commands, deliberately separate, because they answer different
questions:

  assert   Is the NEW run the SAME EXPERIMENT as the historical one?
           Declared identity vs historical report vs new report vs stream
           manifest. Byte-exact on identity fields. Any difference fails.

  compare  Did the same experiment REPEAT? Metric-by-metric relative
           differences against the historical report, judged against the
           reference's own between-bank spread. Never byte equality: the
           source container, the code and the GPU reduction order differ.

Artifact integrity is not experimental identity. `assert` checks the
latter; the stream's hash and count (the former) are verified by the Rust
reader, not here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PRODUCER = "kda_q8_real measurement arm"
STREAM_FORMAT = "represent-observation-stream/v1"
BANK_MANIFEST = "manifest.json"
SCOPE_FIELDS = ("kda_q8_layers", "mla_q8_layers", "shared_q8_layers", "lm_head_q8")
IDENTITY_FIELDS = ("run", "expert_candidate", "bytes")
# The two historical full-bank KlP99 values differ by this much
# (selection 2.382e-3 vs heldout 2.598e-3); cheap->deep differences below
# it sit inside the reference's own spread. Frozen in
# docs/represent/forecasts/represent-real-evidence-1.json.
REFERENCE_KL_P99_SPREAD = 0.0907
LOGIT_METRICS = ("kl_p50", "kl_p95", "kl_p99", "max_logit_delta", "top10_changes", "top1_flips")
ROUTING_COUNTS = (
    "route_flips",
    "positions_with_route_change",
    "layers_with_route_change",
    "first_layer_with_route_change",
)
ROUTING_DISTRIBUTIONS = ("route_margin", "route_weight_mass_moved")
BANK_DISTRIBUTIONS = (
    "top1_mass_displaced",
    "top1_margin",
    "top1_candidate_margin",
    "top10_mass_displaced",
    "top10_margin",
    "top10_candidate_margin",
    "top10_rank_displacement",
)
QUANTILES = ("p50", "p95", "p99", "max")


def load(path: str) -> dict:
    with open(Path(path).expanduser()) as f:
        return json.load(f)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_scope(scope: dict) -> dict:
    """Sorted, deduplicated, raw_env dropped — the transition, not its spelling."""
    out = {}
    for k in SCOPE_FIELDS:
        v = scope[k]
        out[k] = sorted(set(v)) if isinstance(v, list) else bool(v)
    return out


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def eq(self, name: str, a, b) -> None:
        self.rows.append((name, a == b, f"{a!r}" if a == b else f"{a!r} != {b!r}"))

    def ok(self, name: str, cond: bool, detail: str) -> None:
        self.rows.append((name, cond, detail))

    def report(self) -> bool:
        width = max(len(n) for n, _, _ in self.rows)
        for name, passed, detail in self.rows:
            print(f"  {'PASS' if passed else 'FAIL'}  {name.ljust(width)}  {detail}")
        failed = [n for n, p, _ in self.rows if not p]
        print(f"{len(self.rows) - len(failed)}/{len(self.rows)} identity checks passed")
        return not failed


def cmd_assert(args: argparse.Namespace) -> int:
    declared, hist, new = load(args.declared), load(args.historical), load(args.report)
    manifest = load(args.manifest)
    c = Checks()
    # 1. The declaration IS the historical arm.
    for f in IDENTITY_FIELDS:
        c.eq(f"declared.{f} == historical.{f}", declared[f], hist[f])
    # 2. The new run IS the historical arm, on the same bank and gate.
    for f in IDENTITY_FIELDS + ("bank_manifest_sha256",):
        c.eq(f"report.{f} == historical.{f}", new[f], hist[f])
    c.eq("report.gate.id == historical.gate.id", new["gate"]["id"], hist["gate"]["id"])
    # 3. The new run resolved the declared scope, and said so itself.
    c.eq(
        "report.runtime_scope == declared.scope",
        resolved_scope(new["runtime_scope"]),
        resolved_scope(declared["scope"]),
    )
    c.eq("report.declared_identity names the declaration", Path(new["declared_identity"]).name, Path(args.declared).name)
    # 4. The stream is bound to the same transition, from the campaign loop.
    c.eq("manifest.format", manifest["format"], STREAM_FORMAT)
    c.eq("manifest.producer", manifest["producer"], PRODUCER)
    c.eq("manifest.scope == declared.scope", resolved_scope(manifest["scope"]), resolved_scope(declared["scope"]))
    c.eq("manifest.candidate_identity == declared.expert_candidate", manifest["candidate_identity"], declared["expert_candidate"])
    # 5. The stream and the report describe the same run.
    c.eq("manifest.observations == report.positions", manifest["observations"], new["positions"])
    c.eq(
        "manifest.observations == sequences x positions",
        manifest["observations"],
        manifest["sequences"] * manifest["positions_per_sequence"],
    )
    c.eq("manifest.stream_sha256 == report.stream.stream_sha256", manifest["stream_sha256"], new["stream"]["stream_sha256"])
    c.eq("manifest.code_identity == report.code_identity", manifest["code_identity"], new["code_identity"])
    c.ok("manifest.code_identity is stamped", manifest["code_identity"] != "unknown", manifest["code_identity"])
    bank_manifest = Path(manifest["bank_identity"]) / BANK_MANIFEST
    c.eq("sha256(manifest.bank_identity/manifest.json) == report.bank_manifest_sha256", sha256_of(bank_manifest), new["bank_manifest_sha256"])
    if args.sequences is not None:
        c.eq("manifest.sequences", manifest["sequences"], args.sequences)
        c.eq("manifest.sequence_order is 0..N in order", manifest["sequence_order"], list(range(args.sequences)))
    print(f"identity: {new['run']} on bank {new['bank_manifest_sha256'][:12]}")
    return 0 if c.report() else 1


def rel(a: float, b: float) -> float | None:
    if a == 0:
        return None if b == 0 else float("inf")
    return (b - a) / abs(a)


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4e}"
    return str(v)


def cmd_compare(args: argparse.Namespace) -> int:
    hist, new = load(args.historical), load(args.report)
    hb, nb = hist["bank"], new["bank"]
    rows: list[tuple[str, object, object]] = []
    rows.append(("positions", hist["positions"], new["positions"]))
    rows.append(("min_covered_mass", hist["min_covered_mass"], new["min_covered_mass"]))
    for m in LOGIT_METRICS:
        rows.append((f"logits.{m}", hb["logits"][m], nb["logits"][m]))
    for m in ROUTING_COUNTS:
        rows.append((f"routing.{m}", hb["routing"][m], nb["routing"][m]))
    for d in ROUTING_DISTRIBUTIONS:
        for q in QUANTILES:
            rows.append((f"routing.{d}.{q}", hb["routing"][d][q], nb["routing"][d][q]))
    for d in BANK_DISTRIBUTIONS:
        if d in hb and d in nb and hb[d] is not None and nb[d] is not None:
            for q in QUANTILES:
                rows.append((f"{d}.{q}", hb[d][q], nb[d][q]))
    rows.append(("wall_seconds (informational)", hist["wall_seconds"], new["wall_seconds"]))

    out = {"historical": args.historical, "reproduction": args.report, "reference_kl_p99_spread": REFERENCE_KL_P99_SPREAD, "metrics": []}
    print(f"| metric | historical | reproduction | rel diff |")
    print(f"|---|---:|---:|---:|")
    for name, a, b in rows:
        r = rel(a, b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        flag = "" if r is None or abs(r) <= REFERENCE_KL_P99_SPREAD or name.startswith("wall") else " **>spread**"
        print(f"| {name} | {fmt(a)} | {fmt(b)} | {'n/a' if r is None else f'{100*r:+.2f}%'}{flag} |")
        out["metrics"].append({"metric": name, "historical": a, "reproduction": b, "rel_diff": r})

    hf, nf = sorted(hist["verdict_failures"]), sorted(new["verdict_failures"])
    same_class = [x.split(":")[0] for x in hf] == [x.split(":")[0] for x in nf]
    print()
    print(f"verdict passed: historical {hist['verdict_passed']}, reproduction {new['verdict_passed']}")
    print(f"failing criteria: historical {[x.split(':')[0] for x in hf]}, reproduction {[x.split(':')[0] for x in nf]} -> {'SAME CLASS' if same_class else 'DIFFERENT CLASS'}")
    out["verdict"] = {"historical_failures": hf, "reproduction_failures": nf, "same_failure_class": same_class}

    hk, nk = hist["kl_by_position"], new["kl_by_position"]
    means_h = [p["mean"] for p in hk]
    means_n = [p["mean"] for p in nk]
    worst = max(abs(rel(a, b) or 0.0) for a, b in zip(means_h, means_n))
    ratio_h = means_h[0] / min(means_h)
    ratio_n = means_n[0] / min(means_n)
    print(f"kl_by_position mean: worst per-position rel diff {100*worst:.1f}%; U-shape ratio pos0/min historical {ratio_h:.2f}, reproduction {ratio_n:.2f}")
    out["kl_by_position"] = {"worst_mean_rel_diff": worst, "u_ratio_historical": ratio_h, "u_ratio_reproduction": ratio_n}
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(out, indent=2))
        print(f"written {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assert", help="same experiment as the historical arm, or fail")
    a.add_argument("--declared", required=True)
    a.add_argument("--historical", required=True)
    a.add_argument("--report", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--sequences", type=int, default=None)
    a.set_defaults(fn=cmd_assert)
    c = sub.add_parser("compare", help="repeatability against the historical report")
    c.add_argument("--historical", required=True)
    c.add_argument("--report", required=True)
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_compare)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
