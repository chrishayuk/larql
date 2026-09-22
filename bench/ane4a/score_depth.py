"""ANE-4A1 — score one draft depth against the frozen full-depth target bank.

The question this answers is NOT "how close are the distributions" but
"would the target accept what this drafter proposes". Those come apart:
logits can move a great deal without changing the selected token, and
speculative decoding lives on exactly that asymmetry. So top-1 agreement
and target-token rank are the headline, and KL is reported as a
distribution rather than a mean.

Per-category results are kept separate. A drafter that is excellent on
factual prompts and hopeless on arithmetic is a materially different
finding from one that is uniformly mediocre, and an average hides the
difference.

**Provenance is asserted, not trusted.** A depth result is only a depth
result if the draft arm matches the bank's container, backend and format
policy. This project has already produced a "depth effect" that was a
backend effect, so the sidecar written beside each draft plane is
checked before any number is computed, and a mismatch refuses the run.

Teacher-forcing convention: position `i` holds the distribution
predicting token `i+1`. The final position has no target in the
sequence, so rank/containment statistics use positions `0..n-2` while
model-vs-model agreement uses all of them.

Usage:
    python3 score_depth.py <subset.json> <draft-dir> <depth> [out.json]
"""

import json
import os
import sys

import numpy as np

BANK = "/Users/christopherhay/chris-models/qbanks/Qwen3.8-27B/quality-bank-1"
VOCAB = 248320
MODEL_LAYERS = 64
# What the bank was produced with. Any draft compared against it must
# match, or the comparison measures the difference between these instead
# of the difference in depth.
REQUIRED_ENGINE = "production"
REQUIRED_MAX_FORMAT = "bf16"
TOP_K = (5, 20)


def load_plane(path, positions):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != positions * VOCAB:
        raise SystemExit(
            f"REFUSED: {path} holds {a.size} floats, expected "
            f"{positions} x {VOCAB} = {positions * VOCAB}"
        )
    return a.reshape(positions, VOCAB)


def check_provenance(meta_path, entry, depth):
    if not os.path.exists(meta_path):
        raise SystemExit(f"REFUSED: no provenance sidecar at {meta_path}")
    m = json.load(open(meta_path))
    problems = []
    if REQUIRED_ENGINE not in m.get("engine", ""):
        problems.append(f"engine {m.get('engine')!r} is not {REQUIRED_ENGINE}")
    if m.get("max_format_env") != REQUIRED_MAX_FORMAT:
        problems.append(
            f"LARQL_CPU_MAX_FORMAT={m.get('max_format_env')!r}, bank used {REQUIRED_MAX_FORMAT!r}"
        )
    if m.get("draft_depth") != depth:
        problems.append(f"draft_depth {m.get('draft_depth')} != {depth}")
    if m.get("executed_layers") != [0, depth]:
        problems.append(f"executed_layers {m.get('executed_layers')} != [0, {depth}]")
    if m.get("model_layers") != MODEL_LAYERS:
        problems.append(f"model_layers {m.get('model_layers')} != {MODEL_LAYERS}")
    if m.get("positions") != entry["positions"]:
        problems.append(f"positions {m.get('positions')} != {entry['positions']}")
    if m.get("vocab") != VOCAB:
        problems.append(f"vocab {m.get('vocab')} != {VOCAB}")
    if problems:
        raise SystemExit(
            f"REFUSED: {entry['id']} provenance does not match the bank:\n  "
            + "\n  ".join(problems)
        )
    return m


def log_softmax(x):
    x = x.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def score_entry(target, draft, ids):
    """Per-position statistics for one prompt."""
    positions = target.shape[0]
    lt, ld = log_softmax(target), log_softmax(draft)
    pt = np.exp(lt)
    kl = (pt * (lt - ld)).sum(axis=-1)  # KL(target || draft), nats

    t_arg = target.argmax(axis=-1)
    d_arg = draft.argmax(axis=-1)
    agree = t_arg == d_arg

    # Rank of the TARGET's chosen token in the draft's ordering: 0 means
    # the drafter would have proposed it. Computed by counting strictly
    # greater logits, which needs no sort of a 248k row.
    rows = np.arange(positions)
    t_arg_score = draft[rows, t_arg]
    t_arg_rank = (draft > t_arg_score[:, None]).sum(axis=-1)

    # And the rank of the ACTUAL next token, where the sequence has one.
    scored = positions - 1
    next_ids = np.asarray(ids[1:], dtype=np.int64)
    nt_rank_d = np.full(positions, -1, dtype=np.int64)
    nt_rank_t = np.full(positions, -1, dtype=np.int64)
    if scored > 0:
        r = np.arange(scored)
        d_s = draft[r, next_ids]
        t_s = target[r, next_ids]
        nt_rank_d[:scored] = (draft[:scored] > d_s[:, None]).sum(axis=-1)
        nt_rank_t[:scored] = (target[:scored] > t_s[:, None]).sum(axis=-1)

    return {
        "kl": kl,
        "agree": agree,
        "target_argmax_rank_in_draft": t_arg_rank,
        "next_token_rank_draft": nt_rank_d[:scored],
        "next_token_rank_target": nt_rank_t[:scored],
    }


def summarise(label, kl, agree, t_rank, nt_d, nt_t):
    out = {
        "positions": int(agree.size),
        "top1_agreement": float(agree.mean()),
        "kl_p50": float(np.percentile(kl, 50)),
        "kl_p90": float(np.percentile(kl, 90)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_mean": float(kl.mean()),
        "target_argmax_median_rank": float(np.median(t_rank)),
    }
    for k in TOP_K:
        out[f"target_argmax_in_draft_top{k}"] = float((t_rank < k).mean())
    if nt_d.size:
        out["next_token_median_rank_draft"] = float(np.median(nt_d))
        out["next_token_median_rank_target"] = float(np.median(nt_t))
    out["label"] = label
    return out


def main():
    subset_path, draft_dir, depth = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_path = sys.argv[4] if len(sys.argv) > 4 else None
    subset = json.load(open(subset_path))

    per_entry, cats = [], {}
    acc = {k: [] for k in ("kl", "agree", "t_rank", "nt_d", "nt_t")}
    for entry in subset["entries"]:
        eid, pos = entry["id"], entry["positions"]
        meta = check_provenance(os.path.join(draft_dir, f"{eid}.meta.json"), entry, depth)
        target = load_plane(os.path.join(BANK, "ref", f"{eid}.f32"), pos)
        draft = load_plane(os.path.join(draft_dir, f"{eid}.f32"), pos)
        s = score_entry(target, draft, entry["ids"])

        row = summarise(eid, s["kl"], s["agree"], s["target_argmax_rank_in_draft"],
                        s["next_token_rank_draft"], s["next_token_rank_target"])
        row["category"] = entry["category"]
        row["prepare_s"] = meta.get("prepare_s")
        row["ms_per_position"] = meta.get("ms_per_position")
        per_entry.append(row)

        for key, val in (("kl", s["kl"]), ("agree", s["agree"]),
                         ("t_rank", s["target_argmax_rank_in_draft"]),
                         ("nt_d", s["next_token_rank_draft"]),
                         ("nt_t", s["next_token_rank_target"])):
            acc[key].append(val)
        cats.setdefault(entry["category"], {k: [] for k in acc})
        for key, val in (("kl", s["kl"]), ("agree", s["agree"]),
                         ("t_rank", s["target_argmax_rank_in_draft"]),
                         ("nt_d", s["next_token_rank_draft"]),
                         ("nt_t", s["next_token_rank_target"])):
            cats[entry["category"]][key].append(val)

    def cat(d):
        return {k: np.concatenate(v) if v else np.array([]) for k, v in d.items()}

    a = cat(acc)
    overall = summarise("OVERALL", a["kl"], a["agree"], a["t_rank"], a["nt_d"], a["nt_t"])
    by_cat = []
    for name in sorted(cats):
        c = cat(cats[name])
        by_cat.append(summarise(name, c["kl"], c["agree"], c["t_rank"], c["nt_d"], c["nt_t"]))

    hdr = (f"{'':<14}{'pos':>6}{'top1':>8}{'KL p50':>10}{'KL p90':>10}"
           f"{'KL p99':>10}{'t-rank':>9}{'top5':>7}{'top20':>7}")
    print(f"\ndepth {depth} vs full-depth target — {subset['subset']}\n")
    print(hdr)

    def show(r):
        print(f"{r['label']:<14}{r['positions']:>6}{r['top1_agreement']:>8.3f}"
              f"{r['kl_p50']:>10.4f}{r['kl_p90']:>10.4f}{r['kl_p99']:>10.4f}"
              f"{r['target_argmax_median_rank']:>9.0f}"
              f"{r['target_argmax_in_draft_top5']:>7.3f}"
              f"{r['target_argmax_in_draft_top20']:>7.3f}")

    show(overall)
    print()
    for r in by_cat:
        show(r)

    doc = {"experiment": "ANE-4A1 depth ladder", "depth": depth,
           "subset": subset["subset"], "bank": BANK,
           "provenance_asserted": {"engine": REQUIRED_ENGINE,
                                   "max_format": REQUIRED_MAX_FORMAT},
           "overall": overall, "by_category": by_cat, "by_entry": per_entry}
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
