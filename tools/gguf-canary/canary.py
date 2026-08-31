#!/usr/bin/env python3
"""Cross-runtime logit canary: VINDEX3 execution vs llama.cpp on one
frozen token sequence.

Two gates, deliberately narrow:

  TOKENIZATION  both runtimes tokenize the fixed prompt to the same ids
                (checked upstream; the ids recorded here are frozen)
  EXECUTION     full logits at every position, compared per position:
                top-1 match, top-5 overlap, cosine, max|dlogit|, RMS,
                KL(A||B) after softmax

Bit-identical logits are NOT the bar — different kernels and
accumulation orders move low bits legitimately. The bar is agreement of
the distributions within the declared tolerances with no systematic
divergence.

    python3 canary.py A.f32 B.f32 --ids 1,2,3 --vocab N \
        [--label-a vindex] [--label-b llama.cpp] [--record out.json ...]
"""
import json
import sys

import numpy as np

TOL = {
    "top1_agreement": 1.0,      # every position, exactly
    "top5_overlap_min": 0.8,    # >= 4 of 5 shared at every position
    "kl_max": 0.05,             # nats, per position
    "cosine_min": 0.995,
}


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def main(argv):
    a_path, b_path = argv[0], argv[1]
    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default
    ids = [int(x) for x in opt("--ids", "").split(",") if x]
    vocab = int(opt("--vocab"))
    label_a = opt("--label-a", "A")
    label_b = opt("--label-b", "B")

    A = np.fromfile(a_path, dtype=np.float32).astype(np.float64).reshape(-1, vocab)
    B = np.fromfile(b_path, dtype=np.float32).astype(np.float64).reshape(-1, vocab)
    if A.shape != B.shape:
        raise SystemExit(f"shape mismatch: {A.shape} vs {B.shape}")
    n = A.shape[0]
    if ids and len(ids) != n:
        raise SystemExit(f"{len(ids)} ids but {n} positions")

    rows, agree1, overlap5s, kls, cosines, maxds, rmss = [], 0, [], [], [], [], []
    for i in range(n):
        a, b = A[i], B[i]
        ta, tb = int(a.argmax()), int(b.argmax())
        top5a = set(np.argpartition(a, -5)[-5:].tolist())
        top5b = set(np.argpartition(b, -5)[-5:].tolist())
        ov = len(top5a & top5b) / 5.0
        pa, pb = softmax(a), softmax(b)
        kl = float((pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum())
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        d = a - b
        maxd, rms = float(np.abs(d).max()), float(np.sqrt((d * d).mean()))
        agree1 += ta == tb
        overlap5s.append(ov); kls.append(kl); cosines.append(cos)
        maxds.append(maxd); rmss.append(rms)
        rows.append(dict(pos=i, top1_a=ta, top1_b=tb, top1_match=ta == tb,
                         top5_overlap=ov, cosine=cos, kl=kl,
                         max_abs_dlogit=maxd, rms_dlogit=rms))

    summary = dict(
        positions=n,
        top1_agreement=agree1 / n,
        top5_overlap_mean=float(np.mean(overlap5s)),
        top5_overlap_min=float(np.min(overlap5s)),
        cosine_mean=float(np.mean(cosines)),
        cosine_min=float(np.min(cosines)),
        kl_mean=float(np.mean(kls)),
        kl_max=float(np.max(kls)),
        max_abs_dlogit=float(np.max(maxds)),
        rms_dlogit_mean=float(np.mean(rmss)),
    )
    passed = (
        summary["top1_agreement"] >= TOL["top1_agreement"]
        and summary["top5_overlap_min"] >= TOL["top5_overlap_min"]
        and summary["kl_max"] <= TOL["kl_max"]
        and summary["cosine_min"] >= TOL["cosine_min"]
    )

    print(f"\nEXECUTION   {label_a} vs {label_b}")
    print(f"  positions compared      {n}")
    print(f"  top-1 agreement         {summary['top1_agreement']*100:.0f}%")
    print(f"  top-5 overlap           mean {summary['top5_overlap_mean']*100:.0f}%  min {summary['top5_overlap_min']*100:.0f}%")
    print(f"  cosine                  mean {summary['cosine_mean']:.6f}  min {summary['cosine_min']:.6f}")
    print(f"  KL(A||B)                mean {summary['kl_mean']:.5f}  max {summary['kl_max']:.5f} nats")
    print(f"  max |dlogit|            {summary['max_abs_dlogit']:.4f}")
    print(f"  RMS dlogit              {summary['rms_dlogit_mean']:.4f}")
    print(f"\n  per position:")
    print(f"  pos  top1(A)  top1(B)  match  top5  cosine    KL        max|d|")
    for r in rows:
        print(f"  {r['pos']:>3}  {r['top1_a']:>7}  {r['top1_b']:>7}  {'yes' if r['top1_match'] else 'NO ':>5}"
              f"  {r['top5_overlap']*100:>3.0f}%  {r['cosine']:.6f}  {r['kl']:.6f}  {r['max_abs_dlogit']:.4f}")
    print(f"\n  verdict: {'PASS' if passed else 'FAIL'} (tolerances: {TOL})")

    record_path = opt("--record")
    if record_path:
        record = dict(
            prompt=opt("--prompt"),
            token_ids=ids,
            vocab=vocab,
            arm_a=dict(label=label_a, source=opt("--source-a"), dump=a_path),
            arm_b=dict(label=label_b, source=opt("--source-b"), dump=b_path),
            hashes=json.loads(opt("--hashes", "{}")),
            tolerances=TOL,
            summary=summary,
            positions=rows,
            verdict="PASS" if passed else "FAIL",
        )
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  recorded: {record_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
