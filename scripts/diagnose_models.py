#!/usr/bin/env python3
"""Multi-architecture correctness sweep using `larql shannon verify`.

For each model in the architecture matrix, run the three-engine bits/char
comparison and report PASS/FAIL. Three orthogonal axes are exercised:

1. **F32 reference**: LARQL Rust CPU forward vs HF/PyTorch and MLX
   references. Tests the safetensors → ModelWeights → forward path.

2. **Q4K Metal** (when a local vindex is available): the production GPU
   path. Tests larql-compute Metal shaders + vindex packing.

3. **Q4K CPU** (vindex without `--metal`): the CPU fallback used on
   machines without GPU support.

Run with no args for the standard matrix; pass a model id to scope to one.

Env-var workarounds for known-but-not-yet-permanently-fixed bugs are
applied per model (see docs/diagnoses/shannon-cross-engine-divergence.md).
Once those land as proper config-driven parsing, this script will stop
needing the env vars — at which point you'll know the fixes worked.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "gutenberg" / "frankenstein.txt"
DEFAULT_CORPUS_BYTES = 1024

# Expected bits/char penalty for Q4_K_M vs the F32 reference, in percent.
# Q4_K_M ships at ~30 % bpc loss for prose on instruction-tuned models;
# anything outside this band is suspicious (kernel correctness regression
# below, or upstream config drift above). Used as the PASS/FAIL band for
# the Q4K Metal row in the sweep.
Q4K_M_EXPECTED_GAP_PCT: tuple[float, float] = (15.0, 50.0)

# Pre-fix mention of RESULT_PREFIX kept in sync with shannon_score_*.py.
RESULT_PREFIX = "RESULT "


@dataclass
class TestCase:
    """One row of the diagnostic matrix."""
    name: str
    model_id: str
    family: str  # llama, gemma2, gemma3, gemma4, mistral, qwen, smollm, etc.
    env: dict = field(default_factory=dict)
    skip_hf: bool = False  # for very large models where torch CPU is too slow
    note: str = ""
    # Optional Q4K vindex paths for the GPU/CPU vindex-path correctness check.
    q4k_vindex: Optional[Path] = None
    # Expected quantization gap range (Q4K bits/char relative to F32, as
    # a percentage). Defaults to the standard Q4_K_M band; override per
    # case if a particular vindex was built with non-standard settings.
    q4k_expected_gap_pct: tuple[float, float] = Q4K_M_EXPECTED_GAP_PCT


# Architecture matrix. One small representative of each LARQL-supported
# family that has an ungated-or-locally-cached HF model. No env-var
# workarounds — the loader in `larql-models` now parses `rms_norm_eps`
# and the structured `rope_scaling` field directly, so this matrix
# exercises the production loader path with zero overrides.
#
# Architectures still not represented (no small candidate cached): mixtral,
# deepseek, deepseek_v4 (large MoEs); gpt_oss (smallest is 20B); generic
# (catch-all, tested implicitly via the others); tinymodel (internal).
MATRIX: list[TestCase] = [
    TestCase(
        name="SmolLM2-135M",
        model_id="HuggingFaceTB/SmolLM2-135M",
        family="llama (small, no SWA)",
    ),
    # GPT-2: config parsing works (n_embd / n_layer / n_head aliases land
    # in `detect/config_io.rs`), but loading raw GPT-2 safetensors needs
    # tensor-key renaming (wte → embed_tokens, wpe → position_embed, c_attn
    # → fused qkv, h.N.* → layers.N.*). The `gpt2` arch in larql-models is
    # built for the GGUF→HF normalisation path, not raw-safetensors. Skip
    # until the safetensors loader gains the GPT-2 alias map.
    TestCase(
        name="Qwen3-0.6B",
        model_id="Qwen/Qwen3-0.6B",
        family="qwen3",
    ),
    TestCase(
        name="Llama-3.2-1B",
        model_id="meta-llama/Llama-3.2-1B",
        family="llama (llama3 rope_scaling)",
    ),
    TestCase(
        name="Granite-4.0-micro",
        model_id="ibm-granite/granite-4.0-micro",
        family="granite (scaling multipliers)",
    ),
    TestCase(
        name="Granite-4.1-3B",
        model_id="ibm-granite/granite-4.1-3b",
        family="granite (4.1 dense, all 4 scalars + tied embed)",
        note="GraniteForCausalLM v4.1, hidden=2560 / 40L / 40Q / 8KV, vocab=100352, "
             "attention_multiplier=1/64, embedding_multiplier=12, logits_scaling=10, "
             "residual_multiplier=0.22. Same dense arch as 8B/30B with different scalars.",
    ),
    TestCase(
        name="Granite-4.1-8B",
        model_id="ibm-granite/granite-4.1-8b",
        family="granite (4.1 dense, larger head_dim)",
        note="GraniteForCausalLM v4.1, hidden=4096 / 40L / 32Q / 8KV, vocab=100352, "
             "attention_multiplier=1/128 (vs 1/64 on 3B), logits_scaling=16 (vs 10), "
             "embedding_multiplier=12, residual_multiplier=0.22. ~17 GB bf16.",
    ),
    TestCase(
        name="Granite-4.1-30B",
        model_id="ibm-granite/granite-4.1-30b",
        family="granite (4.1 dense, μP init)",
        note="GraniteForCausalLM v4.1, hidden=4096 / 64L / 32Q / 8KV, vocab=100352, "
             "attention_multiplier=1/128, logits_scaling=16, residual_multiplier=0.175 "
             "(vs 0.22 — μP-init scaling), rope_theta=50M (vs 10M on 3B/8B). ~60 GB bf16.",
    ),
    TestCase(
        name="Gemma-2-2B",
        model_id="google/gemma-2-2b",
        family="gemma2 (softcap, post-norms)",
    ),
    TestCase(
        name="Gemma-4-E2B-it",
        model_id="google/gemma-4-E2B-it",
        family="gemma4 (PLE, per-layer geom)",
    ),
    TestCase(
        name="StarCoder2-3B",
        model_id="bigcode/starcoder2-3b",
        family="starcoder2 (attention bias, partial RoPE)",
    ),
    TestCase(
        name="Mistral-7B-v0.1",
        model_id="mistralai/Mistral-7B-v0.1",
        family="mistral (all SWA)",
    ),
    TestCase(
        name="Gemma-3-4B-it",
        model_id="google/gemma-3-4b-it",
        family="gemma3 (mixed SWA + global)",
        q4k_vindex=REPO_ROOT / "output" / "gemma3-4b-q4k-v2.vindex",
    ),
]


def run_q4k_metal(case: TestCase, corpus: Path) -> Optional[dict]:
    """Run `larql shannon encode --vindex --metal` for the Q4K Metal path,
    parse bits/char from the AC-coded payload. Returns None if no vindex."""
    if case.q4k_vindex is None or not case.q4k_vindex.exists():
        return None
    out_path = Path(f"/tmp/diagnose_q4k_{os.getpid()}_{case.name}.bin")
    cmd = [
        str(REPO_ROOT / "target" / "release" / "larql"),
        "shannon", "encode", case.model_id,
        "--in", str(corpus),
        "--out", str(out_path),
        "--vindex", str(case.q4k_vindex),
        "--metal",
    ]
    env = dict(os.environ)
    env.update(case.env)
    start = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - start
    out_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        return {"error": proc.stderr or proc.stdout, "elapsed": elapsed}
    m = re.search(r"bits/char:\s+([\d.]+)", proc.stdout)
    if not m:
        return {"error": "no bits/char in output", "elapsed": elapsed}
    return {
        "bits_per_char": float(m.group(1)),
        "elapsed": elapsed,
    }


def parse_verify_output(text: str) -> Optional[dict]:
    """Extract the structured verify result from `shannon verify --json` stdout.

    Looks for the final `RESULT_PREFIX` line on stdout (the same prefix the
    Python reference scorers emit) and parses its JSON payload. The payload
    schema is set in `crates/larql-cli/src/commands/primary/shannon_cmd.rs`
    (`emit_verify_json`); update both sides in lockstep if it changes.

    Reshapes the engines list into a dict keyed by engine name so existing
    consumers can still do `engines["hf"]["bits_per_char"]`.
    """
    for line in reversed(text.splitlines()):
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(RESULT_PREFIX):].strip())
        except json.JSONDecodeError:
            return None
        rows = {}
        for entry in payload.get("engines", []):
            engine = entry.get("engine")
            if engine is None:
                continue
            rows[engine] = {
                "tokens": entry.get("tokens_scored", 0),
                "bits_per_token": entry.get("bits_per_token", 0.0),
                "bits_per_char": entry.get("bits_per_char", 0.0),
                "total_bits": entry.get("total_bits", 0.0),
                "elapsed_secs": entry.get("elapsed_secs", 0.0),
            }
        max_pair = payload.get("max_pair", ["", ""])
        return {
            "engines": rows,
            "max_delta_pct": payload.get("max_delta_pct", 0.0),
            "max_pair": (max_pair[0], max_pair[1]),
            "reference": payload.get("reference", ""),
            "pass": payload.get("pass", False),
        }
    return None


def run_verify(case: TestCase, corpus: Path, threshold: float, engines: str) -> dict:
    cmd = [
        str(REPO_ROOT / "target" / "release" / "larql"),
        "shannon", "verify", case.model_id,
        "--corpus", str(corpus),
        "--context", "512",
        "--stride", "256",
        "--threshold", str(threshold),
        "--engines", engines,
        "--json",
    ]
    env = dict(os.environ)
    env.update(case.env)
    start = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - start
    parsed = parse_verify_output(proc.stdout + "\n" + proc.stderr)
    return {
        "case": case,
        "exit_code": proc.returncode,
        "elapsed": elapsed,
        "parsed": parsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def normalize_corpus(src: Path, dst: Path, n_bytes: int):
    """Strip BOM + CRLF and truncate corpus to a deterministic byte count.

    A UTF-8 BOM (`\\xef\\xbb\\xbf`) at the head, if present, is removed before
    truncation so the corpus matches what `tail -c +4 | head -c N` produces
    in the docs. CRLF→LF matches Python's `read_text()` default and avoids
    the silent tokenization drift documented in scripts/README_shannon_score.md.
    """
    raw = src.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    raw = raw[:n_bytes]
    lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"")
    lf = lf[:n_bytes]
    dst.write_bytes(lf)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                   help=f"raw corpus (default: {DEFAULT_CORPUS.relative_to(REPO_ROOT)})")
    p.add_argument("--bytes", type=int, default=DEFAULT_CORPUS_BYTES,
                   help=f"bytes to score (default: {DEFAULT_CORPUS_BYTES})")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="pair-wise delta threshold in percent (default: 0.5)")
    p.add_argument("--engines", default="mlx,hf",
                   help="comma list passed to shannon verify (default: mlx,hf)")
    p.add_argument("--model", help="restrict to one model id (substring match against MATRIX names)")
    args = p.parse_args()

    binary = REPO_ROOT / "target" / "release" / "larql"
    if not binary.exists():
        sys.exit(f"missing {binary} — run `cargo build --release -p larql-cli`")

    if shutil.which(".venv/bin/python") is None and not (REPO_ROOT / ".venv" / "bin" / "python").exists():
        sys.exit("missing .venv/bin/python — needed for the MLX and HF reference scorers")

    tmp_corpus = Path(f"/tmp/diagnose_models_corpus_{os.getpid()}.txt")
    normalize_corpus(args.corpus, tmp_corpus, args.bytes)
    print(f"corpus: {tmp_corpus} ({args.bytes} bytes, CRLF-normalized from {args.corpus})")
    print(f"threshold: {args.threshold}%   engines: {args.engines}")
    print()

    selected = [c for c in MATRIX if args.model is None or args.model.lower() in c.name.lower()]
    if not selected:
        sys.exit(f"no matrix entry matches `{args.model}`")

    print("# F32 reference triangle (LARQL Rust vs HF + MLX)")
    print(f"{'name':<22} {'family':<32} {'Δ rust-vs-hf':>14} {'max Δ':>10} {'verdict':>9} {'time':>8}")
    print("-" * 100)

    results = []
    for case in selected:
        r = run_verify(case, tmp_corpus, args.threshold, args.engines)
        results.append(r)
        if r["parsed"] is None:
            print(f"{case.name:<22} {case.family:<32} {'PARSE-FAIL':>14} {'-':>10} {'ERR':>9} {r['elapsed']:>7.1f}s")
            continue
        engines = r["parsed"]["engines"]
        rust_total = engines.get("rust", {}).get("total_bits", 0)
        hf_total = engines.get("hf", {}).get("total_bits", 0)
        delta_rust_hf = (rust_total - hf_total) / max(hf_total, 1.0) * 100.0 if hf_total else 0.0
        verdict = "PASS" if r["exit_code"] == 0 else "FAIL"
        print(f"{case.name:<22} {case.family:<32} {delta_rust_hf:>13.3f}% {r['parsed']['max_delta_pct']:>9.3f}% {verdict:>9} {r['elapsed']:>7.1f}s")

    # Q4K Metal path: separate table. Only models with a local vindex run.
    q4k_cases = [c for c in selected if c.q4k_vindex and c.q4k_vindex.exists()]
    if q4k_cases:
        print()
        print("# Q4K Metal path (vs F32 reference)")
        print(f"{'name':<22} {'F32 bpc':>10} {'Q4K bpc':>10} {'Δ pct':>10} {'verdict':>9} {'time':>8}")
        print("-" * 80)
        for r, case in zip([rr for rr in results if rr["case"] in q4k_cases], q4k_cases):
            engines = (r["parsed"] or {}).get("engines", {})
            f32_bpc = engines.get("hf", {}).get("bits_per_char")
            if f32_bpc is None:
                f32_bpc = engines.get("rust", {}).get("bits_per_char")
            q4k = run_q4k_metal(case, tmp_corpus)
            if q4k is None:
                continue
            if "error" in q4k:
                print(f"{case.name:<22} {f32_bpc:>10.4f} {'ERR':>10} {'-':>10} {'FAIL':>9} {q4k['elapsed']:>7.1f}s")
                print(f"  └─ {q4k['error'].strip().splitlines()[-1] if q4k['error'] else 'no detail'}")
                continue
            gap_pct = (q4k["bits_per_char"] - f32_bpc) / f32_bpc * 100.0
            lo, hi = case.q4k_expected_gap_pct
            verdict = "PASS" if lo <= gap_pct <= hi else "FAIL"
            print(f"{case.name:<22} {f32_bpc:>10.4f} {q4k['bits_per_char']:>10.4f} {gap_pct:>9.1f}% {verdict:>9} {q4k['elapsed']:>7.1f}s")

    print()
    n_pass = sum(1 for r in results if r["exit_code"] == 0)
    n_total = len(results)
    print(f"summary: {n_pass}/{n_total} PASS at threshold {args.threshold}%")
    if any(r["case"].env for r in results):
        print()
        print("env-var workarounds applied to make these pass:")
        for r in results:
            if r["case"].env:
                env_str = " ".join(f"{k}={v}" for k, v in r["case"].env.items())
                print(f"  {r['case'].name:<22} {env_str}")
                if r["case"].note:
                    print(f"  {'':22} ({r['case'].note})")

    tmp_corpus.unlink(missing_ok=True)
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
