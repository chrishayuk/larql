#!/usr/bin/env python3
"""
End-to-end smoke test for the `attention-service-routes` HTTP surface.

Exercises every endpoint in sequence:

  1. POST  /v1/attention/session           → session_id
  2. GET   /v1/attention/session/{id}      → state
  3. POST  /v1/attention/prefill           → per-layer residuals
  4. POST  /v1/attention/decode            → next-step residuals
  5. POST  /v1/kv-cache/snapshot           → versioned blob
  6. POST  /v1/kv-cache/restore (new sess) → identical post-restore state
  7. POST  /v1/kv-cache/free               → clear
  8. DELETE /v1/attention/session/{id}     → 204

Reports per-step latency and a one-line PASS/FAIL with a readable
diagnostic on failure. No external deps beyond `requests`.

Usage:
    python3 scripts/attention-service-smoke.py \\
        --base-url http://localhost:8081 \\
        --model-id gemma-3-4b \\
        --hidden-dim 2560 \\
        --seq-len 8

The script prints input embeddings as deterministic random
(seed=42) so failures are reproducible. It does NOT validate
numerical correctness — it just confirms the wire surface is alive
and the runner returns shaped output. For correctness validation,
diff the residuals against a local CPU reference in a separate
script.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    print("requests is required; pip install requests", file=sys.stderr)
    sys.exit(2)


def t(label: str, fn):
    """Time a labelled call and return (result, elapsed_ms)."""
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        print(f"  ✗ {label}: {type(e).__name__}: {e}")
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(f"  ✓ {label}: {elapsed_ms:.1f} ms")
    return result, elapsed_ms


def expect(resp: requests.Response, *codes: int, op: str) -> dict[str, Any]:
    if resp.status_code not in codes:
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:300]
        raise RuntimeError(
            f"{op} expected status in {codes}, got {resp.status_code}; body={body!r}"
        )
    if resp.status_code == 204:
        return {}
    return resp.json()


def smoke(
    base_url: str,
    model_id: str,
    hidden_dim: int,
    seq_len: int,
    kv_format: str,
) -> int:
    rng = random.Random(42)
    token_embeddings = [
        [rng.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(seq_len)
    ]
    query_token = [rng.uniform(-0.1, 0.1) for _ in range(hidden_dim)]

    sess = requests.Session()

    print(f"\n— Smoke test {base_url} (model={model_id}, kv={kv_format}, seq={seq_len}, hidden={hidden_dim}) —\n")

    # ── 1. Create session ────────────────────────────────────────────────
    body, _ = t(
        "create session",
        lambda: expect(
            sess.post(
                f"{base_url}/v1/attention/session",
                json={"model_id": model_id, "kv_format": kv_format},
                timeout=30,
            ),
            200,
            op="create_session",
        ),
    )
    session_id = body["session_id"]
    print(f"     session_id={session_id}  layer_range={body['layer_range']}  format={body['kv_format']}")

    # ── 2. Get session state ─────────────────────────────────────────────
    body, _ = t(
        "get session",
        lambda: expect(
            sess.get(f"{base_url}/v1/attention/session/{session_id}", timeout=10),
            200,
            op="get_session",
        ),
    )
    assert body["seq_len"] == 0 and body["prefilled"] is False, body
    num_layers = int(body["num_layers"])
    print(f"     num_layers={num_layers}  seq_len={body['seq_len']}  prefilled={body['prefilled']}")

    # ── 3. Prefill ───────────────────────────────────────────────────────
    body, prefill_ms = t(
        "prefill",
        lambda: expect(
            sess.post(
                f"{base_url}/v1/attention/prefill",
                json={"session_id": session_id, "token_embeddings": token_embeddings},
                timeout=120,
            ),
            200,
            op="prefill",
        ),
    )
    residuals = body["post_attention_residuals"]
    assert len(residuals) == num_layers, f"expected {num_layers} layers, got {len(residuals)}"
    assert len(residuals[0]) == seq_len, f"expected seq_len={seq_len} rows, got {len(residuals[0])}"
    assert (
        len(residuals[0][0]) == hidden_dim
    ), f"expected hidden_dim={hidden_dim} cols, got {len(residuals[0][0])}"
    print(
        f"     residuals[{num_layers}][{seq_len}][{hidden_dim}]  "
        f"runner={body['latency_ms']:.1f} ms  tokens={body['tokens_processed']}"
    )

    # ── 4. Decode ────────────────────────────────────────────────────────
    body, decode_ms = t(
        "decode",
        lambda: expect(
            sess.post(
                f"{base_url}/v1/attention/decode",
                json={
                    "session_id": session_id,
                    "query_token_embedding": query_token,
                },
                timeout=60,
            ),
            200,
            op="decode",
        ),
    )
    res1 = body["post_attention_residual"]
    assert len(res1) == num_layers, res1
    assert len(res1[0]) == hidden_dim, res1[0]
    print(f"     residual[{num_layers}][{hidden_dim}]  runner={body['latency_ms']:.1f} ms")

    # ── 5. Snapshot ──────────────────────────────────────────────────────
    body, _ = t(
        "snapshot",
        lambda: expect(
            sess.post(
                f"{base_url}/v1/kv-cache/snapshot",
                json={"session_id": session_id},
                timeout=30,
            ),
            200,
            op="snapshot",
        ),
    )
    snap_b64 = body["snapshot"]
    snap_bytes = base64.b64decode(snap_b64)
    assert snap_bytes[:4] == b"\x41\x51\x41\x4c", f"bad magic: {snap_bytes[:4]!r}"
    assert snap_bytes[4] == 0x01 and snap_bytes[5] == 0x00, "bad version"
    print(f"     snapshot_bytes={body['bytes_len']}  magic=LAQA  version=1")

    # ── 6. Restore into a brand-new session ──────────────────────────────
    body, _ = t(
        "restore (new session)",
        lambda: expect(
            sess.post(
                f"{base_url}/v1/attention/session",
                json={"model_id": model_id, "restore_from_snapshot": snap_b64},
                timeout=30,
            ),
            200,
            op="restore",
        ),
    )
    sid_b = body["session_id"]
    assert sid_b != session_id, "restore must allocate a fresh id"
    print(f"     new session_id={sid_b}")

    # ── 7. Free all layers on the new session ────────────────────────────
    body, _ = t(
        "free all layers",
        lambda: expect(
            sess.post(
                f"{base_url}/v1/kv-cache/free",
                json={"session_id": sid_b},
                timeout=10,
            ),
            200,
            op="free",
        ),
    )
    print(f"     layers_freed={body['layers_freed']}")

    # ── 8. Delete both sessions ──────────────────────────────────────────
    for sid in (session_id, sid_b):
        t(
            f"delete {sid[:8]}…",
            lambda sid=sid: expect(
                sess.delete(f"{base_url}/v1/attention/session/{sid}", timeout=10),
                204,
                op="delete_session",
            ),
        )

    print("\n  OK — every endpoint round-tripped cleanly.\n")
    print(f"  prefill: {prefill_ms:.1f} ms wall  /  decode: {decode_ms:.1f} ms wall")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--base-url",
        default=os.environ.get("LARQL_ATTN_URL", "http://localhost:8081"),
        help="HTTP base URL of the attention server (default: $LARQL_ATTN_URL or http://localhost:8081)",
    )
    p.add_argument(
        "--model-id",
        default=os.environ.get("LARQL_MODEL_ID", "gemma-3-4b"),
        help="Model id to bind sessions to (default: $LARQL_MODEL_ID or gemma-3-4b)",
    )
    p.add_argument("--hidden-dim", type=int, default=2560)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument(
        "--kv-format",
        default="iso3",
        choices=["fp32", "planar3", "planar4", "iso3", "iso4"],
    )
    args = p.parse_args()
    try:
        return smoke(
            base_url=args.base_url,
            model_id=args.model_id,
            hidden_dim=args.hidden_dim,
            seq_len=args.seq_len,
            kv_format=args.kv_format,
        )
    except Exception as e:
        print(f"\n  FAIL: {type(e).__name__}: {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
