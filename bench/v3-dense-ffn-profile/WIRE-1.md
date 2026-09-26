# V3-FFN-WIRE-1: exact binary versus JSON

Implementation revision **`b2f24c1d`**, September 25, 2026. Same M3 Max / 128 GiB
and thread settings as the [initial diagnostic](README.md). Both transports and
the local control use the same optimized CPU build and numerical providers.
Gemma uses BF16-source `Requantise(FusedQ8)` FFN weights; Qwen uses BF16-source
`Decode(BlasF32)`. Both carriers remain exact f32.

The worker validates immutable authority at preparation/open, then accepts a
36-byte header and binary carrier through a persistent HTTP client pool. The
incarnation handle, sequence, layer, width, exact length and finite values are
checked per operation. See the [wire contract](../../docs/ffn/v3-ffn-wire.md).

## Direct byte result

Actual body counts, including every binary header. JSON is the first measured
JSON arm; binary counts were verified on **every position of every binary run**.
Both response and request have the same fixed binary size.

| Model | JSON request bytes/token | Binary request bytes/token | Request reduction | Combined request+response traffic reduction factor |
|---|---:|---:|---:|---:|
| Qwen3 0.6B, 28 layers | 1,316,849 mean | **115,696 exact** | 91.2% | 11.40× |
| Gemma 3 4B, 34 layers | 2,474,730 mean | **349,384 exact** | 85.9% | 7.17× |

Cold discovery/open traffic and HTTP/TLS/TCP headers are excluded. No carrier
quantization, prediction, innovation, remote KV or changed FFN math is involved.

## Timing observations and drift

**Diagnostics, not a promoted performance claim:** peer exclusivity remains
unconfirmed. Six warmup runs precede two nested brackets per model:
local → JSON → binary → JSON → local. Each invocation generates 48 tokens;
summaries exclude the 19-token prompt and first eight continuation steps, leaving
39 matched positions. Every one of the **32 CLI runs matched token IDs** within
its model across local, JSON and binary. The separate fixture gate checks
bitwise logits and crosses the sliding-window boundary.

Raw ranges below include both measured brackets, including drift-rejected ones;
warmups are excluded. They are not pooled estimates or validated speedup ratios.

| Model | Local ms/token | JSON ms/token | Binary ms/token | JSON tok/s | Binary tok/s |
|---|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 22.37–22.95 | 39.41–41.26 | **27.21–29.69** | 24.24–25.37 | **33.68–36.74** |
| Gemma 3 4B | 61.24–63.88 | 93.08–94.72 | **68.53–70.15** | 10.56–10.74 | **14.25–14.59** |

| Model / bracket | Outer local drift | Inner JSON drift | JSON/binary comparison |
|---|---:|---:|---|
| Qwen / 0 | 0.533% | 2.433% | Rejected |
| Qwen / 1 | 2.561% | 3.925% | Rejected |
| Gemma / 0 | 0.581% | 0.227% | Both numerical drift checks pass; exclusivity unconfirmed |
| Gemma / 1 | 2.878% | 1.237% | Rejected |

In Gemma bracket 0, the local midpoint is **61.419 ms/token**, JSON midpoint
**93.186 ms/token**, and binary **68.535 ms/token**. The added time over local is
7.116 ms for binary versus 31.767 ms for JSON. This is one bracket observation,
not evidence of sustained performance across workloads or machines.

## Remaining overhead

Gemma bracket 0, binary arm, mean per token:

| Top-level interval | ms |
|---|---:|
| Attention / entry | 20.052 |
| FFN provider inclusive | 41.644 |
| Residual / re-entry | 0.655 |
| Embedding/head/other | 6.184 |
| **Total** | **68.535** |

Within the FFN call, the worker transform takes 34.036 ms, client encoding
0.085 ms, client decoding 0.055 ms, worker receipt/decoding 0.159 ms, and worker
encoding 0.123 ms. The measured HTTP round trip totals 41.234 ms, with a
6.129 ms remainder after subtracting worker handler time. These are nested
intervals and must not be added to the top-level partition.

The corresponding JSON client encoding was 4.418 ms/token. Qwen client encoding
fell from about 2.013 ms to 0.023 ms in its first measured arm, though its timing
bracket failed the drift check. The remaining transport overhead includes HTTP
and scheduling; it is not a measurement of pure network RTT. LAN and
uninstrumented throughput remain unmeasured.

## Receipts and reproduction

- [Qwen manifest](results/wire-1-qwen3-0.6b-20260925/manifest.json),
  [brackets](results/wire-1-qwen3-0.6b-20260925/brackets.json),
  [per-token binary table](results/wire-1-qwen3-0.6b-20260925/block-0-binary.tsv),
  [all raw traces](results/wire-1-qwen3-0.6b-20260925/raw-traces.tar.gz).
- [Gemma manifest](results/wire-1-gemma3-4b-20260925/manifest.json),
  [brackets](results/wire-1-gemma3-4b-20260925/brackets.json),
  [per-token binary table](results/wire-1-gemma3-4b-20260925/block-0-binary.tsv),
  [all raw traces](results/wire-1-gemma3-4b-20260925/raw-traces.tar.gz).

Archives contain all 16 traces per model and were verified against the originals
using `trace-archive.json`. Manifests include binary and artifact metadata hashes;
`trials.json` records every command and summary. Worker bindings and stdout/stderr
are retained. All measurement workers were stopped after their model's run.

Reproduce with `run_loopback.py MODEL NEW_OUTPUT_DIRECTORY --wire-comparison`.
The ordinary driver defaults explicitly to `--wire json` so the historical
diagnostic remains reproducible despite the CLI's new binary default.
