# Non-protocol diagnostic: the executor split threshold (recorded before P/M)

Written 2026-09-24 01:00 BST, before F closed and before any P/M run. **Not an adjudication input.**

A non-protocol diagnostic build changing `MIN_SPLIT_BYTES`
(`crates/larql-vindex/src/format/vindex3/opplan/exec/cpu/executor.rs`) from 4 MB to 1 MB observed ~38 ms/token and
~76 GB/s for `FusedNvfp4Q8` (`production-nvfp4`: 106 -> 71 ms/token). It was run before P/M. It was not committed to the
experimental branch: the source was restored, and the binary was rebuilt and re-verified at 748 worker slabs per token.
It was run under load (a peer compile), on 16 generated tokens, with no repeats. NVFP4-Q8-1's freeze excludes scheduling
changes, so P/M run with the frozen threshold.

Mechanism, as diagnosed:
- **What falls below the threshold:** Gemma 3 4B's attention projections in NVFP4 are 1.5-2.9 MB, below 4 MB, so they
  run single-threaded. That is ~0.30 GB/token.
- **What stays parallel:** the FFN projections (14.7 MB) split across 6 workers (= 12 P-cores / 2).
- **Evidence the FFN kernel isn't the limit:** `LARQL_CPU_WORKERS=12` changed nothing (48 vs 49 ms/token; 54 vs 50 GB/s).
- **Why:** the threshold is a representation-independent byte count, calibrated when these matrices were bf16-sized
  (>= 8 MB). Compressing the representation moved them under it.

Consequence for M, stated in advance: with the frozen threshold, `FusedNvfp4Q8` was informally at ~52 GB/s against
M's 80 GB/s. An M miss is expected, and the cause is this scheduling geometry, not the kernel. It is a separate
question for its own freeze.

Latency model this implies:
**token = parallel projection work + sub-threshold serial projection work + non-projection floor**,
not bytes / aggregate rate + floor.
