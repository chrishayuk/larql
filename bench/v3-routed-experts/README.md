# V3 routed experts: correctness receipts

Implementation: `85e85100`. This extends the V3 provider boundary to selected
unweighted expert transforms. It is a CPU packed-MXFP4 correctness rung; there
is no throughput or K3 claim. See the [operator contract](../../docs/ffn/v3-routed-experts.md).

Subsequent exact-HTTP timing and byte decomposition is recorded separately in
the [provisional profiling report](PROFILE-1-results.md).

## Real-model result

**Pass:** all 12 candidate runs matched the three pre-refactor controls on
prompt and generated token IDs. Fifteen CLI runs emitted 480 generated IDs
in total (32 per run). The same three prompts are reused across layouts;
these are not 480 independent test cases.

| Placement | Workers | Prose, code, long prompt |
|---|---:|---|
| Pre-refactor local (`dfba977f` binary) | 0 | Saved controls |
| Refactored local (`85e85100`) | 0 | All match |
| All expert banks on one worker | 1 | All match |
| Experts 0–15 / 16–31 in every layer | 2 | All match |
| First 12 layers / remaining layers split 0–7 and 8–31 | 3 | All match |

All candidate runs used identical release binaries and unchanged artifact
metadata. Saved bindings independently cover all 24 × 32 layer/expert pairs
exactly once in every distributed layout. All workers were stopped after the
runs. The [combined verification receipt](results/verification.json) includes
SHA-256 hashes of every raw run record.

The declared owned source-byte ledger (codes, scales and biases) totals
10,165,616,640 bytes in every layout. Its split is 5,082,808,320 bytes per worker
for the two-way expert split, and 5,082,808,320 / 1,270,702,080 / 3,812,106,240
bytes for the mixed layout. These are binding-ledger values, **not measured
RSS**. Workers widen owned matrix rows to F32; physical read accounting is
verified separately by the fixture gate.

Candidate manifests report `dirty: true` because the driver includes untracked
receipt/report files in that check. The tracked implementation and driver
remained unchanged from `85e85100` throughout the candidate runs; binary hashes
are recorded in every manifest. No performance result is promoted.

## Controls and protocol

`check_cli.py` runs the same three raw prompts with greedy generation, recording
32 output IDs per prompt. Prompt lengths are 5, 13 and 151 tokens. The last
crosses GPT-OSS's 128-token sliding window. These prompt definitions reuse the
separate V3-FFN-SLICE-1 C0 reconnaissance; all results here are fresh runs.

The pre-refactor control uses the exact binaries from WIRE-2 (`dfba977f`,
confirmed against its recorded SHA-256 hashes). Its manifest records checkout
HEAD `8ddf800a` and a dirty working tree because the new source was already being
edited; **those edits were not built into the old binary**. Binary hashes,
rather than checkout HEAD, identify this control.

Every run pins 8 CPU workers, 8 Rayon threads, one Accelerate thread and two
Tokio workers. No peer-exclusive measurement window was established. These
receipts compare IDs, not performance. A generated-ID match does not prove
bitwise equality of every real-model intermediate; the fixture tests below
supply bitwise numerical gates on their stated scope.

## Structural and numerical checks

- Existing routed tests: 42 passed, including the independent served-forward
  oracle and latent/shared-branch checks of the local refactor.
- Production expert group: 30 passed, including three placement topologies,
  exact partial-byte reads, worker matrix/bias residency and bitwise logits
  across a four-token fixture window.
- Binary frame and client corruption tests: passed. Signed zero and subnormal
  bits round-trip; wrong handles, sequences, layers, IDs, sizes, counts,
  content type, nonfinite values and changed OPEN bindings refuse.
- Full V3 HTTP integration suite: 23 passed. Routed coverage includes three
  layouts, interleaved sessions, invalid admission before local payload reads,
  malformed requests, worker repeatability and failed-step/fresh-session replay.
- V3 CLI regression tests: 20 passed. Server option admission tests: 8 passed.
- Clippy with warnings denied for all six changed crates, formatting and
  relative documentation links: passed.

A cancellation-sensitive unit test explicitly distinguishes production
selection order from result arrival order. Worker output contains the down
bias before weighting. Selected shard groups execute concurrently; every
request completes or fails before the step returns.

## Reproduction

Build the CPU binaries, then capture local and distributed IDs:

```bash
cargo build --release -p larql-cli -p larql-server --no-default-features
python3 bench/v3-routed-experts/check_cli.py \
  --model /path/to/gpt-oss-20b.vindex3 --out /tmp/routed-local
python3 bench/v3-routed-experts/check_cli.py \
  --model /path/to/gpt-oss-20b.vindex3 --out /tmp/routed-experts \
  --topology experts --reference /tmp/routed-local/ids.json
```

The driver is scoped to GPT-OSS 20B's 24 layers and 32 experts. Other topology
options are `one` (one worker owns all expert banks) and `mixed` (one worker
owns layers 0–11; two workers split experts 0–7 and 8–31 in layers 12–23).
It starts workers sequentially, waits for bindings, and stops them in `finally`.
Each output directory must be new; logs and failed runs are retained.
