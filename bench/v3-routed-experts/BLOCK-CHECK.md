# Block check: GPT-OSS 20B, remote block execution against sequential decode

Status: **correctness evidence only, recorded 2026-11-14.** Nothing here is
timed and no throughput claim is made. The worker still runs one expert
transform per row, so a block changes how many exchanges cross the network,
not how much expert work is done.

## Question

Does the routed block session (`RoutedBlockSession`, see the
[coordinator design](../../docs/ffn/v3-multi-position-coordinator.md)) give,
on a real model, exactly the logits that one-position-at-a-time decode gives
through the same remote experts, including after rejected positions are
discarded?

## Instrument

[`v3_block_check`](../../crates/larql-cli/examples/v3_block_check.rs)
connects to running expert workers and, for each prompt of
[`prompts-v2.json`](prompts-v2.json):

1. decodes the prompt and 64 greedy tokens one position at a time through
   the remote experts, keeping every position's logits as the reference;
2. replays that stream through the block session at widths 1, 2, 4 and 8;
3. replays it with four wrong tokens appended to every four true ones,
   committing only the true ones;
4. replays it as chained partial commits (keep 3 reject 5, keep 2 reject 2,
   keep 4 reject 3, repeating);
5. runs two negative controls on the first block;
6. optionally repeats the sequential decode with resident experts, and
   compares the greedy ids with a saved control.

Every comparison is of f32 bit patterns, per position. Frames are counted at
the transport seam, one per call to the single-row or multi-row route. The
HTTP transport makes one request per such call, with no retry or splitting,
so on these successful runs a frame is one network exchange.

The container is opened as `larql run` and the worker open it on the CPU
production arm: stored bytes, nothing requantised at load.

## Results

Three prompts, 70, 78 and 216 positions (364 in all).

| | One worker | Two workers (experts 0-15, 16-31) |
|---|---|---|
| Mismatched positions, every width and prompt | 0 | 0 |
| Mismatched kept positions, partial and chained commit | 0 | 0 |
| Single-row frames during any whole-block, partial or chained phase | 0 | 0 |
| Negative controls detected | 3 of 3 prompts | 3 of 3 prompts |
| Greedy ids equal the saved control (216-position prompt, 64 ids) | yes | yes |
| Remote sequential equals local decode, all 364 positions | yes | not run |

Frames for the 216-position prompt:

| | One worker | Two workers |
|---|---|---|
| Sequential, single-row | 5,184 | 10,093 |
| Width 2, multi-row | 2,592 | 5,158 |
| Width 4 | 1,296 | 2,588 |
| Width 8 | 648 | 1,296 |

With one worker a block is exactly one frame per routed layer. With two, a
width-8 block contacted both workers at every layer, while a single position
sometimes selects experts on one worker only, which is why sequential decode
needs fewer than two frames per layer on average.

The token streams of the two topologies are identical for all three prompts.

### Negative controls

- One bit flipped in one returned logit is reported at exactly that
  position.
- One expert output value negated inside one multi-row answer is reported as
  a mismatch. The reported positions are a run of consecutive positions
  ending at the block's last (for example 4 to 7), which is what a change to
  one row and everything that later attends to it would give. The receipts
  do not record which row was changed, so that reading is not checked.

A control the checker misses fails the run. The controls' own frames are not
in any of the frame counts above.

## Receipts

| Run | Receipt |
|---|---|
| One worker, with local reference and saved control | [`20261114-02-one-worker`](block-results/20261114-02-one-worker/receipt.json) |
| Two workers, with saved control | [`20261114-03-two-workers`](block-results/20261114-03-two-workers/receipt.json) |
| Superseded first run | [`20261114-01-one-worker`](block-results/20261114-01-one-worker/receipt.json) |

Each of the two corrected run directories also holds the worker bindings
and logs, the source revision and worktree status, and SHA-256 of the
binaries, the model's `index.json` and `system_graph.json`, the workload and
the saved control. The superseded run's directory holds only its receipt,
binding, logs, revision and binary hashes.

**The first run is superseded, not wrong.** Its harness opened the container
under the library's default policy, which may requantise at load, where
`larql run` and the worker use stored bytes. Block and sequential decode
shared that coordinator, so its parity and frame counts hold, but its logits
were not the production arithmetic and its greedy stream left the saved
control at the fourth generated token. The corrected runs match the control
for all 64. That run also predates the negative controls and the local and
saved-control comparisons, and decoded 16 greedy tokens per prompt, not 64:
it was never self-checked, and the comparison with the control was made
afterwards from its recorded stream.

## What this does not show

- Any timing or throughput.
- Speculative decoding: the keep and reject patterns are imposed by the
  harness. A width-8 block is an eight-position verification, not eight
  accepted tokens.
- Local decode against the two-worker topology, or logit bits compared
  directly across the two topologies (their token streams are compared).
- A saved control for the two shorter prompts; one exists for the long
  prompt only.
- Any reduction in expert weight reads.

The receipts record revision `ab303e92` with a modified worktree. The
modifications were this harness, its dev-dependency and the lock file; the
runtime crates were as committed.

**The receipts are not bound to the harness source.** The harness was an
untracked file when the runs were made, and the receipts record a hash of
its built binary, not of its source. Nothing in them proves the committed
file is the one that ran. The worker logs also record no command line or
environment; the expert ranges and backend are in the binding files.

The sequential reference shares the coordinator's attention, router and
head code and the worker's per-row transform with the block path, so a
defect common to both would not show in the block comparison. The local
decode comparison (one worker) and the saved control (token ids, one
prompt) are the checks that reach outside it.
