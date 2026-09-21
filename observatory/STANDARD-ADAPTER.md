# Standard capture → Observatory UI

Status: **draft runner bridge implemented in the UI; not an executor ABI freeze**.
A real Granite Standard record is now exported and replay-audited (2026-09-20).
The recording runner owns serialization and session identity; live transport remains
unimplemented. The executor tap and arithmetic are unchanged.

Use **Open recording** on the arrival screen. JSON stays in the browser. Save
original record downloads the unmodified parsed source, including fields the UI
has not rendered. Replay, Map, Trace, Graph and the inspector share the existing
reducer and semantic selection. A source labelled `executor` is displayed as
**runner-declared execution**, not independently verified evidence.

## Relationship to V3-OBS-1

The adapter mirrors `WriteStats` in
`crates/larql-vindex/src/format/vindex3/opplan/exec/observe_stats.rs`:

| Executor field | UI meaning |
|---|---|
| layer / site / position | Semantic selection; `Attention` or `Ffn` |
| norm | L2 norm of post-add, pre-layer-scale `after` |
| delta_norm | L2 norm of the applied delta; never norm(after) − norm(before) |
| layer_scale | Recorded separately; does not rescale any captured value |
| projection | Exact recorded coordinates, no refit or reconstruction |
| probe | Raw carrier dot output-head row, ordered by predeclared token IDs |

`norm_method` must be `l2-v1`; `probe_method` must be `dot-raw-v1`. There is no
final norm or softmax. A probe difference is not labelled a log-odds estimate.
No full-vocabulary rank/probability is computed. Empty probes are valid.

The stats observer does not supply an entering-carrier norm, embedding sample,
per-site duration, directional write probe, or attention-source distribution.
These remain unavailable. It does not supply run IDs, global sequence, or
monotonic timestamps; the runner must add them. No timing is inferred from
arrival order or neighboring timestamps.

The first bridge admits only `Single`, main-carrier observations in a recorded
3D basis. Other basis dimensions, Bundle, and History refuse explicitly. They
must receive topology-aware adapters, not zero padding or hidden reduction.

## Draft file shape

`schema = larql.observatory.standard.v1`

Required fields:

- `provenance`: `executor` or `synthetic` (test data must use synthetic).
- `run_id`, `model`, and `identity`: nonempty `container`, `plan`, `lowering`,
  `tokenizer`, `session` authority strings supplied by the runner.
- `capture: standard`, `topology: Single`, `norm_method`, `probe_method`.
- `layers`, `program`: each layer has ordered `sites`, each with `site`
  (`Attention`/`Ffn`) and a unique plan-local `operation_id`. A mixer-only layer
  declares one site. No FFN write is inferred from `FfnDone`.
- `tokens`: contiguous `position`, `token_id`, optional `label`. Includes all
  positions named in the record, including decode positions. Redacted labels
  display as token IDs. Optional `prompt` is never reconstructed from labels.
- `observed_positions`: the explicit capture scope. A cached or unobserved
  prompt prefix is not claimed to have observations.
- `probe_tokens`: ordered token IDs with optional labels. Labels are decorated
  with their IDs so identical decoded text cannot merge distinct head rows.
- `basis`: executor `BasisIdentity` fields (`provider`, `id`, `hash_hex`, `dims`,
  `hidden`) plus the runner's `source`. Hash is lowercase hex without a prefix.
- `runtime`: explicit `timing_intrusive` boolean and `device_readbacks` count.
- `coverage`: `complete` or `incomplete`, for carrier writes in the declared
  position scope only. This does not claim full activation/embedding coverage.
- `events`: canonical order with decimal-string u64 `sequence` and
  `timestamp_ns`, plus `run_id` and `kind` on every event.

Event bodies:

| kind | Additional fields |
|---|---|
| RunStarted | Start at sequence `0`; metadata lives in file manifest |
| Tokenized | Tokens live in file manifest |
| CarrierWrite | `stats`: the `WriteStats` fields listed above |
| SiteCompleted | `layer`, `position`, `site: AttentionDone / FfnDone` |
| TokenProduced | `token_id`, `predicting_position`, `text` |
| EventDropped | `count` (positive number); detailed loss metadata may be retained in source |
| RunCompleted | `reason` |
| RunRefused | `reason`; always incomplete capture |

Minimal illustrative write (numbers authored, not measured):

```json
{
  "run_id": "adapter-test",
  "sequence": "2",
  "timestamp_ns": "1000",
  "kind": "CarrierWrite",
  "stats": {
    "layer": 0,
    "site": "Attention",
    "position": 0,
    "norm": 31.25,
    "delta_norm": 2.75,
    "layer_scale": null,
    "projection": [0.125, -0.5, 1.25],
    "probe": [2.125, -1.75]
  }
}
```

See `tests/standard.test.mjs` for a complete authored manifest including a
mixer-only layer and `FfnDone` without a write.

## Validation and replay

- Duplicate deliveries with identical envelopes deduplicate; conflicting IDs
  refuse. Sequence/timestamp integers never pass through floating-point parsing.
- Program order is checked independently for each position. Unknown sites,
  out-of-scope positions and duplicate semantic writes refuse.
- Complete coverage requires RunCompleted, no reported drops or sequence gaps,
  and exactly every declared write for every observed position.
- Incomplete coverage preserves gaps and loss, including loss reported at the
  end with no following observation. Missing terminal events remain incomplete.
- No record is a parity witness merely because its checks pass.

Preflight without running a model:

```sh
cd observatory
node scripts/check-standard.mjs /path/to/runner-record.json
```

For the first real, lossless replay witness:

```sh
node scripts/check-standard.mjs /path/to/runner-record.json --require-executor --require-complete > /path/to/replay-audit.json
```

The report binds to the input file's SHA-256. It compares **every** carrier write
against the source (identity, coordinates, norms, raw probes and layer scale),
checks deduplication and unavailable fields, then saves/reopens through JSON.
It checks every replay boundary for short records, or 33 evenly spaced boundaries
including empty and complete for longer records. It does not claim exhaustive
scrubbing of long logs or browser rendering QA. The browser performs the same
audit on Standard import and displays the result in the Record panel.

`--require-executor` refuses authored fixtures; the provenance remains a runner
declaration, not independent authentication. `--require-complete` refuses partial
records. Neither flag establishes observed/unobserved output parity. Attach the
executor witness separately; do not promote the existing synthetic adapter tests
to a real-model result.

Current integration status: the checked-in Granite record contains 400 writes
across five positions / 40 layers and 804 unique events, with no gaps or drops.
All source values and JSON roundtrip pass the replay audit; 33 prefix boundaries
are checked. Separate observed/unobserved execution produced bit-identical finite
logits at all five positions and the greedy output ` Paris`. Live transport and
browser interaction QA remain unclaimed.

## Executor-session handoff

1. Agree or translate this **draft** runner envelope; no executor type change
   is needed. Do not make the executor emit frontend envelope fields.
2. Export one short Granite CPU Standard record with actual token labels/IDs,
   bound program, identity, basis and probe order. Gemma 3 4B follows admission.
3. Check/import it and compare every UI coordinate, norm and probe with the
   source; save/open and scrub without inference. Preserve the original record.
4. Supply the runner endpoint and its framing/resume/capability contract before
   wiring live delivery. The existing inference WebSocket is not presumed to
   speak this format. UI pause must pause rendering only.
5. Attach the executor's observed/unobserved parity witness separately. Force
   live queue overflow and verify persistent loss plus an independent lossless
   record before calling live observation complete.

The original adapter tests remain authored cases. `tests/measured.test.mjs` now
checks the real recording and the SHA-256 binding of its separate parity witness.
No live/nonblocking transport result or reference-framework parity is claimed.

## Recording runner

```sh
# From the repository root; paths must name new output files.
# Build the separately maintained observatory_record exporter first.
target/release/examples/observatory_record \
  ~/chris-models/granite-4.2-3b.s6.vindex3 \
  /tmp/granite-standard.json 'The capital of France is' ' Paris' ' Berlin'
node observatory/scripts/check-standard.mjs /tmp/granite-standard.json \
  --require-executor --require-complete
```

The runner verifies payload hashes, closes the `target` plan, prepares production
CPU operands once, and observes the canonical decode traversal. It accepts 1–32
prompt tokens and up to eight predeclared single-token probe texts. Head probes
widen only the selected stored BF16/F16/F32 rows via mmap, before execution; they
do not normalize, softcap, or use a full per-layer head pass. Unsupported head
formats refuse when probes are requested. Omit probes for norm/projection-only
capture. Bundle/History and empty plans refuse explicitly.

It emits one greedy output token (lowest-ID tie break), without a chat template.
A fresh unobserved session over the same prepared image checks all vocabulary
logits at all input positions bit-for-bit. This second pass supplies only the
independent parity control, never visualization data. Failed capture/coverage or
parity checks produce no accepted recording. Final files are written through
temporary files with no overwrite; the parity companion is `*.parity.json`.
This is a bounded in-memory lossless recorder, not a nonblocking live queue.
Timing is explicitly intrusive and not representative of unobserved performance.

`run_provenance` preserves the executor's serialized `RunProvenance`. The UI
checks that basis/probe/method identities agree, and shows the execution's pinned
realization classes as **run-wide** evidence. The runner includes its executable
and adapter-source hashes. Container identity hashes the serialized inspected
index and graph, whose payload entries were verified before execution. Plan-local
operation IDs are derived from that exact bound plan's hash plus layer/site.

The first record's default production policy includes 121 BF16→Q8 requantized
operands, 80 fused BF16, 80 BLAS F32 and one gather. This witness does not establish
exact equivalence to an all-BF16 or reference-backend execution.
