# GW-V2 execution record

**Causal run complete:** 22 September 2026 (Europe/London). All 666 executions
and 75 arms are sealed, with zero parity mismatches and 49,284 intervention
firings. [Full results](gw-v2-results.md): 0/35 held-out-clearing primary cells,
no registered early frontier. Latency reporting remains incomplete pending an
acknowledged exclusive window.

The completed checkpoint separates three distinct claims:

| Question | Recorded outcome |
|---|---|
| Execution validity | All 666 executions completed; natural/no-op and full-exact/identity controls passed bit-for-bit. |
| Predictor fidelity surface | All 35 primary and 35 sham cells are recorded. Every primary cell is below 0.80 raw and z-scored point retention on each held-out split. Sham reports are diagnostic only. |
| Registered frontier | No qualifying contiguous 2×2 tile at depths no later than L12; 0/35 primary cells clear held out. |

AMEND-1 was authorized and sealed before V2 model capture, fitting or replay.
It resolves only the fit-sham correspondence and freezes its diagnostic
reporting. The original protocol and primary decisions are unchanged.

| Authority | Canonical SHA-256 identity |
|---|---|
| Original protocol | `830c2ceb97fce4f2d03e74e207d04e54257cd5b38e830a93047a86b6bd4fd006` |
| Population | `d73e0783adc0c26777ed9f0e4b880133814c39d0283cd7d816cd72cd1d37c23c` |
| AMEND-1 | `bb5ebb48e52768e50ff5a26954bf8873489c1ab01d0fec1272e2f52caa3b2d5b` |
| Donor map | `3ed66fc2432fe2d967ab58243faea729d0dbd0ed19f211ae6b58d17d7afec45c` |

The [amendment](../bench/gw-v2/gemma3-4b-it-phase1/gwv2-amend-1.json)
and [complete donor map](../bench/gw-v2/gemma3-4b-it-phase1/gwv2-donor-map.json)
are separate artifacts. Every execution stage verifies the four identities.
The additional execution binding checks each original row, role, offset and
model metadata file against its frozen authority.

The sham is partially shuffled: 42/44 training countries, 522/603 token-position
rows. KNA and STP remain unchanged. The complete registered population is
reported by split, and the shuffled training recipients are reported separately
as a diagnostic. Original matched controls are retained when recipients are
filtered. A held-out row has no donor assignment, so no shuffled-only held-out
subset is manufactured.

## Stage artifacts

- Execution binding: `bench/gw-v2/gemma3-4b-it-phase1/execution.json`.
- Capture: `output/gwv2-gemma3-4b-it-phase1-capture/capture.json`.
- Fitting: `output/gwv2-gemma3-4b-it-phase1-fit/fit.json`.
- Authoritative replay: `output/gwv2-gemma3-4b-it-phase1-replay2/replay.json`.
- Pinned replay executable: `output/gwv2-gemma3-4b-it-phase1-runner/observatory_record`.
- Prefix constructor parity: `output/gwv2-gemma3-4b-it-phase1-prefix/prefix-parity.json`.
- Analysis implementation seal: `bench/gw-v2/gemma3-4b-it-phase1/analysis-implementation.json`.
- Pinned executable image: `output/gwv2-gemma3-4b-it-phase1-image2/execution-image.json`.
- Prefix/predictor cost ledger: `output/gwv2-gemma3-4b-it-phase1-cost2.json`.
- Adjudication: `output/gwv2-gemma3-4b-it-phase1-adjudication.json`.

The adjudication canonical SHA-256 is
`710bdf8568aafea46591668a471bdd765a01a13e28ae457848e143215b42bc0d`.
The runner passed Clippy with warnings denied, 22 Python tests, two targeted
Rust regression tests, and a synthetic report-rendering check. The whole dirty
workspace was not reformatted or subjected to a full workspace test run.

The analysis source was sealed before reading replay outcomes. The registered
bootstrap generator seed is reset separately within each split and diagnostic
subset. V2-specific Rust changes made here after the replay executable was copied
are a type alias, formatting and an additional prefix refusal for V-normalized architectures;
the running executable remains pinned and is not overwritten.

Capture completed all 666 executions and 1,008 subject-token positions with
bit-exact natural source-head and carrier reconstruction. No candidate-effect
readouts were stored in the capture. All 35 primary and 35 sham models were
fitted and sealed before replay. Training diagnostic predictions leave the
recipient subject out; sham training fits additionally omit rows whose frozen
donor is that subject. The donor mapping is never rewired.

The first replay implementation recomputed all layers and the full vocabulary
head for every arm. Its partial files remain in
`output/gwv2-gemma3-4b-it-phase1-replay/`; it was stopped after the replacement
runner passed its exact controls. No numerical candidate outcomes from that
partial run were inspected or used to choose a predictor or modify a gate.
It is not an adjudication input, nor an input to fitting, surface summaries,
bootstrap intervals, gate calculations or performance claims.

The separate, hashed [superseded-run provenance record](../bench/gw-v2/gemma3-4b-it-phase1/superseded-replay-provenance.json)
retains all four partial-file sizes and SHA-256 hashes, execution-tool session
76312, SIGINT termination and exit code 130. Its OS process ID was not retained.
The last retained progress checkpoint was 90 executions. The proximal and
terminal files each contain 94 whole execution-sized records and part of a 95th;
the independently buffered before/timing files have different flushed coverage.
These byte counts are not a claim of complete control coverage, and the exact
completed-execution count at termination is unknown. No `replay.json` was sealed.
Only file sizes and hashes were examined for this provenance addition, not
numeric candidate values. The record identifies the earlier image manifest's
executable hash without claiming that a separate full-replay binary was preserved.

The replacement replay starts every execution afresh. Its natural arm runs the
whole model. Its exact-V arm independently runs the whole intervened model.
The no-op, identity and candidate arms enter the canonical decode interpreter
at L24 from the captured natural carrier, with the original prefix KV state.
They execute L24 attention, its declared intervention, the real L24 FFN and all
later layers. Selected terminal vocabulary rows must match the full exit
bit-for-bit in the natural/no-op and full-exact/identity controls on every row.
This is a harness optimization; natural carrier and non-H1 state remain inputs
and no avoided-work claim follows from replay reuse.

## Reproduction entry points

Run from the repository root. Use new stage directories: the runner refuses
to overwrite tensors or seals. Validate the existing registration and amendment
with `python3 scripts/gwv2_prepare.py validate bench/gw-v2/gemma3-4b-it-phase1/execution.json`.

The stage interfaces are:

```text
observatory_record --gwv2 capture EXECUTION CAPTURE_DIRECTORY
.venv/bin/python scripts/gwv2_fit.py EXECUTION CAPTURE_JSON FIT_DIRECTORY
observatory_record --gwv2 replay EXECUTION REPLAY_DIRECTORY CAPTURE_JSON FIT_JSON
observatory_record --gwv2 inspect EXECUTION IMAGE_DIRECTORY
observatory_record --gwv2 prefix-check EXECUTION PREFIX_DIRECTORY CAPTURE_JSON
.venv/bin/python scripts/gwv2_adjudicate.py EXECUTION REPLAY_JSON FIT_JSON CAPTURE_JSON RESULT_JSON
.venv/bin/python scripts/gwv2_cost.py EXECUTION IMAGE_JSON FIT_JSON PREFIX_JSON COST_JSON
.venv/bin/python scripts/gwv2_report.py RESULT_JSON REPORT_MD --cost COST_JSON
```

The report generator does not adjudicate or select cells. The canonical
adjudication JSON is its authority. Predictors and all replay tensors are
file-hashed; the fitting manifest and analysis implementation additionally have
canonical JSON seals.

## Timing boundary

Operational elapsed times are not performance measurements. The registered
timing protocol still requires warmed baseline/candidate/baseline brackets,
less than 1% bracket drift, and acknowledged exclusivity with every peer
session. A request to coordinate that window was sent to the user while causal
execution continued. Until a window is confirmed, latency claims remain
unavailable.

The separate prefix instrument executes only canonical layers below the source
depth, then the source pre-attention norm and required KV-head V projection.
It reconstructed all 1,008 subject-token positions at every one of the seven
depths bit-for-bit: 7,056 checked vectors, zero mismatches. The sealed cost ledger
contains all 35 cells and all 666 executions per cell. This instrument does not
replace the natural surrounding state in V2. The ledger reports bounded
matvec/attention FLOPs and logical vector/state bytes, not measured DRAM traffic.
The initial `image/` and `cost.json` artifacts are retained. Repeating image
inspection using the pinned replay executable produced identical plan and
residency fields; `image2/` and `cost2.json` bind the final accounting directly to
that executable instead of the earlier full-replay build.
