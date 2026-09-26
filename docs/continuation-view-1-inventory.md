# Continuation read contract: what depends on its shape

**Status: reconnaissance (code reading only).** Read against main
`cf9c5309` (CONTINUATION-MEM-1 and CONTINUATION-PLUGIN-1 both closed). It is the evidence
CONTINUATION-VIEW-1 is frozen against: every claim is a statement about the
code, with its location. The forecast is
[`continuation-view-1.json`](represent/forecasts/continuation-view-1.json);
the costs that justify it were measured by CONTINUATION-MEM-1
([`continuation-mem-1-notes.json`](represent/forecasts/continuation-mem-1-notes.json)).

In this document, `exec/` means
`crates/larql-vindex/src/format/vindex3/opplan/exec/`.

## The contract today

`ContinuationProvider::keys(layer)` and `values(layer)` return
`&[Vec<f32>]`: every appended row, ordered by position from 0, one heap
allocation per row (`exec/kv.rs:101-114`). The module doc ties the shape to
`AttentionStepCall`: "the `&[Vec<f32>]` row-slice shape mirrors
AttentionStepCall and changes only with it" (`exec/kv.rs:27-29`).

MEM-1 attributed three costs to this contract (class a):

- **The mirror.** `canonical/v1` keeps its matrix and a second `Vec<Vec<f32>>`
  copy of every row, only to answer `&[Vec<f32>]`: 313 MB on gemma3-4b at
  1,124 positions, the same size as the K/V data itself.
- **Unreachable rows.** Rows below a sliding window cannot be dropped: 24 MB
  there, about 1.7 GB at 8,192 positions.
- **Per-row allocation.** One heap allocation per row per K and V.

## Dependencies, by kind

### Representation: callers that need the `Vec<Vec<f32>>` shape

This dependency is shallow. No kernel needs the shape.

- **The trait signature.** `exec/kv.rs:111,114`.
- **`AttentionStepCall` fields.** `keys: &'a [Vec<f32>]` and
  `values: &'a [Vec<f32>]` (`exec/backend.rs:827,829`). These are the only
  struct fields of the row type.
- **ConvQkv parameters.** `conv_qkv::layer_forward_with(past_keys: &[Vec<f32>],
  past_values: &[Vec<f32>], base)` (`exec/conv_qkv.rs:105-111`).
- **Implementor storage, 10 implementors.**
  - `RowKvState`: `exec/kv.rs:286,338`.
  - `CanonicalKvState`: `crates/larql-kv/src/vindex3/mod.rs:88,179`. Its
    mirror is built at `:322` and again in `LayerRows::from_matrices` at
    `:63`.
  - The C5 external `HostileRows`:
    `crates/larql-kv/tests/external_continuation_provider/provider.rs:62`. It
    is also compiled into `crates/larql-continuation-fixture`.
  - Six test doubles: `Perturbed`, `RecordingKvState`, `RowsOnly`,
    `RowsOnlyLatent`, `KvOnly`, `NoLatent`.
  - CONTINUATION-MEM-1's measuring wrapper `Measured<P>`
    (`crates/larql-kv/tests/continuation_mem_1/measured.rs`, also compiled
    into `crates/larql-server/tests/continuation_mem_1_handoff`). It is the
    instrument VIEW-1 re-runs, so it migrates with the trait. Its companion
    `Counting<B>` backend (`counting.rs`) only forwards `AttentionStepCall`.
- **Below the step, kernels already abstract storage.** Every attention
  kernel reads rows through `key_of` / `value_of: impl Fn(usize) -> &[f32]`:
  - reference: `exec/reference.rs:308,337`;
  - production: `exec/production.rs:707,721,851,876`;
  - observation: `exec/observe.rs:398`.

  Only the closure bodies that build those functions touch `Vec`.

### Indexing: absolute position 0 = slice index 0

- **Six index expressions, one pair per backend.** The index `p` is the
  absolute position in `start..=position`.
  - Reference: `exec/reference.rs:570,577`.
  - Production: `exec/production.rs:1002,1009`.
  - Device: `exec/device.rs:610,617`.
- **ConvQkv's accessors.** They split history from the batch's new rows on
  `index < past_keys.len()`, and compute
  `visible = past_keys.len() + t + 1` (`exec/conv_qkv.rs:213-228`). The
  separate `base` is used only for RoPE.
- **The invariant `keys(layer).len() == position` is documented, never
  asserted in production.**
  - It is documented at `exec/backend.rs:826-829` ("rows for positions
    `0..position`") and in the module doc at `exec/kv.rs:21-26`.
  - About ten tests rely on it, for example
    `crates/larql-inference/src/vindex3/tests/mod.rs:498-500` and
    `exec/tests/kv.rs:320-321`.
- **`LatentKvRows::rows()` (MLA) follows the same from-0 pattern.** It is
  defined at `exec/continuation.rs:418-430` and indexed at `exec/mla.rs:323`.
  MEM-1 found no class (a) cost there: allocated equals payload.

### Lifetime and borrowing

- **Softmax needs its borrow only for one backend call.**
  - `attention_into_kv` re-borrows around each append
    (`exec/mod.rs:2336-2344`).
  - Decode borrows `state()` for its step (`exec/decode.rs:862-866`).
  - Nothing needs a long-lived slice.
- **ConvQkv is the one real case.**
  - The operator's `key_at` reads past rows *by reference*.
  - The executor deep-copies them (`exec/mod.rs:1446-1447`,
    `exec/decode.rs:831-832`) only to release the `&self` borrow of
    `keys`/`values` before taking `&mut recurrent_state`.
  - MEM-1's quadratic copy (M4: 2·h rows per call) is therefore caused by
    borrowing, not required by the operator.

### Execution: what kernels actually read

- **Kernels read a position range, one row at a time, by absolute
  position.**
  - Sliding layers read `start..=position`.
  - Global layers and ConvQkv read all history.
  - Nothing requires contiguous storage.
- **The window floor is computed inside each backend, twice.**
  - Production has `source_start` (`exec/production.rs:694-705`), which the
    device backend reuses through `aggregate_heads` (`exec/device.rs:602`).
  - Reference has its own duplicate (`exec/reference.rs:356-362`).
- **`LayerKvGeometry.window` reaches providers but is informational**
  (`exec/kv.rs:46`). Nothing guarantees a provider which rows it may drop.
- **Metal needs no device-contiguous K/V.** V3 softmax on Metal
  (`DevicePlanBackend`) runs Q/K/V projections on the device and aggregates
  on the CPU from the provider's rows (`exec/device.rs:12-21, 574-620`).
  V3 has no GPU KV. The Kimi `stack_metal.rs` device state is separate and
  not driven by `DecodeSession`.

## What this implies

**One abstraction suffices.** A view is a logical position range
`[base, end)` over a backing the executor does not see. It exposes
`key(p)` and `value(p) -> &[f32]` for absolute positions, with a checked
refusal outside the range.

- The 6 index expressions and ConvQkv's two accessors become view calls.
- Row-backed, contiguous, windowed and paged storage all fit that shape
  without being named in it.
- **Remote storage fits only through materialised or cache-backed rows.** A
  borrowed `&[f32]` requires the bytes to be local for the borrow's lifetime,
  so direct remote fetching may need a different access model later.
