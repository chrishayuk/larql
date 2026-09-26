# The continuation plane — inventory before it is opened

**Status: measured, not built.** Read against main `43b2e9d9` (PR #531),
with PR #526 (codec and lowering providers loaded from shared libraries)
open beside it. Written so that "open the continuation plane" starts
from what the tree contains, the way the
[lowering inventory](lowering-plane-inventory.md) did for the lowering
plane. The forecast that freezes the transition is
[`continuation-plugin-1.json`](represent/forecasts/continuation-plugin-1.json).

Two plugin planes exist or are being opened. This is the third:

```text
                 VINDEX3 program
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      codec         lowering      continuation
   what bytes?    how computed?   what state survives a step?
```

A codec never names a backend. A lowering never names a codec. A
continuation provider should name neither. It answers what state a
conversation carries between steps, and nothing else.

## What is already right

- **The seam is a trait, and the caller owns the state.**
  `ContinuationProvider` (`…/opplan/exec/kv.rs`) is driven by batch
  prefill and `DecodeSession` alike. There is no batch→decode state
  translation. Its module doc already says that residency, quantisation,
  windowing and checkpointing are policy that composes outside the
  executor.
- **Geometry comes from the plan.** `plan_continuation_geometry` returns
  one `LayerContinuationGeometry` per layer: `Kv`, `LatentKv`,
  `Recurrent`, `KvAndRecurrent` or `Stateless`. Providers never ask
  `ModelArchitecture`. This is the vocabulary a capability declaration
  can match against.
- **Refusal is already typed.** `ContinuationError` distinguishes "this
  provider holds no recurrent/latent state" from "this layer is not
  recurrent/latent". A KV-only provider fails closed on a hybrid layer.
- **Two exact providers exist, and a gate pins them equal.** `RowKvState`
  and `CanonicalKvState` are bit-identical through prefill, resume and
  decode (the VI3-KV-1 gates in `crates/larql-kv/src/vindex3/tests/`).

## What is missing

| fact | baseline |
|---|---|
| provider identity | none; a provider is known by its Rust type |
| registry | none |
| `ContinuationProvider` impls in production source | 2 (`RowKvState`, `CanonicalKvState`); 1 more in tests |
| trait methods | 9 (`prepare`, `append`, `keys`, `values`, `position`, `set_position`, `prepare_continuation`, `recurrent_state`, `latent_state`) |
| non-test construction sites (`CanonicalKvState::new`/`from_cache`, `RowKvState::default`) | 16 across 15 files (CLI 9, server 1, LQL 1, inference 1, vindex 4) |
| V3 provider selection | string match: `run_cmd_vindex3/mod.rs:182` (`"standard" \| "row" \| "no-cache"`), `run_cmd_vindex3/inputs.rs:56` |
| durable handoff | `V3KvHandoff { kv: CanonicalKvState, absorbed_ids }` (`larql-server/src/vindex3.rs:349`), so authority is carried by the concrete type |
| authority on resume | model identity, session, token-prefix match; no provider identity |
| plugin registration | `PluginRegistrar` (PR #526, open) takes codecs and lowerings only |

The handoff row is the one that matters most. A resumed conversation is
interpreted by whatever type the server compiled in. That is safe today
only because there is exactly one type. Once a second provider can hold
server state, the type can no longer serve as the authority.

## `no-cache` is not a provider

`--engine no-cache` replays the whole input history through fresh state
on every step (`larql-inference`). It keeps no continuation. It *drives*
a provider. The plugin model makes this explicit: `no-cache` stays a
runtime mode layered over a provider, and is not registered as one.

## Two stages, two freezes

**Stage 1 — open the provider plane (CONTINUATION-PLUGIN-1).** Identity,
registry, built-ins behind it, authority in handoffs, one hostile
external provider. The trait's method set does not change. The
forecast's F3 counts it.

**Stage 2 — open the read contract (a separate freeze).** Today the
contract requires every appended row to be retained and served from
absolute position 0 (`kv.rs` module doc). `AttentionStepCall` says keys
are "positions `0..position`". Every step backend (`reference.rs:570`,
`production.rs:973`, `device.rs:606`) reads `step.keys[p]` at an
absolute `p`, bounded below by `source_start`. So no windowed, compressed
or device-resident provider can be honest until the executor learns a
retained-row base. That is a contract change with its own falsifiers.
It is a new question, so it gets a new freeze, not a stage-1 wave.
Measurement comes first: CONTINUATION-MEM-1 measures resident bytes by
state form and context length. Then CONTINUATION-VIEW-1 removes the
duplicated row views, and CONTINUATION-WINDOW-1 bounds residency exactly.

Baseline facts for that freeze, recorded now:

- **V3 does not touch the V2 Metal `KVCache`.** No reference to
  `KVCache::new`, `max_seq` or `DEFAULT_GPU_KV_CACHE_MAX_SEQ` exists under
  `larql-inference/src/vindex3` or `…/vindex3/opplan/exec`. The
  4096-row physical-capacity debt in the
  [KV residency contract](kv-residency-contract.md) belongs to the V2
  fused Metal path. On V3 the equivalent debt is host rows growing
  without bound on sliding layers.
- **`CanonicalKvState` holds every row twice.** It keeps the cache's
  `Array2` matrices (the storage authority) and a `Vec<Vec<f32>>` row
  view materialised from them (`larql-kv/src/vindex3/mod.rs`, `append`).
  Host K/V bytes are therefore about 2× `RowKvState`'s for the same
  history. This comes from reading the code and is not yet measured. The
  stage-2 read contract (serving slices from storage) removes the second
  copy, and that is its first forecastable saving.
- **Conv-QKV copies its whole prefix each call.** `provider.keys(..).to_vec()`
  at `exec/mod.rs:1324` and `decode.rs:820`. That is O(N) per layer per
  step. The reference executor is deliberately literal, but a read
  contract that serves views removes the copy.
- **Illustrative size, Gemma 3 4B** (read from the model config, to be
  re-read from the plan at that freeze: 34 layers, 29 sliding at
  `W = 1024` and 5 global, `kv_dim = 1024`, f32 K+V = 8 KiB per row per
  layer). At 4096 positions: full retention is 139 264 rows ≈ 1.06 GiB;
  window-resident retention is 29×1024 + 5×4096 = 50 176 rows ≈ 0.38 GiB
  (−64%). `CanonicalKvState`'s double storage doubles the baseline.

## Where the V2 engine ideas land

Porting `KvEngine` into V3 wholesale is not the plan. Each mechanism
becomes a provider (or is classified as something else) once the
contract can express it:

| V2 engine | V3 reading | needs |
|---|---|---|
| `standard` | `canonical/v1` | stage 1 |
| `standard` windowed / `windowed-checkpoint` | `window-resident/v1`, later `checkpointed/v1` | stage 2 |
| `markov-rs` | residual-canonical, K/V derived | stage 2 + a residual state region |
| `markov-rs-codec` | the same, with a codec-backed cold tier | stage 2 + REPRESENT measurement |
| `turbo-quant` | quantised rows; declares approximate, never exact | stage 2 + a correctness declaration |
| `boundary-kv` / `boundary-per-layer` | observer/archive composition over a provider | — |
| `apollo` | not continuation; a `RetrievalEngine` | — |
| `semantic-promotion` | policy above providers | — |
| `no-cache` | runtime replay mode, not a provider | — |

The `larql-kv` [state policy](../crates/larql-kv/docs/state-policy.md)
already uses the canonical / reconstructible / derivative vocabulary that
these declarations will need. Its words should be reused rather than
reinvented, but only when a provider first exercises them.
