# MarkovResidualCodecEngine — Specification

**Status:** 📝 Draft v0.1 (2026-05-17).
**Audience:** LARQL contributors.
**Scope:** Contract for a KV-cache engine in `larql-kv` that extends
`MarkovResidualEngine` with a codec layer on its cold tier, trading
bounded fidelity loss for cold-tier memory reduction.

This is engine 2 of 3 in the boundary-engine series. The siblings are
[`BoundaryKvEngine`](boundary-kv-engine.md) (transport / save-restore,
no in-session change) and `BoundaryPerLayerEngine` (per-layer codec
policy, specced in [boundary-per-layer-engine.md](boundary-per-layer-engine.md)).
This engine sits in the middle: it compresses live state, but uniformly
across all layers, accepting the per-layer fragility cost in exchange
for a smaller and simpler design surface than the per-layer engine.

---

## 1. Purpose

`MarkovResidualCodecEngine` is a variant of
[`MarkovResidualEngine`](markov-residual-engine.md) that stores
cold-tier residuals through a codec rather than as raw `f32`. The hot
tier is unchanged. The cold tier shrinks by the codec's compression
ratio (2× for `Bf16`, 4× for `Int8Clip3Sigma`), at the cost of a
bounded fidelity loss that breaks `MarkovResidualEngine`'s
bit-identical contract.

The engine exists because `MarkovResidualEngine`'s cold tier is the
dominant memory cost at long context: 4 bytes per layer per cold
position times 34 layers times d=2560 = ~350 KB/token for Gemma 3 4B.
At 128K cold tokens that is ~44 GB before any K/V is reconstructed.
A 2× saving (bf16) is ~22 GB — the difference between a model that
fits and one that does not on a 64 GB consumer machine.

The engine is not a transport format (that is `BoundaryKvEngine`'s
territory). It is not a per-layer policy engine (that is
`BoundaryPerLayerEngine`'s territory). It is uniform compression of
the cold residual tier, with one codec choice for all layers.

## 2. Contract

The engine **must** satisfy the following contracts on any architecture
it claims to support.

### 2.1 Bounded-KL correctness contract

> For any prompt `P` and any decode step `t`, the next-token
> distribution produced by `MarkovResidualCodecEngine` is within a
> calibrated KL bound of the distribution produced by
> `MarkovResidualEngine` on the same model at the same quantisation
> tier, given the same `(prompt, sampling_config)` tuple. The bound
> depends on the configured codec, and is **not** zero for any lossy
> codec.

The bound is per-codec, per-architecture, established by a calibration
sweep (see §4.7). Initial bounds from Exp 43/49 on Gemma 3 4B final
layer:

| Codec | Bytes/vec at d=2560 | KL bound (nats) | Top-1 |
|---|---:|---:|---:|
| `Bf16` | 5120 | ≈ 0 (lossless under bf16 forward) | 100% |
| `Int8Clip3Sigma` | 2564 | ≤ 2.0 (per-position, final-layer) | ≥ 93% |
| `AdaptiveBlockG32` | 1637 | ≤ 0.05 | ≥ 89% |
| `PerGroupInt4G128` | 1364 | ≤ 0.20 | ≥ 80% |

**These bounds are final-layer-only and do not transfer to mid-layer
residuals.** Exp 46 measured `Int8Clip3Sigma` at L12 (Gemma 3 4B):
top-1 = 45%, early-divergence = 80%. A naive application of any of
the above codecs to every layer's cold residuals is expected to be
significantly weaker than the per-codec bound. Per-architecture
calibration (§4.7) must measure the engine's actual end-to-end
contract, not assume it from the final-layer numbers.

This is the load-bearing reason the v0.1 default is `Bf16`: it is the
only codec in the table whose KL bound is robust mid-layer. The other
codecs are present in the configuration surface for users who can
tolerate the weaker contract or who have run their own per-layer
calibration.

### 2.2 State-sufficiency contract

> At any decode step `t`, the engine's persistent state (hot residuals
> + cold codec payloads) is sufficient to reconstruct the inputs
> required by the model's forward pass to produce the logits for
> token `t+1`, modulo the codec's reconstruction error.

This is `MarkovResidualEngine`'s §2.2 weakened by the codec
reconstruction step. The engine still does not retain K/V tensors past
the window in steady state; the cold tier still encodes residuals
rather than K/V; the codec is the only difference.

### 2.3 Memory contract

> Hot-tier state size is identical to `MarkovResidualEngine`:
> `W × L × d × sizeof(f32)`. Cold-tier state size is
> `N_cold × L × bytes_per_residual(codec)` where
> `bytes_per_residual` is codec-defined.

For Gemma 3 4B at `W = 512`, `N_cold = 128K`:

| Codec | Hot tier | Cold tier | Total |
|---|---:|---:|---:|
| `MarkovResidual` (f32 cold) | 170 MiB | 44 GiB | 44 GiB |
| `Bf16` cold | 170 MiB | 22 GiB | 22 GiB |
| `Int8Clip3Sigma` cold | 170 MiB | 11 GiB | 11 GiB |
| `AdaptiveBlockG32` cold | 170 MiB | 7 GiB | 7 GiB |

The hot tier dominates only at short context. The cold tier dominates
beyond ~5K tokens — which is where this engine starts to be worth its
correctness cost.

### 2.4 Determinism contract

> Inherits `MarkovResidualEngine`'s determinism contract. The codec
> itself is deterministic; codec-induced variance in the output
> distribution is bounded by §2.1.

## 3. What the engine does NOT promise

- **Bit-identity with `MarkovResidualEngine`.** This is the explicit
  delta from the parent engine; any caller relying on KL=0 must use
  `MarkovResidualEngine` directly.
- **Bit-identity with `Standard`.** Inherits `MarkovResidualEngine`'s
  in-session bit-identical claim only when configured with
  `codec = Bf16` *and* the model's forward pass is bf16 — otherwise
  the codec adds error on top of `MarkovResidualEngine`'s contract.
  The bf16+bf16 case is the only one whose contract reduces to KL≈0;
  any other configuration is bounded-KL per §2.1.
- **Cross-session transport.** This engine's cold codec payloads are
  on-disk encoding for *one process's own state*, not a transport
  protocol object. There is no `BoundaryContract` taxonomy attached;
  there is no model-revision verification on payload load. Use
  `BoundaryKvEngine` if you need transport.
- **Per-layer codec policy.** This engine applies one codec to every
  cold layer. Per-layer choice lives in `BoundaryPerLayerEngine`.
- **Hot-tier compression.** Hot residuals stay `f32`. Compressing the
  hot tier would weaken the bit-identical-to-itself property the
  engine needs for `recompute_kv` to produce stable K/V across decode
  steps within a chunk.
- **Speedup over `MarkovResidualEngine`.** The codec adds work on
  cold-residual write (encode) and cold-residual read during
  `recompute_kv` (decode). Cold-tier traffic is reduced, which can
  offset the codec cost at very long context, but no speedup is
  contracted at any length.

## 4. Architecture preconditions

Inherits all preconditions from
[`MarkovResidualEngine`](markov-residual-engine.md) §4 unchanged.

Additional preconditions specific to this engine:

### 4.7 Per-architecture, per-codec calibration exists

> For any (architecture, codec) pair the engine is configured to use,
> a calibration record must exist that bounds the end-to-end
> next-token KL divergence vs `MarkovResidualEngine` on a representative
> prompt distribution.

The calibration sweep:

```
prompts:    ≥ 300 across ≥ 6 prompt classes
positions:  ≥ 30 decode steps per prompt
metric:     KL(P_codec || P_markov_residual), per step
threshold:  per-codec, declared in the engine's config
```

The default thresholds for v0.1 are pre-registered:

- `Bf16`: ≤ 0.01 nats (essentially lossless against an f32 forward;
  exact if the forward is bf16).
- `Int8Clip3Sigma`: ≤ 0.5 nats (Contract C territory; not yet
  measured mid-layer).
- `AdaptiveBlockG32`: ≤ 0.1 nats (extrapolated from Exp 49 final-layer
  + a 2× degradation factor for mid-layer; **assumption requiring
  measurement before promotion to default**).

Engines configured with a codec whose calibration record exceeds the
threshold for the target model **must** refuse to construct. They
**must not** silently degrade contract.

### 4.8 Codec is a pure function

> The codec's encode and decode functions must be pure functions of
> their inputs — no internal state, no learned parameters that drift
> across encodes, no calibration that varies with prompt history.

This is satisfied by all v0.1 codecs (the calibration is per-frame,
not per-corpus). SmoothQuant-style pre-normalisation is **not** a
pure-function codec — it modifies the model weights and is therefore a
model migration, not a codec swap (see `BOUNDARY_REF_PROTOCOL.md`
§10.4). It is explicitly out of scope for this engine.

### 4.9 Precondition check is the implementation's responsibility

Same shape as `MarkovResidualEngine` §4.5: a precondition-check entry
point that takes a model handle plus a `codec` choice and returns
"supported" or a structured reason for refusal. Refusal cases include
the new ones in §4.7 and §4.8 above.

## 5. State representation

The persistent state has two tiers, matching `MarkovResidualEngine`'s
structure with the cold tier replaced.

### 5.1 Hot window

Identical to `MarkovResidualEngine` §5.1. Per-layer buffer of up to
`W` residual rows, `f32`, shape `[W, d]`. FIFO eviction into the
cold tier.

### 5.2 Cold tier

Per-layer codec payloads, one payload per evicted residual row:

```
cold[layer]: Vec<CodecPayload>
```

Each `CodecPayload` is the codec's wire form for one residual vector
(e.g., `(scale: f32, bytes: Vec<i8>)` for `Int8Clip3Sigma`). Payload
format is codec-defined.

On `recompute_kv` for cold positions:

1. Decode `cold[layer][position]` → `f32` residual.
2. Apply the existing `recompute_kv` machinery exactly as
   `MarkovResidualEngine` does.

The decode is just-in-time (per decode step, per cold position
touched). Caching the decoded f32 residuals across decode steps would
defeat the memory saving; the engine **must not** cache them.

### 5.3 Codec selection

The codec is set at engine construction time and is constant for the
engine's lifetime. Changing codecs mid-session is out of scope (it
would require re-encoding the entire cold tier and breaks the contract
mid-flight).

### 5.4 What is NOT in the state

Same exclusions as `MarkovResidualEngine` §5.3: no K/V tensors past
the window, no attention outputs, no position indices.

Additionally:

- **No persistent decoded-residual cache.** Decoded cold residuals are
  per-step transients.
- **No codec metadata per payload.** The codec is engine-wide; payloads
  do not need to carry codec identity. (If a future engine variant
  allows per-layer codec choice, that is `BoundaryPerLayerEngine`'s
  territory, not this one's.)

## 6. Operations

The engine exposes the same logical operations as
`MarkovResidualEngine`:

### 6.1 `prefill(prompt_tokens) -> State`

Runs the forward pass over `prompt_tokens`, populating the hot window
and (if `len(prompt_tokens) > W`) the cold tier. Cold-tier rows are
codec-encoded on eviction; the encode runs synchronously inline with
the eviction step.

### 6.2 `decode_step(state, last_token_id) -> (next_logits, new_state)`

Identical to `MarkovResidualEngine::decode_step` except cold residuals
are codec-decoded just-in-time during `recompute_kv`. New hot residuals
that overflow into the cold tier are codec-encoded at the eviction step.

### 6.3 `check_preconditions(model, codec) -> Result<(), PreconditionViolation>`

Validates §4 against a given model and codec choice. Required entry
point; see §4.9.

### 6.4 Optional: state (de)serialisation

If implemented, must round-trip through the §2.1 contract. The cold
tier serialises trivially (already a byte buffer); the hot tier follows
`MarkovResidualEngine` §6.4 (model fingerprint required, hard reject
on mismatch). The codec used to encode the cold tier is part of the
fingerprint.

## 7. Configuration

The engine takes at minimum:

- `W: usize` (hot window size, same default as
  `MarkovResidualEngine`: 512).
- `codec: ColdResidualCodec`. Default `Bf16`. Options at v0.1:
  `Bf16`, `Int8Clip3Sigma`, `AdaptiveBlockG32`, `PerGroupInt4G128`.
- A reference to a model handle satisfying §4.

The engine does not take a sampling config.

### 7.1 Why `Bf16` is the default

`Bf16` is the only v0.1 codec whose §4.7 calibration is robust without
a per-layer sweep, because the f32 → bf16 → f32 roundtrip introduces
error at the float-precision floor rather than at the codec's
quantisation grid. The 2× memory saving over `MarkovResidualEngine`
is the engine's load-bearing value at v0.1.

The other codecs are present for users who:

- Have run per-architecture calibration and accept the resulting
  weaker contract.
- Have measured that their workload tolerates the codec's
  end-to-end KL.
- Are running experimental configurations on a model where the
  fragility profile is being characterised.

Defaulting to `Bf16` keeps the engine's out-of-the-box contract
defensible against the Exp 46 finding without forcing every user to
run their own sweep before they can ship.

## 8. Error modes

Implementations must distinguish at least:

### 8.1 Precondition violation

§4 not satisfied. Hard error at construction.

### 8.2 Codec encode failure

`codec.encode(residual)` returned an error (e.g., NaN in input, codec
internal invariant failure). Hard error; the engine must not silently
fall back to a different encoding or drop the residual.

### 8.3 Codec decode failure

`codec.decode(payload)` returned an error. Hard error during
`recompute_kv`. The engine must not return synthesised K/V from a
failed decode.

### 8.4 Calibration threshold violation at construction

A codec configured whose §4.7 calibration is above the threshold for
the target architecture. Hard refuse at engine construction.

### 8.5 Calibration record missing

A codec configured for which no §4.7 calibration record exists for the
target architecture. Hard refuse; explicit message naming the
(architecture, codec) pair that needs calibration. The engine **must
not** silently fall back to `Bf16` — the user asked for a specific
codec and is owed a clear failure when the calibration is missing.

## 9. Implementation phases

### Phase 1 — Engine skeleton with `Bf16` codec

Implement `MarkovResidualCodecEngine` as a variant of
`MarkovResidualEngine` that takes a `ColdResidualCodec` parameter.
Initial codec: `Bf16` only. Verify the §2.1 contract: KL ≈ 0 vs
`MarkovResidualEngine` on the existing parity fixtures.

### Phase 2 — Per-codec calibration infrastructure

Build the §4.7 calibration sweep harness. Run it for Gemma 3 4B with
`Bf16` (expected to pass) and `Int8Clip3Sigma` (expected to fail
mid-layer, to validate the calibration mechanism actually catches
unsafe configurations).

### Phase 3 — Promote `Int8Clip3Sigma` only if calibration passes

If the Phase 2 calibration of `Int8Clip3Sigma` lands above its
threshold (likely), it stays gated behind a feature flag. The codec is
present in the configuration surface but the engine refuses to
construct without an explicit acknowledgement from the caller that the
contract is weaker than §2.1's default threshold.

### Phase 4 — Cross-architecture support

Re-run the calibration sweep for Llama 3 family and any other target
architecture. Document per-architecture calibration records in the
engine's module-level doc.

### Phase 5 — Additional codecs (optional)

`AdaptiveBlockG32`, `PerGroupInt4G128` from Exp 49. These need their
own calibration sweep and likely their own per-layer characterisation
before they can be defaulted on; they ship as opt-in even after
calibration.

## 10. Open questions

- **Mid-layer codec viability.** The single biggest open question: do
  any of the non-`Bf16` codecs survive per-layer calibration for any
  architecture? Exp 46 says no for the final-layer-calibrated
  `Int8Clip3Sigma`. Possible mitigations: (a) per-layer recalibration
  of the codec's parameters (e.g., a different `clip3σ` threshold per
  layer); (b) layer-class codec mapping (different codec for "early"
  vs "late" layers — but that is `BoundaryPerLayerEngine`'s
  territory, not this one's). For v0.1, the honest answer is
  "non-`Bf16` codecs are experimental and have no calibrated default
  for any architecture."
- **Decode-cache vs decode-jit tradeoff.** §5.2 mandates no decoded-
  residual cache, but if cold-residual touches per step are
  bottlenecked by decode cost, a tiny LRU might be worth it.
  Measurement-first: run Phase 1 with no cache and see whether the
  hot path is decode-bound.
- **Sliding-window-attention interaction.** Gemma 3 uses sliding-window
  attention on some layers; cold residuals past the sliding window are
  unreferenced. The engine still stores them (they could become
  referenced if the window grows), but a future optimisation could
  drop them. Out of scope for v0.1.
- **Codec choice persistence on save/restore.** If state
  serialisation ships (§6.4), the codec used to encode cold payloads
  must be in the saved-state fingerprint. A `Bf16`-encoded saved state
  cannot be loaded into an engine configured for `Int8Clip3Sigma`
  without re-encoding the cold tier first.
- **Interaction with `BoundaryKvEngine`.** The two engines are
  composable: this engine for in-session compression, `BoundaryKvEngine`
  for cross-session transport. The compositional contract is the
  weaker of the two — which means the headline cross-session
  guarantee is the codec's §2.1 bound, not `BoundaryKvEngine`'s
  bit-identical-in-session claim. Worth specifying when the
  composition spec is written.

---

## Appendix A: Relationship to sibling engines

See [`boundary-kv-engine.md`](boundary-kv-engine.md) Appendix A for
the full table. This engine sits between `MarkovResidualEngine` and
`BoundaryPerLayerEngine`: it accepts the bounded-KL cost of cold-tier
compression but applies one codec uniformly across all layers.

## Appendix B: Why this engine is the riskiest of the three

Of the three planned boundary engines, this one has the weakest
defensible contract at v0.1:

- `BoundaryKvEngine` preserves `Standard`'s bit-identical contract
  in-session and is honest about the cross-session contract being
  weaker.
- `BoundaryPerLayerEngine` has the freedom to choose `Bf16` on
  fragile layers and bound the loss precisely.
- This engine applies one codec to every layer, which means either
  (a) the codec is `Bf16` and the engine's value is a flat 2× cold
  saving with KL ≈ 0, or (b) the codec is lossy and the contract
  weakens dramatically due to mid-layer fragility (Exp 46).

The v0.1 default is (a). The engine's existence as a separate spec is
justified by (1) the simpler implementation surface (one codec, no
per-layer policy), (2) the 2× cold-saving being meaningful on its own
for memory-bound deployments, and (3) the existence of users who
prefer the simpler operational model even at the cost of less
flexibility.

If Phase 2 calibration shows that no non-`Bf16` codec survives mid-
layer for any tested architecture, this engine's long-term role
becomes "the simple 2× cold-saver" — a worthwhile addition, but
narrower than the engine's design surface implies. That is the
expected outcome and the spec is written to be defensible at that
outcome.
