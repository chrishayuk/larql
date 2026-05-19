# BoundaryPerLayerEngine — Specification

**Status:** 📝 Draft v0.1 (2026-05-17).
**Audience:** LARQL contributors.
**Scope:** Contract for a KV-cache engine in `larql-kv` that extends
`MarkovResidualEngine` with a per-layer codec policy, exploiting layer
fragility differences to push cold-tier compression past what a
uniform codec can safely achieve.

This is engine 3 of 3 in the boundary-engine series. The siblings are
[`BoundaryKvEngine`](boundary-kv-engine.md) (transport / save-restore,
no in-session change) and
[`MarkovResidualCodecEngine`](markov-residual-codec-engine.md) (uniform
cold-tier codec). This engine is the most expressive of the three and
requires the largest calibration investment to ship safely.

---

## 1. Purpose

`BoundaryPerLayerEngine` is a variant of
[`MarkovResidualEngine`](markov-residual-engine.md) that stores
cold-tier residuals through a *per-layer* codec policy. Different
layers get different codecs — typically `Bf16` for fragility-sensitive
mid-stack layers and `Int8Clip3Sigma` (or stronger) for tolerant
late-stack layers — selected from a per-architecture calibration sweep.

The engine exists because the mid-layer fragility result from Exp 46
(naive `Int8Clip3Sigma` at L12: top-1 = 45%, early-divergence = 80%)
is a *layer-specific* failure, not a codec-wide one. The final layer
of the same model tolerates the same codec at top-1 = 98.7%. A
per-layer policy can exploit this gap: keep `Bf16` where the model is
sensitive, drop to a lossy codec where it is not. The expected payoff,
estimated from Exp 49's per-codec sizes against a depth-fraction
fragility profile (the L22-L25 commitment-band finding referenced in
[the parent series notes](boundary-kv-engine.md)):

| Engine | Codec policy | Cold tier (Gemma 3 4B, 128K tokens) |
|---|---|---:|
| `MarkovResidualEngine` | f32 all layers | 44 GiB |
| `MarkovResidualCodecEngine { Bf16 }` | bf16 all layers | 22 GiB |
| `MarkovResidualCodecEngine { Int8Clip3Sigma }` (unsafe mid-layer) | int8 all layers | 11 GiB |
| **`BoundaryPerLayerEngine` (default policy)** | mixed per fragility | **~14 GiB** |

The mixed-policy number is bigger than the unsafe-uniform-int8 number,
which is the point: the engine trades a smaller compression ratio for
a contract that is actually defensible per layer. If the per-layer
calibration finds that no layer tolerates the lossy codec, the engine
degrades to `MarkovResidualCodecEngine { Bf16 }`-equivalent — which
is fine; it just means the per-layer policy did not earn its
complexity for that model.

The engine is not a transport format (that is `BoundaryKvEngine`'s
territory). It is not a uniform codec (that is
`MarkovResidualCodecEngine`'s territory). It is the codec policy
designed to exploit the layer-fragility structure that the other two
engines necessarily ignore.

## 2. Contract

The engine **must** satisfy the following contracts on any architecture
it claims to support.

### 2.1 Per-layer-bounded correctness contract

> For any prompt `P` and any decode step `t`, the next-token
> distribution produced by `BoundaryPerLayerEngine` is within a
> per-layer-calibrated KL bound of the distribution produced by
> `MarkovResidualEngine` on the same model at the same quantisation
> tier, given the same `(prompt, sampling_config)` tuple. The
> end-to-end bound is the composition of per-layer bounds via the
> calibration sweep in §4.7.

The contract has two levels:

1. **Per-layer codec contract.** Each `(layer, codec)` pair in the
   policy must have a calibration record bounding the layer's
   contribution to end-to-end KL.
2. **Policy composition contract.** The full policy
   `policy: Vec<(layer, codec)>` must have an end-to-end calibration
   record bounding next-token KL vs `MarkovResidualEngine` on a
   representative prompt distribution. End-to-end KL is **not** the
   sum of per-layer KLs — fragility compounds non-linearly through
   the residual stream. The composition record is the binding
   contract; per-layer records are diagnostic.

### 2.2 State-sufficiency contract

> At any decode step `t`, the engine's persistent state (hot residuals
> + cold per-layer codec payloads) is sufficient to reconstruct the
> inputs required by the model's forward pass to produce the logits
> for token `t+1`, modulo each layer's codec reconstruction error.

Inherits `MarkovResidualEngine`'s §2.2 weakened by per-layer codec
reconstruction. The codec for layer `ℓ` is the only thing that varies
between this engine and `MarkovResidualCodecEngine`.

### 2.3 Memory contract

> Hot-tier state size is identical to `MarkovResidualEngine`:
> `W × L × d × sizeof(f32)`. Cold-tier state size is the sum over
> layers of `N_cold × bytes_per_residual(policy[layer])`.

For Gemma 3 4B (`L = 34`, `d = 2560`) at `N_cold = 128K`, with a
fragility-driven default policy (`Bf16` for L0–L24, `Int8Clip3Sigma`
for L25–L33, illustrative):

```
L0..L24  (25 layers, bf16):           128K × 25 × 5120 ≈ 16 GiB
L25..L33 ( 9 layers, int8-clip3σ):    128K ×  9 × 2564 ≈  3 GiB
total cold:                                            ≈ 19 GiB
```

vs. `MarkovResidualCodecEngine { Bf16 }`'s 22 GiB: a 14% saving for the
illustrative policy, **purely from the contract surface this engine
exposes**. The actual default policy comes from §4.7's calibration; the
illustrative numbers above are not a contract.

### 2.4 Determinism contract

Inherits `MarkovResidualCodecEngine` §2.4: deterministic per layer;
codec-induced variance bounded by §2.1.

## 3. What the engine does NOT promise

- **Bit-identity with any uniform-codec engine.** The per-layer policy
  is the engine's defining feature; using it makes this engine
  semantically distinct from both `MarkovResidualEngine` and
  `MarkovResidualCodecEngine`.
- **Bit-identity with `MarkovResidualEngine`.** Only achievable by
  configuring the policy as `Bf16` everywhere — at which point use
  `MarkovResidualCodecEngine { Bf16 }` directly. There is no value in
  configuring this engine as a uniform policy.
- **Policy auto-discovery from a model alone.** The engine does not
  derive the policy from the model's weights. The policy is an output
  of the §4.7 calibration sweep, performed offline, and is a per-
  (model, target-end-to-end-KL-bound) artifact.
- **Per-position codec choice.** The codec is fixed per layer; a
  position-dependent policy (e.g., compress more aggressively at
  high-margin positions) is out of scope for v0.1. It would require
  per-position metadata in the cold tier and changes to the
  `recompute_kv` dispatch. Possible v0.2.
- **Cross-architecture policy portability.** A policy calibrated for
  Gemma 3 4B is not a policy for Llama 3 8B. Engines refuse to
  construct with a policy whose calibration record is for a different
  model fingerprint.
- **Mid-flight policy changes.** Same as
  `MarkovResidualCodecEngine` §5.3: codec choice (here: policy) is set
  at engine construction and constant for the engine's lifetime.

## 4. Architecture preconditions

Inherits all preconditions from
[`MarkovResidualEngine`](markov-residual-engine.md) §4 and
[`MarkovResidualCodecEngine`](markov-residual-codec-engine.md) §4.8
(codec is a pure function).

Additional preconditions specific to this engine:

### 4.7 Per-layer, per-architecture calibration sweep exists

> For any (architecture, policy) pair the engine is configured to
> use, a calibration record must exist that bounds the end-to-end
> next-token KL divergence vs `MarkovResidualEngine` on a
> representative prompt distribution.

The calibration is a two-pass sweep:

**Pass 1 — Per-layer fragility profile.** For each layer `ℓ` in
isolation, with all other layers at `Bf16`, run each candidate codec
and measure end-to-end KL contribution:

```
prompts:    ≥ 300 across ≥ 6 prompt classes
positions:  ≥ 30 decode steps per prompt
metric:     KL(P_codec_at_layer_ℓ || P_bf16_everywhere) per step
output:     per-(layer, codec) KL contribution
```

This pass produces the fragility profile: a per-layer ranking of how
much KL each candidate codec adds when applied to that layer alone.

**Pass 2 — Policy composition validation.** For a candidate policy
(typically the greedy policy from Pass 1: pick the most aggressive
codec for each layer that stays under a per-layer budget), run the
full sweep with the composed policy and measure actual end-to-end KL:

```
metric:     KL(P_policy || P_bf16_everywhere) per step
budget:     caller-declared end-to-end KL ceiling
output:     pass/fail vs budget; per-policy calibration record
```

Pass 2 is the binding measurement (per §2.1). Pass 1 is diagnostic —
it informs which policies are worth running through Pass 2.

### 4.8 Policy has a fingerprint

> A policy is identified by a content hash of its
> `(model_revision, Vec<(layer, codec)>)` tuple. The engine refuses to
> load a calibration record whose policy fingerprint does not match.

This prevents accidental drift: if a calibration record exists for
"policy A on model X" and a caller constructs the engine with "policy
A' on model X", the engine refuses to start, even if A and A' look
similar.

### 4.9 Precondition check is the implementation's responsibility

Same shape as `MarkovResidualEngine` §4.5 and
`MarkovResidualCodecEngine` §4.9, plus refusal cases for missing
calibration records (per §4.7) and policy fingerprint mismatches
(per §4.8).

## 5. State representation

The persistent state has two tiers, matching
`MarkovResidualCodecEngine`'s structure with per-layer codec choice.

### 5.1 Hot window

Identical to `MarkovResidualEngine` §5.1.

### 5.2 Cold tier

Per-layer codec payloads, one payload per evicted residual row, where
each layer can use a different codec:

```
cold[layer]: Vec<CodecPayload>   // codec = policy.codec_for(layer)
```

The codec for each layer is determined by the engine's `policy` at
construction; payloads do not carry codec identity in-band (the policy
is engine-wide and known to be constant).

Decode is just-in-time per cold position (same as
`MarkovResidualCodecEngine` §5.2). The decode dispatches to the codec
the policy assigned to that layer.

### 5.3 Policy

```
policy: BoundaryLayerPolicy {
    model_revision: String,
    entries: Vec<(layer: u16, codec: ColdResidualCodec)>,
    calibration_record_id: String,
}
```

The `calibration_record_id` references the §4.7 sweep output that
covers this policy. The engine refuses to construct without a valid
calibration record.

### 5.4 What is NOT in the state

Same exclusions as `MarkovResidualCodecEngine` §5.4: no decoded-
residual cache, no K/V tensors past the window, etc.

Additionally:

- **No per-position policy metadata.** The codec is a function of
  layer alone (per §3); position is not in the lookup.

## 6. Operations

### 6.1 `prefill(prompt_tokens) -> State`

Identical to `MarkovResidualCodecEngine::prefill`, except the codec
used to encode each evicted cold row is `policy.codec_for(layer)`.

### 6.2 `decode_step(state, last_token_id) -> (next_logits, new_state)`

Identical to `MarkovResidualCodecEngine::decode_step`, except cold
residuals decode through their layer's codec.

### 6.3 `check_preconditions(model, policy) -> Result<(), PreconditionViolation>`

Validates §4 against a given model and policy. Required entry point;
see §4.9.

### 6.4 Optional: state (de)serialisation

If implemented, must round-trip through §2.1's contract. The saved
state carries the policy fingerprint; loading into an engine with a
different policy is a hard refuse.

## 7. Configuration

The engine takes at minimum:

- `W: usize` (hot window size, same default as
  `MarkovResidualEngine`: 512).
- `policy: BoundaryLayerPolicy` (per §5.3). Required.
- `calibration_records: BoundaryCalibrationStore` (per §4.7). Required.
- A reference to a model handle satisfying §4.

There is no codec default — the engine cannot meaningfully default a
policy without a calibration sweep for the specific (architecture,
target-KL) pair. Callers must produce a policy via the calibration
infrastructure and pass it in explicitly.

### 7.1 Default policy generators (not contracts)

The calibration infrastructure (separate from this spec) may provide
helper policies for tested architectures:

- `BoundaryLayerPolicy::bf16_uniform(model)` — degenerate case;
  equivalent to `MarkovResidualCodecEngine { Bf16 }`. Useful as a
  sanity-check baseline.
- `BoundaryLayerPolicy::greedy_under_kl(model, kl_budget)` — runs the
  calibration store's per-layer fragility profile for `model` and
  returns the most aggressive policy that fits under `kl_budget`.

These are not part of the engine's contract; they are convenience
constructors over the calibration store. The engine validates the
output policy via the same §4.7 path regardless of how the policy was
produced.

## 8. Error modes

### 8.1 Precondition violation

§4 not satisfied. Hard error at construction.

### 8.2 Codec encode / decode failure

Same as `MarkovResidualCodecEngine` §8.2 / §8.3.

### 8.3 Calibration record missing or mismatched

Policy references a `calibration_record_id` that does not exist in
the `BoundaryCalibrationStore`, or the record's policy fingerprint
does not match the policy's. Hard refuse at construction.

### 8.4 Calibration record exceeds target KL budget

Calibration record exists but its measured end-to-end KL is above
the caller's declared budget. Hard refuse — the engine cannot
silently degrade the contract.

### 8.5 Layer-policy gap

Policy is missing an entry for some layer in `0..L`. Hard refuse;
the engine does not infer a default codec for unspecified layers.

## 9. Implementation phases

Phased to amortise calibration cost — the calibration infrastructure
is most of the engine's complexity, and it is reusable for the other
two engines.

### Phase 1 — Calibration infrastructure (precursor)

Build the §4.7 sweep harness. This is reusable: it produces both
per-layer fragility profiles (this engine's Pass 1) and uniform-codec
end-to-end records (`MarkovResidualCodecEngine`'s §4.7 records). The
harness writes into a `BoundaryCalibrationStore` (filesystem or
embedded DB).

### Phase 2 — Engine skeleton with bf16-uniform policy

Implement `BoundaryPerLayerEngine` with `BoundaryLayerPolicy::bf16_uniform`
as the sole supported policy. Verify the §2.1 contract reduces to
`MarkovResidualCodecEngine { Bf16 }`'s contract.

### Phase 3 — Per-layer fragility characterisation

Run Phase 1 sweep for Gemma 3 4B with the v0.1 codec menu (`Bf16`,
`Int8Clip3Sigma`, `AdaptiveBlockG32`, `PerGroupInt4G128`). Output: the
per-layer fragility profile for that model.

### Phase 4 — Greedy policy generator + measured policy

Implement `BoundaryLayerPolicy::greedy_under_kl`. Generate policies
at several KL budgets (e.g., 0.05, 0.1, 0.5 nats). Run Pass 2
end-to-end measurement on each. Pick the largest budget whose
end-to-end KL is below its stated bound; that is the v0.1 default
policy generator output for Gemma 3 4B.

### Phase 5 — Cross-architecture support

Re-run Phase 3 + Phase 4 for Llama 3 family. Document the per-
architecture default policy output in the engine's module-level
doc.

### Phase 6 — Per-position policy (optional, v0.2)

If per-position codec choice ever earns its complexity, it slots in
here. Out of scope for v0.1; flagged in §3 as a non-promise.

## 10. Open questions

- **Is the per-layer payoff worth the calibration cost?** The
  illustrative 14% saving in §2.3 is a guess. Phase 3/4 will produce
  the real number; if it's under (say) 20% across tested
  architectures, the engine's existence is hard to justify against
  `MarkovResidualCodecEngine { Bf16 }`. The honest pre-registered
  threshold: if Pass 2 on Gemma 3 4B at a defensible KL budget
  (≤ 0.1 nats) does not beat `Bf16`-uniform by ≥ 15% cold-tier
  saving, this engine ships as "calibration infrastructure for the
  other engines" and does not get its own user-facing
  configuration surface.
- **Layer-class shortcuts.** The MEMORY.md depth-fraction law
  (15/25/38%) suggests layer classes ("early", "middle", "late") with
  characteristic fragility profiles. Could the calibration store
  per-(architecture, layer-class) records instead of per-layer? Would
  shrink the calibration matrix substantially. Open until Phase 3
  data lands.
- **Calibration recalibration cadence.** A policy is calibrated
  against a specific model fingerprint. When the model is fine-tuned
  or quantised to a new tier, the policy is invalid. How often do
  users re-calibrate? Default answer: at every model revision; the
  calibration store keys on revision and refuses cross-revision
  loads (§4.8). Whether this is operationally tolerable depends on
  the calibration sweep's wall-clock cost — needs Phase 1
  measurement.
- **Pass-1-vs-Pass-2 divergence.** Pass 1 measures per-layer
  contributions assuming other layers are at `Bf16`. Pass 2 measures
  the composed policy. The two can diverge if codec choices interact
  (a layer that is fragile in isolation may be tolerated when an
  upstream layer is also lossy, or vice versa). The composition
  contract (§2.1) handles this correctly — Pass 2 is the binding
  measurement — but the greedy policy generator in §7.1 may produce
  policies that are over- or under-compressed when Pass 2 disagrees
  with Pass 1's predictions. Iteration (regenerate greedy policy
  using Pass 2 residual budget) may be needed; this is implementer's
  call.
- **Interaction with `BoundaryKvEngine`.** Same as
  `MarkovResidualCodecEngine` §10's last bullet: composition is
  possible; the contract weakens to the lossier of the two. Worth a
  composition-spec when both engines have shipped.

---

## Appendix A: Relationship to sibling engines

See [`boundary-kv-engine.md`](boundary-kv-engine.md) Appendix A for
the full table. This engine is the most flexible and most
calibration-hungry of the three. It is the engine to reach for when:

- A uniform-codec engine (`MarkovResidualCodecEngine`) cannot fit the
  cold tier under the memory budget at the required end-to-end KL.
- A per-architecture calibration sweep is operationally available.
- The 15%-saving threshold (per §10) is met for the target model.

## Appendix B: Why this engine is the most ambitious of the three

The other two engines have defensible v0.1 contracts on day one:

- `BoundaryKvEngine` ships with bit-identical in-session correctness
  and a cross-session contract calibrated by Exp 44 already.
- `MarkovResidualCodecEngine` ships with `Bf16` as a default that has
  a robust KL ≈ 0 contract without per-layer recalibration.

This engine ships with a contract that *depends on* the calibration
infrastructure having been built (Phase 1) and the sweep having been
run for the target model (Phase 3/4). Until then, the engine has no
defensible default policy and can only be constructed as
`bf16_uniform` — at which point it adds no value over
`MarkovResidualCodecEngine { Bf16 }`.

The engine's existence as a separate spec is justified by the
calibration infrastructure being shared with the other engines and
by the long-term ceiling: if the per-layer fragility profile of a
target model has a strong gradient (e.g., the commitment-band
finding referenced in the parent series notes), the per-layer
engine is the only way to harvest the cold-tier savings that
non-`Bf16` codecs offer mid-stack without breaking the contract
elsewhere.

If Phase 3/4 finds no useful gradient on any tested architecture,
this engine ships as calibration-only and remains a slot for a
future model that does exhibit one.
