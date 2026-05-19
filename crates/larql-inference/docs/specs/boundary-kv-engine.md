# BoundaryKvEngine — Specification

**Status:** 📝 Draft v0.1 (2026-05-17).
**Audience:** LARQL contributors.
**Scope:** Contract for a KV-cache engine in `larql-kv` that emits and
consumes `larql-boundary` frames at chunk boundaries, enabling compact
session save/restore and inter-process state transfer without changing
in-session correctness.

This spec defines *what the engine promises* and *under what
preconditions*. It deliberately does not prescribe Rust API shapes —
those are the implementer's call, subject to the contracts below.

This spec is one of three planned in the boundary-engine series:

- **`BoundaryKvEngine`** *(this spec)* — Standard KV in-session,
  boundary frames as a save/restore + transport format.
- `MarkovResidualCodecEngine` (future spec) — `markov_residual` with a
  codec layer on its cold tier.
- `BoundaryPerLayerEngine` (future spec) — per-layer codec policy with
  fragility-driven layer selection.

The three differ in *how* boundary frames participate in decode, not in
their codec/gate underpinnings. All three depend on `larql-boundary`'s
Phase 1–3 implementation. This engine targets the smallest semantic
delta from `Standard` and is the first deployable v0.1.

---

## 1. Purpose

`BoundaryKvEngine` is a KV-cache engine for transformer LMs that
behaves as `Standard` during a live decode session and additionally
emits compact, contract-bearing `larql-boundary::BoundaryFrame` objects
at chunk boundaries. Those frames are sufficient to *resume* generation
on another process (or after a restart) without retransmitting the
prompt or the live KV cache.

The engine does **not** change attention semantics during a session.
Cold context within a session is held in the same Standard KV
representation it would have under the production cache. The engine's
novelty is the cross-session / cross-process state representation: a
chunk's last-position residual (≈5 KB at int8-clip3σ under the Exp 44
gate, vs ≈360 KB/token of f16 KV at Gemma 3 4B scale) replaces
retransmission of the prompt or KV.

The engine is not a compression scheme over the live KV cache (that is
`MarkovResidualCodecEngine`'s territory). It is not a residual-stream
sufficient-statistic engine for in-session cold attention (that is
`MarkovResidualEngine`'s territory). It is a save/restore + transport
layer.

## 2. Contract

The engine **must** satisfy the following contracts on any architecture
it claims to support.

### 2.1 In-session correctness contract

> For any prompt `P` and any decode step `t` reached *within a single
> session* (no save/restore traversal), the next-token distribution
> produced by `BoundaryKvEngine` is bit-identical to the distribution
> produced by `Standard` on the same model at the same quantisation
> tier, given the same `(prompt, sampling_config)` tuple.

"Bit-identical" is stated at the level of the post-`final_norm`,
post-`lm_head` logits. Hidden-state cosine vs the `Standard` reference
path is exactly `1.000000`.

The "same quantisation tier" qualification is load-bearing, in the
same sense as `MarkovResidualEngine` §2.1 (see
[markov-residual-engine.md](markov-residual-engine.md)). The reference
target is the same model's own `Standard` path at the same tier, not an
FP16 dequantised path.

This is the strong contract. The engine ships with Standard's
guarantees inside any one session.

### 2.2 Cross-session restore contract

> For any session resumed from a `BoundaryFrame` chain produced by this
> engine, the next-token distribution at the first post-restore decode
> step satisfies the contract level carried on the resuming frame.

The contract level is the `BoundaryContract` enum defined by
`larql-boundary` (see Exp 44 calibration; the calibrated v0.1 default
is `ArgmaxNearEquivalentHighMargin` at `min_log_prob_margin = 2.16` for
Gemma 3 4B).

Specifically:

- **`Exact` / `BF16` frame:** bit-identical resume. The hidden state at
  the resume position matches what a within-session Standard decode
  would have produced at the same position.
- **`ArgmaxNearEquivalentHighMargin` (`D-@high`) frame:** the first
  ~5 post-resume tokens match Standard's greedy decode with the
  early-divergence rate calibrated by Exp 44 Track A (4.8% on the
  Frankenstein/Gemma 3 4B fixture, 95% CI ≈ 1.6%–10.7%). Total-window
  divergence over 20 tokens is **not** contracted — cascade compounds
  past step 5 (see Exp 44 README).
- **`ArgmaxNearEquivalentLowMargin` (`D-@low`) frame:** the *immediate*
  next token matches Standard's argmax. No multi-token continuation
  contract.
- **`Calibrating` frame:** not valid for cross-trust restore (see §10.6
  of `BOUNDARY_REF_PROTOCOL.md`).
- **Any frame with `BoundaryAgreement::Disagrees` or `NotChecked`:**
  hard reject. No restore contract.

The contract is established by:

1. `larql-boundary`'s codec roundtrip + gate tests (Phases 1–3, already
   shipped).
2. A new restore-parity test fixture in this crate: run Standard for N
   tokens, capture a boundary frame at token N, resume on a fresh
   engine, verify the first 5 post-resume tokens match Standard's
   greedy decode at the calibrated early-div rate.

### 2.3 State-sufficiency contract (cross-session)

> A `BoundaryFrame` chain plus the resuming process's local model
> handle is sufficient to continue generation from the chain's last
> position, without access to the original prompt or any prior KV.

This is the operational claim that makes save/restore meaningful. The
chain encodes everything the resuming process needs:

- `model_revision` + `tokenizer_revision` (verification — see
  `BOUNDARY_REF_PROTOCOL.md` §10.4).
- One or more boundary residuals (covers cold context summary).
- Optionally, the hot-tier token IDs since the last boundary (for
  bit-identical post-resume continuation; without this, the resume is
  contract-D-@high rather than contract-A).

If the chain omits hot-tier token IDs and the most recent boundary is
`D-@high`, the resume contract is `D-@high`. If the chain includes the
hot-tier tokens since the last boundary, the resume runs a normal
Standard prefill over those tokens starting from the boundary residual,
restoring contract-A within the hot window's worth of context.

### 2.4 Memory contract

> In-session steady-state memory is identical to `Standard` at the same
> hot-window cap. Persistent-state size on save is `O(N_chunks)` where
> each chunk costs the boundary-frame size (≈5 KB compressed, ≈10 KB
> bf16) plus the chunk's token IDs (4 bytes/token).

The engine does not reduce in-session memory. It reduces cross-session
*transport* and cold-storage size.

### 2.5 Determinism contract

> Inherits `Standard`'s determinism contract for in-session decode.
> Save/restore is deterministic up to the codec's contract level —
> a `D-@high` restore is not byte-identical to a `D-@high` restore from
> a different gate-accepted frame, but both satisfy the early-div bound.

## 3. What the engine does NOT promise

Explicit non-contracts so future contributors don't accidentally rely
on behaviour that was never in scope:

- **In-session KV compression.** The engine does not shrink the live
  KV cache. Use `MarkovResidualEngine` for residual-stream replacement;
  the planned `MarkovResidualCodecEngine` for KV-cold-tier codec
  compression.
- **Unbounded context.** The hot tier behaves as `Standard`; the cold
  tier behaves as `Standard` (no eviction by default). A bounded
  variant is `Standard { window_size: Some(N) }` plus this engine
  composed; this spec does not couple them.
- **Mid-layer boundary frames.** `larql-boundary` is final-layer-only
  by construction (Exp 46 showed mid-layer codec failure at every
  tested split). This engine inherits that restriction. Multi-layer
  boundary frames are not in scope; if a future codec proves
  mid-layer-safe, a `BoundaryPerLayerEngine` spec will cover it.
- **Restore from a foreign architecture or quantisation tier.** A
  `BoundaryFrame` is keyed to `model_revision`; mismatched models hard
  reject (see §8.4).
- **Training-time use.** Inference-only.
- **Speedup over Standard at any context length.** This engine is
  Standard plus an extra emit at chunk boundaries. The boundary emit
  costs one `lm_head(final_norm(residual))` per chunk (≈
  O(tokens / chunk_tokens), not O(tokens) — see
  `BOUNDARY_REF_PROTOCOL.md` §8) plus optionally one compressed-residual
  forward pass for `boundary_agreement`. At `chunk_tokens = 512` this is
  sub-1% overhead; default-off configuration matches Standard's
  performance exactly.

## 4. Architecture preconditions

The engine inherits all preconditions from `MarkovResidualEngine` §4
because:

- The cross-session restore path (§2.2) reconstructs K/V from the
  resuming residual, via the same `recompute_kv` machinery that
  `MarkovResidualEngine` uses.
- The in-session path inherits `Standard`'s preconditions (which are a
  subset of `MarkovResidualEngine`'s).

Specifically required:

- **§4.1 Residual stream is a pre-attention sufficient statistic** —
  required for the boundary residual to encode enough state to resume.
- **§4.2 Deterministic norm placement** — required for `recompute_kv`
  during post-resume warmup.
- **§4.3 Position encoding is a function of token position** — required
  so the resuming process can apply correct RoPE at the resume
  position.
- **§4.4 Attention mask is a pure function of position** — required so
  the resuming process can construct a valid mask without the original
  KV-state history.

Additional precondition specific to this engine:

### 4.5 Boundary residual is informative at chunk-end

> The final-layer residual at the chunk's last token must be a
> sufficient statistic for the immediate next-token distribution, to
> the precision asserted by the chosen contract level.

This is implied by §4.1 plus the existence of a calibrated gate from
Exp 44 for the target architecture. New architectures join the
supported set by passing per-architecture gate calibration; they do
not inherit support from Gemma 3 4B's calibration alone. See `larql-
boundary::gate::BoundaryGateConfig::min_log_prob_margin` documentation
for how to express per-architecture thresholds.

### 4.6 Precondition check is the implementation's responsibility

The engine must provide a precondition-check entry point that takes a
model handle and a target contract level, and returns either "supported"
or a structured reason for refusal. Refusal cases include:

- Architecture missing from `MarkovResidualEngine`'s supported set.
- Architecture present but no calibrated `BoundaryGateConfig` for it.
- Caller requested a contract level (e.g., `D-@high`) but only
  `D-@low` calibration exists.

It must not silently fall back to a weaker contract on an unsupported
architecture.

## 5. State representation

The engine's persistent state has three tiers:

### 5.1 Hot KV (in-session)

Identical to `Standard`'s representation. Per-layer K/V tensors for the
last hot-tier tokens. Eviction policy: none by default
(`window_size = None`); FIFO when configured with a bounded window.

### 5.2 Cold KV (in-session)

When the engine is configured with `window_size = None` (default),
there is no cold tier — all KV stays hot. When configured with
`window_size = Some(W)`, cold KV is stored exactly as `Standard
{ window_size: Some(W) }` stores it: the engine inherits, not extends,
that behaviour.

### 5.3 Boundary chain (cross-session)

A list of `BoundaryFrame` objects emitted at chunk boundaries, ordered
by `token_end`. Each frame is the on-wire serialisation defined in
`BOUNDARY_REF_PROTOCOL.md` §5. The chain is **independent** of the live
hot/cold KV — it is not consulted during in-session decode.

When emitted:

- A frame is captured every `chunk_tokens` tokens (default 512).
- Capture position: the final layer's residual at the chunk's last
  token, after `final_norm` is applied — i.e., the input to `lm_head`.
- The capture runs `larql-boundary::metadata::compute` to populate
  agreement / margin / fragility, then `larql-boundary::gate::apply`
  to assign a contract level.
- Frames are written to the configured archive (filesystem, memory
  buffer, or gRPC stream, per `BoundaryArchive` trait — see §6.4).

### 5.4 What is NOT in the state

- Per-layer or pre-attention residuals (those live in
  `MarkovResidualEngine`).
- Mid-layer boundary frames.
- Decoded `BoundaryFrame` payloads stored alongside live KV — a frame is
  emitted on capture and not read back during the same session.

## 6. Operations

The engine exposes, at minimum, the following logical operations. API
shape is the implementer's call.

### 6.1 `prefill(prompt_tokens) -> State`

Identical to `Standard::prefill`. Plus: if any chunk boundaries are
crossed during prefill, emit a frame per boundary (per §5.3).

### 6.2 `decode_step(state, last_token_id) -> (next_logits, new_state)`

Identical to `Standard::decode_step`. Plus: if this step crosses a
chunk boundary (i.e., `state.next_position % chunk_tokens == 0` after
the step), emit a frame.

### 6.3 `resume(boundary_chain, optional_hot_tokens) -> State`

Reconstructs a decode state from a previously emitted boundary chain.

Required behaviour:

1. Verify every frame's `model_revision` and `tokenizer_revision`
   match the live model. Hard reject on mismatch.
2. Verify the chain is contiguous: `frame[i+1].token_start ==
   frame[i].token_end`. Hard reject on gap.
3. Verify every frame's contract level is acceptable for the
   caller's stated resume contract (a caller asking for `D-@high`
   restore must reject chains with any `D-@low`, `Calibrating`,
   `Unknown`, or `Disagrees` frame).
4. Decode the resuming residual (the last frame's payload) via
   `larql-boundary::codec`.
5. From that residual, run `MarkovResidualEngine::recompute_kv` on
   *only the resume position* to bootstrap a single K/V row, which
   becomes the hot tier's seed.
6. If `optional_hot_tokens` is provided, run `Standard::prefill` over
   those tokens with the seeded K/V state. The resume contract is
   then `A` (bit-identical to Standard) for any decode step after the
   prefill completes.
7. If `optional_hot_tokens` is omitted, the engine is positioned at
   the resuming residual; the next `decode_step` produces a
   distribution under the chain's last-frame contract level.

### 6.4 `check_preconditions(model, target_contract) -> Result<(), PreconditionViolation>`

Validates §4 against a given model and the caller's target contract
level. Required entry point; see §4.6.

### 6.5 Archive interface

An archive trait is the engine's only interaction with persistent
storage. The trait surface is intentionally small:

- `append(frame: BoundaryFrame) -> Result<()>` — durably write a frame.
- `load_chain(sequence_id: &str) -> Result<Vec<BoundaryFrame>>` — read
  the chain for a sequence.
- `fsync()` — durability barrier; called per §10's eviction-order
  discussion in `BOUNDARY_REF_PROTOCOL.md` §13 Option A.

Concrete archive implementations (filesystem, in-memory, gRPC) live
outside this spec.

## 7. Configuration

The engine takes at minimum:

- `window_size: Option<usize>` (Standard's hot-tier cap). Default
  `None` (no eviction).
- `chunk_tokens: usize` (boundary emission stride). Default `512`.
  Constraint: `chunk_tokens ≥ 1`.
- `compression: BoundaryCompression` (per `larql-boundary`). Default
  `Int8Clip3Sigma` (compressed) with `Bf16` fallback per gate.
- `gate: BoundaryGateConfig` (per `larql-boundary`). Default per
  architecture; engine refuses construction if no calibration exists
  for the model's architecture.
- `archive: Box<dyn BoundaryArchive>` (per §6.4). Required.
- A reference to a model handle satisfying §4.

The engine does not take a sampling config — sampling is the caller's
concern, as with `MarkovResidualEngine`.

## 8. Error modes

Implementations must distinguish at least:

### 8.1 Precondition violation

§4 not satisfied for the requested contract level. Hard error at engine
construction.

### 8.2 Archive failure during emit

`archive.append` returned an error. The engine **must** treat this as a
hard error and propagate. Silent drop of a frame breaks the
state-sufficiency contract (§2.3) and turns a recoverable session into
an unrecoverable one.

Implementations should follow `BOUNDARY_REF_PROTOCOL.md` §13 Option A
(durability-first): emit the frame and fsync before allowing the
session to free any state that would be needed to recompute the frame
on retry. For this engine specifically, that means: emit + fsync before
allowing the next decode step to proceed past the chunk boundary.

### 8.3 Resume verification failure

Any of §6.3 steps 1–3 failed. Hard error; the engine does not attempt a
partial restore. Caller may retry with a different chain or fall back
to full-prefill resume.

### 8.4 Model / tokenizer revision mismatch

`frame.model_revision != local.model_revision` (or tokenizer).
Hard error. `model_revision` is the canonical identity (a content hash
of weights). `architecture` is a human-readable label only and **must
not** be used for matching — two checkpoints of the same architecture
can have different residual geometry. See `BOUNDARY_REF_PROTOCOL.md`
§10.4 for the model migration interaction (in particular,
SmoothQuant-style pre-normalization is a model migration, not a codec
swap).

### 8.5 Resume from contract-incompatible chain

Caller requested `D-@high` resume; chain contains a `D-@low` or weaker
frame. Hard error. The engine does not auto-degrade contract level —
the contract is the caller's contract with the receiver, not an
engine-internal preference.

### 8.6 Frame-codec roundtrip mismatch

If a decoded frame's residual hash does not match `frame.residual_hash`
(present only when emit set it), this is a corruption signal. Hard
error in debug builds; in release, the implementation's choice, but it
must not produce non-contracted output.

## 9. Implementation phases

Phased to maximise reuse of existing crates and minimise new code paths.

### Phase 1 — Engine skeleton + Standard parity

Implement `BoundaryKvEngine` as a thin wrapper over the existing
`Standard` engine machinery. Boundary capture stubbed (no frames
emitted). Verify in-session correctness contract (§2.1) by passing the
existing `dispatch_parity` test suite.

### Phase 2 — Boundary capture

Wire `chunk_tokens`-triggered capture into the decode loop. Reuse
`larql-boundary::metadata::compute` for the metadata and
`larql-boundary::gate::apply` for the contract assignment. Emit to a
`Vec<BoundaryFrame>` in-memory archive by default; provide a
`FilesystemArchive` for durability.

### Phase 3 — Resume path

Implement `resume(chain, optional_hot_tokens)`. Reuse
`MarkovResidualEngine::recompute_kv` for the post-restore K/V seeding
(per §6.3 step 5). Add the restore-parity test fixture (per §2.2).

### Phase 4 — Calibration & cross-arch support

Run Exp 44 Track A calibration for Llama 3 family (likely already
compatible per `markov-residual-engine.md` §10) and any other target
arch. Document per-architecture `BoundaryGateConfig` defaults in the
engine's module-level doc.

### Phase 5 — gRPC archive (optional)

`GrpcArchive` implementation for the boundary-frame-as-grid-transfer
use case. Out of scope for v0.1; specced separately when grid
integration ships.

## 10. Open questions

Not blocking the spec, but worth tracking:

- **Hot-tokens-since-last-boundary serialisation.** §6.3 step 6 wants
  raw hot tokens to upgrade the resume contract from `D-@high` to `A`.
  Should these ride inside the last `BoundaryFrame` (extending the
  frame schema) or alongside the chain (separate transport object)?
  Leaning toward "separate object" so `BoundaryFrame` stays a pure
  fixed-position checkpoint.
- **Bounded-window composition.** `Standard { window_size: Some(W) }`
  combined with `BoundaryKvEngine` is the obvious "bounded memory +
  cross-session checkpoints" engine. Does it ship as a separate
  `BoundedBoundaryKvEngine` or as the same engine with a
  `window_size: Some(_)` config? Leaning toward the latter (same
  engine, config-driven) on the principle that windowing is a
  Standard-engine feature and this engine inherits Standard's
  configuration surface.
- **Frame-emission overhead measurement.** §3's claim of "sub-1% at
  `chunk_tokens = 512`" is an estimate. Needs measurement on real
  hardware in Phase 2.
- **Restore-then-evict interaction.** What happens if a caller resumes
  from a chain and then immediately overflows the hot window? The
  resume residual lives at the chain's last position; if the hot
  window evicts it, does the engine emit a new boundary frame for it?
  The cleanest answer is yes — emit at eviction time, same as for
  decoded-into-window positions — but this needs to be made explicit.
- **Chain compaction.** A long session emits many boundary frames; for
  cold storage, only the most recent N may matter. Is chain compaction
  an engine concern or an archive concern? Leaning toward archive
  concern (engine emits per chunk; archive decides retention).
- **Calibrating-frame interaction with archive.** Frames emitted in
  `calibration_mode = true` carry `BoundaryContract::Calibrating` and
  must not cross trust boundaries (per `BOUNDARY_REF_PROTOCOL.md`
  §10.6). The engine should either (a) refuse to archive
  `Calibrating` frames to a non-local archive, or (b) require the
  archive to declare its trust boundary. Decision deferred to Phase 2
  implementation.

---

## Appendix A: Relationship to sibling engines

| Engine | What it replaces | In-session contract | Cross-session contract |
|---|---|---|---|
| `Standard` | — | reference | none |
| `MarkovResidualEngine` | live KV with per-layer residuals | bit-identical (KL=0) | not in scope |
| `BoundaryKvEngine` *(this spec)* | nothing in-session; transport for restore | bit-identical via Standard | `A` / `D-@high` per frame |
| `MarkovResidualCodecEngine` *(future)* | `MarkovResidualEngine`'s cold tier with a codec | bounded-KL per layer | not in scope |
| `BoundaryPerLayerEngine` *(future)* | per-layer codec policy | per-layer contract | per-layer per-frame |
| `ApolloEngine` | retrieval-augmented compiled facts via residual injection | task accuracy | n/a |
| `UnlimitedContextEngine` | live KV with per-window checkpoints | exact within window | not in scope |
| `TurboQuantEngine` | live KV with WHT+Lloyd-Max codec | cos ≈ 0.991 | not in scope |

`BoundaryKvEngine` is the only engine whose primary value lives at the
cross-session boundary rather than in-session. The other two planned
boundary engines (`MarkovResidualCodecEngine`,
`BoundaryPerLayerEngine`) push boundary-frame mechanics into in-session
state and accept the resulting correctness-contract weakening.

## Appendix B: Why this engine first

Of the three planned boundary engines, this one has the cleanest
implementation path because it can be built as a near-pure additive
layer over `Standard`: most of the v0.1 contract (§2.1) is satisfied by
delegation. The novel mechanism — chunk-boundary capture + emit + resume —
is isolated to two new code paths (capture in §6.1/§6.2, resume in
§6.3). The codec and gate themselves are already shipped in
`larql-boundary`.

The other two engines require modifying live KV representation, which
weakens the in-session contract and forces every downstream consumer
(attention, FFN, profiling) to handle a new state type. Specifying and
implementing them on top of this one's calibration and archive
infrastructure is the right ordering.
