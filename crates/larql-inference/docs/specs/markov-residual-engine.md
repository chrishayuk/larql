# MarkovResidualEngine — Specification

**Status:** ✅ Shipped (2026-05-09 extraction, 2026-05-16 Q4K hot-path migration).
**Audience:** LARQL contributors.
**Scope:** Contract for a KV-cache-free decode engine in `larql-kv`,
currently validated on Gemma 3 4B, designed to admit other architectures
behind explicit preconditions.

This spec defines *what the engine promises* and *under what preconditions*.
It deliberately does not prescribe Rust API shapes — those are the
implementer's call, subject to the contracts below.

---

## 1. Purpose

`MarkovResidualEngine` is an alternative decode path for transformer LMs
that replaces the per-token K/V cache with the residual stream itself as
the persistent inference state. The production implementation lives in
[`larql_kv::engines::markov_residual`](../../../larql-kv/src/engines/markov_residual/)
(`rs_prefill`, `rs_decode_step`, plus the Q4K hot-path routing through
`attention_decode_step_native` + `ffn_decode_step_native`). It originated
as a research prototype in the retired `kv-cache-benchmark` crate
(2026-05-09 extraction) and was promoted to a first-class engine in
`larql-kv`.

The engine is not a compression scheme layered over a KV cache. It is a
different answer to the question "what state must persist between decode
steps?" — and the answer it gives happens to compress well as a side
effect.

## 2. Contract

The engine **must** satisfy the following contract on any architecture it
claims to support:

### 2.1 Correctness contract

> For any prompt `P` and any decode step `t`, the next-token distribution
> produced by `MarkovResidualEngine` is bit-identical to the distribution
> produced by the reference Standard KV decode path **on the same model
> at the same quantisation tier**, given the same `(prompt,
> sampling_config)` tuple.

"Bit-identical" is stated at the level of the post-`final_norm`,
post-`lm_head` logits. Equivalently: hidden-state cosine vs the reference
path is exactly `1.000000` (cos = 1 forces logit identity under
deterministic final norm + lm_head, which is strictly stronger than
`KL = 0.0` on the output distribution).

The "same quantisation tier" qualification is load-bearing. Running the
engine against a quantised model (Q4_K, FP8, etc.) is a supported
configuration; the comparison target is *that quantised model's own
Standard KV path*, not an FP16 dequantised reference. The engine does
not promise bit-identity across quantisation tiers — that would require
a quantisation-invariant residual representation, which is not part of
this contract and is not measured by the experiments backing it.

The contract is established by the engine's parity tests in
`crates/larql-kv/src/engines/markov_residual/` (`#[ignore]`'d real-model
fixtures) and the dispatch parity gate in
`crates/larql-kv/tests/dispatch_parity.rs`. Any implementation claiming
to satisfy this spec must pass an equivalent suite. New quantisation
tiers join the supported set by adding a same-tier comparison fixture;
they do not inherit support from FP16 validation alone. (The original
`kv-cache-benchmark::tests::test_real_model` test set was retired in
2026-05-16 along with the crate.)

### 2.2 State-sufficiency contract

> At any decode step `t`, the engine's persistent state is sufficient to
> reconstruct the inputs required by the model's forward pass to produce
> the logits for token `t+1`.

This is the actual theoretical claim. The KV cache is *one* sufficient
state; the residual stream (under preconditions in §4) is *another*. The
contract forbids any implementation that achieves correctness by secretly
caching K/V tensors under a different name.

### 2.3 Memory contract

> Persistent state size is `O(W + N_cold)` where `W` is the hot-window
> cap and `N_cold` is the number of tokens beyond the window. The hot
> window contributes a fixed ceiling; the cold tier grows at
> 4 bytes/token.

Specifically, on any supported architecture with `L` layers and hidden
dim `d`:

- **Hot window ceiling:** `W × L × d × sizeof(f32)`, plus implementation
  bookkeeping.
- **Cold tier growth:** 4 bytes per token past `W`.
- **No K/V tensors** retained past the window in the steady-state
  representation. Transient K/V during `recompute_kv` is permitted and
  expected.

For Gemma 3 4B at `W=512`: bare-tensor floor = `512 × 34 × 2560 × 4 =
178 257 920` bytes ≈ **170 MiB**. This is the f32 residual storage
footprint only. The user-visible total ceiling — including per-layer
alignment, cold tier index structures, and any auxiliary state — is
reported by the implementation, not this spec. Do not anchor on
"170 MiB ± a few percent" as the deployable footprint; the formula
gives the floor, the implementation gives the ceiling.

For any other supported architecture, the bare-tensor floor is computed
from that architecture's `L` and `d`.

### 2.4 Determinism contract

> Given the same `(model, prompt, sampling_config, rng_seed)`, the
> engine produces byte-identical outputs across runs on the same
> hardware + BLAS implementation.

Non-determinism from GPU reduction order, mixed-precision accumulation,
or BLAS threading is in scope for the implementer to handle — but the
contract is that the engine does not *add* non-determinism on top of the
reference forward pass.

## 3. What the engine does NOT promise

Explicit non-contracts, so future contributors don't accidentally rely on
behaviour that was never in scope:

- **Cross-architecture generality.** The contract holds only on supported
  architectures (§4). Adding an architecture means passing its
  precondition check, not hoping it works.
- **Cross-model state portability.** Residuals captured from model `A`
  are not meaningful as state for model `B`, even if `A` and `B` share
  hidden dim. State is model-specific.
- **Unbounded context.** The hot window is `W`; the cold tier stores
  token IDs only, and cold-replay cost grows with `N_cold`. This engine
  bounds *memory*, not *cold-replay compute*. Tier 2 (`UnlimitedContextEngine`)
  and Tier 3 (`ApolloEngine`) are the escape hatches for unbounded
  context; they live in sibling engines and are out of scope here.
- **Training-time use.** The engine is inference-only. Gradient flow
  through cold-replayed residuals is not supported and not planned.
- **Speedup over Standard KV at short context.** Measured wall-clock on
  Gemma 3 4B shows parity-to-slight-advantage at short context because
  the engine doesn't carry a growing K/V tensor. Large advantages are
  expected only at long context, where Standard KV's O(N) growth starts
  to hurt.

## 4. Architecture preconditions

The bit-perfect claim is not a statement about transformers in general.
It is a statement about architectures whose forward pass satisfies a
specific set of structural properties. An implementation **must**
validate these properties (statically at compile time where possible,
dynamically at engine construction otherwise) before claiming to support
a given model.

### 4.1 Residual stream is a pre-attention sufficient statistic

The residual stream entering layer `ℓ` must contain all information the
layer's attention + FFN need to produce the residual stream entering
layer `ℓ+1`, conditional on the token being decoded.

**Why this is a precondition:** if a layer reads state from anywhere
other than its residual stream input and the current token (e.g. from a
persistent memory module, a retrieval-augmented external cache, or an
attention sink outside the K/V cache), then cold-replaying residuals is
not sufficient to reproduce the forward pass. The engine's correctness
contract silently breaks.

**How to check:** inspect the model's forward pass. Any read from
persistent state other than `(residual_in, token_id, model_weights)` is
a precondition violation.

**Known compliant:** Gemma 3 4B, Gemma 4 E2B, Gemma 4 E4B (subject to
4.3), Llama 3 family (subject to 4.3).

**Known non-compliant (illustrative, not exhaustive):** architectures
with explicit memory modules (e.g. Memformer-style persistent slots),
retrieval-augmented decoders reading from an external KB between layers,
models using attention sinks implemented as non-residual-stream state,
and any architecture that maintains a learned global summary token whose
state updates *during* decode (some recent long-context schemes). The
general rule is in the precondition itself: any read from persistent
state outside `(residual_in, token_id, model_weights)` is non-compliant,
regardless of whether the specific architecture appears in the list above.

### 4.2 Deterministic RMSNorm / LayerNorm placement

The engine reconstructs K/V from residuals via `recompute_kv` at each
decode step. This requires that the normalization applied to the
residual before the Q/K/V projections is a pure function of the
residual + fixed layer weights, with no stateful component
(running mean/variance, learnable per-step biases derived from position,
etc.).

**Known compliant:** RMSNorm (Gemma, Llama), standard LayerNorm.

**Known non-compliant:** any norm with running statistics updated at
inference time. (No current production LLM falls in this category, but
adapter-based setups sometimes do.)

### 4.3 Position encoding is a function of token position, not cache state

The engine must be able to recompute position-dependent components
(RoPE frequencies, ALiBi slopes, positional embeddings) from the token
index alone, not from cache-internal bookkeeping.

**Why this is a precondition:** cold-replay reconstructs residuals for
tokens at their original positions. If position encoding depended on
*when* a token entered the K/V cache rather than its logical position,
cold-replay would apply the wrong rotations.

**Known compliant:** RoPE (Gemma, Llama), ALiBi, sinusoidal.

**Known non-compliant:** learned positional embeddings with a
cache-state-dependent lookup; any "streaming position" scheme where the
embedding depends on window offset rather than absolute position.

### 4.4 Attention mask is a pure function of position

Similar to 4.3: the attention mask at decode step `t` must be derivable
from token positions alone (causal mask + optional static
document-separator pattern). Masks derived from cache state or
content-dependent routing are not supported.

**Known compliant:** standard causal masks, sliding-window masks with
fixed width.

**Known non-compliant:** content-routed sparse attention (e.g.
router-based MoA), per-query dynamic mask construction that depends on
prior attention outputs.

### 4.5 Precondition check is the implementation's responsibility

The engine must provide a precondition-check entry point that takes a
model handle and returns either "supported" or a structured reason for
refusal. It **must not** silently fall back to a non-bit-perfect
approximation on unsupported architectures. A violated precondition is a
hard error at engine construction, not a warning.

## 5. State representation

The persistent state has two tiers:

### 5.1 Hot window

- Per-layer buffer of up to `W` residual rows, `f32`, shape `[W, d]`.
- Canonical ordering: oldest-to-newest within the window.
- Eviction policy when `position > W`: FIFO — oldest row is evicted
  into the cold tier (as a token ID; the residual is not preserved).

### 5.2 Cold tier

- Append-only vector of token IDs for positions `0..(N - W)`.
- 4 bytes/token (`u32`).
- Reconstruction path at decode step `t`:
  `[cold_token_ids ‖ hot_residuals]` is passed to `recompute_kv`
  before `rs_decode_step` computes the step's output.

### 5.3 What is NOT in the state

- K/V tensors (transient only, during `recompute_kv`).
- Attention outputs, FFN activations (transient only, per step).
- Position indices (derivable from `cold_token_ids.len() + hot_window.len()`).

## 6. Operations

The engine exposes, at minimum, the following logical operations. API
shape is the implementer's call.

### 6.1 `prefill(prompt_tokens) -> State`

Runs the forward pass over `prompt_tokens`, populating the hot window
(and cold tier if `len(prompt_tokens) > W`). Returns initial state.

### 6.2 `decode_step(state, last_token_id) -> (next_logits, new_state)`

Advances state by one token. Under the correctness contract (§2.1),
`next_logits` must be bit-identical to the Standard KV reference path
given the same inputs.

### 6.3 `check_preconditions(model) -> Result<(), PreconditionViolation>`

Validates §4 against a given model. Required entry point; see §4.5.

### 6.4 Optional: state (de)serialization

If implemented, serialized state must round-trip through the correctness
contract: `decode_step` on deserialized state must produce identical
logits to `decode_step` on pre-serialization state.

The two tiers have very different serialisation profiles:

- **Cold tier** is trivial — append-only `u32` token IDs, no model
  fingerprint required (token IDs are vocab-keyed but not residual-
  geometry-keyed). Stable across engine versions for a given tokenizer.
- **Hot window** is the harder case. Residuals are model-specific
  `f32` (or BF16, per §2.1) and are not interpretable against another
  model. If hot-window serialisation ships, the format **must** carry a
  model fingerprint (vindex hash + arch identifier + `residual_dtype`)
  and the loader **must** refuse to load hot-window state whose
  fingerprint does not match the live model. Cross-model loads are a
  hard refuse, not a warning.

  `residual_dtype` is the on-disk dtype of the *serialised residual
  tensor*, not the model weights' quantisation tier. The two can
  differ (e.g. residuals serialised in BF16 from a Q4_K model). Both
  are part of the fingerprint contract; the model's quantisation tier
  appears via the vindex hash, the residual dtype appears as its own
  field.

Format is the implementer's call; stability across engine versions for
the hot window is a separate contract to be specified if/when
serialization ships.

## 7. Configuration

The engine takes at minimum:

- `W` (hot window size). Default: `512`. Constraint: `W ≥ 1`, though
  values below ~128 are likely to trade memory for cold-replay compute
  in ways that dominate wall-clock.
- A reference to a model handle satisfying §4.

The engine does not take a sampling config — it produces logits;
sampling is the caller's concern. This is a deliberate separation: the
correctness contract is about logits, which are a deterministic function
of state + model. Sampling is where non-determinism legitimately enters,
and it lives outside this engine.

## 8. Error modes

Implementations must distinguish at least:

- **Precondition violation** (§4): model is not supported. Hard error at
  construction.
- **Resource exhaustion:** hot window allocation failure, cold tier
  allocation failure. Hard error at construction or during prefill.
- **Cold-replay mismatch:** if the engine ever detects that cold-replay
  produced residuals inconsistent with the previously-stored hot-window
  residuals at the replay boundary, this is a bug, not a recoverable
  error. Panic in debug builds; in release, the implementation's choice,
  but it must not silently produce non-bit-perfect output.

## 9. Migration history (informative)

The migration is complete; this section records what happened:

1. ✅ Lifted `kv-cache-benchmark::real_model::markov_layer` into
   `larql_kv::engines::markov_residual` (2026-05-09 extraction).
2. ✅ `KvStrategy` trait impl was dropped together with the
   `kv-cache-benchmark` crate (2026-05-16). The production trait is
   `larql_kv::KvEngine` (re-exported from `larql_inference::KvEngine`),
   driven by `larql_kv::generation::generate_with_engine`. The
   research-era `KvStrategy` synthetic-encoder trait is gone.
3. ✅ The `#[ignore]`'d real-model test suite lives next to the engine
   at `crates/larql-kv/src/engines/markov_residual/`.
4. ✅ Dropped "Tier 1 / variant iv-dense" naming. The engine is
   `larql_kv::engines::markov_residual::MarkovResidualEngine`.
5. ✅ §4 preconditions documented per architecture in the engine module.
   New architectures must include a precondition-validation record.

## 10. Open questions

Not blocking the migration, but worth tracking:

- **Gemma 4 E4B support.** The latest measured run covers Gemma 3 4B.
  Gemma 4 E2B has been validated end-to-end in the LARQL research stack
  (zero-matmul FFN); E4B has not been run through this engine yet.
  Precondition check per §4 should pass, but "should" is not "has."
- **Llama 3 support.** No blockers anticipated; position encoding
  (RoPE), norm (RMSNorm), and forward-pass structure all appear to
  satisfy §4. Needs empirical validation.
- **Multi-query / grouped-query attention interaction.** Gemma 3 4B uses
  GQA; the reference implementation handles it. Worth confirming that
  the cold-replay path continues to work under MQA (single K/V head) and
  under more aggressive GQA ratios than Gemma's.
- **Quantization interaction.** §2.1 now states the contract: bit-
  identical vs the same model's own Standard KV path *at the same
  quantisation tier*. The remaining open work is fixturing — the current
  measured tier is FP16; Q4_K and FP8 same-tier comparison fixtures need
  to land before those tiers can be claimed as supported.
- **Interaction with Tier 2 / Tier 3 engines.** `UnlimitedContextEngine`
  and `ApolloEngine` build on the same residual-stream machinery. Worth
  deciding whether they share a common trait / base engine with
  `MarkovResidualEngine` or stay as sibling implementations. Out of
  scope for this spec; worth flagging for the Tier 2/3 spec.

---

## Appendix: relationship to the (retired) kv-cache-benchmark ladder

This engine was **Row 3** of the historical benchmark's correctness ladder
(`Markov RS (W=512)`). The other rows are out of scope:

- Row 1 (Standard KV): the reference path the correctness contract is
  stated against.
- Row 2 (TurboQuant): a different engine with a different contract
  (top-1 preserved, not bit-exact).
- Row 4 (`UnlimitedContextEngine` / Tier 2): a different engine, uses
  per-window K/V checkpoints; bit-exact within window, not across.
- Row 5 (`ApolloEngine` / Tier 3): a different engine, uses single-vector
  boundaries + injection; first-token factual, not bit-exact.
- Row 6 (RS Graph Walk): the projected future, requires cracked
  attention; not yet operational.

When migration lands, the benchmark's Row 3 measurement becomes "measure
`larql-inference::MarkovResidualEngine` via the `KvStrategy` adapter,"
rather than "measure our in-tree `real_model::markov_layer`." The number
should not change. If it does, the migration broke something.
