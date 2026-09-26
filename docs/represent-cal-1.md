# REPRESENT-CAL-1 — calibrated encoder recipes

**Status: accepted implementation contract. CAL-1.1 is implemented for dense
softmax single-stream prefixes. CAL-1.2 adds scoped library recipe dispatch;
full-width R4 qualification and admission gates remain open.** Scope: complete the existing NVFP4 GPTQ path, then add matched
unweighted and diagonal-input-weighted scale fitting. Allocation policy stays
fixed. Observer-aware encoding and budget allocation follow this milestone.

REPRESENT compiles model semantics, evidence and resource constraints into an
executable representation. A codec defines how stored values are interpreted;
an encoder recipe defines how those values were chosen. Calibration belongs to
derivation, and measured quality belongs to admission.

## Existing authorities and integration points

This plan extends the current contracts; it does not replace their identities,
evidence rules or runtime fidelity floors.

| Concern | Existing authority | CAL-1 extension |
|---|---|---|
| Allocation policy | [PrecisionMap](../crates/larql-vindex/src/format/vindex3/represent/map.rs) | Keep the same map across encoder comparisons |
| Codec and recipe identity | [CodecIdentity / EncoderRecipe](../crates/larql-vindex/src/format/vindex3/represent/nvfp4_pack.rs) | Explicit recipe selection and derivation parameters |
| Compilation | [RepresentSpec / compile_representation](../crates/larql-vindex/src/format/vindex3/represent/mod.rs) | Dispatch the selected recipe and record the recipe actually executed |
| GPTQ | [gptq module](../crates/larql-vindex/src/format/vindex3/represent/gptq/mod.rs) | Expand supplied-Hessian tensor encoding to capture, one layer, then sequential model compilation |
| Candidate identity | [CandidateRepresentationAuthority](../crates/larql-vindex/src/format/vindex3/represent/candidate_authority.rs) | Bind derivation to independently verified completed payloads |
| Activation evidence | [ActivationAuthority](../crates/larql-vindex/src/format/vindex3/represent/activation.rs) | Bind exact site, input population and candidate-prefix identity |
| Admission | [Contract index](represent-v1-contract-index.md), [optimizer contract index](optimizer-contract-index.md) | Feed existing evidence and promotion machinery with calibrated candidates |

The GPTQ tensor encoder already freezes every scale byte and changes only E2M1
codes. It takes a supplied Hessian and uses scalar factorization on small
fixtures. Its module explicitly leaves calibration capture, production
factorization and REPRESENT dispatch open.

[ENCODER-R4](../bench/prompts/quality-bank-1/ENCODER-R4.md) remains the authority
for the existing GPTQ experiment: sequential candidate-path calibration,
dead-coordinate handling, damping, f64 production arithmetic, N selection,
scale identity and the one-shot Q-bank protocol. CAL-1 does not reopen these
choices or silently add arms to its admission run.

## Three contracts

**Execution.** Resolved representation choices, codec identities and revisions,
geometry, layouts, payload integrity and precision topology supply execution.
The decoder does not fetch calibration data or rerun an encoder. Unknown
encoder provenance does not invalidate an otherwise supported codec.

**Derivation.** A versioned record binds each encoded site to source identity,
recipe identity/revision, all value-affecting parameters, calibration artifact
digest (when required), allocation-policy identity and resolved decisions. Bind
the record to the completed candidate and payload digests; a requested recipe
name alone is not proof of what ran. Preserve existing representation-state
semantics: a provenance-only change must not pretend the executable values
changed. Do not put candidate provenance into `Vindex3Index.extra`, which
participates in source semantic identity.

**Admission.** Measurements bind candidate identity, reference, bank identity,
protocol and instrument to quality, storage and realized resource costs, plus
the policy and decision that use them. Reuse current measurement ingestion and
promotion authorities. Successful encoding is not quality admission; missing
or diagnostic-only evidence must not become an authoritative passing score.

Reproduction needs provenance **and the referenced source/calibration inputs**,
with a supported deterministic numeric implementation. Stored execution does
not need these derivation inputs. Existing explicit fidelity-floor policies
can still require verified attestations before preparation; this separation
does not bypass those policies.

The implementation should introduce typed derivation/calibration records and
validated dispatch, not leave this distinction as free-text documentation.
Keep legacy nearest artifacts readable. Final schema names and persistence
placement must be settled with the first implementation and migration tests.

## Experimental arms

There are five candidate methods plus one null control. All use the same
allocation map, NVFP4 codec ABI, payload byte count and execution kernels.
Identical byte count does not mean identical payload contents or provenance
size. Report payload, complete artifact and runtime residency separately.

| Arm | Recipe / method | Allowed choices | Role |
|---|---|---|---|
| A | `nvfp4-nearest-v1` | Independent codes on nearest's frozen scales | Incumbent |
| A′ | Diagonal H, frozen scales | Same independent code grid | Null control; no production recipe needed |
| B | Proposed `nvfp4-scale-search-v1` | Group scales and codes | Unweighted search control |
| C | Proposed `nvfp4-weighted-scale-v1` | Exactly B's search space and algorithm | Value of diagonal input statistics |
| D | `nvfp4-gptq-v1` | Coupled codes on A's frozen scales | Existing R4 |
| E | Observer-aware method, deferred | Explicitly matched to its comparator | Incremental value of output sensitivity |

For A′, positive diagonal weights cannot change an independent nearest-code
argmin. Use the same tie rule; zero-weight coordinates retain nearest's code.
Require byte equality on fixtures away from floating-point rounding boundaries
and explicit edge-case checks. Any discrepancy must be explained by arithmetic
or a violation of the stated freedom, not counted as fidelity recovery.

B/C must share candidate scale generation, initialization, traversal, iteration
budget, arithmetic and tie-breaking. Only objective weights differ. Start with
the tensor scale frozen to A; search legal stored E4M3 group scales and choose
nearest E2M1 codes at each candidate scale. Include A's scale. Score the actual
decoded stored values, including scale rounding, rather than an ideal float
scale that cannot be persisted. Specify the candidate set and stopping rule
before looking at evaluation results. Unit input weights must reproduce B
exactly; an all-zero group uses a declared deterministic fallback.

B versus C isolates weighting within that search. A versus D isolates
correlated compensation under frozen scales. C versus D measures practical
quality/cost tradeoffs with different freedoms; it does not isolate covariance
information alone. An information-only comparison needs matched scale freedom
in a separately named future experiment.

Reports must identify the comparison class in both their names and structured
metadata: `local-information-ablation` for B/C on identical source weights and
samples, and `sequential-candidate-comparison` for whole-model candidates using
their respective encoded prefixes. Record the compared arms, sites, sample
digests and prefix identities. The latter measures the compiler's system-level
result; it must not be reported as isolating the weighting objective.

## Implementation sequence and acceptance gates

The stages define acceptance independently. Synthetic fixtures establish
mechanics; real model evidence is required for quality and production-cost
conclusions. The CAL-1.1 implementation record below states its qualified scope.

### CAL-1.1 — Calibration artifact and exact capture boundary

Land this boundary independently, before GPTQ dispatch integration. Build the
second-layer candidate-prefix falsifier as an early end-to-end gate in this
stage, using a controlled encoded first-layer replacement; do not defer it
until full-model GPTQ works. The first code PR closes artifact authority and
capture semantics only, without claiming calibrated recipe integration.

Define a versioned artifact manifest with source/model and tokenizer identity,
ordered token-bank digest, masking/position rules, site identity and geometry,
operator boundary, actual sample count, statistic kind and normalization,
numeric precision, payload lengths/digests, and capture implementation identity.
Record the already-encoded candidate prefix that produced those activations.
Distinguish calibration, reconstruction-validation and final admission banks.
For the existing R4 run, preserve its content-verified disjointness and frozen
N-prefix rules; do not substitute the older sensitivity calibration protocol.

Capture the actual inputs to qkv, gate/up and down projections, respecting
their graph boundaries. Shared captures are valid only for sites proven to
consume identical inputs. Accumulate dense H one site at a time; persist bounded
statistics rather than requiring all activations or model tensors in RAM.
Orchestrate candidate-path execution above the substrate dependency boundary;
do not make `larql-vindex` depend on `larql-inference`.

**Gates:** compare streamed statistics with a direct small-matrix reference;
refuse wrong source, site, width, prefix, token digest, corrupt payload and
unsupported numeric data. At a real layer, independently verify each captured
boundary. A second-layer fixture must detect calibration incorrectly using the
canonical prefix after the first layer has changed. An uncaptured site is a
refusal, not an implicit nearest fallback.

### 2. Complete GPTQ compilation

Integrate the R4-qualified accelerated f64 factorization for real site widths;
preserve reduced alive-column semantics and the frozen update-vector accuracy
gate. Connect explicit recipe dispatch to capture/artifact validation, encoding,
candidate authority and derivation persistence. Preflight required sites and
recipe support before producing a publishable output. Keep source files
immutable and mark incomplete outputs as incomplete.

**Gates:** retain all current tensor oracles; add nonuniform diagonal-H null
coverage, multiple rows/groups, dead coordinates and rounding edge cases.
Verify the accelerated path against the numeric reference. Expand one tensor
to one layer to sequential full-model compilation, measuring capture cost
separately from accumulation/factorization. Assert A/D scale-byte identity and
payload-length identity at every stage. For the frozen R4 target, retain its
specific total-payload assertion; it is not a universal NVFP4 byte count.

Reopen persisted candidates and independently verify authority and decoded
values. A stored GPTQ pack must execute without its calibration files. Recipe
mismatch affects reproducibility, not decoding. A transient nearest encoder
must not be used as GPTQ's byte-equivalence oracle: reproduction requires the
matching recipe and inputs, while execution comparison uses the stored pack
and its decoded values. Update the nearest-specific compilation module header
when this dispatch becomes real.

### 3. Close existing R4 admission

Run R4.0-CAL-B only after the candidate encoder exists: choose N mechanically
from held-out reconstruction under the already-frozen rule, then freeze it.
Retain R4.3's one-shot Q-bank comparison and full per-category/saturation report.
No Q-bank feedback enters encoder tuning or N selection.

**Gates:** record selected N, derivation identity, candidate payload identity,
execution verification and the measured admission verdict. Implementation can
pass while the quality verdict fails. A loss is a completed experiment, not a
reason to retune on the admission bank.

### 4. Add matched scale-search recipes

Begin this implementation after the GPTQ path runs end to end and stage 3's
R4 verdict is recorded, whether positive or negative. B/C reuse the established
capture, derivation and admission framework.

Freeze B/C's shared search contract and a separate comparison protocol before
running their evaluation. Consume diagonal statistics through the same
calibration authority; production C must not require dense-H allocation or
factorization. For a local B/C ablation, use identical source weights and input
samples. For sequential model candidates, record each candidate's actual
prefix, since later-layer activations can differ between recipes.

**Gates:** all-one weighting gives identical B/C bytes; a constructed group
with competing errors demonstrates that nonuniform weighting can change the
selected scale and improve its declared weighted objective. Validate zero
weights, finite nonnegative statistics, scale extremes, legal NVFP4 geometry,
determinism and independent stored decode. Within the common enumerated search,
each selected result must score no worse than the included nearest-scale
candidate under its own objective. This is not an end-to-end quality guarantee.

### 5. Comparative admission and milestone closure

Predeclare B/C/D comparisons, quality criteria and measurement budgets on a
separate protocol. A bank already exposed during development is not a fresh
holdout; retain comparability labels and use an untouched qualification bank
where required. Bind results to actual payloads, not recipe names alone.

Report quality distributions and uncertainty alongside capture time, encoding
time, peak memory, calibration artifact size, payload size and realized runtime
cost. Performance claims use the repository's warmed baseline/candidate/baseline
protocol and peer-session exclusivity handshake. Do not infer a kernel-speed
win merely from a changed recipe.

CAL-1 closes when calibrated capture, validated recipes, derivation records,
persisted execution and independently bound admission evidence work end to end
for the scoped model and sites, with both GPTQ and weighted-scale comparisons
reported. A particular recipe need not win. Synthetic success alone, metadata
alone or an unmeasured candidate cannot close the milestone. Allocation policy
changes, MoE generalization and observer research remain separate work.

## Observer follow-up boundary

The starting quadratic is `E[(ΔW x)^T G(x) (ΔW x)]`. The factorized
`tr(G ΔW H ΔW^T)` is exact for constant G with `H = E[xx^T]`; using separate
averages for varying G(x) is an approximation to test.

Constant positive diagonal G does not change independent per-row optima.
It can guide shared resources or coupled parameters. This conclusion does
not cover input-dependent diagonal G(x): row i then has effective input metric
`H_i = E[g_i(x) xx^T]`, which can change its optimum even without a shared
resource. Off-diagonal G additionally couples output errors. Keep these
mechanisms distinct when reopening [Quant-Obs](quant-obs.md), and compare them
against the calibrated incumbent established here.

Register three separate future claims: `observer-constant-diagonal-allocation`
for shared-resource decisions, `observer-input-dependent-diagonal` for the
row-specific H_i mechanism, and `observer-off-diagonal-coupling` for coupled
output errors. Each needs its own comparator and falsifier; a result for one
does not establish either of the others.


## CAL-1.1 implementation record

The library API lives in
[`represent::calibration`](../crates/larql-vindex/src/format/vindex3/represent/calibration/mod.rs).
`PreparedCalibration` owns a prepared CPU image, seals effective source and
candidate operand values (including norms and biases), and captures one selected
projection at a time through the ordinary decode interpreter. The initial scope
is contiguous dense softmax single-stream prefixes; other programs refuse.
The content seal includes the entire target layer, conservatively invalidating
on changes after the selected site, and excludes later layers and the head.
This is a capture-context identity, not a replacement for RepresentationStateId.

Attention and gate/up inputs use the existing operand observer. A new opt-in
backend tap borrows the actual intermediate immediately before down projection;
reference and production CPU paths share their observed/unobserved arithmetic.
Unsupported backends refuse this tap. A prepared image can select another
projection in its target layer without reloading the prefix.

`CalibrationBank::from_token_bank` reads and verifies the existing
`teacher-forced-token-bank/v1` payloads, retaining their bank identity. The
calibration envelope additionally binds ordered masks, sequence resets and
population. The capture checks the tokenizer digest against its own container.
The direct in-memory constructor supports fixtures and pre-frozen external
corpora, with the same tokenizer check; it does not grant those corpora holdout
status or replace R4's separate disjointness gates.

A `represent-calibration/v1` artifact consists of `manifest.json` and
`statistics.f64`: raw uncentered f64 sums, dense X^T X or diagonal sums of squares,
with sample count carried separately. No normalization or damping is applied.
The reader requires an independently supplied expected capture key and checks
context, schema, arithmetic revision, shape, payload length, finite values and
both manifest/payload digests. Writers require a fresh directory and publish
the manifest last. These records establish capture provenance and integrity;
they are not Q-bank admission evidence or encoder derivation records.

The early falsifier replaces layer zero's down matrix with decoded NVFP4 values
through an immutable overlay. Layer one's captured statistics must differ from
the canonical-prefix control and equal an ordinary candidate execution; the
canonical artifact must fail reuse under the candidate's key. Layer zero's
entering Q input stays equal, locating the effect downstream of the replacement.
Other gates cover all six sites, masks and KV resets, bias-content binding,
malformed artifacts, tokenizer mismatch, sealed-bank corruption, deterministic
replay and bit-identical observed/unobserved logits.

The opt-in real-layer check uses fresh text, not Q-bank data. On Granite 4.2 3B,
it checks all six layer-one boundaries over ten positions (width 2,560 for
Q/K/V/gate/up, 8,192 for down). This is boundary qualification only: it does not
establish long-range behavior, calibration sufficiency, quality or throughput.

```bash
cargo test -p larql-vindex --lib calibration --no-default-features
CAL1_CONTAINER=/path/to/granite-4.2-3b.s6.vindex3 cargo test -p larql-vindex \
  --lib calibration_real_layer_inputs --no-default-features -- --ignored --nocapture
```

CAL-1.2 below supplies scoped recipe dispatch and iterative prefix construction.
Accelerated full-width factorization, R4.0-CAL-B/R4.3 and B/C encoding remain
subsequent stages.
Existing plugin-weighted compilation on main is a separate consumer; CAL-1.1
does not pass raw sums to that interface as if they were already normalized
moments or claim that NVFP4 weighted-scale fitting is implemented.


## CAL-1.2 implementation record

[`compile_representation_recipe`](../crates/larql-vindex/src/format/vindex3/represent/recipe.rs)
is the explicit library entry point for `Nvfp4Recipe::Nearest` and
`Nvfp4Recipe::Gptq(&request)`. Both write through the existing REPRESENT
container compiler. The CLI's default and the existing nearest entry points
remain unchanged; nearest requires neither a tokenizer nor calibration and
retains its existing candidate identity and transient byte-parity gate.

A `GptqRequest` names a component, a frozen `CalibrationBank`, and the exact
tensors assigned to GPTQ. Each site requests either `CaptureTo(path)` or
`Existing(path)`. All other eligible projections explicitly use nearest.
The initial schedule covers dense softmax q/k/v/o/gate/up/down in ascending
layer order. CAL-1.1 supplies no o-projection capture boundary, so o is nearest;
requesting GPTQ there refuses. A requested protected, missing or unsupported
site also refuses. This is scoped mixed-recipe integration, not closure of the
uniform R4 arm. Other eligible tensors must be explicitly protected rather than
silently encoded outside the supported schedule.

Before encoding each GPTQ site, the compiler prepares the actual completed
candidate prefix and derives its expected **DenseGram, calibration-population**
key. New capture is persisted, released, and read through the same validated
artifact reader as reuse. Wrong source, prefix, site, bank, execution, statistic
or population refuses; an existing artifact is never replaced by a recapture.
Raw uncentered f64 sums feed the frozen R4 damping and elimination algorithm
without normalization. Tensor scale and every persisted E4M3 scale come from
the original W0 nearest scale derivation; only E2M1 codes can differ.

Completed packs are staged in private segment files and mmap-backed in the
next capture's operand source. They expose their actual NVFP4 dtype and raw
bytes to lowering: using decoded f32 replacements would silently choose a
different CPU kernel. q/k/v/o are installed before FFN capture; gate/up before
down; the completed layer before the next layer. The source stays immutable.
The packed loader also delegates physical splitting to the codec, including
NVFP4's three streams inside one payload; stream count alone does not imply
separate tensor files.

[`DerivationRecord`](../crates/larql-vindex/src/format/vindex3/represent/derivation.rs)
lives in `candidate.json`. Its execution-ordered tensor entries name the recipe
actually run, frozen parameters, source-value digest, calibration manifest
and digest, payload digest and length, dead/alive columns and saturation.
Source semantic identity and allocation-map digest bind the enclosing record.
CandidateRepresentationAuthority validates each entry against the completed
operand seals and includes derivation in its integrity binding. It does not
add calibration to RepresentationStateId or decoding semantics. A segment
containing several recipes declares `nvfp4-site-recipes-v1`; its per-tensor
records are authoritative about which recipe produced each payload.

Memory lifetime is structural: one artifact is captured/read and consumed at
a time, its Vec moves into the Hessian without a raw-matrix clone, raw H is
released before factorization, and the factorization and row workspace are
dropped before the next site. Factorization still requires several matrices
for **that site**. Capture still prepares a prefix image, and staged packed
weights occupy disk/mapped address space; neither is a claim of constant total
model memory. The scalar f64 factorization remains unsuitable for a production
full-width campaign; acceleration and measured resource admission remain open.

Gates cover persisted A/D scale identity and equal byte counts, nontrivial code
changes, exact agreement of captured Grams with ordinary candidate execution,
within-layer and second-layer falsifiers, refusal of a decoded-f32 execution
identity, deterministic replay using existing artifacts, and rejection of
wrong-key/diagonal artifacts before site encoding. The final candidate runs
bit-identically after deleting the calibration directory, and its stored
execution matches the completed packed prefix. An explicit-nearest comparison
also preserves the legacy `candidate.json` bytes without calibration files.
These are implementation gates, not a quality win, Q-bank admission or R4 closure.

```rust,ignore
use larql_vindex::format::vindex3::represent::{
    compile_representation_recipe, Nvfp4Recipe, RepresentSpec,
};
use larql_vindex::format::vindex3::represent::recipe::{
    CalibrationInput, GptqRequest,
};

// bank was created from a verified token bank or an explicitly frozen corpus.
let request = GptqRequest {
    component: "target".into(),
    bank,
    sites: [( // Exact object/tensor names from the operation plan.
        ("target.decoder_stack".into(), "0.self_attn.q_proj.weight".into()),
        CalibrationInput::CaptureTo(calibration_path),
    )].into(),
};
compile_representation_recipe(
    source_path, output_path, &RepresentSpec::nvfp4(), Nvfp4Recipe::Gptq(&request),
)?;
```
