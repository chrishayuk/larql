//! Output types for the semantic representability plan (`larql vindex3 plan`).
//!
//! A plan is a statement over one or more inspected artifacts of **exactly
//! why** the current VINDEX3 container schema can or cannot faithfully encode
//! them. Findings are typed twice:
//!
//! - **category** — what kind of statement this is: representable,
//!   value-mismatched between authorities, unrepresented, or a declared
//!   cross-component interface.
//! - **semantic class** — how much losing it would matter. `consumed` in the
//!   inventory is *not* sufficient: a key can be consumed and still resolve
//!   to a different value than the checkpoint declared, which is why
//!   mismatch findings compare values, not statuses.
//!
//! The verdict is fail-closed: any mismatched/unrepresented finding whose
//! class is execution-, tensor-, or interface-semantic — or whose key the
//! registry has never seen (`unknown`) — makes the plan inadmissible.

use serde::{Deserialize, Serialize};

mod artifacts;
pub use artifacts::*;

/// Current plan schema. Bump on any breaking change to these types.
///
/// v2: the plan carries the built [`SystemGraph`] — representability is
/// defined as "the graph builder placed it", and the graph is the proof.
/// Interfaces are reported as resolved edges rather than candidate guesses.
///
/// v3: findings about config keys carry a [`Carriage`] stage, and the
/// census reports every declared key rather than only the unconsumed
/// ones — so `unrepresented: N` is a count against a stated denominator
/// instead of a lower bound. Adds the `training_only` and `alias`
/// semantic classes.
///
/// v4: the plan names who judged it and what it judged. `planner` carries
/// the planner's package version and its *semantics* version; every
/// artifact carries its `source` — the argument as given and, for a repo,
/// the immutable commit the facts were read at. A verdict without these
/// is not attributable, and a plan of an earlier schema is refused by
/// [`SystemPlan::parse`] rather than read as unattributed.
pub const PLAN_SCHEMA: u32 = 6;

/// The planner's semantics version.
///
/// Bumped **only** when a verdict can change: a new or corrected rule that
/// makes a checkpoint admissible or blocked where it was not before (the
/// sliding-window normalisation flipped six Qwen3 sizes from three
/// blockers to zero — that is a bump). A fix to the CLI, the report's
/// wording or the JSON layout is not. The package version says which
/// build ran; this says whether its answers are comparable with another
/// build's. Anything caching verdicts keys on (source revision, this).
///
/// `plan/tests/identity.rs` pins fixture verdicts against this value, so
/// a change that flips one fails there until the version is bumped.
///
/// **22** — Kimi-K3's latent routed branch (K3-LATENTMOE-1). The routed
/// experts run behind a bottleneck of their own:
/// `routed_expert_down_proj` takes the block input to
/// `routed_expert_hidden_size`, the experts run THERE, the weighted
/// aggregate is normalised, and `routed_expert_up_proj` returns it. Two
/// leaves stop grading `unknown` and become carried facts, and the router
/// and shared experts stay at `hidden_size` — both read the un-projected
/// block input, and the shared branch is summed after the up-projection.
///
/// The bump is not the two leaves finding homes. It is that
/// `routed_expert_hidden_size` becomes **the authority the routed bank's
/// geometry is sized from**: every expert-bank shape contract — packed
/// gate/up and down, both scale streams, the per-expert variants and the
/// down bias — asks `MoeSurface::routed_expert_input_width` instead of
/// reaching for the component's `hidden`. A build that stored the width
/// and kept sizing the bank from `hidden` would report the fact as
/// carried while refusing the checkpoint's real bank on shape; that is
/// the hollow carriage this transition exists to make impossible, and it
/// is why the leaf and the geometry move together rather than in two
/// rungs.
///
/// The norm's epsilon is the LAYER's `rms_norm_eps`, and this inverts the
/// finding of the two K3 rungs before it: `q_a_layernorm` and
/// `kv_a_layernorm` run at `KimiRMSNorm`'s class default `1e-6` because
/// their constructor passes no override, and `routed_expert_norm`'s
/// passes one. Same family, same norm class, a factor of ten apart —
/// carried with the form rather than reached for.
///
/// Presence and truthiness are not shared between adjacent leaves:
/// `routed_expert_hidden_size` selects the form by PRESENCE (`0` selects
/// it and is then refused by name; `null` is absent), while
/// `latent_moe_use_norm` is read by truthiness, so `null`, `false` and
/// absent all mean no norm.
///
/// **21** — Kimi-K3's factorised MLA query (K3-MLA-Q-LORA-1).
/// `q_lora_rank` selects the query FORM — `q_a_proj` -> `q_a_layernorm`
/// -> `q_b_proj` in place of one dense `q_proj` — and the form is
/// declared, never deduced: `q_proj` and `q_b_proj` have the same row
/// count (`Hq*q_head_dim`, 18432 on K3) and differ only in their column
/// count, so an operand-sniffing build would pick the form from the very
/// thing the form decides. `MlaQueryForm` carries it on the surface,
/// `MlaQueryProjection` on the op — typed, so "both" and "neither" are
/// unrepresentable — and closure holds the shipped operands to the
/// declaration from both sides, refusing a `q_proj` under a declared rank
/// and the triple under none.
///
/// `q_a_layernorm`'s epsilon is carried with the form and has its own
/// authority: it runs at `KimiRMSNorm`'s class default `1e-6`, not the
/// layer's `rms_norm_eps`, and NOT by borrowing `kv_a_norm_eps` — the two
/// agree because one class default is used twice, which is a shared cause
/// and not a shared authority.
///
/// This bumps no verdict on any row, and that is the expected result:
/// every MLA config leaf on K3 already graded representable, so the
/// blocker count does not move. What moves is the operand plane —
/// `MLA_LAYER_UNADDRESSED` 3 -> 0, K3's estate 5382 -> 5379 unclassified
/// and 12 -> 9 distinct spellings — and the executor, which reproduces
/// the factorisation to per-boundary parity against a third oracle arm.
/// After it, no remaining K3 text-generation blocker is an
/// attention-semantic blocker.
///
/// **20** — Kimi-K3's FFN combine, and the end of silent activation
/// substitution (K3-ACT-1). `hidden_act: "situ"` names SiTU-GLU,
/// `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)` — a
/// softcapped SwiGLU — and is carried as an
/// `ExpertGatePolicy::SituGlu { beta, linear_beta }` rather than an
/// `Activation` variant, on the reasoning that enum already states for
/// GPT-OSS's clamped GLU. `activation_situ_beta` and
/// `activation_situ_linear_beta` move from Unknown to its parameters,
/// carried `Lowered`; `hidden_act` stops reading `mismatched` because the
/// activation probe now answers from the COMBINE the FFN computes rather
/// than from a nonlinearity field a non-plain policy never reads.
///
/// The second half is not K3's. `ModelArchitecture::activation` used to
/// read `.and_then(from_hf_name).unwrap_or(Silu)`, which gave the same
/// answer to *the config is silent* and *the config declared something
/// this build has never judged* — so a checkpoint declaring `situ`, or
/// BitNet declaring `relu2`, was executed as SwiGLU. The four states are
/// now told apart by `ActivationDeclaration`, and every gate/up kernel
/// selection refuses an unjudged declaration by name instead of
/// substituting one. This moves no finding on any row but K3: the planner
/// was already reporting both specimens honestly, and what changed is
/// what an executor does with them.
///
/// **19** — the two K3 attention output gates, DECLARED (K3-REP-GATE-1).
/// `linear_attn_config.use_full_rank_gate` (the KDA output gate's FORM:
/// one full-rank `g_proj` in place of the low-rank `g_a_proj`/`g_b_proj`
/// pair) and `mla_use_output_gate` (a sigmoid gate on MLA's aggregated
/// value before `o_proj`, the same generic operation the softmax family's
/// `attn_output_gate` resolves to) move from Unknown to ExecutionSemantic,
/// carried `Lowered`: the op plan carries the KDA form as a type and the
/// MLA gate as an optional operand, closure holds the shipped `g_proj` /
/// pair to the declaration from both sides, and the CPU executors compute
/// both gates to per-boundary parity with second oracle arms transcribed
/// from `modeling_kimi_linear.py`. The gate's projection is the ONLY thing
/// the KDA form changes; the MLA gate is a new stage between aggregation
/// and `o_proj`. Every Metal path refuses either declared gate by name.
///
/// Verdicts pinned: a Kimi-shaped estate declaring both keys is
/// admissible where it was blocked by two Unknown findings; the same keys
/// on a component with no KDA / MLA block stay blocked, uncarried — the
/// keys were judged, not waved through. `self_attn.g_proj` gains a role on
/// each attention family under that family's operator; K3's MLA layers
/// still carry the unaddressed q-LoRA triple, which is its own cell.
///
/// **18** — attention-residual TRAVERSAL (K3-ATTNRES-1, transition 2).
/// The topology is executed, not merely addressed: the executor carries
/// an explicit prefix-plus-snapshots state through the decode traversal
/// (2a) and one such state PER POSITION through the batch traversal
/// (2b), each witnessed against a Torch oracle transcribed from
/// `modeling_kimi_linear.py` before any Rust arithmetic was written.
///
/// Two readers move together, as they did at 17 and for the same
/// reason. `ResidualTopology::unimplemented_reason` is DELETED rather
/// than left answering `None` — the state it sat in between wave 19 and
/// this rung — and both the plan report's traversal refusal and the
/// executor's preparation refusal go with it. A dead authority that
/// still answers invites a reader to consult it; the contract for the
/// next topology that cannot be traversed is to bring the authority and
/// its readers back together.
///
/// `attn_res_block_size` moves `Represented` -> `Lowered`, its site now
/// naming the history carrier that reads the period. That reader's
/// COUNT does not move: the leaf was already non-blocking, and a count
/// that changed there would mean the stage name was doing work it
/// should not.
///
/// What does NOT lift: a component declaring the topology and shipping
/// no `attention_residual_exit` object is still blocked by that object's
/// absence, which is the arm that keeps this from having been
/// implemented as "stop refusing attention residuals". Nor does anything
/// else about K3 — its op plan still does not close on `self_attn.g_proj`
/// (K3-REP-GATE-1) and the routed-expert bank is its own rung, so the
/// row moves at the plan level only and the traversal it thereby stops
/// refusing has never run on it.
///
/// **17** — attention residuals are a declared RESIDUAL TOPOLOGY, owned
/// and addressed, and explicitly not traversable (K3-ATTNRES-1,
/// transition 1). Read from Kimi-K3's own `modeling_kimi_linear.py`
/// (`_apply_attn_res`, `_forward_attn_residual`,
/// `_apply_output_attn_res`): the state is ONE prefix sum plus a history
/// of snapshots of it taken every `attn_res_block_size` layers, each
/// sublayer reads a softmax-weighted mix over that history before it
/// runs and adds its result back, and the stack's end reduces the whole
/// history once more before the final norm. A third topology, not a
/// dialect of the second: no stream count makes a `[1, hidden]`
/// projection a Sinkhorn site's mix.
///
/// Three planes move. `attn_res_block_size` is parsed and carried to
/// `ResidualTopology::AttentionResidual.block_size` — `Represented`, not
/// `Lowered`, because no traversal receives it. The exit pair
/// (`output_attn_res_{norm,proj}`) becomes ONE placed object under the
/// declaration, and the generic `norm` name fragment stops sweeping the
/// exit norm into the component's final norm, which had been binding two
/// tensors while claiming to be one. The four per-layer operands
/// classify to four roles of this topology — under the DECLARATION, so a
/// checkpoint shipping the spellings without the period gains no
/// topology from its tensor names — are required on every transformer
/// layer, and are checked at `[hidden]` and `[1, hidden]`.
///
/// What is refused is said by name, and the seam wave 19 retired returns
/// with the variant that needs it: `unimplemented_reason` answers `Some`
/// for this topology alone, and the executor's preparation step and this
/// report both read it, so a plan the report calls executable is one the
/// executor prepares. A component declaring the period and shipping no
/// exit object is refused one step earlier, by the exit's own name — the
/// analogue of the head boundary for the bundle. NO arithmetic is
/// implemented: the traversal and the torch oracle that would judge one
/// are the rung's next artefacts, and a build that lifted this refusal
/// because the operands are addressable would be claiming execution from
/// addressing. Forecast before the code
/// (`forecasts/k3-attnres-1-declare-own-address.json`, scored per
/// reader): K3 33 -> 32, as -1 declaration, -1 ownership, +1 execution
/// surface; a reading of 31 is the fail-open and a BUG. No other cached
/// row declares the period or ships a pair.
///
/// **16** — hyper-connection TRAVERSAL (wave 19). The residual topology's
/// refusal is retired from its one authority
/// (`ResidualTopology::unimplemented_reason`), which the executor's
/// preparation step and this report both read: the decode step (19a)
/// and the batch traversal (19b) carry the bundle, reduce it to one
/// vector at each site and expand the sublayer's output back, and an
/// intermediate-state witness against the reference's oracle fails on
/// every deliberate defect tried. The three topology keys (`hc_mult`,
/// `hc_sinkhorn_iters`, `hc_eps`) leave the unsupported-component table
/// and are carried as execution semantics to the component's residual
/// topology. What remains refused is said by name: a component with the
/// topology and NO `hyper_connection_head` object (GLM-5.3-Flash, `mhc`
/// unexplained) keeps a blocking execution-surface finding, because a
/// whole-stack execution has no declared reduction from the bundle
/// before the final norm — so its count does not move, and a drop there
/// would be capability granted past the head boundary. Forecast before
/// the code (`forecasts/wave19-sinkhorn-traversal.json`, scored per
/// reader): GLM-5.3-Flash 31 -> 31 on this reader and -3 on the key
/// table; DeepSeek-V4 rows -3 each, never reaching this reader.
///
/// **15** — hyper-connection ADDRESSABILITY (wave 18). The six per-layer
/// Sinkhorn site operands (`hc_{attn,ffn}_{fn,base,scale}`) are operand
/// roles, required on every layer of a component that declares the
/// topology, checked against the declared stream count's geometry and
/// bound into the op plan; the head's three bare operands
/// (`hc_head_{fn,base,scale}`) are placed as their own object, and only
/// under the declaration. The op plan therefore no longer refuses on the
/// topology — it runs closure — while the plan report and the executor's
/// preparation step still refuse, through the same
/// `ResidualTopology::unimplemented_reason`, whose text now names the
/// traversal as what is missing rather than the arithmetic (wave 17) or
/// the placement (wave 18). Measured before the code: the two
/// DeepSeek-V4 rows lose their three `hc_head_*` unplaced-group blockers
/// and nothing else moves; GLM-5.3-Flash's execution-surface refusal
/// changes its text and keeps its category; Hy4-preview is untouched
/// because its head is spelled under `model.hc_head` and its topology
/// resolves to none. And a measurement that contradicted the programme's
/// expectation: Kimi-K3's four `*_res_{norm,proj}` operands are
/// `[hidden]` and `[1, hidden]` — not a Sinkhorn site's
/// `[(2 + hc)·hc, hc·hidden]` under any stream count — so they are a
/// different residual topology (AttnRes), not this one's second dialect,
/// and they do not move.
///
/// **14** — hyper-connections are a declared RESIDUAL TOPOLOGY, and
/// explicitly not executable. Read from DeepSeek-V4-Flash's own
/// `inference/model.py`: the state is a bundle of `hc_mult` parallel
/// streams, each sublayer reduces the bundle to one vector and expands
/// its output back, and the weights are computed per token through a
/// projection whose statistics a 20-iteration Sinkhorn split turns into
/// reduce weights, expand weights and a cross-stream combination matrix.
/// `ResidualTopology` states it on the COMPONENT — once the residual
/// means `[.., streams, d]`, the embedding, every branch operator and
/// the head must agree — and a HALF declaration refuses rather than
/// completing itself with one stream. The op plan refuses before reading
/// an operand and the report says so, both through the topology's own
/// `unimplemented_reason`. The component label also stops being
/// family-named: `hyper-connections (GLM-5.x)` appeared verbatim on
/// Tencent and DeepSeek checkpoints, and is now named for the mechanism.
///
/// **13** — LFM2's norm dialect is carried. `operator_norm` and
/// `ffn_norm` are the two-norm PRE-only estate under LFM2's own
/// spelling (`Lfm2DecoderLayer.forward`), and `norm_eps` is its
/// epsilon key. No new execution semantic: the placement is one this
/// build already runs. Registering `lfm2` also stops the identity
/// resolving to `GenericArch`, which was serving Llama-shaped defaults
/// to a stack whose every other layer is a short convolution. Forecast
/// before the code, and deliberately not a GREEN wave: four rows lose
/// three blockers each and NONE clears — the conv mixer's geometry and
/// the `full_attn_idxs` schedule are still absent, and they are
/// execution semantics rather than spellings.
///
/// **12** — three families resolve to their own identities. `olmo2`,
/// `olmo3` and `exaone4` matched no registry entry and fell through to
/// `GenericArch`, which had already chosen PER-HEAD QK norm for OLMo-2 —
/// the wrong reduction for a family whose reference normalises the whole
/// projection. Each entry declares only what its reference establishes:
/// OLMo-2's `QkNormScope::FullProjection` (the operator OLMoE already
/// judges), the 1e-5 `rms_norm_eps` class default both families take, and
/// EXAONE-4's per-head norm applied after the head reshape — its own
/// entry precisely because that one difference is an operator, not a
/// label. Registration resolves a NAME and grants nothing else: a
/// declaration the schema cannot carry still refuses under a registered
/// family. Forecast before the code: three rows clear, four keep the
/// blocker named for each.
///
/// **11** — post-norm placement EXECUTES. Wave 10 could represent it and
/// refused to lower it; the generic executor already applied the wrap
/// norms to each sublayer's OUTPUT before the residual add, and what it
/// could not do was run with NO pre-sublayer norm. Both the batch and the
/// decode path now read the raw residual where the placement says no norm
/// conditions it, and the epsilon QK norm runs at moved off the
/// pre-attention norm's field onto the layer's own `declared_norm_eps` —
/// an epsilon and a placement are unrelated facts, and coupling them is
/// what made this unrepresentable. A post-only stack's single declared
/// epsilon belongs to the post sites, which are the only norm sites it
/// has; a four-norm stack still refuses an unjudged post epsilon, because
/// there the two sites exist and can differ. Forecast before the code:
/// no row clears (identity still blocks all seven), seven rows lose the
/// unsupported-component blocker.
///
/// **10** — a stack may normalise its sublayers' OUTPUT. `NormPlacement`
/// knew two transformer shapes, two-norm and four-norm, and OLMo-2,
/// OLMo-3 and EXAONE-4 declare a third: the sublayer reads the raw
/// residual and its result is normalised before the add
/// (`Olmo2DecoderLayer.forward`, identical in the other two). Their
/// operand estate — both wrap norms, neither pre-norm — matched nothing,
/// so the execution surface refused to build and every probe on those
/// components answered nothing at all. `PostOnly` is recognised from that
/// estate, and the spelling collision is why it is read from which norms
/// EXIST: these families' `post_attention_layernorm` is a true post-norm
/// where a Llama stack's is the pre-FFN norm. The op plan REFUSES it —
/// representable, explicitly not executable, a distinction the closure
/// vocabulary now states in its own defect. Forecast before the code:
/// no row clears, seven surfaces build, sixteen blockers retire because a
/// probe can finally answer, and three answer and still refuse.
///
/// **9** — a wholly-routed family has an FFN, and the always-on shared
/// branch is sized by what the checkpoint declares. The FFN presence rule
/// read only the DENSE width, so `Qwen3_5MoeTextConfig` — which declares
/// no `intermediate_size` at all, because every layer is a routed block —
/// was graded as having no FFN op, and `hidden_act` and
/// `num_experts_per_tok` had nothing to answer to. `FfnSurface`'s dense
/// width becomes optional so that absence is stated rather than written
/// as a zero. Beside it, `shared_expert_intermediate_size` is read (in
/// both declared spellings) and becomes the ONE authority for the shared
/// branch's width: two lineages size it differently and this build was
/// deriving it as `moe_intermediate_size * shared_experts`, which is
/// Kimi's fact and is fourfold wrong on Qwen1.5-MoE. Qwen's gated shared
/// expert — `sigmoid(shared_expert_gate(x)) * shared(x)`, summed with the
/// routed branch — is declared with its own operand, and Qwen3.5-MoE's
/// stacked expert bank is declared as the `PackedBF16` it is. Forecast
/// before the code: exactly five rows clear.
///
/// **8** — two keys read by no implementation, ours or upstream, are
/// read-and-checked rather than graded `Unknown`. Falcon3's
/// `activation: "swiglu"` names the FFN shape (gated, SiLU on the gate) and
/// is judged against the shape the execution surface carries; SmolLM2's
/// `is_llama_config: true` is judged against the family the declared
/// identity resolved to. Same treatment as `use_mrope` / `rope_interleaved`
/// in version 3's wave: never echoed, one value away from a wrong FFN or
/// a wrong family. Forecast before the code: exactly two rows clear.
///
/// **7** — `partial_rotary_factor` is read from inside `rope_parameters`,
/// the transformers-5.x flat form and the only spelling every Qwen3.5
/// checkpoint uses. The parser read the legacy top-level key and Gemma 4's
/// per-layer-type block, so Qwen3.8 (which writes both) resolved while
/// Qwen3.5 lost its fraction: no layer carried one, the partial and
/// multi-axis rotary probes answered nothing, and three leaves refused a
/// family whose text path this build executes. Precedence now mirrors
/// `standardize_rope_params`: top level, then per-type block, then flat
/// block. Forecast before the code: three Qwen3.5 dense rows admissible,
/// three Qwen3.5 MoE rows from six blockers to three, nothing else moves.
///
/// **6** — a declaration a companion switch turns off is inert. Qwen2.5
/// ships `sliding_window: 32768` beside `use_sliding_window: false`; the
/// graph carries no window, which agrees with the checkpoint, and the
/// carriage rule was reporting that agreement as a dropped fact. The
/// companion is read at the same nesting level, so one component's switch
/// cannot silence another's window.
///
/// **5** — MoE routing. A key declared with no value states that its
/// subject does not apply and no longer demands a home (Gemma 4's dense
/// sizes declare `top_k_experts: null`), and Qwen's expert schedule
/// (`decoder_sparse_step`, `mlp_only_layers`) is judged against its
/// value: inert at the uniform all-MoE stack, blocking for any real
/// per-layer topology.
///
/// **24** — the GPT-2 lineage is represented. A family with a learned
/// position table that declares no rotary key resolves every layer to
/// `PositionPolicy::None` (it was graded `Rope { theta: 10000 }` off the
/// parser's default base) and states the table as the head surface's
/// `input_position`; LayerNorm biases (`norm.bias`) and dense-FFN biases
/// (`ffn.bias`) are surface declarations closure holds the operands to;
/// `activation_function`, `n_positions` and `scale_attn_weights` are read
/// and carried; `scale_attn_by_inverse_layer_idx`,
/// `reorder_and_upcast_attn` and `add_cross_attention` are representable
/// only as `false`; `n_ctx` is an alias of `n_positions`; GPT-2's `h.{L}`
/// block list and `wpe` table are placed; and registered mask buffers are
/// reported as excluded. GPT-2 checkpoints move from inadmissible to
/// admissible; no other family's verdict moves.
///
/// **23** — linear rope scaling is represented and Gemma 3's interface
/// spellings are read. `PositionPolicy::Linear` carries `rope_scaling =
/// {linear, factor}` on the layers the architecture says it reaches
/// (Gemma 3: full-attention layers only), so `rope_type: "linear"` and its
/// `factor` are admissible where they used to mismatch. `image_token_index`
/// / `boi_token_index` / `eoi_token_index` / `mm_tokens_per_image` are the
/// image binding under Gemma 3's names and `vision_use_head` is the SigLIP
/// tower's pooling-head fact, so none of the five grades `Unknown` any
/// longer. And a `vocab_size` the config omits is answered by the
/// embedding table's row count, recorded with its provenance, so the text
/// execution surface completes. Forecast before the code: every Gemma 3
/// checkpoint moves from seven text-generation blockers to zero; nothing
/// else moves. Not new mathematics: `larql-compute`'s `rope_freq_plan`
/// has always taken a position divisor.
///
/// **4** — Llama-3 wavelength-band rope scaling is represented.
/// `PositionPolicy::Llama3` carries the block, so a checkpoint declaring
/// `rope_type: "llama3"` is admissible where it used to be refused. Not
/// new mathematics: `larql-compute` has always implemented the family,
/// and the gap was that the container had nowhere to say so.
///
/// Deliberately NOT bumped for plan schema 6: findings gained an `id` and
/// a `cluster`, and each capability now names its blockers, but no verdict
/// moved. The schema says what the document contains; this says whether
/// its answers are comparable. An instrumentation change that shifted this
/// number would make every stored verdict falsely incomparable.
///
/// **3** — decoding-policy and dropout defaults stopped blocking. They
/// are preserved as declared facts and classified for what they are
/// ([`SemanticClass::GenerationPolicy`], [`SemanticClass::TrainingOnly`])
/// instead of grading `Unknown`, and `pretraining_tp` is judged against
/// its VALUE, because HF Llama's forward pass reads it above 1.
///
/// **2** — the architecture-identity gate. A `model_type` no registry
/// entry matches, and a container/text pair that resolve to different
/// architectures, now block instead of passing silently into
/// `GenericArch`'s Llama-shaped defaults. Measured on the conformance
/// corpus: 15 of 42 declared `model_type` strings, across 30 checkpoints.
pub const PLANNER_SEMANTICS_VERSION: u32 = 24;

/// Who judged a plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannerIdentity {
    /// The crate that implements the planner.
    pub package: String,
    /// That crate's package version — which build ran.
    pub package_version: String,
    /// [`PLANNER_SEMANTICS_VERSION`] at the time — whether two verdicts
    /// are comparable.
    pub semantics_version: u32,
}

impl PlannerIdentity {
    /// This build's identity.
    pub fn current() -> Self {
        Self {
            package: env!("CARGO_PKG_NAME").to_string(),
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            semantics_version: PLANNER_SEMANTICS_VERSION,
        }
    }
}

#[cfg(test)]
mod tests;
