//! Interface, metadata, training-only and alias key tables.

use super::super::report::SemanticClass;

#[allow(unused_imports)]
use super::*;

/// Keys that declare a cross-component contract: hidden-state taps, block
/// protocols, special-token roles in a multimodal or drafter interface.
pub const INTERFACE_SEMANTIC_KEYS: &[&str] = &[
    "target_layer_ids",
    "block_size",
    "mask_token_id",
    "image_token_id",
    "video_token_id",
    // The rest of a multimodal join (Gemma 4): the tokens that open,
    // close or stand in for an audio / image span, how many soft tokens
    // an image expands to (declared twice — root and tower), the span
    // kind the text model attends bidirectionally over, and a tower the
    // checkpoint declares it does NOT have (`audio_config: null`).
    "audio_token_id",
    "boi_token_id",
    "eoi_token_id",
    "boa_token_id",
    "eoa_token_id",
    "eoa_token_index",
    "vision_soft_tokens_per_image",
    "default_output_length",
    "use_bidirectional_attention",
    "audio_config",
    // Gemma 3's spellings of the same image join: `Gemma3Config` names
    // the soft token and its delimiters `*_index` where Gemma 4 says
    // `*_id`, and the soft-token count `mm_tokens_per_image`. Read by
    // the interface reader under these names, so they are credited as
    // read and, like Gemma 4's, required by the image capability alone.
    "image_token_index",
    "boi_token_index",
    "eoi_token_index",
    "mm_tokens_per_image",
];

/// Identity facts inert for a forward pass wherever they appear.
pub const METADATA_KEYS: &[&str] = &[
    // HF's dynamic-import map: which Python class to load for this
    // `model_type`. Loader plumbing for another runtime entirely — it
    // names code, not a forward-pass fact, and two checkpoints differing
    // only here compute identical logits.
    "AutoConfig",
    "AutoModel",
    "AutoModelForCausalLM",
    "model_type",
    "tie_word_embeddings",
    // The mamba_ssm lineage spellings of the same fact, read by the same
    // parser fallback chain.
    "tie_embedding_weights",
    "tie_embeddings",
    // Whether the reference runtime fuses the residual add with the norm
    // — a kernel-schedule fact about the SAME operation, the exact class
    // `cache_implementation` sits in: two checkpoints differing only
    // here compute the same function.
    "fused_add_norm",
    // `rope_scaling` as a bare leaf (not recursed into) means its value is
    // not an object — in every checkpoint on hand, `null`. A non-null
    // object never reaches this leaf; it flattens into `rope_type`/
    // `factor`/etc. instead, covered above. So a bare `rope_scaling` fact
    // carries no scaling information to lose — the same claim
    // `max_position_embeddings` makes about itself, just true unconditionally
    // here rather than by schema absence.
    "rope_scaling",
    // HF's serving-time KV-cache implementation selector (`"hybrid"`,
    // `"static"`, …) — which cache *class* generation code should
    // instantiate to hold a mix of sliding/full attention layers
    // efficiently. It names a consequence of the per-layer attention
    // topology, not an independent forward-pass fact: the topology itself
    // is declared elsewhere (`sliding_window` + the architecture's layer
    // alternation, e.g. Gemma 2's fixed period-2 pattern) and VINDEX3
    // already carries *that*, per layer, in the attention table. Two
    // checkpoints differing only in `cache_implementation` compute
    // identical logits for any prompt both can hold.
    "cache_implementation",
];

/// Keys that parameterise *training* and are inert at inference. Each
/// entry must name the training-time path it belongs to, because "we
/// don't run that" is the entire justification for dropping it.
pub const TRAINING_ONLY_KEYS: &[&str] = &[
    // MoE load-balancing auxiliary loss: added to the training objective,
    // never read on a forward pass.
    "router_aux_loss_coef",
    // Whether the model *returns* router logits alongside hidden states —
    // a training/analysis output switch. It changes what is returned, not
    // what is computed, and generic execution returns logits only.
    "output_router_logits",
    // Mamba2 `dt_bias` initialisation bounds (`Mamba2Mixer.__init__`
    // samples dt in [time_step_min, time_step_max], floors it at
    // time_step_floor, and stores softplus⁻¹ of it as the initial
    // `dt_bias`). Once the checkpoint ships a trained `dt_bias` tensor,
    // these parameterise nothing a forward pass reads — the forward-time
    // clamp is `time_step_limit`, which is execution-semantic above.
    "time_step_floor",
    "time_step_min",
    "time_step_max",
    // Mamba1's dt-projection rank. Mamba2's dt is a per-head scalar from
    // the fused `in_proj`; transformers carries the field on
    // `Mamba2Config` for lineage and its `__init__` alone reads it.
    "time_step_rank",
    // Weight-init scaling for the residual projections
    // (`_init_weights` divides by √(2·n_layer) when set) — same class as
    // `initializer_range`, inert once training is over.
    "rescale_prenorm_residual",
    // The mamba_ssm lineage's per-tensor init bounds (OuteAI Mamba2Attn):
    // A's log-uniform sampling range, and the conv/embedding init ranges
    // — the same class as `initializer_range`, split per tensor. Inert
    // once the trained tensors ship.
    "A_initializer_range",
    "conv_initializer_range",
    "emb_initializer_range",
    // Dropout on a classification head this causal-LM config does not
    // have — and dropout is identity at inference regardless.
    "classifier_dropout",
];

/// Redundant spellings: `alias → canonical`. An entry claims the same
/// fact is declared under `canonical` *in the same config* and read
/// there, which the gate verifies — so listing a key here cannot silence
/// it if the canonical spelling is missing or disagrees.
pub const ALIAS_KEYS: &[(&str, &str)] = &[
    // GPT-OSS declares both spellings, with the same value; the parser's
    // alias list reads `num_experts_per_tok`.
    ("experts_per_token", "num_experts_per_tok"),
    // The pre-scaling context length, also declared inside the rope
    // scaling block, which is where the parser reads it.
    (
        "initial_context_length",
        "rope_scaling.original_max_position_embeddings",
    ),
    // The regular-interval spelling of a hybrid interleave ("every Nth
    // layer is full attention"): a compressed encoding of exactly the
    // fact the explicit `layer_types` array states per layer. Qwen3.5
    // declares both; the parser reads the array. Benign only while
    // `layer_types` is genuinely present and consumed in the same
    // config — the gate verifies that, same as every other alias.
    ("full_attention_interval", "layer_types"),
    // Gemma 3's backward-compatible spelling of the same stride
    // (`Gemma3TextConfig.__post_init__` stores the declared
    // `sliding_window_pattern` as `_sliding_window_pattern` and derives
    // `layer_types` from it only when the array is absent). The 270M
    // checkpoints serialise both; the array is read, and the period must
    // corroborate it — `alias_contradicts_canonical` checks it the same
    // way it checks `full_attention_interval`.
    ("_sliding_window_pattern", "layer_types"),
];

/// Reviewed-and-safe-to-drop keys. Empty by design until a key has actually
/// been reviewed; every future entry must carry a justification comment.
pub const IGNORED_SAFE_KEYS: &[&str] = &[];

/// Keys that configure a model COMPONENT this build does not implement:
/// `leaf → component`.
///
/// The point is arithmetic. A component is one piece of engineering
/// whatever its key count, so nine keys naming one absent indexer should
/// read as one job and not nine mysteries. `Unknown` cannot say that —
/// it means "nobody has looked" — and a report made mostly of `unknown`
/// tells you how much was unexamined rather than how much is left.
///
/// # The registration rule
///
/// **An entry requires positive evidence of component ownership, never
/// merely plausible adjacency.**
///
/// Concretely, that rules out the shortcuts that would make this table
/// cheap to extend:
///
/// * no prefix-only registration — `index*` is how these keys were
///   *found*, not why they are grouped;
/// * no regex or pattern rule, ever. A regex over `rope` is what filed
///   `indexer_rope_interleave` under general RoPE, where acting on it
///   would have re-paired the whole model's rotary against the wrong
///   partners;
/// * a key that merely sits beside a registered one stays
///   [`SemanticClass::Unknown`], which is the honest answer.
///
/// What counts as positive evidence is a semantic witness: a geometry
/// self-consistent and distinct from the model's own, a value that is a
/// selection count rather than a width, an array whose length equals the
/// layer count and so is a per-layer schedule for *this* component. The
/// discovery tool may be a pattern; the authority may not be.
///
/// The cost of relaxing this is not a wrong label — it is an engineering
/// estimate that reads tidier than the evidence supports, which is worse
/// than no estimate.
pub const UNSUPPORTED_COMPONENT_KEYS: &[(&str, &str)] = &[
    // GLM-5.3-Flash's learned SPARSE ATTENTION INDEXER: a side network
    // that scores keys so attention can read `index_topk` of them
    // instead of the whole context.
    //
    // Grouped on evidence rather than on the `index` prefix alone. The
    // geometry is self-consistent and separate from the model's own
    // (`index_n_heads: 32`, `index_head_dim: 128`, against the text
    // stack's `qk_head_dim: 256`); `index_topk: 2048` is a selection
    // count, not a width; the pooling trio describes one mechanism; and
    // `indexer_types` carries exactly 45 entries against
    // `num_hidden_layers: 45`, so it is a per-layer schedule for this
    // component the way `layer_types` is for attention.
    //
    // No reference implementation exists to check any of this against:
    // `glm5_next` is absent from transformers 5.5.0 and the repo ships
    // no remote modeling code (72 files, zero `.py`). So the component is
    // NAMED and REFUSED, never guessed at — which is the whole difference
    // between an engineering estimate and a compatibility claim.
    // FP8 ACTIVATION quantisation — a compute-path fact, not a storage
    // one, and the distinction is the whole reason this key is here
    // rather than beside `fmt` and `weight_block_size`.
    //
    // `activation_scheme: "dynamic"` says the reference kernel quantises
    // the ACTIVATION to FP8 at run time and runs an FP8 GEMM. This build
    // dequantises the weights to f32 and runs an f32 GEMM: numerically
    // close, but a different route, and the difference is not the
    // storage codec (which is reproduced bit-exactly). Classifying this
    // as represented would claim an execution path that does not exist,
    // so it is NAMED and REFUSED — the storage support is real and
    // stops precisely here.
    ("activation_scheme", FP8_ACTIVATION_QUANT),
    ("index_head_dim", GLM_SPARSE_INDEXER),
    ("index_n_heads", GLM_SPARSE_INDEXER),
    ("index_topk", GLM_SPARSE_INDEXER),
    ("index_kpool", GLM_SPARSE_INDEXER),
    ("index_kpool_compress", GLM_SPARSE_INDEXER),
    ("index_kpool_always_select_tail", GLM_SPARSE_INDEXER),
    ("index_share_for_mtp_iteration", GLM_SPARSE_INDEXER),
    ("indexer_types", GLM_SPARSE_INDEXER),
    // The indexer's OWN rotary pairing, and the reason this key is worth
    // singling out: a regex over `rope` swept it into the general RoPE
    // cluster, where "fixing" it would have meant applying an interleaved
    // pairing to the whole model. Wrong component AND wrong operator, and
    // the checkpoint declares `true`, so the mistake would have been live
    // rather than latent.
    ("indexer_rope_interleave", GLM_SPARSE_INDEXER),
    // Hyper-connections (`hc_mult`, `hc_sinkhorn_iters`, `hc_eps`) left
    // this table in wave 19: the topology they configure is executed on
    // both traversals, so they are execution semantics carried to the
    // component's residual topology — see [`EXECUTION_SEMANTIC_KEYS`] and
    // the carriage rules. `mhc: true` sits beside them and is NOT listed
    // anywhere: it is a bare boolean whose expansion cannot be checked
    // without a reference, and guessing it into a table is exactly the
    // failure the tables' contract forbids. It stays `unknown`, which is
    // what it is.
];

/// Component label for GLM's learned sparse attention indexer.
pub(super) const GLM_SPARSE_INDEXER: &str = "sparse attention indexer (GLM-5.x)";
/// The FP8 scheme's run-time ACTIVATION quantisation. Named apart from
/// the storage codec because this build reproduces the storage exactly
/// and implements none of this.
pub(super) const FP8_ACTIVATION_QUANT: &str = "FP8 activation quantisation (dynamic per-tensor)";

/// The unimplemented component this leaf configures, if any.
pub fn unsupported_component(leaf: &str) -> Option<&'static str> {
    UNSUPPORTED_COMPONENT_KEYS
        .iter()
        .find(|(key, _)| *key == leaf)
        .map(|(_, component)| *component)
}

/// The canonical spelling this leaf aliases, if it is a registered alias.
pub fn alias_canonical(leaf: &str) -> Option<&'static str> {
    ALIAS_KEYS
        .iter()
        .find(|(alias, _)| *alias == leaf)
        .map(|(_, canonical)| *canonical)
}

/// `generate()` defaults that ship inside `config.json`.
///
/// Transformers moved decoding policy to `generation_config.json` years
/// ago and still reads these from the model config for old checkpoints,
/// so they keep appearing. Every entry selects among a forward pass's
/// outputs or says what `generate()` returns; none changes what the
/// forward pass computes, which is what a container represents.
///
/// The contract is the same as every table here: an entry is a claim
/// about a specific key, not a pattern. `top_k` is listed because it is
/// the sampling cutoff; MoE's `num_experts_per_tok` is the routing
/// top-k and is execution-semantic, which is exactly why membership is
/// by exact leaf name and never by a substring.
pub const GENERATION_POLICY_KEYS: &[&str] = &[
    // Sampling policy.
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
    "repetition_penalty",
    "encoder_repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size",
    "exponential_decay_length_penalty",
    "renormalize_logits",
    "remove_invalid_values",
    // Search policy.
    "num_beams",
    "num_beam_groups",
    "diversity_penalty",
    "early_stopping",
    "num_return_sequences",
    "penalty_alpha",
    // Length policy.
    "max_length",
    "min_length",
    "max_new_tokens",
    "min_new_tokens",
    // Token constraints. Vocabulary *positions*, not model geometry:
    // banning a token changes which of the logits may be selected, never
    // the logits.
    "bad_words_ids",
    "begin_suppress_tokens",
    "suppress_tokens",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "forced_decoder_ids",
    // What `generate()` hands back. `output_hidden_states` and
    // `output_attentions` make the forward pass *retain* intermediates;
    // they do not change the values it computes.
    "output_scores",
    "output_attentions",
    "output_hidden_states",
    "output_logits",
    "return_dict",
    "return_dict_in_generate",
    "num_logits_to_keep",
    "use_cache",
    // Legacy per-task default blocks (`task_specific_params.*`) and the
    // GPT-2-era sequence-classification head knobs, which describe a head
    // no text-generation container builds.
    "summary_type",
    "summary_use_proj",
    "summary_activation",
    "summary_proj_to_labels",
    "summary_first_dropout",
];

/// Regularisation rates. Dropout is the identity at inference — the
/// module is placed but never active outside training — so the RATE
/// parameterises a path a container never runs.
///
/// Value-independent, unlike [`INERT_AT_VALUE`]: a dropout of 0.0 and one
/// of 0.5 compute the same forward pass here.
pub const DROPOUT_KEYS: &[&str] = &[
    "attn_pdrop",
    "embd_pdrop",
    "resid_pdrop",
    "summary_pdrop",
    "attention_dropout",
    "hidden_dropout",
    "embedding_dropout",
    "residual_dropout",
    "classifier_dropout",
    "mlp_dropout",
    "conv_dropout",
    // MoE router noise, added to logits during training only.
    "input_jitter_noise",
    "router_jitter_noise",
];

/// A value a key must hold to be inert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InertValue {
    /// Inert when the declared integer equals this.
    Int(i64),
    /// Inert when the declared array is empty — a schedule that selects
    /// no layers is not a schedule.
    EmptyList,
    /// Inert when the declared boolean equals this.
    Bool(bool),
}

impl InertValue {
    pub(super) fn matches(self, value: &serde_json::Value) -> bool {
        match self {
            Self::Int(want) => value.as_i64() == Some(want),
            Self::EmptyList => value.as_array().is_some_and(|a| a.is_empty()),
            Self::Bool(want) => value.as_bool() == Some(want),
        }
    }

    /// Whether `value` is spelled in this entry's vocabulary at all — an
    /// integer for [`Self::Int`], an array for [`Self::EmptyList`], a
    /// boolean for [`Self::Bool`]. A value in the vocabulary that is not
    /// the inert one is the case the table exists to keep blocking; a
    /// value in ANOTHER vocabulary is a different declaration under the
    /// same leaf, and the leaf's name classification answers for it.
    pub(super) fn speaks_for(self, value: &serde_json::Value) -> bool {
        match self {
            Self::Int(_) => value.is_i64() || value.is_u64(),
            Self::EmptyList => value.is_array(),
            Self::Bool(_) => value.is_boolean(),
        }
    }
}

/// Keys inert **at one value** and execution-semantic at any other.
///
/// `pretraining_tp` is why this table is keyed by value rather than by
/// name. It reads as a training-time knob and is not one: HF Llama's
/// forward pass branches on it, slicing every projection into `tp` shards
/// and summing them, to reproduce the numerics of the tensor-parallel run
/// that trained the weights. At the `1` that every checkpoint in the
/// conformance corpus declares it is exactly a no-op. Listing the NAME
/// would have silenced a key that changes the forward pass the moment a
/// checkpoint ships `2` — a checked default, never an assumed one.
///
/// An entry here must name a value whose inertness is *verified*, not
/// assumed from the key reading like a default.
pub const INERT_AT_VALUE: &[(&str, InertValue)] = &[
    ("pretraining_tp", InertValue::Int(1)),
    // Qwen's MoE layer schedule: which layers route to an expert bank and
    // which run a plain MLP. `decoder_sparse_step` is the stride (HF:
    // `layer_idx % decoder_sparse_step == 0` is a MoE layer) and
    // `mlp_only_layers` names the exceptions.
    //
    // At a stride of 1 with no exceptions, every layer is a MoE layer —
    // which is the uniform stack the graph already builds, so the pair
    // describes what is already represented. Any OTHER value is a real
    // per-layer topology: a stride of 2 makes half the tower dense, and a
    // non-empty exception list carves out named layers. Neither is
    // expressible today, and both keep blocking.
    //
    // Value-keyed rather than name-listed for exactly that reason. This
    // is the same shape as `pretraining_tp` above: a key that reads like
    // a default and is one only at one value.
    ("decoder_sparse_step", InertValue::Int(1)),
    ("mlp_only_layers", InertValue::EmptyList),
    // `Gemma3TextConfig.use_bidirectional_attention` (transformers 5.5):
    // the TEXT decoder's causal-mask switch. `false` is the causal stack
    // every text-generation checkpoint runs — exactly what the graph
    // builds — so it is inert at that value. `true` (EmbeddingGemma)
    // makes `Gemma3Attention.is_causal` false, overlays a bidirectional
    // sliding mask and halves the window: a different forward pass, which
    // keeps blocking. Gemma 4's `"vision"` STRING is a different
    // declaration under the same leaf — the masking policy over image
    // spans, an interface fact — and is classified by name as before (see
    // [`InertValue::speaks_for`]).
    ("use_bidirectional_attention", InertValue::Bool(false)),
];

/// The inert value registered for this leaf, if any.
pub fn inert_at_value(leaf: &str) -> Option<InertValue> {
    INERT_AT_VALUE
        .iter()
        .find(|(key, _)| *key == leaf)
        .map(|(_, v)| *v)
}

/// [`classify_key`], with the declared value available.
///
/// The value settles exactly one question — whether a key registered in
/// [`INERT_AT_VALUE`] is holding the value that makes it inert. Every
/// other classification is by name, so a key not in that table answers
/// identically here and in [`classify_key`].
pub fn classify_key_at(leaf: &str, value: &serde_json::Value) -> SemanticClass {
    match inert_at_value(leaf) {
        // Inert at this value: declared, preserved, and read by nothing a
        // forward pass runs.
        Some(inert) if inert.matches(value) => SemanticClass::TrainingOnly,
        // Registered, and holding some OTHER value of the same vocabulary:
        // this is the case the table exists to keep blocking.
        Some(inert) if inert.speaks_for(value) => SemanticClass::ExecutionSemantic,
        // A value outside the entry's vocabulary is not a non-inert value
        // of this fact; the name classification answers, and an unjudged
        // leaf still grades `Unknown` and blocks.
        Some(_) | None => classify_key(leaf),
    }
}
