//! The [`Position`] slice of [`super::ModelArchitecture`].

use super::{Norms, UNSCALED_POSITION_DIVISOR};
use crate::config::{
    rope_types, DeclaredRopeScaling, InputPositionEncoding, Llama3RopeScaling, ModelConfig,
    PositionPolicy, RotaryFrequencyBasis, YarnRopeScaling,
};

/// Positional encoding: RoPE bases, scaling schemes and per-layer policy.
pub trait Position: Norms {
    /// RoPE base frequency for a given layer.
    /// Gemma3 uses different bases for sliding vs global attention layers.
    ///
    /// Only meaningful on layers whose
    /// [`position_policy_for_layer`](Self::position_policy_for_layer) is
    /// rotary; the policy is the authority on *whether* a layer rotates.
    fn rope_base_for_layer(&self, layer: usize) -> f64 {
        let _ = layer;
        self.config().rope_base
    }

    /// Positional-encoding policy for a given layer.
    ///
    /// A checkpoint declaring `layer_rope_theta` states the policy per layer
    /// — including intentional absence (NoPE), which the HF form spells as a
    /// `0.0` sentinel honoured here and nowhere else. Without the array,
    /// every layer rotates at [`rope_base_for_layer`](Self::rope_base_for_layer).
    ///
    /// The forward path for a family that declares NoPE layers must consume
    /// this policy, not a raw theta — a zero base is degenerate
    /// (`1/0^(i/d)`), never a parameter value.
    ///
    /// A checkpoint that declares `rope_scaling.rope_type = "yarn"` states a
    /// second execution fact — scaled frequencies and an attention amplitude
    /// — for every rotating layer, and the policy carries it
    /// ([`PositionPolicy::Yarn`]) rather than letting it be read once by the
    /// forward path and dropped by everything else ([`Self::yarn_rope_scaling`]
    /// is the config read; this is where it becomes per-layer policy).
    fn position_policy_for_layer(&self, layer: usize) -> PositionPolicy {
        default_position_policy_for_layer(self, layer)
    }

    /// How position enters at the input, before any layer: a learned
    /// absolute table added to the token embedding, when the family has
    /// one ([`TensorKeys::position_embed_key`](super::TensorKeys::position_embed_key)).
    /// Its row count is the declared serving bound
    /// (`max_position_embeddings`, GPT-2's `n_positions`); a family with a
    /// table and no declared bound answers `None` here and operand closure
    /// refuses the unaccounted table by name.
    fn input_position_encoding(&self) -> Option<InputPositionEncoding> {
        self.position_embed_key()?;
        let max_positions = self.config().max_position_embeddings?;
        Some(InputPositionEncoding::LearnedAbsolute { max_positions })
    }

    /// The unscaled rotary policy at `theta`: plain, partial, or
    /// multi-axis, decided by what the config declares.
    ///
    /// Lives in the trait default rather than a family override because
    /// `partial_rotary_factor` and `rope_parameters.mrope_*` are config
    /// facts, not family facts — a model declaring them means the same
    /// thing whatever its `model_type`. A family whose partial rotary
    /// uses a different frequency basis (Gemma 4, HF `proportional`)
    /// overrides [`Self::position_policy_for_layer`] and never reaches
    /// here.
    ///
    /// A declared fraction of `1.0` is a full rotary and stays
    /// [`PositionPolicy::Rope`]: the fraction is then not a partial
    /// rotary fact, and manufacturing a `PartialRope` for it would put
    /// every ordinary model on the partial path.
    fn rotary_policy(&self, theta: f64) -> PositionPolicy {
        let cfg = self.config();
        let Some(rotary_fraction) = cfg.partial_rotary_factor.filter(|f| *f < 1.0) else {
            return PositionPolicy::Rope { theta };
        };
        // HF's `default` rope with `partial_rotary_factor` takes the
        // inverse frequencies over the ROTARY width — `base**(arange(0,
        // dim, 2) / dim)` with `dim = head_dim * factor`. The head-width
        // basis is a different declaration (`proportional`) and a
        // different family's override.
        let basis = RotaryFrequencyBasis::RotaryWidth;
        match (cfg.mrope_section.as_deref(), cfg.mrope_interleaved) {
            (Some([t, h, w]), Some(interleaved)) => PositionPolicy::MRope {
                theta,
                rotary_fraction,
                basis,
                section: [*t, *h, *w],
                interleaved,
            },
            // Either half alone is an incomplete declaration. Resolving
            // it as a plain partial rotary would drop the multi-axis
            // fact silently, so the partial policy is built and the
            // plan's carriage gate refuses the unpaired key — the
            // refusal belongs where refusals are reported, not here.
            _ => PositionPolicy::PartialRope {
                theta,
                rotary_fraction,
                basis,
            },
        }
    }

    /// Fraction of head_dim to apply RoPE to (0.0–1.0).
    /// Models with partial rotary embedding (e.g., 0.25) override per layer.
    /// Default: 1.0 (full rotation).
    fn rotary_fraction_for_layer(&self, _layer: usize) -> f64 {
        1.0
    }

    /// RoPE scaling type (None, "linear", "yarn", "dynamic", "llama3").
    fn rope_scaling_type(&self) -> Option<&str> {
        self.config()
            .rope_scaling
            .as_ref()
            .map(|s| s.scaling_type.as_str())
    }

    /// RoPE scaling factor.
    fn rope_scaling_factor(&self) -> f64 {
        self.config()
            .rope_scaling
            .as_ref()
            .map_or(1.0, |s| s.factor)
    }

    /// Per-layer RoPE position divisor from `rope_scaling`: the linear
    /// `factor` when the checkpoint declares `rope_type: "linear"`,
    /// [`UNSCALED_POSITION_DIVISOR`] otherwise.
    ///
    /// **The read lives in the trait default**, for the reason recorded
    /// on [`Self::yarn_rope_scaling`]: `rope_type: "linear"` is a config
    /// fact, and HF's `_compute_linear_scaling_rope_parameters` applies
    /// it to every rotating layer of any family that declares it. This
    /// used to return `1.0` unconditionally and let Gemma 3 opt in, which
    /// is the shape that doc-comment warns about — a Llama-2 long-context
    /// checkpoint declaring `{type: linear, factor: 2}` was served
    /// unscaled.
    ///
    /// Gemma 3 overrides this to return the `factor` on global layers
    /// only and `1.0` on sliding layers, matching the HF
    /// `Gemma3TextConfig.rope_scaling.full_attention` structure — the
    /// legitimate override: which layers the block reaches is fixed by
    /// the architecture, not by the config.
    fn rope_position_divisor_for_layer(&self, _layer: usize) -> f64 {
        self.linear_rope_scaling()
            .unwrap_or(UNSCALED_POSITION_DIVISOR)
    }

    /// The linear position divisor when the checkpoint declares
    /// `rope_scaling = {rope_type: linear, factor}`; `None` for every
    /// other scaling family and for no block at all.
    ///
    /// The checkpoint-wide declaration. Which layers it reaches is
    /// [`Self::rope_position_divisor_for_layer`]'s answer.
    fn linear_rope_scaling(&self) -> Option<f64> {
        let rs = self.config().rope_scaling.as_ref()?;
        rs.scaling_type
            .eq_ignore_ascii_case(rope_types::ROPE_TYPE_LINEAR)
            .then_some(rs.factor)
    }

    /// `llama3` RoPE scaling parameters when the checkpoint declares them.
    ///
    /// **The read lives in the trait default**, for the reason recorded on
    /// [`Self::yarn_rope_scaling`] below. This used to return `None` and
    /// make each family opt in, which is the shape that doc-comment warns
    /// about: `rope_type: "llama3"` is a *config fact*, and a checkpoint
    /// declaring it was served the wrong frequencies unless its
    /// architecture happened to have overridden this method. Only
    /// `llama.rs` had, so a family arriving with Llama-3 scaling under any
    /// other `model_type` silently lost it.
    ///
    /// The band factors default because HF defaults them; the pre-trained
    /// context window does too — unlike YaRN, whose correction bounds are
    /// undefined without it, `_compute_llama3_parameters` reads
    /// `original_max_position_embeddings` from the config proper when the
    /// block omits it.
    fn llama3_rope_scaling(&self) -> Option<Llama3RopeScaling> {
        let rs = self.config().rope_scaling.as_ref()?;
        if !rs
            .scaling_type
            .eq_ignore_ascii_case(rope_types::ROPE_TYPE_LLAMA3)
        {
            return None;
        }
        Some(Llama3RopeScaling {
            factor: rs.factor,
            low_freq_factor: rs
                .llama3_low_freq_factor
                .unwrap_or(crate::defaults::LLAMA3_LOW_FREQ_FACTOR_DEFAULT),
            high_freq_factor: rs
                .llama3_high_freq_factor
                .unwrap_or(crate::defaults::LLAMA3_HIGH_FREQ_FACTOR_DEFAULT),
            original_max_position_embeddings: rs
                .llama3_original_max_position_embeddings
                .unwrap_or(crate::defaults::LLAMA3_ORIGINAL_MAX_POSITION_EMBEDDINGS_DEFAULT),
        })
    }

    /// `yarn` RoPE scaling parameters when the checkpoint declares them.
    ///
    /// **The read lives in the trait default deliberately.** [§4.7.8] recorded
    /// three recurrences of one shape: a behaviour that is a *config fact*,
    /// read on one architecture, with a trait default silently answering for
    /// everyone else. `rope_type: "yarn"` is a config fact — GPT-OSS and
    /// DeepSeek both ship it — so an architecture must not have to opt in to
    /// being served correctly. Anything that declares YaRN in `config.json`
    /// gets YaRN, and a new family arriving with it needs no code at all.
    ///
    /// [§4.7.8]: ../../../docs/k3-funnel.md
    fn yarn_rope_scaling(&self) -> Option<YarnRopeScaling> {
        let rs = self.config().rope_scaling.as_ref()?;
        if !rs
            .scaling_type
            .eq_ignore_ascii_case(rope_types::ROPE_TYPE_YARN)
        {
            return None;
        }
        Some(YarnRopeScaling {
            factor: rs.factor,
            beta_fast: rs.yarn_beta_fast.unwrap_or(crate::defaults::YARN_BETA_FAST),
            beta_slow: rs.yarn_beta_slow.unwrap_or(crate::defaults::YARN_BETA_SLOW),
            // Required, not defaulted: this is the window the model was
            // *pre-trained* at, and YaRN's correction bounds are defined
            // against it. HF indexes it unconditionally
            // (`rope_parameters_dict["original_max_position_embeddings"]`) and
            // raises if it is absent, so there is no value to inherit. A yarn
            // block without it is malformed and resolves to `None` here —
            // pinned by `yarn_without_original_context_is_not_scaling`.
            original_max_position_embeddings: rs.llama3_original_max_position_embeddings?,
            truncate: rs.yarn_truncate.unwrap_or(crate::defaults::YARN_TRUNCATE),
            mscale: rs.yarn_mscale,
            mscale_all_dim: rs.yarn_mscale_all_dim,
        })
    }

    /// Which frequency-scaling family this checkpoint declares.
    ///
    /// The one place the question is answered. `rope_type` holds a single
    /// value, so the families are alternatives, not a set — asking the two
    /// accessors separately at each call site would invent a "both" state
    /// and leave every site to pick a precedence of its own.
    fn declared_rope_scaling(&self) -> DeclaredRopeScaling {
        if let Some(yarn) = self.yarn_rope_scaling() {
            return DeclaredRopeScaling::Yarn(yarn);
        }
        if let Some(llama3) = self.llama3_rope_scaling() {
            return DeclaredRopeScaling::Llama3(llama3);
        }
        if let Some(factor) = self.linear_rope_scaling() {
            return DeclaredRopeScaling::Linear { factor };
        }
        DeclaredRopeScaling::None
    }
}

/// The family-agnostic position policy — what
/// [`ModelArchitecture::position_policy_for_layer`] resolves unless a
/// family overrides it.
///
/// A free function as well as a trait default so that an override can
/// **narrow** the decision and then defer to it. A family whose config
/// gates rotation on its own key — `granitemoehybrid`'s
/// `position_embedding_type` — has to answer that question first and
/// this one second, and Rust gives an override no way to call the
/// default it replaced. Copying the body into the override instead
/// would leave two resolvers to keep in agreement, which is the exact
/// shape [`PositionPolicy`] exists to prevent.
pub fn default_position_policy_for_layer<A: Position + ?Sized>(
    arch: &A,
    layer: usize,
) -> PositionPolicy {
    // A declared relative scheme decides the policy outright. Checked
    // before every rotary branch because `rope_base` carries a
    // DEFAULT: without this, a checkpoint that declares no rope key at
    // all still resolves to `Rope { theta: 10000 }`, which is a
    // rotation the author never asked for on every layer.
    if let (Some(d_rel), Some(extent)) = (arch.config().d_rel, arch.config().rel_extent) {
        return PositionPolicy::Relative { d_rel, extent };
    }
    // `mla_use_nope` means what it says: the MLA block applies no
    // positional rotation at all.
    //
    // Judged from Kimi Linear's own `modeling_kimi.py`, not from the
    // flag's name, because the config looks self-contradictory — it
    // declares `mla_use_nope: true` *and* `qk_rope_head_dim: 64`. The
    // reference settles it two ways:
    //
    //   1. the file contains **no rotary code whatsoever** — `q_rot`
    //      and `k_rot` are split out and concatenated straight back,
    //      unrotated;
    //   2. `arch.use_nope` is read exactly once, as `assert
    //      arch.use_nope` — the flag is a *precondition*, not a
    //      switch, and the class refuses to run without it.
    //
    // So `qk_rope_head_dim` is a **structural width**, not a rotary
    // subspace: it splits `q_head_dim = 128 + 64 = 192` (q_proj rows
    // 32·192 = 6144, as stored) and gives `kv_a_proj_with_mqa` its
    // extra 64 outputs, broadcast across heads as a shared unrotated K
    // component. The key name is actively misleading and only the
    // reference could settle it.
    //
    // Deliberately keyed on `Some(true)`. `false` is a combination the
    // reference does not implement — its assert fires — so this build
    // has no ground truth for it and must not answer.
    if arch.config().mla_use_nope == Some(true) {
        return PositionPolicy::None;
    }
    // A family that adds learned absolute positions at the input and
    // declares no rotary key rotates nothing. Without this, GPT-2 — whose
    // config states no rope key at all — resolved to `Rope { theta:
    // 10000 }` off the parser's DEFAULT base on every layer.
    if arch.position_embed_key().is_some() && !arch.config().declares_rotary {
        return PositionPolicy::None;
    }
    // The per-layer rotary SCHEDULE, asked before the rotary SHAPE.
    // `no_rope_layers` says whether this layer rotates at all; everything
    // below says how a rotating layer rotates. Composed rather than
    // branched so a scheduled layer still picks up YaRN, a partial
    // rotary, or a per-layer theta — the drop `layer_rope_theta`'s branch
    // already guards against, one key over.
    if !rope_scheduled_for_layer(arch.config(), layer) {
        return PositionPolicy::None;
    }
    // One resolution of "which scaling family did this checkpoint
    // declare", asked once and composed with the per-layer theta below.
    let scaling = arch.declared_rope_scaling();
    // Linear scaling is declared once for the checkpoint but REACHES a
    // layer by the architecture's answer: HF applies Gemma 3's block to
    // its full-attention layers only. Resolved per layer here, before
    // the two branches below, so each sees the divisor this layer
    // actually runs under — `None` on a layer the family leaves plain.
    let scaling = match scaling {
        DeclaredRopeScaling::Linear { .. } => {
            let divisor = arch.rope_position_divisor_for_layer(layer);
            if divisor == UNSCALED_POSITION_DIVISOR {
                DeclaredRopeScaling::None
            } else {
                DeclaredRopeScaling::Linear { factor: divisor }
            }
        }
        declared => declared,
    };
    match arch
        .config()
        .layer_rope_theta
        .as_ref()
        .and_then(|thetas| thetas.get(layer))
    {
        Some(&declared) => {
            // A per-layer theta states WHICH base this layer rotates
            // at. It does not state that the layer stops being a
            // partial or multi-axis rotary, so the config's rotary
            // shape is re-applied at that theta. Without this, any
            // checkpoint declaring `layer_rope_theta` alongside
            // `partial_rotary_factor` would silently rotate the whole
            // head — the same drop this rung exists to close, one
            // branch over.
            match PositionPolicy::from_declared_theta_with_scaling(declared, scaling) {
                PositionPolicy::Rope { theta } => arch.rotary_policy(theta),
                // NoPE has no rotary shape, and YaRN carries its own.
                resolved => resolved,
            }
        }
        None => match scaling {
            DeclaredRopeScaling::Yarn(scaling) => PositionPolicy::Yarn {
                theta: arch.rope_base_for_layer(layer),
                scaling,
            },
            DeclaredRopeScaling::Llama3(scaling) => PositionPolicy::Llama3 {
                theta: arch.rope_base_for_layer(layer),
                scaling,
            },
            // Composed with the rotary SHAPE, not built over it: a linear
            // divisor on a partial or multi-axis rotary has no variant,
            // so that layer keeps its shape and the plan's carriage gate
            // refuses the unpaired `factor` — the same fallthrough
            // `from_declared_theta_with_scaling` takes, so the two
            // branches of this function cannot disagree.
            DeclaredRopeScaling::Linear { factor } => {
                match arch.rotary_policy(arch.rope_base_for_layer(layer)) {
                    PositionPolicy::Rope { theta } => PositionPolicy::Linear { theta, factor },
                    shaped => shaped,
                }
            }
            DeclaredRopeScaling::None => arch.rotary_policy(arch.rope_base_for_layer(layer)),
        },
    }
}

/// Whether the declared rotary schedule rotates `layer` at all.
///
/// Two spellings of one fact, and they are NOT equal partners: both
/// SmolLM3 and Llama 4 build the mask from the interval only
/// `if no_rope_layers is None`, so an explicit mask SUPERSEDES a declared
/// interval rather than being cross-checked against it. Preferring the
/// interval — or reconciling the two — would put this build on a
/// schedule the reference never runs.
///
/// A mask shorter than the stack makes no statement about the layers past
/// its end (both references document "at least the same length as the
/// number of layers" and index it directly), so those layers keep the
/// unscheduled behaviour rather than acquiring a guessed one.
fn rope_scheduled_for_layer(cfg: &ModelConfig, layer: usize) -> bool {
    if let Some(mask) = cfg.no_rope_layers.as_ref() {
        return mask
            .get(layer)
            .copied()
            .is_none_or(PositionPolicy::rope_enabled_by_flag);
    }
    if let Some(interval) = cfg.no_rope_layer_interval {
        return PositionPolicy::rope_enabled_by_interval(layer, interval);
    }
    true
}
