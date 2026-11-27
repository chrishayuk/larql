//! The [`FeedForward`] slice of [`super::ModelArchitecture`].

use super::{Attention, SITU_DEFAULT_BETA};
use crate::config::{
    Activation, ActivationDeclaration, ExpertFormat, ExpertGatePolicy, ExpertRoutingPolicy,
    FfnType, GateUpLayout, HyperConnection, ResidualTopology, SharedExpertGateSpec, SITU_NAME,
};

/// The FFN block: activation, dense FFN, and every Mixture-of-Experts fact.
pub trait FeedForward: Attention {
    /// What this checkpoint's `hidden_act` declaration says — silent, a
    /// judged nonlinearity, the name of a gate policy, or a name this
    /// build has never judged.
    ///
    /// The judgment every FFN decision reads. Overriding it is asserting
    /// the architecture knows better than its own checkpoint.
    fn activation_declaration(&self) -> ActivationDeclaration {
        ActivationDeclaration::judge(self.config().hidden_act.as_deref())
    }

    /// Activation function for the FFN.
    ///
    /// SiLU when the config is **silent** — that is a checked default,
    /// and the only branch that takes it. A judged name maps through the
    /// one table in [`Activation::from_hf_name`].
    ///
    /// The two remaining declarations have no nonlinearity to return, and
    /// this method deliberately does not invent one:
    ///
    /// - [`ActivationDeclaration::NamesGatePolicy`] — the whole combine is
    ///   named, not just the gate's nonlinearity, and
    ///   [`Self::expert_gate_policy`] carries it. The value returned here
    ///   is INERT: [`ExpertGatePolicy`]'s non-`Gated` arms consume no
    ///   `Activation` at all, which
    ///   `situ_policy_makes_the_activation_field_inert` pins.
    /// - [`ActivationDeclaration::Unjudged`] — nothing in this build knows
    ///   what the name means. The value returned here is likewise never
    ///   executed: [`Self::gate_up_is_gelu_tanh`] refuses by name before
    ///   any kernel is selected, which is where the refusal belongs.
    ///
    /// Before K3-ACT-1 this method collapsed *silent* and *unjudged* into
    /// one `unwrap_or(Silu)`, so a checkpoint declaring `situ` or `relu2`
    /// was silently executed as SwiGLU. Two rows in the conformance
    /// estate were in exactly that state.
    /// Whether the dense FFN's up and down projections carry additive
    /// biases. Reads `mlp_bias` when the checkpoint declares it; `None` =
    /// unjudged, and a bias operand shipped under `None` or `Some(false)`
    /// fails VINDEX3 operand closure.
    fn ffn_bias(&self) -> Option<bool> {
        self.config().mlp_bias
    }

    fn activation(&self) -> Activation {
        match self.activation_declaration() {
            ActivationDeclaration::Nonlinearity(activation) => activation,
            ActivationDeclaration::Absent
            | ActivationDeclaration::NamesGatePolicy(_)
            | ActivationDeclaration::Unjudged(_) => Activation::Silu,
        }
    }

    /// Which of the two gate/up kernel families this checkpoint's FFN uses
    /// on the walk / kquant / dense-weight paths — the ONE call those
    /// paths make.
    ///
    /// **Panics, by name, when the declaration cannot be served there**:
    /// a gate policy that is not plain gating
    /// ([`ExpertGatePolicy::SituGlu`], [`ExpertGatePolicy::ClampedGlu`]),
    /// or an activation name this build has never judged. Those paths
    /// compute `act(gate) * up` and nothing else, and handing them a bool
    /// for a declaration they cannot serve is precisely the silent
    /// substitution this refuses.
    ///
    /// A panic rather than a `Result` because that is already the
    /// established contract at this exact seam —
    /// [`Activation::uses_gelu_tanh_gate_up`] panics for
    /// [`Activation::Relu`] with the same reasoning — and because these
    /// paths are reached only after a plan has admitted the model. The
    /// planner refuses both known specimens, so this is a backstop, which
    /// is what it should be.
    fn gate_up_is_gelu_tanh(&self) -> bool {
        let policy = self.expert_gate_policy();
        assert!(
            matches!(policy, ExpertGatePolicy::Gated),
            "the walk/kquant gate-up paths compute `act(gate) * up` and have no kernel for              {policy:?}; refusing rather than substituting plain gating for it"
        );
        // The DECLARATION decides whether to refuse; [`Self::activation`]
        // supplies the answer. Those are two different questions and
        // answering both from the declaration was a real defect: StarCoder2
        // overrides `activation()` to tanh-GELU and declares no
        // `hidden_act` at all, so a derivation reading the config alone
        // silently replaced a family's own judgment with the SiLU
        // fallback — the same shape of bug this rung exists to remove, one
        // level up.
        match self.activation_declaration() {
            ActivationDeclaration::Absent | ActivationDeclaration::Nonlinearity(_) => {
                self.activation().uses_gelu_tanh_gate_up()
            }
            // Unreachable while the policy is `Gated` (a policy name
            // resolves to a non-`Gated` policy above), but stated rather
            // than collapsed into the arm below: the two declarations are
            // different facts and a future policy name must not silently
            // inherit an `unjudged` message.
            ActivationDeclaration::NamesGatePolicy(name) => panic!(
                "`hidden_act: \"{name}\"` names a gate policy, and the walk/kquant gate-up \
                 paths have no kernel for it; refusing rather than substituting plain gating"
            ),
            // Refused even where a family overrides `activation()`: the
            // checkpoint declared a name this build cannot read, the
            // family's answer and the declaration disagree, and the plan
            // already grades that leaf `mismatched`. Computing anything
            // here would be picking a side silently. No in-tree
            // architecture is in this state.
            ActivationDeclaration::Unjudged(name) => panic!(
                "`hidden_act: \"{name}\"` is an activation this build has never judged; the \
                 walk/kquant gate-up paths refuse it rather than computing SiLU in its place"
            ),
        }
    }

    /// FFN type (gated vs standard).
    fn ffn_type(&self) -> FfnType {
        FfnType::Gated
    }

    /// How expert weights are stored in this model.
    fn expert_format(&self) -> ExpertFormat {
        ExpertFormat::PerExpert
    }

    /// How this checkpoint's fused `gate_up` operand splits into the gate and
    /// up branches, or `None` where no fused operand exists (dense models and
    /// per-expert MoE, which store `gate_proj`/`up_proj` separately).
    ///
    /// `None` is *not* a usable default for a packed architecture: readers
    /// must refuse rather than guess, because the two layouts differ by a
    /// silent permutation of rows rather than by anything a shape check
    /// could catch. See [`GateUpLayout`] for the two families that already
    /// disagree.
    fn gate_up_layout(&self) -> Option<GateUpLayout> {
        None
    }

    /// Whether this model uses Mixture of Experts.
    ///
    /// Answered from the **declaration**, not from a family list: a
    /// checkpoint that states a routed-expert count is an MoE whether or
    /// not this build has a registry entry for it. The previous `false`
    /// default meant an unregistered MoE resolved as dense — Kimi Linear,
    /// with 256 experts per layer and top-8 routing, produced an execution
    /// surface saying `ffn: dense, intermediate_size 9216`, which is the
    /// dense-layer width of one layer out of twenty-seven.
    ///
    /// Losing an MoE this way is not a gap in a report. It is a container
    /// that would describe the wrong model.
    fn is_moe(&self) -> bool {
        self.config().num_experts.is_some_and(|experts| experts > 0)
    }

    /// Number of routed experts per layer.
    fn num_experts(&self) -> usize {
        self.config().num_experts.unwrap_or(0)
    }

    /// Number of experts activated per token.
    fn num_experts_per_token(&self) -> usize {
        self.config().num_experts_per_token.unwrap_or(0)
    }

    /// Number of shared (always-active) experts.
    fn num_shared_experts(&self) -> usize {
        self.config().num_shared_experts.unwrap_or(0)
    }

    /// Router weight key for expert selection.
    fn moe_router_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// The routing rule this architecture's MoE block uses.
    ///
    /// Override this, not [`Self::moe_router_type`] — the string form is
    /// derived from it. Returning a *typed* kind is what lets the compute
    /// layer `match` exhaustively instead of comparing strings and falling
    /// back silently on any value it has not heard of.
    fn moe_router_kind(&self) -> crate::config::MoeRouterKind {
        // The declared scoring function decides the rule. Answering the
        // softmax default for a checkpoint that declares `sigmoid` states
        // a routing rule the model does not use — and it is not a small
        // difference: sigmoid scores are independent, so the selected
        // weights do not sum to 1.
        //
        // An unrecognised spelling keeps the default here and is caught by
        // the plan's declared-vs-resolved comparison instead, which can
        // refuse where this signature cannot.
        match self.config().router_activation.as_deref() {
            Some(crate::config::moe_router::ROUTER_ACTIVATION_SIGMOID) => {
                crate::config::MoeRouterKind::Sigmoid
            }
            _ => crate::config::MoeRouterKind::default(),
        }
    }

    /// Router algorithm identifier written into `MoeConfig.router_type` in a
    /// vindex. Serialisation only — dispatch on [`Self::moe_router_kind`].
    fn moe_router_type(&self) -> &str {
        self.moe_router_kind().as_str()
    }

    /// Expert FFN gate weight key.
    fn expert_ffn_gate_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    /// Expert FFN up-projection weight key.
    fn expert_ffn_up_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    /// Expert FFN down-projection weight key.
    fn expert_ffn_down_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    /// Packed gate+up projection blocks key (all experts fused, MXFP4).
    fn packed_gate_up_blocks_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Packed gate+up projection scales key.
    fn packed_gate_up_scales_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Packed down projection blocks key.
    fn packed_down_blocks_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Packed down projection scales key.
    fn packed_down_scales_key(&self, _layer: usize) -> Option<String> {
        None
    }

    //
    // These three tensors exist per layer in the GPT-OSS checkpoint and were
    // silently dropped for the same reason the attention biases were: nothing
    // named them, so extraction never asked and the forward pass never
    // applied them. A key returned here is only half the obligation — the
    // backend has to consume it. See `docs/k3-funnel.md` §4.7.

    /// Router bias key, added to the router logits before top-k selection.
    /// `None` for routers without a bias term.
    fn moe_router_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Per-expert bias on the fused gate+up projection.
    /// Shape `[num_experts, 2 * moe_intermediate_size]`, interleaved on the
    /// last axis exactly as the weight rows are.
    fn packed_gate_up_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Per-expert bias on the expert down projection.
    /// Shape `[num_experts, hidden_size]`.
    fn packed_down_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// How an expert's gate/up projection is combined into the down
    /// projection's input. Defaults to the plain gated FFN every other
    /// MoE architecture in the support table uses.
    fn expert_gate_policy(&self) -> ExpertGatePolicy {
        // Deliberately NOT derived from `swiglu_limit`.
        //
        // A declared clamp says a bound exists; it does not say the layer
        // computes [`ExpertGatePolicy::ClampedGlu`], which is a specific
        // formula — `glu = g·sigmoid(alpha·g)`, `out = (u+1)·glu`, with
        // `alpha = 1.702` — transcribed from GPT-OSS's reference. GLM-5.3-
        // Flash and Inkling-Small both declare a `swiglu_limit` too, and
        // nothing on hand says they share that activation.
        //
        // **That caution was right, and GLM has now been read.** Its
        // reference clamps exactly as GPT-OSS does and then computes
        // `silu(g) * u`, not `(u+1) * g * sigmoid(alpha*g)` — which is
        // why [`ExpertGatePolicy::ClampedGated`] exists as its own
        // variant. Deriving either from `swiglu_limit` would have picked
        // the wrong one for one of the two families, at a measured
        // relative 31.7 on GLM's real expert bank.
        //
        // Resolving the policy from the bound alone would claim they do,
        // on the strength of one shared field name. That is the same
        // inference `layer_types` → Gated DeltaNet made, and it is wrong
        // for the same reason: a declared parameter is not evidence of the
        // operator that consumes it. An architecture that has been judged
        // against its own reference overrides this.
        //
        // `situ` IS read here, and the distinction is exactly the one the
        // paragraph above draws. `swiglu_limit` is a PARAMETER, and a
        // parameter names no operator. `hidden_act: "situ"` is the
        // operator's own NAME — the checkpoint's own module registers it
        // as `ACT2FN["situ"] = SituAndMul` — which is the standing `silu`
        // and `gelu_pytorch_tanh` already have in `HF_ACTIVATION_NAMES`,
        // and the standing `swiglu`/`geglu` already have in
        // `HF_GLU_NAMES`, where one word names a shape AND a
        // nonlinearity. Reading a declared name is not inferring an
        // operator from an adjacent value.
        match self.activation_declaration() {
            ActivationDeclaration::NamesGatePolicy(SITU_NAME) => self.situ_gate_policy(),
            _ => ExpertGatePolicy::Gated,
        }
    }

    /// SiTU-GLU's two softcaps, resolved from the config exactly once.
    ///
    /// `beta` goes through the reference's `beta or 1.0`
    /// (`_get_situ_activation_params`, `modeling_kimi_linear.py` L91),
    /// which is Python truthiness — absent, null AND `0.0` all resolve to
    /// `1.0`. Every consumer downstream therefore receives a `beta` it can
    /// divide by, and none of them repeats this rule.
    ///
    /// `linear_beta` has no such fallback in the reference (L80 branches
    /// on `is not None`), so absence is carried as `None` — the up branch
    /// untouched, a different function from an infinite bound — and a
    /// declared `0.0` is carried verbatim.
    fn situ_gate_policy(&self) -> ExpertGatePolicy {
        let config = self.config();
        let beta = match config.activation_situ_beta {
            Some(declared) if declared != 0.0 => declared as f32,
            _ => SITU_DEFAULT_BETA,
        };
        ExpertGatePolicy::SituGlu {
            beta,
            linear_beta: config.activation_situ_linear_beta.map(|v| v as f32),
        }
    }

    /// How the router's top-k weights are normalised.
    ///
    /// Read from `norm_topk_prob`, which is exactly what that field means:
    /// `true` renormalises the selected weights to sum to 1; `false` or absent
    /// keeps the raw softmax probabilities, which sum to less by however much
    /// mass the unselected experts hold.
    ///
    /// **This default reads the config on purpose.** Baking one routing order
    /// into code that serves both architectures is a rescale of the entire
    /// expert branch, and it has now been got wrong twice — `docs/k3-funnel.md`
    /// §4.7 finding 5, then again in `select_and_normalise` an hour after that
    /// finding was written up. A config-reading default makes a new MoE
    /// architecture correct on arrival instead of correct only if someone
    /// remembers to override; `QwenArch` shipped `norm_topk_prob: true` models
    /// and inherited the wrong order for exactly that reason.
    ///
    /// Override only where the order is fixed by the architecture rather than
    /// by config, as GPT-OSS's is.
    fn expert_routing_policy(&self) -> ExpertRoutingPolicy {
        if self.config().norm_topk_prob.unwrap_or(false) {
            ExpertRoutingPolicy::NormalisedOverSelected
        } else {
            ExpertRoutingPolicy::SoftmaxThenSelect
        }
    }

    /// Shared expert FFN gate weight key.
    fn shared_expert_gate_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Shared expert FFN up-projection weight key.
    fn shared_expert_up_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Shared expert FFN down-projection weight key.
    fn shared_expert_down_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// The always-on shared branch's intermediate width — `None` when the
    /// judgment declares no shared expert at all.
    ///
    /// Two conventions, one answer, so that no caller has to know the
    /// lineage to size the branch:
    ///
    /// - a family that DECLARES the width writes it
    ///   (`shared_expert_intermediate_size` on Qwen MoE,
    ///   `moe_shared_expert_intermediate_size` on Nemotron-H);
    /// - the DeepSeek/Kimi lineage declares a shared-expert COUNT instead
    ///   and sizes one wider FFN at `moe_intermediate_size * count`
    ///   (`KimiMLP`'s `intermediate_size` in `KimiSparseMoeBlock.__init__`).
    ///
    /// The declaration wins where both are present: it is the checkpoint
    /// speaking rather than this build's arithmetic, and the two disagree
    /// on every checkpoint that states both — Qwen1.5-MoE declares 5632
    /// against a routed 1408, Nemotron-3 Nano 3712 against 1856.
    fn shared_expert_intermediate_size(&self) -> Option<usize> {
        let count = self.num_shared_experts();
        if count == 0 {
            // `None` iff there is no branch. A width beside a zero count
            // would be a size for something nothing builds, and the two
            // fields would disagree about whether the branch exists.
            return None;
        }
        self.config()
            .shared_expert_intermediate_size
            .or_else(|| Some(self.moe_intermediate_size() * count))
    }

    /// How this component's residual stream is shaped and recombined.
    ///
    /// Resolved once, here, so that no caller has to decide what an
    /// absent `hc_mult` means. The Sinkhorn-split reference reads
    /// `hc_mult`, `hc_sinkhorn_iters` and `hc_eps` together, so a
    /// checkpoint declaring them apart is one this build has not judged
    /// rather than one to be completed with defaults — it REFUSES rather
    /// than filling in the missing halves.
    ///
    /// **It is not judged INCOMPLETE.** Hy4-preview declares `hc_mult`
    /// and `hc_eps` with no iteration count because its topology runs no
    /// Sinkhorn at all: its `hc_pre.hc_fn` is `[2 * hc, hc * d]` against
    /// the Sinkhorn form's `[(2 + hc) * hc, hc * d]`, it carries two
    /// scales rather than three, no combination block, and an explicit
    /// `hc_magnitude` where the Sinkhorn kernel hardcodes a factor of
    /// two. Recognising that variant needs positive evidence this build
    /// does not yet parse; until then the honest statement is that the
    /// combination is unjudged, not that it is half-written.
    ///
    /// The third judged topology, attention residuals, is read from ONE
    /// key — `attn_res_block_size` — because the reference takes only
    /// one: the snapshot schedule, every layer's read of the history and
    /// the stack's exit reduction all follow from the period. Declaring
    /// it BESIDE the hyper-connection keys is a third thing again, and
    /// refuses for the same reason a partial Sinkhorn declaration does:
    /// a component runs ONE residual programme, and reading either would
    /// discard what the other declares.
    fn residual_topology(&self) -> Result<ResidualTopology, String> {
        let cfg = self.config();
        let sinkhorn = (cfg.hc_streams, cfg.hc_sinkhorn_iters, cfg.hc_eps);
        match (cfg.attn_res_block_size, sinkhorn) {
            (Some(block_size), (None, None, None)) => {
                Ok(ResidualTopology::AttentionResidual { block_size })
            }
            (Some(block_size), (streams, iters, eps)) => Err(format!(
                "two residual topologies declared together (attn_res_block_size \
                 {block_size}, hc_mult {streams:?}, hc_sinkhorn_iters {iters:?}, hc_eps \
                 {eps:?}) — a component runs ONE residual programme, and reading either \
                 would discard what the other declares, so this build chooses neither"
            )),
            (None, (None, None, None)) => Ok(ResidualTopology::SingleStream),
            (None, (Some(streams), Some(sinkhorn_iters), Some(sinkhorn_eps))) => {
                Ok(ResidualTopology::HyperConnection(HyperConnection {
                    streams,
                    sinkhorn_iters,
                    sinkhorn_eps,
                }))
            }
            (None, (streams, iters, eps)) => Err(format!(
                "unjudged hyper-connection declaration (hc_mult {streams:?}, \
                 hc_sinkhorn_iters {iters:?}, hc_eps {eps:?}) — this build lowers only the \
                 Sinkhorn-split form, which reads all three together, and declaring them \
                 apart may mean a DIFFERENT topology rather than an incomplete one, so this \
                 build chooses neither"
            )),
        }
    }

    /// The gate on the shared branch's output, where the family runs one.
    ///
    /// `None` is the DeepSeek/Kimi form — the branch is summed unscaled —
    /// and it is a judgment, not an absence of information: adding a gate
    /// nothing declared, or dropping one that exists, both produce fluent
    /// wrong answers rather than a failure.
    fn shared_expert_branch_gate(&self) -> Option<SharedExpertGateSpec> {
        None
    }

    /// The `[1, hidden_size]` projection operand that
    /// [`Self::shared_expert_branch_gate`] reads. Paired with it: a family
    /// declaring the gate must name the operand, and one declaring no gate
    /// has none to name.
    fn shared_expert_branch_gate_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Whether this model has a hybrid dense-MLP + expert block per layer.
    /// Unlike pure MoE (Mixtral/DeepSeek), both branches run and their outputs are summed.
    fn is_hybrid_moe(&self) -> bool {
        false
    }

    /// Per-expert intermediate (hidden) dimension. 0 for non-MoE models.
    fn moe_intermediate_size(&self) -> usize {
        // The declared expert width. `0` was the old default and it is not
        // a width — an MoE surface carrying it states that each expert
        // projects to nothing, which no closure check can satisfy and no
        // reader can act on.
        self.config().moe_intermediate_size.unwrap_or(0)
    }

    /// Packed stacked gate+up projection key (Gemma 4 PackedBF16 format).
    /// Tensor shape: [num_experts, 2 * moe_intermediate_size, hidden_size].
    fn packed_experts_gate_up_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Packed stacked down projection key (Gemma 4 PackedBF16 format).
    /// Tensor shape: [num_experts, hidden_size, moe_intermediate_size].
    fn packed_experts_down_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Gemma 4 router learned input-scale key (`router.scale`).
    fn moe_router_scale_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Gemma 4 router per-expert output-scale key (`router.per_expert_scale`).
    fn moe_router_per_expert_scale_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Router's own RMS-norm weight, applied to the router's input *before*
    /// the router projection. HF Gemma 4's `Gemma4TextRouter.norm` is
    /// **parameter-free** (`with_scale=False`) so no tensor exists on disk —
    /// see [`moe_router_norm_parameter_free`](Self::moe_router_norm_parameter_free).
    /// This key is provided for architectures that DO ship a learned router
    /// norm weight. Return `None` to fall back to either parameter-free
    /// RMSNorm (when the flag is set) or the experts' pre-norm output.
    fn moe_router_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Whether the router applies a parameter-free RMSNorm to its input
    /// before `router_scale`/projection. Gemma 4 sets this true. When true
    /// AND `moe_router_norm_key` returns `None`, the forward pass runs
    /// `x / sqrt(mean(x²) + eps)` on the raw residual instead of reusing
    /// the experts' pre-norm (which would apply the wrong learned weight).
    fn moe_router_norm_parameter_free(&self) -> bool {
        false
    }

    /// Scalar multiplier applied to the router input after `router.norm` and
    /// after the learned `router.scale` vector. Gemma 4 uses `hidden_size^-0.5`
    /// (called `scalar_root_size` in HF). Return `None` for no scaling.
    fn moe_router_input_scalar(&self) -> Option<f32> {
        None
    }

    /// Outer post-FFN norm for hybrid MoE layers — applied to `(h1 + h2)`
    /// before the residual add, where `h1 = post_ffn_norm_1(dense)` and
    /// `h2 = post_ffn_norm_2(moe)`. HF Gemma 4 stores this at the un-suffixed
    /// key `post_feedforward_layernorm.weight`, while the dense-branch norm
    /// uses the suffixed `_1` variant (see `post_feedforward_layernorm_key`).
    /// Return `None` for architectures that either don't combine via an outer
    /// norm or reuse the dense-branch norm as the outer norm.
    fn moe_post_outer_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Post-FFN norm for dense MLP output in hybrid MoE layers.
    /// Gemma 4 A4B: `post_feedforward_layernorm_1.weight` (replaces the plain variant).
    fn moe_post_ffn1_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Pre-norm applied to the residual before feeding into the expert block.
    /// Gemma 4 A4B: `pre_feedforward_layernorm_2.weight`.
    fn moe_pre_experts_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Post-norm applied to the expert block output.
    /// Gemma 4 A4B: `post_feedforward_layernorm_2.weight`.
    fn moe_post_experts_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Whether the hybrid MoE forward applies a final RMS norm to the
    /// combined (dense + expert) output before adding to the residual.
    ///
    /// Gemma 4 26B A4B: true — matches HF `post_feedforward_layernorm(combined)`.
    /// All other models: false — use `layer_scalar * combined` instead.
    fn moe_has_combined_output_norm(&self) -> bool {
        false
    }
}
