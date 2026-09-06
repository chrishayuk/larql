//! Multi-Latent Attention, executed — transcribed directly from
//! `KimiMLAAttention.forward` in the checkpoint's own `modeling_kimi.py`,
//! not reused from [`super::super::mla::MlaOp`] (the encoded-operand
//! struct; container-facing, no math) or from
//! `format::weights::mla_absorb` (a DIFFERENT technique — DeepSeek-V3's
//! absorbed reformulation, which needs a `q_lora_rank` Kimi does not
//! have — proven identical to nothing yet, so not a substitute here).
//!
//! ```text
//! q_states          = q_proj(x)                                    # [Hq, nope+rope]
//! compressed_kv     = kv_a_proj_with_mqa(x)                        # [kv_lora_rank+rope]
//! k_pass, k_rot     = split(compressed_kv, [kv_lora_rank, rope])
//! k_pass            = kv_b_proj(kv_a_layernorm(k_pass))            # [Hq, nope+v_head_dim]
//! k_nope, v         = split(k_pass, [nope, v_head_dim])
//! key   = concat(k_nope, k_rot)         # k_rot is the SAME vector for every head (MQA)
//! query = concat(q_nope, q_rope)
//! scores            = query · key * (nope+rope)^-0.5
//! weights           = softmax(scores)                               # causal, over CACHED positions
//! attn_value        = weights · v
//! output            = o_proj(attn_value)
//! ```
//!
//! **`mla_use_nope=True` is asserted by the checkpoint's own `__init__`
//! and its `forward` never calls a rotary embedding on `q_rope`/`k_rot`
//! at all** — "RoPE" in `qk_rope_head_dim` is DeepSeek's inherited field
//! name for the SHAPE, not a claim this family rotates it.
//! [`Mutation::TreatSharedKRopeAsPositioned`] makes that a measured
//! property rather than a read comment: applying the crate's own
//! trusted [`super::kernels::rope_rotate`] to the shared K component
//! must move the output once real positions differ, proving the
//! reference's omission is load-bearing, not inert.
//!
//! **A single cached position cannot exercise this operator at all**:
//! softmax over one score is `1.0` regardless of its value, so
//! `attn_value` degenerates to the lone `v` no matter what `q`, `k_nope`
//! or `k_rot` are. Every test and fixture here therefore carries at
//! least two real positions — the minimum at which the attention math
//! is doing anything a bug could hide in.

use larql_models::config::{MlaGeometry, NormType};

use super::continuation::LatentKvRows;
use super::cpu::projector::WeightRows;
use super::kernels::{norm, rope_rotate, softmax};

/// One layer's five operands, f32 and row-major, plus the `kv_a_layernorm`
/// epsilon — `1e-6`, `KimiRMSNorm`'s class DEFAULT, deliberately NOT
/// `config.rms_norm_eps` (`1e-5`, what the layer's own two norms use):
/// `kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)` in the checkpoint's
/// `__init__` passes no `eps`, so it is the one norm in this whole layer
/// that does NOT read the config value. Carried here rather than assumed
/// equal to the layer eps, so that divergence stays visible instead of
/// silently coinciding.
#[derive(Clone, Copy)]
pub struct MlaWeights<'a> {
    /// How this layer builds its query.
    pub query: MlaQueryWeights<'a>,
    /// `[kv_lora_rank+rope, hidden]`.
    pub kv_a_proj: WeightRows<'a>,
    /// `[kv_lora_rank]` — RMSNorm weight over the latent ONLY, never the
    /// shared rope-K half of `compressed_kv`.
    pub kv_a_norm: &'a [f32],
    /// `[Hq·(nope+v_head_dim), kv_lora_rank]`.
    pub kv_b_proj: WeightRows<'a>,
    /// `[hidden, Hq·v_head_dim]`.
    pub o_proj: WeightRows<'a>,
    pub kv_a_norm_eps: f64,
    /// The output gate's projection, `[Hq·v_head_dim, hidden]`, when the
    /// layer declares one (Kimi-K3's `mla_use_output_gate`):
    /// `sigmoid(g_proj(x)) ⊙ attn_value` before `o_proj`, `x` the block's
    /// normalised input. `None` = no gate, the reference's own default.
    pub output_gate: Option<WeightRows<'a>>,
}

/// The per-position cache MLA actually carries: the COMPRESSED latent
/// plus the one shared rope-K, raw — never the decompressed
/// `Hq·(nope+v_head_dim)` a per-head K/V pair would cost. Decompression
/// happens at read time, for every cached position, on every call; that
/// is the operator's real cost profile, not an artefact of this
/// reference being naive.
///
/// Held in the engine's generic per-position row store
/// ([`LatentKvRows`]), not in a type of this operator's own: one row per
/// position, of the width [`MlaGeometry::compressed_kv_width`] declares,
/// is exactly what [`LayerLatentKvGeometry`](super::continuation::LayerLatentKvGeometry)
/// describes. A second state type here would be a second answer to
/// "what does an MLA layer retain", and the point of the continuation
/// schema is that there is one.
pub type MlaState = LatentKvRows;

/// Every boundary the operator crosses, for the CURRENT position only —
/// prior positions' boundaries were already returned by their own call.
/// Names match the pipeline in this module's own doc comment, so a
/// disagreement against the oracle names its own stage.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaTrace {
    /// The query leaving whichever form the layer declared, fused
    /// nope+rope per head, `[Hq·q_head_dim]`.
    ///
    /// Named `q_states` after the reference's own variable, because under
    /// the factorised form nothing computes a `q_proj` at all and the
    /// boundary would otherwise name an operand the layer does not have.
    pub q_states: Vec<f32>,
    /// `q_a_proj(x)`, `[rank]` — `None` under the direct form.
    pub q_a: Option<Vec<f32>>,
    /// `q_a_layernorm(q_a)`, `[rank]` — `None` under the direct form.
    ///
    /// Reported even when a mutation feeds `q_b` something else: this is
    /// the value the norm PRODUCED, and `Mutation::QbFedPreNorm` exists
    /// to prove the executor's next stage consumed it rather than
    /// recomputing something equivalent for display.
    pub q_a_normed: Option<Vec<f32>>,
    /// `q_b_proj(q_a_normed)`, `[Hq·q_head_dim]` — `None` under the
    /// direct form, and equal to [`Self::q_states`] under the low-rank
    /// one.
    pub q_b: Option<Vec<f32>>,
    /// `kv_a_proj_with_mqa(x)`, RAW, `[kv_lora_rank+rope]` — what gets
    /// cached.
    pub compressed_kv: Vec<f32>,
    /// `kv_a_layernorm` applied to the latent half only, `[kv_lora_rank]`
    /// (or the raw latent, under [`Mutation::OmitKvANorm`]).
    pub kv_a_normed: Vec<f32>,
    /// `kv_b_proj(kv_a_normed)`, fused nope-K+V per head,
    /// `[Hq·(nope+v_head_dim)]`.
    pub kv_b: Vec<f32>,
    /// Causal softmax weights, THIS query against every visible cached
    /// position, `[Hq·visible]`, head-major.
    pub attn_weights: Vec<f32>,
    /// Weighted sum of visible `v`, pre-`o_proj`, `[Hq·v_head_dim]`.
    pub attn_value: Vec<f32>,
    /// `sigmoid(g_proj(x))`, `[Hq·v_head_dim]` — `None` on an ungated
    /// layer. Under [`Mutation::SigmoidOmitted`] the raw pre-activation.
    pub output_gate: Option<Vec<f32>>,
    /// `attn_value ⊙ output_gate`, what `o_proj` actually consumed —
    /// `None` on an ungated layer, where `o_proj` reads `attn_value`.
    pub gated_value: Option<Vec<f32>>,
    /// `o_proj(gated_value)` (or `o_proj(attn_value)` ungated), `[hidden]`.
    pub output: Vec<f32>,
}

/// Deliberate defects, for the negative controls. Perturb the REAL
/// function, never a hand-rolled copy — same posture as `exec::kda`'s and
/// `exec::kimi_router`'s own `Mutation` enums.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mutation {
    None,
    /// Rotate the shared K-rope component at its cached position with the
    /// crate's own trusted RoPE kernel — the transform `mla_use_nope`
    /// asserts this family does NOT apply. Must move the output once ≥2
    /// real positions differ; a fixture where it does not is not
    /// exercising this operator (see this module's own doc comment).
    TreatSharedKRopeAsPositioned {
        theta: f64,
    },
    /// Skip `kv_a_layernorm` at EVERY cached position's decompression —
    /// feed the raw latent straight to `kv_b_proj`. Must move the
    /// output: the norm's own gain is not the identity at real weights.
    OmitKvANorm,
    /// The declared output gate not applied: `o_proj` reads `attn_value`.
    /// K3-REP-GATE-1; caught at `gated_value`.
    GateOmitted,
    /// The raw gate pre-activation multiplied in, no sigmoid. Caught at
    /// `output_gate`.
    SigmoidOmitted,
    /// `q_b_proj` fed the UN-normed `q_a`, while `q_a_layernorm`'s output
    /// is still computed and still reported on the trace.
    ///
    /// Output-identical to skipping the norm and trace-DIFFERENT, which
    /// is the entire point: it is the control for the TRACE, and the only
    /// thing that distinguishes "the boundary this executor reports is
    /// the value its next stage consumed" from "the boundary is a
    /// separately-computed display that happens to look right". Caught at
    /// `q_b`, with `q_a_normed` unmoved.
    ///
    /// Meaningless on a direct-form layer, which has no norm to bypass;
    /// `layer_forward` ignores it there, and the parity test asserts that
    /// rather than leaving it to be assumed.
    QbFedPreNorm,
}

/// The query's operands, in whichever form the layer declared.
///
/// `Direct` is Kimi Linear's one dense projection; `LowRank` is Kimi-K3's
/// `q_a_proj -> q_a_layernorm -> q_b_proj` under a declared
/// `q_lora_rank`. The two produce the same object and are consumed
/// identically from the head split onward — which is exactly why the form
/// is carried rather than deduced from the operands' shapes.
#[derive(Clone, Copy)]
pub enum MlaQueryWeights<'a> {
    Direct {
        /// `[Hq·q_head_dim, hidden]`.
        q_proj: WeightRows<'a>,
    },
    LowRank {
        /// `[rank, hidden]`.
        q_a_proj: WeightRows<'a>,
        /// RMSNorm weight over the query latent, `[rank]`.
        q_a_norm: &'a [f32],
        /// `[Hq·q_head_dim, rank]` — the SAME row count as `Direct`'s
        /// `q_proj`, a different column count.
        q_b_proj: WeightRows<'a>,
        /// `q_a_norm`'s epsilon. `KimiRMSNorm(self.q_lora_rank)` passes
        /// none, so it runs at the class default `1e-6` while the layer's
        /// own norms run at `rms_norm_eps` — the same property
        /// [`MlaWeights::kv_a_norm_eps`] carries, arrived at by the same
        /// cause and NOT by sharing that field's authority.
        q_a_norm_eps: f64,
    },
}

/// One token through Multi-Latent Attention. `x` is the ALREADY-NORMED
/// hidden state (the layer applies `input_layernorm` before calling
/// this, same contract `exec::kda::layer_forward` holds). Appends the
/// current position's compressed KV to `state` before reading it back,
/// so `state.len() - 1` after this call is this position's
/// causal index.
///
/// **Causality here is structural, not a runtime check** — deliberately
/// with no `Mutation` to disable it, unlike every other property this
/// module measures. `state` holds no position this call has not itself
/// appended, so there is no "attend to the future" defect this function
/// COULD express: reading `state.rows()` in full already IS the
/// causal set, by the append-then-read contract, not by a bound some
/// omitted check would have enforced. A batched whole-sequence
/// implementation would need its own explicit mask and its own control
/// for omitting it; this one does not, and claims nothing about that
/// other implementation.
pub fn mla_forward(
    x: &[f32],
    hidden: usize,
    weights: MlaWeights<'_>,
    geometry: MlaGeometry,
    state: &mut LatentKvRows,
    mutation: Mutation,
) -> MlaTrace {
    mla_forward_with(
        &super::cpu::physical::ExecutorProjections,
        x,
        hidden,
        weights,
        geometry,
        state,
        mutation,
    )
}

/// [`mla_forward`] with the four projections executed where the caller
/// says — the plan-driven path hands it the BACKEND's projector, so a
/// reference run computes them with the scalar oracle and a production
/// run with the executor's format-aware dispatch. Everything else about
/// this operator (the decompression, the softmax combine) is unchanged
/// f32 transcription either way.
#[allow(clippy::too_many_arguments)]
pub fn mla_forward_with(
    projections: &dyn super::gated_delta::DenseProjections,
    x: &[f32],
    hidden: usize,
    weights: MlaWeights<'_>,
    geometry: MlaGeometry,
    state: &mut LatentKvRows,
    mutation: Mutation,
) -> MlaTrace {
    // The four projections are ordinary linear algebra the caller's
    // infrastructure already does well; this operator's own math (the
    // decompression at read time, the softmax combine) stays a plain f32
    // transcription either way. Measured at P3d-n: MLA was the single
    // largest per-token cost (306.8 ms/tok, only 7 of 27 layers) while
    // still calling `exec::kernels::matvec` — the crate's DELIBERATELY
    // naive "Stage A oracle" reference, never meant as a production
    // path. Which projector runs them changes no arithmetic beyond
    // summation order, so the real-weight parity suite re-run is the
    // check that routing changed no semantics, not a new correctness
    // claim.
    let matvec = |w: WeightRows<'_>, x: &[f32], out: usize| projections.project(w, x, out);
    let heads = geometry.num_heads;
    let nope = geometry.qk_nope_head_dim;
    let rope = geometry.qk_rope_head_dim;
    let v_dim = geometry.v_head_dim;
    let q_head_dim = geometry.q_head_dim();
    let latent = geometry.kv_lora_rank;
    let scaling = (q_head_dim as f64).powf(-0.5) as f32;
    let omit_kv_a_norm = matches!(mutation, Mutation::OmitKvANorm);

    let (q_states, q_a, q_a_normed, q_b) = match weights.query {
        MlaQueryWeights::Direct { q_proj } => {
            (matvec(q_proj, x, heads * q_head_dim), None, None, None)
        }
        MlaQueryWeights::LowRank {
            q_a_proj,
            q_a_norm,
            q_b_proj,
            q_a_norm_eps,
        } => {
            // L419: q_b_proj(q_a_layernorm(q_a_proj(hidden_states))).
            let rank = q_a_norm.len();
            let q_a = matvec(q_a_proj, x, rank);
            let normed = norm(NormType::RmsNorm, &q_a, q_a_norm, 0.0, q_a_norm_eps);
            // The norm is reported whatever the mutation does; only its
            // CONSUMER changes, which is the whole of `QbFedPreNorm`.
            let into_b = match mutation {
                Mutation::QbFedPreNorm => &q_a,
                _ => &normed,
            };
            let q_b = matvec(q_b_proj, into_b, heads * q_head_dim);
            (q_b.clone(), Some(q_a), Some(normed), Some(q_b))
        }
    };
    let compressed_kv = matvec(weights.kv_a_proj, x, geometry.compressed_kv_width());

    state.append(compressed_kv.clone());
    let cur_pos = state.len() - 1;
    let visible = state.len(); // == cur_pos + 1: every cached position, never more

    // ── Read every visible position, decompressing at read time ──
    // This operator's real cost profile: nothing decompressed is ever
    // cached, so every call re-derives every prior position's k_nope/v
    // from its raw compressed latent.
    let mut cur_kv_a_normed = Vec::new();
    let mut cur_kv_b = Vec::new();
    let mut per_pos_kv_b = Vec::with_capacity(visible);
    let mut per_pos_k_rot = Vec::with_capacity(visible);
    for p in 0..visible {
        let entry = &state.rows()[p];
        let latent_input = if omit_kv_a_norm {
            entry[..latent].to_vec()
        } else {
            norm(
                NormType::RmsNorm,
                &entry[..latent],
                weights.kv_a_norm,
                0.0,
                weights.kv_a_norm_eps,
            )
        };
        let decompressed = matvec(weights.kv_b_proj, &latent_input, heads * (nope + v_dim));
        if p == cur_pos {
            cur_kv_a_normed = latent_input;
            cur_kv_b = decompressed.clone();
        }
        per_pos_kv_b.push(decompressed);

        let mut k_rot = entry[latent..].to_vec();
        if let Mutation::TreatSharedKRopeAsPositioned { theta } = mutation {
            rope_rotate(&mut k_rot, p, theta);
        }
        per_pos_k_rot.push(k_rot);
    }

    let mut attn_weights = vec![0.0f32; heads * visible];
    let mut attn_value = vec![0.0f32; heads * v_dim];
    for h in 0..heads {
        let q_nope = &q_states[h * q_head_dim..h * q_head_dim + nope];
        let q_rope = &q_states[h * q_head_dim + nope..h * q_head_dim + nope + rope];

        let mut scores = vec![0.0f32; visible];
        for (p, score) in scores.iter_mut().enumerate() {
            let head_kv = &per_pos_kv_b[p][h * (nope + v_dim)..h * (nope + v_dim) + nope];
            let k_rot = &per_pos_k_rot[p];
            let dot: f32 = q_nope.iter().zip(head_kv).map(|(a, b)| a * b).sum::<f32>()
                + q_rope.iter().zip(k_rot).map(|(a, b)| a * b).sum::<f32>();
            *score = dot * scaling;
        }
        softmax(&mut scores);
        attn_weights[h * visible..(h + 1) * visible].copy_from_slice(&scores);

        for (p, &w) in scores.iter().enumerate() {
            let v = &per_pos_kv_b[p][h * (nope + v_dim) + nope..(h + 1) * (nope + v_dim)];
            for (d, &vd) in v.iter().enumerate() {
                attn_value[h * v_dim + d] += w * vd;
            }
        }
    }

    // The output gate (K3-REP-GATE-1), read from the block INPUT `x` and
    // applied to the AGGREGATE — never to the per-position values, which
    // this executor decompresses from the latent cache at read time and
    // could not gate by their own positions' gates without retaining a
    // gate per cached position it never had.
    let (output_gate, gated_value) = match weights.output_gate {
        None => (None, None),
        Some(g_proj) => {
            let mut gate = matvec(g_proj, x, heads * v_dim);
            if mutation != Mutation::SigmoidOmitted {
                gate.iter_mut().for_each(|g| *g = 1.0 / (1.0 + (-*g).exp()));
            }
            let gated: Vec<f32> = if mutation == Mutation::GateOmitted {
                attn_value.clone()
            } else {
                attn_value.iter().zip(&gate).map(|(a, g)| a * g).collect()
            };
            (Some(gate), Some(gated))
        }
    };
    let output = matvec(
        weights.o_proj,
        gated_value.as_deref().unwrap_or(&attn_value),
        hidden,
    );

    MlaTrace {
        q_states,
        q_a,
        q_a_normed,
        q_b,
        compressed_kv,
        kv_a_normed: cur_kv_a_normed,
        kv_b: cur_kv_b,
        attn_weights,
        attn_value,
        output_gate,
        gated_value,
        output,
    }
}
