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

use super::cpu::projector::{DenseProjector, WeightRows};
use super::kernels::{norm, rope_rotate, softmax};

/// Routed to the crate's existing BLAS projector, same reasoning
/// `exec::kda`'s own local `matvec` states: the projections are ordinary
/// linear algebra infrastructure already does well, and this operator's
/// own math (the decompression-at-read-time recurrence, the softmax
/// combine) stays a plain f32 transcription. Measured at P3d-n: MLA was
/// the single largest per-token cost (306.8 ms/tok, only 7 of 27
/// layers) while still calling `exec::kernels::matvec` — the crate's
/// DELIBERATELY naive "Stage A oracle" reference (see that module's own
/// doc comment), never meant as a production path. Swapping only this
/// function changes no arithmetic — `BlasF32` and the scalar loop agree
/// up to summation-order float noise — so the full real-weight parity
/// suite re-run is the check that acceleration changed no semantics,
/// not a new correctness claim.
fn matvec(w: &[f32], x: &[f32], out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out];
    super::cpu::kernels::BlasF32.project_rows(WeightRows::F32(w), x, &mut y);
    y
}

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
    /// `[Hq·q_head_dim, hidden]`.
    pub q_proj: &'a [f32],
    /// `[kv_lora_rank+rope, hidden]`.
    pub kv_a_proj: &'a [f32],
    /// `[kv_lora_rank]` — RMSNorm weight over the latent ONLY, never the
    /// shared rope-K half of `compressed_kv`.
    pub kv_a_norm: &'a [f32],
    /// `[Hq·(nope+v_head_dim), kv_lora_rank]`.
    pub kv_b_proj: &'a [f32],
    /// `[hidden, Hq·v_head_dim]`.
    pub o_proj: &'a [f32],
    pub kv_a_norm_eps: f64,
}

/// The per-position KV cache MLA actually carries: the COMPRESSED latent
/// plus the one shared rope-K, raw — never the decompressed
/// `Hq·(nope+v_head_dim)` a per-head K/V pair would cost. Decompression
/// happens at read time, for every cached position, on every call; that
/// is the operator's real cost profile, not an artefact of this
/// reference being naive.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MlaState {
    /// One entry per cached position, each `kv_lora_rank+rope` long.
    pub compressed_kv: Vec<Vec<f32>>,
}

impl MlaState {
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Every boundary the operator crosses, for the CURRENT position only —
/// prior positions' boundaries were already returned by their own call.
/// Names match the pipeline in this module's own doc comment, so a
/// disagreement against the oracle names its own stage.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaTrace {
    /// `q_proj(x)`, fused nope+rope per head, `[Hq·q_head_dim]`.
    pub q_proj: Vec<f32>,
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
    /// `o_proj(attn_value)`, `[hidden]`.
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
}

/// One token through Multi-Latent Attention. `x` is the ALREADY-NORMED
/// hidden state (the layer applies `input_layernorm` before calling
/// this, same contract `exec::kda::layer_forward` holds). Appends the
/// current position's compressed KV to `state` before reading it back,
/// so `state.compressed_kv.len() - 1` after this call is this position's
/// causal index.
///
/// **Causality here is structural, not a runtime check** — deliberately
/// with no `Mutation` to disable it, unlike every other property this
/// module measures. `state` holds no position this call has not itself
/// appended, so there is no "attend to the future" defect this function
/// COULD express: reading `state.compressed_kv` in full already IS the
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
    state: &mut MlaState,
    mutation: Mutation,
) -> MlaTrace {
    let heads = geometry.num_heads;
    let nope = geometry.qk_nope_head_dim;
    let rope = geometry.qk_rope_head_dim;
    let v_dim = geometry.v_head_dim;
    let q_head_dim = geometry.q_head_dim();
    let latent = geometry.kv_lora_rank;
    let scaling = (q_head_dim as f64).powf(-0.5) as f32;
    let omit_kv_a_norm = matches!(mutation, Mutation::OmitKvANorm);

    let q_proj = matvec(weights.q_proj, x, heads * q_head_dim);
    let compressed_kv = matvec(weights.kv_a_proj, x, geometry.compressed_kv_width());

    state.compressed_kv.push(compressed_kv.clone());
    let cur_pos = state.compressed_kv.len() - 1;
    let visible = state.compressed_kv.len(); // == cur_pos + 1: every cached position, never more

    // ── Read every visible position, decompressing at read time ──
    // This operator's real cost profile: nothing decompressed is ever
    // cached, so every call re-derives every prior position's k_nope/v
    // from its raw compressed latent.
    let mut cur_kv_a_normed = Vec::new();
    let mut cur_kv_b = Vec::new();
    let mut per_pos_kv_b = Vec::with_capacity(visible);
    let mut per_pos_k_rot = Vec::with_capacity(visible);
    for p in 0..visible {
        let entry = &state.compressed_kv[p];
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
        let q_nope = &q_proj[h * q_head_dim..h * q_head_dim + nope];
        let q_rope = &q_proj[h * q_head_dim + nope..h * q_head_dim + nope + rope];

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

    let output = matvec(weights.o_proj, &attn_value, hidden);

    MlaTrace {
        q_proj,
        compressed_kv,
        kv_a_normed: cur_kv_a_normed,
        kv_b: cur_kv_b,
        attn_weights,
        attn_value,
        output,
    }
}
