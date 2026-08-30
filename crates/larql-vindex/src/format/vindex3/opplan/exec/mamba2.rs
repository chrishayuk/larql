//! Reference Mamba2/SSD: the recurrence, written to be read.
//!
//! Deliberately slow and literal, the same posture as
//! [`super::gated_delta`]: this exists so someone can put it beside
//! `Mamba2Mixer.torch_forward` in `transformers` and compare it stage by
//! stage. Speed is a later rung's problem.
//!
//! One deliberate divergence from the reference's PREFILL path, judged
//! and bounded rather than hidden: HF prefills through the chunked SSD
//! scan and decodes through the per-position recurrence — two
//! evaluations of the same operator that differ only in fp accumulation
//! order. This module runs the recurrence for both, because the
//! recurrence IS the state semantics (the scan is an optimisation of
//! it), and the banked oracle's own self-check measured the two at
//! ≤ 2.1e-4 max-abs with argmax agreement on every probe prompt. The
//! parity gate scores against that bound, not against bit-identity with
//! a reassociated scan.
//!
//! Precision is a judgment, not a default: the reference's own naive
//! path casts x, B and C to fp32 and computes the scan and the gated
//! norm in fp32 whatever the model's bulk dtype — so fp32 state here is
//! a transcription of the reference, and the one place that judgment
//! lives is [`state_geometry`].

// Explicit index loops on purpose, matching the reference's named axes —
// verification against `torch_forward` is the point of the file.
#![allow(clippy::needless_range_loop)]

use larql_models::config::DtBound;

use super::super::Mamba2Op;
use super::continuation::{
    RecurrentBufferGeometry, RecurrentGeometry, RecurrentState, StateInitialization,
};
use super::cpu::WeightRows;
use super::gated_delta::DenseProjections;
use super::timing::{timed, OpClass};

/// Buffer indices this operator assigns.
pub const SSM_STATE: usize = 0;
pub const CONV_HISTORY: usize = 1;

/// This operator's state, in the engine's generic terms: one
/// `head_dim × state_size` matrix per head, plus the causal conv's
/// history over the x|B|C channels.
///
/// Both buffers are fp32 **by judgment, not default**: the checkpoint
/// declares no state dtype for this family, and the reference
/// implementation's own naive path computes the scan in fp32 (explicit
/// `.float()` casts in `torch_forward`) — the state is held at the
/// precision the reference computes at. `conv_dim` is read from the conv
/// operand's closure-verified shape, never re-derived from a hidden size
/// this function was not given.
pub fn state_geometry(op: &Mamba2Op) -> RecurrentGeometry {
    let conv_dim = op.conv1d.shape[0];
    RecurrentGeometry {
        buffers: vec![
            RecurrentBufferGeometry {
                shape: vec![
                    op.geometry.num_heads,
                    op.geometry.head_dim,
                    op.geometry.state_size,
                ],
                dtype: super::super::gated_delta::StateDtype::Float32,
                initialization: StateInitialization::Zeros,
            },
            RecurrentBufferGeometry {
                shape: vec![conv_dim, op.geometry.conv_kernel],
                dtype: super::super::gated_delta::StateDtype::Float32,
                initialization: StateInitialization::Zeros,
            },
        ],
    }
}

/// The nine operands in their resident representations, resolved by the
/// caller through the `Mamba2Op` — the single architecture authority.
pub struct Mamba2Weights<'a> {
    /// The two DENSE projections, in whatever representation they are
    /// resident as.
    pub in_proj: WeightRows<'a>,
    pub out_proj: WeightRows<'a>,
    /// The elementwise glue and the depthwise convolution, always f32.
    pub conv1d: &'a [f32],
    /// `[conv_dim]` — present iff `use_conv_bias`.
    pub conv1d_bias: Option<&'a [f32]>,
    pub a_log: &'a [f32],
    pub d: &'a [f32],
    pub dt_bias: &'a [f32],
    /// The gated RMSNorm's weight `[d_inner]` — present iff `rms_norm`.
    pub norm: Option<&'a [f32]>,
    pub norm_eps: f32,
}

/// Every boundary the operator crosses, kept so a disagreement names its
/// own stage instead of being debugged backwards from the layer output.
#[derive(Debug, Default)]
pub struct Mamba2Planes {
    /// Post-conv, post-SiLU x|B|C: `[T][conv_dim]`.
    pub conv: Vec<Vec<f32>>,
    /// Discretised per-head timestep, post-softplus, post-clamp: `[T][heads]`.
    pub dt: Vec<Vec<f32>>,
    /// The recurrence's own output, pre-norm: `[T][d_inner]`.
    pub core: Vec<Vec<f32>>,
    /// `[T][hidden]`.
    pub output: Vec<Vec<f32>>,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    // The numerically stable form; large x must not overflow exp.
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Apply the declared `time_step_limit` clamp. An unbounded side clamps
/// nothing — that is the declaration, not a missing case.
fn clamp_dt(dt: f32, min: DtBound, max: DtBound) -> f32 {
    let dt = match min {
        DtBound::Finite(v) => dt.max(v as f32),
        DtBound::Unbounded => dt,
    };
    match max {
        DtBound::Finite(v) => dt.min(v as f32),
        DtBound::Unbounded => dt,
    }
}

/// The whole operator: hidden states in, layer output out, state advanced.
///
/// Stage order is the specification, from `torch_forward`:
///
/// ```text
/// in_proj → split z | xBC | dt          (z first — the gate)
/// causal conv over xBC, + bias, SiLU    (the gate is NOT convolved)
/// split x | B | C
/// dt = clamp(softplus(dt + dt_bias))    per-head scalar
/// per position, per head h (group g = h / (heads/groups)):
///   S = S · exp(dt·A) + dt · x ⊗ B[g]   decay, then rank-1 write
///   y = S · C[g] + D · x                read AFTER the write
/// y = norm(y · silu(z))                 gate first, then normalise
/// out_proj
/// ```
///
/// The read-after-write ordering is why a single-position test cannot
/// validate this: the current position reads state it has just written.
pub fn layer_forward_with(
    op: &Mamba2Op,
    w: &Mamba2Weights<'_>,
    hidden: &[Vec<f32>],
    state: &mut RecurrentState,
    proj: &dyn DenseProjections,
) -> Mamba2Planes {
    let g = op.geometry;
    let (heads, head_dim, n_state) = (g.num_heads, g.head_dim, g.state_size);
    let d_inner = heads * head_dim;
    let conv_dim = w.conv1d.len() / g.conv_kernel;
    let group_width = g.n_groups * n_state;
    let heads_per_group = heads / g.n_groups;
    let kernel = g.conv_kernel;
    let in_rows = 2 * d_inner + 2 * group_width + heads;
    let t_len = hidden.len();
    let mut planes = Mamba2Planes::default();
    let _site = super::cpu::ledger::in_site(super::cpu::ledger::Site::Recurrent);

    // Stage 1: the fused projection, for every position. `hidden` is the
    // layer input, so every position's operand exists before the
    // recurrence has run.
    let rows: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
    let mixed: Vec<Vec<f32>> = proj.project_many(w.in_proj, &rows, in_rows);
    // Split offsets: z | xBC | dt.
    let xbc_at = d_inner;
    let dt_at = d_inner + conv_dim;

    // Stage 2: depthwise causal convolution over the x|B|C channels ONLY
    // — the gate is never convolved — then the bias, then SiLU. The
    // window reaches back `kernel-1` positions, which may lie before this
    // batch: the durable history answers there, exactly as the reference
    // cache does, and its zeros are correct at sequence start because the
    // buffer was initialised to zeros.
    let history_len = kernel - 1;
    let history: Vec<f32> = state.buffer(CONV_HISTORY).cells().to_vec();
    let past = |c: usize, back: usize| -> f32 {
        // `back` = 1 means the position immediately before the batch.
        let slot = kernel - back;
        history[c * kernel + slot]
    };
    let mut conv: Vec<Vec<f32>> = vec![vec![0.0; conv_dim]; t_len];
    let convolution = timed(OpClass::DeltaConv);
    for c in 0..conv_dim {
        let taps = &w.conv1d[c * kernel..(c + 1) * kernel];
        let bias = w.conv1d_bias.map_or(0.0, |b| b[c]);
        for t in 0..t_len {
            let mut acc = 0.0f32;
            for (i, tap) in taps.iter().enumerate() {
                // Causal: left-padded by kernel-1, so tap i reads
                // position t - (kernel-1) + i.
                let offset = t as isize - (kernel as isize - 1) + i as isize;
                if offset < 0 {
                    let back = (-offset) as usize;
                    if back <= history_len {
                        acc += tap * past(c, back);
                    }
                    continue;
                }
                if (offset as usize) < t_len {
                    acc += tap * mixed[offset as usize][xbc_at + c];
                }
            }
            conv[t][c] = silu(acc + bias);
        }
    }
    drop(convolution);

    // `out_proj` is downstream of the recurrence; collected and projected
    // after the loop, one weight traversal for every position.
    let mut gated: Vec<Vec<f32>> = Vec::with_capacity(t_len);

    for t in 0..t_len {
        // Stage 3: split the convolved plane, and discretise dt.
        let x = &conv[t][..d_inner];
        let b_plane = &conv[t][d_inner..d_inner + group_width];
        let c_plane = &conv[t][d_inner + group_width..d_inner + 2 * group_width];
        let gates = timed(OpClass::DeltaGates);
        let dt: Vec<f32> = (0..heads)
            .map(|h| {
                clamp_dt(
                    softplus(mixed[t][dt_at + h] + w.dt_bias[h]),
                    g.dt_limit_min,
                    g.dt_limit_max,
                )
            })
            .collect();
        drop(gates);

        // Stage 4: the recurrence — decay, rank-1 write, read AFTER.
        let recurrence = timed(OpClass::DeltaRecurrence);
        let mut core = vec![0.0f32; d_inner];
        let cells = state.buffer_mut(SSM_STATE).cells_mut();
        for h in 0..heads {
            let group = h / heads_per_group;
            let b_row = &b_plane[group * n_state..(group + 1) * n_state];
            let c_row = &c_plane[group * n_state..(group + 1) * n_state];
            let a = -w.a_log[h].exp();
            let decay = (dt[h] * a).exp();
            let head = &mut cells[h * head_dim * n_state..(h + 1) * head_dim * n_state];
            for d in 0..head_dim {
                let xd = x[h * head_dim + d];
                let row = &mut head[d * n_state..(d + 1) * n_state];
                let mut y = 0.0f32;
                for n in 0..n_state {
                    row[n] = row[n] * decay + dt[h] * b_row[n] * xd;
                    y += row[n] * c_row[n];
                }
                core[h * head_dim + d] = y + w.d[h] * xd;
            }
        }
        drop(recurrence);

        // Stage 5: the gated norm — gate FIRST, then normalise over the
        // FULL inner width (the reference's `MambaRMSNormGated`: variance
        // over the whole last dimension, unlike DeltaNet's per-head
        // norm), then the weight. `rms_norm: false` declares no norm; the
        // gate still applies.
        let gated_norm = timed(OpClass::DeltaGatedNorm);
        let mut normed = vec![0.0f32; d_inner];
        for i in 0..d_inner {
            normed[i] = core[i] * silu(mixed[t][i]);
        }
        if let Some(weight) = w.norm {
            let var: f32 = normed.iter().map(|v| v * v).sum::<f32>() / d_inner as f32;
            let inv = 1.0 / (var + w.norm_eps).sqrt();
            for i in 0..d_inner {
                normed[i] = weight[i] * normed[i] * inv;
            }
        }
        drop(gated_norm);

        gated.push(normed);
        planes.conv.push(conv[t].clone());
        planes.dt.push(dt);
        planes.core.push(core);
    }

    // Stage 6: one traversal of `out_proj` for every position.
    if let Some(width) = hidden.first().map(Vec::len) {
        let gated_rows: Vec<&[f32]> = gated.iter().map(Vec::as_slice).collect();
        planes.output = proj.project_many(w.out_proj, &gated_rows, width);
    }

    // Roll the convolution history forward: the last `kernel` positions
    // of the PRE-convolution x|B|C plane, oldest first — the same window
    // the reference cache keeps.
    {
        let history = state.buffer_mut(CONV_HISTORY);
        let cells = history.cells_mut();
        for c in 0..conv_dim {
            for slot in 0..kernel {
                let back = kernel - 1 - slot;
                let value = if back < t_len {
                    mixed[t_len - 1 - back][xbc_at + c]
                } else {
                    let older = back - t_len;
                    if older < history_len {
                        past(c, older + 1)
                    } else {
                        0.0
                    }
                };
                cells[c * kernel + slot] = value;
            }
        }
    }
    planes
}
