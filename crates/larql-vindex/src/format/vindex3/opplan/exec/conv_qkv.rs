//! Reference conv-QKV attention: the hybrid block, written to be read.
//!
//! Deliberately slow and literal, the same posture as [`super::mamba2`]:
//! this exists so someone can put it beside `Mamba2Attention.forward` in
//! `modeling_mamba2attn.py` and compare it stage by stage. Speed is a
//! later rung's problem.
//!
//! The operator is fully defined by its persisted semantics — the
//! [`ConvQkvOp`]'s geometry and operands. Nothing here knows a family
//! name: fused projection width, conv kernel, rotary width and base, GQA
//! head counts and bias presence all come from the op, so a container is
//! sufficient to execute this layer with no source lookup — the deletion
//! invariant, held per operator.
//!
//! Precision is a judgment, not a default: the reference upcasts
//! attention scores to fp32 for the softmax (`attn_weights ...
//! dtype=torch.float32`), and this executor computes every stage in f32
//! over widened operands — so the f32 conv history is a transcription of
//! the reference's own compute precision, recorded at
//! [`conv_history_geometry`].

// Explicit index loops on purpose, matching the reference's named axes —
// verification against `Mamba2Attention.forward` is the point of the file.
#![allow(clippy::needless_range_loop)]

use super::super::conv_qkv::ConvQkvOp;
use super::continuation::{
    RecurrentBufferGeometry, RecurrentGeometry, RecurrentState, StateInitialization,
};
use super::cpu::WeightRows;
use super::gated_delta::DenseProjections;
use super::timing::{timed, OpClass};

/// Buffer indices this operator assigns. One buffer: the conv history.
/// (The K/V rows live on the provider's KV side — the layer's OTHER
/// continuation region.)
pub const CONV_HISTORY: usize = 0;

/// The fixed half of this operator's continuation, in the engine's
/// generic terms: the causal conv's history over the FULL fused QKV —
/// the last `conv_kernel` positions of the PRE-conv projection, exactly
/// the window the reference cache keeps (`conv_states[layer]`, full
/// kernel width, left-padded). The KV half is declared beside it by the
/// continuation planner; this function owns only the buffer shape.
pub fn conv_history_geometry(op: &ConvQkvOp) -> RecurrentGeometry {
    RecurrentGeometry::single(RecurrentBufferGeometry {
        shape: vec![op.geometry.qkv_rows(), op.geometry.conv_kernel],
        dtype: super::super::gated_delta::StateDtype::Float32,
        initialization: StateInitialization::Zeros,
    })
}

/// The four operands in their resident representations, resolved by the
/// caller through the `ConvQkvOp` — the single architecture authority.
pub struct ConvQkvWeights<'a> {
    /// The two DENSE projections, in whatever representation they are
    /// resident as.
    pub in_proj: WeightRows<'a>,
    pub out_proj: WeightRows<'a>,
    /// The depthwise convolution taps, always f32: `[qkv_rows · kernel]`.
    pub conv1d: &'a [f32],
    /// `[qkv_rows]` — present iff `use_conv_bias`.
    pub conv1d_bias: Option<&'a [f32]>,
}

/// Every boundary the operator crosses, kept so a disagreement names its
/// own stage instead of being debugged backwards from the layer output.
#[derive(Debug, Default)]
pub struct ConvQkvPlanes {
    /// Post-conv fused QKV (NO activation follows the conv): `[T][qkv_rows]`.
    pub conv: Vec<Vec<f32>>,
    /// Concatenated attention output, pre-`out_proj`: `[T][Hq·Dh]`.
    pub attn: Vec<Vec<f32>>,
    /// `[T][hidden]`.
    pub output: Vec<Vec<f32>>,
    /// This batch's K rows, post-conv post-rotary: `[T][Hkv·Dh]`. The
    /// caller appends them to the provider — the executor computes, the
    /// provider persists, and the split keeps this function ignorant of
    /// how rows are stored.
    pub keys: Vec<Vec<f32>>,
    /// This batch's V rows: `[T][Hkv·Dh]`.
    pub values: Vec<Vec<f32>>,
}

/// The whole operator: hidden states in, layer output out, conv history
/// advanced. `past_keys`/`past_values` are the rows already persisted
/// for this layer (empty at sequence start); `base` is the absolute
/// position of `hidden[0]` — the rotary angle is a function of absolute
/// position, so a decode step at position 40 must not rotate like a
/// prefill at position 0.
///
/// Stage order is the specification, from `Mamba2Attention.forward`:
///
/// ```text
/// in_proj → fused qkv                    (bias iff use_attention_qkv_bias)
/// causal conv over the FULL qkv, + bias  (NO activation — unlike the mixer)
/// split q | k | v                        (head_dim · Hq | · Hkv | · Hkv)
/// partial rotary on the leading rotary_dim dims of q and k
///   (rotate-half, inv_freq over the rotary width, absolute positions)
/// append k, v to the cache; GQA: q head h reads kv head h / (Hq/Hkv)
/// scores = q·k / √head_dim over the causal prefix, softmax in f32
/// out = scores · v, heads concatenated → out_proj
/// ```
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_with(
    op: &ConvQkvOp,
    w: &ConvQkvWeights<'_>,
    hidden: &[Vec<f32>],
    state: &mut RecurrentState,
    past_keys: &[Vec<f32>],
    past_values: &[Vec<f32>],
    base: usize,
    proj: &dyn DenseProjections,
) -> ConvQkvPlanes {
    let g = op.geometry;
    let (heads, kv_heads, head_dim) = (g.num_heads, g.num_kv_heads, g.head_dim);
    let qkv_rows = g.qkv_rows();
    let kernel = g.conv_kernel;
    let rotary = g.rotary_dim;
    let q_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let heads_per_kv = heads / kv_heads;
    let t_len = hidden.len();
    let mut planes = ConvQkvPlanes::default();
    let _site = super::cpu::ledger::in_site(super::cpu::ledger::Site::Attention);

    // Stage 1: the fused projection, for every position. The declared
    // bias switches are `false` on the judged checkpoint; a `true` was
    // blocked at admission (no bias role is judged yet), so no bias is
    // added here — by declaration, not omission.
    let rows: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
    let mixed: Vec<Vec<f32>> = proj.project_many(w.in_proj, &rows, qkv_rows);

    // Stage 2: depthwise causal convolution over the FULL fused QKV,
    // then the bias — and nothing else: the reference applies no
    // activation here, where the mixer's conv applies SiLU. The window
    // reaches back `kernel-1` positions, which may lie before this
    // batch: the durable history answers there, and its zeros are
    // correct at sequence start because the buffer was initialised to
    // zeros.
    let history_len = kernel - 1;
    let history: Vec<f32> = state.buffer(CONV_HISTORY).cells().to_vec();
    let past = |c: usize, back: usize| -> f32 {
        // `back` = 1 means the position immediately before the batch.
        let slot = kernel - back;
        history[c * kernel + slot]
    };
    let mut conv: Vec<Vec<f32>> = vec![vec![0.0; qkv_rows]; t_len];
    let convolution = timed(OpClass::DeltaConv);
    for c in 0..qkv_rows {
        let taps = &w.conv1d[c * kernel..(c + 1) * kernel];
        let bias = w.conv1d_bias.map_or(0.0, |b| b[c]);
        for t in 0..t_len {
            let mut acc = 0.0f32;
            for (i, tap) in taps.iter().enumerate() {
                let offset = t as isize - (kernel as isize - 1) + i as isize;
                if offset < 0 {
                    let back = (-offset) as usize;
                    if back <= history_len {
                        acc += tap * past(c, back);
                    }
                    continue;
                }
                if (offset as usize) < t_len {
                    acc += tap * mixed[offset as usize][c];
                }
            }
            conv[t][c] = acc + bias;
        }
    }
    drop(convolution);

    // Stage 3: split q|k|v and rotate the leading `rotary_dim` dims of
    // each q and k head — rotate-half convention, frequencies over the
    // ROTARY width (`theta^(-2i/rotary_dim)`), at ABSOLUTE positions.
    let half = rotary / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (g.rope_theta as f32).powf(-(2.0 * i as f32) / rotary as f32))
        .collect();
    let rotate = |head: &mut [f32], position: usize| {
        for (i, freq) in inv_freq.iter().enumerate() {
            let angle = position as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let a = head[i];
            let b = head[half + i];
            head[i] = a * cos - b * sin;
            head[half + i] = b * cos + a * sin;
        }
    };
    let gates = timed(OpClass::DeltaGates);
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(t_len);
    for t in 0..t_len {
        let position = base + t;
        let mut q = conv[t][..q_width].to_vec();
        let mut k = conv[t][q_width..q_width + kv_width].to_vec();
        let v = conv[t][q_width + kv_width..].to_vec();
        for h in 0..heads {
            rotate(&mut q[h * head_dim..h * head_dim + rotary], position);
        }
        for h in 0..kv_heads {
            rotate(&mut k[h * head_dim..h * head_dim + rotary], position);
        }
        queries.push(q);
        planes.keys.push(k);
        planes.values.push(v);
    }
    drop(gates);

    // Stage 4: causal softmax attention over the persisted prefix plus
    // this batch, scores in f32 at 1/√head_dim, GQA by head grouping.
    let attention = timed(OpClass::DeltaRecurrence);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let key_at = |index: usize| -> &[f32] {
        if index < past_keys.len() {
            &past_keys[index]
        } else {
            &planes.keys[index - past_keys.len()]
        }
    };
    let value_at = |index: usize| -> &[f32] {
        if index < past_values.len() {
            &past_values[index]
        } else {
            &planes.values[index - past_values.len()]
        }
    };
    for t in 0..t_len {
        let visible = past_keys.len() + t + 1;
        let q = &queries[t];
        let mut out = vec![0.0f32; q_width];
        for h in 0..heads {
            let kv_head = h / heads_per_kv;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0.0f32; visible];
            let mut max = f32::NEG_INFINITY;
            for (index, score) in scores.iter_mut().enumerate() {
                let k_row = key_at(index);
                let k_head = &k_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                let dot: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
                *score = dot * scale;
                max = max.max(*score);
            }
            let mut denom = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                denom += *score;
            }
            let out_head = &mut out[h * head_dim..(h + 1) * head_dim];
            for (index, score) in scores.iter().enumerate() {
                let weight = score / denom;
                let v_row = value_at(index);
                let v_head = &v_row[kv_head * head_dim..(kv_head + 1) * head_dim];
                for d in 0..head_dim {
                    out_head[d] += weight * v_head[d];
                }
            }
        }
        planes.attn.push(out);
        planes.conv.push(conv[t].clone());
    }
    drop(attention);

    // Stage 5: one traversal of `out_proj` for every position.
    if let Some(width) = hidden.first().map(Vec::len) {
        let attn_rows: Vec<&[f32]> = planes.attn.iter().map(Vec::as_slice).collect();
        planes.output = proj.project_many(w.out_proj, &attn_rows, width);
    }

    // Roll the convolution history forward: the last `kernel` positions
    // of the PRE-convolution fused QKV, oldest first — the same window
    // the reference cache keeps.
    {
        let history = state.buffer_mut(CONV_HISTORY);
        let cells = history.cells_mut();
        for c in 0..qkv_rows {
            for slot in 0..kernel {
                let back = kernel - 1 - slot;
                let value = if back < t_len {
                    mixed[t_len - 1 - back][c]
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
