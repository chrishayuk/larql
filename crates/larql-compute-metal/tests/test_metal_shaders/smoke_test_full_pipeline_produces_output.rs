//! Smoke test: full pipeline produces output

use super::*;

const HIDDEN: usize = 256;
// Deliberately != HIDDEN: the down projection's input is `INTER` wide, so a
// down matvec fed a `HIDDEN`-wide operand is wrong *and* reads past it.
const INTER: usize = 512;
const HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const Q_DIM: usize = HEADS * HEAD_DIM;

/// Q4_0 weights for one layer, every matrix filled with a constant.
struct Q4_0Weights {
    wq: Vec<u8>,
    wk: Vec<u8>,
    wv: Vec<u8>,
    wo: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

impl Q4_0Weights {
    fn constant(wo: f32, gate: f32, up: f32) -> Self {
        let q = |v: f32, n: usize| quantize_q4_0(&vec![v; n]);
        Self {
            wq: q(0.01, Q_DIM * HIDDEN),
            wk: q(0.01, Q_DIM * HIDDEN),
            wv: q(0.01, Q_DIM * HIDDEN),
            wo: q(wo, HIDDEN * Q_DIM),
            gate: q(gate, INTER * HIDDEN),
            up: q(up, INTER * HIDDEN),
            down: q(0.01, HIDDEN * INTER),
        }
    }
}

fn layer<'a>(
    w: &'a Q4_0Weights,
    norm: &'a [f32],
    ffn_type: larql_compute::FfnType,
) -> larql_compute::FullPipelineLayer<'a> {
    // Q4_0 packs its f16 scale inside each 18-byte block, so no external
    // scale array exists. This fixture used to fabricate one out of
    // *input*-quantization scales — the wrong object entirely — which
    // `QuantWeight::new` now refuses.
    larql_compute::FullPipelineLayer {
        attn_sinks: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_o_bias: None,
        attn_softcap: 0.0,
        wq: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &w.wq,
            larql_compute::QuantAux::None,
        ),
        wk: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &w.wk,
            larql_compute::QuantAux::None,
        ),
        wv: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &w.wv,
            larql_compute::QuantAux::None,
        ),
        wo: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &w.wo,
            larql_compute::QuantAux::None,
        ),
        gate: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &w.gate,
            larql_compute::QuantAux::None,
        ),
        up: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &w.up,
            larql_compute::QuantAux::None,
        ),
        down: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &w.down,
            larql_compute::QuantAux::None,
        ),
        input_norm: norm,
        post_attn_norm: norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 1.0,
        has_post_norms: false,
        activation: larql_compute::Activation::Silu,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        norm_type: larql_compute::NormType::RmsNorm,
        ffn_type,
        attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        head_dim: HEAD_DIM,
        num_q_heads: HEADS,
        num_kv_heads: HEADS,
        rope_base: 10000.0,
        rotary_dim: 0,
        rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
            HEAD_DIM,
            0_usize,
            10000.0_f64,
        ),
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        q_norm_weight: None,
        k_norm_weight: None,
        ffn_up_bias: None,
        ffn_down_bias: None,
        moe: None,
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
        kv_shared_source: None,
        residual_multiplier: 1.0,
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
    }
}

#[test]
fn full_pipeline_seq1_produces_nonzero() {
    let metal = get_metal();
    let weights = Q4_0Weights::constant(0.01, 0.01, 0.01);
    let norm = vec![1.0f32; HIDDEN];
    let x: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 * 0.01).sin()).collect();
    let layer = layer(&weights, &norm, larql_compute::FfnType::Gated);

    let result = metal.full_pipeline_q4(
        &[layer],
        &x,
        HIDDEN,
        INTER,
        1,     // seq_len
        false, // use_qk_norm
        0.0,   // softcap
    );
    assert!(result.is_some(), "full_pipeline_q4 should return Some");
    let output = result.unwrap();
    assert_eq!(output.len(), HIDDEN);
    // Finiteness is checked separately from magnitude. `v.abs() > 1e-6` is
    // false for NaN, so a NaN-filled result used to fail here with the message
    // "output should be nonzero" — which sent the investigation hunting for a
    // zeroed buffer when in fact every element was NaN. See the construction
    // lock in `backend::MetalBackend::with_options`.
    let non_finite = output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "Pipeline output has {non_finite}/{} non-finite values; head={:?}",
        output.len(),
        &output[..output.len().min(8)]
    );
    assert!(
        output.iter().any(|&v| v.abs() > 1e-6),
        "Pipeline output is finite but all-zero; head={:?}",
        &output[..output.len().min(8)]
    );
}

/// With the O projection and the FFN's first matrix zeroed, the activation
/// is exactly zero, so the layer must return its input unchanged:
/// `h_post_attn = x + 0` and `h_out = h_post_attn + down(0)`.
///
/// Regression guard for the Q4_0 down projection being handed the Q8 copy
/// of the *FFN-normed hidden state* (`HIDDEN` wide) instead of a Q8 copy of
/// the *activation* (`INTER` wide). That computed `down(q8(norm(h)))`,
/// which is nonzero here, and read `INTER - HIDDEN` int8s and their scales
/// past the end of the staging buffers. The out-of-bounds bytes belonged to
/// whatever sat next to them in GPU memory, which is why
/// `full_pipeline_seq1_produces_nonzero` failed with all-inf/NaN output
/// only when many backends were allocating at once (about half of all runs
/// of this binary while another process kept the GPU busy) and never alone.
/// This test does not depend on what the stray bytes hold: the wrong
/// operand alone makes the output differ from `x`.
fn assert_zero_activation_is_identity(ffn_type: larql_compute::FfnType) {
    let metal = get_metal();
    // Gated: gate = 0 → silu(0)·up = 0. Standard: up = 0 → act(0) = 0.
    let weights = match ffn_type {
        larql_compute::FfnType::Gated => Q4_0Weights::constant(0.0, 0.0, 0.01),
        _ => Q4_0Weights::constant(0.0, 0.01, 0.0),
    };
    let norm = vec![1.0f32; HIDDEN];
    let x: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 * 0.01).sin()).collect();
    let layer = layer(&weights, &norm, ffn_type);

    let out = metal
        .full_pipeline_q4(&[layer], &x, HIDDEN, INTER, 1, false, 0.0)
        .expect("full_pipeline_q4 should return Some");
    assert_eq!(out.len(), HIDDEN);
    let max_diff = max_diff(&out, &x);
    assert!(
        max_diff < 1e-6,
        "{ffn_type:?} FFN with a zero activation changed the residual by {max_diff}; \
         the Q4_0 down projection is not reading the activation. head={:?}",
        &out[..8]
    );
}

#[test]
fn q4_0_gated_down_projection_reads_the_activation() {
    assert_zero_activation_is_identity(larql_compute::FfnType::Gated);
}

#[test]
fn q4_0_standard_down_projection_reads_the_activation() {
    assert_zero_activation_is_identity(larql_compute::FfnType::Standard);
}
