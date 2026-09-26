//! STEP-PROFILE-1: the single-position lowering marks each projection
//! GEMV as its own stage.
//!
//! `--profile` prices a stage against the bytes its class reads. The
//! O projection, the FFN gate/up and the FFN down used to share a stage
//! with the norms, activation and residual around them, so a class's
//! GB/s mixed GEMV time with glue time. The lowering now marks
//! `attn.o_proj`, `ffn.gate_up` and `ffn.down` apart from that glue.
//!
//! These tests pin the exact mark sequence for every branch the split
//! touches — fused and unfused down, fused and unfused O, with and
//! without the attention gate — because a mark in the wrong place does
//! not fail anything else: the numbers stay right and the profile
//! silently prices glue as GEMV. Production's `SingleEncoder` ignores the
//! marks, so the arithmetic is covered by the existing parity tests.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::attention::{
    AttnScratch, AttnShape, AttnWeights, LoweredPosition,
};
use larql_compute_metal::lowering::ffn::{FfnActivation, FfnScratch, FfnShape, FfnWeights};
use larql_compute_metal::lowering::profile::{Stage, StageEncoders};
use larql_compute_metal::lowering::{LoweredMatrix, PostNorm};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;
use metal::ComputeCommandEncoderRef;

const HIDDEN: usize = 256;
const INTER: usize = 512;
const NUM_Q: usize = 8;
const NUM_KV: usize = 2;
const HEAD_DIM: usize = 32;
const Q_ROWS: usize = NUM_Q * HEAD_DIM;
const KV_ROWS: usize = NUM_KV * HEAD_DIM;
const T: usize = 10;
const EPS: f32 = 1e-5;
const SCORE_SCALE: f32 = 0.176_776_7; // 1/sqrt(32)
const THETA: f64 = 10_000.0;

/// Records each stage run in encode order — consecutive marks of the same
/// stage are one run, as the profiler counts them — and hands back the
/// one real encoder, so the lowering executes exactly as in production.
struct Recorder<'a> {
    enc: &'a ComputeCommandEncoderRef,
    runs: Vec<Stage>,
}

impl StageEncoders for Recorder<'_> {
    fn stage(&mut self, stage: Stage) -> &ComputeCommandEncoderRef {
        if self.runs.last() != Some(&stage) {
            self.runs.push(stage);
        }
        self.enc
    }
}

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 0.6
        })
        .collect()
}

fn gpu() -> MetalBackend {
    MetalBackend::new().expect("Metal device: these tests must not skip without one")
}

/// A device-resident NVFP4 matrix, leaked for the test's lifetime so the
/// `LoweredMatrix` borrows stay valid (and distinct allocations never
/// alias in the buffer cache).
fn resident(gpu: &MetalBackend, rows: usize, cols: usize, seed: u32) -> LoweredMatrix<'static> {
    let m = nvfp4::quantize(&det(rows * cols, seed), rows, cols).expect("quantise");
    let m = Box::leak(Box::new(m));
    LoweredMatrix::Nvfp4 {
        packed: Box::leak(Box::new(gpu.lowering_weight(&m.packed))),
        packed_offset: 0,
        scales: Box::leak(Box::new(gpu.lowering_weight(&m.scales))),
        scales_offset: 0,
        tensor_scale: m.tensor_scale,
    }
}

/// The stage runs one dense FFN encode produces, with or without the
/// four-norm post-FFN norm (without it the residual folds into the down
/// write).
fn ffn_runs(post_norm: bool) -> Vec<Stage> {
    let gpu = gpu();
    let h_in = gpu.lowering_upload(&det(HIDDEN, 1)).unwrap();
    let norm_w = gpu.lowering_upload(&vec![1.0f32; HIDDEN]).unwrap();
    let post_w = gpu.lowering_upload(&vec![1.0f32; HIDDEN]).unwrap();
    let post_scratch = gpu.lowering_scratch(HIDDEN);
    let h_out = gpu.lowering_scratch(HIDDEN);
    let (normed, g, u, a, d) = (
        gpu.lowering_scratch(HIDDEN),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(HIDDEN),
    );
    let w = FfnWeights {
        gate: resident(&gpu, INTER, HIDDEN, 2),
        up: resident(&gpu, INTER, HIDDEN, 3),
        down: resident(&gpu, HIDDEN, INTER, 4),
        norm_weight: &norm_w,
        post_norm: post_norm.then_some(PostNorm {
            weight: &post_w,
            eps: EPS,
            weight_offset: 0.0,
            scratch: &post_scratch,
        }),
    };
    let s = FfnScratch {
        normed: &normed,
        gate: &g,
        up: &u,
        act: &a,
        down: &d,
    };
    let shape = FfnShape {
        hidden: HIDDEN,
        intermediate: INTER,
        norm_eps: EPS,
        norm_weight_offset: 0.0,
        activation: FfnActivation::Silu,
        residual_scale: None,
    };
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut rec = Recorder {
        enc,
        runs: Vec::new(),
    };
    gpu.encode_gated_ffn(&mut rec, &h_in, &h_out, &w, &s, &shape);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    rec.runs
}

#[test]
fn four_norm_ffn_marks_each_projection_between_its_glue() {
    // pre-norm | gate/up | activation | down | post-norm + residual
    assert_eq!(
        ffn_runs(true),
        [
            Stage::DenseFfn,
            Stage::FfnGateUp,
            Stage::DenseFfn,
            Stage::FfnDown,
            Stage::DenseFfn,
        ]
    );
}

#[test]
fn two_norm_ffn_marks_the_residual_folded_down_as_the_down_stage() {
    // pre-norm | gate/up | activation | down (+ residual in the write)
    assert_eq!(
        ffn_runs(false),
        [
            Stage::DenseFfn,
            Stage::FfnGateUp,
            Stage::DenseFfn,
            Stage::FfnDown,
        ]
    );
}

/// Which attention-output branch an encode takes.
#[derive(Clone, Copy)]
struct AttnCase {
    /// A judged sigmoid gate on the attention output.
    gate: bool,
    /// An output-projection bias: keeps the O projection unfused.
    o_bias: bool,
}

/// The stage runs after attention over the cache: the output side the
/// split touches.
fn attn_tail_runs(case: AttnCase) -> Vec<Stage> {
    let gpu = gpu();
    let h_in = gpu.lowering_upload(&det(HIDDEN, 5)).unwrap();
    let norm_w = gpu.lowering_upload(&vec![1.0f32; HIDDEN]).unwrap();
    let k_cache = gpu.lowering_upload(&det(T * KV_ROWS, 6)).unwrap();
    let v_cache = gpu.lowering_upload(&det(T * KV_ROWS, 7)).unwrap();
    let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| (THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64)) as f32)
        .collect();
    let inv_freq = gpu.lowering_upload(&inv_freq).unwrap();
    let o_bias = gpu.lowering_upload(&det(HIDDEN, 8)).unwrap();
    let h_out = gpu.lowering_scratch(HIDDEN);
    let normed = gpu.lowering_scratch(HIDDEN);
    let q = gpu.lowering_scratch(Q_ROWS);
    let gate = gpu.lowering_scratch(Q_ROWS);
    let concat = gpu.lowering_scratch(Q_ROWS);
    let gated = gpu.lowering_scratch(Q_ROWS);
    let attn_out = gpu.lowering_scratch(HIDDEN);
    let w = AttnWeights {
        q: resident(&gpu, Q_ROWS, HIDDEN, 9),
        k: resident(&gpu, KV_ROWS, HIDDEN, 10),
        v: resident(&gpu, KV_ROWS, HIDDEN, 11),
        o: resident(&gpu, HIDDEN, Q_ROWS, 12),
        gate: case.gate.then(|| resident(&gpu, Q_ROWS, HIDDEN, 13)),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: case.o_bias.then_some(&o_bias),
        sinks: None,
        qk_norm: None,
        norm_weight: &norm_w,
        post_norm: None,
    };
    let s = AttnScratch {
        normed: &normed,
        q: &q,
        k_cache: &k_cache,
        v_cache: &v_cache,
        gate: &gate,
        concat: &concat,
        gated: &gated,
        attn_out: &attn_out,
        inv_freq: &inv_freq,
        splitk: None,
    };
    let shape = AttnShape {
        hidden: HIDDEN,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        head_dim: HEAD_DIM,
        norm_eps: EPS,
        norm_weight_offset: 0.0,
        qk_norm_eps: EPS,
        parameter_free_q: false,
        parameter_free_k: false,
        parameter_free_v: false,
        query_scale: None,
        score_scale: SCORE_SCALE,
        position: LoweredPosition::Scaled {
            theta: THETA,
            amplitude: 1.0,
        },
        window: None,
        softcap: None,
        position_index: T - 1,
        kv_len: T,
        residual_scale: None,
    };
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut rec = Recorder {
        enc,
        runs: Vec::new(),
    };
    gpu.encode_attention(&mut rec, &h_in, &h_out, &w, &s, &shape);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let core = rec
        .runs
        .iter()
        .position(|s| *s == Stage::AttnCore)
        .unwrap_or_else(|| panic!("no attn.core run in {:?}", rec.runs));
    rec.runs[core + 1..].to_vec()
}

#[test]
fn unfused_o_projection_is_its_own_stage_before_the_glue() {
    // o-proj | bias + residual
    assert_eq!(
        attn_tail_runs(AttnCase {
            gate: false,
            o_bias: true,
        }),
        [Stage::AttnOProj, Stage::AttnOut]
    );
}

#[test]
fn residual_folded_o_projection_is_the_whole_tail() {
    // o-proj with the residual in its write: no glue stage, and no empty
    // glue stage opened ahead of it (an empty sampled stage is pure
    // drain in the profile).
    assert_eq!(
        attn_tail_runs(AttnCase {
            gate: false,
            o_bias: false,
        }),
        [Stage::AttnOProj]
    );
}

#[test]
fn gated_attention_prices_the_gate_as_glue_ahead_of_the_projection() {
    // sigmoid gate | o-proj | bias + residual
    assert_eq!(
        attn_tail_runs(AttnCase {
            gate: true,
            o_bias: true,
        }),
        [Stage::AttnOut, Stage::AttnOProj, Stage::AttnOut]
    );
}
