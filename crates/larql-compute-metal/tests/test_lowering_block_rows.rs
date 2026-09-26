//! VERIFY-N: the multi-position FFN, head and stack lowerings against
//! their single-position encodes.
//!
//! Each `*_rows` encode over a block of `ROWS` positions must produce, row
//! for row, what the single-position encode produces for that position —
//! that is the whole contract of a verify block: it predicts exactly what
//! `step` would have. So each test runs the block once and the same
//! positions one at a time, and compares.
//!
//! The comparison is a tolerance, not bitwise: the block's projections run
//! the multi-RHS NVFP4 matmuls where a single position runs the (fused)
//! GEMV — same per-row element order, different grouping, gated at 1e-5
//! in `test_lowering_matmul_rows`. Every other op in these blocks
//! (`rms_norm_rows`, the combine, residual, head scale/softcap) shares its
//! per-row arithmetic with the single-position kernel.
//!
//! Parity alone cannot show that the block READ a judged fact, so each
//! test carries a control that must miss parity by orders of magnitude:
//! the FFN block against the other activations' single-position outputs,
//! the head block against the other multiplier/softcap setting, and the
//! stack block against the stack with its layers' FFNs swapped.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::attention::{AttnShape, AttnWeights, LoweredPosition};
use larql_compute_metal::lowering::ffn::{FfnActivation, FfnScratch, FfnShape, FfnWeights};
use larql_compute_metal::lowering::head::{HeadScratch, HeadShape, HeadWeights};
use larql_compute_metal::lowering::profile::SingleEncoder;
use larql_compute_metal::lowering::stack::{
    Checkpoint, LayerFfnLowering, LayerLowering, StackScratch,
};
use larql_compute_metal::lowering::{LoweredMatrix, PostNorm};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

const HIDDEN: usize = 256;
const INTER: usize = 512;
/// Positions per verify block. Not a power of two, so the block's
/// matmuls decompose across more than one arm.
const ROWS: usize = 5;
const EPS: f32 = 1e-5;
const POST_EPS: f32 = 1e-6;
/// Centred norm convention (`1 + w`) for the branch norms.
const OFFSET: f32 = 1.0;
/// Glimmer's final norm is uncentred.
const HEAD_OFFSET: f32 = 0.0;
const RESIDUAL_SCALE: f32 = 0.5;
/// Kimi-K3's SiTU-GLU bounds (`test_lowering_situ`).
const SITU_BETA: f32 = 4.0;
const SITU_LINEAR_BETA: f32 = 25.0;
/// Not a multiple of 8 or 16: every matmul arm has a tail.
const VOCAB: usize = 1000;
/// Glimmer's output multiplier and final logit softcap.
const HEAD_MULTIPLIER: f32 = 0.196;
const HEAD_SOFTCAP: f32 = 20.0;
/// Final-norm weight level that puts the multiplied logits at the
/// softcap's own scale (std ~18 here), where tanh is nonlinear. At unit
/// weights the logits sit near 0.1 and the softcap is the identity to
/// 1e-5 — its control would be blind.
const HEAD_NORM_GAIN: f32 = 40.0;
const WEIGHT_AMPLITUDE: f32 = 0.5;
/// The matmul-vs-GEMV bound of `test_lowering_matmul_rows`.
const PARITY: f64 = 1e-5;
/// A control must miss parity by at least this factor.
const CONTROL_OVER_PARITY: f64 = 100.0;

// Stack geometry: no measured attention row, so serial throughout.
const LAYERS: usize = 2;
const NUM_Q: usize = 4;
const NUM_KV: usize = 1;
const HEAD_DIM: usize = 32;
const Q_ROWS: usize = NUM_Q * HEAD_DIM;
const KV_ROWS: usize = NUM_KV * HEAD_DIM;
/// Cache positions before the block.
const BASE: usize = 12;
const CACHE_POSITIONS: usize = BASE + ROWS;
const THETA: f64 = 10_000.0;

fn det(n: usize, seed: u32, amplitude: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(29);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * amplitude
        })
        .collect()
}

fn rel_rms(reference: &[f32], got: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        num += (*a as f64 - *b as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn run(gpu: &MetalBackend, encode: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode(enc);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Fail, never skip: a shader that does not compile makes `new()` return
/// None, and a skipped gate reads as a pass.
fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal backend (shader library must compile)")
}

/// An NVFP4 matrix, quantised and LEAKED: `lowering_weight` caches the
/// device copy on the host allocation's (ptr, len), so weight bytes must
/// outlive every later allocation that could reuse their address.
fn matrix(n: usize, k: usize, seed: u32) -> &'static nvfp4::Nvfp4Matrix {
    Box::leak(Box::new(
        nvfp4::quantize(&det(n * k, seed, WEIGHT_AMPLITUDE), n, k).expect("quantise"),
    ))
}

/// Device residency for one NVFP4 matrix.
struct Resident {
    packed: metal::Buffer,
    scales: metal::Buffer,
    tensor_scale: f32,
}

impl Resident {
    fn new(gpu: &MetalBackend, m: &nvfp4::Nvfp4Matrix) -> Self {
        Self {
            packed: gpu.lowering_weight(&m.packed),
            scales: gpu.lowering_weight(&m.scales),
            tensor_scale: m.tensor_scale,
        }
    }
    fn lowered(&self) -> LoweredMatrix<'_> {
        LoweredMatrix::Nvfp4 {
            packed: &self.packed,
            packed_offset: 0,
            scales: &self.scales,
            scales_offset: 0,
            tensor_scale: self.tensor_scale,
        }
    }
}

fn assert_rows_match(what: &str, reference: &[f32], got: &[f32], width: usize) -> f64 {
    assert!(got.iter().all(|v| v.is_finite()), "{what}: non-finite");
    let mut worst = 0.0f64;
    for r in 0..ROWS {
        let span = r * width..(r + 1) * width;
        let e = rel_rms(&reference[span.clone()], &got[span]);
        assert!(
            e < PARITY,
            "{what}: row {r} off its single position: {e:.3e}"
        );
        worst = worst.max(e);
    }
    eprintln!("{what}: worst row rel_rms {worst:.3e}");
    worst
}

fn assert_control(what: &str, parity: f64, moved: f64) {
    eprintln!(
        "{what}: control moved {moved:.3e} ({:.0}x parity)",
        moved / parity.max(1e-12)
    );
    assert!(
        moved > parity * CONTROL_OVER_PARITY && moved > PARITY,
        "control BLIND: `{what}` moved the block only {moved:.3e} (parity {parity:.3e})"
    );
}

// ── FFN ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct FfnCase {
    activation: FfnActivation,
    post_norm: bool,
    residual_scale: Option<f32>,
}

const FFN_CASES: [FfnCase; 3] = [
    // Four-norm placement with a residual scale.
    FfnCase {
        activation: FfnActivation::Silu,
        post_norm: true,
        residual_scale: Some(RESIDUAL_SCALE),
    },
    // Two-norm placement: the single position folds the residual into the
    // down projection; the block adds it separately.
    FfnCase {
        activation: FfnActivation::GeluTanh,
        post_norm: false,
        residual_scale: None,
    },
    FfnCase {
        activation: FfnActivation::SituGlu {
            beta: SITU_BETA,
            linear_beta: Some(SITU_LINEAR_BETA),
        },
        post_norm: true,
        residual_scale: None,
    },
];

/// `(block, per-position)` outputs of one FFN case, `[ROWS, HIDDEN]` each.
fn ffn_case(gpu: &MetalBackend, c: &FfnCase) -> (Vec<f32>, Vec<f32>) {
    let (gate, up, down) = (
        Resident::new(gpu, matrix(INTER, HIDDEN, 1)),
        Resident::new(gpu, matrix(INTER, HIDDEN, 2)),
        Resident::new(gpu, matrix(HIDDEN, INTER, 3)),
    );
    let upload = |x: &[f32]| gpu.lowering_upload(x).expect("upload");
    let norm = upload(&det(HIDDEN, 4, WEIGHT_AMPLITUDE));
    let post = upload(&det(HIDDEN, 5, WEIGHT_AMPLITUDE));
    let h = det(ROWS * HIDDEN, 6, 1.0);
    let post_single = gpu.lowering_scratch(HIDDEN);
    let w = FfnWeights {
        gate: gate.lowered(),
        up: up.lowered(),
        down: down.lowered(),
        norm_weight: &norm,
        post_norm: c.post_norm.then_some(PostNorm {
            weight: &post,
            eps: POST_EPS,
            weight_offset: OFFSET,
            scratch: &post_single,
        }),
    };
    let shape = FfnShape {
        hidden: HIDDEN,
        intermediate: INTER,
        norm_eps: EPS,
        norm_weight_offset: OFFSET,
        activation: c.activation,
        residual_scale: c.residual_scale,
    };
    let scratch = |width: usize| {
        (
            gpu.lowering_scratch(width * HIDDEN),
            gpu.lowering_scratch(width * INTER),
            gpu.lowering_scratch(width * INTER),
            gpu.lowering_scratch(width * INTER),
            gpu.lowering_scratch(width * HIDDEN),
        )
    };

    let (normed, g, u, act, dn) = scratch(ROWS);
    let block_scratch = FfnScratch {
        normed: &normed,
        gate: &g,
        up: &u,
        act: &act,
        down: &dn,
    };
    let h_in = upload(&h);
    let h_out = gpu.lowering_scratch(ROWS * HIDDEN);
    let post_rows = gpu.lowering_scratch(ROWS * HIDDEN);
    run(gpu, |enc| {
        gpu.encode_gated_ffn_rows(
            &mut SingleEncoder(enc),
            &h_in,
            &h_out,
            &w,
            &block_scratch,
            &post_rows,
            &shape,
            ROWS,
        )
        .expect("the block encodes")
    });
    let block = gpu
        .lowering_readback(&h_out, ROWS * HIDDEN)
        .expect("readback");

    let (normed1, g1, u1, act1, dn1) = scratch(1);
    let single_scratch = FfnScratch {
        normed: &normed1,
        gate: &g1,
        up: &u1,
        act: &act1,
        down: &dn1,
    };
    let ins: Vec<metal::Buffer> = (0..ROWS)
        .map(|r| upload(&h[r * HIDDEN..(r + 1) * HIDDEN]))
        .collect();
    let outs: Vec<metal::Buffer> = (0..ROWS).map(|_| gpu.lowering_scratch(HIDDEN)).collect();
    run(gpu, |enc| {
        for (i, o) in ins.iter().zip(&outs) {
            gpu.encode_gated_ffn(&mut SingleEncoder(enc), i, o, &w, &single_scratch, &shape);
        }
    });
    let each = outs
        .iter()
        .flat_map(|b| gpu.lowering_readback(b, HIDDEN).expect("readback"))
        .collect();
    (block, each)
}

#[test]
fn ffn_rows_match_each_position_under_every_activation() {
    let gpu = backend();
    let results: Vec<(Vec<f32>, Vec<f32>)> = FFN_CASES.iter().map(|c| ffn_case(&gpu, c)).collect();
    let mut worst = 0.0f64;
    for (c, (block, each)) in FFN_CASES.iter().zip(&results) {
        let what = format!("ffn rows, {:?}", c.activation);
        worst = worst.max(assert_rows_match(&what, each, block, HIDDEN));
    }
    // The block must have run ITS activation: each block output is far
    // from the single positions of the SAME placement under the next
    // case's activation.
    for (i, (c, (block, _))) in FFN_CASES.iter().zip(&results).enumerate() {
        let other = FFN_CASES[(i + 1) % FFN_CASES.len()].activation;
        let (_, each_other) = ffn_case(
            &gpu,
            &FfnCase {
                activation: other,
                ..*c
            },
        );
        assert_control(
            &format!("{:?} block vs {other:?} positions", c.activation),
            worst,
            rel_rms(&each_other, block),
        );
    }
}

// ── head ────────────────────────────────────────────────────────────

/// `(block, per-position)` logits, `[ROWS, VOCAB]` each.
fn head_case(
    gpu: &MetalBackend,
    multiplier: Option<f32>,
    softcap: Option<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let proj = Resident::new(gpu, matrix(VOCAB, HIDDEN, 7));
    let upload = |x: &[f32]| gpu.lowering_upload(x).expect("upload");
    let norm: Vec<f32> = det(HIDDEN, 8, WEIGHT_AMPLITUDE)
        .iter()
        .map(|w| w + HEAD_NORM_GAIN)
        .collect();
    let norm = upload(&norm);
    let h = det(ROWS * HIDDEN, 9, 1.0);
    let w = HeadWeights {
        projection: proj.lowered(),
        norm_weight: &norm,
    };
    let shape = HeadShape {
        hidden: HIDDEN,
        vocab: VOCAB,
        norm_eps: EPS,
        norm_weight_offset: HEAD_OFFSET,
        multiplier,
        softcap,
    };

    let (normed, raw) = (
        gpu.lowering_scratch(ROWS * HIDDEN),
        gpu.lowering_scratch(ROWS * VOCAB),
    );
    let h_in = upload(&h);
    let logits = gpu.lowering_scratch(ROWS * VOCAB);
    run(gpu, |enc| {
        gpu.encode_head_rows(
            &mut SingleEncoder(enc),
            &h_in,
            &logits,
            &w,
            &HeadScratch {
                normed: &normed,
                raw_logits: &raw,
            },
            &shape,
            ROWS,
        )
        .expect("the block encodes")
    });
    let block = gpu
        .lowering_readback(&logits, ROWS * VOCAB)
        .expect("readback");

    let (normed1, raw1) = (gpu.lowering_scratch(HIDDEN), gpu.lowering_scratch(VOCAB));
    let ins: Vec<metal::Buffer> = (0..ROWS)
        .map(|r| upload(&h[r * HIDDEN..(r + 1) * HIDDEN]))
        .collect();
    let outs: Vec<metal::Buffer> = (0..ROWS).map(|_| gpu.lowering_scratch(VOCAB)).collect();
    run(gpu, |enc| {
        for (i, o) in ins.iter().zip(&outs) {
            gpu.encode_head(
                &mut SingleEncoder(enc),
                i,
                o,
                &w,
                &HeadScratch {
                    normed: &normed1,
                    raw_logits: &raw1,
                },
                &shape,
            );
        }
    });
    let each = outs
        .iter()
        .flat_map(|b| gpu.lowering_readback(b, VOCAB).expect("readback"))
        .collect();
    (block, each)
}

#[test]
fn head_rows_match_each_position_and_read_multiplier_and_softcap() {
    let gpu = backend();
    let (block, each) = head_case(&gpu, Some(HEAD_MULTIPLIER), Some(HEAD_SOFTCAP));
    let parity = assert_rows_match("head rows, multiplier + softcap", &each, &block, VOCAB);
    let (bare_block, bare_each) = head_case(&gpu, None, None);
    let bare = assert_rows_match("head rows, bare", &bare_each, &bare_block, VOCAB);
    let parity = parity.max(bare);
    // Each op on its own: the block is far from positions lacking it.
    let (_, no_softcap) = head_case(&gpu, Some(HEAD_MULTIPLIER), None);
    assert_control(
        "head block vs positions without softcap",
        parity,
        rel_rms(&no_softcap, &block),
    );
    let (_, no_multiplier) = head_case(&gpu, None, Some(HEAD_SOFTCAP));
    assert_control(
        "head block vs positions without multiplier",
        parity,
        rel_rms(&no_multiplier, &block),
    );
}

// ── stack ───────────────────────────────────────────────────────────

struct StackLayerHost {
    q: &'static nvfp4::Nvfp4Matrix,
    k: &'static nvfp4::Nvfp4Matrix,
    v: &'static nvfp4::Nvfp4Matrix,
    o: &'static nvfp4::Nvfp4Matrix,
    fg: &'static nvfp4::Nvfp4Matrix,
    fu: &'static nvfp4::Nvfp4Matrix,
    fd: &'static nvfp4::Nvfp4Matrix,
    norms: [Vec<f32>; 4],
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

fn stack_layer_host(l: u32) -> StackLayerHost {
    let s = |i: u32| l * 100 + i;
    StackLayerHost {
        q: matrix(Q_ROWS, HIDDEN, s(10)),
        k: matrix(KV_ROWS, HIDDEN, s(11)),
        v: matrix(KV_ROWS, HIDDEN, s(12)),
        o: matrix(HIDDEN, Q_ROWS, s(13)),
        fg: matrix(INTER, HIDDEN, s(14)),
        fu: matrix(INTER, HIDDEN, s(15)),
        fd: matrix(HIDDEN, INTER, s(16)),
        norms: [17, 18, 19, 20].map(|i| det(HIDDEN, s(i), WEIGHT_AMPLITUDE)),
        k_cache: det(CACHE_POSITIONS * KV_ROWS, s(21), 1.0),
        v_cache: det(CACHE_POSITIONS * KV_ROWS, s(22), 1.0),
    }
}

struct StackLayerDevice {
    mats: [Resident; 7],
    norms: [metal::Buffer; 4],
}

fn attn_shape(l: usize, position_index: usize) -> AttnShape {
    AttnShape {
        hidden: HIDDEN,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        head_dim: HEAD_DIM,
        norm_eps: EPS,
        norm_weight_offset: OFFSET,
        qk_norm_eps: EPS,
        parameter_free_q: true,
        parameter_free_k: true,
        parameter_free_v: false,
        query_scale: None,
        score_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        // Layer 0 rotates, layer 1 is NoPE.
        position: if l == 0 {
            LoweredPosition::Rope { theta: THETA }
        } else {
            LoweredPosition::None
        },
        window: None,
        softcap: None,
        position_index,
        kv_len: position_index + 1,
        residual_scale: None,
    }
}

/// The layer program for position `position_index` (the block's base for
/// the rows encode). Layer 0 is four-norm placement, layer 1 two-norm;
/// `swap_ffn` exchanges their FFN activations (the control).
#[allow(clippy::too_many_arguments)]
fn layers<'a>(
    dev: &'a [StackLayerDevice],
    caches: &'a [(metal::Buffer, metal::Buffer)],
    inv_freq: &'a metal::Buffer,
    post_attn: &'a metal::Buffer,
    post_ffn: &'a metal::Buffer,
    position_index: usize,
    swap_ffn: bool,
    gate: Option<&'a Resident>,
) -> Vec<LayerLowering<'a>> {
    dev.iter()
        .zip(caches)
        .enumerate()
        .map(|(l, (d, (k_cache, v_cache)))| {
            let four_norm = l == 0;
            let activation = if (l == 0) != swap_ffn {
                FfnActivation::Silu
            } else {
                FfnActivation::GeluTanh
            };
            LayerLowering {
                attn: AttnWeights {
                    q: d.mats[0].lowered(),
                    k: d.mats[1].lowered(),
                    v: d.mats[2].lowered(),
                    o: d.mats[3].lowered(),
                    gate: gate.map(Resident::lowered),
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,
                    o_bias: None,
                    sinks: None,
                    qk_norm: None,
                    norm_weight: &d.norms[0],
                    post_norm: four_norm.then_some(PostNorm {
                        weight: &d.norms[1],
                        eps: POST_EPS,
                        weight_offset: OFFSET,
                        scratch: post_attn,
                    }),
                },
                attn_shape: attn_shape(l, position_index),
                ffn: LayerFfnLowering::Dense {
                    weights: FfnWeights {
                        gate: d.mats[4].lowered(),
                        up: d.mats[5].lowered(),
                        down: d.mats[6].lowered(),
                        norm_weight: &d.norms[2],
                        post_norm: four_norm.then_some(PostNorm {
                            weight: &d.norms[3],
                            eps: POST_EPS,
                            weight_offset: OFFSET,
                            scratch: post_ffn,
                        }),
                    },
                    shape: FfnShape {
                        hidden: HIDDEN,
                        intermediate: INTER,
                        norm_eps: EPS,
                        norm_weight_offset: OFFSET,
                        activation,
                        residual_scale: None,
                    },
                },
                k_cache,
                v_cache,
                inv_freq,
            }
        })
        .collect()
}

/// Scratch for `width` positions. Index map as `StackScratch`'s fields.
fn stack_buffers(gpu: &MetalBackend, width: usize) -> Vec<metal::Buffer> {
    [
        HIDDEN, HIDDEN, HIDDEN, Q_ROWS, Q_ROWS, Q_ROWS, Q_ROWS, HIDDEN, HIDDEN, HIDDEN, INTER,
        INTER, INTER, HIDDEN, HIDDEN,
    ]
    .iter()
    .map(|n| gpu.lowering_scratch(width * n))
    .collect()
}

fn stack_scratch(b: &[metal::Buffer]) -> StackScratch<'_> {
    StackScratch {
        h_a: &b[0],
        h_b: &b[1],
        attn_normed: &b[2],
        q: &b[3],
        gate: &b[4],
        concat: &b[5],
        gated: &b[6],
        attn_out: &b[7],
        attn_post: &b[8],
        ffn_normed: &b[9],
        ffn_gate: &b[10],
        ffn_up: &b[11],
        ffn_act: &b[12],
        ffn_down: &b[13],
        ffn_post: &b[14],
        hybrid: None,
        splitk: None,
    }
}

struct StackRig {
    host: Vec<StackLayerHost>,
    dev: Vec<StackLayerDevice>,
    inv_freq: metal::Buffer,
    h: Vec<f32>,
}

fn stack_rig(gpu: &MetalBackend) -> StackRig {
    let host: Vec<StackLayerHost> = (0..LAYERS as u32).map(stack_layer_host).collect();
    let dev = host
        .iter()
        .map(|w| StackLayerDevice {
            mats: [w.q, w.k, w.v, w.o, w.fg, w.fu, w.fd].map(|m| Resident::new(gpu, m)),
            norms: [0, 1, 2, 3].map(|i| gpu.lowering_upload(&w.norms[i]).expect("upload")),
        })
        .collect();
    let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64) as f32)
        .collect();
    StackRig {
        host,
        dev,
        inv_freq: gpu.lowering_upload(&inv_freq).expect("upload"),
        h: det(ROWS * HIDDEN, 23, 1.0),
    }
}

fn fresh_caches(gpu: &MetalBackend, rig: &StackRig) -> Vec<(metal::Buffer, metal::Buffer)> {
    rig.host
        .iter()
        .map(|w| {
            (
                gpu.lowering_upload(&w.k_cache).expect("upload"),
                gpu.lowering_upload(&w.v_cache).expect("upload"),
            )
        })
        .collect()
}

fn read_caches(gpu: &MetalBackend, caches: &[(metal::Buffer, metal::Buffer)]) -> Vec<Vec<f32>> {
    caches
        .iter()
        .flat_map(|(k, v)| [k, v])
        .map(|b| {
            gpu.lowering_readback(b, CACHE_POSITIONS * KV_ROWS)
                .expect("readback")
        })
        .collect()
}

/// The block through `encode_stack_rows`: `(output, caches)`.
fn stack_block(gpu: &MetalBackend, rig: &StackRig, swap_ffn: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
    let caches = fresh_caches(gpu, rig);
    let bufs = stack_buffers(gpu, ROWS);
    let scratch = stack_scratch(&bufs);
    let h_in = gpu.lowering_upload(&rig.h).expect("upload");
    let ls = layers(
        &rig.dev,
        &caches,
        &rig.inv_freq,
        &bufs[8],
        &bufs[14],
        BASE,
        swap_ffn,
        None,
    );
    let mut out = None;
    run(gpu, |enc| {
        out = Some(
            gpu.encode_stack_rows(&mut SingleEncoder(enc), &h_in, &ls, &scratch, ROWS)
                .expect("the block encodes"),
        );
    });
    let out = out.expect("encoded");
    assert!(
        std::ptr::eq(out, scratch.h_a) || std::ptr::eq(out, scratch.h_b),
        "the block's output must be one of the stack's ping-pong buffers"
    );
    (
        gpu.lowering_readback(out, ROWS * HIDDEN).expect("readback"),
        read_caches(gpu, &caches),
    )
}

/// The same positions through `encode_stack`, one at a time, each row's
/// final hidden state captured by a checkpoint after the last layer.
fn stack_each(gpu: &MetalBackend, rig: &StackRig) -> (Vec<f32>, Vec<Vec<f32>>) {
    let caches = fresh_caches(gpu, rig);
    let bufs = stack_buffers(gpu, 1);
    let scratch = stack_scratch(&bufs);
    let ins: Vec<metal::Buffer> = (0..ROWS)
        .map(|r| {
            gpu.lowering_upload(&rig.h[r * HIDDEN..(r + 1) * HIDDEN])
                .expect("upload")
        })
        .collect();
    let caps: Vec<metal::Buffer> = (0..ROWS).map(|_| gpu.lowering_scratch(HIDDEN)).collect();
    run(gpu, |enc| {
        for r in 0..ROWS {
            let ls = layers(
                &rig.dev,
                &caches,
                &rig.inv_freq,
                &bufs[8],
                &bufs[14],
                BASE + r,
                false,
                None,
            );
            gpu.encode_stack(
                &mut SingleEncoder(enc),
                &ins[r],
                &ls,
                &scratch,
                &[Checkpoint {
                    after_layer: LAYERS - 1,
                    into: &caps[r],
                }],
            );
        }
    });
    let out = caps
        .iter()
        .flat_map(|b| gpu.lowering_readback(b, HIDDEN).expect("readback"))
        .collect();
    (out, read_caches(gpu, &caches))
}

#[test]
fn stack_rows_match_each_position_through_every_layer() {
    let gpu = backend();
    let rig = stack_rig(&gpu);
    let (block, block_caches) = stack_block(&gpu, &rig, false);
    let (each, each_caches) = stack_each(&gpu, &rig);
    let parity = assert_rows_match("stack rows", &each, &block, HIDDEN);
    // Every layer's block slots, K and V, against the per-position writes;
    // the prefix before the block untouched by either.
    let prefix = BASE * KV_ROWS;
    for (i, (b, e)) in block_caches.iter().zip(&each_caches).enumerate() {
        assert!(
            b[..prefix]
                .iter()
                .zip(&e[..prefix])
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "cache {i}: the block wrote before its base"
        );
        let err = rel_rms(&e[prefix..], &b[prefix..]);
        assert!(err < PARITY, "cache {i}: block slots off: {err:.3e}");
    }
    // Control: the block with its layers' FFN activations swapped.
    let (swapped, _) = stack_block(&gpu, &rig, true);
    assert_control(
        "stack block with swapped FFN activations",
        parity,
        rel_rms(&each, &swapped),
    );
}

#[test]
fn stack_rows_propagates_the_attention_gate_refusal() {
    let gpu = backend();
    let rig = stack_rig(&gpu);
    let caches = fresh_caches(&gpu, &rig);
    let bufs = stack_buffers(&gpu, ROWS);
    let scratch = stack_scratch(&bufs);
    let h_in = gpu.lowering_upload(&rig.h).expect("upload");
    let gate = Resident::new(&gpu, matrix(Q_ROWS, HIDDEN, 30));
    let ls = layers(
        &rig.dev,
        &caches,
        &rig.inv_freq,
        &bufs[8],
        &bufs[14],
        BASE,
        false,
        Some(&gate),
    );
    let mut err = None;
    run(&gpu, |enc| {
        err = gpu
            .encode_stack_rows(&mut SingleEncoder(enc), &h_in, &ls, &scratch, ROWS)
            .err();
    });
    let err = err.expect("a gated layer has no multi-position lowering");
    assert!(err.contains("output-gate"), "{err}");
}
