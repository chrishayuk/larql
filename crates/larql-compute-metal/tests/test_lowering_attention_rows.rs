//! VERIFY-N: `encode_attention_rows` against per-position
//! `encode_attention`, and the attention arms each one reaches.
//!
//! A verify block is correct only if row `i` produces what `step` would
//! have produced at position `base + i` — the same hidden-state output
//! and the same K/V written into the cache. So every case here runs the
//! block once through the rows lowering and once as `ROWS` consecutive
//! single-position encodes over an identical copy of the cache, and
//! compares both.
//!
//! The comparison is a tolerance, not bitwise: the block's projections
//! run the multi-RHS NVFP4 matmuls where a single position runs the
//! fused-segment GEMV (same per-row element order, different grouping —
//! `test_lowering_matmul_rows` gates that at 1e-5), and a block that
//! falls back from split-K to seqpar reassociates the attention sum
//! against the per-position split-K dispatches
//! (`test_kernel_kv_attention_splitk`). Both stay inside the same 1e-5.
//!
//! Parity alone cannot show WHICH kernel ran, so each case also asserts
//! the route witness: how many serial / seqpar / split-K attention
//! dispatches the block encoded against the per-position path's. And a
//! causality control re-encodes row 0 with the whole block visible, which
//! must miss parity by orders of magnitude — otherwise the gate could not
//! tell a causal block from one where early rows read later rows' keys.
//!
//! | case | attention arm the block takes |
//! |---|---|
//! | unmeasured geometry | one `kv_attention_rows` dispatch (serial) |
//! | Gemma geometry, no split-K scratch | one `kv_attention_seqpar_rows` dispatch |
//! | Gemma geometry, span >= 128 | one split-K op over the block |
//! | split-K scratch sized for one row | split-K refused (`fits`), seqpar rows |
//! | block straddling the 512 split-K tier | split-K refused (chunk counts differ), seqpar rows |
//! | block past the short-kernel span | per-row loop on the long kernel |
//! | gpt-oss geometry straddling its 512 slice tier | per-row loop (slice counts differ) |

#![cfg(target_os = "macos")]

use std::sync::atomic::Ordering;
use std::sync::Mutex;

use larql_compute_metal::lowering::attention::{
    AttnScratch, AttnShape, AttnWeights, LoweredPosition, QkNormWeights,
};
use larql_compute_metal::lowering::profile::SingleEncoder;
use larql_compute_metal::lowering::{LoweredMatrix, PostNorm};
use larql_compute_metal::ops::kv_seqpar::SeqparRequest;
use larql_compute_metal::ops::kv_splitk::{ml_part_len, SplitKScratch, SPLITK_MAX_CHUNKS};
use larql_compute_metal::{route_witness, MetalBackend};
use larql_models::quant::nvfp4;

const HIDDEN: usize = 256;
/// Positions per verify block. Not a power of two, so the block's
/// projections decompose across more than one matmul arm.
const ROWS: usize = 5;
const EPS: f32 = 1e-5;
const QK_EPS: f32 = 1e-6;
const POST_EPS: f32 = 1e-6;
/// Centred norm convention (`1 + w`).
const OFFSET: f32 = 1.0;
const THETA: f64 = 10_000.0;
const QUERY_SCALE: f32 = 1.7;
/// Gemma 2's attention logit softcap.
const SOFTCAP: f32 = 50.0;
const RESIDUAL_SCALE: f32 = 0.5;
const WEIGHT_AMPLITUDE: f32 = 0.5;
const SINK_AMPLITUDE: f32 = 2.0;
/// See the module doc: `test_lowering_matmul_rows`'s matmul-vs-GEMV
/// bound. Measured worst across these cases ~1e-6 (M3 Max), so a
/// reassociated attention merge fits with an order of magnitude spare.
const PARITY: f64 = 1e-5;
/// A block that let row 0 see rows 1.. must miss parity by this factor.
const CAUSAL_CONTROL_OVER_PARITY: f64 = 100.0;

/// The route witness is process-global; every test in this binary that
/// reads it holds this lock so no other test's encodes land in its delta.
static WITNESS_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug)]
struct Geom {
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
}

impl Geom {
    fn q_rows(self) -> usize {
        self.num_q * self.head_dim
    }
    fn kv_rows(self) -> usize {
        self.num_kv * self.head_dim
    }
}

/// No row in `ops::attention_geometry`: serial at every span when the
/// seqpar request is unset, and never split-K.
const UNMEASURED: Geom = Geom {
    num_q: 4,
    num_kv: 1,
    head_dim: 32,
};
/// Gemma 3 4B's measured row: seqpar(4) at every span, and split-K from
/// span 128 (8 chunks) and 512 (16 chunks).
const GEMMA: Geom = Geom {
    num_q: 8,
    num_kv: 4,
    head_dim: 256,
};
/// gpt-oss-20b's measured row: 8 slices below span 512, 12 from 512.
const GPT_OSS: Geom = Geom {
    num_q: 64,
    num_kv: 8,
    head_dim: 64,
};

/// Short context: every row well inside the short kernel, below split-K.
const SHORT_BASE: usize = 12;
/// `SHORT_BASE`'s rows under a sliding window narrower than the block's
/// cache, so every row's span is clamped to it.
const WINDOW: usize = 8;
/// Spans 201..=205: inside Gemma's split-K tier [128, 512).
const SPLITK_BASE: usize = 200;
/// Spans 510..=514: straddle the 512 tier boundary (Gemma: 8 → 16 split-K
/// chunks; gpt-oss: 8 → 12 seqpar slices).
const TIER_512_BASE: usize = 509;
/// Spans past `SHORT_ATTENTION_SPAN` (1024): the long kernel.
const LONG_BASE: usize = 1100;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Variant {
    /// Biases on all four projections, weighted QK norm, V norm, RoPE,
    /// sinks, softcap, post-attention norm and a residual scale.
    Full,
    /// Parameter-free Q/K norm, query scale, NoPE, no post norm, no bias
    /// — the per-position path folds the residual into the o-proj.
    Plain,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    geom: Geom,
    base: usize,
    window: Option<usize>,
    variant: Variant,
    /// Positions the split-K scratch holds; `None` = no scratch.
    splitk_rows: Option<usize>,
}

/// Route-witness readings for the three lowered attention arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct Witness {
    serial: u64,
    seqpar: u64,
    splitk: u64,
}

fn witness() -> Witness {
    Witness {
        serial: route_witness::LOWERED_ATTEND_SERIAL.load(Ordering::Relaxed),
        seqpar: route_witness::LOWERED_ATTEND_SEQPAR.load(Ordering::Relaxed),
        splitk: route_witness::LOWERED_ATTEND_SPLITK.load(Ordering::Relaxed),
    }
}

impl Witness {
    fn since(self, earlier: Witness) -> Witness {
        Witness {
            serial: self.serial - earlier.serial,
            seqpar: self.seqpar - earlier.seqpar,
            splitk: self.splitk - earlier.splitk,
        }
    }
}

fn det(n: usize, seed: u32, amplitude: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(9);
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
/// None, and a skipped gate reads as a pass. The seqpar request is pinned
/// to `Unset` (the measured policy) so an operator's `LARQL_KV_SEQPAR`
/// cannot move these cases onto other arms.
fn backend() -> MetalBackend {
    let mut gpu = MetalBackend::new().expect("Metal backend (shader library must compile)");
    gpu.decode_flags.kv_seqpar = SeqparRequest::Unset;
    gpu
}

/// Host-side operands for one geometry.
struct Operands {
    q: nvfp4::Nvfp4Matrix,
    k: nvfp4::Nvfp4Matrix,
    v: nvfp4::Nvfp4Matrix,
    o: nvfp4::Nvfp4Matrix,
    norm: Vec<f32>,
    post: Vec<f32>,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    o_bias: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    sinks: Vec<f32>,
    inv_freq: Vec<f32>,
    h: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

fn operands(g: Geom, cache_positions: usize) -> Operands {
    let mk = |n: usize, k: usize, seed: u32| {
        nvfp4::quantize(&det(n * k, seed, WEIGHT_AMPLITUDE), n, k).expect("quantise")
    };
    Operands {
        q: mk(g.q_rows(), HIDDEN, 1),
        k: mk(g.kv_rows(), HIDDEN, 2),
        v: mk(g.kv_rows(), HIDDEN, 3),
        o: mk(HIDDEN, g.q_rows(), 4),
        norm: det(HIDDEN, 5, WEIGHT_AMPLITUDE),
        post: det(HIDDEN, 6, WEIGHT_AMPLITUDE),
        q_bias: det(g.q_rows(), 7, WEIGHT_AMPLITUDE),
        k_bias: det(g.kv_rows(), 8, WEIGHT_AMPLITUDE),
        v_bias: det(g.kv_rows(), 9, WEIGHT_AMPLITUDE),
        o_bias: det(HIDDEN, 10, WEIGHT_AMPLITUDE),
        q_norm: det(g.head_dim, 11, WEIGHT_AMPLITUDE),
        k_norm: det(g.head_dim, 12, WEIGHT_AMPLITUDE),
        sinks: det(g.num_q, 13, SINK_AMPLITUDE),
        inv_freq: (0..g.head_dim / 2)
            .map(|i| THETA.powf(-2.0 * i as f64 / g.head_dim as f64) as f32)
            .collect(),
        h: det(ROWS * HIDDEN, 14, 1.0),
        k_cache: det(cache_positions * g.kv_rows(), 15, 1.0),
        v_cache: det(cache_positions * g.kv_rows(), 16, 1.0),
    }
}

/// An NVFP4 matrix resident as `(packed, scales)`.
fn lowered<'a>(
    (packed, scales): &'a (metal::Buffer, metal::Buffer),
    m: &nvfp4::Nvfp4Matrix,
) -> LoweredMatrix<'a> {
    LoweredMatrix::Nvfp4 {
        packed,
        packed_offset: 0,
        scales,
        scales_offset: 0,
        tensor_scale: m.tensor_scale,
    }
}

fn shape(c: &Case, position_index: usize) -> AttnShape {
    let full = c.variant == Variant::Full;
    AttnShape {
        hidden: HIDDEN,
        num_q_heads: c.geom.num_q,
        num_kv_heads: c.geom.num_kv,
        head_dim: c.geom.head_dim,
        norm_eps: EPS,
        norm_weight_offset: OFFSET,
        qk_norm_eps: QK_EPS,
        parameter_free_q: !full,
        parameter_free_k: !full,
        parameter_free_v: full,
        query_scale: (!full).then_some(QUERY_SCALE),
        score_scale: 1.0 / (c.geom.head_dim as f32).sqrt(),
        position: if full {
            LoweredPosition::Rope { theta: THETA }
        } else {
            LoweredPosition::None
        },
        window: c.window,
        softcap: full.then_some(SOFTCAP),
        position_index,
        kv_len: position_index + 1,
        residual_scale: full.then_some(RESIDUAL_SCALE),
    }
}

/// Everything one case produced, from both paths.
struct Outcome {
    /// `[ROWS, HIDDEN]` from the block, and from ROWS single positions.
    rows_out: Vec<f32>,
    each_out: Vec<f32>,
    /// Whole caches after each path.
    rows_k: Vec<f32>,
    rows_v: Vec<f32>,
    each_k: Vec<f32>,
    each_v: Vec<f32>,
    rows_witness: Witness,
    each_witness: Witness,
    /// Row 0 re-encoded against the block's caches with the WHOLE block
    /// visible (`kv_len = base + ROWS`) — what a non-causal block would
    /// have computed for it.
    row0_sees_block: Vec<f32>,
}

fn run_case(gpu: &MetalBackend, c: &Case) -> Outcome {
    let g = c.geom;
    let (q_rows, kv_rows) = (g.q_rows(), g.kv_rows());
    let cache_positions = c.base + ROWS;
    // Leaked: `lowering_weight` caches device copies on the host
    // allocation's (ptr, len), so weight bytes must never be freed and
    // their address reused by a later case's different weights.
    let op: &'static Operands = Box::leak(Box::new(operands(g, cache_positions)));
    let full = c.variant == Variant::Full;

    let weight = |m: &nvfp4::Nvfp4Matrix| {
        (
            gpu.lowering_weight(&m.packed),
            gpu.lowering_weight(&m.scales),
        )
    };
    let (q, k, v, o) = (weight(&op.q), weight(&op.k), weight(&op.v), weight(&op.o));
    let up = |x: &[f32]| gpu.lowering_upload(x).expect("upload");
    let (norm, post) = (up(&op.norm), up(&op.post));
    let (q_bias, k_bias, v_bias, o_bias) = (
        up(&op.q_bias),
        up(&op.k_bias),
        up(&op.v_bias),
        up(&op.o_bias),
    );
    let (q_norm, k_norm, sinks, inv_freq) = (
        up(&op.q_norm),
        up(&op.k_norm),
        up(&op.sinks),
        up(&op.inv_freq),
    );
    let post_single = gpu.lowering_scratch(HIDDEN);
    let w = AttnWeights {
        q: lowered(&q, &op.q),
        k: lowered(&k, &op.k),
        v: lowered(&v, &op.v),
        o: lowered(&o, &op.o),
        gate: None,
        q_bias: full.then_some(&q_bias),
        k_bias: full.then_some(&k_bias),
        v_bias: full.then_some(&v_bias),
        o_bias: full.then_some(&o_bias),
        sinks: full.then_some(&sinks),
        qk_norm: full.then_some(QkNormWeights {
            q: &q_norm,
            k: &k_norm,
            weight_offset: OFFSET,
        }),
        norm_weight: &norm,
        post_norm: full.then_some(PostNorm {
            weight: &post,
            eps: POST_EPS,
            weight_offset: OFFSET,
            scratch: &post_single,
        }),
    };

    // Split-K partials, sized as a caller would size them.
    let splitk_bufs = c.splitk_rows.map(|rows| {
        let (o_len, ml_len) = SplitKScratch::lens(rows, g.num_q, q_rows);
        (
            gpu.lowering_scratch(o_len),
            gpu.lowering_scratch(ml_len),
            rows,
        )
    });
    let splitk = splitk_bufs
        .as_ref()
        .map(|(o_part, ml_part, rows)| SplitKScratch {
            o_part,
            ml_part,
            rows: *rows,
            max_q_heads: g.num_q,
            max_q_rows: q_rows,
        });

    // ── the block ─────────────────────────────────────────────────
    let (rk, rv) = (up(&op.k_cache), up(&op.v_cache));
    let h_in = up(&op.h);
    let h_out = gpu.lowering_scratch(ROWS * HIDDEN);
    let (normed, q_buf, concat, attn_out, post_rows) = (
        gpu.lowering_scratch(ROWS * HIDDEN),
        gpu.lowering_scratch(ROWS * q_rows),
        gpu.lowering_scratch(ROWS * q_rows),
        gpu.lowering_scratch(ROWS * HIDDEN),
        gpu.lowering_scratch(ROWS * HIDDEN),
    );
    let (gate, gated) = (gpu.lowering_scratch(q_rows), gpu.lowering_scratch(q_rows));
    let rows_scratch = AttnScratch {
        normed: &normed,
        q: &q_buf,
        k_cache: &rk,
        v_cache: &rv,
        gate: &gate,
        concat: &concat,
        gated: &gated,
        attn_out: &attn_out,
        inv_freq: &inv_freq,
        splitk: splitk.as_ref(),
    };
    let before = witness();
    run(gpu, |enc| {
        gpu.encode_attention_rows(
            &mut SingleEncoder(enc),
            &h_in,
            &h_out,
            &w,
            &rows_scratch,
            &post_rows,
            &shape(c, c.base),
            ROWS,
        )
        .expect("the block encodes")
    });
    let rows_witness = witness().since(before);
    let rows_out = gpu
        .lowering_readback(&h_out, ROWS * HIDDEN)
        .expect("readback");
    let rows_k = gpu
        .lowering_readback(&rk, cache_positions * kv_rows)
        .expect("k");
    let rows_v = gpu
        .lowering_readback(&rv, cache_positions * kv_rows)
        .expect("v");

    // ── the same block, one position at a time ────────────────────
    let (ek, ev) = (up(&op.k_cache), up(&op.v_cache));
    let h_each: Vec<metal::Buffer> = (0..ROWS)
        .map(|i| up(&op.h[i * HIDDEN..(i + 1) * HIDDEN]))
        .collect();
    let out_each: Vec<metal::Buffer> = (0..ROWS).map(|_| gpu.lowering_scratch(HIDDEN)).collect();
    let (s_normed, s_q, s_concat, s_out) = (
        gpu.lowering_scratch(HIDDEN),
        gpu.lowering_scratch(q_rows),
        gpu.lowering_scratch(q_rows),
        gpu.lowering_scratch(HIDDEN),
    );
    let each_scratch = |k_cache, v_cache| AttnScratch {
        normed: &s_normed,
        q: &s_q,
        k_cache,
        v_cache,
        gate: &gate,
        concat: &s_concat,
        gated: &gated,
        attn_out: &s_out,
        inv_freq: &inv_freq,
        splitk: splitk.as_ref(),
    };
    let before = witness();
    run(gpu, |enc| {
        for i in 0..ROWS {
            gpu.encode_attention(
                &mut SingleEncoder(enc),
                &h_each[i],
                &out_each[i],
                &w,
                &each_scratch(&ek, &ev),
                &shape(c, c.base + i),
            );
        }
    });
    let each_witness = witness().since(before);
    let each_out: Vec<f32> = out_each
        .iter()
        .flat_map(|b| gpu.lowering_readback(b, HIDDEN).expect("readback"))
        .collect();
    let each_k = gpu
        .lowering_readback(&ek, cache_positions * kv_rows)
        .expect("k");
    let each_v = gpu
        .lowering_readback(&ev, cache_positions * kv_rows)
        .expect("v");

    // ── causality control: row 0 with the whole block visible ─────
    let control_out = gpu.lowering_scratch(HIDDEN);
    run(gpu, |enc| {
        gpu.encode_attention(
            &mut SingleEncoder(enc),
            &h_each[0],
            &control_out,
            &w,
            &each_scratch(&rk, &rv),
            &AttnShape {
                kv_len: c.base + ROWS,
                ..shape(c, c.base)
            },
        );
    });
    let row0_sees_block = gpu
        .lowering_readback(&control_out, HIDDEN)
        .expect("readback");

    Outcome {
        rows_out,
        each_out,
        rows_k,
        rows_v,
        each_k,
        each_v,
        rows_witness,
        each_witness,
        row0_sees_block,
    }
}

/// Parity of the block against the per-position path — outputs row by
/// row, the block's K/V slots, and an untouched cache prefix — plus the
/// causality control. Returns the worst output residual.
fn assert_block_matches_positions(c: &Case, out: &Outcome) -> f64 {
    let kv_rows = c.geom.kv_rows();
    assert!(
        out.rows_out.iter().all(|v| v.is_finite()),
        "{c:?}: non-finite block output"
    );
    let mut worst = 0.0f64;
    for i in 0..ROWS {
        let span = i * HIDDEN..(i + 1) * HIDDEN;
        let e = rel_rms(&out.each_out[span.clone()], &out.rows_out[span]);
        eprintln!("{c:?} row {i}: rel_rms {e:.3e}");
        assert!(
            e < PARITY,
            "{c:?}: row {i} off its single position: {e:.3e}"
        );
        worst = worst.max(e);
    }
    let prefix = c.base * kv_rows;
    for (name, rows, each) in [
        ("K", &out.rows_k, &out.each_k),
        ("V", &out.rows_v, &out.each_v),
    ] {
        assert!(
            rows[..prefix]
                .iter()
                .zip(&each[..prefix])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{c:?}: the block wrote {name} cache positions before its base"
        );
        let e = rel_rms(&each[prefix..], &rows[prefix..]);
        assert!(e < PARITY, "{c:?}: block {name} slots off: {e:.3e}");
    }
    let moved = rel_rms(&out.row0_sees_block, &out.rows_out[..HIDDEN]);
    eprintln!(
        "{c:?} causality control: {:.0}x parity",
        moved / worst.max(1e-12)
    );
    assert!(
        moved > worst * CAUSAL_CONTROL_OVER_PARITY && moved > PARITY,
        "{c:?}: row 0 attending the whole block moved it only {moved:.3e} \
         (parity {worst:.3e}) — the parity gate could not see a non-causal block"
    );
    worst
}

fn check(gpu: &MetalBackend, c: Case, rows_expected: Witness, each_expected: Witness) {
    let out = run_case(gpu, &c);
    assert_block_matches_positions(&c, &out);
    assert_eq!(
        out.rows_witness, rows_expected,
        "{c:?}: the block took the wrong attention arm"
    );
    assert_eq!(
        out.each_witness, each_expected,
        "{c:?}: the per-position reference took the wrong attention arm"
    );
}

const ONE_OP: u64 = 1;
const PER_ROW: u64 = ROWS as u64;

#[test]
fn serial_block_is_one_rows_dispatch_at_parity_with_each_position() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    for (window, variant) in [(None, Variant::Full), (Some(WINDOW), Variant::Plain)] {
        check(
            &gpu,
            Case {
                geom: UNMEASURED,
                base: SHORT_BASE,
                window,
                variant,
                splitk_rows: None,
            },
            Witness {
                serial: ONE_OP,
                ..Witness::default()
            },
            Witness {
                serial: PER_ROW,
                ..Witness::default()
            },
        );
    }
}

#[test]
fn seqpar_block_is_one_rows_dispatch_at_parity_with_each_position() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    for variant in [Variant::Full, Variant::Plain] {
        check(
            &gpu,
            Case {
                geom: GEMMA,
                base: SHORT_BASE,
                window: None,
                variant,
                splitk_rows: None,
            },
            Witness {
                seqpar: ONE_OP,
                ..Witness::default()
            },
            Witness {
                seqpar: PER_ROW,
                ..Witness::default()
            },
        );
    }
}

#[test]
fn splitk_block_is_one_splitk_op_at_parity_with_each_position() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // Full carries sinks (merge pass) and softcap (partial pass).
    for variant in [Variant::Full, Variant::Plain] {
        check(
            &gpu,
            Case {
                geom: GEMMA,
                base: SPLITK_BASE,
                window: None,
                variant,
                splitk_rows: Some(ROWS),
            },
            Witness {
                splitk: ONE_OP,
                ..Witness::default()
            },
            Witness {
                splitk: PER_ROW,
                ..Witness::default()
            },
        );
    }
}

#[test]
fn splitk_scratch_too_small_for_the_block_falls_back_to_seqpar_rows() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // Scratch for ONE position: every single-position op still splits,
    // the block does not fit and must take the seqpar rows kernel rather
    // than overrun the partial buffers.
    check(
        &gpu,
        Case {
            geom: GEMMA,
            base: SPLITK_BASE,
            window: None,
            variant: Variant::Full,
            splitk_rows: Some(1),
        },
        Witness {
            seqpar: ONE_OP,
            ..Witness::default()
        },
        Witness {
            splitk: PER_ROW,
            ..Witness::default()
        },
    );
}

#[test]
fn splitk_block_straddling_a_chunk_tier_falls_back_to_seqpar_rows() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // Rows below span 512 split into 8 chunks, rows from 512 into 16: one
    // split-K op cannot carry both, so the block takes seqpar rows while
    // each position still splits at its own chunk count.
    check(
        &gpu,
        Case {
            geom: GEMMA,
            base: TIER_512_BASE,
            window: None,
            variant: Variant::Plain,
            splitk_rows: Some(ROWS),
        },
        Witness {
            seqpar: ONE_OP,
            ..Witness::default()
        },
        Witness {
            splitk: PER_ROW,
            ..Witness::default()
        },
    );
}

#[test]
fn block_past_the_short_span_runs_the_long_kernel_per_row() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // No rows kernel holds more than the short kernel's scores, so the
    // block loops the per-position dispatch — the same kernel `step` runs.
    check(
        &gpu,
        Case {
            geom: GEMMA,
            base: LONG_BASE,
            window: None,
            variant: Variant::Plain,
            splitk_rows: None,
        },
        Witness {
            seqpar: PER_ROW,
            ..Witness::default()
        },
        Witness {
            seqpar: PER_ROW,
            ..Witness::default()
        },
    );
}

#[test]
fn block_straddling_a_seqpar_slice_tier_runs_per_row() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // gpt-oss's rows below span 512 take 8 slices and from 512 take 12;
    // one rows dispatch carries one threadgroup width, so it must loop.
    check(
        &gpu,
        Case {
            geom: GPT_OSS,
            base: TIER_512_BASE,
            window: None,
            variant: Variant::Plain,
            splitk_rows: None,
        },
        Witness {
            seqpar: PER_ROW,
            ..Witness::default()
        },
        Witness {
            seqpar: PER_ROW,
            ..Witness::default()
        },
    );
}

#[test]
fn attention_rows_refuses_the_output_gate_by_name() {
    let gpu = backend();
    let g = UNMEASURED;
    // Leaked for the same reason as `run_case`'s operands.
    let m: &'static nvfp4::Nvfp4Matrix = Box::leak(Box::new(
        nvfp4::quantize(
            &det(g.q_rows() * HIDDEN, 1, WEIGHT_AMPLITUDE),
            g.q_rows(),
            HIDDEN,
        )
        .expect("quantise"),
    ));
    let (packed, scales) = (
        gpu.lowering_weight(&m.packed),
        gpu.lowering_weight(&m.scales),
    );
    let mat = || LoweredMatrix::Nvfp4 {
        packed: &packed,
        packed_offset: 0,
        scales: &scales,
        scales_offset: 0,
        tensor_scale: m.tensor_scale,
    };
    let buf = gpu.lowering_scratch(ROWS * g.q_rows().max(HIDDEN));
    let w = AttnWeights {
        q: mat(),
        k: mat(),
        v: mat(),
        o: mat(),
        gate: Some(mat()),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        sinks: None,
        qk_norm: None,
        norm_weight: &buf,
        post_norm: None,
    };
    let s = AttnScratch {
        normed: &buf,
        q: &buf,
        k_cache: &buf,
        v_cache: &buf,
        gate: &buf,
        concat: &buf,
        gated: &buf,
        attn_out: &buf,
        inv_freq: &buf,
        splitk: None,
    };
    let c = Case {
        geom: g,
        base: SHORT_BASE,
        window: None,
        variant: Variant::Plain,
        splitk_rows: None,
    };
    let mut err = None;
    run(&gpu, |enc| {
        err = gpu
            .encode_attention_rows(
                &mut SingleEncoder(enc),
                &buf,
                &buf,
                &w,
                &s,
                &buf,
                &shape(&c, c.base),
                ROWS,
            )
            .err();
    });
    let err = err.expect("a gated attention op has no multi-position lowering");
    assert!(err.contains("output-gate"), "{err}");
}

#[test]
fn splitk_scratch_sizes_and_fits_the_ops_it_was_sized_for() {
    let gpu = backend();
    let g = GEMMA;
    let (o_len, ml_len) = SplitKScratch::lens(ROWS, g.num_q, g.q_rows());
    // Every chunk of every head of every row gets its own partial.
    assert_eq!(o_len, ROWS * g.q_rows() * SPLITK_MAX_CHUNKS);
    assert_eq!(ml_len, ml_part_len(ROWS, g.num_q, SPLITK_MAX_CHUNKS));
    let (o_part, ml_part) = (gpu.lowering_scratch(o_len), gpu.lowering_scratch(ml_len));
    let s = SplitKScratch {
        o_part: &o_part,
        ml_part: &ml_part,
        rows: ROWS,
        max_q_heads: g.num_q,
        max_q_rows: g.q_rows(),
    };
    assert!(s.fits(ROWS, g.num_q, g.q_rows()), "exactly the sized op");
    assert!(s.fits(1, g.num_q / 2, g.q_rows() / 2), "a smaller op");
    assert!(!s.fits(ROWS + 1, g.num_q, g.q_rows()), "one row too many");
    assert!(!s.fits(ROWS, g.num_q + 1, g.q_rows()), "one head too many");
    assert!(
        !s.fits(ROWS, g.num_q, g.q_rows() + 1),
        "one float too many per row"
    );
}
