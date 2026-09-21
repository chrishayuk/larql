//! G6b, first fragment: does the GPU-lowered gated FFN compute what the
//! interpreter's CPU-glue realisation computes?
//!
//! The lowered path moves the norm, the SiLU-GLU and the residual onto
//! the GPU, so unlike G6a's scheduling-only comparison the arithmetic
//! realisation genuinely changes and float reassociation is legitimate.
//! The bar is therefore a tolerance, judged in the same units the
//! production-parity work uses — max abs, relative RMS, cosine — not
//! bit equality.
//!
//! ## Controls
//!
//! Agreement alone would not show the lowering *read the plan*. A
//! lowering that ignored the norm epsilon, dropped the centred-norm
//! offset, or silently used the wrong activation would still produce
//! finite, plausible, nearly-correct numbers. So each judged fact gets a
//! negative arm that must break parity:
//!
//! - **norm weight offset** — Glimmer's centred convention (`1 + w`).
//!   Dropping it is the single likeliest silent lowering bug.
//! - **activation** — SiLU-GLU vs plain GLU.
//! - **residual** — present vs omitted.
//!
//! If a control does *not* break parity, the corresponding assertion in
//! the positive arm is vacuous and the test says so rather than passing.
//!
//! Control strength is judged **relative to the parity residual**, not
//! against an absolute constant. A fixed threshold is a guess about the
//! fixture: the residual control below moves rel_rms to 2.4e-3, which
//! looks small next to an arbitrary 1e-2 bar and is in fact 2500x the
//! 9.6e-7 the lowering itself achieves — overwhelmingly distinguishable.
//! What makes a control meaningful is that its effect dwarfs the noise
//! the positive arm tolerates, so that is what gets asserted.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::ffn::{FfnActivation, FfnScratch, FfnShape, FfnWeights};
use larql_compute_metal::lowering::profile::SingleEncoder;
use larql_compute_metal::lowering::LoweredMatrix;
use larql_models::quant::nvfp4;

const HIDDEN: usize = 512;
const INTER: usize = 1408;
const EPS: f32 = 1e-5;
/// Glimmer's centred-norm convention.
const NORM_OFFSET: f32 = 1.0;
/// Muse-Glimmer's post-block epsilon — three orders of magnitude below
/// the pre-block one. Reusing the pre-norm value here is the silent
/// four-norm bug this fixture exists to catch.
const POST_EPS: f32 = 1e-8;

fn deterministic(n: usize, seed: u32) -> Vec<f32> {
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

/// The reference: the same program on the CPU, in f32, written straight
/// from the plan's op order. Independent of the Metal code under test.
///
/// Takes every judged fact explicitly — including the two a control
/// perturbs — because a reference that hard-coded them could not model
/// the defects the controls exist to detect.
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    h: &[f32],
    norm_w: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    offset: f32,
    silu: bool,
    residual: bool,
    post: PostNormMode<'_>,
) -> Vec<f32> {
    let ms = h.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    let normed: Vec<f32> = h
        .iter()
        .zip(norm_w)
        .map(|(x, w)| x * inv * (offset + w))
        .collect();
    let mv = |m: &[f32], x: &[f32], n: usize, k: usize| -> Vec<f32> {
        (0..n)
            .map(|r| (0..k).map(|c| m[r * k + c] * x[c]).sum())
            .collect()
    };
    let g = mv(gate, &normed, INTER, HIDDEN);
    let u = mv(up, &normed, INTER, HIDDEN);
    let act: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(gv, uv)| {
            if silu {
                (gv / (1.0 + (-gv).exp())) * uv
            } else {
                gv * uv
            }
        })
        .collect();
    let d = mv(down, &act, HIDDEN, INTER);
    let rms_norm = |v: &[f32], w: &[f32], eps: f32| -> Vec<f32> {
        let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        v.iter()
            .zip(w)
            .map(|(x, wv)| x * inv * (1.0 + wv))
            .collect()
    };
    match post {
        // The judged shape: normalise the branch, then add.
        PostNormMode::BeforeResidual(w, eps) => {
            let n = rms_norm(&d, w, eps);
            h.iter().zip(&n).map(|(a, b)| a + b).collect()
        }
        // The plausible-but-wrong shape: add, then normalise the sum.
        PostNormMode::AfterResidual(w, eps) => {
            let summed: Vec<f32> = h.iter().zip(&d).map(|(a, b)| a + b).collect();
            rms_norm(&summed, w, eps)
        }
        PostNormMode::None if residual => h.iter().zip(&d).map(|(a, b)| a + b).collect(),
        PostNormMode::None => d,
    }
}

/// How (and whether) a post-block norm joins the residual stream.
#[derive(Clone, Copy)]
enum PostNormMode<'a> {
    /// Normalise the branch output, then add — the judged semantics.
    BeforeResidual(&'a [f32], f32),
    /// Add, then normalise the sum — what the name "post-FFN norm"
    /// could plausibly be read to mean, and a different model.
    AfterResidual(&'a [f32], f32),
    None,
}

/// How far a control must exceed the parity residual to demonstrate that
/// the positive arm could have detected the corresponding defect.
const CONTROL_MARGIN: f64 = 100.0;

struct Metrics {
    max_abs: f32,
    rel_rms: f64,
    cosine: f64,
}

fn compare(reference: &[f32], got: &[f32]) -> Metrics {
    let max_abs = reference
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (mut num, mut den, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        let (a, b) = (*a as f64, *b as f64);
        num += (a - b) * (a - b);
        den += a * a;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    Metrics {
        max_abs,
        rel_rms: (num / den).sqrt(),
        cosine: dot / (na.sqrt() * nb.sqrt()),
    }
}

/// A control must move the result far enough above the parity residual
/// that the positive arm would have caught the defect it models.
fn assert_control(what: &str, perturbed: &[f32], got: &[f32], parity_rel_rms: f64) {
    let c = compare(perturbed, got);
    let ratio = c.rel_rms / parity_rel_rms;
    eprintln!(
        "  control `{what}`: rel_rms {:.3e} = {ratio:.0}x the parity residual",
        c.rel_rms
    );
    assert!(
        ratio > CONTROL_MARGIN,
        "control `{what}` moves the result only {ratio:.1}x the parity residual          ({:.3e} vs {parity_rel_rms:.3e}) — the positive assertion cannot          distinguish this defect, so passing it proves nothing",
        c.rel_rms
    );
}

/// Run the lowered FFN once under SiLU-GLU and return its output.
#[allow(clippy::too_many_arguments)]
fn run_lowered(
    gpu: &larql_compute_metal::MetalBackend,
    h: &[f32],
    norm_w: &[f32],
    gate: &nvfp4::Nvfp4Matrix,
    up: &nvfp4::Nvfp4Matrix,
    down: &nvfp4::Nvfp4Matrix,
    offset: f32,
    post_norm_w: Option<&[f32]>,
    post_eps: f32,
) -> Vec<f32> {
    run_lowered_act(
        gpu,
        h,
        norm_w,
        gate,
        up,
        down,
        offset,
        post_norm_w,
        post_eps,
        FfnActivation::Silu,
    )
}

/// The same run, with the plan's gate COMBINE stated explicitly.
///
/// Split out so the SiTU arm can be driven through the identical
/// lowering — the point of that test is that only `shape.activation`
/// differs, so anything else varying between the arms would confound it.
#[allow(clippy::too_many_arguments)]
fn run_lowered_act(
    gpu: &larql_compute_metal::MetalBackend,
    h: &[f32],
    norm_w: &[f32],
    gate: &nvfp4::Nvfp4Matrix,
    up: &nvfp4::Nvfp4Matrix,
    down: &nvfp4::Nvfp4Matrix,
    offset: f32,
    post_norm_w: Option<&[f32]>,
    post_eps: f32,
    activation: FfnActivation,
) -> Vec<f32> {
    let h_in = gpu.lowering_upload(h).expect("upload");
    let norm_buf = gpu.lowering_upload(norm_w).expect("upload");
    let h_out = gpu.lowering_scratch(HIDDEN);
    let (normed, g, u, a, d) = (
        gpu.lowering_scratch(HIDDEN),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(HIDDEN),
    );
    let post_buf = post_norm_w.map(|w| gpu.lowering_upload(w).expect("upload"));
    let post_scratch = gpu.lowering_scratch(HIDDEN);
    let w = FfnWeights {
        gate: LoweredMatrix::Nvfp4 {
            packed: &gpu.lowering_weight(&gate.packed),
            packed_offset: 0,
            scales: &gpu.lowering_weight(&gate.scales),
            scales_offset: 0,
            tensor_scale: gate.tensor_scale,
        },
        up: LoweredMatrix::Nvfp4 {
            packed: &gpu.lowering_weight(&up.packed),
            packed_offset: 0,
            scales: &gpu.lowering_weight(&up.scales),
            scales_offset: 0,
            tensor_scale: up.tensor_scale,
        },
        down: LoweredMatrix::Nvfp4 {
            packed: &gpu.lowering_weight(&down.packed),
            packed_offset: 0,
            scales: &gpu.lowering_weight(&down.scales),
            scales_offset: 0,
            tensor_scale: down.tensor_scale,
        },
        norm_weight: &norm_buf,
        post_norm: post_buf
            .as_ref()
            .map(|b| larql_compute_metal::lowering::PostNorm {
                weight: b,
                eps: post_eps,
                weight_offset: NORM_OFFSET,
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
        norm_weight_offset: offset,
        activation,
    };

    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_gated_ffn(&mut SingleEncoder(enc), &h_in, &h_out, &w, &s, &shape);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out = gpu.lowering_readback(&h_out, HIDDEN).expect("readback");
    // `w` borrows the uploaded buffers; end its borrow before recycling.
    for b in [h_in, norm_buf, h_out, normed, g, u, a, d, post_scratch] {
        gpu.recycle_lowering_scratch(b);
    }
    if let Some(b) = post_buf {
        gpu.recycle_lowering_scratch(b);
    }
    out
}

#[test]
fn lowered_ffn_matches_the_cpu_program_and_reads_its_judged_facts() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let h = deterministic(HIDDEN, 1);
    let norm_w = deterministic(HIDDEN, 2);
    let gate_f = deterministic(INTER * HIDDEN, 3);
    let up_f = deterministic(INTER * HIDDEN, 4);
    let down_f = deterministic(HIDDEN * INTER, 5);

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();

    // The reference consumes the *quantised* weights, so the comparison
    // isolates the lowering from quantisation error — which Q2 already
    // measured separately and which would otherwise dominate here.
    let gate_q = nvfp4::round_trip(&gate_f, INTER, HIDDEN).unwrap();
    let up_q = nvfp4::round_trip(&up_f, INTER, HIDDEN).unwrap();
    let down_q = nvfp4::round_trip(&down_f, HIDDEN, INTER).unwrap();

    let post_w = deterministic(HIDDEN, 6);
    let reference = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::BeforeResidual(&post_w, POST_EPS),
    );
    let got = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        Some(&post_w),
        POST_EPS,
    );

    let m = compare(&reference, &got);
    eprintln!(
        "lowered FFN vs CPU program: max_abs {:.3e}  rel_rms {:.3e}  cosine {:.9}",
        m.max_abs, m.rel_rms, m.cosine
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "lowered FFN produced non-finite output"
    );
    assert!(
        m.rel_rms < 1e-4 && m.cosine > 0.999_999,
        "lowered FFN disagrees with its own program: rel_rms {:.3e}, cosine {:.9}",
        m.rel_rms,
        m.cosine
    );

    // ── Control 1: the centred-norm offset is read ──────────────────
    let no_offset = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        0.0,
        true,
        true,
        PostNormMode::BeforeResidual(&post_w, POST_EPS),
    );
    assert_control("centred-norm offset", &no_offset, &got, m.rel_rms);

    // ── Control 2: the activation is SiLU-GLU, not plain GLU ────────
    let plain_glu = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        false,
        true,
        PostNormMode::BeforeResidual(&post_w, POST_EPS),
    );
    assert_control("SiLU-GLU activation", &plain_glu, &got, m.rel_rms);

    // ── Control 3: the residual is applied ──────────────────────────
    let no_residual = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        false,
        PostNormMode::None,
    );
    assert_control("FFN residual", &no_residual, &got, m.rel_rms);

    // ── Control 4: the post-FFN norm exists at all ──────────────────
    let no_post = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::None,
    );
    assert_control("post-FFN norm omitted", &no_post, &got, m.rel_rms);

    // Control 5 (post-norm epsilon) is deliberately NOT asserted here.
    // At this fixture's magnitudes the branch output has mean-square
    // ~11, so `sqrt(ms + 1e-5)` and `sqrt(ms + 1e-8)` differ by ~4.4e-7
    // relative — *below* this lowering's own ~9e-7 parity residual. The
    // control would fire at 1.0x and prove nothing, which is a fact
    // about where epsilon is observable, not about the lowering.
    // `post_norm_epsilon_is_read_where_it_is_observable` tests it in the
    // regime where the distinction exists.

    // ── Control 6: normalise the branch, THEN add — not add then
    //    normalise the sum. "Post-FFN norm" reads both ways; only one
    //    is the interpreter's.
    let after = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::AfterResidual(&post_w, POST_EPS),
    );
    assert_control(
        "post-norm applied after the residual",
        &after,
        &got,
        m.rel_rms,
    );
}

/// The post-norm epsilon, tested where it is observable.
///
/// Epsilon only moves the result when it is a real fraction of the
/// branch output's mean-square. The main fixture's branch has ms ~11, so
/// 1e-5 and 1e-8 are indistinguishable there — below the lowering's own
/// error. Here the down projection is scaled down until ms ~1e-4, where
/// 1e-5 shifts the RMS by ~5% and the two epsilons are unmistakable.
///
/// GPU-vs-GPU on purpose: the question is whether the plan's epsilon is
/// *plumbed through* to the kernel, and comparing two lowered runs
/// answers exactly that without a reference in between.
/// Two-norm placement folds the residual add into the down-projection
/// write (A-5b rung 2a). Every other test here passes `Some(post_norm)`,
/// which takes the four-norm branch — so the fused path shipped
/// unexercised, and stayed that way when it grew byte offsets for the
/// packed-operand layout. That is the branch where dropping an offset
/// would compute a different matrix's rows with the residual added on
/// top: finite, plausible, wrong.
#[test]
fn two_norm_placement_folds_the_residual_into_the_down_write() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let h = deterministic(HIDDEN, 31);
    let norm_w = deterministic(HIDDEN, 32);
    let gate_f = deterministic(INTER * HIDDEN, 33);
    let up_f = deterministic(INTER * HIDDEN, 34);
    let down_f = deterministic(HIDDEN * INTER, 35);

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();

    // Reference consumes the QUANTISED weights, so this isolates the
    // lowering from representation error (measured separately in Q2).
    let gate_q = nvfp4::round_trip(&gate_f, INTER, HIDDEN).unwrap();
    let up_q = nvfp4::round_trip(&up_f, INTER, HIDDEN).unwrap();
    let down_q = nvfp4::round_trip(&down_f, HIDDEN, INTER).unwrap();
    // `run_lowered` takes no activation argument — the harness fixes SiLU,
    // so the reference must too. Passing `false` here compares against a
    // GELU program and reports a kernel divergence that is really a
    // fixture mistake (rel_rms 0.56 on the first run).
    let expect = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::None,
    );

    // post_norm = None is what selects the fused branch.
    let got = run_lowered(&gpu, &h, &norm_w, &gate, &up, &down, NORM_OFFSET, None, EPS);
    let m = compare(&expect, &got);
    assert!(
        m.rel_rms < 1e-4,
        "fused-residual down projection diverged: rel_rms {}, max_abs {}",
        m.rel_rms,
        m.max_abs
    );
    // The residual really is added, not dropped: without it the output
    // would be the branch alone, which differs from h by construction.
    assert!(
        got.iter().zip(&h).any(|(o, i)| (o - i).abs() > 1e-6),
        "output equals the input residual — the FFN branch was not added"
    );
}

#[test]
fn post_norm_epsilon_is_read_where_it_is_observable() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let h = deterministic(HIDDEN, 1);
    let norm_w = deterministic(HIDDEN, 2);
    let post_w = deterministic(HIDDEN, 6);
    let gate_f = deterministic(INTER * HIDDEN, 3);
    let up_f = deterministic(INTER * HIDDEN, 4);
    // Scaled so the branch output's mean-square lands near 1e-4.
    let down_f: Vec<f32> = deterministic(HIDDEN * INTER, 5)
        .iter()
        .map(|v| v * 3e-3)
        .collect();

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();

    let with_post = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        Some(&post_w),
        POST_EPS,
    );
    let with_pre = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        Some(&post_w),
        EPS,
    );

    // Both runs include the residual, which is identical between them and
    // dwarfs the branch; compare the branch contribution alone.
    //
    // Note the branch is measured *after* the post-norm, so its own
    // mean-square is ~1 by construction whatever the down projection's
    // scale — the magnitude that decides observability is the pre-norm
    // one, which is not visible from outside the encoder. The scaling of
    // `down_f` above is what puts it in range; the assertion below is on
    // the effect, not on a proxy for it.
    let branch_post: Vec<f32> = with_post.iter().zip(&h).map(|(a, b)| a - b).collect();
    let branch_pre: Vec<f32> = with_pre.iter().zip(&h).map(|(a, b)| a - b).collect();
    let m = compare(&branch_post, &branch_pre);
    // Judged against the lowering's own parity residual, as everywhere
    // else: ~9e-7 on this fixture.
    let ratio = m.rel_rms / 8.921e-7;
    eprintln!(
        "post-norm eps 1e-8 vs 1e-5: rel_rms {:.3e} = {ratio:.0}x the parity residual",
        m.rel_rms
    );
    assert!(
        ratio > CONTROL_MARGIN,
        "the plan's post-norm epsilon must reach the kernel: swapping 1e-8 for 1e-5 \
         moved the branch only {ratio:.1}x the parity residual ({:.3e})",
        m.rel_rms
    );
}

/// Does the dense-FFN lowering REACH its SiTU-GLU arm?
///
/// `tests/test_lowering_situ.rs` already qualifies the `situ_glu` kernel
/// against the scalar authority, but it binds `bind_situ_glu` directly.
/// Nothing there — or anywhere — drives
/// `lowering::ffn::encode_gate_up_act_from_normed`'s
/// `FfnActivation::SituGlu` arm, so a lowering that ignored the plan's
/// combine and ran `geglu_silu` would keep every one of those tests
/// green. A proven kernel nobody dispatches is not an executed kernel.
///
/// The witness is an analytic identity rather than a second transcription
/// of the reference. `situ_glu` computes
///
/// ```text
/// beta * tanh(g / beta) * sigmoid(g) * u        (linear_beta = None)
/// ```
///
/// and `beta * tanh(g / beta) -> g` as `beta` grows, so at a large beta
/// with the up branch uncapped the combine IS `silu(g) * u`. That gives
/// the arm a real external reference — the same `cpu_reference` the SiLU
/// parity test uses — with only `shape.activation` changed.
///
/// Paired, because the identity alone is satisfied by a lowering that
/// never reached the arm at all: at GLM/K3-scale parameters the SiTU
/// combine must MOVE the answer, far past the parity residual. One arm
/// says the lowering computes SiTU's formula; the other says it is not
/// quietly computing SiLU.
#[test]
fn the_lowering_dispatches_situ_glu_and_binds_its_parameters() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let h = deterministic(HIDDEN, 11);
    let norm_w = deterministic(HIDDEN, 12);
    let gate_f = deterministic(INTER * HIDDEN, 13);
    let up_f = deterministic(INTER * HIDDEN, 14);
    let down_f = deterministic(HIDDEN * INTER, 15);

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();
    let gate_q = nvfp4::round_trip(&gate_f, INTER, HIDDEN).unwrap();
    let up_q = nvfp4::round_trip(&up_f, INTER, HIDDEN).unwrap();
    let down_q = nvfp4::round_trip(&down_f, HIDDEN, INTER).unwrap();

    // `beta` large enough that `beta*tanh(g/beta)` is `g` to well inside
    // f32, small enough that `g/beta` is not flushed: the normed gate
    // here is O(1), so g/beta ~ 1e-4 and the cubic term ~1e-9 relative.
    const WIDE_BETA: f32 = 1.0e4;

    let reference = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::None,
    );
    let wide = run_lowered_act(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        None,
        EPS,
        FfnActivation::SituGlu {
            beta: WIDE_BETA,
            linear_beta: None,
        },
    );

    assert!(
        wide.iter().all(|v| v.is_finite()),
        "the SiTU lowering produced non-finite output"
    );
    let m = compare(&reference, &wide);
    eprintln!(
        "SiTU(beta={WIDE_BETA:.0e}, no cap) vs the SiLU program: max_abs {:.3e}  \
         rel_rms {:.3e}  cosine {:.9}",
        m.max_abs, m.rel_rms, m.cosine
    );
    assert!(
        m.rel_rms < 1e-4 && m.cosine > 0.999_999,
        "at a wide beta the SiTU combine must BE silu(g)*u, so the lowering \
         disagreeing here means it bound the kernel's parameters wrongly: \
         rel_rms {:.3e}, cosine {:.9}",
        m.rel_rms,
        m.cosine
    );

    // The pairing: at real parameters the arm must move the answer, or
    // the agreement above is equally consistent with the lowering having
    // ignored `FfnActivation::SituGlu` and run `geglu_silu`.
    let capped = run_lowered_act(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        None,
        EPS,
        FfnActivation::SituGlu {
            beta: 4.0,
            linear_beta: Some(25.0),
        },
    );
    assert!(
        capped.iter().all(|v| v.is_finite()),
        "the capped SiTU lowering produced non-finite output"
    );
    assert_control("SiTU softcap at K3 parameters", &capped, &wide, m.rel_rms);

    // And `linear_beta: None` is a DIFFERENT function from an infinite
    // bound, not a spelling of it — the flag the kernel binds separately.
    // Capping only the up branch, at the same beta, must move the answer
    // too, or `has_linear` is not reaching the shader.
    let up_capped = run_lowered_act(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        None,
        EPS,
        FfnActivation::SituGlu {
            beta: WIDE_BETA,
            linear_beta: Some(0.5),
        },
    );
    assert_control(
        "SiTU up-branch cap (has_linear)",
        &up_capped,
        &wide,
        m.rel_rms,
    );
}
