//! K3-ACT-1: does the Metal `situ_glu` kernel compute what
//! `larql_compute::MoeGateRule::combine` computes?
//!
//! The CPU rule is the arithmetic authority and is itself pinned against
//! a Torch oracle transcribed from Kimi-K3's own `SituAndMul.forward`
//! (`ffn/expert_weight/tests/situ_parity.rs`). This test closes the other
//! half: the GPU transcription of that same expression, so the two tiers
//! cannot drift.
//!
//! ## Controls
//!
//! Agreement alone would not show the kernel READ its parameters. A
//! kernel that ignored `beta`, ignored `linear_beta`, or dropped the
//! `has_linear` flag would still produce finite, plausible, monotone
//! numbers on the same inputs. So each judged fact gets a negative arm
//! that must break agreement by far more than the parity residual:
//!
//! - **beta** — the gate softcap, 4.0 against 25.0.
//! - **linear_beta present vs absent** — a different function, not a
//!   larger bound.
//! - **plain SwiGLU** — the substitution this whole rung removes.
//!
//! Inputs deliberately straddle the softcap: `tanh(g/4)` saturates past
//! |g| ~ 12 and a fixture living only there would make every control read
//! zero while the test passed.

#![cfg(target_os = "macos")]

use larql_compute::MoeGateRule;
use larql_compute_metal::kernels::ffn::bind_situ_glu;
use larql_compute_metal::MetalBackend;

/// Kimi-K3's declared parameters (`config.json`, `text_config`).
const BETA: f32 = 4.0;
const LINEAR_BETA: f32 = 25.0;

/// Enough elements to cross a threadgroup boundary, so the bounds guard
/// in the kernel is exercised rather than assumed.
const N: usize = 1029;

/// Max absolute disagreement admitted between the two tiers. Both compute
/// the same expression in f32; only `tanh`/`exp` differ between Metal's
/// libm and Rust's, at the last ulp of values whose magnitude is bounded
/// by `beta * linear_beta`.
const TOLERANCE: f32 = 1e-4;

/// Gate values spanning the near-linear, transition and saturated bands
/// of `tanh(g / beta)`, on both signs.
fn gates(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.037).sin() * 14.0) + ((i % 7) as f32 - 3.0))
        .collect()
}

/// Up values reaching well past `linear_beta`, so its softcap does work.
/// Deliberately NOT the gate sequence: a symmetric fixture cannot see a
/// branch swap.
fn ups(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.019).cos() * 45.0) - ((i % 5) as f32))
        .collect()
}

/// Run `situ_glu` on the GPU for one set of parameters.
fn run_gpu(
    gpu: &MetalBackend,
    gate: &[f32],
    up: &[f32],
    beta: f32,
    linear: Option<f32>,
) -> Vec<f32> {
    let g = gpu.lowering_upload(gate).expect("upload gate");
    let u = gpu.lowering_upload(up).expect("upload up");
    let out = gpu.lowering_scratch(gate.len());

    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let pipeline = &gpu.ffn.situ_glu_pipeline;
    bind_situ_glu(
        enc,
        pipeline,
        (&g, 0),
        (&u, 0),
        (&out, 0),
        gate.len() as u32,
        beta,
        linear,
        false,
    );
    larql_compute_metal::lowering::dispatch_linear(enc, pipeline, gate.len());
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let read = gpu.lowering_readback(&out, gate.len()).expect("readback");
    for b in [g, u, out] {
        gpu.recycle_lowering_scratch(b);
    }
    read
}

fn cpu(gate: &[f32], up: &[f32], rule: MoeGateRule) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| rule.combine(*g, *u))
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn the_metal_kernel_agrees_with_the_scalar_authority_on_both_arms() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (gate, up) = (gates(N), ups(N));

    for (name, linear) in [("k3", Some(LINEAR_BETA)), ("no_linear_cap", None)] {
        let want = cpu(
            &gate,
            &up,
            MoeGateRule::SituGlu {
                beta: BETA,
                linear_beta: linear,
            },
        );
        let got = run_gpu(&gpu, &gate, &up, BETA, linear);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        let delta = max_abs(&got, &want);
        assert!(
            delta <= TOLERANCE * scale,
            "arm {name}: max abs {delta} exceeds {TOLERANCE} x {scale}"
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "arm {name}: the kernel produced a non-finite value"
        );
    }
}

/// The kernel reads `beta`, `linear_beta` and `has_linear` — each is
/// shown to change the answer by far more than the parity residual.
///
/// Without this the parity test above is satisfiable by a kernel that
/// ignores all three and happens to be close on this fixture.
#[test]
fn each_parameter_the_kernel_binds_changes_its_answer() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (gate, up) = (gates(N), ups(N));
    let reference = run_gpu(&gpu, &gate, &up, BETA, Some(LINEAR_BETA));
    let scale = reference
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(1.0);
    let residual = TOLERANCE * scale;

    // `beta` is read: the gate softcap at 25 is a different function
    // from the one at 4.
    let other_beta = run_gpu(&gpu, &gate, &up, LINEAR_BETA, Some(LINEAR_BETA));
    assert!(
        max_abs(&other_beta, &reference) > residual * 100.0,
        "changing beta moved the kernel's output by only {}",
        max_abs(&other_beta, &reference)
    );

    // `has_linear` is read: absence is a different function, not a
    // larger bound.
    let no_linear = run_gpu(&gpu, &gate, &up, BETA, None);
    assert!(
        max_abs(&no_linear, &reference) > residual * 100.0,
        "dropping linear_beta moved the kernel's output by only {}",
        max_abs(&no_linear, &reference)
    );

    // `linear_beta`'s VALUE is read, not just its presence.
    let other_linear = run_gpu(&gpu, &gate, &up, BETA, Some(3.0));
    assert!(
        max_abs(&other_linear, &reference) > residual * 100.0,
        "changing linear_beta moved the kernel's output by only {}",
        max_abs(&other_linear, &reference)
    );

    // And the substitution the rung removes: plain SwiGLU on the same
    // inputs is nowhere near SiTU at these parameters.
    let swiglu = cpu(
        &gate,
        &up,
        MoeGateRule::Gated(larql_compute::Activation::Silu),
    );
    assert!(
        max_abs(&swiglu, &reference) > residual * 100.0,
        "SwiGLU and SiTU differ by only {} on this fixture — it is in the wrong regime to \
         witness the substitution",
        max_abs(&swiglu, &reference)
    );
}
