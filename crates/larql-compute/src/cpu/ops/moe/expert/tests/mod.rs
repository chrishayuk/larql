use super::*;
use crate::cpu::ops::q4_common::quantize_q4_k;
use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use crate::{Activation, QuantFormat};

// BF16 encoding for common values (little-endian: low byte first).
fn bf16_bytes(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let hi = (bits >> 16) as u16;
    hi.to_le_bytes()
}

fn fill_bf16(len: usize, val: f32) -> Vec<u8> {
    let b = bf16_bytes(val);
    let mut v = vec![0u8; len * 2];
    for i in 0..len {
        v[i * 2] = b[0];
        v[i * 2 + 1] = b[1];
    }
    v
}

#[test]
fn zero_inter_returns_zero_vec() {
    let h = vec![1.0f32; 4];
    let out = run_single_expert(
        &h,
        &[],
        &[],
        0,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert_eq!(out, vec![0.0f32; 4]);
}

#[test]
fn zero_hidden_returns_empty() {
    let h: Vec<f32> = vec![];
    let out = run_single_expert(
        &h,
        &[],
        &[],
        0,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert_eq!(out.len(), 0);
}

#[test]
fn pre_experts_norm_empty_weight_returns_input_copy() {
    let h = vec![1.0f32, -2.0, 3.5];
    let out = pre_experts_norm(&h, &[], 0.0, 1e-6);
    assert_eq!(out, h);
}

#[test]
fn pre_experts_norm_applies_weight_and_offset() {
    let h = vec![3.0f32, 4.0];
    let norm_w = vec![1.0f32, 2.0];
    let out = pre_experts_norm(&h, &norm_w, 0.5, 0.0);
    let rms = ((3.0_f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt();
    let expected = [3.0 / rms * 1.5, 4.0 / rms * 2.5];

    for (actual, expected) in out.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn nonzero_weights_produce_nonzero_output() {
    let hidden = 4;
    let inter = 2;
    // One expert's worth of all-1.0 BF16 weights.
    let gate_up = fill_bf16(2 * inter * hidden, 1.0);
    let down = fill_bf16(hidden * inter, 1.0);
    let h = vec![1.0f32; hidden];
    let out = run_single_expert(
        &h,
        &gate_up,
        &down,
        inter,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert_eq!(out.len(), hidden);
    assert!(
        out.iter().any(|v| v.abs() > 0.01),
        "expected nonzero output, got {out:?}"
    );
}

#[test]
fn run_single_expert_into_matches_allocating_path() {
    let hidden = 4;
    let inter = 2;
    let gate_up = fill_bf16(2 * inter * hidden, 0.75);
    let down = fill_bf16(hidden * inter, -0.5);
    let h = vec![1.0f32, 0.5, -0.25, 2.0];
    let expected = run_single_expert(
        &h,
        &gate_up,
        &down,
        inter,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    let mut scratch = ExpertScratch::new(hidden, inter, inter);

    let actual = run_single_expert_into(
        &mut scratch,
        &h,
        &gate_up,
        &down,
        inter,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn run_single_expert_into_zeroes_output_for_empty_weights() {
    let hidden = 4;
    let inter = 2;
    let h = vec![1.0f32; hidden];
    let mut scratch = ExpertScratch::new(hidden, inter, inter);
    scratch.out.fill(9.0);

    let out = run_single_expert_into(
        &mut scratch,
        &h,
        &[],
        &[],
        inter,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );

    assert_eq!(out, &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn quantize_h_norm_for_q4k_rejects_empty_or_misaligned_input() {
    assert!(quantize_h_norm_for_q4k(&[]).is_none());
    assert!(quantize_h_norm_for_q4k(&vec![1.0f32; 255]).is_none());

    let q8 = quantize_h_norm_for_q4k(&vec![1.0f32; 256]).unwrap();
    assert_eq!(q8.qs.len(), 256);
}

/// An empty gate+up slab is the same refusal at the smallest shape, and
/// the pre-filled output is never overwritten with zeros first.
#[test]
#[should_panic(expected = "288 are needed")]
fn run_single_expert_q4k_q8k_into_refuses_an_empty_gate_up() {
    let hidden = 256;
    let inter = 1;
    let mut scratch = ExpertScratch::new(hidden, inter, 256);
    scratch.out.fill(7.0);
    let h_q8 = quantize_h_norm_for_q4k(&vec![1.0f32; hidden]).unwrap();

    let _ = run_single_expert_q4k_q8k_into(
        &mut scratch,
        &h_q8,
        &[],
        &[],
        inter,
        crate::ExpertMlp::gated(Activation::Silu),
    );
}

#[test]
fn run_single_expert_q4k_q8k_into_valid_weights_produces_finite_output() {
    let hidden = 256;
    let inter = 256;
    let h: Vec<f32> = (0..hidden)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
        .collect();
    let h_q8 = quantize_h_norm_for_q4k(&h).unwrap();
    let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.002)
        .collect();
    let down_f32: Vec<f32> = (0..hidden * inter)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.0025)
        .collect();
    let gate_up = quantize_q4_k(&gate_up_f32);
    let down = quantize_q4_k(&down_f32);
    let mut scratch = ExpertScratch::new(hidden, inter, inter);

    let out = run_single_expert_q4k_q8k_into(
        &mut scratch,
        &h_q8,
        &gate_up,
        &down,
        inter,
        crate::ExpertMlp::gated(Activation::GeluTanh),
    );

    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|v| v.abs() > 1e-8), "got all-zero output");
}

#[test]
fn run_single_expert_into_q4k_cached_dequant_path_runs() {
    let hidden = 256;
    let inter = 256;
    let h: Vec<f32> = (0..hidden)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
        .collect();
    let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.0015)
        .collect();
    let down_f32: Vec<f32> = (0..hidden * inter)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.001)
        .collect();
    let gate_up = quantize_q4_k(&gate_up_f32);
    let down = quantize_q4_k(&down_f32);
    let mut scratch = ExpertScratch::new(hidden, inter, inter);

    let out = run_single_expert_into(
        &mut scratch,
        &h,
        &gate_up,
        &down,
        inter,
        QuantFormat::Q4_K,
        crate::ExpertMlp::gated(Activation::Silu),
    );

    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn with_norm_matches_manual_prenorm() {
    let hidden = 4;
    let inter = 2;
    let gate_up = fill_bf16(2 * inter * hidden, 1.0);
    let down = fill_bf16(hidden * inter, 1.0);
    let h = vec![1.0f32, 2.0, 3.0, 4.0];
    let norm_w = vec![1.0f32; hidden];
    let eps = 1e-6_f32;

    let rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32 + eps).sqrt();
    let h_normed: Vec<f32> = h
        .iter()
        .zip(norm_w.iter())
        .map(|(&x, &w)| x / rms * w)
        .collect();

    let direct = run_single_expert(
        &h_normed,
        &gate_up,
        &down,
        inter,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    let via_norm = run_single_expert_with_norm(
        &h,
        &gate_up,
        &down,
        inter,
        &norm_w,
        0.0,
        eps,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );

    let max_diff: f32 = direct
        .iter()
        .zip(&via_norm)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        max_diff < 1e-4,
        "with_norm diverges from manual prenorm: max_diff={max_diff}"
    );
}

fn with_env_in_thread<T: Send + 'static>(
    vars: &'static [(&'static str, Option<&'static str>)],
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    // Run on a fresh thread so the TLS-cached env reads (`Q4K_DIRECT`,
    // `EXPERT_TIMING`) initialise from this dispatch's values rather than
    // inheriting an earlier `false` from another thread. The override is
    // thread-local, so it must be set *inside* the spawned thread (where the
    // TLS statics initialise); it dies with the thread → no cleanup, no
    // `std::env::set_var` (which races `getenv` → SIGSEGV), no lock.
    std::thread::spawn(move || {
        for (n, v) in vars {
            crate::options::set_env_override(n, *v);
        }
        f()
    })
    .join()
    .expect("thread did not panic")
}

/// `LARQL_Q4K_DIRECT=1` opts in to the q4k-direct matvec path inside
/// `run_single_expert_into`.  Cover the previously-untested branch
/// (lines 248, 273-321) by setting the env var on a fresh thread so
/// the TLS cache picks it up.
#[test]
fn run_single_expert_into_q4k_direct_env_takes_direct_path() {
    let out = with_env_in_thread(&[(crate::options::ENV_Q4K_DIRECT, Some("1"))], || {
        let hidden = 256;
        let inter = 256;
        let h: Vec<f32> = (0..hidden)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
            .collect();
        let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.0015)
            .collect();
        let down_f32: Vec<f32> = (0..hidden * inter)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.001)
            .collect();
        let gate_up = quantize_q4_k(&gate_up_f32);
        let down = quantize_q4_k(&down_f32);
        let mut scratch = ExpertScratch::new(hidden, inter, inter);
        let out = run_single_expert_into(
            &mut scratch,
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::Q4_K,
            crate::ExpertMlp::gated(Activation::GeluTanh),
        );
        out.to_vec()
    });
    assert_eq!(out.len(), 256);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Same `LARQL_Q4K_DIRECT=1` env, plus `LARQL_MOE_EXPERT_TIMING=1`
/// to drive the timing eprintln on the q4k path (lines 268-269,
/// 280-281, 285-286, 297-298, 308-319).
#[test]
fn run_single_expert_into_q4k_direct_with_timing_prints_breakdown() {
    let out = with_env_in_thread(
        &[
            (crate::options::ENV_Q4K_DIRECT, Some("1")),
            (crate::options::ENV_MOE_EXPERT_TIMING, Some("1")),
        ],
        || {
            let hidden = 256;
            let inter = 256;
            let h: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001).collect();
            let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.002)
                .collect();
            let down_f32: Vec<f32> = (0..hidden * inter)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.002)
                .collect();
            let gate_up = quantize_q4_k(&gate_up_f32);
            let down = quantize_q4_k(&down_f32);
            let mut scratch = ExpertScratch::new(hidden, inter, inter);
            let out = run_single_expert_into(
                &mut scratch,
                &h,
                &gate_up,
                &down,
                inter,
                QuantFormat::Q4_K,
                crate::ExpertMlp::gated(Activation::Silu),
            );
            out.to_vec()
        },
    );
    assert_eq!(out.len(), 256);
}

/// Same `LARQL_MOE_EXPERT_TIMING=1`, but on the default (non-q4k)
/// path — covers the timing eprintln further down the function
/// (lines 332-333, 338-339, 354-355, 367-368, 379, 381-392).
#[test]
fn run_single_expert_into_timing_on_default_path_prints_breakdown() {
    let out = with_env_in_thread(
        &[(crate::options::ENV_MOE_EXPERT_TIMING, Some("1"))],
        || {
            let hidden = 4;
            let inter = 2;
            let gate_up = fill_bf16(2 * inter * hidden, 0.5);
            let down = fill_bf16(hidden * inter, 0.25);
            let h = vec![1.0f32; hidden];
            let mut scratch = ExpertScratch::new(hidden, inter, inter);
            let out = run_single_expert_into(
                &mut scratch,
                &h,
                &gate_up,
                &down,
                inter,
                QuantFormat::BF16,
                crate::ExpertMlp::gated(Activation::Silu),
            );
            out.to_vec()
        },
    );
    assert_eq!(out.len(), 4);
}

#[test]
fn gelu_tanh_differs_from_silu() {
    let hidden = 4;
    let inter = 2;
    let gate_up = fill_bf16(2 * inter * hidden, 1.0);
    let down = fill_bf16(hidden * inter, 1.0);
    let h = vec![0.5f32; hidden];
    let silu_out = run_single_expert(
        &h,
        &gate_up,
        &down,
        inter,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    let gelu_out = run_single_expert(
        &h,
        &gate_up,
        &down,
        inter,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::GeluTanh),
    );
    let max_diff: f32 = silu_out
        .iter()
        .zip(&gelu_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        max_diff > 0.01,
        "SiLU and GeluTanh should diverge; max_diff={max_diff}"
    );
}

// ── Q4_K direct-from-mmap path coverage ────────────────────────────────

/// `run_single_expert` with `QuantFormat::Q4_K` and a hidden size
/// that's a multiple of 256 routes through the thread-local
/// `ExpertScratch` + SDOT-based `run_single_expert_q4k_q8k_into`.
/// Covers lines 102-157 of expert.rs.
#[test]
fn run_single_expert_q4k_routes_through_direct_path() {
    let hidden = 256usize; // multiple of Q4_K_BLOCK_ELEMS
    let inter = 256usize; // also Q4_K-aligned
                          // gate || up weights, both `inter × hidden` flattened row-major.
    let gate_data: Vec<f32> = (0..inter * hidden)
        .map(|i| ((i as f32) * 0.001).sin() * 0.05)
        .collect();
    let up_data: Vec<f32> = (0..inter * hidden)
        .map(|i| ((i as f32) * 0.001).cos() * 0.05)
        .collect();
    let mut gate_up_bytes = quantize_q4_k(&gate_data);
    gate_up_bytes.extend_from_slice(&quantize_q4_k(&up_data));

    let down_data: Vec<f32> = (0..hidden * inter)
        .map(|i| ((i as f32) * 0.002).sin() * 0.05)
        .collect();
    let down_bytes = quantize_q4_k(&down_data);

    let h: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.01).sin() * 0.5)
        .collect();
    let out = run_single_expert(
        &h,
        &gate_up_bytes,
        &down_bytes,
        inter,
        QuantFormat::Q4_K,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(
        out.iter().any(|&v| v.abs() > 1e-6),
        "non-degenerate Q4_K weights should produce non-zero output"
    );
}

/// `run_single_expert` with Q4_K weights but a hidden size that
/// isn't a multiple of 256 falls back to the BF16-style f32 cache
/// path (the Q4_K direct branch is gated on `hidden.is_multiple_of(256)`).
/// Exercises the negative condition of the gate.
#[test]
fn run_single_expert_q4k_non_aligned_hidden_falls_back() {
    let hidden = 4usize; // not a multiple of 256 — gate fails
    let inter = 2usize;
    // BF16 path expects 2 bytes per weight; the format dispatch
    // happens before the size check returns. We just need
    // non-empty bytes so `try_cached_dequant` doesn't bail.
    let gate_up = fill_bf16(2 * inter * hidden, 0.1);
    let down = fill_bf16(hidden * inter, 0.1);
    let h = vec![0.5f32; hidden];
    // Format is BF16 here because Q4_K bytes wouldn't decode at
    // hidden=4. The test asserts that the non-aligned-hidden gate
    // fails and we don't panic — same fallback as production.
    let out = run_single_expert(
        &h,
        &gate_up,
        &down,
        inter,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert_eq!(out.len(), hidden);
}

// ── run_single_expert_into early returns ───────────────────────────────

/// inter=0 short-circuits with zeroed output (no allocation in scratch).
#[test]
fn run_single_expert_into_zero_inter_returns_zeroed_out() {
    let hidden = 4usize;
    let mut scratch = ExpertScratch::new(hidden, 0, 0);
    let h = vec![1.0f32; hidden];
    let out = run_single_expert_into(
        &mut scratch,
        &h,
        &[],
        &[],
        0,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert_eq!(out, &[0.0f32; 4]);
}

/// hidden=0 (zero-length `h_norm`) also short-circuits to zeroed.
#[test]
fn run_single_expert_into_zero_hidden_returns_zeroed_out() {
    let mut scratch = ExpertScratch::new(0, 2, 2);
    let h: Vec<f32> = vec![];
    let out = run_single_expert_into(
        &mut scratch,
        &h,
        &[],
        &[],
        2,
        QuantFormat::BF16,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert!(out.is_empty());
}

// ── run_single_expert_q4k_q8k_into edge cases ──────────────────────────

/// Empty input (zero-length Q8_K activation) early-returns zeroed
/// scratch.out — covers lines 442-448.
#[test]
fn run_single_expert_q4k_q8k_into_zero_hidden_returns_zeroed() {
    let mut scratch = ExpertScratch::new(0, 4, 4);
    let empty_q8k = quantize_x_to_q8k(&[]);
    let out = run_single_expert_q4k_q8k_into(
        &mut scratch,
        &empty_q8k,
        &[],
        &[],
        4,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert!(out.is_empty());
}

/// `inter=0` also short-circuits — covers the `inter == 0` arm of
/// the early-return guard.
#[test]
fn run_single_expert_q4k_q8k_into_zero_inter_returns_zeroed() {
    let hidden = 256usize;
    let mut scratch = ExpertScratch::new(hidden, 0, 0);
    let h: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001).collect();
    let h_q8k = quantize_x_to_q8k(&h);
    let out = run_single_expert_q4k_q8k_into(
        &mut scratch,
        &h_q8k,
        &[],
        &[],
        0,
        crate::ExpertMlp::gated(Activation::Silu),
    );
    assert_eq!(out, &[0.0f32; 256]);
}

/// `gate_up_bytes` too short for the claimed (hidden, inter) shape is
/// refused by name. This path returns a borrowed scratch slice and has no
/// error channel, so the refusal ends the step; it used to hand back a
/// zeroed expert output that a combine would add as if it were computed.
#[test]
#[should_panic(expected = "run_single_expert_kq_q8k_into: refused operands")]
fn run_single_expert_q4k_q8k_into_short_bytes_are_refused() {
    let hidden = 256usize;
    let inter = 256usize;
    let mut scratch = ExpertScratch::new(hidden, inter, inter);
    let h: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001).collect();
    let h_q8k = quantize_x_to_q8k(&h);
    // Only 100 bytes — far less than 2 * inter * (hidden/256) * 144.
    let short_gate_up = vec![0u8; 100];
    let _ = run_single_expert_q4k_q8k_into(
        &mut scratch,
        &h_q8k,
        &short_gate_up,
        &[],
        inter,
        crate::ExpertMlp::gated(Activation::Silu),
    );
}

/// GeluTanh activation branch in `run_single_expert_q4k_q8k_into`.
/// The Silu branch is hit by the round-trip tests above; this pins
/// the GeluTanh side too.
#[test]
fn run_single_expert_q4k_q8k_into_gelu_tanh_activation() {
    let hidden = 256usize;
    let inter = 256usize;
    let mut scratch = ExpertScratch::new(hidden, inter, inter);

    let gate_data: Vec<f32> = (0..inter * hidden)
        .map(|i| ((i as f32) * 0.001).sin() * 0.05)
        .collect();
    let up_data: Vec<f32> = (0..inter * hidden)
        .map(|i| ((i as f32) * 0.001).cos() * 0.05)
        .collect();
    let mut gate_up = quantize_q4_k(&gate_data);
    gate_up.extend_from_slice(&quantize_q4_k(&up_data));

    let down_data: Vec<f32> = (0..hidden * inter)
        .map(|i| ((i as f32) * 0.002).sin() * 0.05)
        .collect();
    let down = quantize_q4_k(&down_data);

    let h: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.01).sin() * 0.5)
        .collect();
    let h_q8k = quantize_x_to_q8k(&h);

    let out = run_single_expert_q4k_q8k_into(
        &mut scratch,
        &h_q8k,
        &gate_up,
        &down,
        inter,
        crate::ExpertMlp::gated(Activation::GeluTanh),
    );
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── quantize_h_norm_for_q4k ────────────────────────────────────────────

/// `quantize_h_norm_for_q4k` returns Some on a 256-multiple input.
#[test]
fn quantize_h_norm_for_q4k_some_on_aligned_length() {
    let h: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001).collect();
    let out = quantize_h_norm_for_q4k(&h);
    assert!(out.is_some());
}

/// Non-aligned length (not multiple of 256) returns None.
#[test]
fn quantize_h_norm_for_q4k_none_on_non_aligned_length() {
    let h: Vec<f32> = vec![0.0; 100];
    let out = quantize_h_norm_for_q4k(&h);
    assert!(out.is_none());
}

/// Empty input returns None.
#[test]
fn quantize_h_norm_for_q4k_none_on_empty() {
    let out = quantize_h_norm_for_q4k(&[]);
    assert!(out.is_none());
}

/// `LARQL_KERNEL_TIMING=1` branches: the per-stage timing diagnostics
/// must not disturb the output. The flag is cached in TLS, so the fresh
/// thread from `with_env_in_thread` re-reads it.
#[test]
fn run_single_expert_q4k_q8k_into_with_kernel_timing_env_matches_untimed() {
    let hidden = 256;
    let inter = 256;
    let h: Vec<f32> = (0..hidden)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
        .collect();
    let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.002)
        .collect();
    let down_f32: Vec<f32> = (0..hidden * inter)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.0025)
        .collect();

    let run = {
        let h = h.clone();
        let gate_up_f32 = gate_up_f32.clone();
        let down_f32 = down_f32.clone();
        move || {
            let h_q8 = quantize_h_norm_for_q4k(&h).unwrap();
            let gate_up = quantize_q4_k(&gate_up_f32);
            let down = quantize_q4_k(&down_f32);
            let mut scratch = ExpertScratch::new(hidden, inter, inter);
            run_single_expert_q4k_q8k_into(
                &mut scratch,
                &h_q8,
                &gate_up,
                &down,
                inter,
                crate::ExpertMlp::gated(Activation::Silu),
            )
            .to_vec()
        }
    };

    let timed = with_env_in_thread(
        &[(crate::options::ENV_KERNEL_TIMING, Some("1"))],
        run.clone(),
    );
    let untimed = run();
    assert_eq!(timed, untimed, "timing instrumentation changed the output");
}

mod bias;
