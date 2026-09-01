//! The TTFA prefill prediction — the falsifiable experiment behind the
//! "<500 ms first-turn TTFA" gate (`docs/tts-funnel.md`).
//!
//! Step 5 measured TTFA at ~2.0 s, almost entirely the 343-row prefill,
//! and the Q4 speech prefill is a *row-oriented* path: per activation
//! row, quantise to Q8K and re-walk the packed weights with the decode
//! matvec. The decode kernel is compute/issue-bound (not
//! bandwidth-bound — `docs/q4k-decode-kernel.md`), so the candidate
//! that should win is the one that amortises the per-super-block
//! decode across the sequence — and it already exists:
//! `q4k_matmul_into`, the amortised matmul the vindex serving path
//! prefills through. The speech path simply never wired it in.
//!
//! Method: real checkpoint backbone matrices (28 layers × 7
//! projections), a fixture-sized activation block, best-of-`REPS` for
//! each candidate over the whole stack:
//!
//! 1. row path — what `--speak --q4` does at prefill today;
//! 2. amortised Q4 matmul — decode each super-block once, FMA across
//!    all rows (`q4_common::matmul`);
//! 3. fp32 BLAS GEMM — the fp32-resident oracle (what a no-quant build
//!    pays);
//! 4. dequantise-whole-matrix + fp32 GEMM — for the record, the
//!    approach the matmul's doc comment argues against.
//!
//! The prediction substitutes each candidate's GEMM total into the
//! measured prefill band. If the amortised matmul does not move
//! prefill decisively toward the gate, the lever is Metal, not CPU
//! wiring — either way we learn before implementing.
//!
//! NOT FOR CI (needs the checkpoint, and it is a benchmark):
//!
//! ```text
//! MOSS_TTS_REALTIME_DIR=<snapshot> \
//! cargo test --release -p larql-kv --test moss_prefill_bench -- --ignored --nocapture
//! ```

mod common;

use std::time::Instant;

use larql_compute::cpu::ops::q4_common::{dequantize_q4_k, q4k_matmul_into, quantize_q4_k};
use larql_compute::cpu::ops::q4k_q8k_dot::{q4k_q8k_gemm_into, q4k_q8k_matvec_parallel};
use larql_compute::quantize_x_to_q8k;
use ndarray::Array2;

use common::model_dir_from_env;
use larql_inference::larql_models::loading::safetensors::load_model_dir;

/// The parity fixture's prefill length — 343 rows (system + voice-clone
/// splice + assistant header + 12-token lead).
const SEQ: usize = 343;
/// Best-of repetitions per candidate.
const REPS: usize = 3;
/// The registry tag the row-path kernel dispatch expects.
const Q4K_FORMAT_TAG: &str = "Q4_K";
/// The step-5 measured Q4 prefill on the clean CPU build, for the
/// prediction context (`docs/tts-funnel.md` step-5 gate log).
const MEASURED_PREFILL_SECONDS: f64 = 1.97;

/// One backbone projection: packed bytes + the fp32 original.
struct PrefillOp {
    packed: Vec<u8>,
    f32_matrix: Array2<f32>,
    rows: usize,
    cols: usize,
}

#[test]
#[ignore]
fn q4_prefill_prediction() {
    let model_dir = model_dir_from_env();
    let weights = load_model_dir(&model_dir).expect("backbone load");

    // ── The prefill op mix: every backbone projection, once ──
    let arch = &weights.arch;
    let mut ops: Vec<PrefillOp> = Vec::new();
    for layer in 0..weights.num_layers {
        for key in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            let matrix = weights.tensors.get(&key).expect(&key).to_owned();
            let (rows, cols) = matrix.dim();
            let data = matrix.as_standard_layout().as_slice().unwrap().to_vec();
            ops.push(PrefillOp {
                packed: quantize_q4_k(&data),
                f32_matrix: matrix.to_owned().into_dimensionality().unwrap(),
                rows,
                cols,
            });
        }
    }
    let flops: f64 = ops
        .iter()
        .map(|op| 2.0 * op.rows as f64 * op.cols as f64 * SEQ as f64)
        .sum();
    println!(
        "prefill op mix: {} matrices, seq {SEQ}, {:.2} TFLOP total",
        ops.len(),
        flops / 1e12
    );

    // Activation blocks per op shape, deterministic values.
    let activations: Vec<Array2<f32>> = ops
        .iter()
        .map(|op| {
            Array2::from_shape_fn((SEQ, op.cols), |(row, col)| {
                ((row * 31 + col * 7) % 23) as f32 * 0.01 - 0.11
            })
        })
        .collect();

    // ── 1. Row path: today's speech prefill ──
    let row_path = best_seconds(|| {
        let mut sink = 0.0f32;
        for (op, x) in ops.iter().zip(&activations) {
            let mut out_row = vec![0.0f32; op.rows];
            for row in x.rows() {
                let q8 = quantize_x_to_q8k(row.as_slice().unwrap());
                q4k_q8k_matvec_parallel(
                    &mut out_row,
                    &q8,
                    &op.packed,
                    op.rows,
                    op.cols,
                    Q4K_FORMAT_TAG,
                )
                .expect("bench shapes are valid");
                sink += out_row[0];
            }
        }
        sink
    });

    // ── 2. Amortised Q4 matmul: decode once, FMA across the sequence ──
    let amortised = best_seconds(|| {
        let mut sink = 0.0f32;
        for (op, x) in ops.iter().zip(&activations) {
            let mut out = vec![0.0f32; SEQ * op.rows];
            q4k_matmul_into(
                &mut out,
                x.as_slice().unwrap(),
                &op.packed,
                op.rows,
                op.cols,
                SEQ,
            );
            sink += out[0];
        }
        sink
    });

    // ── 2b. Blocked Q4K×Q8K GEMM: one dispatch per matrix, L1-hot
    // weight rows, activations quantised once (the granularity fix the
    // row path's ~67k fork/joins demanded) ──
    let blocked = best_seconds(|| {
        let mut sink = 0.0f32;
        for (op, x) in ops.iter().zip(&activations) {
            let q8k: Vec<_> = x
                .rows()
                .into_iter()
                .map(|row| quantize_x_to_q8k(row.as_slice().unwrap()))
                .collect();
            let mut out = vec![0.0f32; SEQ * op.rows];
            q4k_q8k_gemm_into(&mut out, &q8k, &op.packed, op.rows, op.cols);
            sink += out[0];
        }
        sink
    });

    // ── 3. fp32 BLAS GEMM: the fp32-resident oracle ──
    let f32_gemm = best_seconds(|| {
        let mut sink = 0.0f32;
        for (op, x) in ops.iter().zip(&activations) {
            let y = x.dot(&op.f32_matrix.t());
            sink += y[[0, 0]];
        }
        sink
    });

    // ── 4. Dequantise-whole + fp32 GEMM, for the record ──
    let dequant_gemm = best_seconds(|| {
        let mut sink = 0.0f32;
        for (op, x) in ops.iter().zip(&activations) {
            let wf = dequantize_q4_k(&op.packed, op.rows * op.cols);
            let w = Array2::from_shape_vec((op.rows, op.cols), wf).unwrap();
            let y = x.dot(&w.t());
            sink += y[[0, 0]];
        }
        sink
    });

    // ── Numerics: the amortised matmul against fp32, one op ──
    let probe = &ops[0];
    let x0 = &activations[0];
    let mut amortised_out = vec![0.0f32; SEQ * probe.rows];
    q4k_matmul_into(
        &mut amortised_out,
        x0.as_slice().unwrap(),
        &probe.packed,
        probe.rows,
        probe.cols,
        SEQ,
    );
    let reference = x0.dot(&probe.f32_matrix.t());
    let (mut dot, mut ref_sq, mut got_sq) = (0.0f64, 0.0f64, 0.0f64);
    for (index, &r) in reference.as_slice().unwrap().iter().enumerate() {
        let g = amortised_out[index];
        dot += r as f64 * g as f64;
        ref_sq += (r as f64).powi(2);
        got_sq += (g as f64).powi(2);
    }
    let cosine = dot / (ref_sq.sqrt() * got_sq.sqrt());

    // ── The prediction ──
    println!(
        "\nrow path (today)      {:.0} ms  ({:.0} GFLOPS)",
        row_path * 1000.0,
        flops / row_path / 1e9
    );
    println!(
        "amortised q4 matmul   {:.0} ms  ({:.0} GFLOPS, {:.1}x vs row path)",
        amortised * 1000.0,
        flops / amortised / 1e9,
        row_path / amortised
    );
    println!(
        "blocked q4k·q8k gemm  {:.0} ms  ({:.0} GFLOPS, {:.1}x vs row path)",
        blocked * 1000.0,
        flops / blocked / 1e9,
        row_path / blocked
    );
    println!(
        "fp32 BLAS GEMM        {:.0} ms  ({:.0} GFLOPS)",
        f32_gemm * 1000.0,
        flops / f32_gemm / 1e9
    );
    println!(
        "dequant-whole + GEMM  {:.0} ms  (for the record)",
        dequant_gemm * 1000.0
    );
    println!("amortised-matmul cosine vs fp32 (first op): {cosine:.6}");
    let overhead = MEASURED_PREFILL_SECONDS - row_path;
    println!(
        "\nmeasured Q4 prefill   {MEASURED_PREFILL_SECONDS:.2} s; GEMV term {:.2} s -> non-GEMM overhead {:.2} s",
        row_path, overhead
    );
    for (name, candidate) in [
        ("blocked q4k·q8k gemm", blocked),
        ("fp32 BLAS GEMM", f32_gemm),
    ] {
        println!(
            "prediction with {name}: prefill ≈ {:.0} ms (gate: TTFA < 500 ms)",
            (overhead + candidate) * 1000.0
        );
    }
}

fn best_seconds(mut pass: impl FnMut() -> f32) -> f64 {
    let mut best = f64::INFINITY;
    let mut sink = 0.0f32;
    for _ in 0..REPS {
        let start = Instant::now();
        sink += pass();
        best = best.min(start.elapsed().as_secs_f64());
    }
    assert!(sink.is_finite());
    best
}
