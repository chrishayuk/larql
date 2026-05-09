//! End-to-end synthetic speculative-step microbench.
//!
//! Composes the three phase 2/3 CUDA kernels at realistic
//! Gemma 3 4B Q4_K_M shapes:
//!
//!   1. `q4k_batched::matvec_batched` — projection layers at M_TILE=N
//!   2. `attn_tree::tree_decode_attention` — masked attention at q_tokens=N
//!   3. `sampling::verify_tree_gpu` — rejection-sampling verification
//!
//! Synthetic inputs (no real model required). Reports per-step wall
//! time at M=1 (single-token decode baseline) vs M=5 (depth-2
//! branches=2 5-node tree). Updates the empirical perf model from
//! cuda-decode-perf-results-followup with composition-level numbers.
//!
//! Run with `--ignored` to opt in:
//!   LARQL_CUDA_AVAILABLE=1 cargo test --release \
//!     -p larql-compute --features cuda \
//!     --test bench_cuda_speculative_step -- --ignored --nocapture

#![cfg(any(feature = "cuda", feature = "cuda-oxide"))]

use std::time::Instant;

use larql_compute::cuda::attn_tree::{tree_decode_attention, TreeAttentionOpts};
use larql_compute::cuda::q4k_batched::matvec_batched;
use larql_compute::cuda::sampling::{verify_tree_gpu, verify_tree_gpu_parallel};
use larql_compute::cuda::CudaBackend;

const Q4K_BLOCK_BYTES: usize = 144;
const Q4K_BLOCK_ELEMS: usize = 256;

// Gemma 3 4B Q4_K_M shapes.
const HIDDEN: usize = 2560;
const FFN_INTERMEDIATE: usize = 10240;
const NUM_Q_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const VOCAB: usize = 256_000;
const SEQ_LEN: usize = 2048;
const NUM_LAYERS: usize = 26;

fn gpu_or_skip() -> Option<CudaBackend> {
    if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
        return None;
    }
    CudaBackend::new().ok()
}

fn synth_bytes(rows: usize, hidden: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = hidden / Q4K_BLOCK_ELEMS;
    let n_bytes = rows * blocks_per_row * Q4K_BLOCK_BYTES;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n_bytes)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) & 0xff) as u8
        })
        .collect()
}

fn synth_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0xCBF2_9CE4_8422_2325);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFFFF) as f32 / 65536.0) - 0.5
        })
        .collect()
}

/// Time one full single-layer decode step (projections + attention).
/// Excludes embedding lookup and lm_head — those are first / last
/// layer concerns benched separately.
#[allow(clippy::too_many_arguments)]
fn time_one_layer(
    backend: &CudaBackend,
    qkv_w: &[u8],
    wo_w: &[u8],
    gate_up_w: &[u8],
    down_w: &[u8],
    x: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    mask_bits: &[u32],
    m: usize,
    ctx_len: usize,
    words_per_row: usize,
) -> f64 {
    let t0 = Instant::now();
    // proj_qkv
    let _qkv = matvec_batched(
        backend,
        qkv_w,
        x,
        (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
        HIDDEN,
        m,
    )
    .unwrap();
    // attention — uses synthetic Q/K/V we already have
    let q_in = synth_f32(m * NUM_Q_HEADS * HEAD_DIM, 0xA1);
    let opts = TreeAttentionOpts {
        q_tokens: m,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        ctx_len,
        words_per_row,
        attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        softcap: 0.0,
    };
    let _attn = tree_decode_attention(backend, &q_in, k_cache, v_cache, mask_bits, opts).unwrap();
    // proj_wo
    let attn_flat = synth_f32(m * HIDDEN, 0xA2);
    let _wo = matvec_batched(backend, wo_w, &attn_flat, HIDDEN, HIDDEN, m).unwrap();
    // FFN: gate_up
    let _gate_up = matvec_batched(backend, gate_up_w, x, FFN_INTERMEDIATE, HIDDEN, m).unwrap();
    // FFN: down
    let down_in = synth_f32(m * FFN_INTERMEDIATE, 0xA3);
    let _down = matvec_batched(backend, down_w, &down_in, HIDDEN, FFN_INTERMEDIATE, m).unwrap();
    t0.elapsed().as_secs_f64() * 1000.0
}

#[test]
#[ignore]
fn bench_synthetic_speculative_step_at_gemma_shapes() {
    let Some(backend) = gpu_or_skip() else {
        eprintln!("LARQL_CUDA_AVAILABLE != 1; skipping");
        return;
    };

    eprintln!("\n=== Synthetic Gemma 3 4B Q4_K_M speculative-step microbench ===");
    eprintln!(
        "hidden={HIDDEN}  FFN={FFN_INTERMEDIATE}  q_heads={NUM_Q_HEADS}  kv_heads={NUM_KV_HEADS}"
    );
    eprintln!("head_dim={HEAD_DIM}  vocab={VOCAB}  seq_len={SEQ_LEN}  layers={NUM_LAYERS}");

    // One copy of each weight — reused across iterations and across
    // (M=1 vs M=5) to keep weight upload cost identical between runs.
    let qkv_w = synth_bytes((NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, HIDDEN, 0xB1);
    let wo_w = synth_bytes(HIDDEN, HIDDEN, 0xB2);
    let gate_up_w = synth_bytes(FFN_INTERMEDIATE, HIDDEN, 0xB3);
    let down_w = synth_bytes(HIDDEN, FFN_INTERMEDIATE, 0xB4);
    let k_cache = synth_f32(SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, 0xB5);
    let v_cache = synth_f32(SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, 0xB6);

    for m in &[1usize, 5] {
        let m = *m;
        let x = synth_f32(m * HIDDEN, 0xC0 + m as u64);
        let words_per_row = SEQ_LEN.div_ceil(32);
        // Full mask: every q sees every cache position.
        let mut mask = vec![0u32; m * words_per_row];
        for q in 0..m {
            for j in 0..SEQ_LEN {
                mask[q * words_per_row + j / 32] |= 1u32 << (j % 32);
            }
        }

        // Warmup
        for _ in 0..3 {
            let _ = time_one_layer(
                &backend,
                &qkv_w,
                &wo_w,
                &gate_up_w,
                &down_w,
                &x,
                &k_cache,
                &v_cache,
                &mask,
                m,
                SEQ_LEN,
                words_per_row,
            );
        }

        let iters = 30;
        let mut layer_ms = 0.0;
        for _ in 0..iters {
            layer_ms += time_one_layer(
                &backend,
                &qkv_w,
                &wo_w,
                &gate_up_w,
                &down_w,
                &x,
                &k_cache,
                &v_cache,
                &mask,
                m,
                SEQ_LEN,
                words_per_row,
            );
        }
        layer_ms /= iters as f64;

        let layers_ms = layer_ms * NUM_LAYERS as f64;

        eprintln!(
            "\n  M={m}: per-layer {layer_ms:.3} ms  → all {NUM_LAYERS} layers {layers_ms:.2} ms"
        );

        // verify_tree at vocab=256k: time it standalone since it's
        // a single per-step cost (not per-layer).
        if m > 1 {
            // Synthesize per-node target probs (uniform-ish).
            let p_uniform = vec![1.0_f32 / VOCAB as f32; VOCAB];
            let p_target_rows: Vec<Vec<f32>> = (0..m).map(|_| p_uniform.clone()).collect();
            let drafts: Vec<i32> = (0..m as i32).collect();
            let pdraft = vec![1.0_f32 / VOCAB as f32; m];

            // Warmup
            for _ in 0..2 {
                let _ =
                    verify_tree_gpu(&backend, &p_target_rows, &drafts, &pdraft, VOCAB, 0).unwrap();
            }

            let viters = 5;
            let t0 = Instant::now();
            for _ in 0..viters {
                let _ =
                    verify_tree_gpu(&backend, &p_target_rows, &drafts, &pdraft, VOCAB, 0).unwrap();
            }
            let verify_serial_ms = t0.elapsed().as_secs_f64() * 1000.0 / viters as f64;

            // Parallel kernel.
            for _ in 0..2 {
                let _ =
                    verify_tree_gpu_parallel(&backend, &p_target_rows, &drafts, &pdraft, VOCAB, 0)
                        .unwrap();
            }
            let t1 = Instant::now();
            for _ in 0..viters {
                let _ =
                    verify_tree_gpu_parallel(&backend, &p_target_rows, &drafts, &pdraft, VOCAB, 0)
                        .unwrap();
            }
            let verify_parallel_ms = t1.elapsed().as_secs_f64() * 1000.0 / viters as f64;
            let speedup = verify_serial_ms / verify_parallel_ms;
            eprintln!(
                "  M={m}: verify_tree at vocab={VOCAB}: serial {verify_serial_ms:.3} ms  parallel {verify_parallel_ms:.3} ms  speedup {speedup:.1}×"
            );

            let total_step_serial = layers_ms + verify_serial_ms;
            let total_step_parallel = layers_ms + verify_parallel_ms;
            eprintln!(
                "  M={m}: TOTAL step (layers + verify) = serial {total_step_serial:.2} ms  parallel {total_step_parallel:.2} ms"
            );

            // ms/tok for various α (parallel verify path).
            for alpha in &[0.5_f64, 0.6, 0.7, 0.8] {
                let exp_tokens: f64 = (0..m).map(|k| alpha.powi(k as i32)).sum::<f64>();
                let ms_per_tok = total_step_parallel / exp_tokens;
                eprintln!(
                    "    α={alpha}: expected_tokens/step={exp_tokens:.2} → {ms_per_tok:.2} ms/tok (parallel)"
                );
            }
        }
    }

    eprintln!("\n=== Notes ===");
    eprintln!("- Synthetic weights / activations; absolute numbers are upper-bounds on production");
    eprintln!("  (each call uploads weights to device — production keeps them resident).");
    eprintln!("- verify_tree at vocab=256k uses a single-thread serial inner loop today;");
    eprintln!("  parallel reduction is the next perf step. Parity is locked at this scope.");
}
