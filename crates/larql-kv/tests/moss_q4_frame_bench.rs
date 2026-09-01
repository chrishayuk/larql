//! The Q4 frame prediction — the falsifiable bandwidth experiment.
//!
//! The step-5 stage split says a MOSS frame is memory-bandwidth-bound:
//! ~6.8 GB of fp32 backbone weights once + ~1.06 GB of depth weights
//! sixteen times (`docs/tts-funnel.md` step-5 gate log). If that model
//! of the cost is right, Q4_K (~4.5 bits/weight) should cut GEMV time by
//! a large factor and the frame should land near the 80 ms budget with
//! no other change. If it doesn't, the kernels, the packing, or the
//! bandwidth model is wrong — either way we learn.
//!
//! Method: real checkpoint matrices, the real per-frame op mix (28
//! backbone layers ×1 use, 4 depth layers ×16 uses, 16 heads ×1),
//! fp32 BLAS GEMV vs `quantize_q4_k` + `q4k_matvec::dispatch`, best of
//! `REPS` frame-equivalents each. Attention/cache/norm overhead is
//! everything the GEMVs don't account for; the prediction substitutes
//! the Q4 GEMV total into the measured frame time.
//!
//! NOT FOR CI (needs the checkpoint, and it is a benchmark):
//!
//! ```text
//! MOSS_TTS_REALTIME_DIR=<snapshot> \
//! cargo test --release -p larql-kv --test moss_q4_frame_bench -- --ignored --nocapture
//! ```

mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::cpu::ops::q4k_matvec;
use larql_compute::cpu::ops::q4k_q8k_dot::{q4k_q8k_matvec_into, q4k_q8k_matvec_parallel};
use larql_compute::quantize_x_to_q8k;
use ndarray::{Array1, Array2};

use common::model_dir_from_env;
use larql_inference::larql_models::loading::safetensors::load_model_dir;
use larql_inference::larql_models::speech::moss_tts_realtime::{
    depth_transformer_model, load_moss_tts_aux_from_safetensors, MossTtsRealtimeConfig,
};

const REPS: usize = 5;

/// One weight matrix and how many times a single frame multiplies it.
struct FrameOp {
    name: &'static str,
    matrix: Array2<f32>,
    uses_per_frame: usize,
}

#[test]
#[ignore]
fn q4_frame_prediction() {
    let model_dir = model_dir_from_env();
    let weights = load_model_dir(&model_dir).expect("backbone load");
    let config_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::path::Path::new(&model_dir).join("config.json")).unwrap(),
    )
    .unwrap();
    let config = MossTtsRealtimeConfig::from_config_json(&config_json).unwrap();
    let aux = load_moss_tts_aux_from_safetensors(&model_dir, weights.arch.as_ref(), config.clone())
        .expect("aux load");
    let depth = depth_transformer_model(aux.local, &config).expect("depth adapter");

    // ── The per-frame op mix, from the real tensors ──
    let mut ops: Vec<FrameOp> = Vec::new();
    let arch = &weights.arch;
    for layer in 0..weights.num_layers {
        for (name, key) in [
            ("bb.q", arch.attn_q_key(layer)),
            ("bb.k", arch.attn_k_key(layer)),
            ("bb.v", arch.attn_v_key(layer)),
            ("bb.o", arch.attn_o_key(layer)),
            ("bb.gate", arch.ffn_gate_key(layer)),
            ("bb.up", arch.ffn_up_key(layer)),
            ("bb.down", arch.ffn_down_key(layer)),
        ] {
            let matrix = weights.tensors.get(&key).expect(&key).to_owned();
            ops.push(FrameOp {
                name,
                matrix,
                uses_per_frame: 1,
            });
        }
    }
    let depth_arch = &depth.weights.arch;
    for layer in 0..depth.weights.num_layers {
        for (name, key) in [
            ("dt.q", depth_arch.attn_q_key(layer)),
            ("dt.k", depth_arch.attn_k_key(layer)),
            ("dt.v", depth_arch.attn_v_key(layer)),
            ("dt.o", depth_arch.attn_o_key(layer)),
            ("dt.gate", depth_arch.ffn_gate_key(layer)),
            ("dt.up", depth_arch.ffn_up_key(layer)),
            ("dt.down", depth_arch.ffn_down_key(layer)),
        ] {
            let matrix = depth.weights.tensors.get(&key).expect(&key).to_owned();
            ops.push(FrameOp {
                name,
                matrix,
                uses_per_frame: config.rvq,
            });
        }
    }
    for head in &depth.lm_heads {
        ops.push(FrameOp {
            name: "dt.head",
            matrix: head.clone(),
            uses_per_frame: 1,
        });
    }

    let op_bytes = |op: &FrameOp| op.matrix.len() * std::mem::size_of::<f32>() * op.uses_per_frame;
    let total_bytes_f32: usize = ops.iter().map(op_bytes).sum();
    println!(
        "frame op mix: {} matrices, {:.1} GB fp32 moved per frame",
        ops.len(),
        total_bytes_f32 as f64 / 1e9
    );
    // Which op class carries the mass — the bandwidth model above claims the
    // depth transformer's ×rvq reuse dominates, and this is where that shows.
    let mut by_class: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for op in &ops {
        let entry = by_class.entry(op.name).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += op_bytes(op);
    }
    for (name, (count, bytes)) in &by_class {
        println!(
            "  {name:<8} {count:>3} matrices  {:.2} GB/frame  ({:.1}%)",
            *bytes as f64 / 1e9,
            100.0 * *bytes as f64 / total_bytes_f32 as f64
        );
    }

    // ── fp32 GEMV timing (BLAS via ndarray dot) ──
    let frame_f32 = best_frame_seconds(|| {
        let mut sink = 0.0f32;
        for op in &ops {
            let x = Array1::<f32>::from_elem(op.matrix.ncols(), 0.5);
            for _ in 0..op.uses_per_frame {
                let y = op.matrix.dot(&x);
                sink += y[0];
            }
        }
        sink
    });

    // ── Quantize (row-major Q4_K superblocks) and re-time ──
    let quantize_start = Instant::now();
    let packed: Vec<(usize, usize, usize, Vec<u8>)> = ops
        .iter()
        .map(|op| {
            let (rows, cols) = op.matrix.dim();
            let data = op.matrix.as_slice().expect("standard layout").to_vec();
            (rows, cols, op.uses_per_frame, quantize_q4_k(&data))
        })
        .collect();
    let q4_bytes: usize = packed
        .iter()
        .map(|(_, _, uses, blob)| blob.len() * uses)
        .sum();
    println!(
        "quantized in {:.1}s: {:.2} GB moved per frame at Q4_K",
        quantize_start.elapsed().as_secs_f64(),
        q4_bytes as f64 / 1e9
    );

    let frame_q4 = best_frame_seconds(|| {
        let mut sink = 0.0f32;
        for (rows, cols, uses, blob) in &packed {
            let x = vec![0.5f32; *cols];
            for _ in 0..*uses {
                let y = q4k_matvec::dispatch(blob, &x, *rows, *cols);
                sink += y[0];
            }
        }
        sink
    });

    // ── The production kernel: Q8K activations + NEON Q4K·Q8K dot ──
    let frame_q4k_q8k = best_frame_seconds(|| {
        let mut sink = 0.0f32;
        for (rows, cols, uses, blob) in &packed {
            let x = vec![0.5f32; *cols];
            let q8k = quantize_x_to_q8k(&x);
            let mut out = vec![0.0f32; *rows];
            for _ in 0..*uses {
                q4k_q8k_matvec_into(&mut out, &q8k, blob, *rows, *cols)
                    .expect("bench shapes are valid");
                sink += out[0];
            }
        }
        sink
    });

    // ── Same kernel, row-parallel ──
    let frame_q4k_parallel = best_frame_seconds(|| {
        let mut sink = 0.0f32;
        for (rows, cols, uses, blob) in &packed {
            let x = vec![0.5f32; *cols];
            let q8k = quantize_x_to_q8k(&x);
            let mut out = vec![0.0f32; *rows];
            for _ in 0..*uses {
                q4k_q8k_matvec_parallel(&mut out, &q8k, blob, *rows, *cols, "Q4_K")
                    .expect("bench shapes are valid");
                sink += out[0];
            }
        }
        sink
    });

    // ── The prediction ──
    // Measured whole-frame time carries GEMV + everything else; swap the
    // GEMV term. The measured p50 band from the step-5 runs is the
    // context, printed for the reader rather than baked in.
    println!("\nfp32 GEMV total       {:.1} ms/frame", frame_f32 * 1000.0);
    println!(
        "q4k dequant-dot       {:.1} ms/frame (naive kernel, for the record)",
        frame_q4 * 1000.0
    );
    println!(
        "q4k·q8k single-core   {:.1} ms/frame ({:.2}x vs fp32)",
        frame_q4k_q8k * 1000.0,
        frame_f32 / frame_q4k_q8k
    );
    println!(
        "q4k·q8k parallel      {:.1} ms/frame ({:.2}x vs fp32)",
        frame_q4k_parallel * 1000.0,
        frame_f32 / frame_q4k_parallel
    );
    let best_q4 = frame_q4k_q8k.min(frame_q4k_parallel);
    println!(
        "prediction: measured frame p50 (190-230 ms band) - fp32 GEMV + best q4 = {:.0}-{:.0} ms",
        190.0 - frame_f32 * 1000.0 + best_q4 * 1000.0,
        230.0 - frame_f32 * 1000.0 + best_q4 * 1000.0,
    );
    println!("realtime budget       80 ms");
}

fn best_frame_seconds(mut frame: impl FnMut() -> f32) -> f64 {
    let mut best = f64::INFINITY;
    let mut sink = 0.0f32;
    for _ in 0..REPS {
        let start = Instant::now();
        sink += frame();
        best = best.min(start.elapsed().as_secs_f64());
    }
    assert!(sink.is_finite());
    best
}
