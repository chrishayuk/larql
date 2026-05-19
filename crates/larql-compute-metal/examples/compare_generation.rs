//! Token generation benchmarks: simulates actual decode-time inference.
//!
//! Tests the production case: seq=1 per token with KV cache,
//! vs seq=6 without cache. Shows the multiplier from KV caching.
//!
//! Usage:
//!   cargo run --release -p larql-compute --features metal --example bench_generation

extern crate blas_src;

use larql_compute::cpu::q4;
use larql_compute::cpu::q4::quantize_q4_0;
use larql_compute::cpu_backend;
use ndarray::Array2;
use std::time::Instant;

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

struct Timer {
    n: usize,
}
impl Timer {
    fn run<F: FnMut()>(&self, name: &str, mut f: F) -> f64 {
        f();
        let t0 = Instant::now();
        for _ in 0..self.n {
            f();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / self.n as f64;
        let tps = 1000.0 / ms;
        println!("  {name:55} {ms:>7.2}ms  ({tps:>5.1} tok/s)");
        ms
    }
}

fn main() {
    let hidden = 2560;
    let inter = 10240;
    let head_dim = 320;
    #[allow(unused_variables)]
    let num_q = 8;
    let num_kv = 4;
    let kv_dim = num_kv * head_dim;
    let cpu = cpu_backend();
    let t = Timer { n: 10 };

    println!("=== Token Generation Benchmarks ===");
    println!("Simulating decode: seq=1 per token (KV cached)\n");

    // Build 21 layers of Q4 data
    let mut layers_q4: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    for l in 0..21u64 {
        let g: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f64 + l as f64 * 1e7) * 0.0001).cos() as f32)
            .collect();
        let u: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f64 + l as f64 * 2e7) * 0.0002).sin() as f32)
            .collect();
        let d: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f64 + l as f64 * 3e7) * 0.0003).cos() as f32)
            .collect();
        let mut dt = vec![0.0f32; hidden * inter];
        for r in 0..inter {
            for c in 0..hidden {
                dt[c * inter + r] = d[r * hidden + c];
            }
        }
        layers_q4.push((quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
    }

    // Build attention weights for 21 layers
    let attn_wq: Vec<Vec<f32>> = (0..21)
        .map(|l| {
            (0..hidden * hidden)
                .map(|i| ((i + l * 1000) as f32 * 0.0001).cos())
                .collect()
        })
        .collect();
    let attn_wk: Vec<Vec<f32>> = (0..21)
        .map(|l| {
            (0..kv_dim * hidden)
                .map(|i| ((i + l * 2000) as f32 * 0.0002).sin())
                .collect()
        })
        .collect();
    let attn_wv: Vec<Vec<f32>> = (0..21)
        .map(|l| {
            (0..kv_dim * hidden)
                .map(|i| ((i + l * 3000) as f32 * 0.0003).cos())
                .collect()
        })
        .collect();
    let attn_wo: Vec<Vec<f32>> = (0..21)
        .map(|l| {
            (0..hidden * hidden)
                .map(|i| ((i + l * 4000) as f32 * 0.0004).sin())
                .collect()
        })
        .collect();

    // ── 1. Prefill (seq=6, no KV cache) ──
    println!("--- 1. Prefill: seq=6, 21 layers (no KV cache) ---\n");

    t.run("CPU f32 prefill (seq=6, 4 attn proj × 21 layers)", || {
        let h = synth(6, hidden, 42);
        for l in 0..21 {
            let wq = Array2::from_shape_vec((hidden, hidden), attn_wq[l].clone()).unwrap();
            let wk = Array2::from_shape_vec((kv_dim, hidden), attn_wk[l].clone()).unwrap();
            let wv = Array2::from_shape_vec((kv_dim, hidden), attn_wv[l].clone()).unwrap();
            let wo = Array2::from_shape_vec((hidden, hidden), attn_wo[l].clone()).unwrap();
            let _ = cpu.matmul_transb(h.view(), wq.view());
            let _ = cpu.matmul_transb(h.view(), wk.view());
            let _ = cpu.matmul_transb(h.view(), wv.view());
            let _ = cpu.matmul_transb(h.view(), wo.view());
        }
    });

    // ── 2. Decode: seq=1 with KV cache (CPU) ──
    println!("\n--- 2. Decode: seq=1 per token, 21 layers (KV cached) ---\n");

    // CPU Q4 decode (seq=1)
    t.run("CPU C Q4 decode (seq=1, FFN only, 21 layers)", || {
        let mut h: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        for (gate_q4, up_q4, down_t_q4) in &layers_q4 {
            let g = q4::q4_matvec(gate_q4, &h, inter, hidden);
            let u = q4::q4_matvec(up_q4, &h, inter, hidden);
            let mut act = vec![0.0f32; inter];
            for i in 0..inter {
                act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i];
            }
            h = q4::q4_matvec(down_t_q4, &act, hidden, inter);
        }
    });

    // CPU f32 BLAS decode (seq=1, attention only — 4 projections)
    t.run(
        "CPU f32 decode (seq=1, attn 4 proj only, 21 layers)",
        || {
            let h = synth(1, hidden, 42);
            for l in 0..21 {
                let wq = Array2::from_shape_vec((hidden, hidden), attn_wq[l].clone()).unwrap();
                let wk = Array2::from_shape_vec((kv_dim, hidden), attn_wk[l].clone()).unwrap();
                let wv = Array2::from_shape_vec((kv_dim, hidden), attn_wv[l].clone()).unwrap();
                let wo = Array2::from_shape_vec((hidden, hidden), attn_wo[l].clone()).unwrap();
                let _ = cpu.matmul_transb(h.view(), wq.view());
                let _ = cpu.matmul_transb(h.view(), wk.view());
                let _ = cpu.matmul_transb(h.view(), wv.view());
                // O proj after attention: [1, hidden] @ [hidden, hidden]^T
                let _ = cpu.matmul_transb(h.view(), wo.view());
            }
        },
    );

    // CPU full decode (seq=1, attn + FFN)
    t.run("CPU full decode (seq=1, attn + Q4 FFN, 21 layers)", || {
        let mut h: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        for l in 0..21 {
            // Attention: 4 projections (simulate)
            let h_arr = Array2::from_shape_vec((1, hidden), h.clone()).unwrap();
            let wq = Array2::from_shape_vec((hidden, hidden), attn_wq[l].clone()).unwrap();
            let wk = Array2::from_shape_vec((kv_dim, hidden), attn_wk[l].clone()).unwrap();
            let wv = Array2::from_shape_vec((kv_dim, hidden), attn_wv[l].clone()).unwrap();
            let wo = Array2::from_shape_vec((hidden, hidden), attn_wo[l].clone()).unwrap();
            let _ = cpu.matmul_transb(h_arr.view(), wq.view());
            let _ = cpu.matmul_transb(h_arr.view(), wk.view());
            let _ = cpu.matmul_transb(h_arr.view(), wv.view());
            let _ = cpu.matmul_transb(h_arr.view(), wo.view());
            // FFN: Q4
            let (gate_q4, up_q4, down_t_q4) = &layers_q4[l];
            let g = q4::q4_matvec(gate_q4, &h, inter, hidden);
            let u = q4::q4_matvec(up_q4, &h, inter, hidden);
            let mut act = vec![0.0f32; inter];
            for i in 0..inter {
                act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i];
            }
            h = q4::q4_matvec(down_t_q4, &act, hidden, inter);
        }
    });

    // ── 3. Metal decode (seq=1) ──
    println!("\n--- 3. Metal decode: seq=1, 21 layers ---\n");
    #[cfg(target_os = "macos")]
    {
        if let Some(ref metal) = larql_compute_metal::MetalBackend::new() {
            // Metal full layer at seq=1
            t.run("Metal full layer (seq=1, 21 layers, 1 cmd/layer)", || {
                let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
                // We can't easily chain layers without reading back, so benchmark one layer
                // and multiply
                for l in 0..21 {
                    let (gate_q4, up_q4, down_t_q4) = &layers_q4[l];
                    let _ = metal.full_layer_direct(
                        &attn_wq[l],
                        &attn_wk[l],
                        &attn_wv[l],
                        &attn_wo[l],
                        gate_q4,
                        up_q4,
                        down_t_q4,
                        &x,
                        1,
                        hidden,
                        num_q,
                        num_kv,
                        head_dim,
                        inter,
                        1.0 / (head_dim as f32).sqrt(),
                    );
                }
            });

            // Metal Q4 FFN only at seq=1
            t.run("Metal Q4 FFN only (seq=1, 21 layers)", || {
                let mut h: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
                for (gate_q4, up_q4, down_t_q4) in &layers_q4 {
                    let (q8, sc) = q4::quantize_to_q8(&h);
                    let g = metal.q4_matvec_direct(gate_q4, &q8, &sc, inter, hidden);
                    let u = metal.q4_matvec_direct(up_q4, &q8, &sc, inter, hidden);
                    let mut act = vec![0.0f32; inter];
                    for i in 0..inter {
                        act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i];
                    }
                    h = metal.q4_f32_matvec_direct(down_t_q4, &act, hidden, inter);
                }
            });
        }
    }
    #[cfg(not(target_os = "macos"))]
    println!("  (Metal not enabled)");

    // ── 4. Comparison summary ──
    println!("\n--- 4. Summary ---\n");
    println!("  Ollama (Q4 Metal, KV cache):     ~10ms/token → ~100 tok/s");
    println!("  LARQL target with Metal + KV:     ~25ms/token → ~40 tok/s");
    println!("  LARQL current (f32, no KV):      ~220ms/token → ~4.5 tok/s");

    println!("\n=== Done ===");
}
