//! Single-thread roofline microbench for the Q4_K × Q8_K decode matvec —
//! the C12 inner loop (see `crates/larql-compute/docs/q4k-decode-kernel.md`).
//!
//! Run: `cargo bench -p larql-compute --bench q4k_q8k_matvec`
//!
//! Purpose: the C12 gap to llama.cpp (1.73× per-core) is documented in two
//! conflicting ways — the DIAGNOSIS doc calls it "memory-system-level"
//! (DRAM-bandwidth bound), the SPEC calls it compute/scheduling bound
//! (33→21 cycles/super-block). Those imply *different* fixes. This bench
//! settles it **before** writing hand-asm (avoid the build-then-measure
//! trap): it times the production NEON kernel single-threaded at real
//! Gemma 3 4B matvec shapes and reports GB/s on the Q4_K weight stream.
//!
//! Read it as a roofline: `ffn_gate_up` (10240×2560, ~14 MB) blows past L2
//! (DRAM-bound); `attn_proj` (2560×2560, ~3.7 MB) is closer to
//! cache-resident. If both report the **same GB/s**, the kernel is
//! compute-bound and hand-asm scheduling is the lever. If the large shape
//! is markedly slower per byte, it is DRAM-bound and only memory-level
//! parallelism (two-super-block interleave + prefetch) — not ALU
//! scheduling — can help.
//!
//! The `scalar` rows are the portable reference (also the parity oracle);
//! the `neon` rows are today's production path. A future `asm` kernel adds
//! a third row here and must stay bit-identical to `scalar`.

extern crate blas_src;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_matvec_scalar, quantize_x_to_q8k, Q8KActivation,
};
// The NEON + hand-asm kernels are aarch64-only (`#[cfg(target_arch = "aarch64")]`
// at their definitions); importing them unconditionally breaks the x86_64 build
// (CI runs benches via `--all-targets`). Gate the import + their bench arms.
#[cfg(target_arch = "aarch64")]
use larql_compute::cpu::ops::q4_common::quantize_q6_k;
#[cfg(target_arch = "aarch64")]
use larql_compute::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_gate_up_asm, q4k_q8k_gate_up_neon, q4k_q8k_matvec_asm, q4k_q8k_matvec_neon,
    q6k_q8k_matvec_asm, q6k_q8k_matvec_neon,
};

const BLOCK_BYTES: usize = 144;
const ELEMS_PER_BLOCK: usize = 256;

/// Deterministic non-trivial f32 fill (no rand dep). Mixed sin/cos so the
/// quantiser sees a realistic spread of magnitudes per super-block.
fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            ((f * 0.0173 + seed).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 0.6
        })
        .collect()
}

struct Case {
    name: &'static str,
    rows: usize,
    cols: usize,
}

fn weight_bytes(rows: usize, cols: usize) -> u64 {
    (rows * (cols / ELEMS_PER_BLOCK) * BLOCK_BYTES) as u64
}

fn bench_q4k_q8k(c: &mut Criterion) {
    // Gemma 3 4B: hidden=2560, intermediate=10240.
    let cases = [
        // The hot one: FFN gate/up. ~14 MB weight stream per matvec —
        // far past M3 Max's per-core L2, so this is the DRAM-bound case.
        Case {
            name: "ffn_gate_up",
            rows: 10240,
            cols: 2560,
        },
        // FFN down shape as Q4_K (real `down` is Q6_K, but same byte
        // volume, transposed aspect — probes whether aspect ratio / row
        // length changes achieved bandwidth).
        Case {
            name: "ffn_down_shape",
            rows: 2560,
            cols: 10240,
        },
        // Attention projection: ~3.7 MB, closer to cache-resident — the
        // roofline contrast against ffn_gate_up.
        Case {
            name: "attn_proj",
            rows: 2560,
            cols: 2560,
        },
    ];

    let mut group = c.benchmark_group("q4k_q8k_matvec");
    // Single sample-size knob: these are short kernels, give criterion room.
    group.sample_size(60);

    for case in &cases {
        let Case { name, rows, cols } = *case;
        assert_eq!(cols % ELEMS_PER_BLOCK, 0, "cols must be a multiple of 256");

        let w_f32 = synth(rows * cols, 0.3);
        let w_q4 = quantize_q4_k(&w_f32);
        let x = synth(cols, 1.1);
        let q8: Q8KActivation = quantize_x_to_q8k(&x);
        let mut out = vec![0.0f32; rows];

        group.throughput(Throughput::Bytes(weight_bytes(rows, cols)));

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("neon", name), &(), |b, _| {
            b.iter(|| {
                q4k_q8k_matvec_neon(&mut out, &q8, &w_q4, rows, cols)
                    .expect("bench shapes are valid");
                std::hint::black_box(out[0]);
            });
        });

        // C12 hand-asm kernel (bit-exact with scalar/neon — see parity test).
        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("asm", name), &(), |b, _| {
            b.iter(|| {
                q4k_q8k_matvec_asm(&mut out, &q8, &w_q4, rows, cols)
                    .expect("bench shapes are valid");
                std::hint::black_box(out[0]);
            });
        });

        // Scalar reference — only on the smaller shapes (it's ~10-30× slower;
        // running it on the 14 MB shape wastes wall-clock for no extra signal).
        if rows * cols <= 2560 * 2560 {
            group.bench_with_input(BenchmarkId::new("scalar", name), &(), |b, _| {
                b.iter(|| {
                    q4k_q8k_matvec_scalar(&mut out, &q8, &w_q4, rows, cols)
                        .expect("bench shapes are valid");
                    std::hint::black_box(out[0]);
                });
            });
        }
    }

    group.finish();

    // Fused gate+up pair (C12): the kernel `kquant_ffn_forward_layer` and the
    // remote expert server's q8k_wire path call. Same Gemma 3 4B gate/up
    // shape; throughput counts BOTH weight streams (the fusion shares the
    // activation loads, not the weight bytes).
    #[cfg(target_arch = "aarch64")]
    {
        let rows = 10240usize;
        let cols = 2560usize;
        let g_q4 = quantize_q4_k(&synth(rows * cols, 0.3));
        let u_q4 = quantize_q4_k(&synth(rows * cols, 0.9));
        let x = synth(cols, 1.1);
        let q8: Q8KActivation = quantize_x_to_q8k(&x);
        let mut g_out = vec![0.0f32; rows];
        let mut u_out = vec![0.0f32; rows];

        let mut group = c.benchmark_group("q4k_q8k_gate_up");
        group.sample_size(60);
        group.throughput(Throughput::Bytes(2 * weight_bytes(rows, cols)));
        group.bench_with_input(BenchmarkId::new("neon", "ffn_gate_up"), &(), |b, _| {
            b.iter(|| {
                q4k_q8k_gate_up_neon(&mut g_out, &mut u_out, &q8, &g_q4, &u_q4, rows, cols)
                    .expect("bench shapes are valid");
                std::hint::black_box(g_out[0] + u_out[0]);
            });
        });
        group.bench_with_input(BenchmarkId::new("asm", "ffn_gate_up"), &(), |b, _| {
            b.iter(|| {
                q4k_q8k_gate_up_asm(&mut g_out, &mut u_out, &q8, &g_q4, &u_q4, rows, cols)
                    .expect("bench shapes are valid");
                std::hint::black_box(g_out[0] + u_out[0]);
            });
        });
        group.finish();
    }

    // Q6_K matvec (C12): the `down`-projection / attention-V format. Gemma 3
    // 4B down shape; throughput on the 210-byte/256-elem Q6_K stream.
    #[cfg(target_arch = "aarch64")]
    {
        let rows = 2560usize;
        let cols = 10240usize;
        let w_q6 = quantize_q6_k(&synth(rows * cols, 0.3));
        let x = synth(cols, 1.1);
        let q8: Q8KActivation = quantize_x_to_q8k(&x);
        let mut out = vec![0.0f32; rows];

        let mut group = c.benchmark_group("q6k_q8k_matvec");
        group.sample_size(60);
        group.throughput(Throughput::Bytes(
            (rows * (cols / ELEMS_PER_BLOCK) * 210) as u64,
        ));
        group.bench_with_input(BenchmarkId::new("neon", "ffn_down"), &(), |b, _| {
            b.iter(|| {
                q6k_q8k_matvec_neon(&mut out, &q8, &w_q6, rows, cols)
                    .expect("bench shapes are valid");
                std::hint::black_box(out[0]);
            });
        });
        group.bench_with_input(BenchmarkId::new("asm", "ffn_down"), &(), |b, _| {
            b.iter(|| {
                q6k_q8k_matvec_asm(&mut out, &q8, &w_q6, rows, cols)
                    .expect("bench shapes are valid");
                std::hint::black_box(out[0]);
            });
        });
        group.finish();
    }
}

/// C12 decomposition: split the per-super-block cost of `q4k_q8k_matvec_asm`
/// into (a) the `asm!` block itself and (b) the per-block Rust glue
/// (`unpack_scales_mins` + the i32 scale array + the scalar `sum2` loop + 2×
/// f16→f32 + the f32 epilogue). The full matvec measures (a)+(b) with
/// whatever out-of-order overlap the core finds; comparing the three numbers
/// says whether the glue is exposed (worth folding into the asm) or already
/// hidden (only intra-asm instruction-count reduction can pay).
#[cfg(target_arch = "aarch64")]
fn bench_sb_decomposition(c: &mut Criterion) {
    use larql_compute::cpu::ops::q4_common::f16_to_f32;
    use larql_compute::cpu::ops::q4k_q8k_dot::{q4k_sb_sum1_asm, unpack_scales_mins};

    // One row's worth of super-blocks, attn_proj-like width, repeated over a
    // weight buffer big enough to defeat L1 but stay in the same cache regime
    // as the full-matvec bench (~3.7 MB).
    let cols = 2560usize;
    let rows = 1024usize;
    let n_sb_per_row = cols / ELEMS_PER_BLOCK;
    let n_sb = rows * n_sb_per_row;
    let w_q4 = quantize_q4_k(&synth(rows * cols, 0.3));
    let x = synth(cols, 1.1);
    let q8: Q8KActivation = quantize_x_to_q8k(&x);

    // Pre-extracted per-SB inputs for the asm-only arm (the extraction IS
    // the glue — it must not be timed inside this arm).
    let sb_scales: Vec<[i32; 8]> = (0..n_sb)
        .map(|i| {
            let block = &w_q4[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
            let (sc, _mn) = unpack_scales_mins(&block[4..16]);
            [
                sc[0] as i32,
                sc[1] as i32,
                sc[2] as i32,
                sc[3] as i32,
                sc[4] as i32,
                sc[5] as i32,
                sc[6] as i32,
                sc[7] as i32,
            ]
        })
        .collect();

    let mut group = c.benchmark_group("q4k_sb_decomposition");
    group.sample_size(60);
    // Throughput in super-blocks so criterion reports per-SB time directly.
    group.throughput(Throughput::Elements(n_sb as u64));

    group.bench_function("asm_only", |b| {
        b.iter(|| {
            let mut acc = 0i64;
            for i in 0..n_sb {
                let sb_in_row = i % n_sb_per_row;
                let quants = w_q4[i * BLOCK_BYTES + 16..].as_ptr();
                let act = q8.qs[sb_in_row * ELEMS_PER_BLOCK..].as_ptr();
                // SAFETY: same contracts as the production caller.
                let s = unsafe { q4k_sb_sum1_asm(quants, act, sb_scales[i].as_ptr()) };
                acc += s as i64;
            }
            std::hint::black_box(acc)
        });
    });

    group.bench_function("glue_only", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for i in 0..n_sb {
                let sb_in_row = i % n_sb_per_row;
                let block = &w_q4[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
                let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let (scales, mins) = unpack_scales_mins(&block[4..16]);
                let sc = [
                    scales[0] as i32,
                    scales[1] as i32,
                    scales[2] as i32,
                    scales[3] as i32,
                    scales[4] as i32,
                    scales[5] as i32,
                    scales[6] as i32,
                    scales[7] as i32,
                ];
                std::hint::black_box(sc.as_ptr());
                let q8_sums = &q8.sums[sb_in_row * 8..sb_in_row * 8 + 8];
                let d_y = q8.d[sb_in_row];
                let mut sum2_acc: i32 = 0;
                for s in 0..8 {
                    sum2_acc += mins[s] as i32 * q8_sums[s] as i32;
                }
                // Fake sum1 stands in for the asm result; the epilogue math is
                // the real per-SB f32 work.
                let sum1 = i as i32;
                acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2_acc as f32;
            }
            std::hint::black_box(acc)
        });
    });

    group.bench_function("full_matvec", |b| {
        let mut out = vec![0.0f32; rows];
        b.iter(|| {
            q4k_q8k_matvec_asm(&mut out, &q8, &w_q4, rows, cols).expect("bench shapes are valid");
            std::hint::black_box(out[0]);
        });
    });

    group.bench_function("full_matvec_v2", |b| {
        use larql_compute::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_asm_v2;
        let mut out = vec![0.0f32; rows];
        b.iter(|| {
            q4k_q8k_matvec_asm_v2(&mut out, &q8, &w_q4, rows, cols)
                .expect("bench shapes are valid");
            std::hint::black_box(out[0]);
        });
    });

    group.bench_function("full_matvec_v3", |b| {
        use larql_compute::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_asm_v3;
        let mut out = vec![0.0f32; rows];
        b.iter(|| {
            q4k_q8k_matvec_asm_v3(&mut out, &q8, &w_q4, rows, cols)
                .expect("bench shapes are valid");
            std::hint::black_box(out[0]);
        });
    });

    group.finish();
}

/// Effective-bandwidth shape sweep (C12, post-roofline-crossover): the SAME
/// rayon-chunked matvec the production decode path runs (par_chunks_mut(32)
/// over rows → single-thread asm kernel per chunk), measured at the real
/// per-layer shapes. If GB/s climbs steeply with matrix size, the per-call
/// fork-join tax on small per-layer matvecs is where DRAM goes idle — the
/// 26B decode issues ~180 of these sections per token.
#[cfg(target_arch = "aarch64")]
fn bench_mt_shapes(c: &mut Criterion) {
    use larql_compute::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_asm_v3;
    use rayon::prelude::*;

    // (label, rows, cols) — production 26B shapes plus a big amortised
    // reference. K/V 2048×2816 (3.2 MB), Q 4096×2816, O 2816×4096,
    // dense gate/up 2112×2816, lm_head-class 65536×2816 (~104 MB).
    let cases: &[(&str, usize, usize)] = &[
        ("kv_proj_2048x2816", 2048, 2816),
        ("q_proj_4096x2816", 4096, 2816),
        ("o_proj_2816x4096", 2816, 4096),
        ("dense_gu_2112x2816", 2112, 2816),
        ("big_65536x2816", 65536, 2816),
    ];

    let mut group = c.benchmark_group("q8k_mt_shapes");
    group.sample_size(30);

    for &(label, rows, cols) in cases {
        let w_q4 = quantize_q4_k(&synth(rows * cols, 0.3));
        let x = synth(cols, 1.1);
        let q8: Q8KActivation = quantize_x_to_q8k(&x);
        let bytes_per_row = (cols / ELEMS_PER_BLOCK) * BLOCK_BYTES;
        let mut out = vec![0.0f32; rows];

        group.throughput(Throughput::Bytes((rows * bytes_per_row) as u64));
        group.bench_function(label, |b| {
            b.iter(|| {
                // Mirror q8k_direct_proj / matvec_q4k_or_q6k_q8k exactly:
                // 32-row chunks, one single-thread kernel call per chunk.
                out.par_chunks_mut(32).enumerate().for_each(|(ci, chunk)| {
                    let row_start = ci * 32;
                    let n = chunk.len().min(rows.saturating_sub(row_start));
                    if n == 0 {
                        return;
                    }
                    let w = &w_q4[row_start * bytes_per_row..(row_start + n) * bytes_per_row];
                    q4k_q8k_matvec_asm_v3(&mut chunk[..n], &q8, w, n, cols)
                        .expect("bench shapes are valid");
                });
                std::hint::black_box(out[0]);
            });
        });
    }

    // Expert-granularity arm: 8 parallel tasks (top-8 experts), each a
    // SEQUENTIAL gate+up+down for one expert (single-thread kernels) — the
    // production cpu_moe_forward shape. 8 tasks on 8 threads = any straggler
    // idles a core; compare GB/s against the row-chunked arms above.
    {
        let inter = 704usize;
        let hidden = 2816usize;
        let n_experts = 8usize;
        let gu_rows = 2 * inter; // gate+up stacked
        let per_expert_gu = quantize_q4_k(&synth(gu_rows * hidden, 0.3));
        let per_expert_dn = quantize_q4_k(&synth(hidden * 768, 0.7)); // 704→768 padded
        let x = synth(hidden, 1.1);
        let q8: Q8KActivation = quantize_x_to_q8k(&x);
        let act = synth(768, 0.5);
        let q8_act: Q8KActivation = quantize_x_to_q8k(&act);
        let total_bytes = n_experts
            * (gu_rows * (hidden / ELEMS_PER_BLOCK) * BLOCK_BYTES
                + hidden * (768 / ELEMS_PER_BLOCK) * BLOCK_BYTES);

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_function("experts_8x_gate_up_down", |b| {
            b.iter(|| {
                let acc: f32 = (0..n_experts)
                    .into_par_iter()
                    .map(|_| {
                        let mut gu_out = vec![0.0f32; gu_rows];
                        let mut dn_out = vec![0.0f32; hidden];
                        q4k_q8k_matvec_asm_v3(&mut gu_out, &q8, &per_expert_gu, gu_rows, hidden)
                            .expect("bench shapes are valid");
                        q4k_q8k_matvec_asm_v3(&mut dn_out, &q8_act, &per_expert_dn, hidden, 768)
                            .expect("bench shapes are valid");
                        gu_out[0] + dn_out[0]
                    })
                    .sum();
                std::hint::black_box(acc)
            });
        });
    }

    group.finish();
}

/// Achieved effective bandwidth through the **production** parallel entry
/// point, at the same shapes as `bench_mt_shapes`.
///
/// Why this arm exists: `bench_mt_shapes` hand-rolls `par_chunks_mut(32)` over
/// **rayon**, which is what production ran when the standing "larql extracts
/// ~47 GB/s vs llama.cpp ~70" figure was recorded (C12, 2026-06-12). Production
/// no longer runs that: `q4k_q8k_matvec_parallel` routes through the
/// spin-barrier pool, default-on since 2026-06-13, which was worth +28% e2e
/// precisely because it removed the rayon fork-join tax these arms still pay.
/// So the rayon arms measure a path nothing executes any more, and the 47 GB/s
/// number derived from them has been quoted ever since as though it described
/// the shipping stack.
///
/// Running both at identical shapes turns that into a measured delta instead of
/// an assumption, and makes the achieved figure directly comparable to the
/// attainable ceiling from `examples/membw_probe.rs` (~127 GB/s read on M3 Max).
///
/// `LARQL_SPIN_POOL=0` flips this arm back to rayon for a same-binary A/B —
/// spell it inline before the command, never via a shell variable
/// (`project_spin_barrier_pool`: `env $FLAGS` does not word-split under zsh and
/// silently drops all but the first flag).
#[cfg(target_arch = "aarch64")]
fn bench_mt_production(c: &mut Criterion) {
    use larql_compute::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_parallel;

    let cases: &[(&str, usize, usize)] = &[
        ("kv_proj_2048x2816", 2048, 2816),
        ("q_proj_4096x2816", 4096, 2816),
        ("o_proj_2816x4096", 2816, 4096),
        ("dense_gu_2112x2816", 2112, 2816),
        ("big_65536x2816", 65536, 2816),
    ];

    let mut group = c.benchmark_group("q8k_mt_production");
    group.sample_size(30);

    for &(label, rows, cols) in cases {
        let w_q4 = quantize_q4_k(&synth(rows * cols, 0.3));
        let x = synth(cols, 1.1);
        let q8: Q8KActivation = quantize_x_to_q8k(&x);
        let bytes_per_row = (cols / ELEMS_PER_BLOCK) * BLOCK_BYTES;
        let mut out = vec![0.0f32; rows];

        group.throughput(Throughput::Bytes((rows * bytes_per_row) as u64));
        group.bench_function(label, |b| {
            b.iter(|| {
                q4k_q8k_matvec_parallel(&mut out, &q8, &w_q4, rows, cols, "Q4_K")
                    .expect("bench shapes are valid");
                std::hint::black_box(out[0]);
            });
        });
    }

    group.finish();
}

#[cfg(target_arch = "aarch64")]
criterion_group!(
    benches,
    bench_q4k_q8k,
    bench_sb_decomposition,
    bench_mt_shapes,
    bench_mt_production
);
#[cfg(not(target_arch = "aarch64"))]
criterion_group!(benches, bench_q4k_q8k);
criterion_main!(benches);
