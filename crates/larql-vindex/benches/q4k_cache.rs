//! Q4_K dequant cache vs row-level — measures the trade-off the LRU
//! bound (`set_kquant_ffn_cache_max_layers`) controls.
//!
//! Two strategies for serving full-K FFN compute on Q4_K bytes:
//!
//! 1. **Cached**: dequantise the whole layer to f32 once
//!    (`dequantize_q4_k` over intermediate × hidden), then do plain
//!    f32 scaled-adds across all `K` features. Pays a big up-front
//!    decode cost; amortises across K. This is what `kquant_ffn_layer`
//!    populates and the CPU per-position fallback uses.
//!
//! 2. **Row**: for each feature, fused `q4k_row_scaled_add` directly
//!    against the Q4_K bytes. No allocation, no caching, but `K`
//!    independent decode passes.
//!
//! At what K does row beat cache? This bench answers that for two
//! production-relevant shapes. The result decides whether the LRU
//! bound default should stay 0 (unlimited) or move to a sane cap.
//!
//! Run: `cargo bench -p larql-vindex --bench q4k_cache`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_vindex::quant::registry::lookup;

fn synth_block(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            (u * 1.5).clamp(-2.5, 2.5)
        })
        .collect()
}

/// Pre-encode one layer's down matrix as Q4_K bytes. Returns
/// (bytes, intermediate, hidden).
fn make_q4k_layer(intermediate: usize, hidden: usize) -> (Vec<u8>, usize, usize) {
    let f32_data = synth_block(intermediate * hidden, 0xc0ffee);
    let q4k_bytes = quantize_q4_k(&f32_data);
    (q4k_bytes, intermediate, hidden)
}

/// "Cached" strategy: dequantise the whole layer once, then iterate
/// features doing plain f32 scaled-adds. Mirrors what
/// `kquant_ffn_layer` + caller does, minus the Arc/lock overhead.
fn cached_full_k_scaled_add(
    bytes: &[u8],
    intermediate: usize,
    hidden: usize,
    k: usize,
) -> Vec<f32> {
    let info = lookup("Q4_K").expect("Q4_K registered");
    let n = intermediate * hidden;
    let f32_layer = (info.dequantize)(bytes, n).expect("dequant");
    let mut out = vec![0.0f32; hidden];
    for feat in 0..k.min(intermediate) {
        let row = &f32_layer[feat * hidden..(feat + 1) * hidden];
        let alpha = 0.001 * feat as f32;
        for (o, &r) in out.iter_mut().zip(row.iter()) {
            *o += alpha * r;
        }
    }
    out
}

/// "Row" strategy: fused dequant + scaled-add per feature. Mirrors
/// `kquant_ffn_row_scaled_add` (the path the row-level optimisation
/// uses).
fn row_level_scaled_add(bytes: &[u8], _intermediate: usize, hidden: usize, k: usize) -> Vec<f32> {
    let info = lookup("Q4_K").expect("Q4_K registered");
    let scaled_add = info.row_scaled_add.expect("row_scaled_add");
    let bytes_per_row = info.bytes_per_row(hidden).expect("aligned");
    let mut out = vec![0.0f32; hidden];
    for feat in 0..k {
        let start = feat * bytes_per_row;
        let end = start + bytes_per_row;
        if end > bytes.len() {
            break;
        }
        let alpha = 0.001 * feat as f32;
        scaled_add(&bytes[start..end], alpha, &mut out).expect("scaled_add");
    }
    out
}

fn bench_cached_vs_row(c: &mut Criterion) {
    let mut group = c.benchmark_group("q4k_cached_vs_row");

    let configs: &[(&str, usize, usize, usize)] = &[
        // (label, intermediate, hidden, k)
        ("gemma3-4b-K100", 10_240, 2560, 100), // sparse decode
        ("gemma3-4b-K1024", 10_240, 2560, 1024), // medium decode
        ("gemma3-4b-fullK", 10_240, 2560, 10_240), // full-K branch
    ];

    for &(label, intermediate, hidden, k) in configs {
        let (bytes, _, _) = make_q4k_layer(intermediate, hidden);
        group.throughput(Throughput::Elements(k as u64));

        group.bench_with_input(
            BenchmarkId::new("cached", label),
            &(bytes.clone(), intermediate, hidden, k),
            |b, (bytes, i, h, k)| b.iter(|| cached_full_k_scaled_add(bytes, *i, *h, *k)),
        );

        group.bench_with_input(
            BenchmarkId::new("row", label),
            &(bytes, intermediate, hidden, k),
            |b, (bytes, i, h, k)| b.iter(|| row_level_scaled_add(bytes, *i, *h, *k)),
        );
    }
    group.finish();
}

/// W2 — down leg specifically. Down is stored `[hidden, intermediate]`
/// on disk (PyTorch `nn.Linear` orientation). The legacy
/// `kquant_ffn_layer` cache amortises the transpose by dequantising the
/// whole layer once. The W2 fix emits a feature-major Q4_K down file
/// at extract time so per-feature decode is a single row dequant —
/// no transpose, no cache, no Mutex.
///
/// This bench compares both paths by simulating one full pass of K
/// scaled-adds:
/// - `cache_transpose`: dequantise the `[hidden, intermediate]` layer
///   to f32, transpose to feature-major, then plain scaled-add per
///   feature. Models the legacy `kquant_ffn_row_scaled_add_via_cache`.
/// - `feature_major`: per feature, fused `q4k_row_scaled_add` against
///   feature-major Q4_K bytes. Models `kquant_down_feature_scaled_add`.
fn bench_down_cache_vs_feature_major(c: &mut Criterion) {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let mut group = c.benchmark_group("q4k_down_cache_vs_feature_major");

    // Production-relevant Gemma 3 4B dims for down.
    let intermediate = 10_240usize;
    let hidden = 2560usize;

    // Pre-encode a feature-major down (already transposed, then Q4_K).
    let f32_data = synth_block(intermediate * hidden, 0xfacef00d);
    let fm_q4k_bytes = quantize_q4_k(&f32_data);

    // Pre-encode the legacy [hidden, intermediate] orientation: same
    // values, indexed differently. The cache path dequants this and
    // transposes to feature-major before scaled-add.
    let mut hi_layout = vec![0.0f32; intermediate * hidden];
    for feat in 0..intermediate {
        for h in 0..hidden {
            hi_layout[h * intermediate + feat] = f32_data[feat * hidden + h];
        }
    }
    let hi_q4k_bytes = quantize_q4_k(&hi_layout);

    for &k in &[100usize, 1024, 10_240] {
        group.throughput(Throughput::Elements(k as u64));

        // Cache + transpose path.
        group.bench_with_input(
            BenchmarkId::new("cache_transpose", k),
            &(hi_q4k_bytes.clone(), k),
            |b, (bytes, k_in)| {
                let k_local = *k_in;
                b.iter(|| {
                    let info = lookup("Q4_K").unwrap();
                    let n = intermediate * hidden;
                    let dequant = (info.dequantize)(bytes, n).unwrap();
                    // Transpose to feature-major: [intermediate, hidden].
                    let mut feature_major = vec![0.0f32; n];
                    for h in 0..hidden {
                        let src = &dequant[h * intermediate..(h + 1) * intermediate];
                        for (feat, &v) in src.iter().enumerate() {
                            feature_major[feat * hidden + h] = v;
                        }
                    }
                    // Scaled-add per feature into a hidden-dim accumulator.
                    let mut out = vec![0.0f32; hidden];
                    for feat in 0..k_local.min(intermediate) {
                        let row = &feature_major[feat * hidden..(feat + 1) * hidden];
                        let alpha = 0.001 * feat as f32;
                        for (o, &r) in out.iter_mut().zip(row.iter()) {
                            *o += alpha * r;
                        }
                    }
                    out
                })
            },
        );

        // Feature-major Q4_K row decode.
        group.bench_with_input(
            BenchmarkId::new("feature_major", k),
            &(fm_q4k_bytes.clone(), k),
            |b, (bytes, k_in)| {
                let k_local = *k_in;
                b.iter(|| {
                    let info = lookup("Q4_K").unwrap();
                    let scaled_add = info.row_scaled_add.unwrap();
                    let bytes_per_row = info.bytes_per_row(hidden).unwrap();
                    let mut out = vec![0.0f32; hidden];
                    for feat in 0..k_local {
                        let start = feat * bytes_per_row;
                        let end = start + bytes_per_row;
                        if end > bytes.len() {
                            break;
                        }
                        let alpha = 0.001 * feat as f32;
                        scaled_add(&bytes[start..end], alpha, &mut out).unwrap();
                    }
                    out
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cached_vs_row,
    bench_down_cache_vs_feature_major
);
criterion_main!(benches);
