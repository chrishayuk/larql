use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kv_cache_benchmark::markov_residual::MarkovResidual;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::rotorquant::RotorQuantStrategy;
use kv_cache_benchmark::standard_kv::StandardKv;
use kv_cache_benchmark::turboquant::TurboQuant;
use kv_cache_benchmark::*;
use rand::prelude::*;

fn bench_encode(c: &mut Criterion) {
    let config = ModelConfig::gemma_4b();
    let dim = config.kv_dim();

    let mut rng = StdRng::seed_from_u64(42);
    let keys: Vec<Vec<f32>> = (0..200)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();
    let values: Vec<Vec<f32>> = (0..200)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();

    let mut group = c.benchmark_group("encode");

    group.bench_function("standard_kv", |b| {
        let s = StandardKv;
        b.iter(|| s.encode(&keys, &values))
    });
    group.bench_function("turboquant_4bit", |b| {
        let s = TurboQuant::new(4);
        b.iter(|| s.encode(&keys, &values))
    });
    group.bench_function("turboquant_3bit", |b| {
        let s = TurboQuant::new(3);
        b.iter(|| s.encode(&keys, &values))
    });
    group.bench_function("markov_residual", |b| {
        let s = MarkovResidual::new(512);
        b.iter(|| s.encode(&keys, &values))
    });
    group.bench_function("rotorquant_iso3", |b| {
        let s = RotorQuantStrategy::iso3();
        b.iter(|| s.encode(&keys, &values))
    });
    group.bench_function("rotorquant_planar3", |b| {
        let s = RotorQuantStrategy::planar3();
        b.iter(|| s.encode(&keys, &values))
    });
    group.bench_function("rotorquant_iso4", |b| {
        let s = RotorQuantStrategy::iso4();
        b.iter(|| s.encode(&keys, &values))
    });
    group.bench_function("rotorquant_planar4", |b| {
        let s = RotorQuantStrategy::planar4();
        b.iter(|| s.encode(&keys, &values))
    });

    group.finish();
}

fn bench_wht(c: &mut Criterion) {
    let mut group = c.benchmark_group("wht");
    for dim in [128, 256] {
        let x: Vec<f32> = (0..dim)
            .map(|i| (i as f32 - dim as f32 / 2.0) / 100.0)
            .collect();
        group.bench_with_input(BenchmarkId::new("wht", dim), &x, |b, x| {
            b.iter(|| kv_cache_benchmark::turboquant::rotation::wht(x))
        });
    }
    group.finish();
}

fn bench_memory_sweep(c: &mut Criterion) {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let rq_iso3 = RotorQuantStrategy::iso3();
    let rq_planar3 = RotorQuantStrategy::planar3();
    let rq_iso4 = RotorQuantStrategy::iso4();
    let rq_planar4 = RotorQuantStrategy::planar4();
    let strategies: Vec<&dyn KvStrategy> = vec![
        &standard,
        &tq4,
        &markov,
        &rq_iso3,
        &rq_planar3,
        &rq_iso4,
        &rq_planar4,
    ];
    let lengths = benchmark::CONTEXT_LENGTHS;
    c.bench_function("memory_sweep", |b| {
        b.iter(|| benchmark::memory_sweep(&config, &strategies, lengths))
    });
}

/// Accuracy metric microbenchmarks — no model weights required.
///
/// These measure the overhead of the accuracy helpers that validate engine
/// hidden-state correctness (cosine, KL, softmax). Useful for understanding
/// how much the correctness checks add to a real-model test run.
fn bench_accuracy_metrics(c: &mut Criterion) {
    use larql_inference::engines::accuracy::{
        cosine_similarity, js_divergence, kl_divergence, mse, softmax,
    };

    let hidden = 2560usize; // Gemma 3 4B hidden_dim
    let mut rng = StdRng::seed_from_u64(99);
    let a: Vec<f32> = (0..hidden)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();
    let b: Vec<f32> = (0..hidden)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();

    let mut group = c.benchmark_group("accuracy");
    group.throughput(Throughput::Elements(hidden as u64));

    group.bench_function("cosine_similarity/2560", |bench| {
        bench.iter(|| cosine_similarity(&a, &b))
    });
    group.bench_function("mse/2560", |bench| bench.iter(|| mse(&a, &b)));

    // Softmax + KL on a 1K-token subset (fast enough for CI)
    let vocab = 1000usize;
    let logits: Vec<f32> = (0..vocab).map(|i| (i as f32) * 0.01).collect();
    let p = softmax(&logits);
    let raw_q: Vec<f32> = (0..vocab).map(|_| rng.gen_range(0.0f32..1.0f32)).collect();
    let q_sum: f32 = raw_q.iter().sum();
    let q: Vec<f32> = raw_q.iter().map(|x| x / q_sum).collect();

    group.bench_function("softmax/1k_vocab", |bench| bench.iter(|| softmax(&logits)));
    group.bench_function("kl_divergence/1k_vocab", |bench| {
        bench.iter(|| kl_divergence(&p, &q))
    });
    group.bench_function("js_divergence/1k_vocab", |bench| {
        bench.iter(|| js_divergence(&p, &q))
    });

    group.finish();
}

/// EngineKind dispatch overhead — construction, parsing, and engine creation.
/// Measures the metadata / dispatch path without a forward pass.
fn bench_engine_kind(c: &mut Criterion) {
    use larql_inference::engines::EngineKind;

    let mut group = c.benchmark_group("engine_kind");

    group.bench_function("from_name/markov-rs", |b| {
        b.iter(|| EngineKind::from_name("markov-rs"))
    });
    group.bench_function("from_name/unlimited-context", |b| {
        b.iter(|| EngineKind::from_name("unlimited-context"))
    });
    group.bench_function("build/markov_rs_W512", |b| {
        b.iter(|| {
            EngineKind::MarkovResidual {
                window_size: Some(512),
            }
            .build(larql_compute::cpu_backend())
        })
    });
    group.bench_function("build/unlimited_context_W512", |b| {
        b.iter(|| {
            EngineKind::UnlimitedContext { window_size: 512 }.build(larql_compute::cpu_backend())
        })
    });

    group.finish();
}

/// Memory accounting at different context lengths.
/// Models how fast engines can report their state size as context grows —
/// relevant for multi-turn systems that need to decide when to evict.
fn bench_engine_memory_accounting(c: &mut Criterion) {
    // Gemma 3 4B geometry
    let layers = 34usize;
    let kv_heads = 4usize;
    let head_dim = 256usize;
    let kv_dim = kv_heads * head_dim;
    let hidden = 2560usize;

    let mut group = c.benchmark_group("engine_memory");

    for &seq_len in &[512usize, 4096, 32768, 131072, 370_000] {
        let window = seq_len.min(512);

        group.bench_with_input(
            BenchmarkId::new("markov_rs_hot_bytes", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    // Hot-window bytes: W × layers × hidden_dim × 4 (f32)
                    window * layers * hidden * 4
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("standard_kv_bytes_fp16", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    // Standard KV (FP16): seq × layers × 2 × kv_dim × 2 bytes
                    seq_len * layers * 2 * kv_dim * 2
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compression_ratio", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let std_kv = seq_len * layers * 2 * kv_dim * 2;
                    let markov_hot = window * layers * hidden * 4;
                    let markov_cold = seq_len.saturating_sub(window) * 4; // 4B/token cold
                    let markov_total = markov_hot + markov_cold;
                    if markov_total > 0 {
                        std_kv as f64 / markov_total as f64
                    } else {
                        0.0
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_wht,
    bench_memory_sweep,
    bench_accuracy_metrics,
    bench_engine_kind,
    bench_engine_memory_accounting,
);
criterion_main!(benches);
