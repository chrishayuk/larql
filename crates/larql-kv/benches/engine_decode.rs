//! Criterion microbenchmarks for the KV engines on synthetic weights.
//!
//! Times prefill (8-token prompt) and a single decode step on the
//! synthetic test model. The fixture is small so these benches run
//! quickly and don't depend on a vindex on disk; for end-to-end
//! real-model numbers run `larql bench <vindex> --engine <spec>` from
//! the CLI. (The retired `kv-cache-benchmark::kv_strategies` synthetic
//! comparator was deprecated in 2026-05-16 — it measured random-vector
//! encode/decode, not real decode steady-state.)
//!
//! Engines covered:
//! - `standard` (production K/V cache, unbounded)
//! - `standard:window=4` (sliding-window K/V)
//! - `no-cache` (full re-forward per step, debug fallback)
//! - `markov-rs` (residual-stream replacement)
//! - `unlimited-context` (per-window K/V checkpoints)
//! - `turbo-quant-4bit` (WHT + Lloyd-Max 4-bit codec)
//! - `apollo` (boundary-residual injection)

use criterion::{criterion_group, criterion_main, Criterion};
use larql_inference::cpu_engine_backend;
use larql_inference::ffn::WeightFfn;
use larql_inference::test_utils::make_test_weights;
use larql_kv::EngineKind;

fn all_engines() -> Vec<(&'static str, EngineKind)> {
    vec![
        ("standard", EngineKind::Standard { window_size: None }),
        (
            "standard-window-4",
            EngineKind::Standard {
                window_size: Some(4),
            },
        ),
        ("no-cache", EngineKind::NoCache),
        (
            "markov-rs",
            EngineKind::MarkovResidual { window_size: None },
        ),
        (
            "unlimited-context",
            EngineKind::UnlimitedContext { window_size: 4 },
        ),
        ("turbo-quant-4bit", EngineKind::TurboQuant { bits: 4 }),
        (
            "apollo",
            EngineKind::Apollo {
                injection_layer: 1,
                inject_coefficient: 8.0,
                top_k: 4,
            },
        ),
    ]
}

fn bench_prefill(c: &mut Criterion) {
    let weights = make_test_weights();
    let prompt: Vec<u32> = (0..8).collect();
    let ffn = WeightFfn { weights: &weights };

    let mut group = c.benchmark_group("prefill");
    for (name, kind) in all_engines() {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut engine = kind.clone().build(cpu_engine_backend());
                let _ = engine.prefill(&weights, &ffn, &prompt);
            });
        });
    }
    group.finish();
}

fn bench_decode_step(c: &mut Criterion) {
    let weights = make_test_weights();
    let prompt: Vec<u32> = (0..8).collect();
    let ffn = WeightFfn { weights: &weights };

    let mut group = c.benchmark_group("decode_step");
    for (name, kind) in all_engines() {
        group.bench_function(name, |b| {
            // Pre-warm: prefill once, then time a single decode_step.
            let mut engine = kind.clone().build(cpu_engine_backend());
            let _ = engine.prefill(&weights, &ffn, &prompt);
            b.iter(|| {
                let _ = engine.decode_step(&weights, &ffn, 1);
            });
        });
    }
    group.finish();
}

/// Step-4 parity-relevant bench: end-to-end token generation through
/// `generate_with_engine` (Standard) vs the legacy `generate_cached_backend`.
/// If `Standard` is a parity-preserving wrapper, this benchmark
/// quantifies the dispatch-trait overhead — should be a wash.
fn bench_engine_vs_legacy_generation(c: &mut Criterion) {
    use larql_inference::test_utils::make_test_tokenizer;
    use larql_kv::generation::{generate_cached_backend, generate_with_engine};
    use larql_kv::StandardEngine;

    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..8).collect();
    let max = 8;

    let mut group = c.benchmark_group("generate");

    group.bench_function("legacy_generate_cached_backend", |b| {
        b.iter(|| {
            generate_cached_backend(
                &weights,
                &tokenizer,
                &ffn,
                &prompt,
                max,
                None,
                None,
                |_, _| {},
            );
        });
    });

    group.bench_function("engine_dispatch_standard", |b| {
        b.iter(|| {
            let mut engine = StandardEngine::new(None);
            generate_with_engine(
                &mut engine as &mut dyn larql_kv::KvEngine,
                &weights,
                &tokenizer,
                &ffn,
                &prompt,
                max,
                |_, _| {},
            );
        });
    });

    // A5: async dispatch on `CpuBackend` is a degenerate `Ready*` wrapper.
    // Expected: bit-identical token stream + tok/s within criterion noise
    // of the sync path. Confirms the `BackendSlot::Async` branch +
    // `Ready*` handle allocations don't introduce overhead on the CPU
    // path (the only path that matters until A4 lands real Metal
    // deferred dispatch).
    group.bench_function("engine_dispatch_standard_async", |b| {
        use larql_inference::AsyncComputeBackend;
        b.iter(|| {
            let backend: Box<dyn AsyncComputeBackend> = Box::new(larql_compute::CpuBackend);
            let mut engine = StandardEngine::with_async_backend(None, backend);
            generate_with_engine(
                &mut engine as &mut dyn larql_kv::KvEngine,
                &weights,
                &tokenizer,
                &ffn,
                &prompt,
                max,
                |_, _| {},
            );
        });
    });

    group.finish();
}

/// Compares the per-layer dispatch helpers directly: sync vs async on
/// CpuBackend. Isolates the `attention_*_async` + `read_hidden` + `flush`
/// overhead from the surrounding generate-loop / sampling work.
fn bench_helpers_sync_vs_async(c: &mut Criterion) {
    use larql_inference::kv_dispatch::helpers::{
        kv_decode_step_via_dispatch, kv_decode_step_via_dispatch_async, kv_prefill_via_dispatch,
        kv_prefill_via_dispatch_async,
    };

    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..8).collect();
    let cpu = larql_compute::CpuBackend;

    let mut group = c.benchmark_group("helpers");

    group.bench_function("prefill_sync", |b| {
        b.iter(|| {
            let _ = kv_prefill_via_dispatch(&cpu, &weights, &ffn, &prompt, None, None).unwrap();
        });
    });

    group.bench_function("prefill_async", |b| {
        b.iter(|| {
            let _ =
                kv_prefill_via_dispatch_async(&cpu, &weights, &ffn, &prompt, None, None).unwrap();
        });
    });

    group.bench_function("decode_step_sync", |b| {
        let (_, mut handles) =
            kv_prefill_via_dispatch(&cpu, &weights, &ffn, &prompt, None, None).unwrap();
        let mut pos = prompt.len();
        b.iter(|| {
            let _ =
                kv_decode_step_via_dispatch(&cpu, &weights, &ffn, &mut handles, 1, pos, None, None);
            pos += 1;
        });
    });

    group.bench_function("decode_step_async", |b| {
        let (_, mut handles) =
            kv_prefill_via_dispatch_async(&cpu, &weights, &ffn, &prompt, None, None).unwrap();
        let mut pos = prompt.len();
        b.iter(|| {
            let _ = kv_decode_step_via_dispatch_async(
                &cpu,
                &weights,
                &ffn,
                &mut handles,
                1,
                pos,
                None,
                None,
            );
            pos += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_prefill,
    bench_decode_step,
    bench_engine_vs_legacy_generation,
    bench_helpers_sync_vs_async,
);
criterion_main!(benches);
