//! Criterion benches for the Exp 53 `ShardSource` lookup path.
//!
//! Two scenarios at a few `(n_entries, d, k)` shapes:
//!
//! - `ShardSource::Cache` — flat `Vec<f32>` in-memory KNN. Mirrors the
//!   Python prototype's data layout; gives the upper bound on what
//!   "no vindex overhead" looks like.
//! - `ShardSource::Vindex` — production. Queries a `PatchedVindex`
//!   loaded with `insert_feature` patches via `gate_knn` +
//!   `ffn_row_into`.
//!
//! These are **in-process** micro-benches: no tonic, no TCP. They
//! isolate the dispatch + KNN cost, not the wire path. The Python
//! prototype's 0.085 ms loopback baseline includes TCP framing; ours
//! beats it by orders of magnitude because criterion measures pure
//! compute. Use them to catch regressions in the KNN inner loop, not
//! to compare end-to-end latency.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_models::TopKEntry;
use larql_server::shard_query::{ShardCache, ShardSource};
use larql_vindex::{FeatureMeta, PatchedVindex, VectorIndex};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;

/// Construct an in-memory `ShardCache` source with `n` pre-normed
/// entries at hidden dim `d`. Inputs span the unit basis so a query
/// equal to `[1, 0, …]` returns row 0 with cos = 1.
fn make_cache_source(n: usize, d: usize) -> ShardSource {
    let mut cache = ShardCache::new(0.5);
    // Build n unit-norm vectors: row i has a 1 at column (i % d), 0
    // elsewhere. Outputs are arbitrary but deterministic.
    let mut inputs = vec![0.0f32; n * d];
    let mut outputs = vec![0.0f32; n * d];
    for i in 0..n {
        inputs[i * d + (i % d)] = 1.0;
        for j in 0..d {
            outputs[i * d + j] = ((i + j) as f32) * 0.1;
        }
    }
    cache.seed_from_normed(0, inputs, outputs, n, d).unwrap();
    ShardSource::cache(Arc::new(RwLock::new(cache)))
}

/// Construct a `PatchedVindex` source with `n` gate-only patches at
/// layer 0. No down weights wired — the lookup returns a miss-with-
/// best_sim, which still exercises the full `gate_knn` matvec.
/// (Wiring real down rows would require f32 mmap-backed FFN storage,
/// which the production deploy uses but isn't worth setting up here
/// just to add ~1 µs per call to the read path.)
fn make_vindex_source(n: usize, d: usize) -> ShardSource {
    let base = VectorIndex::new(vec![None], vec![None], 1, d);
    let mut patched = PatchedVindex::new(base);
    for i in 0..n {
        let mut gate = vec![0.0f32; d];
        gate[i % d] = 1.0;
        let meta = FeatureMeta {
            top_token: format!("f{i}"),
            top_token_id: i as u32,
            c_score: 1.0,
            top_k: vec![TopKEntry {
                token: format!("f{i}"),
                token_id: i as u32,
                logit: 1.0,
            }],
        };
        patched.insert_feature(0, i, gate, meta);
    }
    ShardSource::vindex(Arc::new(RwLock::new(patched)), 0.5)
}

/// Standard query: unit vector along axis 0. Hits the synthetic
/// row 0 at cos = 1.
fn query_vec(d: usize) -> Vec<f32> {
    let mut q = vec![0.0f32; d];
    q[0] = 1.0;
    q
}

fn bench_cache_lookup(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("shard_query/cache_lookup");
    // (n_entries, d) — matches the Python prototype's order of magnitude
    // (n≈50, d=2816 for Gemma 3 4B) plus a smaller and a larger sweep
    // point for shape sensitivity.
    for &(n, d) in &[(16usize, 256usize), (64, 1024), (256, 1024), (64, 2816)] {
        let source = make_cache_source(n, d);
        let q = query_vec(d);
        for &k in &[1usize, 4] {
            let id = BenchmarkId::new(format!("n{n}_d{d}_k{k}"), n * d);
            group.bench_with_input(id, &q, |b, q| {
                b.iter(|| {
                    rt.block_on(async { black_box(source.lookup(0, black_box(q), k, 0.5).await) })
                });
            });
        }
    }
    group.finish();
}

fn bench_vindex_lookup(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("shard_query/vindex_lookup");
    for &(n, d) in &[(16usize, 256usize), (64, 1024), (256, 1024)] {
        let source = make_vindex_source(n, d);
        let q = query_vec(d);
        // Only k=1: the vindex path with no down weights does
        // gate_knn → tau check → fall-through-miss for both k values;
        // exercising k=4 here would measure the same code.
        let id = BenchmarkId::new(format!("n{n}_d{d}_k1"), n * d);
        group.bench_with_input(id, &q, |b, q| {
            b.iter(|| {
                rt.block_on(async { black_box(source.lookup(0, black_box(q), 1, 0.5).await) })
            });
        });
    }
    group.finish();
}

/// Cross-layer mutation scenario: mutate layer 1 between every
/// query at layer 0. With whole-cache invalidation, every mutation
/// would evict layer 0's cache and the next query rebuilds it.
/// With per-layer invalidation (2026-05-16), layer 0 stays hot —
/// the iter measures cache-hit cost + a layer-1 patch insert.
///
/// Useful as a regression guard: if per-layer invalidation breaks
/// and falls back to whole-cache clear, this bench inflates by the
/// rebuild cost (≈ matvec time at layer 0's n×d shape).
fn bench_vindex_cross_layer_mutation(c: &mut Criterion) {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("shard_query/vindex_cross_layer_mutation");
    for &(n, d) in &[(64usize, 1024usize), (256, 1024)] {
        // Build a PatchedVindex with patches at BOTH layer 0 (the
        // query target) and layer 1 (the mutation target). Layer 1
        // gets a single patch initially; the bench loop replaces /
        // re-inserts it on every iter.
        let base = VectorIndex::new(vec![None, None], vec![None, None], 2, d);
        let mut patched = PatchedVindex::new(base);
        let meta = |feat: u32| FeatureMeta {
            top_token: format!("f{feat}"),
            top_token_id: feat,
            c_score: 1.0,
            top_k: vec![TopKEntry {
                token: format!("f{feat}"),
                token_id: feat,
                logit: 1.0,
            }],
        };
        for i in 0..n {
            let mut gate = vec![0.0f32; d];
            gate[i % d] = 1.0;
            patched.insert_feature(0, i, gate, meta(i as u32));
        }
        // Seed one patch at layer 1 so the cross-layer mutation
        // path is exercised consistently (insert vs. update has the
        // same invalidation cost on this code path).
        patched.insert_feature(1, 0, vec![1.0f32; d], meta(1000));
        let vindex = Arc::new(RwLock::new(patched));
        let source = ShardSource::vindex(Arc::clone(&vindex), 0.5);
        let q = query_vec(d);

        // Warm the layer-0 cache once outside the timed loop.
        rt.block_on(async { source.lookup(0, &q, 1, 0.5).await });

        let id = BenchmarkId::new(format!("n{n}_d{d}_k1"), n * d);
        group.bench_with_input(id, &q, |b, q| {
            let mut counter: u32 = 0;
            b.iter(|| {
                // Cross-layer mutation: re-insert at layer 1 with a
                // slightly different gate so it isn't optimised away.
                counter = counter.wrapping_add(1);
                let new_gate = vec![counter as f32; d];
                {
                    let mut w = vindex.blocking_write();
                    w.insert_feature(1, 0, new_gate, meta(counter));
                }
                // Then query layer 0 — should still hit the cache
                // because layer 1's invalidation is layer-scoped.
                rt.block_on(async { black_box(source.lookup(0, black_box(q), 1, 0.5).await) })
            });
        });
    }
    group.finish();
}

/// A/B counterpart to [`bench_vindex_cross_layer_mutation`]: simulates
/// what the old whole-cache invalidation would have cost by forcing
/// layer 0's cache to rebuild after every cross-layer mutation. Uses
/// `insert_feature(layer=0, feat=0, …)` to re-insert the same gate
/// vector at layer 0 — a no-op data change that still invalidates
/// the cache via the public mutator path. The delta between this
/// bench and the per-layer one is the savings of per-layer
/// invalidation in cross-layer workloads.
fn bench_vindex_cross_layer_mutation_with_full_invalidation(c: &mut Criterion) {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("shard_query/vindex_cross_layer_mutation_full_invalidation");
    for &(n, d) in &[(64usize, 1024usize), (256, 1024)] {
        let base = VectorIndex::new(vec![None, None], vec![None, None], 2, d);
        let mut patched = PatchedVindex::new(base);
        let meta = |feat: u32| FeatureMeta {
            top_token: format!("f{feat}"),
            top_token_id: feat,
            c_score: 1.0,
            top_k: vec![TopKEntry {
                token: format!("f{feat}"),
                token_id: feat,
                logit: 1.0,
            }],
        };
        for i in 0..n {
            let mut gate = vec![0.0f32; d];
            gate[i % d] = 1.0;
            patched.insert_feature(0, i, gate, meta(i as u32));
        }
        patched.insert_feature(1, 0, vec![1.0f32; d], meta(1000));
        let vindex = Arc::new(RwLock::new(patched));
        let source = ShardSource::vindex(Arc::clone(&vindex), 0.5);
        let q = query_vec(d);
        // Same warmup as the per-layer bench so both start cache-hot.
        rt.block_on(async { source.lookup(0, &q, 1, 0.5).await });
        let mut feat0_gate = vec![0.0f32; d];
        feat0_gate[0] = 1.0;

        let id = BenchmarkId::new(format!("n{n}_d{d}_k1"), n * d);
        group.bench_with_input(id, &q, |b, q| {
            let mut counter: u32 = 0;
            b.iter(|| {
                counter = counter.wrapping_add(1);
                let new_gate_l1 = vec![counter as f32; d];
                {
                    let mut w = vindex.blocking_write();
                    // Cross-layer mutation as before…
                    w.insert_feature(1, 0, new_gate_l1, meta(counter));
                    // …plus a no-op data update at layer 0 that
                    // re-invalidates layer 0's cache. This simulates
                    // what the prior whole-cache `g.clear()` did to
                    // every layer on every mutation.
                    w.insert_feature(0, 0, feat0_gate.clone(), meta(0));
                }
                rt.block_on(async { black_box(source.lookup(0, black_box(q), 1, 0.5).await) })
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cache_lookup,
    bench_vindex_lookup,
    bench_vindex_cross_layer_mutation,
    bench_vindex_cross_layer_mutation_with_full_invalidation,
);
criterion_main!(benches);
