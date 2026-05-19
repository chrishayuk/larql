//! Criterion benchmarks for the grid routing hot-path (ADR-0012).
//!
//! Measures ns/op for the operations that run on every inference request
//! going through the router.
//!
//! Run with:
//!   cargo bench -p larql-router --bench routing
//!
//! All bench IDs use server counts and layer counts, not model names.

use std::collections::HashMap;
use std::time::Instant;

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use parking_lot::RwLock;

use larql_router::grid::{GridState, ServerEntry};
use larql_router_protocol::LayerLatency;

const SERVER_COUNTS: &[(usize, &str)] = &[(1, "1srv"), (10, "10srv"), (100, "100srv")];
const LAYER_COUNTS: &[(usize, &str)] = &[(30, "30layers"), (62, "62layers")];

fn make_entry(id: usize, layer_start: u32, layer_end: u32) -> ServerEntry {
    ServerEntry {
        server_id: format!("srv-{id}"),
        listen_url: format!("http://10.0.0.{id}:8080"),
        model_id: "bench-model".into(),
        layer_start,
        layer_end,
        vindex_hash: format!("hash-{id}"),
        cpu_pct: 0.0,
        ram_used: 4 * 1024 * 1024 * 1024,
        requests_in_flight: id as u32 % 10,
        last_seen: Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
    }
}

/// Build a worst-case state: `n_servers` each owning all `n_layers`
/// layers (full replication). Useful as an upper bound but not what
/// real deployments look like.
fn build_state(n_servers: usize, n_layers: usize) -> GridState {
    let mut state = GridState::default();
    for i in 0..n_servers {
        state.register(make_entry(i, 0, (n_layers - 1) as u32));
    }
    state
}

/// Build a production-shape grid: a model with `n_layers` layers
/// partitioned into `n_shards` contiguous slices, each slice
/// replicated `n_replicas` times. Total servers = n_shards ×
/// n_replicas, replicas-per-layer = n_replicas (constant).
fn build_realistic_state(n_layers: usize, n_shards: usize, n_replicas: usize) -> GridState {
    let mut state = GridState::default();
    let layers_per_shard = n_layers / n_shards;
    for shard_idx in 0..n_shards {
        let layer_start = (shard_idx * layers_per_shard) as u32;
        let layer_end = if shard_idx == n_shards - 1 {
            (n_layers - 1) as u32
        } else {
            ((shard_idx + 1) * layers_per_shard - 1) as u32
        };
        for replica_idx in 0..n_replicas {
            let server_id = shard_idx * n_replicas + replica_idx;
            state.register(make_entry(server_id, layer_start, layer_end));
        }
    }
    state
}

// ── route() hot path ──────────────────────────────────────────────────────────

fn bench_route_single_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_single_layer");
    for &(n_servers, slabel) in SERVER_COUNTS {
        let state = build_state(n_servers, 30);
        group.bench_with_input(BenchmarkId::new(slabel, n_servers), &n_servers, |b, _| {
            b.iter(|| state.route(Some("bench-model"), 15));
        });
    }
    group.finish();
}

// ── route_all() — full forward pass routing ───────────────────────────────────

fn bench_route_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_all");
    for &(n_servers, slabel) in SERVER_COUNTS {
        for &(n_layers, llabel) in LAYER_COUNTS {
            let state = build_state(n_servers, n_layers);
            let layers: Vec<usize> = (0..n_layers).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("{slabel}_{llabel}"), n_servers * n_layers),
                &layers,
                |b, layers| {
                    b.iter(|| state.route_all(Some("bench-model"), layers));
                },
            );
        }
    }
    group.finish();
}

// ── update_heartbeat() — load metric update ───────────────────────────────────

fn bench_heartbeat_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/heartbeat_update");
    for &(n_servers, slabel) in SERVER_COUNTS {
        let mut state = build_state(n_servers, 30);
        let server_ids: Vec<String> = (0..n_servers).map(|i| format!("srv-{i}")).collect();
        let layer_stats: Vec<LayerLatency> = (0..30u32)
            .map(|l| LayerLatency {
                layer: l,
                avg_ms: 2.0,
                p99_ms: 5.0,
            })
            .collect();
        group.bench_with_input(
            BenchmarkId::new(slabel, n_servers),
            &server_ids,
            |b, ids| {
                b.iter(|| {
                    // Update the first server's heartbeat.
                    state.update_heartbeat(&ids[0], 50.0, 2 << 30, 5, layer_stats.clone(), 0.0);
                });
            },
        );
    }
    group.finish();
}

// ── Production-shape route() — replicas-per-layer stays at n_replicas ────────

/// Production-shape scenarios: (n_shards, n_replicas, label).
/// Total servers = n_shards × n_replicas; replicas-per-layer = n_replicas.
const REALISTIC_TOPOLOGIES: &[(usize, usize, &str)] = &[
    (2, 2, "2shards_x2"),   // 4 servers
    (5, 2, "5shards_x2"),   // 10 servers
    (10, 2, "10shards_x2"), // 20 servers
    (10, 3, "10shards_x3"), // 30 servers
    (20, 2, "20shards_x2"), // 40 servers
];

fn bench_route_single_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_realistic");
    for &(n_shards, n_replicas, label) in REALISTIC_TOPOLOGIES {
        let state = build_realistic_state(30, n_shards, n_replicas);
        group.bench_with_input(
            BenchmarkId::new(label, n_shards * n_replicas),
            &(),
            |b, _| {
                // Route the middle layer — typical request shape.
                b.iter(|| state.route(Some("bench-model"), 15));
            },
        );
    }
    group.finish();
}

fn bench_route_all_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_all_realistic");
    let scenarios = &[
        (30, 2, 2, "30layers_2shards_x2"),
        (30, 5, 2, "30layers_5shards_x2"),
        (30, 10, 2, "30layers_10shards_x2"),
        (62, 10, 2, "62layers_10shards_x2"),
        (62, 20, 2, "62layers_20shards_x2"),
    ];
    for &(n_layers, n_shards, n_replicas, label) in scenarios {
        let state = build_realistic_state(n_layers, n_shards, n_replicas);
        let layers: Vec<usize> = (0..n_layers).collect();
        group.bench_with_input(
            BenchmarkId::new(label, n_shards * n_replicas),
            &layers,
            |b, layers| {
                b.iter(|| state.route_all(Some("bench-model"), layers));
            },
        );
    }
    group.finish();
}

// ── ADR-0018: MoE expert-routing benches ─────────────────────────────────────

/// Build a per-(layer, expert-range) MoE topology mirroring real
/// large-MoE deployments (DeepSeek-V3-style): one server per (layer,
/// expert-subset) tuple, no layer overlap across servers.
///
/// Args:
///   `n_layers`            — total layers in the model
///   `n_expert_shards`     — how many expert shards split each layer
///   `experts_per_shard`   — width of each expert range (sets the
///                           total expert_count per layer = shards × width)
///   `n_replicas`          — replicas per (layer, expert-range) tuple
///
/// Returns a `GridState` with
/// `n_layers × n_expert_shards × n_replicas` servers registered.
fn build_moe_state(
    n_layers: usize,
    n_expert_shards: usize,
    experts_per_shard: u32,
    n_replicas: usize,
) -> GridState {
    let mut state = GridState::default();
    let mut sid = 0;
    for layer in 0..n_layers as u32 {
        for shard_idx in 0..n_expert_shards as u32 {
            let expert_start = shard_idx * experts_per_shard;
            let expert_end = expert_start + experts_per_shard - 1;
            for _ in 0..n_replicas {
                let mut entry = make_entry(sid, layer, layer);
                entry.expert_start = expert_start;
                entry.expert_end = expert_end;
                state.register(entry);
                sid += 1;
            }
        }
    }
    state
}

/// `route_expert()` against a single (layer, expert) — the hot path
/// inside per-token MoE fan-out. K2 / DeepSeek-V3 style topologies.
fn bench_route_expert_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_expert_single");
    let scenarios = &[
        // (n_layers, n_expert_shards, experts_per_shard, n_replicas, label)
        (1, 4, 64, 1, "1layer_4shards_x256experts"), // V3-like single layer
        (1, 8, 16, 1, "1layer_8shards_x128experts"), // K2-like single layer
        (60, 4, 64, 1, "60layers_4shards_x256experts_full_v3"),
        (60, 4, 64, 2, "60layers_4shards_x256experts_x2"),
    ];
    for &(n_layers, n_expert_shards, experts_per_shard, n_replicas, label) in scenarios {
        let state = build_moe_state(n_layers, n_expert_shards, experts_per_shard, n_replicas);
        let total_servers = n_layers * n_expert_shards * n_replicas;
        // Pick a middle expert from a middle layer for the lookup.
        let target_layer = (n_layers / 2) as u32;
        let target_expert = (n_expert_shards as u32 / 2) * experts_per_shard + 1;
        group.bench_with_input(BenchmarkId::new(label, total_servers), &(), |b, _| {
            b.iter(|| state.route_expert(Some("bench-model"), target_layer, target_expert));
        });
    }
    group.finish();
}

/// `route_all_experts()` — batched per-token fan-out. Models a
/// top-K expert selection across many layers.
fn bench_route_all_experts(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_all_experts");
    let scenarios = &[
        // (n_layers, n_expert_shards, experts_per_shard, top_k, label)
        (32, 4, 2, 2, "32layers_top2_mixtral_8x"), // Mixtral 8×7B
        (60, 4, 64, 6, "60layers_top6_v3"),        // DeepSeek-V3 (top-6 of 256)
        (60, 4, 64, 8, "60layers_top8_v3_aggr"),   // V3 aggressive
        (80, 4, 32, 8, "80layers_top8_kimi_style"), // K2-ish 80 layers
    ];
    for &(n_layers, n_expert_shards, experts_per_shard, top_k, label) in scenarios {
        let state = build_moe_state(n_layers, n_expert_shards, experts_per_shard, 1);
        // Build the (layer, expert_id) request list — top-K per layer.
        // Spread the top-K samples across the full expert range so each
        // one lands on a valid shard (avoiding bench short-circuits).
        let total_experts_per_layer = n_expert_shards as u32 * experts_per_shard;
        let stride = total_experts_per_layer / top_k as u32;
        let layer_experts: Vec<(usize, u32)> = (0..n_layers)
            .flat_map(|layer| (0..top_k as u32).map(move |k| (layer, k * stride)))
            .collect();
        let n_pairs = layer_experts.len();
        group.bench_with_input(
            BenchmarkId::new(label, n_pairs),
            &layer_experts,
            |b, layer_experts| {
                b.iter(|| state.route_all_experts(Some("bench-model"), layer_experts));
            },
        );
    }
    group.finish();
}

// ── Single register: cost of one rebuild_route_table call ────────────────────

/// Measure the cost of *one* server joining a grid of size N. Each
/// `register()` triggers exactly one `rebuild_route_table()` over
/// N+1 servers, so this isolates the per-rebuild cost that the
/// `register_cascade` bench below conflates across N registrations.
fn bench_single_register(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/single_register");
    for &(n_servers, slabel) in SERVER_COUNTS {
        for &(n_layers, llabel) in LAYER_COUNTS {
            group.bench_with_input(
                BenchmarkId::new(format!("{slabel}_{llabel}"), n_servers * n_layers),
                &(n_servers, n_layers),
                |b, &(ns, nl)| {
                    b.iter_batched(
                        || build_state(ns, nl),
                        |mut state| {
                            // One register = one rebuild over (ns + 1) servers.
                            state.register(make_entry(ns, 0, (nl - 1) as u32));
                            state
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

// ── register_cascade — building a grid from scratch (N registers) ────────────

/// Build an N-server grid from empty by calling `register()` N times.
/// This is O(N² × L) because each register triggers a full
/// `rebuild_route_table()` over the growing set. Useful as a
/// worst-case "cold start" measurement but not representative of
/// real per-join cost — for that, use [`bench_single_register`].
fn bench_register_cascade(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/register_cascade");
    for &(n_servers, slabel) in SERVER_COUNTS {
        for &(n_layers, llabel) in LAYER_COUNTS {
            group.bench_with_input(
                BenchmarkId::new(format!("{slabel}_{llabel}"), n_servers * n_layers),
                &(n_servers, n_layers),
                |b, &(ns, nl)| {
                    b.iter(|| {
                        let mut state = GridState::default();
                        for i in 0..ns {
                            state.register(make_entry(i, 0, (nl - 1) as u32));
                        }
                        state
                    });
                },
            );
        }
    }
    group.finish();
}

// ── ADR-0020: saturation filter overhead ─────────────────────────────────────

/// Builds a production-shape topology and pre-loads every replica's
/// `requests_in_flight` to a known level. Used to measure the cost of
/// the ADR-0020 saturation filter in `route()` against the no-filter
/// baseline.
fn build_loaded_state(
    n_shards: usize,
    n_replicas: usize,
    n_layers: usize,
    in_flight_per_replica: u32,
) -> GridState {
    let mut state = build_realistic_state(n_layers, n_shards, n_replicas);
    let server_ids: Vec<String> = state.servers().map(|(id, _)| id.clone()).collect();
    for id in server_ids {
        state.update_heartbeat(&id, 50.0, 1 << 30, in_flight_per_replica, Vec::new(), 0.0);
    }
    state
}

/// ADR-0020 — three scenarios across the same topology:
///   * ceiling=None: filter disabled (baseline)
///   * ceiling=Some(N), all replicas under N: filter walks but never trims
///   * ceiling=Some(N), all replicas at N: filter trims every candidate
///     → `route()` short-circuits to `None` (503 in production)
fn bench_route_saturation_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/saturation_filter");
    let scenarios: &[(usize, usize, &str)] = &[
        (5, 2, "5shards_x2"),
        (10, 2, "10shards_x2"),
        (20, 2, "20shards_x2"),
    ];
    for &(n_shards, n_replicas, label) in scenarios {
        // Each replica at in_flight = 4. With ceiling=16 they're all
        // well below; with ceiling=4 they're all at the ceiling.
        let mut state = build_loaded_state(n_shards, n_replicas, 30, 4);

        // Baseline: filter disabled.
        state.set_saturation_ceiling(None);
        group.bench_with_input(
            BenchmarkId::new(format!("{label}_no_filter"), n_shards * n_replicas),
            &(),
            |b, _| b.iter(|| state.route(Some("bench-model"), 15)),
        );

        // Filter enabled but no replica saturated — measures pure
        // filter overhead on the success path.
        state.set_saturation_ceiling(Some(16));
        group.bench_with_input(
            BenchmarkId::new(format!("{label}_filter_all_unsat"), n_shards * n_replicas),
            &(),
            |b, _| b.iter(|| state.route(Some("bench-model"), 15)),
        );

        // Every replica at ceiling — `route()` returns `None`. Confirms
        // the saturation short-circuit doesn't degrade vs the success
        // path, and gives operators a number to put next to the 503
        // counter.
        state.set_saturation_ceiling(Some(4));
        group.bench_with_input(
            BenchmarkId::new(format!("{label}_filter_all_sat"), n_shards * n_replicas),
            &(),
            |b, _| b.iter(|| state.route(Some("bench-model"), 15)),
        );
    }
    group.finish();
}

// ── Concurrent route() — RwLock contention (Throughput P1) ───────────────────

/// Drive `route()` from N parallel tokio tasks against a single
/// `Arc<RwLock<GridState>>` — matches the lock shape used in
/// `crates/larql-router/src/http.rs::AppState::resolve_all`. Surfaces
/// read-lock contention before production does.
///
/// Criterion's `Throughput::Elements` reports req/s; per-route ns
/// drops out of the timing data. Topology is fixed at the production-
/// shape middle scenario (10 shards × 2 replicas, 30 layers) so the
/// only axis is worker count.
fn bench_route_concurrent(c: &mut Criterion) {
    const ROUTES_PER_WORKER: usize = 256;
    let worker_counts: &[usize] = &[1, 4, 8, 16];

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    let grid = Arc::new(RwLock::new(build_realistic_state(30, 10, 2)));

    let mut group = c.benchmark_group("routing/route_concurrent");
    for &n_workers in worker_counts {
        group.throughput(Throughput::Elements((n_workers * ROUTES_PER_WORKER) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_workers),
            &n_workers,
            |b, &n_workers| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::with_capacity(n_workers);
                        for w in 0..n_workers {
                            let grid = grid.clone();
                            handles.push(tokio::spawn(async move {
                                // Each worker rotates through layers so the
                                // bench isn't degenerate on one route_table
                                // bucket; that would understate contention
                                // if `route()` ever cached recent results.
                                for i in 0..ROUTES_PER_WORKER {
                                    let layer = ((w * ROUTES_PER_WORKER + i) % 30) as u32;
                                    let _ = grid.read().route(Some("bench-model"), layer);
                                }
                            }));
                        }
                        for h in handles {
                            h.await.unwrap();
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_route_single_layer,
    bench_route_all,
    bench_route_single_realistic,
    bench_route_all_realistic,
    bench_route_expert_single,
    bench_route_all_experts,
    bench_heartbeat_update,
    bench_single_register,
    bench_register_cascade,
    bench_route_saturation_filter,
    bench_route_concurrent,
);
criterion_main!(benches);
