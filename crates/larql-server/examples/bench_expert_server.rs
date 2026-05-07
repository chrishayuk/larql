//! Expert-server benchmark — measures real latency, RSS, and mmap behaviour
//! for the remote-MoE expert endpoints against a hybrid-MoE vindex.
//!
//! What this measures (mirrors `bench_embed_server`'s harness, but for the
//! `POST /v1/expert/{layer}/{id}` and `POST /v1/expert/batch` paths):
//!
//!   1. Vindex load time + RSS (full vs `--ffn-only`)
//!   2. First-touch weight-load cost (lazy `get_or_load_weights()`)
//!   3. Single-expert HTTP round-trip latency, warm
//!   4. Batch endpoint latency at K = `top_k_experts`, warm
//!   5. End-to-end `RemoteMoeBackend::forward_moe` (router + dispatch + combine)
//!   6. Local `cpu_moe_forward` floor (no HTTP, same weights)
//!   7. Optional two-shard split: spawn two in-process servers with
//!      `expert_filter = (0..mid)` and `(mid+1..n-1)`, drive through a
//!      multi-shard `RemoteMoeBackend`, measure parallel-dispatch overhead.
//!
//! Usage:
//!   cargo run --release -p larql-server --example bench_expert_server -- \
//!     output/gemma4-26b-a4b-q4k.vindex
//!
//!   # Two-shard split (in-process):
//!   cargo run --release -p larql-server --example bench_expert_server -- \
//!     output/gemma4-26b-a4b-q4k.vindex --two-shard
//!
//! NOTE: in-process two-shard mode shares mmaps, so RSS numbers conflate the
//! two shards. Use single-shard mode for honest RSS; use two-shard mode for
//! parallel-dispatch latency.

use std::path::PathBuf;
use std::sync::{atomic::AtomicU64, Arc};
use std::time::{Duration, Instant};

use tokio::net::TcpListener;

use larql_inference::{
    cpu_moe_forward, MoeLayerWeights, MoeRouterWeights, RemoteMoeBackend, ShardConfig,
};
use larql_server::{
    bootstrap::{load_single_vindex, LoadVindexOptions},
    cache::DescribeCache,
    routes::single_model_router,
    session::SessionManager,
    state::{AppState, LoadedModel},
};

// ── Memory + timing harness ───────────────────────────────────────────────────

fn mem_mb() -> (u64, u64) {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=,vsz=", "-p", &pid])
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            let parts: Vec<&str> = s.split_whitespace().collect();
            let rss = parts
                .first()
                .and_then(|p| p.parse::<u64>().ok())
                .unwrap_or(0);
            let vsz = parts
                .get(1)
                .and_then(|p| p.parse::<u64>().ok())
                .unwrap_or(0);
            (rss / 1024, vsz / 1024)
        }
        Err(_) => (0, 0),
    }
}

fn checkpoint(label: &str, started: Instant, baseline: (u64, u64)) -> (u64, u64) {
    let (rss, vsz) = mem_mb();
    let dr = rss as i64 - baseline.0 as i64;
    println!(
        "  [{:>5.1}s]  {label:<48}  RSS={rss:>6} MB  Δ={dr:>+7} MB  VSZ={vsz:>7} MB",
        started.elapsed().as_secs_f64()
    );
    (rss, vsz)
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f64 * p).round() as usize;
    samples[idx]
}

fn time_ms<F: FnOnce() -> R, R>(f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1000.0)
}

fn bench_remote<F: FnMut() -> Result<(), String>>(
    name: &str,
    warmup: usize,
    iters: usize,
    mut f: F,
) {
    for _ in 0..warmup {
        let _ = f();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f().expect("bench iteration");
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = percentile(&mut samples.clone(), 0.50);
    let p99 = percentile(&mut samples, 0.99);
    println!(
        "  {:<46}  mean={:>7.2} ms  p50={:>7.2} ms  p99={:>7.2} ms  ({} iters)",
        name, mean, p50, p99, iters
    );
}

// ── Server bootstrap helpers ──────────────────────────────────────────────────

fn make_app_state(model: LoadedModel) -> Arc<AppState> {
    Arc::new(AppState {
        models: vec![Arc::new(model)],
        started_at: Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(3600),
        describe_cache: DescribeCache::new(60),
    })
}

async fn spawn_server(model: LoadedModel) -> String {
    let state = make_app_state(model);
    let router = single_model_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// Spawn an in-process server bound to BOTH a TCP socket and a Unix
/// domain socket, returning `(http_url, unix_url)`.  The two listeners
/// share the same `AppState`, so the bench can A/B the same shard via
/// different transports.
async fn spawn_server_with_uds(model: LoadedModel, uds_path: &std::path::Path) -> (String, String) {
    let state = make_app_state(model);
    let router_tcp = single_model_router(state.clone());
    let router_uds = single_model_router(state);

    let tcp_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tcp_addr = tcp_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(tcp_listener, router_tcp).await.unwrap();
    });

    // Unlink any leftover socket from a prior run.
    let _ = std::fs::remove_file(uds_path);
    let uds_listener = tokio::net::UnixListener::bind(uds_path).expect("UDS bind");
    tokio::spawn(async move {
        axum::serve(uds_listener, router_uds).await.unwrap();
    });

    (
        format!("http://{tcp_addr}"),
        format!("unix://{}", uds_path.display()),
    )
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    // Minimal tracing — load_single_vindex emits info!() lines we want to see.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .try_init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: bench_expert_server <vindex_path> [OPTIONS]\n\n\
             OPTIONS:\n  \
               --ffn-only           Skip the f16 gate-vector warmup (faster boot, lazy decode)\n  \
               --two-shard          Spin up 2 in-process shards instead of 1\n  \
               --uds                Bind a Unix domain socket alongside TCP and route the\n                        \
                                    forward_moe call through it (compares ~150 µs/call savings\n                        \
                                    vs TCP loopback).  Sets `--moe-shards unix:///tmp/larql-bench.sock`.\n  \
               --wire f32|f16       Wire format for the layer-batch endpoint.  f16 halves wire\n                        \
                                    bytes; on loopback the f32↔f16 conversion CPU cancels the\n                        \
                                    saving (use on real LAN).  Default f32.\n\n\
             EXAMPLES:\n  \
               cargo run --release -p larql-server --example bench_expert_server -- \\\n      \
                 output/gemma4-26b-a4b-q4k.vindex\n  \
               cargo run --release -p larql-server --example bench_expert_server -- \\\n      \
                 output/gemma4-26b-a4b-q4k.vindex --uds --wire f16"
        );
        std::process::exit(1);
    }
    let vindex_path = PathBuf::from(&args[1]);
    let ffn_only = args.iter().any(|a| a == "--ffn-only");
    let two_shard = args.iter().any(|a| a == "--two-shard");
    let use_uds = args.iter().any(|a| a == "--uds");
    let wire_f16 = args
        .windows(2)
        .find(|w| w[0] == "--wire")
        .map(|w| w[1].as_str() == "f16")
        .unwrap_or(false);

    // The client picks the wire format via env var (read at the first
    // shard.call_layer_batch call by `RemoteMoeBackend`).  Set it here
    // before any shard-side I/O so the choice is sticky.
    if wire_f16 {
        // SAFETY: single-threaded — we're still in the bench's main fn
        // before tokio is built and before any rayon work.
        unsafe {
            std::env::set_var("LARQL_MOE_WIRE_F16", "1");
        }
    }

    println!("LARQL Expert Server Benchmark");
    println!("══════════════════════════════");
    println!("Vindex:    {}", vindex_path.display());
    println!(
        "Mode:      {}",
        if ffn_only { "--ffn-only" } else { "full" }
    );
    println!(
        "Shards:    {}",
        if two_shard { "2 (in-process)" } else { "1" }
    );
    println!(
        "Transport: {}",
        if use_uds {
            "Unix domain socket"
        } else {
            "TCP HTTP"
        }
    );
    println!(
        "Wire:      {}",
        if wire_f16 {
            "f16 (LARQL_MOE_WIRE_F16=1)"
        } else {
            "f32 (default)"
        }
    );
    println!();

    let started = Instant::now();
    let baseline = mem_mb();
    println!("Memory checkpoints:");
    println!("  [  0.0s]  {:<48}  RSS={:>6} MB", "baseline", baseline.0);

    // ── Load primary shard ────────────────────────────────────────────────────
    let opts_a = LoadVindexOptions {
        no_infer: false,
        ffn_only,
        embed_only: false,
        layer_range: None,
        max_gate_cache_layers: 0,
        max_q4k_cache_layers: 0,
        hnsw: None,
        warmup_hnsw: false,
        release_mmap_after_request: false,
        // For one-shard mode, "owns all experts". For two-shard mode, owns the
        // first half — but we set this *after* peeking at num_experts below.
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
    };

    let path_str = args[1].clone();
    let (model_a, load_a_ms) =
        time_ms(|| load_single_vindex(&path_str, opts_a.clone()).expect("load vindex"));
    let after_load_a = checkpoint("after vindex load (shard A)", started, baseline);
    println!("  Shard A load: {:.0} ms", load_a_ms);

    // ── Inspect MoE config ────────────────────────────────────────────────────
    let mc = model_a
        .config
        .model_config
        .as_ref()
        .expect("vindex missing model_config");
    let moe = mc
        .moe
        .as_ref()
        .expect("vindex is not MoE — no `moe` block in model_config");
    let num_experts = moe.num_experts;
    let top_k = moe.top_k;
    let moe_inter = moe.moe_intermediate_size.unwrap_or(0);
    let hidden = model_a.config.hidden_size;
    let num_layers = model_a.config.num_layers;

    println!();
    println!("Model:        {}", model_a.config.model);
    println!("Layers:       {}", num_layers);
    println!("Hidden:       {}", hidden);
    println!("Experts:      {}  (top-K = {})", num_experts, top_k);
    println!("MoE inter:    {}", moe_inter);
    println!("Quant:        {:?}", model_a.config.quant);
    println!("Hybrid MoE:   {}", moe.hybrid);
    println!();

    // ── Force lazy weight load (cheaper to time it explicitly here) ───────────
    let (_, weights_load_ms) = time_ms(|| {
        let _weights = model_a
            .get_or_load_weights()
            .expect("get_or_load_weights on shard A");
    });
    let after_weights = checkpoint("after get_or_load_weights (shard A)", started, baseline);
    println!("  Weights load: {:.0} ms", weights_load_ms);

    // Snapshot everything we need from `weights` into owned data so we can
    // freely move/swap `model_a` later (e.g. for the two-shard re-load).
    // `gu_bytes_owned` / `dn_bytes_owned` carry per-expert byte slices for
    // the bench layer — read from the per-layer Q4_K mmap entries when the
    // vindex carries them, otherwise from the legacy BF16 monolith strides.
    let (
        gu_bytes_owned,
        dn_bytes_owned,
        bench_format,
        router_proj,
        router_scale,
        router_per_expert_scale,
        router_norm,
        pre_experts_norm,
        post_experts_norm,
        router_norm_parameter_free,
        router_input_scalar,
        activation,
        norm_offset,
        eps,
        layer_routers,
    ) = {
        let weights = model_a.get_or_load_weights().unwrap();
        let arch = &*weights.arch;
        let layer = num_layers / 2;
        let (gu_owned, dn_owned, fmt): (Vec<Vec<u8>>, Vec<Vec<u8>>, larql_inference::QuantFormat) =
            if weights.has_per_layer_ffn() {
                let mut gu_v = Vec::with_capacity(num_experts);
                let mut dn_v = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    let (gu, dn) = weights
                        .get_layer_entry_bytes(layer, e)
                        .expect("per-layer entry");
                    gu_v.push(gu.to_vec());
                    dn_v.push(dn.to_vec());
                }
                (gu_v, dn_v, larql_inference::QuantFormat::Q4_K)
            } else {
                let gate_up_key = arch
                    .packed_experts_gate_up_key(layer)
                    .expect("packed gate_up key");
                let down_key = arch
                    .packed_experts_down_key(layer)
                    .expect("packed down key");
                let gu_all = weights
                    .get_packed_bytes(&gate_up_key)
                    .expect("packed gate_up bytes");
                let dn_all = weights
                    .get_packed_bytes(&down_key)
                    .expect("packed down bytes");
                let gu_stride = 2 * moe_inter * hidden * 2;
                let dn_stride = hidden * moe_inter * 2;
                let gu_v: Vec<Vec<u8>> = (0..num_experts)
                    .map(|e| gu_all[e * gu_stride..(e + 1) * gu_stride].to_vec())
                    .collect();
                let dn_v: Vec<Vec<u8>> = (0..num_experts)
                    .map(|e| dn_all[e * dn_stride..(e + 1) * dn_stride].to_vec())
                    .collect();
                (gu_v, dn_v, larql_inference::QuantFormat::BF16)
            };
        let total_gu: usize = gu_owned.iter().map(|b| b.len()).sum();
        let total_dn: usize = dn_owned.iter().map(|b| b.len()).sum();
        println!(
            "  Packed experts (layer {layer}, format={fmt:?}): gate_up={:.1} MB, down={:.1} MB \
             across {} experts",
            total_gu as f64 / 1e6,
            total_dn as f64 / 1e6,
            num_experts
        );

        let rp = arch
            .moe_router_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .cloned()
            .expect("router_proj for bench layer");
        let rs = arch
            .moe_router_scale_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .cloned()
            .unwrap_or_default();
        let rps = arch
            .moe_router_per_expert_scale_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .cloned()
            .unwrap_or_default();
        let rn = arch
            .moe_router_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .cloned()
            .unwrap_or_default();
        let pre = arch
            .moe_pre_experts_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .cloned()
            .unwrap_or_default();
        let post = arch
            .moe_post_experts_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .cloned()
            .unwrap_or_default();
        let rnpf = arch.moe_router_norm_parameter_free();
        let ris = arch.moe_router_input_scalar().unwrap_or(1.0);
        let act = larql_inference::activation_from_arch(arch);
        let no = arch.norm_weight_offset();
        let ep = arch.norm_eps();

        let layer_rs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0
            ..num_layers)
            .map(|l| {
                (
                    arch.moe_router_key(l)
                        .and_then(|k| weights.vectors.get(&k))
                        .cloned()
                        .unwrap_or_default(),
                    arch.moe_router_scale_key(l)
                        .and_then(|k| weights.vectors.get(&k))
                        .cloned()
                        .unwrap_or_default(),
                    arch.moe_router_per_expert_scale_key(l)
                        .and_then(|k| weights.vectors.get(&k))
                        .cloned()
                        .unwrap_or_default(),
                    arch.moe_router_norm_key(l)
                        .and_then(|k| weights.vectors.get(&k))
                        .cloned()
                        .unwrap_or_default(),
                    arch.moe_pre_experts_norm_key(l)
                        .and_then(|k| weights.vectors.get(&k))
                        .cloned()
                        .unwrap_or_default(),
                    arch.moe_post_experts_norm_key(l)
                        .and_then(|k| weights.vectors.get(&k))
                        .cloned()
                        .unwrap_or_default(),
                )
            })
            .collect();

        (
            gu_owned, dn_owned, fmt, rp, rs, rps, rn, pre, post, rnpf, ris, act, no, ep, layer_rs,
        )
    };
    let layer = num_layers / 2;

    // Prepare a residual (fixed seed: not from inference, but stable).
    let h_input: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32 + 1.0) * 0.0007).sin())
        .collect();

    let _ = (after_load_a, after_weights);

    // Apply expert_filter on shard A if two-shard mode.
    let mid = num_experts / 2;
    let model_a = if two_shard {
        // Re-open shard A with expert_filter. Cheap — vindex is already mmapped.
        // (The current LoadedModel doesn't allow mutating expert_filter post-load,
        // so we re-load.  This load is fast because the kernel pages are warm.)
        drop(model_a);
        let opts_a2 = LoadVindexOptions {
            expert_filter: Some((0, mid - 1)),
            ..opts_a.clone()
        };
        let m = load_single_vindex(&path_str, opts_a2).expect("re-load shard A");
        m.get_or_load_weights().ok();
        m
    } else {
        model_a
    };

    // ── Spawn server(s) ───────────────────────────────────────────────────────
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    // When --uds is set, bind both TCP and UDS on shard A and route the
    // bench client through the unix:// URL.  Two-shard mode keeps shard B
    // on TCP only — UDS is fundamentally same-host so multi-shard UDS
    // doesn't change the picture.
    let uds_path_a = std::path::PathBuf::from("/tmp/larql-bench-a.sock");
    let url_a = if use_uds {
        let (http_url, unix_url) = runtime.block_on(spawn_server_with_uds(model_a, &uds_path_a));
        println!();
        println!("Shard A:  TCP {http_url}");
        println!("          UDS {unix_url}  ← bench client routes through this");
        unix_url
    } else {
        let u = runtime.block_on(spawn_server(model_a));
        println!();
        println!(
            "Shard A:  {u}  experts={}",
            if two_shard {
                format!("0..{}", mid - 1)
            } else {
                format!("0..{}", num_experts - 1)
            }
        );
        u
    };

    let url_b = if two_shard {
        let opts_b = LoadVindexOptions {
            expert_filter: Some((mid, num_experts - 1)),
            ..opts_a.clone()
        };
        let (model_b, load_b_ms) = time_ms(|| load_single_vindex(&path_str, opts_b).unwrap());
        let _ = checkpoint("after vindex load (shard B)", started, baseline);
        println!("  Shard B load: {:.0} ms", load_b_ms);
        model_b.get_or_load_weights().ok();
        let _ = checkpoint("after weights (shard B)", started, baseline);
        let url = runtime.block_on(spawn_server(model_b));
        println!("Shard B:  {url}  experts={}..{}", mid, num_experts - 1);
        Some(url)
    } else {
        None
    };

    // ── Build RemoteMoeBackend client ─────────────────────────────────────────
    let shards: Vec<ShardConfig> = if let Some(url_b) = url_b.as_ref() {
        vec![
            ShardConfig::new(0, mid - 1, url_a.clone()).with_timeout(Duration::from_secs(30)),
            ShardConfig::new(mid, num_experts - 1, url_b.clone())
                .with_timeout(Duration::from_secs(30)),
        ]
    } else {
        vec![ShardConfig::new(0, num_experts - 1, url_a.clone())
            .with_timeout(Duration::from_secs(30))]
    };
    let backend = RemoteMoeBackend::connect(shards).expect("RemoteMoeBackend::connect");

    // Tiny sleep so axum is fully bound before first request.
    runtime.block_on(async {
        tokio::time::sleep(Duration::from_millis(50)).await;
    });

    // ── Bench: end-to-end forward_moe ─────────────────────────────────────────
    println!();
    println!("── End-to-end forward_moe (router + dispatch + combine) ──");
    let router = MoeRouterWeights {
        router_proj: &router_proj,
        router_scale: &router_scale,
        router_per_expert_scale: &router_per_expert_scale,
        router_norm: &router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        pre_experts_norm: &pre_experts_norm,
        post_experts_norm: &post_experts_norm,
        num_experts,
        top_k,
    };

    bench_remote(
        &format!(
            "forward_moe layer={layer} top_k={top_k} ({})",
            if two_shard { "2 shards" } else { "1 shard" }
        ),
        5,
        50,
        || {
            backend
                .forward_moe(layer, &h_input, &router, norm_offset, eps)
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
    );
    let _ = checkpoint("after forward_moe warm", started, baseline);

    // ── Bench: local cpu_moe_forward floor (no HTTP) ──────────────────────────
    println!();
    println!("── Local floor: cpu_moe_forward (no HTTP, same weights) ──");
    // Per-expert byte tables already snapshotted per format above.
    let experts_gate_up_local: Vec<&[u8]> = gu_bytes_owned.iter().map(|v| v.as_slice()).collect();
    let experts_down_local: Vec<&[u8]> = dn_bytes_owned.iter().map(|v| v.as_slice()).collect();
    let layer_w = MoeLayerWeights {
        experts_gate_up: experts_gate_up_local,
        experts_down: experts_down_local,
        router_proj: &router_proj,
        router_scale: &router_scale,
        router_per_expert_scale: &router_per_expert_scale,
        router_norm: &router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        pre_experts_norm: &pre_experts_norm,
        post_ffn1_norm: &[],
        post_experts_norm: &post_experts_norm,
        num_experts,
        top_k,
        intermediate_size: moe_inter,
        activation,
        expert_data_format: bench_format,
    };
    bench_remote(
        &format!("cpu_moe_forward layer={layer} top_k={top_k}"),
        5,
        50,
        || {
            let _ = cpu_moe_forward(&h_input, &layer_w, norm_offset, eps);
            Ok(())
        },
    );

    // ── Bench: walking layers 0..num_layers via forward_moe ───────────────────
    // Simulates one decode-step's worth of MoE blocks across all layers.
    println!();
    println!("── Multi-layer fan-out (1 decode step worth of MoE blocks) ──");

    // Filter to MoE-bearing layers (some hybrid layers have no router).
    let moe_layers: Vec<usize> = layer_routers
        .iter()
        .enumerate()
        .filter(|(_, (rp, _, _, _, _, _))| !rp.is_empty())
        .map(|(i, _)| i)
        .collect();

    if !moe_layers.is_empty() {
        println!(
            "  MoE-bearing layers: {}/{}  (first={}, last={})",
            moe_layers.len(),
            num_layers,
            moe_layers.first().unwrap(),
            moe_layers.last().unwrap()
        );

        // Warm: 3 full sweeps before timing.
        for _ in 0..3 {
            for &l in &moe_layers {
                let r = &layer_routers[l];
                let router = MoeRouterWeights {
                    router_proj: &r.0,
                    router_scale: &r.1,
                    router_per_expert_scale: &r.2,
                    router_norm: &r.3,
                    router_norm_parameter_free,
                    router_input_scalar,
                    pre_experts_norm: &r.4,
                    post_experts_norm: &r.5,
                    num_experts,
                    top_k,
                };
                let _ = backend.forward_moe(l, &h_input, &router, norm_offset, eps);
            }
        }

        let mut sweep_samples: Vec<f64> = Vec::with_capacity(20);
        for _ in 0..20 {
            let t = Instant::now();
            for &l in &moe_layers {
                let r = &layer_routers[l];
                let router = MoeRouterWeights {
                    router_proj: &r.0,
                    router_scale: &r.1,
                    router_per_expert_scale: &r.2,
                    router_norm: &r.3,
                    router_norm_parameter_free,
                    router_input_scalar,
                    pre_experts_norm: &r.4,
                    post_experts_norm: &r.5,
                    num_experts,
                    top_k,
                };
                backend
                    .forward_moe(l, &h_input, &router, norm_offset, eps)
                    .expect("multi-layer forward_moe");
            }
            sweep_samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mean = sweep_samples.iter().sum::<f64>() / sweep_samples.len() as f64;
        let p50 = percentile(&mut sweep_samples.clone(), 0.50);
        let p99 = percentile(&mut sweep_samples, 0.99);
        let per_layer = mean / moe_layers.len() as f64;
        println!(
            "  full sweep ({} layers):  mean={:.2} ms  p50={:.2} ms  p99={:.2} ms  ({:.2} ms/layer)",
            moe_layers.len(),
            mean,
            p50,
            p99,
            per_layer
        );
    } else {
        println!("  No MoE-bearing layers found — skipping multi-layer sweep");
    }

    // ── Final memory ──────────────────────────────────────────────────────────
    println!();
    let final_rss = checkpoint("steady state", started, baseline);
    let total_alloc = (final_rss.0 as i64) - (baseline.0 as i64);
    println!();
    println!(
        "Total RSS allocated:  {:>+7} MB    Total time: {:.1} s",
        total_alloc,
        started.elapsed().as_secs_f64()
    );
}
