//! GQA-RE-READ — does a query head re-reading its KV head's cache cost
//! DRAM traffic, at Gemma 3 4B's attention geometry?
//!
//! The VERIFY-N long-context arm put LARQL's context slope at ~4x MLX's
//! (2.97 vs 0.71 ms/token over ~800 tokens of average context). Two
//! structural candidates: f32 KV (2x MLX's bf16) and the one-threadgroup-
//! per-QUERY-head dispatch, where Gemma's 8 query heads over 4 KV heads
//! read every K/V row twice. The second only costs if the duplicate read
//! misses cache — two threadgroups sharing ~2 MB of one KV head's f32 K+V
//! at 1K may well meet in the SLC. This bench decides it before a
//! GQA-grouped kernel is written.
//!
//! ## Arms (all the production seqpar kernel, 4 slices = the Gemma row)
//!
//! | arm        | TGs | distinct KV streams | reads per stream |
//! |------------|-----|---------------------|------------------|
//! | 4q / 4kv   | 4   | 4                   | 1                |
//! | 8q / 4kv   | 8   | 4                   | 2  ← production  |
//! | 8q / 8kv   | 8   | 8                   | 1                |
//! | 16q / 4kv  | 16  | 4                   | 4                |
//! | 16q / 16kv | 16  | 16                  | 1                |
//!
//! Only 8 threadgroups on a 40-core device means TG count and bytes are
//! confounded; the pairs at fixed TG count separate them:
//!
//! - `8q/4kv ≈ 8q/8kv` → a shared re-read costs as much as a distinct
//!   read: grouping halves real traffic → build the GQA-grouped kernel.
//! - `8q/4kv ≈ 4q/4kv` and `8q/8kv` slower → the re-read is cache-hot and
//!   bytes do bind: grouping buys little; half-precision KV is the lever.
//! - all arms ≈ equal → per-threadgroup latency bound, bytes do not bind:
//!   neither lever as forecast; the lever is more threadgroups per head.
//!
//! The comparison is the SLOPE in span, not the intercept: a differing
//! intercept with an equal slope is scheduling, not KV traffic.
//!
//! Methodology as `bench_attention_span`: `LAYERS` dispatches per command
//! buffer (serial encoder, so they do not overlap — as in decode), each
//! with its own cache so the batch reads cold; median of `ITERS`; the
//! production arm is measured again after the others and a row whose two
//! readings disagree by more than `DRIFT_LIMIT_PCT` is flagged.
//!
//! Run: `cargo run --release -p larql-compute-metal --example bench_attention_gqa`

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("bench_attention_gqa requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute_metal::ops::kv_cache::{
        LayerKVCache, LONG_ATTENTION_SPAN, SHORT_ATTENTION_SPAN,
    };
    use larql_compute_metal::MetalBackend;

    // Gemma 3 4B attention geometry (system_graph.json of the vindex).
    const HEAD_DIM: usize = 256;
    /// Gemma's measured seqpar row: `(256, 8, 4) tiers [(0, 4)]`.
    const SLICES: usize = 4;
    /// One dispatch per decoder layer.
    const LAYERS: usize = 34;
    const WARMUP: usize = 8;
    const ITERS: usize = 40;
    const DRIFT_LIMIT_PCT: f64 = 5.0;
    /// Widest KV-head count of any arm; every cache is allocated at it
    /// and narrower arms use its prefix (the kernel strides by `num_kv`).
    const MAX_KV_HEADS: usize = 16;

    /// (q heads, kv heads). Index 1 is production.
    const ARMS: &[(usize, usize)] = &[(4, 4), (8, 4), (8, 8), (16, 4), (16, 16)];
    const PRODUCTION: usize = 1;
    const SPANS: &[u32] = &[128, 256, 512, 1024, 1536, 2048, 3072, 4096];

    let Some(metal) = MetalBackend::new() else {
        eprintln!("no Metal device");
        return;
    };
    let bufs = metal.bufs();

    let caches: Vec<LayerKVCache> = (0..LAYERS)
        .map(|_| LayerKVCache::new(bufs, LONG_ATTENTION_SPAN, MAX_KV_HEADS, HEAD_DIM))
        .collect();
    let max_q = ARMS.iter().map(|a| a.0).max().unwrap();
    let q = bufs.transient_from_f32(&vec![0.01f32; max_q * HEAD_DIM]);
    let out = bufs.output((max_q * HEAD_DIM * 4) as u64);
    let sinks = bufs.output((max_q * 4) as u64);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let threads = (SLICES * HEAD_DIM) as u64;

    let measure = |span: u32, nq: usize, nkv: usize| -> f64 {
        let pipeline = if span > SHORT_ATTENTION_SPAN {
            &metal.attention.kv_attend_seqpar_long_pipeline
        } else {
            &metal.attention.kv_attend_seqpar_pipeline
        };
        let mut times: Vec<f64> = Vec::with_capacity(ITERS);
        for i in 0..WARMUP + ITERS {
            let t = std::time::Instant::now();
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for c in caches.iter() {
                let vals: [u32; 4] = [span, HEAD_DIM as u32, nq as u32, nkv as u32];
                let win = 0u32;
                let has_sinks = 0u32;
                let softcap = 0.0f32;
                enc.set_compute_pipeline_state(pipeline);
                enc.set_buffer(0, Some(&q), 0);
                enc.set_buffer(1, Some(&c.k_cache), 0);
                enc.set_buffer(2, Some(&c.v_cache), 0);
                enc.set_buffer(3, Some(&out), 0);
                for (slot, v) in vals.iter().enumerate() {
                    enc.set_bytes(
                        4 + slot as u64,
                        4,
                        v as *const u32 as *const std::ffi::c_void,
                    );
                }
                enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(9, 4, &win as *const u32 as *const std::ffi::c_void);
                enc.set_buffer(10, Some(&sinks), 0);
                enc.set_bytes(11, 4, &has_sinks as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(12, 4, &softcap as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    metal::MTLSize::new(nq as u64, 1, 1),
                    metal::MTLSize::new(threads, 1, 1),
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if i >= WARMUP {
                times.push(t.elapsed().as_secs_f64() * 1e6 / LAYERS as f64);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };

    println!("GQA-RE-READ: seqpar attention at Gemma 3 4B geometry (head_dim {HEAD_DIM}, {SLICES} slices)");
    println!("  {LAYERS} dispatches/cmdbuf, one cache each; us per dispatch (median of {ITERS})");
    println!("  GB/s counts DISTINCT K+V bytes (span x kv x head_dim x 4 B x 2)");
    println!();
    print!("  {:>5}", "span");
    for &(nq, nkv) in ARMS {
        print!("  {:>16}", format!("{nq}q/{nkv}kv us GB/s"));
    }
    println!("  {:>7}", "drift");
    println!("  {}", "-".repeat(7 + 18 * ARMS.len() + 9));

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for &span in SPANS {
        // The kernel reads its span from slot 4, not the cache struct.
        let times: Vec<f64> = ARMS
            .iter()
            .map(|&(nq, nkv)| measure(span, nq, nkv))
            .collect();
        let (pq, pkv) = ARMS[PRODUCTION];
        let close = measure(span, pq, pkv);
        let drift = 100.0 * (close - times[PRODUCTION]) / times[PRODUCTION];
        print!("  {span:>5}");
        for (&(_, nkv), &us) in ARMS.iter().zip(&times) {
            let bytes = span as f64 * nkv as f64 * HEAD_DIM as f64 * 4.0 * 2.0;
            print!("  {us:>8.2} {:>7.1}", bytes / (us * 1e-6) / 1e9);
        }
        let flag = if drift.abs() > DRIFT_LIMIT_PCT {
            "  UNUSABLE"
        } else {
            ""
        };
        println!("  {drift:>+6.1}%{flag}");
        rows.push(times);
    }

    // Least-squares slope in span: us per 1K rows, per arm.
    let xs: Vec<f64> = SPANS.iter().map(|&s| s as f64 / 1000.0).collect();
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    println!();
    println!(
        "  slope (us per 1K span, per layer) and intercept (us), least squares over all spans"
    );
    let mut slopes = Vec::new();
    for (a, &(nq, nkv)) in ARMS.iter().enumerate() {
        let ys: Vec<f64> = rows.iter().map(|r| r[a]).collect();
        let my = ys.iter().sum::<f64>() / n;
        let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
        let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
        let slope = sxy / sxx;
        slopes.push(slope);
        println!(
            "  {:>10}  slope {slope:>8.2}  intercept {:>7.2}  x{LAYERS} layers: {:>6.3} ms per 1K ctx",
            format!("{nq}q/{nkv}kv"),
            my - slope * mx,
            slope * LAYERS as f64 / 1000.0
        );
    }
    let p = slopes[PRODUCTION];
    println!();
    println!("  slope ratios vs production 8q/4kv:");
    for (a, &(nq, nkv)) in ARMS.iter().enumerate() {
        println!(
            "  {:>10}  {:>5.2}x",
            format!("{nq}q/{nkv}kv"),
            slopes[a] / p
        );
    }
}
