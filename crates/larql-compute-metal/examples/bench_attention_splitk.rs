//! SPLITK-1 — split-K attention vs production seqpar at Gemma 3 4B's
//! geometry, swept over chunk count and in-threadgroup slices.
//!
//! GQA-RE-READ (`bench_attention_gqa`) found seqpar per-threadgroup
//! latency bound with 8 threadgroups on 40 cores. Split-K gives each
//! head `n_chunks` threadgroups; this locates the occupancy / merge
//! crossover per span, which becomes the planner's tiers.
//!
//! Method as `bench_attention_gqa`: `LAYERS` serial dispatches per command
//! buffer, one cold cache each, median of `ITERS`, and the seqpar baseline
//! measured again after the arms (a row drifting past `DRIFT_LIMIT_PCT`
//! is flagged). Split-K time includes its merge dispatch.
//!
//! Run: `cargo run --release -p larql-compute-metal --example bench_attention_splitk`

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("bench_attention_splitk requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute_metal::ops::kv_cache::{
        LayerKVCache, LONG_ATTENTION_SPAN, SHORT_ATTENTION_SPAN,
    };
    use larql_compute_metal::ops::kv_splitk::{
        min_chunks, ml_part_len, o_part_len, SplitKDispatch,
    };
    use larql_compute_metal::MetalBackend;

    const HEAD_DIM: usize = 256;
    const NUM_Q: usize = 8;
    const NUM_KV: usize = 4;
    /// Production seqpar slices at Gemma's row.
    const PROD_SLICES: usize = 4;
    const LAYERS: usize = 34;
    const WARMUP: usize = 8;
    const ITERS: usize = 40;
    const DRIFT_LIMIT_PCT: f64 = 5.0;
    const CHUNKS: &[usize] = &[2, 4, 8, 16, 32, 64];
    const SLICES: &[usize] = &[1, 2, 4];
    const SPANS: &[u32] = &[32, 128, 256, 512, 1024, 1536, 2048, 3072, 4096];

    let Some(metal) = MetalBackend::new() else {
        eprintln!("no Metal device");
        return;
    };
    let bufs = metal.bufs();
    let caches: Vec<LayerKVCache> = (0..LAYERS)
        .map(|_| LayerKVCache::new(bufs, LONG_ATTENTION_SPAN, NUM_KV, HEAD_DIM))
        .collect();
    let max_chunks = *CHUNKS.iter().max().unwrap();
    let q = bufs.transient_from_f32(&vec![0.01f32; NUM_Q * HEAD_DIM]);
    let out = bufs.output((NUM_Q * HEAD_DIM * 4) as u64);
    let sinks = bufs.output((NUM_Q * 4) as u64);
    let o_part = bufs.output((o_part_len(1, NUM_Q, max_chunks, HEAD_DIM) * 4) as u64);
    let ml_part = bufs.output((ml_part_len(1, NUM_Q, max_chunks) * 4) as u64);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // `None` = production seqpar; `Some((chunks, slices))` = split-K.
    let measure = |span: u32, arm: Option<(usize, usize)>| -> f64 {
        let mut times: Vec<f64> = Vec::with_capacity(ITERS);
        for i in 0..WARMUP + ITERS {
            let t = std::time::Instant::now();
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for c in caches.iter() {
                match arm {
                    None => {
                        let pipeline = if span > SHORT_ATTENTION_SPAN {
                            &metal.attention.kv_attend_seqpar_long_pipeline
                        } else {
                            &metal.attention.kv_attend_seqpar_pipeline
                        };
                        let u = |i: u64, v: u32| {
                            enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void)
                        };
                        let f = |i: u64, v: f32| {
                            enc.set_bytes(i, 4, &v as *const f32 as *const std::ffi::c_void)
                        };
                        enc.set_compute_pipeline_state(pipeline);
                        enc.set_buffer(0, Some(&q), 0);
                        enc.set_buffer(1, Some(&c.k_cache), 0);
                        enc.set_buffer(2, Some(&c.v_cache), 0);
                        enc.set_buffer(3, Some(&out), 0);
                        u(4, span);
                        u(5, HEAD_DIM as u32);
                        u(6, NUM_Q as u32);
                        u(7, NUM_KV as u32);
                        f(8, scale);
                        u(9, 0);
                        enc.set_buffer(10, Some(&sinks), 0);
                        u(11, 0);
                        f(12, 0.0);
                        enc.dispatch_thread_groups(
                            metal::MTLSize::new(NUM_Q as u64, 1, 1),
                            metal::MTLSize::new((PROD_SLICES * HEAD_DIM) as u64, 1, 1),
                        );
                    }
                    Some((n_chunks, slices)) => SplitKDispatch {
                        q: &q,
                        q_offset: 0,
                        k_cache: &c.k_cache,
                        v_cache: &c.v_cache,
                        out: &out,
                        out_offset: 0,
                        o_part: &o_part,
                        ml_part: &ml_part,
                        sinks: None,
                        placeholder: &sinks,
                        kv_len: span,
                        head_dim: HEAD_DIM,
                        num_q_heads: NUM_Q,
                        num_kv_heads: NUM_KV,
                        score_scale: scale,
                        window: 0,
                        softcap: 0.0,
                        n_chunks,
                        slices,
                        rows: 1,
                    }
                    .encode(&metal.attention, enc),
                }
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

    println!("SPLITK-1: split-K vs seqpar({PROD_SLICES}) at ({HEAD_DIM}, {NUM_Q}q, {NUM_KV}kv)");
    println!("  {LAYERS} dispatches/cmdbuf, cold caches; cells = speedup over seqpar (us in brackets for best)");
    println!();
    print!("  {:>5} {:>8}", "span", "seqpar");
    for &s in SLICES {
        for &c in CHUNKS {
            print!(" {:>6}", format!("s{s}c{c}"));
        }
    }
    println!("  {:>14} {:>7}", "best", "drift");

    for &span in SPANS {
        let base = measure(span, None);
        print!("  {span:>5} {base:>8.2}");
        let mut best = (f64::MAX, 0, 0);
        for &s in SLICES {
            for &c in CHUNKS {
                if c < min_chunks(span) || c as u32 > span {
                    print!(" {:>6}", "-");
                    continue;
                }
                let t = measure(span, Some((c, s)));
                if t < best.0 {
                    best = (t, s, c);
                }
                print!(" {:>5.2}x", base / t);
            }
        }
        let close = measure(span, None);
        let drift = 100.0 * (close - base) / base;
        let flag = if drift.abs() > DRIFT_LIMIT_PCT {
            " UNUSABLE"
        } else {
            ""
        };
        println!(
            "  {:>14} {drift:>+6.1}%{flag}",
            format!("s{}c{} {:.1}us", best.1, best.2, best.0)
        );
    }
}
