//! SPLITK-1 gate: split-K attention against the production seqpar kernel.
//!
//! Split-K normalises after the weighted-V sum and merges chunk states
//! with a max correction, so it reassociates the arithmetic and is gated
//! at a tolerance, not bitwise — the call KV-B1 made for seqpar. The V
//! fixture is adversarial (mixed sign, 10^3 magnitude spread) so a chunk
//! dropped, doubled, or merged with the wrong max moves the result far
//! outside reassociation noise; `negative_control_*` proves that.
//!
//! Covered: chunk counts 1..16 including more chunks than keys (empty
//! chunks), spans across the 1024 chunk bound, a sliding window, sinks
//! (merge-only), softcap (pass-1-only), and the rows kernel for a verify
//! block against per-position seqpar dispatches.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute_metal::ops::kv_cache::{attention_span, LayerKVCache, SHORT_ATTENTION_SPAN};
use larql_compute_metal::ops::kv_splitk::{min_chunks, ml_part_len, o_part_len, SplitKDispatch};
use larql_compute_metal::MetalBackend;

const CAPACITY: usize = 4096;
/// Production seqpar slices at Gemma's row, used for the reference and
/// inside each split-K chunk.
const SLICES: usize = 4;
const MAX_ROWS: usize = 8;
const MAX_CHUNKS: usize = 16;

/// Reassociation tolerance, relative to the output's own scale: the
/// seqpar gate's 1e-4.
const TOL: f32 = 1e-4;

#[derive(Clone, Copy)]
struct Geometry {
    head_dim: usize,
    num_q: usize,
    num_kv: usize,
}

/// Gemma 3 4B (the target) and gpt-oss-20b (narrow heads, 8:1 GQA).
const GEMMA: Geometry = Geometry {
    head_dim: 256,
    num_q: 8,
    num_kv: 4,
};
const GPT_OSS: Geometry = Geometry {
    head_dim: 64,
    num_q: 64,
    num_kv: 8,
};

#[derive(Clone, Copy, Default)]
struct Semantics {
    window: u32,
    softcap: f32,
    sinks: bool,
}

fn lcg(len: usize, seed: u64, f: impl Fn(usize, f32) -> f32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            f(i, ((s >> 33) as f32) / (u32::MAX as f32))
        })
        .collect()
}

struct Rig {
    metal: MetalBackend,
    g: Geometry,
    cache: LayerKVCache,
    q: metal::Buffer,
    out: metal::Buffer,
    sinks: metal::Buffer,
    o_part: metal::Buffer,
    ml_part: metal::Buffer,
}

fn rig(g: Geometry) -> Rig {
    let metal = MetalBackend::new().expect("Metal device");
    let bufs = metal.bufs();
    let cache = LayerKVCache::new(bufs, CAPACITY, g.num_kv, g.head_dim);
    let n = CAPACITY * g.num_kv * g.head_dim;
    let k = lcg(n, 0x51, |_, u| u * 2.0 - 1.0);
    let v = lcg(n, 0xADDE, |i, u| {
        let mag = 10f32.powf(u * 3.0 - 1.5);
        if i % 2 == 0 {
            mag
        } else {
            -mag
        }
    });
    unsafe {
        std::ptr::copy_nonoverlapping(k.as_ptr(), cache.k_cache.contents() as *mut f32, n);
        std::ptr::copy_nonoverlapping(v.as_ptr(), cache.v_cache.contents() as *mut f32, n);
    }
    let q = bufs.transient_from_f32(&lcg(MAX_ROWS * g.num_q * g.head_dim, 0x99, |_, u| {
        u * 2.0 - 1.0
    }));
    let sinks = bufs.transient_from_f32(&lcg(g.num_q, 0x5157, |_, u| u * 4.0 - 2.0));
    let out = bufs.output((MAX_ROWS * g.num_q * g.head_dim * 4) as u64);
    let o_part = bufs.output((o_part_len(MAX_ROWS, g.num_q, MAX_CHUNKS, g.head_dim) * 4) as u64);
    let ml_part = bufs.output((ml_part_len(MAX_ROWS, g.num_q, MAX_CHUNKS) * 4) as u64);
    Rig {
        metal,
        g,
        cache,
        q,
        out,
        sinks,
        o_part,
        ml_part,
    }
}

fn scale(g: Geometry) -> f32 {
    1.0 / (g.head_dim as f32).sqrt()
}

/// Production seqpar at one position `T = kv_len`, Q row `row`.
fn seqpar(r: &Rig, kv_len: u32, row: usize, sem: Semantics) -> Vec<f32> {
    let g = r.g;
    let m = &r.metal;
    let long = attention_span(kv_len, sem.window) > SHORT_ATTENTION_SPAN;
    let pipeline = if long {
        &m.attention.kv_attend_seqpar_long_pipeline
    } else {
        &m.attention.kv_attend_seqpar_pipeline
    };
    let row_bytes = (row * g.num_q * g.head_dim * 4) as u64;
    let cmd = m.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let u = |i: u64, v: u32| enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void);
    let f = |i: u64, v: f32| enc.set_bytes(i, 4, &v as *const f32 as *const std::ffi::c_void);
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&r.q), row_bytes);
    enc.set_buffer(1, Some(&r.cache.k_cache), 0);
    enc.set_buffer(2, Some(&r.cache.v_cache), 0);
    enc.set_buffer(3, Some(&r.out), 0);
    u(4, kv_len);
    u(5, g.head_dim as u32);
    u(6, g.num_q as u32);
    u(7, g.num_kv as u32);
    f(8, scale(g));
    u(9, sem.window);
    enc.set_buffer(10, Some(&r.sinks), 0);
    u(11, sem.sinks as u32);
    f(12, sem.softcap);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(g.num_q as u64, 1, 1),
        metal::MTLSize::new((SLICES * g.head_dim) as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    larql_compute_metal::buffers::read_buffer_f32(&r.out, g.num_q * g.head_dim)
}

fn dispatch<'a>(
    r: &'a Rig,
    kv_len: u32,
    rows: usize,
    n_chunks: usize,
    sem: Semantics,
) -> SplitKDispatch<'a> {
    let g = r.g;
    SplitKDispatch {
        q: &r.q,
        q_offset: 0,
        k_cache: &r.cache.k_cache,
        v_cache: &r.cache.v_cache,
        out: &r.out,
        out_offset: 0,
        o_part: &r.o_part,
        ml_part: &r.ml_part,
        sinks: sem.sinks.then_some(&r.sinks),
        placeholder: &r.sinks,
        kv_len,
        head_dim: g.head_dim,
        num_q_heads: g.num_q,
        num_kv_heads: g.num_kv,
        score_scale: scale(g),
        window: sem.window,
        softcap: sem.softcap,
        n_chunks,
        slices: SLICES,
        rows,
    }
}

/// Split-K over `rows` positions from `kv_len`; `[rows, num_q, head_dim]`.
fn splitk(r: &Rig, kv_len: u32, rows: usize, n_chunks: usize, sem: Semantics) -> Vec<f32> {
    let cmd = r.metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    dispatch(r, kv_len, rows, n_chunks, sem).encode(&r.metal.attention, enc);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    larql_compute_metal::buffers::read_buffer_f32(&r.out, rows * r.g.num_q * r.g.head_dim)
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0f32, f32::max)
}

fn check(r: &Rig, spans: &[u32], chunk_counts: &[usize], sem: Semantics, label: &str) {
    for &span in spans {
        let reference = seqpar(r, span, 0, sem);
        let floor = min_chunks(attention_span(span, sem.window));
        for &c in chunk_counts.iter().filter(|&&c| c >= floor) {
            let got = splitk(r, span, 1, c, sem);
            let rel = max_rel(&reference, &got);
            assert!(
                rel <= TOL,
                "{label}: span {span}, {c} chunks: max rel {rel:.3e} > {TOL:.0e}\n  \
                 ref[..4]={:?}\n  got[..4]={:?}",
                &reference[..4],
                &got[..4]
            );
        }
    }
}

const SPANS: &[u32] = &[
    1, 2, 3, 31, 33, 255, 257, 1023, 1024, 1025, 2047, 2048, 3000, 4096,
];
const CHUNKS: &[usize] = &[1, 2, 3, 4, 8, 16];

#[test]
fn splitk_matches_seqpar_at_gemma_geometry() {
    check(&rig(GEMMA), SPANS, CHUNKS, Semantics::default(), "gemma");
}

#[test]
fn splitk_matches_seqpar_at_gpt_oss_geometry_with_sinks() {
    let sem = Semantics {
        sinks: true,
        ..Default::default()
    };
    check(&rig(GPT_OSS), SPANS, CHUNKS, sem, "gpt-oss+sinks");
}

#[test]
fn splitk_honours_a_sliding_window() {
    let sem = Semantics {
        window: 1024,
        ..Default::default()
    };
    check(
        &rig(GEMMA),
        &[1000, 1024, 1025, 1500, 4096],
        CHUNKS,
        sem,
        "window",
    );
}

#[test]
fn splitk_applies_softcap_per_score() {
    let sem = Semantics {
        softcap: 5.0,
        ..Default::default()
    };
    check(&rig(GEMMA), &[33, 1025, 3000], CHUNKS, sem, "softcap");
}

/// A verify block: row `i` at cache length `kv_len + i`, all rows in one
/// pass-1 and one merge dispatch, against per-position seqpar.
#[test]
fn splitk_rows_matches_per_position_seqpar() {
    let r = rig(GEMMA);
    let g = r.g;
    let row_len = g.num_q * g.head_dim;
    for (kv_len, n_chunks) in [(20u32, 2usize), (1020, 4), (2040, 4), (3000, 8)] {
        let sem = Semantics::default();
        let got = splitk(&r, kv_len, MAX_ROWS, n_chunks, sem);
        for row in 0..MAX_ROWS {
            let reference = seqpar(&r, kv_len + row as u32, row, sem);
            let rel = max_rel(&reference, &got[row * row_len..(row + 1) * row_len]);
            assert!(
                rel <= TOL,
                "rows: kv_len {kv_len}, row {row}, {n_chunks} chunks: max rel {rel:.3e}"
            );
        }
    }
}

/// The gate must be able to fail: corrupt one chunk's exp-sum between the
/// passes and require the merged output to leave tolerance.
#[test]
fn negative_control_a_corrupted_chunk_state_fails_the_gate() {
    let r = rig(GEMMA);
    let (span, n_chunks) = (2048u32, 4usize);
    let sem = Semantics::default();
    let reference = seqpar(&r, span, 0, sem);
    let d = dispatch(&r, span, 1, n_chunks, sem);

    let cmd = r.metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    d.encode_partial(&r.metal.attention, enc);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    // Head 0, chunk 1: double its l.
    unsafe {
        let ml = r.ml_part.contents() as *mut f32;
        *ml.add(3) *= 2.0;
    }
    let cmd = r.metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    d.encode_merge(&r.metal.attention, enc);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let got = larql_compute_metal::buffers::read_buffer_f32(&r.out, g_len(&r));
    let rel = max_rel(&reference, &got);
    assert!(rel > 100.0 * TOL, "corrupted chunk state passed: {rel:.3e}");
}

fn g_len(r: &Rig) -> usize {
    r.g.num_q * r.g.head_dim
}
