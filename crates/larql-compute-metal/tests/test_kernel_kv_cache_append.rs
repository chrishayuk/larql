#![cfg(target_os = "macos")]

//! Per-kernel tests for `kv_cache_append` and the prefill→decode KV cache
//! layout/stride hand-off.
//!
//! ## Why a focused file
//!
//! `kv_cache_append` is the kernel decode dispatches once per layer per
//! token to merge a freshly-projected K/V into the cache. Production
//! prefill bypasses it (writes the cache via `copy_nonoverlapping` on
//! the underlying Metal buffer) — so any layout disagreement between the
//! prefill bulk-copy path and the decode-time append path produces a
//! cache that *looks* right at one position and wrong elsewhere. The
//! end-to-end consequence is the still-open
//! `decode_consistency_gemma4_31b_dense` parity gap (cos=0.996586 at L0,
//! drifting to cos≈0.76 at L59).
//!
//! The pre-existing `test_kernel_kv_attention` pins `kv_attention` once
//! the cache is populated; this file pins what gets *into* the cache.
//!
//! ## What it asserts
//!
//! 1. **`kv_cache_append` direct correctness** — writes `new_k` / `new_v`
//!    into the right `[pos * num_kv * head_dim ..]` slot, byte-for-byte.
//! 2. **Round-trip with `kv_attention`** — after appending one position,
//!    `kv_attention(T=pos+1)` produces the same answer as a fresh CPU
//!    `kv_attention` over the same K/V buffers. Catches any layout-
//!    interpretation disagreement between the writer and the reader.
//! 3. **Prefill→decode hand-off** — emulate Metal prefill's bulk
//!    `copy_nonoverlapping` of an `[N, num_kv * head_dim]` block of K/V
//!    into `LayerKVCache.{k,v}_cache`, set `current_len = N`, then
//!    `kv_cache_append` at pos=N, then `kv_attention(T=N+1)`. Compare
//!    against a CPU reference over all N+1 positions. This is the exact
//!    sequence production decode does on the first decode step after
//!    prefill — if prefill stores K/V in a different layout than decode
//!    reads them, this test fails before the parity suite would.
//!
//! Geometries cover all four production architectures, with the
//! Gemma 4 31B global-layer shape (32×4×512, head_dim=512) called out
//! since it's where the parity gap lives.

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::{cos_sim, get_metal, max_diff};

use larql_compute_metal::ops::kv_cache::{encode_kv_append, encode_kv_attend, LayerKVCache};

// ── CPU reference ───────────────────────────────────────────────────────────

/// Causal-masked GQA softmax-weighted attention. Same routine the
/// `test_kernel_kv_attention` file uses, kept private here so this
/// binary doesn't depend on it.
#[allow(clippy::too_many_arguments)]
fn cpu_kv_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    t: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_q * head_dim];
    let reps = num_q / num_kv;
    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let mut scores = vec![0.0f32; t];
        for (ki, score) in scores.iter_mut().enumerate() {
            let k_off = ki * num_kv * head_dim + kv_h * head_dim;
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += (q[q_off + d] as f64) * (k_cache[k_off + d] as f64);
            }
            *score = (dot as f32) * scale;
        }
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        for e in exps.iter_mut() {
            *e /= sum_exp;
        }
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for (ki, &exp) in exps.iter().enumerate() {
                let v_off = ki * num_kv * head_dim + kv_h * head_dim;
                acc += (exp as f64) * (v_cache[v_off + d] as f64);
            }
            out[q_off + d] = acc as f32;
        }
    }
    out
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build a `LayerKVCache` sized for `(max_seq, num_kv, head_dim)`.
fn make_layer_cache(
    metal: &larql_compute_metal::MetalBackend,
    max_seq: usize,
    num_kv: usize,
    head_dim: usize,
) -> LayerKVCache {
    LayerKVCache::new(metal.bufs(), max_seq, num_kv, head_dim)
}

/// Read `len` floats from a Metal buffer.
fn read_f32(buf: &metal::Buffer, len: usize) -> Vec<f32> {
    larql_compute_metal::buffers::read_buffer_f32(buf, len)
}

/// Drive `kv_cache_append` once at `cache.current_len`. Mirrors the
/// production decode contract: the append shader reads `pos` from
/// `current_len`, but the caller is responsible for bumping
/// `current_len` *after* the matching `kv_attention` dispatch (which
/// itself reads `T = current_len + 1`). This helper deliberately does
/// not bump — see the caller-side loops which manage the position
/// counter explicitly.
fn append_one(
    metal: &larql_compute_metal::MetalBackend,
    cache: &LayerKVCache,
    new_k: &[f32],
    new_v: &[f32],
) {
    assert_eq!(new_k.len(), cache.num_kv_heads * cache.head_dim);
    assert_eq!(new_v.len(), cache.num_kv_heads * cache.head_dim);
    let new_k_buf = metal.bufs().transient_from_f32(new_k);
    let new_v_buf = metal.bufs().transient_from_f32(new_v);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_kv_append(
        enc,
        cache,
        &metal.attention.kv_append_pipeline,
        &new_k_buf,
        &new_v_buf,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Drive `kv_attention` against a populated cache. Returns
/// `[num_q * head_dim]`.
fn attend(
    metal: &larql_compute_metal::MetalBackend,
    cache: &LayerKVCache,
    q: &[f32],
    num_q: usize,
    scale: f32,
    window: u32,
) -> Vec<f32> {
    let q_buf = metal.bufs().transient_from_f32(q);
    let out_buf = metal.bufs().output((num_q * cache.head_dim * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_kv_attend(
        enc,
        cache,
        &metal.attention.kv_attend_pipeline,
        Some(&metal.attention.kv_attend_long_pipeline),
        &q_buf,
        &out_buf,
        num_q,
        scale,
        window,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    read_f32(&out_buf, num_q * cache.head_dim)
}

/// Deterministic synthetic `[seq * num_kv * head_dim]` buffer that
/// varies along all three axes — any indexing bug in the cache writer
/// (transposed, off-by-stride, head-major instead of position-major)
/// produces visibly wrong output.
fn synth_kv(seq: usize, num_kv: usize, head_dim: usize, salt: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(seq * num_kv * head_dim);
    for p in 0..seq {
        for h in 0..num_kv {
            for d in 0..head_dim {
                let i = (p * num_kv * head_dim + h * head_dim + d) as f32;
                let pf = p as f32;
                let hf = h as f32;
                let df = d as f32;
                v.push(
                    (salt + 0.011 * i).sin() * 0.3
                        + (0.07 * pf + 0.13 * hf).cos() * 0.2
                        + (0.005 * df + 0.31 * hf).sin() * 0.15,
                );
            }
        }
    }
    v
}

fn synth_q(num_q: usize, head_dim: usize, salt: f32) -> Vec<f32> {
    (0..num_q * head_dim)
        .map(|i| ((salt + 0.017 * i as f32).sin() + 0.3 * ((i >> 4) as f32).cos()) * 0.4)
        .collect()
}

// ── 1. kv_cache_append direct correctness ──────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn assert_append_writes_exact_bytes(
    label: &str,
    max_seq: usize,
    num_kv: usize,
    head_dim: usize,
    target_pos: usize,
) {
    let metal = get_metal();
    let mut cache = make_layer_cache(&metal, max_seq, num_kv, head_dim);
    cache.current_len = target_pos;

    let kv_total = num_kv * head_dim;
    let new_k: Vec<f32> = (0..kv_total).map(|i| 0.5 + 0.001 * i as f32).collect();
    let new_v: Vec<f32> = (0..kv_total).map(|i| -0.5 + 0.001 * i as f32).collect();

    append_one(&metal, &cache, &new_k, &new_v);

    let k_full = read_f32(&cache.k_cache, max_seq * kv_total);
    let v_full = read_f32(&cache.v_cache, max_seq * kv_total);

    // Target slot must equal the input element-wise; every other slot
    // must be untouched (the cache buffer is freshly allocated, so 0.0).
    let off = target_pos * kv_total;
    let k_slot = &k_full[off..off + kv_total];
    let v_slot = &v_full[off..off + kv_total];
    let k_diff = max_diff(&new_k, k_slot);
    let v_diff = max_diff(&new_v, v_slot);
    assert!(
        k_diff == 0.0 && v_diff == 0.0,
        "kv_cache_append {label}: target slot bytes don't match input \
         (k_diff={k_diff:.3e} v_diff={v_diff:.3e})",
    );
    for p in 0..max_seq {
        if p == target_pos {
            continue;
        }
        let off = p * kv_total;
        for d in 0..kv_total {
            assert_eq!(
                k_full[off + d],
                0.0,
                "kv_cache_append {label}: K cache pos {p} d {d} = {} (should be 0 — \
                 indicates the writer scattered into the wrong slot or the kernel \
                 striped output across multiple positions)",
                k_full[off + d],
            );
            assert_eq!(
                v_full[off + d],
                0.0,
                "kv_cache_append {label}: V cache pos {p} d {d} != 0 (writer scatter bug)"
            );
        }
    }
}

#[test]
fn append_writes_only_target_slot_llama2() {
    // Llama-2 7B: 8 KV heads × 128 dim. Append at a non-zero pos to
    // catch any "always writes pos 0" bug.
    assert_append_writes_exact_bytes("llama2", /*max_seq*/ 32, 8, 128, /*pos*/ 7);
}

#[test]
fn append_writes_only_target_slot_gemma3_4b() {
    assert_append_writes_exact_bytes("gemma3-4b", 32, 4, 256, 18);
}

#[test]
fn append_writes_only_target_slot_gemma4_sliding() {
    assert_append_writes_exact_bytes("gemma4 sliding", 32, 16, 256, 11);
}

#[test]
fn append_writes_only_target_slot_gemma4_global() {
    // Gemma 4 31B global: 4 KV heads × 512 dim — the parity-bug suspect
    // geometry. With max_seq=32 the full cache is 32 * 4 * 512 = 65536
    // floats; we want to confirm only the target slice gets touched.
    assert_append_writes_exact_bytes("gemma4 global", 32, 4, 512, 18);
}

#[test]
fn append_at_pos_zero_clears_otherwise_only_writes_one() {
    // Edge case: pos=0 (first prefill-less decode token).
    assert_append_writes_exact_bytes("pos0", 16, 4, 256, 0);
}

// ── 2. kv_cache_append round-trips through kv_attention ────────────────────

/// Fill the cache via repeated `append_one`, then attend at the next
/// position with a fresh Q. Compare against a CPU reference over the
/// same K/V/Q. This catches any disagreement between the writer's
/// indexing (`pos * num_kv * head_dim + tid`) and the reader's
/// (`K_cache + t * num_kv * head_dim + kv_head * head_dim + d`).
#[allow(clippy::too_many_arguments)]
fn assert_append_roundtrip(
    label: &str,
    seq: usize, // tokens to append
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
) {
    let metal = get_metal();
    let max_seq = seq.max(64);
    let mut cache = make_layer_cache(&metal, max_seq, num_kv, head_dim);

    let kv_total = num_kv * head_dim;
    let mut k_all = Vec::with_capacity(seq * kv_total);
    let mut v_all = Vec::with_capacity(seq * kv_total);
    // Mirror production decode: encode_kv_append reads pos from
    // current_len. To populate positions 0..seq-1, set current_len = p
    // before each append; never bump past seq-1, because the subsequent
    // attend reads T = current_len + 1.
    for p in 0..seq {
        cache.current_len = p;
        // Distinct salt per position so a "wrote everything to pos 0"
        // bug shows up as identical attention output across queries.
        let nk: Vec<f32> = (0..kv_total)
            .map(|i| ((p as f32 + 1.0) * 0.13 + 0.011 * i as f32).sin() * 0.3)
            .collect();
        let nv: Vec<f32> = (0..kv_total)
            .map(|i| ((p as f32 + 1.0) * 0.17 - 0.013 * i as f32).cos() * 0.25)
            .collect();
        append_one(&metal, &cache, &nk, &nv);
        k_all.extend_from_slice(&nk);
        v_all.extend_from_slice(&nv);
    }
    // current_len = seq - 1; encode_kv_attend will compute T = seq.
    assert_eq!(cache.current_len, seq - 1);

    let q = synth_q(num_q, head_dim, 0.43);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let metal_out = attend(&metal, &cache, &q, num_q, scale, /*window*/ 0);
    let cpu_out = cpu_kv_attention(&q, &k_all, &v_all, seq, num_q, num_kv, head_dim, scale);

    let diff = max_diff(&cpu_out, &metal_out);
    let cos = cos_sim(&cpu_out, &metal_out);
    assert!(
        diff < 1e-3 && cos > 0.999999,
        "append-roundtrip {label} (seq={seq} num_q={num_q} num_kv={num_kv} head_dim={head_dim}): \
         max_abs={diff:.3e} cos={cos:.6}",
    );
}

#[test]
fn append_roundtrip_llama2_t8() {
    assert_append_roundtrip("llama2 t=8", 8, 32, 8, 128);
}

#[test]
fn append_roundtrip_gemma3_4b_t18() {
    assert_append_roundtrip("gemma3-4b t=18", 18, 8, 4, 256);
}

#[test]
fn append_roundtrip_gemma4_sliding_t18() {
    assert_append_roundtrip("gemma4 sliding t=18", 18, 32, 16, 256);
}

#[test]
fn append_roundtrip_gemma4_global_t18() {
    // Decode-bug suspect geometry. If the cache layout disagrees between
    // append and attention readers at head_dim=512, this is where it
    // first shows up — same axis as the still-open parity gap.
    assert_append_roundtrip("gemma4 global t=18", 18, 32, 4, 512);
}

// ── 3. Prefill→decode KV cache hand-off ────────────────────────────────────

/// Production prefill writes the cache via `copy_nonoverlapping` of an
/// `[N, num_kv * head_dim]` block into `k_cache.contents()` at offset 0,
/// then sets `current_len = N`. Decode then runs `kv_cache_append` at
/// pos=N and `kv_attention` at T=N+1.
///
/// If the prefill bulk-copy and the append-shader disagree about layout
/// (e.g. one is `[seq, kv_h, head_d]` and the other is
/// `[kv_h, seq, head_d]`), the parity gap on the open Gemma 4 31B test
/// would land here at L0 with the same cos=0.996586 signature.
///
/// Note: this test exercises the **storage / read** contract only. It
/// uses synthetic K/V values rather than running the real prefill
/// (RoPE, V-norm, QK-norm, projection) — the per-shader correctness of
/// those upstream stages is covered by the dedicated `test_kernel_*`
/// files. What's tested here is purely whether what prefill *stores* is
/// what decode *reads*.
#[allow(clippy::too_many_arguments)]
fn assert_prefill_handoff(
    label: &str,
    n_prefill: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
) {
    let metal = get_metal();
    let max_seq = (n_prefill + 16).max(64);
    let mut cache = make_layer_cache(&metal, max_seq, num_kv, head_dim);

    let kv_total = num_kv * head_dim;

    // Synth K/V for prefill positions 0..N.
    let k_prefill = synth_kv(n_prefill, num_kv, head_dim, 0.21);
    let v_prefill = synth_kv(n_prefill, num_kv, head_dim, 0.71);

    // Emulate prefill's bulk write — exactly what `full_pipeline.rs:914-933`
    // does (post-commit copy_nonoverlapping into k_cache/v_cache
    // contents at offset 0).
    unsafe {
        let k_dst = cache.k_cache.contents() as *mut f32;
        let v_dst = cache.v_cache.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(k_prefill.as_ptr(), k_dst, k_prefill.len());
        std::ptr::copy_nonoverlapping(v_prefill.as_ptr(), v_dst, v_prefill.len());
    }
    // Production prefill leaves current_len at n_prefill — reflects "n
    // tokens cached so far, the next one to write goes at slot
    // n_prefill". Mirror that exactly here.
    cache.current_len = n_prefill;

    // Now run the append path for position N. encode_kv_append reads
    // pos from current_len (= n_prefill), writes there. Production
    // decode does *not* bump current_len before the matching attend.
    let new_k: Vec<f32> = (0..kv_total)
        .map(|i| ((n_prefill as f32 + 1.0) * 0.13 + 0.011 * i as f32).sin() * 0.3)
        .collect();
    let new_v: Vec<f32> = (0..kv_total)
        .map(|i| ((n_prefill as f32 + 1.0) * 0.17 - 0.013 * i as f32).cos() * 0.25)
        .collect();
    append_one(&metal, &cache, &new_k, &new_v);
    // Leave current_len at n_prefill — encode_kv_attend will compute
    // T = n_prefill + 1, attending over positions 0..n_prefill.

    // Build the full reference K/V to compare attention against.
    let mut k_full = k_prefill.clone();
    k_full.extend_from_slice(&new_k);
    let mut v_full = v_prefill.clone();
    v_full.extend_from_slice(&new_v);

    let q = synth_q(num_q, head_dim, 0.91);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total = n_prefill + 1;
    let metal_out = attend(&metal, &cache, &q, num_q, scale, 0);
    let cpu_out = cpu_kv_attention(&q, &k_full, &v_full, total, num_q, num_kv, head_dim, scale);

    let diff = max_diff(&cpu_out, &metal_out);
    let cos = cos_sim(&cpu_out, &metal_out);
    assert!(
        diff < 1e-3 && cos > 0.999999,
        "prefill→decode hand-off {label} \
         (n_prefill={n_prefill} num_q={num_q} num_kv={num_kv} head_dim={head_dim}): \
         max_abs={diff:.3e} cos={cos:.6}\n\
         cpu[..8]={:?}\nmtl[..8]={:?}",
        &cpu_out[..8.min(cpu_out.len())],
        &metal_out[..8.min(metal_out.len())],
    );
}

#[test]
fn prefill_handoff_llama2_n18() {
    // Matches `decode_consistency_llama2_7b`'s "Capital of France is"
    // length pattern — 5–6 wordpiece tokens after the chat-template wrap.
    assert_prefill_handoff("llama2 n=18", 18, 32, 8, 128);
}

#[test]
fn prefill_handoff_gemma3_4b_n18() {
    assert_prefill_handoff("gemma3-4b n=18", 18, 8, 4, 256);
}

#[test]
fn prefill_handoff_gemma4_sliding_n18() {
    assert_prefill_handoff("gemma4 sliding n=18", 18, 32, 16, 256);
}

#[test]
fn prefill_handoff_gemma4_global_n18() {
    // The decode-vs-prefill parity gap on Gemma 4 31B drifts from
    // cos=0.996586 at L0 to cos≈0.76 at L59. If the bulk-copy →
    // kv_cache_append → kv_attention chain has a layout disagreement
    // at this exact geometry, this test fails before any other.
    assert_prefill_handoff("gemma4 global n=18", 18, 32, 4, 512);
}

#[test]
fn prefill_handoff_long_context_n128() {
    // Stress the bulk-copy stride at a longer prefill — useful for the
    // long-context regression suite and for catching any
    // `seq_len * num_kv * head_dim` overflow into u32.
    assert_prefill_handoff("long n=128", 128, 8, 2, 128);
}
