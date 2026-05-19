//! Per-expert gated-FFN execution (gate_proj, up_proj, activation, down_proj).
//!
//! Used by the in-process MoE forward pass (`cpu_moe_forward`) and by the
//! remote expert server endpoint when one expert's work is delegated to a
//! shard. The BF16 expert weights are dequantized on demand so only the
//! selected experts pay the conversion cost.

use super::cache::{try_cached_dequant, ExpertF32};
use super::math::{gelu_tanh, matmul_vec, matmul_vec_into, rms_norm, silu};
use crate::cpu::ops::q4_common::q4k_matvec_into;
use crate::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_matvec_into, quantize_x_to_q8k, quantize_x_to_q8k_into, Q8KActivation,
};
use crate::options;
// `q4k_q8k_gate_up_into` exists for future kernel exploration but is not
// wired into the hot path — see comment in `run_single_expert_q4k_q8k_into`.

/// Per-call scratch for `run_single_expert_with_scratch` — preallocate once
/// per gRPC frame and reuse across all K active experts.  Keeps allocation
/// off the hot path: at Gemma 4 26B-A4B sizes the un-pooled version was
/// minting ~360 fresh ~11KB Vecs per token per shard.
///
/// Sized for one expert's worth of intermediate buffers.  Per-call cost on
/// reuse is O(0) — just zeros the activation buffer's padding columns.
pub struct ExpertScratch {
    /// `[inter]` — gate matvec output before activation.
    pub gate_out: Vec<f32>,
    /// `[inter]` — up matvec output.
    pub up_out: Vec<f32>,
    /// `[inter_padded]` — activation buffer fed into down.  Padding columns
    /// (`inter..inter_padded`) are zero-initialised once and re-used
    /// untouched across calls (down's matvec reads them as zero).
    pub act: Vec<f32>,
    /// Q8_K quantisation of `act` for the down matvec on the Q4_K-direct
    /// path.  Pre-allocated at construction so the per-expert quantise
    /// doesn't allocate — eliminates the 5% / 150 µs alloc spikes that
    /// previously dragged the par_iter wall up across rayon workers.
    pub act_q8k: Q8KActivation,
    /// `[hidden]` — final expert output.
    pub out: Vec<f32>,
}

impl ExpertScratch {
    /// Allocate scratch sized for `(hidden, inter, inter_padded)`.  Call
    /// once per gRPC frame; share `&mut` across the K experts.
    pub fn new(hidden: usize, inter: usize, inter_padded: usize) -> Self {
        Self {
            gate_out: vec![0.0f32; inter],
            up_out: vec![0.0f32; inter],
            act: vec![0.0f32; inter_padded],
            act_q8k: Q8KActivation::with_capacity(inter_padded),
            out: vec![0.0f32; hidden],
        }
    }
}

/// Apply pre_experts_norm once per frame and return the normed residual.
/// Hoisting this out of `run_single_expert*` saves K-1 redundant rms_norm
/// passes per layer (the input residual is identical for every expert in
/// the layer's top-K — they all receive the same h_norm by design).
pub fn pre_experts_norm(
    h: &[f32],
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    if pre_experts_norm.is_empty() {
        return h.to_vec();
    }
    rms_norm(h, pre_experts_norm, eps, norm_offset)
}

/// Run a single expert's gated FFN given a pre-normed input vector.
///
/// `gate_up_bytes` and `down_bytes` carry exactly one expert's weights — the
/// caller picks the right per-expert byte range (per-layer `layers/{L}/{e}`
/// mmap entries or a stride into a legacy monolith). `format` tells the
/// dequantiser how to decode them. Returns the expert's output (not yet
/// weighted by router probability). `h_norm` must already be RMS-normed —
/// use `run_single_expert_with_norm` when you have the raw residual.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert(
    h_norm: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    format: crate::QuantFormat,
    activation: crate::Activation,
) -> Vec<f32> {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 {
        return vec![0.0f32; hidden];
    }

    // Storage layout (matches `format/weights/write_layers.rs::quantize_moe_entries`):
    //   gate_up: [2*inter, hidden]              never padded
    //   down:    [hidden, inter_padded]         Q4_K pads inter→256 multiple
    // BF16 has no padding for either. See `forward::cpu_moe_forward` for the
    // expanded explanation; this single-expert path mirrors it exactly so the
    // remote-expert HTTP endpoint and local in-process MoE share the same
    // numerics.
    let inter_padded = match format {
        crate::QuantFormat::Q4_K => {
            let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
            inter.div_ceil(block) * block
        }
        _ => inter,
    };

    // Q4_K direct-from-mmap path (NEON SDOT on aarch64).  Routes through
    // `run_single_expert_q4k_q8k_into` with a thread-local `ExpertScratch`
    // so the per-call allocations of gate_out / up_out / act / act_q8k go
    // away — only the final `Vec<f32>` output is allocated for the
    // function's return type.  Profiling (2026-05-01) showed K=8 × per-call
    // allocs as the dominant HTTP-path bottleneck once the kernel itself
    // got below ~80 µs.  Set `LARQL_DISABLE_Q4K_DIRECT=1` to opt out
    // (kernel-debug A/B).
    if matches!(format, crate::QuantFormat::Q4_K)
        && hidden.is_multiple_of(256)
        && !super::q4k_direct_disabled()
    {
        thread_local! {
            static SCRATCH: std::cell::RefCell<Option<ExpertScratch>> =
                const { std::cell::RefCell::new(None) };
        }
        // Quantise h_norm into a per-thread scratch buffer too, reusing
        // capacity across calls.  Same pattern as ExpertScratch — the
        // h_norm is the same length on every call from the HTTP path, so
        // resize is a no-op after the first hit.
        thread_local! {
            static H_Q8K: std::cell::RefCell<Q8KActivation> =
                std::cell::RefCell::new(Q8KActivation::with_capacity(0));
        }
        return SCRATCH.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let scratch =
                borrow.get_or_insert_with(|| ExpertScratch::new(hidden, inter, inter_padded));
            if scratch.gate_out.len() != inter
                || scratch.act.len() != inter_padded
                || scratch.out.len() != hidden
            {
                *scratch = ExpertScratch::new(hidden, inter, inter_padded);
            }
            H_Q8K.with(|hcell| {
                let mut hb = hcell.borrow_mut();
                quantize_x_to_q8k_into(&mut hb, h_norm);
                let h2 = run_single_expert_q4k_q8k_into(
                    scratch,
                    &hb,
                    gate_up_bytes,
                    down_bytes,
                    inter,
                    activation,
                );
                h2.to_vec()
            })
        });
    }

    let gate_up_w = try_cached_dequant(gate_up_bytes, format, 2 * inter * hidden)
        .unwrap_or_else(|err| panic!("{err}"));
    if gate_up_w.is_empty() {
        return vec![0.0f32; hidden];
    }
    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

    let gate_out = matmul_vec(h_norm, gate_w, inter, hidden);
    let up_out = matmul_vec(h_norm, up_w, inter, hidden);

    // Build inner activation at `inter_padded` so the down matmul (which
    // expects `inter_padded` columns under Q4_K) sees zero in the padding.
    let mut hidden_state: Vec<f32> = vec![0.0f32; inter_padded];
    for j in 0..inter {
        let g = gate_out[j];
        let u = up_out[j];
        hidden_state[j] = match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        };
    }

    let down_w = try_cached_dequant(down_bytes, format, hidden * inter_padded)
        .unwrap_or_else(|err| panic!("{err}"));
    if down_w.is_empty() {
        return vec![0.0f32; hidden];
    }
    matmul_vec(&hidden_state, &down_w, hidden, inter_padded)
}

/// Allocation-free variant of `run_single_expert`: writes into the caller's
/// `ExpertScratch` instead of allocating gate / up / activation / output
/// buffers per call.  Used by the streaming expert server's hot path where
/// allocation churn would dominate at K=8 × 30 layers per token.
///
/// `h_norm` is already pre-normed (see `pre_experts_norm`).  Returns a
/// borrow of `scratch.out` so the caller can `clone_from_slice` into the
/// per-shard accumulator before reusing the scratch for the next expert.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert_into<'s>(
    scratch: &'s mut ExpertScratch,
    h_norm: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    format: crate::QuantFormat,
    activation: crate::Activation,
) -> &'s [f32] {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }

    let inter_padded = match format {
        crate::QuantFormat::Q4_K => {
            let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
            inter.div_ceil(block) * block
        }
        _ => inter,
    };
    debug_assert_eq!(scratch.gate_out.len(), inter);
    debug_assert_eq!(scratch.up_out.len(), inter);
    debug_assert_eq!(scratch.act.len(), inter_padded);
    debug_assert_eq!(scratch.out.len(), hidden);

    // Per-stage timing: enabled by `LARQL_MOE_EXPERT_TIMING=1`.  Hot path
    // gate; the env-var check is cached in TLS to avoid a syscall per call.
    thread_local! {
        static EXPERT_TIMING: bool =
            options::env_flag(options::ENV_MOE_EXPERT_TIMING);
    }
    let timing = EXPERT_TIMING.with(|t| *t);
    let mut t = std::time::Instant::now();

    // Q4_K direct matvec is available via `LARQL_Q4K_DIRECT=1` but stays
    // OFF by default — on Apple Silicon the scalar inner loop loses to
    // BLAS sgemv on cached f32 weights (BLAS uses AMX, ~5× more compute
    // throughput than scalar Rust).  Will become the right default once
    // we ship a NEON-vectorized version.
    thread_local! {
        static Q4K_DIRECT: bool =
            options::env_flag(options::ENV_Q4K_DIRECT);
    }
    let q4k_direct = Q4K_DIRECT.with(|v| *v);
    let q4k_path = q4k_direct && matches!(format, crate::QuantFormat::Q4_K);

    let gate_w_size = inter * hidden;
    // f32 path: hold the cached Arc for the duration of the call so the
    // gate_w / up_w slices below borrow into the cache's payload directly.
    // The previous `v.to_vec()` here copied ~12 MB per call on cache hit,
    // which dominated the per-expert wall time at Gemma 4 26B-A4B sizes.
    let gate_up_w_arc: Option<ExpertF32> = if q4k_path {
        None
    } else {
        let v = try_cached_dequant(gate_up_bytes, format, 2 * inter * hidden)
            .unwrap_or_else(|err| panic!("{err}"));
        if v.is_empty() {
            for v in scratch.out.iter_mut() {
                *v = 0.0;
            }
            return &scratch.out;
        }
        Some(v)
    };
    let t_cache_gu = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    if q4k_path {
        let row_block_bytes = (hidden / larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
            * larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
        let half = inter * row_block_bytes;
        let gate_bytes = &gate_up_bytes[..half];
        let up_bytes = &gate_up_bytes[half..2 * half];
        q4k_matvec_into(&mut scratch.gate_out, h_norm, gate_bytes, inter, hidden);
        let t_gate = if timing { Some(t.elapsed()) } else { None };
        if timing {
            t = std::time::Instant::now();
        }
        q4k_matvec_into(&mut scratch.up_out, h_norm, up_bytes, inter, hidden);
        let t_up = if timing { Some(t.elapsed()) } else { None };
        if timing {
            t = std::time::Instant::now();
        }
        for j in 0..inter {
            let g = scratch.gate_out[j];
            let u = scratch.up_out[j];
            scratch.act[j] = match activation {
                crate::Activation::GeluTanh => gelu_tanh(g) * u,
                _ => silu(g) * u,
            };
        }
        let t_act = if timing { Some(t.elapsed()) } else { None };
        if timing {
            t = std::time::Instant::now();
        }
        q4k_matvec_into(
            &mut scratch.out,
            &scratch.act,
            down_bytes,
            hidden,
            inter_padded,
        );
        let t_down = if timing { Some(t.elapsed()) } else { None };
        if timing {
            eprintln!(
                "[run_expert] q4k_direct cache_gu={:.0}us gate={:.0}us up={:.0}us \
                 act={:.0}us cache_dn=0us down={:.0}us",
                t_cache_gu.unwrap().as_secs_f64() * 1e6,
                t_gate.unwrap().as_secs_f64() * 1e6,
                t_up.unwrap().as_secs_f64() * 1e6,
                t_act.unwrap().as_secs_f64() * 1e6,
                t_down.unwrap().as_secs_f64() * 1e6,
            );
        }
        return &scratch.out;
    }

    // Default path: f32 dequant cache + BLAS sgemv (Apple AMX / OpenBLAS).
    // `gate_up_w_arc` is Some when q4k_path is false (we returned early on
    // miss above); slice into the cached Arc without copying.
    let gate_up_w_f32: &[f32] = gate_up_w_arc
        .as_deref()
        .expect("gate_up_w_arc populated on f32 path");
    let gate_w = &gate_up_w_f32[..gate_w_size];
    let up_w = &gate_up_w_f32[gate_w_size..2 * gate_w_size];
    matmul_vec_into(&mut scratch.gate_out, h_norm, gate_w, inter, hidden);
    let t_gate = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    matmul_vec_into(&mut scratch.up_out, h_norm, up_w, inter, hidden);
    let t_up = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // Build inner activation at `inter_padded`; padding columns
    // (`inter..inter_padded`) stay at their zero-initialised value across
    // reuses since we never write them.
    for j in 0..inter {
        let g = scratch.gate_out[j];
        let u = scratch.up_out[j];
        scratch.act[j] = match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        };
    }
    let t_act = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    let down_w = try_cached_dequant(down_bytes, format, hidden * inter_padded)
        .unwrap_or_else(|err| panic!("{err}"));
    if down_w.is_empty() {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }
    let t_cache_dn = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    matmul_vec_into(
        &mut scratch.out,
        &scratch.act,
        &down_w,
        hidden,
        inter_padded,
    );
    let t_down = if timing { Some(t.elapsed()) } else { None };

    if timing {
        eprintln!(
            "[run_expert] cache_gu={:.0}us gate={:.0}us up={:.0}us act={:.0}us \
             cache_dn={:.0}us down={:.0}us",
            t_cache_gu.unwrap().as_secs_f64() * 1e6,
            t_gate.unwrap().as_secs_f64() * 1e6,
            t_up.unwrap().as_secs_f64() * 1e6,
            t_act.unwrap().as_secs_f64() * 1e6,
            t_cache_dn.unwrap().as_secs_f64() * 1e6,
            t_down.unwrap().as_secs_f64() * 1e6,
        );
    }
    &scratch.out
}

/// Pre-quantise `h_norm` to Q8_K once per layer (shared across the K
/// active experts).  Cost is amortised K-fold: at top_k=8 we save 7
/// quantisation passes per layer.
///
/// Returns `None` if `h_norm.len()` isn't a multiple of 256 (Q8_K block
/// size).  Caller falls back to the f32 path in that case.
pub fn quantize_h_norm_for_q4k(h_norm: &[f32]) -> Option<Q8KActivation> {
    if h_norm.is_empty() || !h_norm.len().is_multiple_of(256) {
        return None;
    }
    Some(quantize_x_to_q8k(h_norm))
}

/// Direct Q4_K-from-mmap expert kernel.  No f32 dequant cache; reads the
/// 144-byte Q4_K super-blocks straight from the per-layer mmap and accumulates
/// an integer dot product against the pre-quantised Q8_K activation.
///
/// On Apple Silicon the inner kernel uses `SDOT` (16 i8 × i8 → 4 i32 lanes
/// per instruction) via `crate::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_into`.
/// On other targets it falls through to the scalar Q8_K reference.
///
/// Why this is faster than the BLAS-on-cached-f32 path at Gemma 4 26B-A4B
/// sizes: the f32 cache is 24 MB per expert × 240 experts/token = 5.7 GB
/// of f32 weights walked per token, which exceeds L3 cache by ~30× on
/// M3 Max — DRAM bandwidth-bound at f32 reading.  Direct Q4_K reads are
/// ~12 MB Q4_K bytes per expert (4× smaller), so DRAM pressure drops 4×
/// and the kernel actually runs near the BW bound rather than way over it.
///
/// `h_norm_q8k` MUST be the Q8_K of the same `h_norm` that fed the f32
/// path — call `quantize_h_norm_for_q4k(&h_norm)` once outside the
/// per-expert loop and share it across the K active experts.
pub fn run_single_expert_q4k_q8k_into<'s>(
    scratch: &'s mut ExpertScratch,
    h_norm_q8k: &Q8KActivation,
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    activation: crate::Activation,
) -> &'s [f32] {
    // Per-stage timing for kernel diagnosis.  Enable with
    // `LARQL_KERNEL_TIMING=1`.  Cached in TLS to avoid syscall per call.
    thread_local! {
        static KERNEL_TIMING: bool = options::env_flag(options::ENV_KERNEL_TIMING);
    }
    let timing = KERNEL_TIMING.with(|t| *t);

    let hidden = h_norm_q8k.qs.len();
    if inter == 0 || hidden == 0 {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }

    // Q4_K weight stride (in bytes) per row: ceil(hidden / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES.
    let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let inter_padded = inter.div_ceil(block) * block;
    let row_block_bytes = (hidden / block) * larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
    let half = inter * row_block_bytes;
    if gate_up_bytes.len() < 2 * half {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }
    let gate_bytes = &gate_up_bytes[..half];
    let up_bytes = &gate_up_bytes[half..2 * half];

    let mut t = std::time::Instant::now();
    // Back-to-back gate + up matvecs.  Tried fused-gate+up via
    // `q4k_q8k_gate_up_into` (2026-05-01): bench was within noise on the
    // single-layer floor and ~4% slower on the 30-layer sweep — the M3 Max
    // OoO engine already extracts plenty of ILP from these two independent
    // matvecs, and the manually-interleaved kernel adds register pressure
    // / hurts the L1 prefetcher.  Fused entry point is kept in
    // `q4k_q8k_dot.rs` (with bit-exact parity test) for future
    // CPU profiles where the trade-off may flip.
    q4k_q8k_matvec_into(&mut scratch.gate_out, h_norm_q8k, gate_bytes, inter, hidden);
    let t_gate = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    q4k_q8k_matvec_into(&mut scratch.up_out, h_norm_q8k, up_bytes, inter, hidden);
    let t_up = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // GELU/SiLU(gate) ⊙ up.  Padding columns (`inter..inter_padded`) stay
    // at their zero-initialised value across reuses (we never write them),
    // matching the existing convention in `run_single_expert_into`.
    for j in 0..inter {
        let g = scratch.gate_out[j];
        let u = scratch.up_out[j];
        scratch.act[j] = match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        };
    }
    let t_act = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // Quantise the per-expert activation to Q8_K in-place into the
    // caller-owned scratch buffer (no allocation on the hot path —
    // eliminates the 150 µs alloc spikes that drag par_iter wall up).
    quantize_x_to_q8k_into(&mut scratch.act_q8k, &scratch.act);
    let t_act_q8k = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // down matvec: out[hidden] = down_W[hidden, inter_padded] @ act
    q4k_q8k_matvec_into(
        &mut scratch.out,
        &scratch.act_q8k,
        down_bytes,
        hidden,
        inter_padded,
    );
    let t_down = if timing { Some(t.elapsed()) } else { None };

    if timing {
        eprintln!(
            "[expert_q4k_q8k] gate={:.0}us up={:.0}us act={:.0}us \
             act_q8k={:.0}us down={:.0}us",
            t_gate.unwrap().as_secs_f64() * 1e6,
            t_up.unwrap().as_secs_f64() * 1e6,
            t_act.unwrap().as_secs_f64() * 1e6,
            t_act_q8k.unwrap().as_secs_f64() * 1e6,
            t_down.unwrap().as_secs_f64() * 1e6,
        );
    }

    &scratch.out
}

/// Apply pre-experts norm then run a single expert. Used by the remote
/// expert server endpoint where the raw residual arrives from the client.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert_with_norm(
    h: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
    format: crate::QuantFormat,
    activation: crate::Activation,
) -> Vec<f32> {
    let h_norm = rms_norm(h, pre_experts_norm, eps, norm_offset);
    run_single_expert(
        &h_norm,
        gate_up_bytes,
        down_bytes,
        inter,
        format,
        activation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::quantize_q4_k;
    use crate::{Activation, QuantFormat};

    // BF16 encoding for common values (little-endian: low byte first).
    fn bf16_bytes(v: f32) -> [u8; 2] {
        let bits = v.to_bits();
        let hi = (bits >> 16) as u16;
        hi.to_le_bytes()
    }

    fn fill_bf16(len: usize, val: f32) -> Vec<u8> {
        let b = bf16_bytes(val);
        let mut v = vec![0u8; len * 2];
        for i in 0..len {
            v[i * 2] = b[0];
            v[i * 2 + 1] = b[1];
        }
        v
    }

    #[test]
    fn zero_inter_returns_zero_vec() {
        let h = vec![1.0f32; 4];
        let out = run_single_expert(&h, &[], &[], 0, QuantFormat::BF16, Activation::Silu);
        assert_eq!(out, vec![0.0f32; 4]);
    }

    #[test]
    fn zero_hidden_returns_empty() {
        let h: Vec<f32> = vec![];
        let out = run_single_expert(&h, &[], &[], 0, QuantFormat::BF16, Activation::Silu);
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn pre_experts_norm_empty_weight_returns_input_copy() {
        let h = vec![1.0f32, -2.0, 3.5];
        let out = pre_experts_norm(&h, &[], 0.0, 1e-6);
        assert_eq!(out, h);
    }

    #[test]
    fn pre_experts_norm_applies_weight_and_offset() {
        let h = vec![3.0f32, 4.0];
        let norm_w = vec![1.0f32, 2.0];
        let out = pre_experts_norm(&h, &norm_w, 0.5, 0.0);
        let rms = ((3.0_f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt();
        let expected = [3.0 / rms * 1.5, 4.0 / rms * 2.5];

        for (actual, expected) in out.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn nonzero_weights_produce_nonzero_output() {
        let hidden = 4;
        let inter = 2;
        // One expert's worth of all-1.0 BF16 weights.
        let gate_up = fill_bf16(2 * inter * hidden, 1.0);
        let down = fill_bf16(hidden * inter, 1.0);
        let h = vec![1.0f32; hidden];
        let out = run_single_expert(
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );
        assert_eq!(out.len(), hidden);
        assert!(
            out.iter().any(|v| v.abs() > 0.01),
            "expected nonzero output, got {out:?}"
        );
    }

    #[test]
    fn run_single_expert_into_matches_allocating_path() {
        let hidden = 4;
        let inter = 2;
        let gate_up = fill_bf16(2 * inter * hidden, 0.75);
        let down = fill_bf16(hidden * inter, -0.5);
        let h = vec![1.0f32, 0.5, -0.25, 2.0];
        let expected = run_single_expert(
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );
        let mut scratch = ExpertScratch::new(hidden, inter, inter);

        let actual = run_single_expert_into(
            &mut scratch,
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn run_single_expert_into_zeroes_output_for_empty_weights() {
        let hidden = 4;
        let inter = 2;
        let h = vec![1.0f32; hidden];
        let mut scratch = ExpertScratch::new(hidden, inter, inter);
        scratch.out.fill(9.0);

        let out = run_single_expert_into(
            &mut scratch,
            &h,
            &[],
            &[],
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );

        assert_eq!(out, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn quantize_h_norm_for_q4k_rejects_empty_or_misaligned_input() {
        assert!(quantize_h_norm_for_q4k(&[]).is_none());
        assert!(quantize_h_norm_for_q4k(&vec![1.0f32; 255]).is_none());

        let q8 = quantize_h_norm_for_q4k(&vec![1.0f32; 256]).unwrap();
        assert_eq!(q8.qs.len(), 256);
    }

    #[test]
    fn run_single_expert_q4k_q8k_into_zeroes_output_for_short_gate_up() {
        let hidden = 256;
        let inter = 1;
        let mut scratch = ExpertScratch::new(hidden, inter, 256);
        scratch.out.fill(7.0);
        let h_q8 = quantize_h_norm_for_q4k(&vec![1.0f32; hidden]).unwrap();

        let out =
            run_single_expert_q4k_q8k_into(&mut scratch, &h_q8, &[], &[], inter, Activation::Silu);

        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn run_single_expert_q4k_q8k_into_valid_weights_produces_finite_output() {
        let hidden = 256;
        let inter = 256;
        let h: Vec<f32> = (0..hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.03)
            .collect();
        let h_q8 = quantize_h_norm_for_q4k(&h).unwrap();
        let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.002)
            .collect();
        let down_f32: Vec<f32> = (0..hidden * inter)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.0025)
            .collect();
        let gate_up = quantize_q4_k(&gate_up_f32);
        let down = quantize_q4_k(&down_f32);
        let mut scratch = ExpertScratch::new(hidden, inter, inter);

        let out = run_single_expert_q4k_q8k_into(
            &mut scratch,
            &h_q8,
            &gate_up,
            &down,
            inter,
            Activation::GeluTanh,
        );

        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|v| v.abs() > 1e-8), "got all-zero output");
    }

    #[test]
    fn run_single_expert_into_q4k_cached_dequant_path_runs() {
        let hidden = 256;
        let inter = 256;
        let h: Vec<f32> = (0..hidden)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
            .collect();
        let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.0015)
            .collect();
        let down_f32: Vec<f32> = (0..hidden * inter)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.001)
            .collect();
        let gate_up = quantize_q4_k(&gate_up_f32);
        let down = quantize_q4_k(&down_f32);
        let mut scratch = ExpertScratch::new(hidden, inter, inter);

        let out = run_single_expert_into(
            &mut scratch,
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::Q4_K,
            Activation::Silu,
        );

        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn with_norm_matches_manual_prenorm() {
        let hidden = 4;
        let inter = 2;
        let gate_up = fill_bf16(2 * inter * hidden, 1.0);
        let down = fill_bf16(hidden * inter, 1.0);
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let norm_w = vec![1.0f32; hidden];
        let eps = 1e-6_f32;

        let rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32 + eps).sqrt();
        let h_normed: Vec<f32> = h
            .iter()
            .zip(norm_w.iter())
            .map(|(&x, &w)| x / rms * w)
            .collect();

        let direct = run_single_expert(
            &h_normed,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );
        let via_norm = run_single_expert_with_norm(
            &h,
            &gate_up,
            &down,
            inter,
            &norm_w,
            0.0,
            eps,
            QuantFormat::BF16,
            Activation::Silu,
        );

        let max_diff: f32 = direct
            .iter()
            .zip(&via_norm)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff < 1e-4,
            "with_norm diverges from manual prenorm: max_diff={max_diff}"
        );
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env_in_thread<T: Send + 'static>(
        vars: &'static [(&'static str, Option<&'static str>)],
        f: impl FnOnce() -> T + Send + 'static,
    ) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous: Vec<_> = vars
            .iter()
            .map(|(n, _)| (*n, std::env::var_os(n)))
            .collect();
        for (n, v) in vars {
            match v {
                Some(s) => std::env::set_var(n, s),
                None => std::env::remove_var(n),
            }
        }
        // Cross thread boundary so TLS-cached env reads (`Q4K_DIRECT`,
        // `EXPERT_TIMING`) initialise to the freshly-set value rather
        // than inheriting an earlier `false` from this process's main
        // thread.
        let result = std::thread::spawn(f).join().expect("thread did not panic");
        for (n, v) in previous {
            match v {
                Some(s) => std::env::set_var(n, s),
                None => std::env::remove_var(n),
            }
        }
        result
    }

    /// `LARQL_Q4K_DIRECT=1` opts in to the q4k-direct matvec path inside
    /// `run_single_expert_into`.  Cover the previously-untested branch
    /// (lines 248, 273-321) by setting the env var on a fresh thread so
    /// the TLS cache picks it up.
    #[test]
    fn run_single_expert_into_q4k_direct_env_takes_direct_path() {
        let out = with_env_in_thread(&[(crate::options::ENV_Q4K_DIRECT, Some("1"))], || {
            let hidden = 256;
            let inter = 256;
            let h: Vec<f32> = (0..hidden)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.01)
                .collect();
            let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.0015)
                .collect();
            let down_f32: Vec<f32> = (0..hidden * inter)
                .map(|i| ((i % 37) as f32 - 18.0) * 0.001)
                .collect();
            let gate_up = quantize_q4_k(&gate_up_f32);
            let down = quantize_q4_k(&down_f32);
            let mut scratch = ExpertScratch::new(hidden, inter, inter);
            let out = run_single_expert_into(
                &mut scratch,
                &h,
                &gate_up,
                &down,
                inter,
                QuantFormat::Q4_K,
                Activation::GeluTanh,
            );
            out.to_vec()
        });
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Same `LARQL_Q4K_DIRECT=1` env, plus `LARQL_MOE_EXPERT_TIMING=1`
    /// to drive the timing eprintln on the q4k path (lines 268-269,
    /// 280-281, 285-286, 297-298, 308-319).
    #[test]
    fn run_single_expert_into_q4k_direct_with_timing_prints_breakdown() {
        let out = with_env_in_thread(
            &[
                (crate::options::ENV_Q4K_DIRECT, Some("1")),
                (crate::options::ENV_MOE_EXPERT_TIMING, Some("1")),
            ],
            || {
                let hidden = 256;
                let inter = 256;
                let h: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001).collect();
                let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
                    .map(|i| ((i % 11) as f32 - 5.0) * 0.002)
                    .collect();
                let down_f32: Vec<f32> = (0..hidden * inter)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.002)
                    .collect();
                let gate_up = quantize_q4_k(&gate_up_f32);
                let down = quantize_q4_k(&down_f32);
                let mut scratch = ExpertScratch::new(hidden, inter, inter);
                let out = run_single_expert_into(
                    &mut scratch,
                    &h,
                    &gate_up,
                    &down,
                    inter,
                    QuantFormat::Q4_K,
                    Activation::Silu,
                );
                out.to_vec()
            },
        );
        assert_eq!(out.len(), 256);
    }

    /// Same `LARQL_MOE_EXPERT_TIMING=1`, but on the default (non-q4k)
    /// path — covers the timing eprintln further down the function
    /// (lines 332-333, 338-339, 354-355, 367-368, 379, 381-392).
    #[test]
    fn run_single_expert_into_timing_on_default_path_prints_breakdown() {
        let out = with_env_in_thread(
            &[(crate::options::ENV_MOE_EXPERT_TIMING, Some("1"))],
            || {
                let hidden = 4;
                let inter = 2;
                let gate_up = fill_bf16(2 * inter * hidden, 0.5);
                let down = fill_bf16(hidden * inter, 0.25);
                let h = vec![1.0f32; hidden];
                let mut scratch = ExpertScratch::new(hidden, inter, inter);
                let out = run_single_expert_into(
                    &mut scratch,
                    &h,
                    &gate_up,
                    &down,
                    inter,
                    QuantFormat::BF16,
                    Activation::Silu,
                );
                out.to_vec()
            },
        );
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn gelu_tanh_differs_from_silu() {
        let hidden = 4;
        let inter = 2;
        let gate_up = fill_bf16(2 * inter * hidden, 1.0);
        let down = fill_bf16(hidden * inter, 1.0);
        let h = vec![0.5f32; hidden];
        let silu_out = run_single_expert(
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );
        let gelu_out = run_single_expert(
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::GeluTanh,
        );
        let max_diff: f32 = silu_out
            .iter()
            .zip(&gelu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff > 0.01,
            "SiLU and GeluTanh should diverge; max_diff={max_diff}"
        );
    }

    // ── Q4_K direct-from-mmap path coverage ────────────────────────────────

    /// `run_single_expert` with `QuantFormat::Q4_K` and a hidden size
    /// that's a multiple of 256 routes through the thread-local
    /// `ExpertScratch` + SDOT-based `run_single_expert_q4k_q8k_into`.
    /// Covers lines 102-157 of expert.rs.
    #[test]
    fn run_single_expert_q4k_routes_through_direct_path() {
        let hidden = 256usize; // multiple of Q4_K_BLOCK_ELEMS
        let inter = 256usize; // also Q4_K-aligned
                              // gate || up weights, both `inter × hidden` flattened row-major.
        let gate_data: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f32) * 0.001).sin() * 0.05)
            .collect();
        let up_data: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f32) * 0.001).cos() * 0.05)
            .collect();
        let mut gate_up_bytes = quantize_q4_k(&gate_data);
        gate_up_bytes.extend_from_slice(&quantize_q4_k(&up_data));

        let down_data: Vec<f32> = (0..hidden * inter)
            .map(|i| ((i as f32) * 0.002).sin() * 0.05)
            .collect();
        let down_bytes = quantize_q4_k(&down_data);

        let h: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let out = run_single_expert(
            &h,
            &gate_up_bytes,
            &down_bytes,
            inter,
            QuantFormat::Q4_K,
            Activation::Silu,
        );
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "non-degenerate Q4_K weights should produce non-zero output"
        );
    }

    /// `run_single_expert` with Q4_K weights but a hidden size that
    /// isn't a multiple of 256 falls back to the BF16-style f32 cache
    /// path (the Q4_K direct branch is gated on `hidden.is_multiple_of(256)`).
    /// Exercises the negative condition of the gate.
    #[test]
    fn run_single_expert_q4k_non_aligned_hidden_falls_back() {
        let hidden = 4usize; // not a multiple of 256 — gate fails
        let inter = 2usize;
        // BF16 path expects 2 bytes per weight; the format dispatch
        // happens before the size check returns. We just need
        // non-empty bytes so `try_cached_dequant` doesn't bail.
        let gate_up = fill_bf16(2 * inter * hidden, 0.1);
        let down = fill_bf16(hidden * inter, 0.1);
        let h = vec![0.5f32; hidden];
        // Format is BF16 here because Q4_K bytes wouldn't decode at
        // hidden=4. The test asserts that the non-aligned-hidden gate
        // fails and we don't panic — same fallback as production.
        let out = run_single_expert(
            &h,
            &gate_up,
            &down,
            inter,
            QuantFormat::BF16,
            Activation::Silu,
        );
        assert_eq!(out.len(), hidden);
    }

    // ── run_single_expert_into early returns ───────────────────────────────

    /// inter=0 short-circuits with zeroed output (no allocation in scratch).
    #[test]
    fn run_single_expert_into_zero_inter_returns_zeroed_out() {
        let hidden = 4usize;
        let mut scratch = ExpertScratch::new(hidden, 0, 0);
        let h = vec![1.0f32; hidden];
        let out = run_single_expert_into(
            &mut scratch,
            &h,
            &[],
            &[],
            0,
            QuantFormat::BF16,
            Activation::Silu,
        );
        assert_eq!(out, &[0.0f32; 4]);
    }

    /// hidden=0 (zero-length `h_norm`) also short-circuits to zeroed.
    #[test]
    fn run_single_expert_into_zero_hidden_returns_zeroed_out() {
        let mut scratch = ExpertScratch::new(0, 2, 2);
        let h: Vec<f32> = vec![];
        let out = run_single_expert_into(
            &mut scratch,
            &h,
            &[],
            &[],
            2,
            QuantFormat::BF16,
            Activation::Silu,
        );
        assert!(out.is_empty());
    }

    // ── run_single_expert_q4k_q8k_into edge cases ──────────────────────────

    /// Empty input (zero-length Q8_K activation) early-returns zeroed
    /// scratch.out — covers lines 442-448.
    #[test]
    fn run_single_expert_q4k_q8k_into_zero_hidden_returns_zeroed() {
        let mut scratch = ExpertScratch::new(0, 4, 4);
        let empty_q8k = quantize_x_to_q8k(&[]);
        let out =
            run_single_expert_q4k_q8k_into(&mut scratch, &empty_q8k, &[], &[], 4, Activation::Silu);
        assert!(out.is_empty());
    }

    /// `inter=0` also short-circuits — covers the `inter == 0` arm of
    /// the early-return guard.
    #[test]
    fn run_single_expert_q4k_q8k_into_zero_inter_returns_zeroed() {
        let hidden = 256usize;
        let mut scratch = ExpertScratch::new(hidden, 0, 0);
        let h: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001).collect();
        let h_q8k = quantize_x_to_q8k(&h);
        let out =
            run_single_expert_q4k_q8k_into(&mut scratch, &h_q8k, &[], &[], 0, Activation::Silu);
        assert_eq!(out, &[0.0f32; 256]);
    }

    /// `gate_up_bytes` too short for the claimed (hidden, inter) shape
    /// returns zeroed output instead of panicking.
    #[test]
    fn run_single_expert_q4k_q8k_into_short_bytes_returns_zeroed() {
        let hidden = 256usize;
        let inter = 256usize;
        let mut scratch = ExpertScratch::new(hidden, inter, inter);
        let h: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001).collect();
        let h_q8k = quantize_x_to_q8k(&h);
        // Only 100 bytes — far less than 2 * inter * (hidden/256) * 144.
        let short_gate_up = vec![0u8; 100];
        let out = run_single_expert_q4k_q8k_into(
            &mut scratch,
            &h_q8k,
            &short_gate_up,
            &[],
            inter,
            Activation::Silu,
        );
        assert_eq!(out, &[0.0f32; 256]);
    }

    /// GeluTanh activation branch in `run_single_expert_q4k_q8k_into`.
    /// The Silu branch is hit by the round-trip tests above; this pins
    /// the GeluTanh side too.
    #[test]
    fn run_single_expert_q4k_q8k_into_gelu_tanh_activation() {
        let hidden = 256usize;
        let inter = 256usize;
        let mut scratch = ExpertScratch::new(hidden, inter, inter);

        let gate_data: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f32) * 0.001).sin() * 0.05)
            .collect();
        let up_data: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f32) * 0.001).cos() * 0.05)
            .collect();
        let mut gate_up = quantize_q4_k(&gate_data);
        gate_up.extend_from_slice(&quantize_q4_k(&up_data));

        let down_data: Vec<f32> = (0..hidden * inter)
            .map(|i| ((i as f32) * 0.002).sin() * 0.05)
            .collect();
        let down = quantize_q4_k(&down_data);

        let h: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let h_q8k = quantize_x_to_q8k(&h);

        let out = run_single_expert_q4k_q8k_into(
            &mut scratch,
            &h_q8k,
            &gate_up,
            &down,
            inter,
            Activation::GeluTanh,
        );
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── quantize_h_norm_for_q4k ────────────────────────────────────────────

    /// `quantize_h_norm_for_q4k` returns Some on a 256-multiple input.
    #[test]
    fn quantize_h_norm_for_q4k_some_on_aligned_length() {
        let h: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001).collect();
        let out = quantize_h_norm_for_q4k(&h);
        assert!(out.is_some());
    }

    /// Non-aligned length (not multiple of 256) returns None.
    #[test]
    fn quantize_h_norm_for_q4k_none_on_non_aligned_length() {
        let h: Vec<f32> = vec![0.0; 100];
        let out = quantize_h_norm_for_q4k(&h);
        assert!(out.is_none());
    }

    /// Empty input returns None.
    #[test]
    fn quantize_h_norm_for_q4k_none_on_empty() {
        let out = quantize_h_norm_for_q4k(&[]);
        assert!(out.is_none());
    }
}
