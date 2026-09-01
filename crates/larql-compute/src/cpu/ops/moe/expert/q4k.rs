use super::scratch::ExpertScratch;
use crate::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_matvec_into, q6k_q8k_matvec_into, quantize_x_to_q8k, quantize_x_to_q8k_into,
    Q8KActivation,
};
use crate::cpu::ops::KernelShapeError;
use crate::options;

/// One integer matvec, dispatched by the store's weight format. The two
/// kernels share the Q8_K activation and differ only in super-block
/// decode; naming the dispatch once keeps the three call sites below from
/// each re-spelling the format match.
#[inline]
fn kq_q8k_matvec_into(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
    format: crate::QuantFormat,
) -> Result<(), KernelShapeError> {
    match format {
        crate::QuantFormat::Q6_K => q6k_q8k_matvec_into(out, q8k_x, w, rows, cols),
        _ => q4k_q8k_matvec_into(out, q8k_x, w, rows, cols),
    }
}

/// The expert paths below return a borrowed scratch slice and have no
/// error channel; a refused operand shape is a defect in the caller's
/// slab arithmetic, and the step ends here rather than with zeros.
fn refuse(kernel: &'static str, e: KernelShapeError) -> ! {
    panic!("{kernel}: {e}")
}

/// A gate+up slab shorter than two `inter`-row halves: refused by name,
/// with the same fields the kernels report.
fn short_gate_up_slab(
    kernel: &'static str,
    have: usize,
    need: usize,
    inter: usize,
    cols: usize,
) -> ! {
    panic!(
        "{}",
        KernelShapeError {
            kernel,
            out_len: inter,
            rows: inter,
            x_len: cols,
            cols,
            weight_bytes: have,
            needed_bytes: need,
        }
    )
}

/// Bytes per weight row at `cols` elements under `format`. `cols` must be
/// a whole number of super-blocks — the padded-row invariant the writer
/// guarantees and `MoeLayerWeights::gate_up_cols` derives.
#[inline]
fn row_block_bytes(cols: usize, format: crate::QuantFormat) -> usize {
    let (block_elems, block_bytes) = format
        .packed_block_layout()
        .expect("integer expert path requires a block format");
    cols / block_elems * block_bytes
}

/// Row-parallel sibling of [`run_single_expert_kq_q8k_into`] for layers
/// whose ACTIVE expert count cannot fill the thread pool.
///
/// The block forward parallelises across active experts, which is the
/// right schedule when top-k ≥ the pool width (Gemma's top-8 on 8
/// threads). GPT-OSS routes top-4, so expert-parallel left half the
/// machine idle on every layer — the experts here instead run serially
/// (in the caller) and each matvec fans its rows across the pool via
/// `q4k_q8k_matvec_parallel`. Same math, same kernels, different
/// schedule; output differs from the serial form only by f32
/// accumulation order inside a row, which is none at all (rows are
/// independent).
pub fn run_single_expert_kq_q8k_parallel_into<'s>(
    scratch: &'s mut ExpertScratch,
    h_norm_q8k: &Q8KActivation,
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    format: crate::QuantFormat,
    mlp: crate::ExpertMlp<'_>,
) -> &'s [f32] {
    use crate::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_parallel;

    let cols = h_norm_q8k.qs.len();
    let hidden_out = scratch.out.len();
    if inter == 0 || cols == 0 {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }
    let (block, _) = format
        .packed_block_layout()
        .expect("integer expert path requires a block format");
    let inter_padded = inter.div_ceil(block) * block;
    let half = inter * row_block_bytes(cols, format);
    if gate_up_bytes.len() < 2 * half {
        short_gate_up_slab(
            "run_single_expert_kq_q8k_parallel_into",
            gate_up_bytes.len(),
            2 * half,
            inter,
            cols,
        );
    }
    let tag = format.registry_tag();
    q4k_q8k_matvec_parallel(
        &mut scratch.gate_out,
        h_norm_q8k,
        &gate_up_bytes[..half],
        inter,
        cols,
        tag,
    )
    .unwrap_or_else(|e| refuse("run_single_expert_kq_q8k_parallel_into", e));
    q4k_q8k_matvec_parallel(
        &mut scratch.up_out,
        h_norm_q8k,
        &gate_up_bytes[half..2 * half],
        inter,
        cols,
        tag,
    )
    .unwrap_or_else(|e| refuse("run_single_expert_kq_q8k_parallel_into", e));
    for j in 0..inter {
        let g = scratch.gate_out[j] + mlp.gate_bias(j);
        let u = scratch.up_out[j] + mlp.up_bias(j);
        scratch.act[j] = mlp.rule.combine(g, u);
    }
    super::super::within_expert::prune_act(&mut scratch.act, inter);
    quantize_x_to_q8k_into(&mut scratch.act_q8k, &scratch.act);
    q4k_q8k_matvec_parallel(
        &mut scratch.out,
        &scratch.act_q8k,
        down_bytes,
        hidden_out,
        inter_padded,
        tag,
    )
    .unwrap_or_else(|e| refuse("run_single_expert_kq_q8k_parallel_into", e));
    mlp.add_down_bias(&mut scratch.out);
    &scratch.out
}
// `q4k_q8k_gate_up_into` exists for future kernel exploration but is not
// wired into the hot path — see comment in `run_single_expert_q4k_q8k_into`.

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
    mlp: crate::ExpertMlp<'_>,
) -> &'s [f32] {
    run_single_expert_kq_q8k_into(
        scratch,
        h_norm_q8k,
        gate_up_bytes,
        down_bytes,
        inter,
        crate::QuantFormat::Q4_K,
        mlp,
    )
}

/// Format-general sibling of [`run_single_expert_q4k_q8k_into`]: the same
/// integer expert kernel over Q4_K or Q6_K super-blocks.
///
/// `h_norm_q8k` is quantised at the gate/up matrices' STORED row width —
/// padded to a super-block boundary by the writer — so `qs.len()` IS the
/// weight column count; zero padding contributes zero to every dot.
#[allow(clippy::too_many_arguments)]
pub fn run_single_expert_kq_q8k_into<'s>(
    scratch: &'s mut ExpertScratch,
    h_norm_q8k: &Q8KActivation,
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    format: crate::QuantFormat,
    mlp: crate::ExpertMlp<'_>,
) -> &'s [f32] {
    // Per-stage timing for kernel diagnosis.  Enable with
    // `LARQL_KERNEL_TIMING=1`.  Cached in TLS to avoid syscall per call.
    thread_local! {
        static KERNEL_TIMING: bool = options::env_flag(options::ENV_KERNEL_TIMING);
    }
    let timing = KERNEL_TIMING.with(|t| *t);

    // Gate/up column count = the activation's quantised width (the STORED,
    // block-padded row width). The down matvec's output rows stay at the
    // LOGICAL hidden — `scratch.out`'s length — which the padding never
    // changes; conflating the two reads gate rows at the wrong stride.
    let cols = h_norm_q8k.qs.len();
    let hidden_out = scratch.out.len();
    if inter == 0 || cols == 0 {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }

    let (block, _) = format
        .packed_block_layout()
        .expect("integer expert path requires a block format");
    let inter_padded = inter.div_ceil(block) * block;
    let half = inter * row_block_bytes(cols, format);
    if gate_up_bytes.len() < 2 * half {
        short_gate_up_slab(
            "run_single_expert_kq_q8k_into",
            gate_up_bytes.len(),
            2 * half,
            inter,
            cols,
        );
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
    kq_q8k_matvec_into(
        &mut scratch.gate_out,
        h_norm_q8k,
        gate_bytes,
        inter,
        cols,
        format,
    )
    .unwrap_or_else(|e| refuse("run_single_expert_kq_q8k_into", e));
    let t_gate = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    kq_q8k_matvec_into(
        &mut scratch.up_out,
        h_norm_q8k,
        up_bytes,
        inter,
        cols,
        format,
    )
    .unwrap_or_else(|e| refuse("run_single_expert_kq_q8k_into", e));
    let t_up = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // Combine gate ⊙ up under the layer's rule, biases first.  Padding
    // columns (`inter..inter_padded`) stay at their zero-initialised value
    // across reuses (we never write them), matching the existing
    // convention in `run_single_expert_into`.
    for j in 0..inter {
        let g = scratch.gate_out[j] + mlp.gate_bias(j);
        let u = scratch.up_out[j] + mlp.up_bias(j);
        scratch.act[j] = mlp.rule.combine(g, u);
    }
    let t_act = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // Within-expert feature routing (aim-validation probe). No-op (one relaxed
    // atomic load) unless a schedule is installed via
    // `super::within_expert::set_routing` — keeps this hot path byte-exact in
    // production. Prunes the `inter` post-activation features to the configured
    // subset before the `down` matvec; padding columns stay zero.
    super::super::within_expert::prune_act(&mut scratch.act, inter);

    // Quantise the per-expert activation to Q8_K in-place into the
    // caller-owned scratch buffer (no allocation on the hot path —
    // eliminates the 150 µs alloc spikes that drag par_iter wall up).
    quantize_x_to_q8k_into(&mut scratch.act_q8k, &scratch.act);
    let t_act_q8k = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // down matvec: out[hidden] = down_W[hidden, inter_padded] @ act
    kq_q8k_matvec_into(
        &mut scratch.out,
        &scratch.act_q8k,
        down_bytes,
        hidden_out,
        inter_padded,
        format,
    )
    .unwrap_or_else(|e| refuse("run_single_expert_kq_q8k_into", e));
    mlp.add_down_bias(&mut scratch.out);
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
