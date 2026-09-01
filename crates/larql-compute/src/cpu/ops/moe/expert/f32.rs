use super::super::cache::{try_cached_dequant, ExpertF32};
use super::super::math::{matmul_vec, matmul_vec_into};
use super::q4k::run_single_expert_kq_q8k_into;
use super::scratch::ExpertScratch;
use crate::cpu::ops::q4_common::q4k_matvec_into;
use crate::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k_into, Q8KActivation};
use crate::options;

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
    mlp: crate::ExpertMlp<'_>,
) -> Vec<f32> {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 {
        return vec![0.0f32; hidden];
    }

    // Storage layout (matches `format/weights/write_layers.rs::quantize_moe_entries`):
    //   gate_up: [2*inter, hidden]              never padded
    //   down:    [hidden, inter_padded]         block formats pad inter→block multiple
    // BF16/F32 have no padding for either. The padding rule is the same
    // authority the writer uses (`MoeWeightLayout::down_cols` goes through
    // `packed_block_layout` too), so a new block format cannot pad on one
    // side only. See `forward::cpu_moe_forward` for the expanded explanation;
    // this single-expert path mirrors it exactly so the remote-expert HTTP
    // endpoint and local in-process MoE share the same numerics.
    let inter_padded = match format.packed_block_layout() {
        Some((block_elems, _)) => inter.div_ceil(block_elems) * block_elems,
        None => inter,
    };
    // The gate/up matrices' STORED row width (block-padded by the writer).
    // Both paths read rows at this stride and pad the input to match.
    let weight_cols = crate::stored_gate_up_cols(gate_up_bytes.len(), inter, format, hidden);
    let h_w: std::borrow::Cow<'_, [f32]> = if weight_cols == hidden {
        std::borrow::Cow::Borrowed(h_norm)
    } else {
        let mut padded = vec![0.0f32; weight_cols];
        padded[..hidden].copy_from_slice(h_norm);
        std::borrow::Cow::Owned(padded)
    };

    // Integer direct-from-mmap path (NEON SDOT on aarch64), Q4_K or Q6_K.
    // Routes through `run_single_expert_kq_q8k_into` with a thread-local
    // `ExpertScratch` so the per-call allocations of gate_out / up_out /
    // act / act_q8k go away — only the final `Vec<f32>` output is
    // allocated for the function's return type.  Profiling (2026-05-01)
    // showed K=8 × per-call allocs as the dominant HTTP-path bottleneck
    // once the kernel itself got below ~80 µs.  Set
    // `LARQL_DISABLE_Q4K_DIRECT=1` to opt out (kernel-debug A/B).
    if matches!(format, crate::QuantFormat::Q4_K | crate::QuantFormat::Q6_K)
        && weight_cols.is_multiple_of(256)
        && !super::super::q4k_direct_disabled()
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
                quantize_x_to_q8k_into(&mut hb, &h_w);
                let h2 = run_single_expert_kq_q8k_into(
                    scratch,
                    &hb,
                    gate_up_bytes,
                    down_bytes,
                    inter,
                    format,
                    mlp,
                );
                h2.to_vec()
            })
        });
    }

    let gate_up_w = try_cached_dequant(gate_up_bytes, format, 2 * inter * weight_cols)
        .unwrap_or_else(|err| panic!("{err}"));
    if gate_up_w.is_empty() {
        return vec![0.0f32; hidden];
    }
    let gate_w = &gate_up_w[..inter * weight_cols];
    let up_w = &gate_up_w[inter * weight_cols..2 * inter * weight_cols];

    let gate_out = matmul_vec(&h_w, gate_w, inter, weight_cols);
    let up_out = matmul_vec(&h_w, up_w, inter, weight_cols);

    // Build inner activation at `inter_padded` so the down matmul (which
    // expects `inter_padded` columns under block formats) sees zero in the
    // padding. Biases join before the combine, matching the reference
    // (`ExpertWeightFfn::run_expert` adds them right after the matmuls).
    let mut hidden_state: Vec<f32> = vec![0.0f32; inter_padded];
    for j in 0..inter {
        let g = gate_out[j] + mlp.gate_bias(j);
        let u = up_out[j] + mlp.up_bias(j);
        hidden_state[j] = mlp.rule.combine(g, u);
    }

    let down_w = try_cached_dequant(down_bytes, format, hidden * inter_padded)
        .unwrap_or_else(|err| panic!("{err}"));
    if down_w.is_empty() {
        return vec![0.0f32; hidden];
    }
    let mut out = matmul_vec(&hidden_state, &down_w, hidden, inter_padded);
    mlp.add_down_bias(&mut out);
    out
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
    mlp: crate::ExpertMlp<'_>,
) -> &'s [f32] {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }

    let inter_padded = match format.packed_block_layout() {
        Some((block_elems, _)) => inter.div_ceil(block_elems) * block_elems,
        None => inter,
    };
    // Stored gate/up row width (block-padded by the writer); the input is
    // zero-padded to match when they differ. See `run_single_expert`.
    let weight_cols = crate::stored_gate_up_cols(gate_up_bytes.len(), inter, format, hidden);
    let h_w: std::borrow::Cow<'_, [f32]> = if weight_cols == hidden {
        std::borrow::Cow::Borrowed(h_norm)
    } else {
        let mut padded = vec![0.0f32; weight_cols];
        padded[..hidden].copy_from_slice(h_norm);
        std::borrow::Cow::Owned(padded)
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

    let gate_w_size = inter * weight_cols;
    // f32 path: hold the cached Arc for the duration of the call so the
    // gate_w / up_w slices below borrow into the cache's payload directly.
    // The previous `v.to_vec()` here copied ~12 MB per call on cache hit,
    // which dominated the per-expert wall time at Gemma 4 26B-A4B sizes.
    let gate_up_w_arc: Option<ExpertF32> = if q4k_path {
        None
    } else {
        let v = try_cached_dequant(gate_up_bytes, format, 2 * inter * weight_cols)
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
        let row_block_bytes = (weight_cols / larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
            * larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
        let half = inter * row_block_bytes;
        let gate_bytes = &gate_up_bytes[..half];
        let up_bytes = &gate_up_bytes[half..2 * half];
        q4k_matvec_into(&mut scratch.gate_out, &h_w, gate_bytes, inter, weight_cols)
            .unwrap_or_else(|e| panic!("run_single_expert_into: {e}"));
        let t_gate = if timing { Some(t.elapsed()) } else { None };
        if timing {
            t = std::time::Instant::now();
        }
        q4k_matvec_into(&mut scratch.up_out, &h_w, up_bytes, inter, weight_cols)
            .unwrap_or_else(|e| panic!("run_single_expert_into: {e}"));
        let t_up = if timing { Some(t.elapsed()) } else { None };
        if timing {
            t = std::time::Instant::now();
        }
        for j in 0..inter {
            let g = scratch.gate_out[j] + mlp.gate_bias(j);
            let u = scratch.up_out[j] + mlp.up_bias(j);
            scratch.act[j] = mlp.rule.combine(g, u);
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
        )
        .unwrap_or_else(|e| panic!("run_single_expert_into: {e}"));
        mlp.add_down_bias(&mut scratch.out);
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
    matmul_vec_into(&mut scratch.gate_out, &h_w, gate_w, inter, weight_cols);
    let t_gate = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    matmul_vec_into(&mut scratch.up_out, &h_w, up_w, inter, weight_cols);
    let t_up = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // Build inner activation at `inter_padded`; padding columns
    // (`inter..inter_padded`) stay at their zero-initialised value across
    // reuses since we never write them.
    for j in 0..inter {
        let g = scratch.gate_out[j] + mlp.gate_bias(j);
        let u = scratch.up_out[j] + mlp.up_bias(j);
        scratch.act[j] = mlp.rule.combine(g, u);
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
    mlp.add_down_bias(&mut scratch.out);
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
