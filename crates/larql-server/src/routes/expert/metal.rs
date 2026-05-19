//! Metal MoE expert dispatch.
//!
//! Currently opt-in (`LARQL_USE_METAL_EXPERTS=1`) while the inter=704
//! accuracy bug on Gemma 4 26B-A4B-it is being debugged. See
//! `larql-compute/ROADMAP.md → "Open: Metal MoE expert kernel — accuracy
//! bug at inter=704"` for kernel-side investigation.
//!
//! When the bug is fixed and this becomes default-on, the only thing
//! to change is `metal::run_experts_metal_batch`'s opt-in gate at the
//! top of the function.

#![cfg(all(feature = "metal-experts", target_os = "macos"))]

use std::sync::Arc;
use std::time::Instant;

use larql_compute_metal::{MetalBackend, MoeScratch};

use crate::env_flags;
use crate::error::ServerError;
use crate::state::AppState;

use super::cpu::run_experts_cpu_batch;

/// Run a layer's pre-selected experts on the Metal GPU and return the weighted
/// sum of their outputs.  Returns `Ok(None)` when Metal is unavailable, the
/// model is not hybrid-MoE, or per-layer Q4_K weights are missing — caller
/// should fall back to the per-expert CPU path.
///
/// `h_post_attn` is the residual the streaming RPC carries (pre-norm not yet
/// applied).  `expert_ids` and `expert_weights` are already client-routed (no
/// router run on the server).  Returns the weighted sum WITHOUT post-experts
/// norm; the client applies post-norm once after summing across shards.
pub fn run_experts_metal_batch(
    state: &AppState,
    layer: usize,
    h_post_attn: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> Result<Option<Vec<f32>>, ServerError> {
    let timing_enabled = env_flags::moe_timing_enabled();
    // 2026-04-30 ACCURACY ISSUE: the Metal MoE expert dispatch (both
    // `run_experts_preselected_metal` and `run_experts_prestaged_metal`,
    // and the in-process `gpu_moe_dispatch_with_scratch`) produces
    // numerically wrong expert outputs for Gemma 4 26B-A4B-it (inter=704,
    // hidden=2816, top_k=8). Cosine similarity vs CPU reference ≈ 0.7;
    // |metal| consistently ~70% of |cpu|. Same model produces "Paris"
    // via CPU experts and "answer is in the context of France" via Metal
    // experts. Bug appears to be in the q4k_ffn_gate_up + GELU + q4k_matvec
    // chain when applied to the 704-wide intermediate dim — the same
    // shaders work correctly for dense FFN at inter=2560/10240/21504.
    // Until the kernel is fixed, default to CPU expert dispatch even on
    // a build that linked the Metal backend.  Set LARQL_USE_METAL_EXPERTS=1
    // to opt back in (e.g. for kernel-debugging runs).
    if !env_flags::use_metal_experts() || env_flags::disable_metal_experts() {
        return Ok(None);
    }
    let t_start = Instant::now();

    let model = state.model_or_err(None)?;
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let arch = &*weights.arch;
    let t_state = t_start.elapsed();

    if !arch.is_hybrid_moe() || !weights.has_per_layer_ffn() {
        return Ok(None);
    }

    let backend_slot = model.metal_backend.get_or_init(MetalBackend::new);
    let Some(backend) = backend_slot.as_ref() else {
        return Ok(None);
    };

    let hidden = model.config.hidden_size;
    if h_post_attn.len() != hidden {
        return Err(ServerError::BadRequest(format!(
            "residual length {} != hidden_size {hidden}",
            h_post_attn.len()
        )));
    }
    let inter = arch.moe_intermediate_size();
    let top_k = arch.num_experts_per_token();

    let t_pre = Instant::now();
    // Apply pre_experts_norm on CPU (cheap; matches the per-expert CPU path's
    // behaviour in `run_single_expert_with_norm`).
    //   out[i] = h[i] / sqrt(mean(h²) + eps) * (norm[i] + norm_offset)
    let h_norm: Vec<f32> = if let Some(norm_key) = arch.moe_pre_experts_norm_key(layer) {
        if let Some(pre_norm) = weights.vectors.get(&norm_key) {
            let eps = arch.norm_eps();
            let norm_offset = arch.norm_weight_offset();
            let pre_norm = pre_norm.as_slice();
            let rms = (h_post_attn.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
            h_post_attn
                .iter()
                .zip(pre_norm.iter())
                .map(|(x, w)| x / rms * (w + norm_offset))
                .collect()
        } else {
            h_post_attn.to_vec()
        }
    } else {
        h_post_attn.to_vec()
    };
    let t_norm = t_pre.elapsed();

    // get_expert_bytes maps expert_id → (gate_up_bytes, down_bytes) mmap slices.
    let get_expert_bytes =
        |eid: usize| -> Option<(&[u8], &[u8])> { weights.get_layer_entry_bytes(layer, eid) };

    // Pre-stage per-expert weights as cache-backed Metal buffers.
    let t_buf_start = Instant::now();
    let mut expert_bufs: Vec<(
        larql_compute_metal::MetalBuffer,
        larql_compute_metal::MetalBuffer,
    )> = Vec::with_capacity(expert_ids.len());
    let mut filtered_weights: Vec<f32> = Vec::with_capacity(expert_ids.len());
    for (i, &eid) in expert_ids.iter().enumerate() {
        if let Some((gu, dn)) = weights.get_layer_entry_bytes(layer, eid) {
            expert_bufs.push((
                backend.cached_buffer_for_bytes(gu),
                backend.cached_buffer_for_bytes(dn),
            ));
            filtered_weights.push(expert_weights[i]);
        }
    }
    let t_bufs = t_buf_start.elapsed();

    // Look up (or create + cache) the MoE scratch for this layer's shape.
    let t_scratch_start = Instant::now();
    let scratch_key = (top_k, hidden, inter);
    let mut scratch_cache = model.moe_scratches.lock().expect("moe_scratches poisoned");
    let scratch = scratch_cache
        .entry(scratch_key)
        .or_insert_with(|| Arc::new(MoeScratch::new_public(backend, top_k, hidden, inter)));
    let t_scratch = t_scratch_start.elapsed();

    let t_gpu_start = Instant::now();
    // 2026-04-30: switched from `run_experts_prestaged_metal` (per-expert
    // pre-cached buffers, per-expert dispatch) back to
    // `run_experts_preselected_metal` (byte-copy into shared scratch,
    // ONE big dispatch for all K experts). The prestaged variant produces
    // numerically wrong expert outputs (cos≈0.7 vs CPU reference, |metal|
    // consistently ~70% of |cpu|).
    let _ = (&expert_bufs, &filtered_weights);
    let result = backend.run_experts_preselected_metal(
        &h_norm,
        expert_ids,
        expert_weights,
        scratch.as_ref(),
        get_expert_bytes,
    );
    let t_gpu = t_gpu_start.elapsed();

    // LARQL_METAL_VS_CPU_DEBUG=1 — recompute via CPU and print element-wise
    // max diff. Used to localise the metal-experts accuracy bug. Slow
    // (every layer × every token does both paths), so opt-in only.
    if env_flags::metal_vs_cpu_debug() {
        match run_experts_cpu_batch(state, layer, h_post_attn, expert_ids, expert_weights) {
            Ok(cpu_out) => {
                let max_abs_diff = result
                    .iter()
                    .zip(cpu_out.iter())
                    .fold(0.0f32, |acc, (m, c)| acc.max((m - c).abs()));
                let metal_norm = (result.iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt();
                let cpu_norm = (cpu_out.iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt();
                let cos = {
                    let dot: f32 = result.iter().zip(cpu_out.iter()).map(|(a, b)| a * b).sum();
                    let na: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let nb: f32 = cpu_out.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if na > 0.0 && nb > 0.0 {
                        dot / (na * nb)
                    } else {
                        f32::NAN
                    }
                };
                eprintln!(
                    "[metal-vs-cpu] L{layer:02} K={} max|Δ|={max_abs_diff:.4e} \
                     |metal|={metal_norm:.4} |cpu|={cpu_norm:.4} cos={cos:.6}",
                    expert_ids.len()
                );
            }
            Err(e) => {
                eprintln!("[metal-vs-cpu] L{layer:02} cpu reference failed: {e}");
            }
        }
    }

    if timing_enabled {
        eprintln!(
            "[expert_metal_batch] layer={layer} experts={} state={:.2}ms norm={:.2}ms \
             scratch={:.2}ms bufs={:.2}ms gpu={:.2}ms total={:.2}ms",
            expert_ids.len(),
            t_state.as_secs_f32() * 1000.0,
            t_norm.as_secs_f32() * 1000.0,
            t_scratch.as_secs_f32() * 1000.0,
            t_bufs.as_secs_f32() * 1000.0,
            t_gpu.as_secs_f32() * 1000.0,
            t_start.elapsed().as_secs_f32() * 1000.0,
        );
    }

    Ok(Some(result))
}
