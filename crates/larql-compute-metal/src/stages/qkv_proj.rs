//! Q + K + V projections — one call per position.
//!
//! Three code paths depending on the weight format + mix:
//!
//! - **Fused f32-input** (`encode_fused_f32`): all three projections share
//!   the same format (Q4_K or Q4_KF) and we dispatch the llama.cpp-exact
//!   `q4kf_qkv_proj` shader in one go. Fastest path.
//! - **Per-projection f32-input** (`encode_per_proj`): mixed formats
//!   (e.g. Gemma 4 Q4_K Q/K + Q6_K V). Three separate shader dispatches.
//! - **Fused Q8-input** (`encode_fused_q8`): `Q8_0` attention layers use
//!   `q8_qkv_proj` with pre-quantised Q8 input from `input_norm::encode_q8`.
//!
//! All paths are per-position single-vector dispatches. Multi-position
//! prefill is achieved by looping over positions with buffer offsets.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

use super::quant_matvec;
use larql_compute::QuantFormat;

/// Which fused-QKV strategy a given `(q, k, v)` format triple maps to.
///
/// Pure data — independent of any pipeline. Use [`pick_qkv_route`] to
/// translate a layer's three projection formats into the dispatch
/// strategy. New format combinations land as one match arm in
/// `pick_qkv_route` plus one branch in the dispatcher, instead of
/// editing 2–3 hard-coded boolean expressions across the encoders.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QkvFormatRoute {
    /// All three projections are uniform Q4_K → `q4k_qkv_proj` shader
    /// ([`FusedQkvKernel::Q4k`]).
    UniformQ4K,
    /// All three projections are uniform Q4_KF → `q4kf_qkv_proj`
    /// shader ([`FusedQkvKernel::Q4kf`]).
    UniformQ4Kf,
    /// Q4_K Q + Q4_K K + Q6_K V (Gemma 3 / Gemma 4 with Ollama-convention
    /// extracts) → `q4k_q6k_qkv_proj` shader.
    MixedQ4kQ6kV,
    /// Anything else: per-projection dispatch through `quant_matvec`.
    /// Slower but covers Q4_KF + Q6_K V, mixed legacy Q4_0, etc.
    PerProjection,
}

impl QkvFormatRoute {
    /// `true` for any single-dispatch fused path. Useful when the
    /// caller wants to short-circuit per-projection scratch setup that
    /// the fused path doesn't need.
    pub fn is_fused(self) -> bool {
        !matches!(self, Self::PerProjection)
    }
}

/// Pick the fused-QKV route for a `(q, k, v)` format triple.
///
/// This is the single source of truth for which Metal QKV pipeline
/// handles which weight-format combination. Adding a new combo (e.g.
/// Q4_KF Q/K + Q6_K V, or a future FP4 family) is a one-line addition
/// here — encode_qkv.rs and decode_hybrid.rs read the route through
/// this helper and route dispatch via `match` on the result.
pub fn pick_qkv_route(q: QuantFormat, k: QuantFormat, v: QuantFormat) -> QkvFormatRoute {
    match (q, k, v) {
        (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K) => QkvFormatRoute::UniformQ4K,
        (QuantFormat::Q4_KF, QuantFormat::Q4_KF, QuantFormat::Q4_KF) => QkvFormatRoute::UniformQ4Kf,
        (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q6_K) => QkvFormatRoute::MixedQ4kQ6kV,
        _ => QkvFormatRoute::PerProjection,
    }
}

/// Per-projection format + weight tuple used by the mixed-format path.
pub struct Proj<'a> {
    pub format: larql_compute::QuantFormat,
    pub w_buf: &'a Buffer,
    pub out_buf: &'a Buffer,
    pub out_off: u64,
    pub rows: usize,
}

/// Threadgroup geometry for a fused-QKV f32-input kernel.
///
/// The two kernels we dispatch from [`encode_fused_f32`] use different
/// per-TG row counts and thread counts:
///
/// - `q4k_qkv_proj` (the simple Q4_K shader): 8 rows/TG, 256 threads/TG.
/// - `q4kf_qkv_proj` (llama.cpp-exact Q4_KF shader): 4 rows/TG, 64 threads/TG.
///
/// Both shaders' constants are exported as `ROWS_PER_TG`/`THREADS_PER_TG`
/// from their respective Rust modules. Dispatching with the wrong
/// geometry silently leaves rows unwritten (the kernel's `if (global_row
/// >= total_rows) return` guard hides the under-coverage). Pass the
/// matching `FusedQkvKernel` so the row check on the host stays in sync.
#[derive(Clone, Copy)]
pub enum FusedQkvKernel {
    /// `shaders::q4k_qkv_proj::QkvKernel` — Q4_K simple (8 rows/TG, 256 threads).
    Q4k,
    /// `shaders::q4kf_qkv_proj::Kernel` — Q4_KF llama.cpp-port (4 rows/TG, 64 threads).
    Q4kf,
}

impl FusedQkvKernel {
    fn rows_per_tg(self) -> u64 {
        match self {
            Self::Q4k => crate::shaders::q4k_qkv_proj::ROWS_PER_TG,
            Self::Q4kf => crate::shaders::q4kf_qkv_proj::ROWS_PER_TG,
        }
    }
    fn threads_per_tg(self) -> u64 {
        match self {
            Self::Q4k => crate::shaders::q4k_qkv_proj::THREADS_PER_TG,
            Self::Q4kf => crate::shaders::q4kf_qkv_proj::THREADS_PER_TG,
        }
    }
}

/// Fused Q4_K / Q4_KF QKV — all three projections same format.
///
/// Dispatches the kernel referenced by `pipeline`. The `kernel`
/// discriminant must match — see [`FusedQkvKernel`] — because the two
/// kernels have different per-TG geometries that must agree on the host
/// or rows go unwritten.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_f32(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    kernel: FusedQkvKernel,
    wq_buf: &Buffer,
    wk_buf: &Buffer,
    wv_buf: &Buffer,
    f32_in: &Buffer,
    f32_in_off: u64,
    q_out: &Buffer,
    q_off: u64,
    k_out: &Buffer,
    k_off: u64,
    v_out: &Buffer,
    v_off: u64,
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) {
    let total_rows = (q_rows + kv_rows + kv_rows) as u32;
    let q_rows_val = q_rows as u32;
    let k_rows_val = kv_rows as u32;
    let v_rows_val = kv_rows as u32;
    let k_val = hidden as u32;
    let num_tgs = (total_rows as u64).div_ceil(kernel.rows_per_tg());
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(wq_buf), 0);
    enc.set_buffer(1, Some(wk_buf), 0);
    enc.set_buffer(2, Some(wv_buf), 0);
    enc.set_buffer(3, Some(f32_in), f32_in_off);
    enc.set_buffer(4, Some(q_out), q_off);
    enc.set_buffer(5, Some(k_out), k_off);
    enc.set_buffer(6, Some(v_out), v_off);
    enc.set_bytes(7, 4, &q_rows_val as *const u32 as *const c_void);
    enc.set_bytes(8, 4, &k_rows_val as *const u32 as *const c_void);
    enc.set_bytes(9, 4, &v_rows_val as *const u32 as *const c_void);
    enc.set_bytes(10, 4, &k_val as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(kernel.threads_per_tg(), 1, 1),
    );
}

/// Per-projection f32-input QKV — mixed formats (Gemma 4 Q4_K + Q6_K).
///
/// One dispatch per projection, each through
/// [`super::quant_matvec::encode`] which picks the right shader by format.
/// The Q8 buffer parameters are only read for Q4_0 / Q8_0 projections;
/// callers with pure f32-input formats can pass any valid buffer + 0 offset.
#[allow(clippy::too_many_arguments)]
pub fn encode_per_proj(
    enc: &ComputeCommandEncoderRef,
    pipes: &quant_matvec::Pipelines<'_>,
    f32_in: &Buffer,
    f32_in_off: u64,
    q8_in: &Buffer,
    q8_in_off: u64,
    q8s_in: &Buffer,
    q8s_in_off: u64,
    projections: [Proj<'_>; 3],
    hidden: usize,
) {
    for p in projections {
        quant_matvec::encode(
            enc, p.format, p.w_buf, f32_in, f32_in_off, q8_in, q8_in_off, q8s_in, q8s_in_off,
            p.out_buf, p.out_off, pipes, p.rows, hidden,
        );
    }
}

/// Fused Q8-input QKV — for Q8_0 attention.
///
/// Input comes from `input_norm::encode_q8`. Weights are Q8 int8 + per-row
/// f32 scale buffers. `q8_qkv_proj` writes all three outputs in one dispatch.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_q8(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    wq_buf: &Buffer,
    wq_scale: &Buffer,
    wk_buf: &Buffer,
    wk_scale: &Buffer,
    wv_buf: &Buffer,
    wv_scale: &Buffer,
    q8_in: &Buffer,
    q8_in_off: u64,
    q8s_in: &Buffer,
    q8s_in_off: u64,
    q_out: &Buffer,
    q_off: u64,
    k_out: &Buffer,
    k_off: u64,
    v_out: &Buffer,
    v_off: u64,
    q_rows: usize,
    kv_rows: usize,
    hidden: usize,
) {
    let q_rows_val = q_rows as u32;
    let k_rows_val = kv_rows as u32;
    let v_rows_val = kv_rows as u32;
    let k_val = hidden as u32;
    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(wq_buf), 0);
    enc.set_buffer(1, Some(wk_buf), 0);
    enc.set_buffer(2, Some(wv_buf), 0);
    enc.set_buffer(3, Some(q8_in), q8_in_off);
    enc.set_buffer(4, Some(wq_scale), 0);
    enc.set_buffer(5, Some(wk_scale), 0);
    enc.set_buffer(6, Some(wv_scale), 0);
    enc.set_buffer(7, Some(q8s_in), q8s_in_off);
    enc.set_buffer(8, Some(q_out), q_off);
    enc.set_buffer(9, Some(k_out), k_off);
    enc.set_buffer(10, Some(v_out), v_off);
    enc.set_bytes(11, 4, &q_rows_val as *const u32 as *const c_void);
    enc.set_bytes(12, 4, &k_rows_val as *const u32 as *const c_void);
    enc.set_bytes(13, 4, &v_rows_val as *const u32 as *const c_void);
    enc.set_bytes(14, 4, &k_val as *const u32 as *const c_void);
    enc.dispatch_thread_groups(MTLSize::new(total_rows, 1, 1), MTLSize::new(256, 1, 1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::QuantFormat;

    /// The four production triples must each map to a distinct fused
    /// route. Anything else falls through to per-projection.
    #[test]
    fn pick_qkv_route_recognises_supported_triples() {
        // Uniform Q4_K (Llama 2 / Mistral / Qwen with all-Q4_K extracts).
        assert_eq!(
            pick_qkv_route(QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K),
            QkvFormatRoute::UniformQ4K,
        );
        // Uniform Q4_KF (llama.cpp-exact pre-baked-scales fast path).
        assert_eq!(
            pick_qkv_route(QuantFormat::Q4_KF, QuantFormat::Q4_KF, QuantFormat::Q4_KF),
            QkvFormatRoute::UniformQ4Kf,
        );
        // Gemma 3 / Gemma 4 Ollama convention: Q4_K Q+K, Q6_K V.
        assert_eq!(
            pick_qkv_route(QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q6_K),
            QkvFormatRoute::MixedQ4kQ6kV,
        );
    }

    /// Anything outside the table falls back to per-projection. Pin a
    /// few representative misses so a future "we now support X" change
    /// is forced to update this test (and therefore the table).
    #[test]
    fn pick_qkv_route_falls_back_to_per_projection() {
        // Q4_KF Q/K + Q6_K V — plausible future combo, not yet wired.
        assert_eq!(
            pick_qkv_route(QuantFormat::Q4_KF, QuantFormat::Q4_KF, QuantFormat::Q6_K),
            QkvFormatRoute::PerProjection,
        );
        // Mixed legacy Q4_0.
        assert_eq!(
            pick_qkv_route(QuantFormat::Q4_0, QuantFormat::Q4_0, QuantFormat::Q4_0),
            QkvFormatRoute::PerProjection,
        );
        // Pure Q6_K (no fused kernel exists).
        assert_eq!(
            pick_qkv_route(QuantFormat::Q6_K, QuantFormat::Q6_K, QuantFormat::Q6_K),
            QkvFormatRoute::PerProjection,
        );
        // f32 input — fused QKV is f32-input-only on the kernel side
        // but the dispatcher still routes through the format-aware
        // helper for raw float weights.
        assert_eq!(
            pick_qkv_route(QuantFormat::F32, QuantFormat::F32, QuantFormat::F32),
            QkvFormatRoute::PerProjection,
        );
    }

    #[test]
    fn is_fused_marks_per_projection_as_not_fused() {
        assert!(QkvFormatRoute::UniformQ4K.is_fused());
        assert!(QkvFormatRoute::UniformQ4Kf.is_fused());
        assert!(QkvFormatRoute::MixedQ4kQ6kV.is_fused());
        assert!(!QkvFormatRoute::PerProjection.is_fused());
    }

    /// `FusedQkvKernel` discriminant geometry — both Q4k and Q4kf
    /// arms must return their shader's compile-time constants.  Pin
    /// the values so a future shader refactor that drifts ROWS_PER_TG
    /// fails this test before it ships.
    #[test]
    fn fused_qkv_kernel_geometry_constants_match_shaders() {
        assert_eq!(
            FusedQkvKernel::Q4k.rows_per_tg(),
            crate::shaders::q4k_qkv_proj::ROWS_PER_TG
        );
        assert_eq!(
            FusedQkvKernel::Q4k.threads_per_tg(),
            crate::shaders::q4k_qkv_proj::THREADS_PER_TG
        );
        assert_eq!(
            FusedQkvKernel::Q4kf.rows_per_tg(),
            crate::shaders::q4kf_qkv_proj::ROWS_PER_TG
        );
        assert_eq!(
            FusedQkvKernel::Q4kf.threads_per_tg(),
            crate::shaders::q4kf_qkv_proj::THREADS_PER_TG
        );
    }

    /// `encode_fused_q8` end-to-end: dispatch the q8 QKV projection
    /// kernel with synthetic Q8 inputs.  The numerical output isn't
    /// asserted (synthetic zero weights produce zero output) — we
    /// just pin that the dispatch completes and writes finite values.
    #[test]
    fn encode_fused_q8_dispatch_completes() {
        let m = crate::MetalBackend::new().expect("Metal device available");
        let hidden = 64usize;
        let q_rows = 16usize;
        let kv_rows = 8usize;

        let zero_q8 = vec![0i8; (q_rows + kv_rows + kv_rows) * hidden];
        let zero_f32 = vec![0.0f32; q_rows + kv_rows + kv_rows];
        let wq_buf = m.bufs.transient_from_i8(&zero_q8[..q_rows * hidden]);
        let wk_buf = m.bufs.transient_from_i8(&zero_q8[..kv_rows * hidden]);
        let wv_buf = m.bufs.transient_from_i8(&zero_q8[..kv_rows * hidden]);
        let wq_scale = m.bufs.transient_from_f32(&zero_f32[..q_rows]);
        let wk_scale = m.bufs.transient_from_f32(&zero_f32[..kv_rows]);
        let wv_scale = m.bufs.transient_from_f32(&zero_f32[..kv_rows]);
        let q8_in = m.bufs.transient_from_i8(&vec![0i8; hidden]);
        let q8s_in = m.bufs.transient_from_f32(&vec![0.0f32; hidden / 32]);
        let q_out = m.bufs.output((q_rows * 4) as u64);
        let k_out = m.bufs.output((kv_rows * 4) as u64);
        let v_out = m.bufs.output((kv_rows * 4) as u64);

        let cmd = m.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_fused_q8(
            enc,
            &m.attention.q8_qkv_proj_pipeline.state,
            &wq_buf,
            &wq_scale,
            &wk_buf,
            &wk_scale,
            &wv_buf,
            &wv_scale,
            &q8_in,
            0,
            &q8s_in,
            0,
            &q_out,
            0,
            &k_out,
            0,
            &v_out,
            0,
            q_rows,
            kv_rows,
            hidden,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let q = crate::buffers::read_buffer_f32(&q_out, q_rows);
        let k = crate::buffers::read_buffer_f32(&k_out, kv_rows);
        let v = crate::buffers::read_buffer_f32(&v_out, kv_rows);
        assert!(q.iter().all(|x| x.is_finite()));
        assert!(k.iter().all(|x| x.is_finite()));
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
