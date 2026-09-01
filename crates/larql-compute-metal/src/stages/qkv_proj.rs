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
    /// All three projections are Q8_0 → `q8_qkv_proj` shader against
    /// pre-quantised Q8 input plus external weight scales. Previously a
    /// hand-written triple-equality bypass at two call sites; a bucket
    /// here so the route table owns it (capability audit, slice 1).
    FusedQ8,
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

/// What the QKV kernels consume as their input vector for a given
/// weight format: raw f32 activations, or the pre-quantised Q8
/// (int8 + per-32-block scales) pair produced by `rms_norm_q8`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QkvInputEncoding {
    F32,
    Q8,
}

/// The input encoding a single projection format requires, or `None`
/// for formats no Metal QKV kernel serves (floats, BitNet ternary).
fn input_encoding_of(f: QuantFormat) -> Option<QkvInputEncoding> {
    match f {
        QuantFormat::Q4_K | QuantFormat::Q4_KF | QuantFormat::Q6_K => Some(QkvInputEncoding::F32),
        // Q5_K is an f32-input k-quant like its siblings above, but no
        // Metal kernel serves it yet. `None` keeps that a planning gap
        // rather than a route that dispatches nothing.
        QuantFormat::Q5_K => None,
        QuantFormat::Q4_0 | QuantFormat::Q8_0 => Some(QkvInputEncoding::Q8),
        // MXFP4 joins the `None` set for a different reason than the
        // floats: kernels exist, but they are expert kernels taking a
        // separate e8m0 binding, and no QKV projection is MXFP4 in any
        // supported checkpoint (attention weights sit outside the MoE
        // quantisation in the OCP layout). If one ever is, this wants a
        // real route, not a silent `Q8`.
        QuantFormat::BF16
        | QuantFormat::F16
        | QuantFormat::F32
        | QuantFormat::I2S
        | QuantFormat::MXFP4 => None,
    }
}

/// A validated QKV execution plan: which kernel arrangement runs the
/// three projections, and which input encoding the norm stage must
/// produce for them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct QkvPlan {
    pub route: QkvFormatRoute,
    pub input: QkvInputEncoding,
}

/// Pick the fused-QKV route for a `(q, k, v)` format triple.
///
/// Prefer [`plan_qkv`], which also derives (and validates) the input
/// encoding; this remains for callers that only need the route.
pub fn pick_qkv_route(q: QuantFormat, k: QuantFormat, v: QuantFormat) -> QkvFormatRoute {
    match (q, k, v) {
        (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K) => QkvFormatRoute::UniformQ4K,
        (QuantFormat::Q4_KF, QuantFormat::Q4_KF, QuantFormat::Q4_KF) => QkvFormatRoute::UniformQ4Kf,
        (QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q6_K) => QkvFormatRoute::MixedQ4kQ6kV,
        (QuantFormat::Q8_0, QuantFormat::Q8_0, QuantFormat::Q8_0) => QkvFormatRoute::FusedQ8,
        _ => QkvFormatRoute::PerProjection,
    }
}

/// The single source of truth for how a `(q, k, v)` format triple
/// executes: kernel route plus the input encoding the norm stage must
/// produce. Every QKV dispatcher — decode, prefill, hybrid — consults
/// this instead of testing one operand's format (the single-operand
/// tests were audit findings F2 and the prefill `uses_f32_input` gate,
/// both of which mis-served the other two operands).
///
/// # Panics
/// - if any projection's format has no Metal QKV kernel (floats, I2S);
/// - if the three formats disagree on input encoding — no kernel
///   arrangement can serve one dispatch with two input encodings, and
///   the previous behaviour was to prep one encoding and hand the
///   other-formats' kernels an unpopulated buffer.
pub fn plan_qkv(q: QuantFormat, k: QuantFormat, v: QuantFormat) -> QkvPlan {
    let (eq, ek, ev) = match (
        input_encoding_of(q),
        input_encoding_of(k),
        input_encoding_of(v),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => panic!(
            "no Metal QKV kernel serves (wq={q:?}, wk={k:?}, wv={v:?}); \
             supported formats: Q4_K, Q4_KF, Q6_K, Q4_0, Q8_0"
        ),
    };
    assert!(
        eq == ek && ek == ev,
        "QKV formats (wq={q:?}, wk={k:?}, wv={v:?}) disagree on input \
         encoding ({eq:?}/{ek:?}/{ev:?}); one dispatch cannot serve two \
         input encodings"
    );
    QkvPlan {
        route: pick_qkv_route(q, k, v),
        input: eq,
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
    /// `shaders::q4k_q6k_qkv_proj::Kernel` — mixed Q4_K Q/K + Q6_K V
    /// (4 rows/TG, 128 threads). Same 0-10 binding table as the
    /// uniform kernels.
    Q4kQ6kV,
}

impl FusedQkvKernel {
    fn rows_per_tg(self) -> u64 {
        match self {
            Self::Q4k => crate::shaders::q4k_qkv_proj::ROWS_PER_TG,
            Self::Q4kf => crate::shaders::q4kf_qkv_proj::ROWS_PER_TG,
            Self::Q4kQ6kV => crate::shaders::q4k_q6k_qkv_proj::ROWS_PER_TG,
        }
    }
    fn threads_per_tg(self) -> u64 {
        match self {
            Self::Q4k => crate::shaders::q4k_qkv_proj::THREADS_PER_TG,
            Self::Q4kf => crate::shaders::q4kf_qkv_proj::THREADS_PER_TG,
            Self::Q4kQ6kV => crate::shaders::q4k_q6k_qkv_proj::THREADS_PER_TG,
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
    // Every fused f32-input QKV kernel derives its superblock count as
    // `K / 256` (truncating); a misaligned K silently drops the tail
    // and desynchronises the row stride (audit F17).
    assert!(
        hidden.is_multiple_of(256),
        "fused QKV kernels require hidden % 256 == 0; got {hidden}"
    );
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
    assert!(
        hidden <= crate::shaders::q8_attn_proj::MAX_K,
        "q8_qkv_proj stages its input in threadgroup memory capped at K = {}; \
         hidden {hidden} would corrupt threadgroup memory (audit F13)",
        crate::shaders::q8_attn_proj::MAX_K,
    );
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
    // One threadgroup covers ROWS_PER_TG rows (one per simdgroup); the
    // previous `total_rows` grid over-dispatched 8x, with every excess
    // TG re-staging the Q8 input into threadgroup memory before its
    // simdgroups discovered they were all out of range (capability
    // audit F19). Geometry from the shader module the pipeline was
    // built from, not a literal.
    use crate::shaders::q8_attn_proj as q8;
    enc.dispatch_thread_groups(
        MTLSize::new(total_rows.div_ceil(q8::ROWS_PER_TG), 1, 1),
        MTLSize::new(q8::THREADS_PER_TG, 1, 1),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::QuantFormat;

    /// The plan carries route + input encoding as one validated fact.
    #[test]
    fn plan_qkv_derives_route_and_input_encoding() {
        use QuantFormat::*;
        let cases = [
            (
                (Q4_K, Q4_K, Q4_K),
                QkvFormatRoute::UniformQ4K,
                QkvInputEncoding::F32,
            ),
            (
                (Q4_KF, Q4_KF, Q4_KF),
                QkvFormatRoute::UniformQ4Kf,
                QkvInputEncoding::F32,
            ),
            (
                (Q4_K, Q4_K, Q6_K),
                QkvFormatRoute::MixedQ4kQ6kV,
                QkvInputEncoding::F32,
            ),
            // Q8_0 was a hand-written triple check at two call sites;
            // now a route bucket owned by the table.
            (
                (Q8_0, Q8_0, Q8_0),
                QkvFormatRoute::FusedQ8,
                QkvInputEncoding::Q8,
            ),
            // Per-projection with a coherent encoding on each side.
            (
                (Q6_K, Q6_K, Q6_K),
                QkvFormatRoute::PerProjection,
                QkvInputEncoding::F32,
            ),
            (
                (Q4_0, Q4_0, Q4_0),
                QkvFormatRoute::PerProjection,
                QkvInputEncoding::Q8,
            ),
        ];
        for ((q, k, v), route, input) in cases {
            let plan = plan_qkv(q, k, v);
            assert_eq!(plan.route, route, "{q:?}/{k:?}/{v:?}");
            assert_eq!(plan.input, input, "{q:?}/{k:?}/{v:?}");
        }
    }

    /// One dispatch cannot serve two input encodings: a triple mixing
    /// an f32-input format with a Q8-input format previously had one
    /// encoding prepped and handed the other formats' kernels an
    /// unpopulated buffer. Now refused at planning time.
    #[test]
    #[should_panic(expected = "disagree on input encoding")]
    fn plan_qkv_refuses_mixed_input_encodings() {
        let _ = plan_qkv(QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_0);
    }

    /// Float attention weights have no Metal QKV kernel; the silent
    /// alternative was quant kernels reading float bytes.
    #[test]
    #[should_panic(expected = "no Metal QKV kernel")]
    fn plan_qkv_refuses_float_formats() {
        let _ = plan_qkv(QuantFormat::F32, QuantFormat::F32, QuantFormat::F32);
    }

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

        // The kernel indexes weight scales as `[rows * blocks]` (see
        // `Wqs` in shaders/q8_attn_proj.rs), not `[rows]`. Allocating
        // one scale per row left `row_scales[b]` reading past the end of
        // the buffer for every row but the first — usually landing on
        // zeroed memory, but occasionally on another test's leftovers,
        // which is why this test failed non-finite roughly one run in
        // three under a full suite and passed in isolation.
        // `blocks = K / 32` in the kernel is the legacy (Q8_0) block size.
        // Ask for it rather than spelling 32 here — re-spelling a layout
        // constant locally is what produced the under-allocation below.
        let blocks = hidden / larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;
        let zero_q8 = vec![0i8; (q_rows + kv_rows + kv_rows) * hidden];
        let zero_f32 = vec![0.0f32; (q_rows + kv_rows + kv_rows) * blocks];
        let wq_buf = m.bufs.transient_from_i8(&zero_q8[..q_rows * hidden]);
        let wk_buf = m.bufs.transient_from_i8(&zero_q8[..kv_rows * hidden]);
        let wv_buf = m.bufs.transient_from_i8(&zero_q8[..kv_rows * hidden]);
        let wq_scale = m.bufs.transient_from_f32(&zero_f32[..q_rows * blocks]);
        let wk_scale = m.bufs.transient_from_f32(&zero_f32[..kv_rows * blocks]);
        let wv_scale = m.bufs.transient_from_f32(&zero_f32[..kv_rows * blocks]);
        let q8_in = m.bufs.transient_from_i8(&vec![0i8; hidden]);
        let q8s_in = m.bufs.transient_from_f32(&vec![0.0f32; blocks]);
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
        crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/stages/qkv_proj.rs:569",
        )
        .expect("command buffer completed");

        let q = crate::buffers::read_buffer_f32(&q_out, q_rows);
        let k = crate::buffers::read_buffer_f32(&k_out, kv_rows);
        let v = crate::buffers::read_buffer_f32(&v_out, kv_rows);
        assert!(q.iter().all(|x| x.is_finite()));
        assert!(k.iter().all(|x| x.is_finite()));
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
