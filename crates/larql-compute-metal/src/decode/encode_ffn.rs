//! Step 6 of the decode pipeline: format-aware FFN dispatch.
//!
//! Three production paths on the same `(gate, up, down)` triplet:
//!   - **Q4_KF** — llama.cpp-exact kernel; fused gate+up; `act_buf` then
//!     down via `quant_matvec` (mixed-quant aware).
//!   - **Q4_K** — our kernel; fused gate+up; down via `quant_matvec`
//!     (Gemma 3 4B ships Q6_K down even when gate/up are Q4_K).
//!   - **Q4_0** (legacy) — Q8-input matvec for gate/up; `q4.f32_matvec`
//!     for down.
//!
//! Used to live inline in `decode_token_with_moe_fn`; pulled out here
//! so `decode/mod.rs` stays readable. Behaviour is byte-identical to
//! the original block.
//!
//! All buffer + pipeline references are held in `FfnBufs` and
//! `FfnDims` so the encoder method has a manageable signature.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::MetalBackend;
use larql_compute::FullPipelineLayer;

/// Max `inter_padded` for which the fused Q4_K GEGLU+down kernel is
/// known to be NaN-free.
///
/// Set after Gemma 4 31B (`inter = 21504`) hit a data-dependent NaN at
/// layer 11 despite clean gate/up inputs and finite weight scales (see
/// the block doc on the dispatch site below). 16384 covers Gemma 3 4B
/// (`inter = 10240`), Gemma 4 26B-A4B (`inter = 2112`), Llama 2 7B
/// (`inter = 11008`), Mistral 7B (`inter = 14336`); larger intermediate
/// sizes fall through to the separated GEGLU + matvec path until the
/// fused-kernel NaN root cause is found.
const MAX_FUSED_GEGLU_DOWN_INTER: usize = 16384;

/// Buffer references the FFN block reads or writes. The encoder is
/// passed separately so the method can also borrow `&self`.
pub(super) struct FfnBufs<'a> {
    // Weights for this layer
    pub gate_w: &'a metal::Buffer,
    pub up_w: &'a metal::Buffer,
    pub down_w: &'a metal::Buffer,
    // Inputs
    pub ffn_norm_out: &'a metal::Buffer, // f32 input (Q4_K / Q4_KF paths)
    pub ffn_q8: &'a metal::Buffer,       // Q8 input bytes (Q4_0 path)
    pub ffn_q8s: &'a metal::Buffer,      // Q8 input scales (Q4_0 path)
    // Scratch (gate output reused even on non-gated paths)
    pub gate_out_scratch: &'a metal::Buffer,
    pub up_out: &'a metal::Buffer,
    pub act_buf: &'a metal::Buffer,
    // Output
    pub down_out: &'a metal::Buffer,
}

#[derive(Copy, Clone)]
pub(super) struct FfnDims {
    pub hidden: usize,
    pub inter: usize,
    /// `inter` rounded up to the next multiple of 256 — used by the Q4K
    /// down dispatch when storage is per-row-padded super-blocks.
    pub inter_padded: usize,
}

impl MetalBackend {
    /// Encode the full FFN block (gate / up / activation / down) into
    /// the encoder. `ffn_uses_kquant` selects the path; the function
    /// returns the same `down_out` buffer the caller passed in via
    /// `bufs`. No commit/flush — the caller owns encoder lifecycle.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_ffn_step(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: FfnBufs<'_>,
        dims: FfnDims,
        ffn_uses_kquant: bool,
    ) {
        let FfnDims {
            hidden,
            inter,
            inter_padded,
        } = dims;
        let inter_val = inter as u32;
        let inter_padded_val = inter_padded as u32;
        let hidden_val = hidden as u32;

        let ffn_is_q4kf = layer.gate.format == larql_compute::QuantFormat::Q4_KF;

        if ffn_is_q4kf {
            self.encode_q4kf_ffn(enc, layer, &bufs, hidden, inter, hidden_val, inter_val);
        } else if ffn_uses_kquant {
            self.encode_q4k_ffn(
                enc,
                layer,
                &bufs,
                hidden,
                inter,
                inter_padded,
                hidden_val,
                inter_val,
                inter_padded_val,
            );
        } else {
            self.encode_q4_0_ffn(enc, layer, &bufs, hidden, inter, hidden_val, inter_val);
        }
    }

    // ── Q4_KF (GGUF) ─────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn encode_q4kf_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
        hidden_val: u32,
        inter_val: u32,
    ) {
        use crate::shaders::q4kf_ffn_gate_up as q4kf_gu;
        use crate::shaders::q4kf_qkv_proj as q4kf;
        let n_tgs_down = (hidden as u64).div_ceil(q4kf::ROWS_PER_TG);

        if layer.is_gated() {
            // Fused gate+up
            let n_tgs_per_mat = (inter as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
            enc.set_compute_pipeline_state(&self.ffn.q4kf_ffn_gate_up_pipeline.state);
            enc.set_buffer(0, Some(bufs.gate_w), 0);
            enc.set_buffer(1, Some(bufs.up_w), 0);
            enc.set_buffer(2, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
            enc.set_buffer(4, Some(bufs.up_out), 0);
            enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1),
            );

            // GEGLU
            self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);

            // Down — format-aware (mixed Q4_KF + Q6_K is a real config)
            self.encode_qmv_down(enc, layer, bufs, hidden, inter);
            let _ = n_tgs_down;
        } else {
            // Standard FFN: up + activation + down
            let n_tgs_up = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
            enc.set_compute_pipeline_state(&self.attention.q4kf_proj_pipeline.state);
            enc.set_buffer(0, Some(bufs.up_w), 0);
            enc.set_buffer(1, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(2, Some(bufs.up_out), 0);
            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs_up, 1, 1),
                MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
            );

            self.encode_activation(
                enc,
                layer,
                bufs.up_out,
                bufs.act_buf,
                inter_val,
                inter as u64,
            );

            enc.set_compute_pipeline_state(&self.attention.q4kf_proj_pipeline.state);
            enc.set_buffer(0, Some(bufs.down_w), 0);
            enc.set_buffer(1, Some(bufs.act_buf), 0);
            enc.set_buffer(2, Some(bufs.down_out), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs_down, 1, 1),
                MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
            );
        }
    }

    // ── Q4_K ─────────────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn encode_q4k_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
        inter_padded: usize,
        hidden_val: u32,
        inter_val: u32,
        inter_padded_val: u32,
    ) {
        use crate::shaders::q4k_ffn_gate_up as q4k_gu;
        // Pull `q4k_matvec` dispatch geometry from the bound pipeline so
        // dispatches work for both 4sg and 8sg variants. Hardcoding the
        // 4sg constants while dispatching the 8sg pipeline (production
        // default since 2026-04-28) leaves rows 4..7 of each TG unwritten.
        // Same fix as `trait_impl/quant_matvec.rs::q4k_matvec` and
        // `moe_dispatch.rs`.
        let q4k_matvec_rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let q4k_matvec_threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let n_tgs_down = (hidden as u64).div_ceil(q4k_matvec_rows_per_tg);

        if layer.is_gated() {
            // Variant selection. Production **default is 8sg** as of
            // 2026-04-28 — see below.
            //
            //   - **Default (8sg)**: 8 simdgroups per TG (256 threads,
            //     8 rows/TG). Bit-identical output to the older 4sg
            //     kernel (same math, only TG geometry changed). End-to-end
            //     +2.1% throughput on quiet GPU (12.96 → 12.69 ms/tok,
            //     5-iter median), no regression on long prompts, full
            //     greedy-decode parity validated on a 5-prompt corpus.
            //     First positive end-to-end perf result this session.
            //
            //   - `LARQL_GATE_UP_8SG=0`: opt-OUT to the older 4sg kernel
            //     (production until 2026-04-28). Emergency escape hatch.
            //
            //   - `LARQL_F16_ACC=1`: f16 inner accumulator. Kernel-isolated
            //     1.79× but end-to-end at parity on quiet GPU. Kept as
            //     opt-in for future hardware/fusion scenarios.
            use crate::shaders::q4k_ffn_gate_up_8sg as q4k_gu_8sg;
            use crate::shaders::q4k_ffn_gate_up_coop as q4k_gu_coop;
            // `LARQL_GATE_UP_COOP=1`: cooperative scale-loading variant.
            // Tried 2026-05-01 — null end-to-end (kernel-isolated ALU
            // diagnosis was misleading). Kept opt-in.
            //
            // Flags snapshot at `MetalBackend::new()` (see
            // `metal::flags::DecodeFlags`) — the decode hot path is
            // ~34 layers/tok and `getenv` per layer per flag was a
            // measurable syscall tax.
            let use_coop = self.decode_flags.gate_up_coop;
            let use_4sg = self.decode_flags.gate_up_use_4sg;
            let use_f16 = self.decode_flags.f16_acc;
            let (pipeline, rows_per_tg, threads_per_tg) = if use_coop {
                // Cooperative wins over the other flags — it's the
                // newest variant under measurement.
                (
                    &self.ffn.q4k_ffn_gate_up_coop_pipeline.state,
                    q4k_gu_coop::ROWS_PER_TG,
                    q4k_gu_coop::THREADS_PER_TG,
                )
            } else if use_4sg && use_f16 {
                (
                    &self.ffn.q4k_ffn_gate_up_f16acc_pipeline.state,
                    q4k_gu::ROWS_PER_TG,
                    q4k_gu::THREADS_PER_TG,
                )
            } else if use_4sg {
                (
                    &self.ffn.q4k_ffn_gate_up_pipeline.state,
                    q4k_gu::ROWS_PER_TG,
                    q4k_gu::THREADS_PER_TG,
                )
            } else {
                // Default (8sg) — and f16 is incompatible-untested with
                // 8sg dispatch, so 8sg wins if both flags conflict.
                let _ = use_f16;
                (
                    &self.ffn.q4k_ffn_gate_up_8sg_pipeline.state,
                    q4k_gu_8sg::ROWS_PER_TG,
                    q4k_gu_8sg::THREADS_PER_TG,
                )
            };
            let n_tgs_per_mat = (inter as u64).div_ceil(rows_per_tg);
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(bufs.gate_w), 0);
            enc.set_buffer(1, Some(bufs.up_w), 0);
            enc.set_buffer(2, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
            enc.set_buffer(4, Some(bufs.up_out), 0);
            enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                MTLSize::new(threads_per_tg, 1, 1),
            );

            // Fast path: down is Q4_K → fused activation+down kernel
            // skips the GEGLU dispatch and the inter-sized activation
            // buffer write/read. Verified parity against the separated
            // path in `test_kernel_q4k_geglu_down.rs`.
            //
            // **Q6_K fusion is NOT engaged here.** The Q6_K fused
            // kernels (`q6k_geglu_silu_down` / `q6k_geglu_gelu_tanh_down`)
            // are built, TG-memory-cached, and parity-tested, but routing
            // them on production gemma3-4b-q4k-v2 regresses decode
            // 67.9 → 62.2 tok/s even with TG caching. Root cause: with
            // GELU-tanh the fused inner loop recomputes tanh(gate[i]) once
            // per output row, so 2560 rows = 2560× more tanh() calls than
            // the separated `geglu_gelu_tanh` dispatch. Gate/up bandwidth
            // was never the bottleneck — the 4× intra-TG redundancy the
            // TG-cache fix targeted was L2-cached in practice (gate/up =
            // 80 KB, well within M3 Max GPU L2). Re-enable once a cheaper
            // activation variant avoids the per-row tanh explosion.
            //
            // Slow path: Q6_K / Q4_KF / Q4_0 / Q8_0 → separated
            // GEGLU then format-aware down dispatch.
            // `LARQL_FUSED_Q6K_DOWN=1` was attempted 2026-05-01 to
            // route Q6_K-down + GELU-tanh through a cached-activation
            // fused kernel (`q6k_geglu_gelu_tanh_down_cached_pipeline`).
            // Both the new cached kernel AND the existing production
            // `q6k_geglu_gelu_tanh_down_pipeline` (which a prior memory
            // claimed was "parity-tested") produce wrong output on the
            // current `interleaved_kquant.bin` layout — model emits "The"
            // and stops (early EOS / NaN propagation). Likely the
            // kernel's Q6_K block layout offsets drifted vs the
            // writer in `format/weights/write_kquant`. Real fix needs a
            // kernel-level parity test against the CPU q6k_matvec
            // reference before re-engaging. Until then the env var is
            // a no-op (keeps the kernel and pipeline registered as
            // dead code for the investigation in
            // `larql-inference/ROADMAP.md` G-3 follow-up).
            let use_fused_q6k_down = self.decode_flags.fused_q6k_down
                && layer.down.format == larql_compute::QuantFormat::Q6_K
                && matches!(layer.activation, larql_compute::Activation::GeluTanh);
            if use_fused_q6k_down {
                let kh = &self.ffn.q6k_geglu_gelu_tanh_down_pipeline;
                let n_tgs = (hidden as u64).div_ceil(kh.rows_per_tg);
                enc.set_compute_pipeline_state(&kh.state);
                enc.set_buffer(0, Some(bufs.down_w), 0);
                enc.set_buffer(1, Some(bufs.gate_out_scratch), 0);
                enc.set_buffer(2, Some(bufs.up_out), 0);
                enc.set_buffer(3, Some(bufs.down_out), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                // Note: pass `inter` (not `inter_padded`) — matches the
                // kernel-level parity test in
                // `tests/test_kernel_q6k_geglu_down.rs::metal_fused_q6k_geglu_down`
                // which uses `inter` as K. For Gemma 3 4B `inter == inter_padded`
                // so the difference is moot, but consistency with the
                // verified test path matters.
                enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    metal::MTLSize::new(n_tgs, 1, 1),
                    metal::MTLSize::new(kh.threads_per_tg, 1, 1),
                );
            } else if layer.down.format == larql_compute::QuantFormat::Q4_K
                && inter_padded <= MAX_FUSED_GEGLU_DOWN_INTER
                && self.decode_flags.fused_down
            {
                // Fused GEGLU+down for small-to-medium intermediate sizes.
                //
                // Guard rationale and revival trigger documented on
                // `MAX_FUSED_GEGLU_DOWN_INTER` at the top of this file.
                // Override: LARQL_FUSED_DOWN=0 disables for all sizes;
                //           LARQL_FUSED_DOWN=1 with the guard temporarily
                //           raised (for investigation only).
                self.encode_q4k_fused_geglu_down(
                    enc,
                    layer,
                    bufs,
                    hidden,
                    inter_padded,
                    hidden_val,
                    inter_padded_val,
                );
            } else {
                self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
                use crate::stages::quant_matvec::{self as qmv, Pipelines};
                let pipes = Pipelines {
                    q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
                    q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
                    q6k_matvec: &self.quant.q6k_matvec_pipeline,
                    q4_matvec: &self.q4.matvec,
                    q4k_matmul: None,
                };
                qmv::encode(
                    enc,
                    layer.down.format,
                    bufs.down_w,
                    bufs.act_buf,
                    0,
                    bufs.act_buf,
                    0,
                    bufs.act_buf,
                    0, // Q8 unused for f32 input
                    bufs.down_out,
                    0,
                    &pipes,
                    hidden,
                    inter_padded,
                );
            } // close `else { unfused geglu+matvec chain }`
            let _ = n_tgs_down;
        } else {
            let n_tgs_up = (inter as u64).div_ceil(q4k_matvec_rows_per_tg);
            enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(bufs.up_w), 0);
            enc.set_buffer(1, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(2, Some(bufs.up_out), 0);
            enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs_up, 1, 1),
                MTLSize::new(q4k_matvec_threads_per_tg, 1, 1),
            );

            self.encode_activation(
                enc,
                layer,
                bufs.up_out,
                bufs.act_buf,
                inter_val,
                inter as u64,
            );

            enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(bufs.down_w), 0);
            enc.set_buffer(1, Some(bufs.act_buf), 0);
            enc.set_buffer(2, Some(bufs.down_out), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(
                4,
                4,
                &inter_padded_val as *const u32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(n_tgs_down, 1, 1),
                MTLSize::new(q4k_matvec_threads_per_tg, 1, 1),
            );
        }
    }

    // ── Q4_0 (legacy Q8 input path) ──────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn encode_q4_0_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
        hidden_val: u32,
        inter_val: u32,
    ) {
        // Geometry travels with the q4 matvec KernelHandle — single source
        // of truth, can't drift from the kernel's row map.
        let kernel = &self.q4.matvec;
        let n_tgs_ffn = (inter as u64).div_ceil(kernel.rows_per_tg);
        let tg_size = MTLSize::new(kernel.threads_per_tg, 1, 1);

        if layer.is_gated() {
            // Gate
            enc.set_compute_pipeline_state(&kernel.state);
            enc.set_buffer(0, Some(bufs.gate_w), 0);
            enc.set_buffer(1, Some(bufs.ffn_q8), 0);
            enc.set_buffer(2, Some(bufs.ffn_q8s), 0);
            enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), tg_size);
            // Up (reuse pipeline + bindings, swap matrix and out)
            enc.set_buffer(0, Some(bufs.up_w), 0);
            enc.set_buffer(3, Some(bufs.up_out), 0);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), tg_size);

            self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
        } else {
            enc.set_compute_pipeline_state(&kernel.state);
            enc.set_buffer(0, Some(bufs.up_w), 0);
            enc.set_buffer(1, Some(bufs.ffn_q8), 0);
            enc.set_buffer(2, Some(bufs.ffn_q8s), 0);
            enc.set_buffer(3, Some(bufs.up_out), 0);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), tg_size);

            self.encode_activation(
                enc,
                layer,
                bufs.up_out,
                bufs.act_buf,
                inter_val,
                inter as u64,
            );
        }

        // Down via Q4_0 f32-input matvec (fixed pipeline, no
        // format-aware routing — Q4_0 vindexes are uniform-format).
        enc.set_compute_pipeline_state(&self.q4.f32_matvec);
        enc.set_buffer(0, Some(bufs.down_w), 0);
        enc.set_buffer(1, Some(bufs.act_buf), 0);
        enc.set_buffer(2, Some(bufs.down_out), 0);
        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    // ── Shared sub-steps ─────────────────────────────────────────────────────

    fn encode_geglu(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        inter_val: u32,
        inter_threads: u64,
    ) {
        crate::stages::ffn::assert_metal_activation_supported(
            layer.activation,
            "metal::decode::encode_geglu",
        );
        let geglu = match layer.activation {
            larql_compute::Activation::Silu => &self.ffn.geglu_pipeline,
            larql_compute::Activation::GeluTanh => &self.ffn.geglu_gelu_tanh_pipeline,
            // assert above prevents reaching here.
            larql_compute::Activation::GeluExact | larql_compute::Activation::ReLU => {
                unreachable!()
            }
        };
        enc.set_compute_pipeline_state(geglu);
        enc.set_buffer(0, Some(bufs.gate_out_scratch), 0);
        enc.set_buffer(1, Some(bufs.up_out), 0);
        enc.set_buffer(2, Some(bufs.act_buf), 0);
        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new(inter_threads, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Fused `activation(gate) * up → q4k_matvec(W_down)` in one
    /// dispatch, replacing the separated GEGLU + Q4_K down pair.
    ///
    /// Only fires when `layer.down.format == Q4_K` — gated by the
    /// caller. Picks `silu_down` or `gelu_tanh_down` based on the
    /// layer's activation. Behaviour pinned by
    /// `test_kernel_q4k_geglu_down.rs::*_gemma3_4b_ffn`.
    #[allow(clippy::too_many_arguments)]
    fn encode_q4k_fused_geglu_down(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        _inter_padded: usize,
        hidden_val: u32,
        inter_padded_val: u32,
    ) {
        crate::stages::ffn::assert_metal_activation_supported(
            layer.activation,
            "metal::decode::encode_q4k_fused_geglu_down",
        );
        let kernel = match layer.activation {
            larql_compute::Activation::Silu => &self.ffn.q4k_geglu_silu_down_pipeline,
            larql_compute::Activation::GeluTanh => &self.ffn.q4k_geglu_gelu_tanh_down_pipeline,
            larql_compute::Activation::GeluExact | larql_compute::Activation::ReLU => {
                unreachable!()
            }
        };
        Self::dispatch_fused_geglu_down(enc, kernel, bufs, hidden, hidden_val, inter_padded_val);
    }

    // Q6_K fused-geglu+down thunk previously lived here as
    // `encode_q6k_fused_geglu_down` under `#[allow(dead_code)]`. Removed
    // 2026-05-09: GELU-tanh fusion regresses on production Q6_K shapes
    // (see `encode_q4k_ffn` block doc for the per-row tanh explosion).
    // Production routes Q6_K down via the separated chain (GEGLU dispatch
    // + format-aware matvec). The pipelines `q6k_geglu_silu_down_pipeline`
    // and `q6k_geglu_gelu_tanh_down_cached_pipeline` are still built and
    // available for opt-in benchmarking; reviving the thunk is a 16-line
    // copy of `encode_q4k_fused_geglu_down`.

    /// Shared dispatch body for the Q4_K / Q6_K fused activation+down
    /// kernels. Both kernel families share the same buffer signature
    /// `(W_down, gate, up, out, N, K)` and per-row simdgroup geometry
    /// — only the dequantisation and the activation differ. Pulled
    /// out so adding a future format (FP4? Q3_K?) is one new
    /// `encode_X_fused_geglu_down` thunk.
    fn dispatch_fused_geglu_down(
        enc: &ComputeCommandEncoderRef,
        kernel: &crate::kernels::KernelHandle,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        hidden_val: u32,
        inter_padded_val: u32,
    ) {
        let n_tgs_down = (hidden as u64).div_ceil(kernel.rows_per_tg);
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(bufs.down_w), 0);
        enc.set_buffer(1, Some(bufs.gate_out_scratch), 0);
        enc.set_buffer(2, Some(bufs.up_out), 0);
        enc.set_buffer(3, Some(bufs.down_out), 0);
        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(
            5,
            4,
            &inter_padded_val as *const u32 as *const std::ffi::c_void,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(n_tgs_down, 1, 1),
            MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    // ── Profile-split helpers ────────────────────────────────────────────────
    // Used only when LARQL_PROFILE_SPLIT=1. Each encodes exactly one half of
    // the FFN so a commit/wait boundary between them measures gate+up vs
    // act+down separately. Caller must not commit between the two halves of
    // the same layer — only between gate_up_phase and down_phase.

    /// Encode the gate+up dispatch only. Writes to `bufs.gate_out_scratch`
    /// and `bufs.up_out`; does NOT encode activation or down.
    pub(super) fn encode_ffn_gate_up_phase(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        dims: FfnDims,
        ffn_uses_kquant: bool,
    ) {
        let FfnDims { hidden, inter, .. } = dims;
        let inter_val = inter as u32;
        let hidden_val = hidden as u32;
        let ffn_is_q4kf = layer.gate.format == larql_compute::QuantFormat::Q4_KF;

        if ffn_is_q4kf {
            use crate::shaders::q4kf_ffn_gate_up as q4kf_gu;
            use crate::shaders::q4kf_qkv_proj as q4kf;
            if layer.is_gated() {
                let n = (inter as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
                enc.set_compute_pipeline_state(&self.ffn.q4kf_ffn_gate_up_pipeline.state);
                enc.set_buffer(0, Some(bufs.gate_w), 0);
                enc.set_buffer(1, Some(bufs.up_w), 0);
                enc.set_buffer(2, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
                enc.set_buffer(4, Some(bufs.up_out), 0);
                enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(n * 2, 1, 1),
                    MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1),
                );
            } else {
                let n = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
                enc.set_compute_pipeline_state(&self.attention.q4kf_proj_pipeline.state);
                enc.set_buffer(0, Some(bufs.up_w), 0);
                enc.set_buffer(1, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(2, Some(bufs.up_out), 0);
                enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(n, 1, 1),
                    MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
                );
            }
        } else if ffn_uses_kquant {
            let rows = self.ffn.q4k_ffn_gate_up_8sg_pipeline.rows_per_tg;
            let tgs = self.ffn.q4k_ffn_gate_up_8sg_pipeline.threads_per_tg;
            if layer.is_gated() {
                let n = (inter as u64).div_ceil(rows);
                enc.set_compute_pipeline_state(&self.ffn.q4k_ffn_gate_up_8sg_pipeline.state);
                enc.set_buffer(0, Some(bufs.gate_w), 0);
                enc.set_buffer(1, Some(bufs.up_w), 0);
                enc.set_buffer(2, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
                enc.set_buffer(4, Some(bufs.up_out), 0);
                enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n * 2, 1, 1), MTLSize::new(tgs, 1, 1));
            } else {
                let rpt = self.quant.q4k_matvec_pipeline.rows_per_tg;
                let tpt = self.quant.q4k_matvec_pipeline.threads_per_tg;
                let n = (inter as u64).div_ceil(rpt);
                enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
                enc.set_buffer(0, Some(bufs.up_w), 0);
                enc.set_buffer(1, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(2, Some(bufs.up_out), 0);
                enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), MTLSize::new(tpt, 1, 1));
            }
        } else {
            // Q4_0 path
            let kernel = &self.q4.matvec;
            let n = (inter as u64).div_ceil(kernel.rows_per_tg);
            let tg = MTLSize::new(kernel.threads_per_tg, 1, 1);
            if layer.is_gated() {
                enc.set_compute_pipeline_state(&kernel.state);
                enc.set_buffer(0, Some(bufs.gate_w), 0);
                enc.set_buffer(1, Some(bufs.ffn_q8), 0);
                enc.set_buffer(2, Some(bufs.ffn_q8s), 0);
                enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), tg);
                enc.set_buffer(0, Some(bufs.up_w), 0);
                enc.set_buffer(3, Some(bufs.up_out), 0);
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), tg);
            } else {
                enc.set_compute_pipeline_state(&kernel.state);
                enc.set_buffer(0, Some(bufs.up_w), 0);
                enc.set_buffer(1, Some(bufs.ffn_q8), 0);
                enc.set_buffer(2, Some(bufs.ffn_q8s), 0);
                enc.set_buffer(3, Some(bufs.up_out), 0);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), tg);
            }
        }
    }

    /// Encode the activation (GEGLU/SiLU) + down dispatch only. Reads from
    /// `bufs.gate_out_scratch` / `bufs.up_out` written by `encode_ffn_gate_up_phase`.
    pub(super) fn encode_ffn_down_phase(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        dims: FfnDims,
        ffn_uses_kquant: bool,
    ) {
        let FfnDims {
            hidden,
            inter,
            inter_padded,
        } = dims;
        let inter_val = inter as u32;
        let inter_padded_val = inter_padded as u32;
        let hidden_val = hidden as u32;
        let ffn_is_q4kf = layer.gate.format == larql_compute::QuantFormat::Q4_KF;

        if ffn_is_q4kf {
            if layer.is_gated() {
                self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
                self.encode_qmv_down(enc, layer, bufs, hidden, inter);
            } else {
                self.encode_activation(
                    enc,
                    layer,
                    bufs.up_out,
                    bufs.act_buf,
                    inter_val,
                    inter as u64,
                );
                use crate::shaders::q4kf_qkv_proj as q4kf;
                let n = (hidden as u64).div_ceil(q4kf::ROWS_PER_TG);
                enc.set_compute_pipeline_state(&self.attention.q4kf_proj_pipeline.state);
                enc.set_buffer(0, Some(bufs.down_w), 0);
                enc.set_buffer(1, Some(bufs.act_buf), 0);
                enc.set_buffer(2, Some(bufs.down_out), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(n, 1, 1),
                    MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
                );
            }
        } else if ffn_uses_kquant {
            if layer.is_gated() {
                let use_fused_q6k = self.decode_flags.fused_q6k_down
                    && layer.down.format == larql_compute::QuantFormat::Q6_K
                    && matches!(layer.activation, larql_compute::Activation::GeluTanh);
                if layer.down.format == larql_compute::QuantFormat::Q4_K {
                    self.encode_q4k_fused_geglu_down(
                        enc,
                        layer,
                        bufs,
                        hidden,
                        inter_padded,
                        hidden_val,
                        inter_padded_val,
                    );
                } else if use_fused_q6k {
                    let kh = &self.ffn.q6k_geglu_gelu_tanh_down_pipeline;
                    let n_tgs = (hidden as u64).div_ceil(kh.rows_per_tg);
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(bufs.down_w), 0);
                    enc.set_buffer(1, Some(bufs.gate_out_scratch), 0);
                    enc.set_buffer(2, Some(bufs.up_out), 0);
                    enc.set_buffer(3, Some(bufs.down_out), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(n_tgs, 1, 1),
                        metal::MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                } else {
                    self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
                    self.encode_qmv_down(enc, layer, bufs, hidden, inter_padded);
                }
            } else {
                self.encode_activation(
                    enc,
                    layer,
                    bufs.up_out,
                    bufs.act_buf,
                    inter_val,
                    inter as u64,
                );
                let rpt = self.quant.q4k_matvec_pipeline.rows_per_tg;
                let tpt = self.quant.q4k_matvec_pipeline.threads_per_tg;
                let n = (hidden as u64).div_ceil(rpt);
                enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
                enc.set_buffer(0, Some(bufs.down_w), 0);
                enc.set_buffer(1, Some(bufs.act_buf), 0);
                enc.set_buffer(2, Some(bufs.down_out), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(
                    4,
                    4,
                    &inter_padded_val as *const u32 as *const std::ffi::c_void,
                );
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), MTLSize::new(tpt, 1, 1));
            }
        } else {
            // Q4_0
            if layer.is_gated() {
                self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
            } else {
                self.encode_activation(
                    enc,
                    layer,
                    bufs.up_out,
                    bufs.act_buf,
                    inter_val,
                    inter as u64,
                );
            }
            enc.set_compute_pipeline_state(&self.q4.f32_matvec);
            enc.set_buffer(0, Some(bufs.down_w), 0);
            enc.set_buffer(1, Some(bufs.act_buf), 0);
            enc.set_buffer(2, Some(bufs.down_out), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
    }

    fn encode_activation(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        in_buf: &metal::Buffer,
        out_buf: &metal::Buffer,
        inter_val: u32,
        inter_threads: u64,
    ) {
        crate::stages::ffn::assert_metal_activation_supported(
            layer.activation,
            "metal::decode::encode_activation",
        );
        let pipe = match layer.activation {
            larql_compute::Activation::Silu => &self.ffn.silu_pipeline,
            larql_compute::Activation::GeluTanh => &self.ffn.gelu_tanh_pipeline,
            larql_compute::Activation::GeluExact | larql_compute::Activation::ReLU => {
                unreachable!()
            }
        };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(in_buf), 0);
        enc.set_buffer(1, Some(out_buf), 0);
        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new(inter_threads, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn encode_qmv_down(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
    ) {
        use crate::stages::quant_matvec::{self as qmv, Pipelines};
        let pipes = Pipelines {
            q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
            q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
            q6k_matvec: &self.quant.q6k_matvec_pipeline,
            q4_matvec: &self.q4.matvec,
            q4k_matmul: None,
        };
        qmv::encode(
            enc,
            layer.down.format,
            bufs.down_w,
            bufs.act_buf,
            0,
            bufs.act_buf,
            0,
            bufs.act_buf,
            0,
            bufs.down_out,
            0,
            &pipes,
            hidden,
            inter,
        );
    }
}
