//! `Capability` — what a backend says it can accelerate.
//!
//! `ComputeBackend` exposes many `Option<…>`-returning methods; each
//! is a "try and see" capability probe. That's awkward because callers
//! have to call the method, check for `None`, and fall back. The
//! [`Capability`] enum lets the caller branch *before* the call:
//!
//! ```ignore
//! if backend.supports(Capability::F32Gemv) {
//!     backend.f32_gemv(w, x).unwrap()
//! } else {
//!     backend.matmul_transb(q_row, w).row(0).to_vec()
//! }
//! ```
//!
//! A backend lists what it can do via [`crate::ComputeBackend::supports`].
//! Default impl returns `false` for everything; override to enable.

/// What a backend can accelerate. Independent flags — a backend
/// typically says yes to several.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Specialised f32 row-per-simdgroup gemv (lm-head logits).
    F32Gemv,
    /// f16-weight gemv (saves the 2× clone for tied-embedding lm-head).
    F16Gemv,
    /// Per-format quant matvec via [`crate::ComputeBackend::quant_matvec`].
    QuantMatVec,
    /// Q4 vector-matrix scatter (down-projection's transposed shape).
    Q4VecMat,
    /// Batched gate+up Q4 matvec for prefill seq>1.
    Q4PairBatch,
    /// Full-pipeline Q4 attention + FFN in one command buffer.
    FullPipelineQ4,
    /// Multi-layer Q4 FFN chain in one command buffer.
    MultiLayerQ4Ffn,
    /// KV-cached single-token decode (`decode_token`).
    DecodeToken,
    /// Decode with a remote-MoE callback (`decode_token_with_moe`).
    DecodeMoe,
    /// Per-stage timing decode (`decode_token_split_profile`).
    DecodeProfile,
    /// Multi-position prefill with KV cache population (`prefill_q4`).
    PrefillQ4,

    // ── Capabilities introduced by openspec change `cuda-and-rotorquant-kv` ──
    /// Backend dispatches via CUDA / cuBLAS / cudarc. Set by `CudaBackend`.
    Cuda,
    /// Fused Flash-Attention-v2-style attention kernel is available.
    FlashAttentionV2,
    /// KV-cache compression via RotorQuant `iso3` / `planar3` / `iso4` / `planar4`.
    KvCompressionRotorQuant,
}
