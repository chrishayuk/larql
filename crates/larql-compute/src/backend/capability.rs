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
    /// Decode using local GPU dispatch for Q4_K per-layer expert tensors.
    DecodeQ4KMoe,
    /// Per-stage timing decode (`decode_token_split_profile`).
    DecodeProfile,
    /// Multi-position prefill with KV cache population (`prefill_kquant`).
    PrefillQ4,
    /// Heterogeneous attention geometry: layers in the same model can
    /// have different `head_dim`, `num_kv_heads`, `num_q_heads`,
    /// `rope_base`, or sliding-window flags. Required by Gemma 4 31B
    /// (50 sliding-attention layers at head_dim=256/16-kv plus 10
    /// global-attention layers at head_dim=512/4-kv). A backend that
    /// returns `false` here will be rejected by callers before decode
    /// rather than silently corrupting KV state at the first
    /// non-uniform layer. The dense-uniform case (every layer reports
    /// the same geometry) is supported by every backend regardless of
    /// this capability.
    HeterogeneousAttention,

    // ── KvDispatch (engine-facing intents) ────────────────────────────
    // See `crates/larql-inference/docs/specs/compute-backend-redesign.md`
    // §5 for the intent vocabulary these capabilities cover. Defaults to
    // `false`; backends override per intent as native kernels land. The
    // current `CpuBackend` will say `true` for the f32 CPU-fallback
    // variants of these once Step 2b lands; Metal/Vulkan say `true` for
    // the fused/specialised variants as those land in steps 5/6.
    /// Native fused single-kernel attention step over a `KvHandle`.
    /// Backends without this fall back to decomposed dispatch.
    FusedAttentionStep,
    /// Specialised windowed-attention shader variant (window size
    /// baked into the pipeline via specialisation constants).
    /// `standard:window=N` benefits when present.
    WindowedAttentionStep,
    /// Native K/V codec kernel (TurboQuant compress/decompress on
    /// the GPU). Backends without this fall back to dequant → f32
    /// append → requant.
    NativeKvCodec,
    /// Boundary-residual upload can be pipelined with the first
    /// attention dispatch (Metal explicit barrier, Vulkan equivalent).
    /// Apollo's compressed path benefits when present.
    PipelinedBoundaryUpload,
    /// Fused `residual_add + rmsnorm` in one kernel (D-RMS-FUSE target).
    FusedResidualNorm,
    /// Backend can allocate a device-resident `KvHandle` (Metal
    /// `MTLBuffer`, Vulkan `VkBuffer`). CPU returns a host-allocated
    /// handle. Strictly, every backend will say `true` here; the flag
    /// exists so future "compute backend that doesn't store K/V at all"
    /// configurations (e.g., a router proxy) can opt out.
    KvHandleNative,
}
