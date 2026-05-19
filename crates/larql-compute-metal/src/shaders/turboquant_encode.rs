//! TurboQuant encode shader: fused WHT + Lloyd-Max quantize for KV vectors.
//!
//! Input:  fp32 vector buffer [batch × d]
//! Output: packed quantized indices + norms
//!
//! Algorithm per vector:
//!   1. Compute L2 norm (simdgroup reduce)
//!   2. Normalize in-place
//!   3. Apply deterministic sign flips (decorrelation)
//!   4. WHT butterfly passes (log2(d) stages)
//!   5. Apply sign flips again (symmetric — makes transform self-inverse)
//!   6. Lloyd-Max centroid lookup (table in constant memory)
//!   7. Pack 4-bit indices into uint32 output
//!
//! One threadgroup per vector. Threadgroup size = d (e.g., 256).
//! Expected: <0.01ms per vector on M3 Max.
//!
//! ## Retention rationale (ADR-017)
//!
//! **Status**: experimental. Built into `all_shaders()` for diag
//! (`metal/diag/shader_bench.rs` benchmarks it) but not wired into
//! production decode/prefill — KV-vector quantisation remains as
//! TurboQuant on the Rust side.
//!
//! Kept on disk because (a) it pairs with `turboquant_decode` and
//! demonstrates the WHT-fused quantise loop in pure-MSL form, and
//! (b) future KV-cache compression experiments can re-engage the
//! pair via the diag bench harness without re-deriving the kernel.
//!
//! **Removal trigger**: if KV-cache compression gets a non-WHT
//! production direction (rotational tokenisation, rank-1 sketch,
//! etc.), demote both `turboquant_*` shaders.

pub const SHADER: &str = r#"
// TurboQuant 4-bit encode: normalize → sign flip → WHT → sign flip → quantize → pack.
// One threadgroup per vector. Threadgroup size = d (e.g., 256).

// Lloyd-Max 4-bit centroids for N(0, sigma) after WHT.
constant float tq4_centroids[16] = {
    -0.1089, -0.0782, -0.0588, -0.0427,
    -0.0283, -0.0148, -0.0050,  0.0050,
     0.0148,  0.0283,  0.0427,  0.0588,
     0.0782,  0.1089,  0.1500,  0.2000
};

constant float tq4_boundaries[15] = {
    -0.0936, -0.0685, -0.0508, -0.0355,
    -0.0216, -0.0099,  0.0000,  0.0099,
     0.0216,  0.0355,  0.0508,  0.0685,
     0.0936,  0.1295,  0.1750
};

// Deterministic sign flip: bit 16 of (i * 2654435761)
static inline bool tq_sign_flip(uint i) {
    return ((i * 2654435761u) >> 16) & 1u;
}

kernel void turboquant_encode_4bit(
    device const float* input    [[buffer(0)]],  // [batch, d] — input vectors
    device float*       norms    [[buffer(1)]],  // [batch] — output norms
    device uchar*       packed   [[buffer(2)]],  // [batch, d/2] — packed 4-bit indices
    constant uint&      d        [[buffer(3)]],  // vector dimension (power of 2)
    constant uint&      batch    [[buffer(4)]],  // number of vectors
    uint  elem    [[thread_position_in_threadgroup]],
    uint  vec_idx [[threadgroup_position_in_grid]],
    threadgroup float* shared [[threadgroup(0)]])
{
    if (vec_idx >= batch || elem >= d) return;

    uint base = vec_idx * d;
    shared[elem] = input[base + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1: L2 norm (thread 0 computes, all read)
    float sum_sq = 0.0f;
    if (elem == 0) {
        for (uint i = 0; i < d; i++) {
            sum_sq += shared[i] * shared[i];
        }
        norms[vec_idx] = sqrt(sum_sq);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = norms[vec_idx];
    float inv_norm = (norm > 1e-12f) ? (1.0f / norm) : 0.0f;

    // Step 2: Normalize
    shared[elem] *= inv_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Sign flips (D matrix)
    if (tq_sign_flip(elem)) shared[elem] = -shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: WHT butterfly passes (rename 'half' → 'hstep' to avoid MSL keyword)
    for (uint hstep = 1; hstep < d; hstep *= 2) {
        uint blk = hstep * 2;
        uint blk_idx = elem / blk;
        uint within = elem % blk;
        if (within < hstep) {
            uint j = blk_idx * blk + within;
            float a = shared[j];
            float b = shared[j + hstep];
            shared[j] = a + b;
            shared[j + hstep] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize by 1/sqrt(d) for orthonormality
    shared[elem] *= 1.0f / sqrt(float(d));

    // Step 5: Sign flips again (symmetric D·H·D)
    if (tq_sign_flip(elem)) shared[elem] = -shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 6: Lloyd-Max quantize
    float y = shared[elem];
    uint idx = 0;
    for (uint b = 0; b < 15; b++) {
        if (y > tq4_boundaries[b]) idx = b + 1;
    }

    // Step 7: Pack 4-bit — two indices per byte
    // Even elements write low nibble, odd elements write high nibble.
    // Barrier ensures even threads write first.
    uint pack_offset = vec_idx * (d / 2) + elem / 2;
    if (elem % 2 == 0) {
        packed[pack_offset] = uchar(idx & 0x0F);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (elem % 2 == 1) {
        packed[pack_offset] |= uchar((idx & 0x0F) << 4);
    }
}
"#;
