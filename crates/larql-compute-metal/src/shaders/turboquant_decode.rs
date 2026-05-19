//! TurboQuant decode shader: fused centroid lookup + inverse WHT + rescale.
//!
//! Input:  packed 4-bit indices + norms
//! Output: fp32 vector buffer [batch × d]
//!
//! Algorithm per vector:
//!   1. Unpack 4-bit indices
//!   2. Centroid lookup → fp32 coordinates
//!   3. Apply sign flips (D)
//!   4. WHT butterfly passes (self-inverse)
//!   5. Apply sign flips (D)
//!   6. Rescale by stored norm
//!
//! One threadgroup per vector. Threadgroup size = d.
//!
//! ## Retention rationale (ADR-017)
//!
//! **Status**: experimental — paired with `turboquant_encode`. Same
//! retention story: built into `all_shaders()` for diag, not wired
//! into production. See `turboquant_encode.rs` for the full rationale
//! and removal trigger; the two ship together.

pub const SHADER: &str = r#"
// TurboQuant 4-bit decode: unpack → centroids → sign flip → WHT → sign flip → rescale.
// Uses tq4_centroids and tq_sign_flip from turboquant_encode.

kernel void turboquant_decode_4bit(
    device const float* norms    [[buffer(0)]],  // [batch] — stored norms
    device const uchar* packed   [[buffer(1)]],  // [batch, d/2] — packed 4-bit indices
    device float*       output   [[buffer(2)]],  // [batch, d] — decoded vectors
    constant uint&      d        [[buffer(3)]],
    constant uint&      batch    [[buffer(4)]],
    uint  elem    [[thread_position_in_threadgroup]],
    uint  vec_idx [[threadgroup_position_in_grid]],
    threadgroup float* shared [[threadgroup(0)]])
{
    if (vec_idx >= batch || elem >= d) return;

    // Step 1: Unpack 4-bit index
    uint pack_offset = vec_idx * (d / 2) + elem / 2;
    uchar byte_val = packed[pack_offset];
    uint idx = (elem % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);

    // Step 2: Centroid lookup
    shared[elem] = tq4_centroids[idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Sign flips (D)
    if (tq_sign_flip(elem)) shared[elem] = -shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: WHT (self-inverse with the D·H·D construction)
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

    shared[elem] *= 1.0f / sqrt(float(d));

    // Step 5: Sign flips again (D)
    if (tq_sign_flip(elem)) shared[elem] = -shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 6: Rescale by norm
    output[vec_idx * d + elem] = shared[elem] * norms[vec_idx];
}
"#;
