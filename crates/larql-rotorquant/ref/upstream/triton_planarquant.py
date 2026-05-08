"""
Triton kernels for PlanarQuant: GPU-accelerated 2D Givens rotation quantization.

Kernels:
  1. triton_planar2_fused — rotate(cos θ, sin θ) → quantize → inverse rotate

This is the Triton equivalent of csrc/planar2_fused_kernel.cu.
The simplest kernel in the rotation family — only 4 FMAs per 2D pair.

Compared to other Triton kernels:
  - IsoQuant Triton: 4 loads/stores per group, 16 FMAs (quaternion multiply)
  - RotorQuant Triton: 8 loads/stores per group, ~28 FMAs (Clifford product)
  - PlanarQuant Triton: 2 loads/stores per group, 4 FMAs (Givens rotation)

Reference: ParaMind2025/isoquant (planar2_fused_kernel.cu)
"""

import torch
import triton
import triton.language as tl


# ── Quantize nearest centroid ────────────────────────────────────────

@triton.jit
def _quantize_nearest(val, centroids_ptr, n_levels: tl.constexpr):
    """Find nearest centroid for a scalar value."""
    best_val = tl.load(centroids_ptr)
    best_dist = tl.abs(val - best_val)
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        d = tl.abs(val - c)
        mask = d < best_dist
        best_dist = tl.where(mask, d, best_dist)
        best_val = tl.where(mask, c, best_val)
    return best_val


# ── PlanarQuant fused kernel ────────────────────────────────────────
# Forward:  v_rot = R(θ) v       [cos θ · v0 - sin θ · v1,  sin θ · v0 + cos θ · v1]
# Quantize: scalar Lloyd-Max on both components
# Inverse:  v̂ = R(-θ) v_q        [cos θ · q0 + sin θ · q1, -sin θ · q0 + cos θ · q1]

@triton.jit
def _planar2_fused_kernel(
    input_ptr, output_ptr,
    rot2_ptr,        # (n_groups, 2) as [cos θ, sin θ]
    centroids_ptr,   # (n_levels,)
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_in_b, stride_in_d,
    stride_out_b, stride_out_d,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    # Load rotation params (cos θ, sin θ) for each group
    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    # Load input 2D pair
    d0 = g_offs * 2
    v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                  mask=g_mask & (d0 < emb_dim), other=0.0)
    v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                  mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)

    # Forward rotation: 4 FMAs
    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1

    # Quantize both scalars
    q0 = _quantize_nearest(r0, centroids_ptr, n_levels)
    q1 = _quantize_nearest(r1, centroids_ptr, n_levels)

    # Inverse rotation (transpose = negate sin): 4 FMAs
    f0 = cos_t * q0 + sin_t * q1
    f1 = -sin_t * q0 + cos_t * q1

    # Store output
    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             f0, mask=g_mask & (d0 < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             f1, mask=g_mask & ((d0 + 1) < emb_dim))


# ── Quantize-only kernel (returns indices, no inverse rotation) ──────

@triton.jit
def _planar2_quantize_kernel(
    input_ptr, indices_ptr,
    rot2_ptr, centroids_ptr,
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_in_b, stride_in_d,
    stride_idx_b, stride_idx_d,
    BLOCK_G: tl.constexpr,
):
    """Quantize only — stores uint8 indices (for KV cache compression)."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    d0 = g_offs * 2
    v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                  mask=g_mask & (d0 < emb_dim), other=0.0)
    v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                  mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)

    # Forward rotation
    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1

    # Find nearest centroid index for r0
    best_idx0 = tl.zeros_like(r0).to(tl.int32)
    best_dist0 = tl.abs(r0 - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        dd = tl.abs(r0 - c)
        mask = dd < best_dist0
        best_dist0 = tl.where(mask, dd, best_dist0)
        best_idx0 = tl.where(mask, i, best_idx0)

    # Find nearest centroid index for r1
    best_idx1 = tl.zeros_like(r1).to(tl.int32)
    best_dist1 = tl.abs(r1 - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        dd = tl.abs(r1 - c)
        mask = dd < best_dist1
        best_dist1 = tl.where(mask, dd, best_dist1)
        best_idx1 = tl.where(mask, i, best_idx1)

    tl.store(indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
             best_idx0.to(tl.int8),
             mask=g_mask & (d0 < emb_dim))
    tl.store(indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
             best_idx1.to(tl.int8),
             mask=g_mask & ((d0 + 1) < emb_dim))


# ── Dequantize-only kernel (indices → reconstructed vectors) ─────────

@triton.jit
def _planar2_dequantize_kernel(
    indices_ptr, output_ptr,
    rot2_ptr, centroids_ptr,
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_idx_b, stride_idx_d,
    stride_out_b, stride_out_d,
    BLOCK_G: tl.constexpr,
):
    """Dequantize indices back to vectors via inverse rotation."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    d0 = g_offs * 2

    # Load indices and look up centroid values
    idx0 = tl.load(indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
                    mask=g_mask & (d0 < emb_dim), other=0).to(tl.int32)
    idx1 = tl.load(indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
                    mask=g_mask & ((d0 + 1) < emb_dim), other=0).to(tl.int32)

    q0 = tl.load(centroids_ptr + idx0, mask=g_mask, other=0.0)
    q1 = tl.load(centroids_ptr + idx1, mask=g_mask, other=0.0)

    # Inverse rotation
    f0 = cos_t * q0 + sin_t * q1
    f1 = -sin_t * q0 + cos_t * q1

    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             f0, mask=g_mask & (d0 < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             f1, mask=g_mask & ((d0 + 1) < emb_dim))


# ── Python wrappers ──────────────────────────────────────────────────

def triton_planar2_fused(
    input: torch.Tensor,       # [batch, emb_dim]
    rot2: torch.Tensor,        # [n_groups, 2] as [cos θ, sin θ]
    centroids: torch.Tensor,   # [n_levels]
) -> torch.Tensor:
    """Fused PlanarQuant pipeline: normalize → rotate → quantize → inverse rotate → rescale.

    Single kernel launch for the full quantize-dequantize roundtrip.
    The fastest rotation-based quantization kernel.
    """
    batch_size, emb_dim = input.shape
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    # Norm separation (done on GPU, outside kernel)
    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty_like(input_f32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _planar2_fused_kernel[grid](
        input_f32, output, rot2_f32, c_f32,
        batch_size, emb_dim, n_groups, n_levels,
        input_f32.stride(0), input_f32.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    output = output * norms
    return output.to(input.dtype)


def triton_planar2_quantize(
    input: torch.Tensor,       # [batch, emb_dim]
    rot2: torch.Tensor,        # [n_groups, 2]
    centroids: torch.Tensor,   # [n_levels]
) -> tuple:
    """Quantize only — returns (indices, norms) for KV cache storage."""
    batch_size, emb_dim = input.shape
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    indices = torch.empty(batch_size, emb_dim, dtype=torch.int8, device=input.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _planar2_quantize_kernel[grid](
        input_f32, indices, rot2_f32, c_f32,
        batch_size, emb_dim, n_groups, n_levels,
        input_f32.stride(0), input_f32.stride(1),
        indices.stride(0), indices.stride(1),
        BLOCK_G=BLOCK_G,
    )

    return indices, norms.squeeze(-1)


def triton_planar2_dequantize(
    indices: torch.Tensor,     # [batch, emb_dim] int8
    norms: torch.Tensor,       # [batch]
    rot2: torch.Tensor,        # [n_groups, 2]
    centroids: torch.Tensor,   # [n_levels]
) -> torch.Tensor:
    """Dequantize indices back to vectors via inverse rotation + rescale."""
    batch_size, emb_dim = indices.shape
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty(batch_size, emb_dim, dtype=torch.float32, device=indices.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _planar2_dequantize_kernel[grid](
        indices, output, rot2_f32, c_f32,
        batch_size, emb_dim, n_groups, n_levels,
        indices.stride(0), indices.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    if norms.dim() == 1:
        norms = norms.unsqueeze(-1)
    return output * norms
