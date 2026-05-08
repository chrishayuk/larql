"""
Triton kernels for IsoQuant: GPU-accelerated quaternion 4D block quantization.

Kernels:
  1. triton_iso_full_fused  — q_L v q̄_R → quantize → q̄_L v̂ q_R (full SO(4))
  2. triton_iso_fast_fused  — q_L v → quantize → q̄_L v̂ (single isoclinic)

These are the Triton equivalents of the CUDA kernels in isoclinic_fused_kernel.cu.
Portable across NVIDIA and AMD GPUs, auto-tuned block sizes.

Compared to RotorQuant Triton (Clifford Cl(3,0)):
  - 4 components per block vs 8 (half the loads/stores)
  - 16 FMAs per quat multiply vs ~28 per sparse geometric product
  - Clean 4D alignment (no tail handling for power-of-2 dims)
"""

import torch
import triton
import triton.language as tl


# ── Quaternion primitives ─────────────────────────────────────────────

@triton.jit
def _quat_mul(aw, ax, ay, az, bw, bx, by, bz):
    """Hamilton product of two quaternions. 16 FMAs."""
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return rw, rx, ry, rz


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


# ── IsoQuant-Full fused kernel ────────────────────────────────────────
# Forward:  T(v) = q_L * v * conj(q_R)
# Quantize: scalar Lloyd-Max on all 4 components
# Inverse:  T⁻¹(v̂) = conj(q_L) * v̂ * q_R

@triton.jit
def _iso_full_fused_kernel(
    input_ptr, output_ptr,
    ql_ptr, qr_ptr, centroids_ptr,
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

    # Load quaternion pair for each group
    ql_w = tl.load(ql_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    ql_x = tl.load(ql_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    ql_y = tl.load(ql_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    ql_z = tl.load(ql_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    qr_w = tl.load(qr_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    qr_x = tl.load(qr_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    qr_y = tl.load(qr_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    qr_z = tl.load(qr_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    # Load input 4D block
    d0 = g_offs * 4
    v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                  mask=g_mask & (d0 < emb_dim), other=0.0)
    v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                  mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)
    v2 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d,
                  mask=g_mask & ((d0 + 2) < emb_dim), other=0.0)
    v3 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 3) * stride_in_d,
                  mask=g_mask & ((d0 + 3) < emb_dim), other=0.0)

    # Forward: temp = q_L * v
    tw, tx, ty, tz = _quat_mul(ql_w, ql_x, ql_y, ql_z, v0, v1, v2, v3)
    # Forward: rotated = temp * conj(q_R)
    rw, rx, ry, rz = _quat_mul(tw, tx, ty, tz, qr_w, -qr_x, -qr_y, -qr_z)

    # Quantize all 4 components
    qw = _quantize_nearest(rw, centroids_ptr, n_levels)
    qx = _quantize_nearest(rx, centroids_ptr, n_levels)
    qy = _quantize_nearest(ry, centroids_ptr, n_levels)
    qz = _quantize_nearest(rz, centroids_ptr, n_levels)

    # Inverse: temp2 = conj(q_L) * v̂
    tw2, tx2, ty2, tz2 = _quat_mul(ql_w, -ql_x, -ql_y, -ql_z, qw, qx, qy, qz)
    # Inverse: restored = temp2 * q_R
    fw, fx, fy, fz = _quat_mul(tw2, tx2, ty2, tz2, qr_w, qr_x, qr_y, qr_z)

    # Store output
    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             fw, mask=g_mask & (d0 < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             fx, mask=g_mask & ((d0 + 1) < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
             fy, mask=g_mask & ((d0 + 2) < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 3) * stride_out_d,
             fz, mask=g_mask & ((d0 + 3) < emb_dim))


# ── IsoQuant-Fast fused kernel ────────────────────────────────────────
# Forward:  T(v) = q_L * v
# Quantize: scalar Lloyd-Max on all 4 components
# Inverse:  T⁻¹(v̂) = conj(q_L) * v̂

@triton.jit
def _iso_fast_fused_kernel(
    input_ptr, output_ptr,
    ql_ptr, centroids_ptr,
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

    # Load single quaternion per group
    ql_w = tl.load(ql_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    ql_x = tl.load(ql_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    ql_y = tl.load(ql_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    ql_z = tl.load(ql_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    # Load input 4D block
    d0 = g_offs * 4
    v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                  mask=g_mask & (d0 < emb_dim), other=0.0)
    v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                  mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)
    v2 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d,
                  mask=g_mask & ((d0 + 2) < emb_dim), other=0.0)
    v3 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 3) * stride_in_d,
                  mask=g_mask & ((d0 + 3) < emb_dim), other=0.0)

    # Forward: rotated = q_L * v (single multiply)
    rw, rx, ry, rz = _quat_mul(ql_w, ql_x, ql_y, ql_z, v0, v1, v2, v3)

    # Quantize all 4 components
    qw = _quantize_nearest(rw, centroids_ptr, n_levels)
    qx = _quantize_nearest(rx, centroids_ptr, n_levels)
    qy = _quantize_nearest(ry, centroids_ptr, n_levels)
    qz = _quantize_nearest(rz, centroids_ptr, n_levels)

    # Inverse: restored = conj(q_L) * v̂ (single multiply)
    fw, fx, fy, fz = _quat_mul(ql_w, -ql_x, -ql_y, -ql_z, qw, qx, qy, qz)

    # Store output
    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             fw, mask=g_mask & (d0 < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             fx, mask=g_mask & ((d0 + 1) < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
             fy, mask=g_mask & ((d0 + 2) < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 3) * stride_out_d,
             fz, mask=g_mask & ((d0 + 3) < emb_dim))


# ── Python wrappers ───────────────────────────────────────────────────

def triton_iso_full_fused(
    input: torch.Tensor,     # [batch, emb_dim]
    q_left: torch.Tensor,    # [n_groups, 4]
    q_right: torch.Tensor,   # [n_groups, 4]
    centroids: torch.Tensor, # [n_levels]
) -> torch.Tensor:
    """Fused IsoQuant-Full pipeline: normalize → q_L v q̄_R → quantize → q̄_L v̂ q_R → rescale.

    Single kernel launch for the full quantize-dequantize roundtrip.
    """
    batch_size, emb_dim = input.shape
    n_groups = q_left.shape[0]
    n_levels = centroids.shape[0]

    # Norm separation
    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()

    ql_f32 = q_left.float().contiguous()
    qr_f32 = q_right.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty_like(input_f32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _iso_full_fused_kernel[grid](
        input_f32, output, ql_f32, qr_f32, c_f32,
        batch_size, emb_dim, n_groups, n_levels,
        input_f32.stride(0), input_f32.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    output = output * norms
    return output.to(input.dtype)


def triton_iso_fast_fused(
    input: torch.Tensor,     # [batch, emb_dim]
    q_left: torch.Tensor,    # [n_groups, 4]
    centroids: torch.Tensor, # [n_levels]
) -> torch.Tensor:
    """Fused IsoQuant-Fast pipeline: normalize → q_L v → quantize → q̄_L v̂ → rescale.

    Single kernel launch. Fastest quantize-dequantize path.
    """
    batch_size, emb_dim = input.shape
    n_groups = q_left.shape[0]
    n_levels = centroids.shape[0]

    # Norm separation
    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()

    ql_f32 = q_left.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty_like(input_f32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _iso_fast_fused_kernel[grid](
        input_f32, output, ql_f32, c_f32,
        batch_size, emb_dim, n_groups, n_levels,
        input_f32.stride(0), input_f32.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    output = output * norms
    return output.to(input.dtype)
