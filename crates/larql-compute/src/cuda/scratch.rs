//! `cuda-decode-cuda-graph` Phase 1: pre-allocated per-decode scratch
//! buffers. CUDA Graphs require stable device pointers across replays
//! — every per-call `device_alloc_uninit` would create a fresh
//! `cuMemAllocAsync` node inside the captured graph, which is both
//! wasteful (alloc+free per replay) and changes the captured kernel
//! arguments. By moving every intermediate buffer into a long-lived
//! `DecodeScratch` we keep the kernel arg pointers stable, so the
//! captured graph just re-runs the kernels on the same memory.
//!
//! The struct is keyed by `(hidden, q_dim, kv_dim, inter, head_dim)`
//! plus the `max_seq` of the KV cache. A shape mismatch invalidates
//! the scratch and forces a re-allocation (and re-capture).

use cudarc::driver::CudaSlice;

use super::driver::Driver;
use super::elem::Q8_1Buf;
use super::error::CudaInitError;

/// Shape that uniquely identifies the buffer sizing for a decode
/// pipeline. Two scratch buffers with equal `Shape` are bit-for-bit
/// reusable — captured graphs are also keyed by this shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodeScratchShape {
    pub hidden: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub inter: usize,
    pub head_dim: usize,
}

impl DecodeScratchShape {
    pub fn matches(&self, other: &Self) -> bool {
        self == other
    }
}

/// Pre-allocated per-decode buffers. Owned by `CudaBackend` and reused
/// across every `decode_token_device` call with the same shape. The
/// CUDA Graph capture path writes its kernel outputs into these
/// buffers, so the captured graph references stable device pointers
/// across replays.
pub(crate) struct DecodeScratch {
    pub shape: DecodeScratchShape,

    // Running residual; persistent across layers within one decode.
    pub h: CudaSlice<f32>,

    // Pre-attn pipeline.
    pub h_attn: CudaSlice<f32>,
    pub h_attn_q8_1: Q8_1Buf,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn_out: CudaSlice<f32>,
    pub attn_out_q8_1: Q8_1Buf,
    pub attn_delta: CudaSlice<f32>,
    pub attn_normed: CudaSlice<f32>,

    // FFN pipeline.
    pub h_ffn: CudaSlice<f32>,
    pub h_ffn_q8_1: Q8_1Buf,
    pub gate: CudaSlice<f32>,
    pub up: CudaSlice<f32>,
    pub act: CudaSlice<f32>,
    pub act_q8_1: Q8_1Buf,
    pub ffn_delta: CudaSlice<f32>,
    pub ffn_normed: CudaSlice<f32>,

    // Device-side pos (`int* pos_dev`). The captured graph reads its
    // attention kernel's `pos` from this buffer; the host writes the
    // current `pos` into it before each replay (one i32 → 4 B htod).
    pub pos: CudaSlice<i32>,
}

impl DecodeScratch {
    /// Allocate every per-decode buffer at the given shape. Cheap
    /// once the alloc pool is warm — all sizes fit in a few MB.
    pub fn allocate(drv: &Driver, shape: DecodeScratchShape) -> Result<Self, CudaInitError> {
        let alloc_q8_1 = |n: usize| -> Result<Q8_1Buf, CudaInitError> {
            debug_assert!(n.is_multiple_of(32));
            let n_blocks = n / 32;
            let bytes = drv
                .stream
                .alloc_zeros::<u8>(n_blocks * 36)
                .map_err(|e| CudaInitError::DriverMissing(format!("alloc q8_1 scratch: {e:?}")))?;
            Ok(Q8_1Buf { bytes, n_blocks })
        };

        Ok(DecodeScratch {
            shape,
            h: drv.device_alloc_uninit(shape.hidden)?,
            h_attn: drv.device_alloc_uninit(shape.hidden)?,
            h_attn_q8_1: alloc_q8_1(shape.hidden)?,
            q: drv.device_alloc_uninit(shape.q_dim)?,
            k: drv.device_alloc_uninit(shape.kv_dim)?,
            v: drv.device_alloc_uninit(shape.kv_dim)?,
            attn_out: drv.device_alloc_uninit(shape.q_dim)?,
            attn_out_q8_1: alloc_q8_1(shape.q_dim)?,
            attn_delta: drv.device_alloc_uninit(shape.hidden)?,
            attn_normed: drv.device_alloc_uninit(shape.hidden)?,
            h_ffn: drv.device_alloc_uninit(shape.hidden)?,
            h_ffn_q8_1: alloc_q8_1(shape.hidden)?,
            gate: drv.device_alloc_uninit(shape.inter)?,
            up: drv.device_alloc_uninit(shape.inter)?,
            act: drv.device_alloc_uninit(shape.inter)?,
            act_q8_1: alloc_q8_1(shape.inter)?,
            ffn_delta: drv.device_alloc_uninit(shape.hidden)?,
            ffn_normed: drv.device_alloc_uninit(shape.hidden)?,
            pos: drv
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| CudaInitError::DriverMissing(format!("alloc pos_dev: {e:?}")))?,
        })
    }
}
