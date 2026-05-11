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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Cache key for the spec batched-seq scratch + graph maps.
/// `cuda-spec-branching-tree` T3.1 extends the original
/// `(seq_len, DecodeScratchShape)` tuple with an `is_tree`
/// discriminator so the linear-chain and tree-mask attention paths
/// don't share a captured graph (they call different kernels).
///
/// Different in-flight tree shapes with the same `seq_len` (e.g. PLD
/// returning a 4-node linear chain one iter and a 4-node tree the
/// next) are already distinguished by `is_tree`. Tree shapes that
/// differ in node count (depth=2 branches=2 → 7 nodes vs depth=3
/// branches=1 → 4 nodes) are distinguished by `seq_len`. We don't
/// need a finer tree_layout_hash because the kernel reads the
/// ancestor bitsets from the scratch buffer at replay time, so the
/// captured launch is correct for any bitset layout sharing the same
/// `(seq_len, shape)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpecScratchKey {
    pub seq_len: usize,
    pub shape: DecodeScratchShape,
    /// `false` = linear-chain causal kernel; `true` = tree-mask
    /// kernel with per-node ancestor bitsets.
    pub is_tree: bool,
}

impl SpecScratchKey {
    pub fn linear(seq_len: usize, shape: DecodeScratchShape) -> Self {
        Self {
            seq_len,
            shape,
            is_tree: false,
        }
    }

    pub fn tree(seq_len: usize, shape: DecodeScratchShape) -> Self {
        Self {
            seq_len,
            shape,
            is_tree: true,
        }
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

/// Phase A of `cuda-spec-cuda-graph`: pre-allocated scratch buffers
/// for the spec batched seq forward path (`decode_tokens_speculative_
/// seq_device`). Same role as [`DecodeScratch`] but every buffer is
/// sized for `seq_len > 1` rows. Reused across spec iters at the same
/// `(seq_len, shape)`; shape mismatch invalidates and re-allocates.
///
/// Per-call savings: ~714 device_alloc calls collapse into one-time
/// alloc at first use. Also gives the CUDA Graph capture (Phase C) a
/// stable pointer set across replays.
pub(crate) struct SpecDecodeScratch {
    pub shape: DecodeScratchShape,
    pub seq_len: usize,

    // Running residual; persistent across layers within one spec iter.
    pub h: CudaSlice<f32>,

    // Pre-attn pipeline (per-seq).
    pub h_attn: CudaSlice<f32>,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn_out: CudaSlice<f32>,
    pub attn_delta: CudaSlice<f32>,
    pub attn_normed: CudaSlice<f32>,

    // FFN pipeline (per-seq).
    pub h_ffn: CudaSlice<f32>,
    pub gate: CudaSlice<f32>,
    pub up: CudaSlice<f32>,
    pub act: CudaSlice<f32>,
    pub ffn_delta: CudaSlice<f32>,
    pub ffn_normed: CudaSlice<f32>,

    // f16 ping-pong buffers for cuBLAS hgemm (input + output).
    // Sized for the maximum projection across the layer pipeline
    // (typically `inter` for FFN, `q_dim` for attention output).
    pub x_f16_in: CudaSlice<half::f16>,
    pub gemm_out_f16: CudaSlice<half::f16>,

    // Device-side base_pos slot (Phase B will let the captured graph
    // re-read this between replays at different cache positions).
    pub base_pos: CudaSlice<i32>,

    // `cuda-spec-branching-tree` T3.1: per-tree-node ancestor bitsets.
    // Sized for `seq_len` u64 entries (= cap of 64 nodes — matches
    // `SpecConfig::tree_nodes()` cap). Unused on the linear-chain
    // path; written by the tree dispatch path before each replay so a
    // captured graph references a stable device pointer.
    pub ancestors: CudaSlice<u64>,
}

impl SpecDecodeScratch {
    /// Maximum projection out_dim we need scratch space for. Conservative:
    /// covers attention's q_dim and FFN's inter (whichever is larger).
    fn max_proj_out_dim(shape: DecodeScratchShape) -> usize {
        shape
            .q_dim
            .max(shape.kv_dim)
            .max(shape.inter)
            .max(shape.hidden)
    }

    pub fn allocate(
        drv: &Driver,
        shape: DecodeScratchShape,
        seq_len: usize,
    ) -> Result<Self, CudaInitError> {
        let n = seq_len;
        let max_out = Self::max_proj_out_dim(shape);
        Ok(SpecDecodeScratch {
            shape,
            seq_len,
            h: drv.device_alloc_uninit(n * shape.hidden)?,
            h_attn: drv.device_alloc_uninit(n * shape.hidden)?,
            q: drv.device_alloc_uninit(n * shape.q_dim)?,
            k: drv.device_alloc_uninit(n * shape.kv_dim)?,
            v: drv.device_alloc_uninit(n * shape.kv_dim)?,
            attn_out: drv.device_alloc_uninit(n * shape.q_dim)?,
            attn_delta: drv.device_alloc_uninit(n * shape.hidden)?,
            attn_normed: drv.device_alloc_uninit(n * shape.hidden)?,
            h_ffn: drv.device_alloc_uninit(n * shape.hidden)?,
            gate: drv.device_alloc_uninit(n * shape.inter)?,
            up: drv.device_alloc_uninit(n * shape.inter)?,
            act: drv.device_alloc_uninit(n * shape.inter)?,
            ffn_delta: drv.device_alloc_uninit(n * shape.hidden)?,
            ffn_normed: drv.device_alloc_uninit(n * shape.hidden)?,
            x_f16_in: unsafe {
                drv.stream
                    .alloc::<half::f16>(n * shape.hidden.max(shape.inter))
                    .map_err(|e| CudaInitError::DriverMissing(format!("alloc x_f16_in: {e:?}")))?
            },
            gemm_out_f16: unsafe {
                drv.stream.alloc::<half::f16>(n * max_out).map_err(|e| {
                    CudaInitError::DriverMissing(format!("alloc gemm_out_f16: {e:?}"))
                })?
            },
            base_pos: drv
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| CudaInitError::DriverMissing(format!("alloc base_pos: {e:?}")))?,
            ancestors: drv
                .stream
                .alloc_zeros::<u64>(n)
                .map_err(|e| CudaInitError::DriverMissing(format!("alloc ancestors: {e:?}")))?,
        })
    }
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
