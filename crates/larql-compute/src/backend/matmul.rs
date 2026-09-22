//! `MatMul` — f32 / f16 matmul + gemv operations.
//!
//! Covers the dense linear-algebra surface: square matmul, transposed
//! matmul, batched matmul, and the specialised single-row gemvs the
//! lm-head uses in autoregressive decode (where `M = 1` makes the
//! 32×32 tiled sgemm waste 31/32 threads).

use ndarray::{Array2, ArrayView2};

/// A single matmul operation for batch dispatch.
pub struct MatMulOp {
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub transpose_b: bool,
}

/// One NVFP4 matrix as a batched call consumes it: packed e2m1 codes,
/// E4M3 group scales, the matrix's f32 tensor scale, and `(n, k)`.
///
/// Named because the tensor scale is not foldable into the scale stream —
/// E4M3 cannot represent the product — so the tuple genuinely carries
/// five things and a reader needs to know which is which.
pub type Nvfp4Operand<'a> = (&'a [u8], &'a [u8], f32, usize, usize);

/// Dense linear-algebra primitives that don't depend on quantisation.
pub trait MatMul {
    /// C = A × B where A is [m, k] and B is [k, n].
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;

    /// C = A × B^T where A is [m, k] and B is [n, k].
    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;

    /// Multiple matmuls in one submission. Default: serial dispatch.
    /// GPU backends can override with parallel command buffer encoding.
    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter()
            .map(|op| {
                if op.transpose_b {
                    self.matmul_transb(op.a.view(), op.b.view())
                } else {
                    self.matmul(op.a.view(), op.b.view())
                }
            })
            .collect()
    }

    /// Dedicated row-per-simdgroup gemv for single-row × large-N × large-K.
    /// Computes `out[N] = W[N, K] · x[K]`. Backends that lack a specialised
    /// kernel should return `None`; callers fall back to `matmul_transb`.
    ///
    /// Motivating use-case: LM-head logits in autoregressive decode where
    /// the 32×32 tiled sgemm wastes 31/32 threads at `M = 1`.
    fn f32_gemv(&self, _w: ArrayView2<f32>, _x: &[f32]) -> Option<Vec<f32>> {
        None
    }

    /// GPU gemv + GPU argmax without materialising the full output Vec.
    /// Returns `(token_id, score)` for the top-1 element.
    /// Saves ~0.33ms on Metal by reading back only 8 KB partial results
    /// instead of 1 MB (262K × f32). Returns `None` if not specialised.
    fn f32_gemv_topk1(&self, _w: ArrayView2<f32>, _x: &[f32]) -> Option<(u32, f32)> {
        None
    }

    /// f16 gemv + GPU argmax. Used by the lm_head greedy-decode path on
    /// tied-embed models (Gemma 3/4) where the f16 mmap'd embeddings are
    /// the lm_head matrix and the bench / production both pick top-1.
    /// Returns `None` if not specialised.
    fn f16_gemv_topk1(
        &self,
        _w_f16: &[u8],
        _x: &[f32],
        _n: usize,
        _k: usize,
    ) -> Option<(u32, f32)> {
        None
    }

    /// f16 gemv + GPU partial top-K. Generalises [`Self::f16_gemv_topk1`]
    /// to `top_k > 1` (capped at the kernel's `K_TOPK` constant). Returns
    /// `None` when not specialised or `top_k` exceeds the per-TG capacity.
    fn f16_gemv_topk(
        &self,
        _w_f16: &[u8],
        _x: &[f32],
        _n: usize,
        _k: usize,
        _top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        None
    }

    /// Like [`Self::f32_gemv`] but skips the internal CPU-vs-GPU flop
    /// threshold. Use when the caller has already decided the work is
    /// worth a GPU dispatch — e.g. the per-layer gate matmul that fires
    /// once per feature-set per token and accumulates across 34–60 layers.
    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        self.f32_gemv(w, x)
    }

    /// Same shape as [`Self::f32_gemv`] but the weight matrix is f16
    /// packed as little-endian IEEE-half bytes, `n * k * 2` long. Lets
    /// the LM head run directly on the mmap'd f16 embeddings without a
    /// 2× f32 clone. Backends without a specialised kernel return
    /// `None`.
    fn f16_gemv(&self, _w_f16: &[u8], _x: &[f32], _n: usize, _k: usize) -> Option<Vec<f32>> {
        None
    }

    /// Like [`Self::f16_gemv`] but skips the internal flop threshold.
    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        self.f16_gemv(w_f16, x, n, k)
    }

    /// Exact-schema diagnostic for dense FFNs: split the columns of a
    /// row-major f16 `[hidden, intermediate]` down projection into fixed
    /// channel blocks and return the squared L2 norm of each block's raw
    /// output vector.
    ///
    /// For block `b`, this computes
    /// `sum_o (sum_c W[o,c] * inner[c])^2` over that block's channel range.
    /// It is an observer primitive, not an execution shortcut: callers still
    /// run the complete FFN normally. Backends without a specialised kernel
    /// return `None` and diagnostic callers must either use their scalar
    /// authority or fail loudly.
    fn f16_ffn_block_contributions(
        &self,
        _down_f16: &[u8],
        _inner: &[f32],
        _hidden: usize,
        _intermediate: usize,
        _block_channels: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Several f16 matrices applied to **one** input vector, as one
    /// device submission where the backend supports it.
    ///
    /// `weights` holds `(w_f16, n, k)` per matrix — every `k` must equal
    /// `x.len()`. A decode step is full of this shape (Q/K/V and an
    /// attention gate all read the attention input; FFN up and gate read
    /// the FFN input), and submitting them together amortises the
    /// per-submission synchronisation and the input upload that dominate
    /// a serialised gemv-per-matmul decode.
    ///
    /// The default is the sequential force gemvs — bit-identical results,
    /// no batching — so a backend only overrides this for the submission
    /// win, never for different arithmetic.
    fn f16_gemv_multi(
        &self,
        weights: &[(&[u8], usize, usize)],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        weights
            .iter()
            .map(|&(w, n, k)| self.f16_gemv_force(w, x, n, k))
            .collect()
    }

    /// Residency hint: these byte regions will be read repeatedly; make
    /// them device-resident now if the backend can.
    ///
    /// Purely an execution-state action — it computes nothing and must
    /// change no number. Motivation: a driver's wired-page collector
    /// un-wires buffers that sit idle between submissions, and a decode
    /// loop that walks tens of GB per token then pays a re-wire on
    /// every touch (measured 10× on a 60 GB f16 working set). One
    /// command buffer referencing everything re-wires it all at memcpy
    /// speed, and steps fast enough to stay under the collector's idle
    /// threshold keep themselves wired thereafter.
    fn wire_resident(&self, _buffers: &[&[u8]]) {}

    /// MXFP4 gemv: `out[N] = W[N, K] · x[K]` consuming the packed
    /// nibble stream and the e8m0 scale stream directly (the two live
    /// in separate buffers — see `mxfp4_matvec`'s layout doc: per row,
    /// `K/32` groups of 16 packed bytes lo-nibble-first plus one scale
    /// byte each). `None` when the backend has no MXFP4 kernel — the
    /// established loud-missing-capability answer.
    fn mxfp4_gemv(
        &self,
        _packed: &[u8],
        _scales: &[u8],
        _x: &[f32],
        _n: usize,
        _k: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Several MXFP4 matrices against one input vector, one submission
    /// where the backend supports it — the same shape and rationale as
    /// [`Self::f16_gemv_multi`]. `weights` holds
    /// `(packed, scales, n, k)` per matrix. Default: sequential
    /// [`Self::mxfp4_gemv`] calls, bit-identical results.
    fn mxfp4_gemv_multi(
        &self,
        weights: &[(&[u8], &[u8], usize, usize)],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        weights
            .iter()
            .map(|&(packed, scales, n, k)| self.mxfp4_gemv(packed, scales, x, n, k))
            .collect()
    }

    /// NVFP4 gemv: `out[N] = W[N, K] · x[K]` from the packed nibble
    /// stream, the **E4M3** group-scale stream (`K/16` groups of 8
    /// packed bytes lo-nibble-first plus one scale byte each), and the
    /// single `tensor_scale` both scale levels are expressed relative
    /// to.
    ///
    /// The extra scalar is the whole difference from [`Self::mxfp4_gemv`]
    /// at this seam, and it is not foldable into the scale stream: E4M3
    /// cannot represent the product, which is exactly why the format
    /// carries two levels. `None` when the backend has no NVFP4 kernel.
    fn nvfp4_gemv(
        &self,
        _packed: &[u8],
        _scales: &[u8],
        _tensor_scale: f32,
        _x: &[f32],
        _n: usize,
        _k: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Several NVFP4 matrices against one input vector, one submission
    /// where the backend supports it. `weights` holds
    /// `(packed, scales, tensor_scale, n, k)` per matrix. Default:
    /// sequential [`Self::nvfp4_gemv`] calls, bit-identical results.
    fn nvfp4_gemv_multi(&self, weights: &[Nvfp4Operand<'_>], x: &[f32]) -> Option<Vec<Vec<f32>>> {
        weights
            .iter()
            .map(|&(packed, scales, tensor_scale, n, k)| {
                self.nvfp4_gemv(packed, scales, tensor_scale, x, n, k)
            })
            .collect()
    }
}
