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

    /// Same shape as [`Self::f16_gemv`] but the weight matrix holds
    /// **bf16** codes, packed little-endian, `n * k * 2` bytes long.
    ///
    /// bf16 is not f16: it is the top 16 bits of the f32 it denotes, so
    /// widening is exact and a backend must decode it as such. The two
    /// share a width and nothing else, which is why this is a separate
    /// method rather than a format flag on the f16 one — a caller that
    /// picked the wrong one would get unrelated numbers, not a rounding
    /// difference. Backends without a specialised kernel return `None`.
    fn bf16_gemv(&self, _w_bf16: &[u8], _x: &[f32], _n: usize, _k: usize) -> Option<Vec<f32>> {
        None
    }

    /// Like [`Self::bf16_gemv`] but skips the internal flop threshold.
    fn bf16_gemv_force(&self, w_bf16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        self.bf16_gemv(w_bf16, x, n, k)
    }

    /// Several bf16 matrices applied to **one** input vector, as one
    /// device submission where the backend supports it — the bf16
    /// sibling of [`Self::f16_gemv_multi`], and for the same reason: a
    /// gated FFN's gate and up projections read the same activation, so
    /// submitting them together amortises the synchronisation that
    /// dominates a gemv-per-matmul decode.
    ///
    /// The default is the sequential force gemvs — bit-identical
    /// results, no batching — so a backend only overrides this for the
    /// submission win, never for different arithmetic.
    fn bf16_gemv_multi(
        &self,
        weights: &[(&[u8], usize, usize)],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        weights
            .iter()
            .map(|&(w, n, k)| self.bf16_gemv_force(w, x, n, k))
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A backend that supplies only the two required matmuls, so every
    /// default body of the trait runs as written. Trait defaults are only
    /// exercised by an implementor that declines to override them, and
    /// every real backend overrides what it supports — the contract they
    /// encode ("no kernel ⇒ `None`, never a wrong answer") is what a *new*
    /// backend relies on.
    struct NaiveBackend;

    impl MatMul for NaiveBackend {
        fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
            a.dot(&b)
        }

        fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
            a.dot(&b.t())
        }
    }

    /// A backend with the single-matrix gemvs but no batched submission:
    /// the `_multi` defaults must be the sequential calls, and the result
    /// arrives per matrix in call order.
    struct SingleGemvBackend;

    /// Marker value each fake gemv writes so the caller can tell which
    /// arm produced a row.
    const F16_MARK: f32 = 16.0;
    const MXFP4_MARK: f32 = 4.0;
    const NVFP4_MARK: f32 = 44.0;

    impl MatMul for SingleGemvBackend {
        fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
            a.dot(&b)
        }

        fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
            a.dot(&b.t())
        }

        fn f16_gemv(&self, _w: &[u8], _x: &[f32], n: usize, _k: usize) -> Option<Vec<f32>> {
            Some(vec![F16_MARK; n])
        }

        fn mxfp4_gemv(
            &self,
            _packed: &[u8],
            _scales: &[u8],
            _x: &[f32],
            n: usize,
            _k: usize,
        ) -> Option<Vec<f32>> {
            Some(vec![MXFP4_MARK; n])
        }

        fn nvfp4_gemv(
            &self,
            _packed: &[u8],
            _scales: &[u8],
            tensor_scale: f32,
            _x: &[f32],
            n: usize,
            _k: usize,
        ) -> Option<Vec<f32>> {
            Some(vec![NVFP4_MARK * tensor_scale; n])
        }
    }

    const K: usize = 4;
    const N: usize = 3;

    /// The batched default dispatches each op to the matmul its
    /// `transpose_b` flag names, in order.
    #[test]
    fn batch_default_dispatches_each_op_by_its_transpose_flag() {
        let b = NaiveBackend;
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let m = array![[5.0f32, 6.0], [7.0, 8.0]];
        let out = b.matmul_batch(&[
            MatMulOp {
                a: a.clone(),
                b: m.clone(),
                transpose_b: false,
            },
            MatMulOp {
                a: a.clone(),
                b: m.clone(),
                transpose_b: true,
            },
        ]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], b.matmul(a.view(), m.view()));
        assert_eq!(out[1], b.matmul_transb(a.view(), m.view()));
        assert_ne!(out[0], out[1], "the flag must select a different product");
    }

    /// Every specialised gemv defaults to "no kernel here" — `None`, so a
    /// caller falls back rather than trusting a fabricated vector.
    #[test]
    fn gemv_defaults_refuse_rather_than_guess() {
        let b = NaiveBackend;
        let w = Array2::<f32>::zeros((N, K));
        let x = vec![0.5f32; K];
        let bytes = vec![0u8; N * K * 2];

        assert!(b.f32_gemv(w.view(), &x).is_none());
        assert!(b.f32_gemv_force(w.view(), &x).is_none());
        assert!(b.f32_gemv_topk1(w.view(), &x).is_none());
        assert!(b.f16_gemv(&bytes, &x, N, K).is_none());
        assert!(b.f16_gemv_force(&bytes, &x, N, K).is_none());
        assert!(b.f16_gemv_topk1(&bytes, &x, N, K).is_none());
        assert!(b.f16_gemv_topk(&bytes, &x, N, K, 2).is_none());
        assert!(b.mxfp4_gemv(&bytes, &bytes, &x, N, K).is_none());
        assert!(b.nvfp4_gemv(&bytes, &bytes, 1.0, &x, N, K).is_none());
        // A residency hint on a backend with no notion of residency is a
        // no-op, not an error.
        b.wire_resident(&[&bytes]);
    }

    /// With no single-matrix kernel the multi defaults are `None` as a
    /// whole — one unsupported matrix refuses the submission, so a caller
    /// cannot receive a partially fabricated batch.
    #[test]
    fn multi_defaults_refuse_when_the_single_gemv_refuses() {
        let b = NaiveBackend;
        let x = vec![0.5f32; K];
        let bytes = vec![0u8; N * K * 2];
        assert!(b.f16_gemv_multi(&[(&bytes, N, K)], &x).is_none());
        assert!(b.mxfp4_gemv_multi(&[(&bytes, &bytes, N, K)], &x).is_none());
        assert!(b
            .nvfp4_gemv_multi(&[(&bytes, &bytes, 1.0, N, K)], &x)
            .is_none());
    }

    /// With single-matrix kernels present, the multi defaults are exactly
    /// the sequential calls: one output per matrix, in order, each the
    /// row its own kernel produced (the NVFP4 tensor scale reaches the
    /// per-matrix call).
    /// **The bf16 trait defaults answer, and answer conservatively.**
    ///
    /// A backend with no bf16 kernel must say `None` rather than decode
    /// the codes some other way: bf16 is the top 16 bits of an f32 and
    /// f16 is not, so a caller handed the wrong decoder gets unrelated
    /// numbers rather than a rounding difference. `bf16_gemv_force`
    /// delegates to `bf16_gemv`, and `bf16_gemv_multi` is the
    /// sequential force gemvs — so a backend overrides the multi only
    /// for the submission win, never for different arithmetic.
    #[test]
    fn the_bf16_defaults_decline_rather_than_guess() {
        let b = SingleGemvBackend;
        let bytes = vec![0u8; N * K * 2];
        let x = vec![1.0f32; K];
        assert_eq!(
            b.bf16_gemv(&bytes, &x, N, K),
            None,
            "no bf16 kernel means None, never an f16 decode of the same bytes"
        );
        assert_eq!(
            b.bf16_gemv_force(&bytes, &x, N, K),
            None,
            "force delegates to the same answer"
        );
        assert_eq!(
            b.bf16_gemv_multi(&[(&bytes[..], N, K), (&bytes[..], N, K)], &x),
            None,
            "a multi over declining gemvs declines as a whole, rather than \
             returning a shorter list a caller would index blindly"
        );
    }

    /// A backend that DOES implement `bf16_gemv` gets the multi default
    /// for free, in argument order — the property that makes overriding
    /// it a pure submission optimisation.
    #[test]
    fn the_bf16_multi_default_is_the_sequential_gemvs_in_order() {
        const BF16_MARK: f32 = 116.0;
        struct Bf16Backend;
        impl MatMul for Bf16Backend {
            fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
                a.dot(&b)
            }
            fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
                a.dot(&b.t())
            }
            fn bf16_gemv(&self, _w: &[u8], _x: &[f32], n: usize, _k: usize) -> Option<Vec<f32>> {
                Some(vec![BF16_MARK; n])
            }
        }
        let b = Bf16Backend;
        let bytes = vec![0u8; N * K * 2];
        let x = vec![1.0f32; K];
        let second_n = N + 3;
        let got = b
            .bf16_gemv_multi(&[(&bytes[..], N, K), (&bytes[..], second_n, K)], &x)
            .expect("a backend with a kernel answers");
        assert_eq!(
            got,
            vec![vec![BF16_MARK; N], vec![BF16_MARK; second_n]],
            "results follow the ARGUMENT order and each keeps its own width"
        );
    }

    #[test]
    fn multi_defaults_are_the_sequential_single_gemvs_in_order() {
        let b = SingleGemvBackend;
        let x = vec![0.5f32; K];
        let bytes = vec![0u8; N * K * 2];
        let second_n = N + 1;

        let f16 = b
            .f16_gemv_multi(&[(&bytes, N, K), (&bytes, second_n, K)], &x)
            .unwrap();
        assert_eq!(f16, vec![vec![F16_MARK; N], vec![F16_MARK; second_n]]);

        let mx = b
            .mxfp4_gemv_multi(&[(&bytes, &bytes, N, K), (&bytes, &bytes, second_n, K)], &x)
            .unwrap();
        assert_eq!(mx, vec![vec![MXFP4_MARK; N], vec![MXFP4_MARK; second_n]]);

        let scale_a = 2.0f32;
        let scale_b = 0.5f32;
        let nv = b
            .nvfp4_gemv_multi(
                &[
                    (&bytes, &bytes, scale_a, N, K),
                    (&bytes, &bytes, scale_b, second_n, K),
                ],
                &x,
            )
            .unwrap();
        assert_eq!(
            nv,
            vec![
                vec![NVFP4_MARK * scale_a; N],
                vec![NVFP4_MARK * scale_b; second_n]
            ]
        );
        // And the `_force` variants route to the same single kernels.
        assert_eq!(b.f16_gemv_force(&bytes, &x, N, K), Some(vec![F16_MARK; N]));
    }
}
