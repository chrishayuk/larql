//! LM-head KNN dispatch — Q4_K, f16, and f32 backend paths plus the
//! shared `top_k_sorted` reduce.
//!
//! `lm_head_knn_backend` picks the cheapest available format; the
//! `_skip_q4k` variant exists for backends whose Q4_K matvec has
//! reduction-tree drift on close-call tokens. Both paths share
//! `top_k_sorted` for the K-largest extraction so a future tweak (e.g.
//! widening the argmax fast path) lands in one place.

use crate::index::core::VectorIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stride32Mode {
    Disabled,
    Fallback,
    First,
}

fn lm_head_stride32_mode() -> Stride32Mode {
    match std::env::var("LARQL_LM_HEAD_STRIDE32") {
        Ok(v) if matches!(v.as_str(), "1" | "true" | "on" | "yes") => Stride32Mode::First,
        Ok(v) if matches!(v.as_str(), "0" | "false" | "off" | "no") => Stride32Mode::Disabled,
        _ => Stride32Mode::Fallback,
    }
}

impl VectorIndex {
    /// KNN against lm_head via a ComputeBackend. Tries paths in order:
    ///   1. Q4 matvec on `lm_head_q4.bin` (when present and backend has q4).
    ///   2. f16 gemv on the mmap'd embeddings (tied-embed models only).
    ///   3. f32 BLAS fallback via `lm_head_knn`.
    ///
    /// `top_k == 1` uses the GPU-argmax fast paths on backends that
    /// implement them, returning a single `(token_id, score)` without
    /// the 1MB scores readback + 262K-element CPU sort that the general
    /// path requires. Bench (greedy decode) takes this path.
    pub fn lm_head_knn_backend(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(u32, f32)> {
        // 1. Q4_K path — ~1 ms on Metal (mmap file or synthesized from f16 embeddings).
        //
        // The on-disk `lm_head_q4.bin` is written by `format/weights/write_kquant`
        // as **Q4_K** (144 bytes per 256 elements with sub-block scales/mins).
        // Earlier code dispatched `q4_matvec` (which is Q4_0 — 18 bytes per 32
        // elements with one f16 scale): the byte-rate happens to match
        // (0.5625 B/element) so file size was identical, but the kernel read
        // Q4_K bytes as Q4_0 scales/quants and silently produced garbage
        // logits. Symptom: multilingual gibberish under `--metal` on any
        // vindex with a fresh `lm_head_q4.bin` (e.g. gemma3-4b-v2 extracted
        // 2026-04-27). Routing through `q4k_matvec` (which takes raw f32 x,
        // no Q8 step) restores the format match.
        if backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
            let q4_bytes: Option<&[u8]> = self.storage.lm_head_kquant_view().map(|b| b.as_ref());
            if let Some(q4_data) = q4_bytes {
                let vocab = self.vocab_size;
                let hidden = self.hidden_size;
                if vocab > 0 {
                    if let Some(x) = query.as_slice() {
                        if let Some(scores_vec) = backend.q4k_matvec(q4_data, x, vocab, hidden) {
                            return Self::top_k_sorted(scores_vec, top_k);
                        }
                    }
                }
            }
        }
        // 2. f16 path — tied-embed Gemma, ~2× the bandwidth of Q4 but still
        //    half of f32 and avoids a 5.6 GB heap allocation on 31B.
        if let Some(hits) = self.lm_head_f16_backend_hits(query, top_k, backend) {
            return hits;
        }
        // 3. f32 BLAS fallback.
        self.lm_head_knn(query, top_k)
    }

    /// Diagnostic alternative to `lm_head_knn_backend` — skips the
    /// production `q4k_matvec` path and tries stable-reduction
    /// alternatives in this order:
    ///
    ///   1. **Stride-32 Q4_K matvec** (`backend.q4k_matvec_stride32`) on
    ///      the same Q4_K bytes — same bandwidth as production
    ///      `q4k_matvec` (~327 MB/token), but with `f16_gemv`'s
    ///      reduction tree. ~2.95 ms/tok lm_head on Gemma 3 4B v2.
    ///   2. f16 GEMV on `embeddings.bin` mmap (tied-embed only).
    ///      Fallback when Q4_K bytes aren't populated. ~3.88 ms/tok.
    ///   3. f32 BLAS fallback (`lm_head_knn`).
    ///
    /// **History:** before 2026-05-02 this was the production default,
    /// because `lm_head_knn_backend` (which calls `q4k_matvec`) was
    /// producing argmax drift on close-call tokens. Root cause turned
    /// out to be a dispatch geometry mismatch in `MetalBackend::q4k_matvec`,
    /// not a kernel-level reduction-tree drift. With the dispatch fix,
    /// `q4k_matvec` is correct AND ~1.10 ms/tok faster than stride-32,
    /// so the canonical chain is now the default and this path is
    /// reachable via `LARQL_LM_HEAD_SKIP_Q4K=1` as a diagnostic A/B.
    /// See `PERFORMANCE.md` "Decision: lm_head dispatch order" for
    /// the full root-cause write-up.
    ///
    /// Env-var overrides (within this fallback chain):
    ///   - `LARQL_LM_HEAD_STRIDE32=0` — disable stride-32 entirely; go
    ///     straight to f16 (then f32). Used to A/B the stride-32 win.
    ///
    /// `lm_head_topk` in `larql-inference::layer_graph::generate::lm_head`
    /// routes here only when `LARQL_LM_HEAD_SKIP_Q4K=1` is set on a
    /// non-CPU backend; the canonical path is `lm_head_knn_backend`.
    pub fn lm_head_knn_backend_skip_q4k(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(u32, f32)> {
        let stride32_mode = lm_head_stride32_mode();

        // 1. Default stable path: stride-32 Q4_K matvec — Q4_K bandwidth
        //    win + f16_gemv's stable reduction tree. Skipped when
        //    `LARQL_LM_HEAD_STRIDE32=0`.
        if stride32_mode != Stride32Mode::Disabled {
            if let Some(hits) = self.lm_head_stride32_backend_hits(query, top_k, backend) {
                return hits;
            }
        }

        // 2. f16 GEMV fallback for vindexes lacking Q4_K lm_head bytes.
        if let Some(hits) = self.lm_head_f16_backend_hits(query, top_k, backend) {
            return hits;
        }

        // 3. f32 BLAS last resort.
        self.lm_head_knn(query, top_k)
    }

    fn lm_head_f16_backend_hits(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(u32, f32)>> {
        if let Some(f16_view) = self.storage.lm_head_f16_view() {
            let f16_mmap: &[u8] = f16_view.as_ref();
            let vocab = self.vocab_size;
            let hidden = self.hidden_size;
            if vocab > 0 {
                let expected = vocab * hidden * 2;
                if f16_mmap.len() >= expected {
                    if let Some(x) = query.as_slice() {
                        if top_k == 1 {
                            if let Some((idx, score)) =
                                backend.f16_gemv_topk1(&f16_mmap[..expected], x, vocab, hidden)
                            {
                                return Some(vec![(idx, score)]);
                            }
                        } else if let Some(hits) =
                            backend.f16_gemv_topk(&f16_mmap[..expected], x, vocab, hidden, top_k)
                        {
                            if !hits.is_empty() {
                                return Some(hits);
                            }
                        }
                        if let Some(scores_vec) =
                            backend.f16_gemv(&f16_mmap[..expected], x, vocab, hidden)
                        {
                            return Some(Self::top_k_sorted(scores_vec, top_k));
                        }
                    }
                }
            }
        }
        None
    }

    fn lm_head_stride32_backend_hits(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(u32, f32)>> {
        if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
            return None;
        }
        let q4_data: &[u8] = self.storage.lm_head_kquant_view().map(|b| b.as_ref())?;
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        if vocab == 0 {
            return None;
        }
        let x = query.as_slice()?;
        backend
            .q4k_matvec_stride32(q4_data, x, vocab, hidden)
            .map(|scores_vec| Self::top_k_sorted(scores_vec, top_k))
    }

    /// Sort `scores` by descending value and keep the top `top_k`. Shared
    /// by the Q4 / f16 / f32 paths above.
    ///
    /// Uses a size-K min-heap instead of `select_nth_unstable_by` so we
    /// don't materialise a 2MB `Vec<(u32, f32)>` for a 262K-vocab lm_head
    /// only to throw away 262K-K of it. For typical K=1..5 on Gemma 3 4B
    /// this drops the CPU portion of lm_head from ~0.5ms to ~50µs.
    ///
    /// Visibility note: `pub(super)` so the `mod tests` in `lm_head/mod.rs`
    /// can keep its existing `VectorIndex::top_k_sorted(...)` call sites
    /// after the M9 file split.
    pub(super) fn top_k_sorted(scores: Vec<f32>, top_k: usize) -> Vec<(u32, f32)> {
        if scores.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let k = top_k.min(scores.len());

        // Argmax fast path — no heap, single linear scan.
        if k == 1 {
            let mut best_i: u32 = 0;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &s) in scores.iter().enumerate() {
                if s.is_finite() && s > best_v {
                    best_v = s;
                    best_i = i as u32;
                }
            }
            if best_v == f32::NEG_INFINITY {
                return Vec::new();
            }
            return vec![(best_i, best_v)];
        }

        // Min-heap of size K, smallest score at index 0. We push until full,
        // then replace-and-sift-down whenever we see something larger than
        // the current min.
        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);

        fn sift_down(h: &mut [(f32, u32)], mut i: usize) {
            let n = h.len();
            loop {
                let mut smallest = i;
                let l = 2 * i + 1;
                let r = 2 * i + 2;
                if l < n && h[l].0 < h[smallest].0 {
                    smallest = l;
                }
                if r < n && h[r].0 < h[smallest].0 {
                    smallest = r;
                }
                if smallest == i {
                    break;
                }
                h.swap(i, smallest);
                i = smallest;
            }
        }

        for (i, &s) in scores.iter().enumerate() {
            if !s.is_finite() {
                continue;
            }
            if heap.len() < k {
                heap.push((s, i as u32));
                if heap.len() == k {
                    for j in (0..k / 2).rev() {
                        sift_down(&mut heap, j);
                    }
                }
            } else if s > heap[0].0 {
                heap[0] = (s, i as u32);
                sift_down(&mut heap, 0);
            }
        }
        if heap.len() < k && heap.len() > 1 {
            for j in (0..heap.len() / 2).rev() {
                sift_down(&mut heap, j);
            }
        }

        heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        heap.into_iter().map(|(s, i)| (i, s)).collect()
    }

    /// KNN against lm_head: find top-K tokens by dot product with query vector.
    /// Single BLAS gemv: query[1, hidden] @ lm_head[vocab, hidden]^T → [1, vocab].
    /// Then top-K selection. Returns (token_id, score) sorted by score descending.
    pub fn lm_head_knn(&self, query: &ndarray::Array1<f32>, top_k: usize) -> Vec<(u32, f32)> {
        let mmap_view = match self.storage.lm_head_f32_view() {
            Some(v) => v,
            None => return vec![],
        };
        let mmap: &[u8] = mmap_view.as_ref();
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        if vocab == 0 {
            return vec![];
        }

        let expected = vocab * hidden * 4;
        if mmap.len() < expected {
            return vec![];
        }

        // Zero-copy: reinterpret mmap as [vocab, hidden] f32 matrix
        let data = unsafe {
            let ptr = mmap.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, vocab * hidden)
        };
        let lm_view = ndarray::ArrayView2::from_shape((vocab, hidden), data).unwrap();

        // gemv via larql-compute: scores = query @ lm_head^T → [1, vocab]
        let hidden = self.hidden_size;
        let x = query.view().into_shape_with_order((1, hidden)).unwrap();
        let cpu = larql_compute::CpuBackend;
        use larql_compute::MatMul;
        let result = cpu.matmul_transb(x, lm_view); // [1, hidden] @ [vocab, hidden]^T → [1, vocab]
        let scores = ndarray::Array1::from_vec(result.into_raw_vec_and_offset().0);

        // Top-K selection
        let mut indexed: Vec<(u32, f32)> = scores
            .iter()
            .copied()
            .enumerate()
            .map(|(i, s)| (i as u32, s))
            .collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed
    }
}

#[cfg(test)]
mod tests {
    //! Coverage for the f32 BLAS fallback path (`lm_head_knn`) plus the
    //! `Stride32Mode` env-var dispatch and the early-return guards on
    //! the f16 / Q4_K backend paths. The Q4_K matvec happy path needs
    //! a real ComputeBackend with `supports_quant(Q4_K)`; that's covered by the
    //! Metal integration tests, not here.
    use ndarray::Array1;

    use super::*;
    use crate::format::filenames::LM_HEAD_BIN;

    /// Build a `VectorIndex` with a synthetic `lm_head.bin` mmap'd in.
    /// `lm_head` is row-major `[vocab, hidden]` f32.
    fn vindex_with_lm_head(vocab: usize, hidden: usize, data: &[f32]) -> VectorIndex {
        assert_eq!(data.len(), vocab * hidden);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(LM_HEAD_BIN);
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).unwrap();

        let mut v = VectorIndex::empty(1, hidden);
        v.load_lm_head(tmp.path())
            .expect("synthetic lm_head loads cleanly");
        // Hold the tempdir for the test lifetime — the mmap's underlying
        // file stays open via memmap2 even if the dir is dropped on Linux,
        // but on macOS the file needs to outlive the mmap.
        std::mem::forget(tmp);
        assert_eq!(v.vocab_size, vocab);
        v
    }

    #[test]
    fn lm_head_knn_returns_empty_when_no_mmap() {
        let v = VectorIndex::empty(1, 4);
        let q = Array1::from_vec(vec![1.0_f32; 4]);
        assert!(v.lm_head_knn(&q, 5).is_empty());
    }

    #[test]
    fn lm_head_knn_returns_empty_when_vocab_zero() {
        // Force lm_head_mmap to be Some but vocab_size = 0.
        let mut v = VectorIndex::empty(1, 4);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(LM_HEAD_BIN);
        std::fs::write(&path, [0u8; 16]).unwrap();
        v.load_lm_head(tmp.path()).unwrap();
        // Override vocab_size to 0 to hit the `vocab == 0` guard.
        v.vocab_size = 0;
        std::mem::forget(tmp);
        let q = Array1::from_vec(vec![1.0_f32; 4]);
        assert!(v.lm_head_knn(&q, 5).is_empty());
    }

    #[test]
    fn lm_head_knn_returns_empty_when_mmap_too_small() {
        // Manually construct: tiny lm_head mmap with vocab=99 declared
        // → required bytes vastly exceeds actual mmap size.
        let mut v = VectorIndex::empty(1, 4);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(LM_HEAD_BIN);
        let bytes: Vec<u8> = (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect();
        std::fs::write(&path, &bytes).unwrap();
        v.load_lm_head(tmp.path()).unwrap();
        // Manually inflate vocab to force a too-small check.
        v.vocab_size = 99;
        std::mem::forget(tmp);
        let q = Array1::from_vec(vec![1.0_f32; 4]);
        assert!(v.lm_head_knn(&q, 5).is_empty());
    }

    #[test]
    fn lm_head_knn_returns_top_k_in_descending_score_order() {
        // lm_head is [4, 2] — 4 vocab tokens × 2 hidden dims.
        // Each row is a "token's projection vector". The query
        // selects which row scores highest.
        let lm_head: Vec<f32> = vec![
            1.0, 0.0, // token 0: aligned with [1, 0]
            0.5, 0.5, // token 1: 45° in xy
            0.0, 1.0, // token 2: aligned with [0, 1]
            -1.0, -1.0, // token 3: anti-aligned with [1, 1]
        ];
        let v = vindex_with_lm_head(4, 2, &lm_head);
        // Query [1, 1] → scores: [1.0, 1.0, 1.0, -2.0]
        let q = Array1::from_vec(vec![1.0_f32, 1.0]);
        let hits = v.lm_head_knn(&q, 3);
        assert_eq!(hits.len(), 3);
        // Top 3 should be tokens 0, 1, 2 (in some order — they all tie at 1.0).
        let mut ids: Vec<u32> = hits.iter().map(|(id, _)| *id).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2]);
        // Lowest score (token 3 = -2.0) is excluded.
        assert!(hits.iter().all(|(id, _)| *id != 3));
    }

    #[test]
    fn lm_head_knn_top_k_zero_returns_full_unsorted_pairs() {
        // The truncate guard requires k > 0; with k = 0 we return the
        // full indexed set sorted by score (no truncation). This pins
        // the actual contract — callers passing top_k=0 should treat
        // it as "give me everything".
        let lm_head: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let v = vindex_with_lm_head(2, 2, &lm_head);
        let q = Array1::from_vec(vec![1.0_f32, 0.0]);
        let hits = v.lm_head_knn(&q, 0);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn lm_head_knn_top_k_larger_than_vocab_returns_all() {
        let lm_head: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let v = vindex_with_lm_head(2, 2, &lm_head);
        let q = Array1::from_vec(vec![1.0_f32, 0.0]);
        let hits = v.lm_head_knn(&q, 99);
        assert_eq!(hits.len(), 2);
    }

    // ── lm_head_stride32_mode ──

    /// RAII env-var override for LARQL_LM_HEAD_STRIDE32. Restores prior
    /// value on drop.
    struct EnvSet {
        prev: Option<String>,
    }
    impl EnvSet {
        fn set(value: Option<&str>) -> Self {
            let prev = std::env::var("LARQL_LM_HEAD_STRIDE32").ok();
            match value {
                Some(v) => std::env::set_var("LARQL_LM_HEAD_STRIDE32", v),
                None => std::env::remove_var("LARQL_LM_HEAD_STRIDE32"),
            }
            Self { prev }
        }
    }
    impl Drop for EnvSet {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var("LARQL_LM_HEAD_STRIDE32", v),
                None => std::env::remove_var("LARQL_LM_HEAD_STRIDE32"),
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn stride32_mode_unset_falls_back() {
        let _g = EnvSet::set(None);
        assert_eq!(lm_head_stride32_mode(), Stride32Mode::Fallback);
    }

    #[test]
    #[serial_test::serial]
    fn stride32_mode_truthy_values_select_first() {
        for val in ["1", "true", "on", "yes"] {
            let _g = EnvSet::set(Some(val));
            assert_eq!(lm_head_stride32_mode(), Stride32Mode::First, "value {val}");
        }
    }

    #[test]
    #[serial_test::serial]
    fn stride32_mode_falsy_values_select_disabled() {
        for val in ["0", "false", "off", "no"] {
            let _g = EnvSet::set(Some(val));
            assert_eq!(
                lm_head_stride32_mode(),
                Stride32Mode::Disabled,
                "value {val}"
            );
        }
    }

    #[test]
    #[serial_test::serial]
    fn stride32_mode_unknown_value_falls_back() {
        let _g = EnvSet::set(Some("maybe"));
        assert_eq!(lm_head_stride32_mode(), Stride32Mode::Fallback);
    }

    // ── Backend dispatch coverage ─────────────────────────────────
    //
    // These tests exercise the `lm_head_knn_backend` /
    // `lm_head_knn_backend_skip_q4k` paths on a `CpuBackend`. The
    // Cpu backend has Q4 support (`supports_quant(Q4_K)` is true), so the Q4
    // stride-32 / matvec branches fire when Q4 bytes are present.
    // f32 fallback fires when neither Q4 nor f16 is loaded.

    #[test]
    #[serial_test::serial]
    fn lm_head_knn_backend_falls_back_to_f32_when_no_q4_or_f16() {
        // Stride-32 disabled so the backend tries f16 (none) → f32.
        let _g = EnvSet::set(Some("0"));
        let lm_head: Vec<f32> = vec![
            1.0, 0.0, //
            0.0, 1.0, //
        ];
        let v = vindex_with_lm_head(2, 2, &lm_head);
        let cpu = larql_compute::CpuBackend;
        let q = Array1::from_vec(vec![1.0_f32, 0.0]);
        let hits = v.lm_head_knn_backend(&q, 1, &cpu);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0, "token 0 [1, 0] best matches query [1, 0]");
    }

    #[test]
    #[serial_test::serial]
    fn lm_head_knn_backend_skip_q4k_falls_back_to_f32() {
        let _g = EnvSet::set(Some("0"));
        let lm_head: Vec<f32> = vec![
            1.0, 0.0, //
            0.0, 1.0, //
        ];
        let v = vindex_with_lm_head(2, 2, &lm_head);
        let cpu = larql_compute::CpuBackend;
        let q = Array1::from_vec(vec![0.0_f32, 1.0]);
        let hits = v.lm_head_knn_backend_skip_q4k(&q, 1, &cpu);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    /// `lm_head_stride32_backend_hits` short-circuits when no Q4 mmap
    /// is loaded on the storage façade. Exercised through the public
    /// `lm_head_knn_backend` with stride32 enabled.
    #[test]
    #[serial_test::serial]
    fn lm_head_stride32_path_returns_none_without_q4_mmap() {
        let _g = EnvSet::set(Some("1"));
        let lm_head: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let v = vindex_with_lm_head(2, 2, &lm_head);
        // Storage has no lm_head_q4 — stride32 returns None, falls
        // through to f16 (none) → f32, returns the f32 dot.
        let cpu = larql_compute::CpuBackend;
        let q = Array1::from_vec(vec![1.0_f32, 0.0]);
        let hits = v.lm_head_knn_backend(&q, 1, &cpu);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
    }

    /// `lm_head_f16_backend_hits` returns None when vocab_size is 0.
    /// We force the path by populating the f16 mmap then setting
    /// vocab_size = 0 — the early-return on `vocab > 0` fires.
    #[test]
    #[serial_test::serial]
    fn lm_head_f16_path_returns_none_when_vocab_zero() {
        let _g = EnvSet::set(Some("0")); // skip stride32

        // f16-encoded 2×2 lm_head.
        let f16_bytes = larql_models::quant::half::encode_f16(&[1.0, 0.0, 0.0, 1.0]);
        let mut anon = memmap2::MmapOptions::new()
            .len(f16_bytes.len())
            .map_anon()
            .unwrap();
        anon.copy_from_slice(&f16_bytes);
        let mmap = anon.make_read_only().unwrap();

        let mut v = VectorIndex::empty(1, 2);
        v.set_lm_head_f16_mmap(std::sync::Arc::new(mmap));
        // Force vocab_size = 0 so the f16 path bails on the inner guard.
        v.vocab_size = 0;
        let cpu = larql_compute::CpuBackend;
        let q = Array1::from_vec(vec![1.0_f32, 0.0]);
        // f16 returns None (vocab=0), f32 fallback also vocab=0 → empty.
        let hits = v.lm_head_knn_backend(&q, 1, &cpu);
        assert!(hits.is_empty());
    }
}
