//! LM-head loaders + KNN.
//!
//! Loads the output projection (vocab × hidden) in one of three formats:
//!
//! - **Q4_K** (`lm_head_q4.bin`): GPU Q4 matvec, ~1 ms on Metal.
//! - **f16**: adopted from the vindex's `embeddings.bin` when that file
//!   is IEEE-half (tied-embedding Gemma / Llama). Drives Metal's
//!   `f16_gemv` shader — half the memory-bandwidth of f32 without the
//!   5.6 GB heap clone that a dequantised lm_head would need on 31B.
//! - **f32** (`lm_head.bin` or cloned from `embed`): CPU BLAS fallback.
//!
//! `lm_head_knn_backend` dispatches in the order above, using the
//! cheapest available backend path for the loaded lm_head representation.
//! Sibling to `super::walk` (FFN) and `super::attn` (attention).
//!
//! Per-concern layout (M9 cleanup, 2026-05-01):
//! - `loaders.rs` — file/mmap loaders + the f16-derived synth path
//! - `knn.rs`     — the three KNN dispatch paths and the shared
//!                  `top_k_sorted` reduce.
//! Constants, the `read_lm_head_manifest_kind` helper, and the unit
//! tests (which span loader + KNN seams) stay here.

use larql_models::quant::ggml::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS,
};

use crate::format::filenames::*;

mod knn;
mod loaders;

/// Numerator/denominator used to back-derive `vocab_size` from a Q4-packed
/// lm_head file's byte length. Q4_K (144 B / 256 elems) and Q4_0 (18 B / 32
/// elems) both rate at 0.5625 B/element, i.e. `9/16`. Knowing only the file
/// size and `hidden_size`, the inverse is `vocab = bytes * 16 / (hidden * 9)`.
pub(super) const Q4_BYTES_PER_ELEM_NUM: usize = 9;
pub(super) const Q4_BYTES_PER_ELEM_DEN: usize = 16;

// Compile-time invariants — if either constant ever changes, this assertion
// catches the byte-rate calc immediately rather than producing silent vocab
// inference drift.
const _: () = assert!(
    Q4_K_BLOCK_BYTES * Q4_BYTES_PER_ELEM_DEN == Q4_K_BLOCK_ELEMS * Q4_BYTES_PER_ELEM_NUM,
    "Q4_K byte rate drift: 144/256 must equal 9/16",
);
const _: () = assert!(
    Q4_0_BLOCK_BYTES * Q4_BYTES_PER_ELEM_DEN == Q4_0_BLOCK_ELEMS * Q4_BYTES_PER_ELEM_NUM,
    "Q4_0 byte rate drift: 18/32 must equal 9/16",
);

/// Read the manifest entry for `lm_head.weight` from `weight_manifest.json`,
/// if the manifest exists and contains an entry for that key. Returns `None`
/// when the manifest is absent (older vindexes) or doesn't list lm_head.
///
/// Used by `load_lm_head_kquant` to assert the on-disk file matches the format
/// the reader is about to dispatch. The Q4_K-vs-Q4_0 byte-rate collision
/// (0.5625 B/elem in both formats) made silent format mismatches invisible
/// to file-size validation; checking the manifest's `kind` discriminator
/// catches the mismatch at load-time.
pub(super) fn read_lm_head_manifest_kind(dir: &std::path::Path) -> Option<String> {
    let manifest_path = dir.join(WEIGHT_MANIFEST_JSON);
    let text = std::fs::read_to_string(&manifest_path).ok()?;
    let entries: Vec<crate::format::weights::write_f32::WeightEntry> =
        serde_json::from_str(&text).ok()?;
    entries
        .into_iter()
        .find(|e| e.key == "lm_head.weight")
        .map(|e| e.kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::VectorIndex;
    use std::sync::Arc;

    /// `top_k_sorted` is the shared reduce used by Q4 / f16 / f32 paths.
    /// Pin the contract: descending by score, capped at `top_k`.
    #[test]
    fn top_k_sorted_descending_and_capped() {
        let scores = vec![0.5f32, 0.1, 0.9, 0.3, 0.7];
        let top3 = VectorIndex::top_k_sorted(scores.clone(), 3);
        let tokens: Vec<u32> = top3.iter().map(|(t, _)| *t).collect();
        let probs: Vec<f32> = top3.iter().map(|(_, s)| *s).collect();
        assert_eq!(
            tokens,
            vec![2, 4, 0],
            "expect descending-by-score token order"
        );
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);

        // top_k larger than input → no truncation, but still sorted.
        let all = VectorIndex::top_k_sorted(scores, 99);
        assert_eq!(all.len(), 5);
        let probs: Vec<f32> = all.iter().map(|(_, s)| *s).collect();
        assert!(probs.windows(2).all(|w| w[0] >= w[1]));
    }

    /// `top_k = 0` returns an empty Vec, never the input.
    #[test]
    fn top_k_sorted_zero_returns_empty() {
        let scores = vec![0.5f32, 0.1, 0.9];
        let out = VectorIndex::top_k_sorted(scores, 0);
        assert!(out.is_empty());
    }

    /// Empty score vector → empty output (no panic).
    #[test]
    fn top_k_sorted_empty_input_returns_empty() {
        let out = VectorIndex::top_k_sorted(Vec::new(), 5);
        assert!(out.is_empty());
    }

    /// `top_k = 1` takes the argmax fast path. Filter is `is_finite()` —
    /// NaN, +∞ and -∞ are all skipped (matching `backend_lm_head_topk` in
    /// the inference crate). Test pins this contract: the highest finite
    /// score wins, regardless of any ±∞ entries.
    #[test]
    fn top_k_sorted_k1_argmax_skips_non_finite() {
        let scores = vec![0.2f32, f32::NAN, 0.9, f32::NEG_INFINITY, 0.5, f32::INFINITY];
        let out = VectorIndex::top_k_sorted(scores, 1);
        assert_eq!(out.len(), 1, "expected one finite winner");
        assert_eq!(out[0].0, 2, "highest finite score is 0.9 at idx 2");
        assert!((out[0].1 - 0.9).abs() < 1e-6);
    }

    /// All-NaN scores yield an empty argmax (no garbage token id).
    #[test]
    fn top_k_sorted_k1_all_nan_returns_empty() {
        let scores = vec![f32::NAN; 10];
        let out = VectorIndex::top_k_sorted(scores, 1);
        assert!(out.is_empty());
    }

    /// Heap path (k=3) skips non-finite values and returns sorted descending.
    #[test]
    fn top_k_sorted_heap_skips_non_finite() {
        let scores = vec![0.1f32, f32::NAN, 0.9, 0.5, f32::NEG_INFINITY, 0.3];
        let out = VectorIndex::top_k_sorted(scores, 3);
        let tokens: Vec<u32> = out.iter().map(|(t, _)| *t).collect();
        assert_eq!(tokens, vec![2, 3, 5]);
    }

    /// Fewer finite values than k → return only the finite ones, sorted.
    #[test]
    fn top_k_sorted_heap_fewer_finite_than_k() {
        let scores = vec![0.7f32, f32::NAN, 0.3, f32::NAN, f32::NAN];
        let out = VectorIndex::top_k_sorted(scores, 5);
        let tokens: Vec<u32> = out.iter().map(|(t, _)| *t).collect();
        assert_eq!(tokens, vec![0, 2]);
    }

    /// Tied scores: return is descending by score; tied tokens are still
    /// distinct (no duplicate index). Stability of which tied index wins
    /// is implementation-defined.
    #[test]
    fn top_k_sorted_handles_ties() {
        let scores = vec![0.5f32, 0.7, 0.5, 0.7, 0.1];
        let out = VectorIndex::top_k_sorted(scores, 3);
        assert_eq!(out.len(), 3);
        let probs: Vec<f32> = out.iter().map(|(_, s)| *s).collect();
        assert!(probs.windows(2).all(|w| w[0] >= w[1]));
        let tokens: std::collections::HashSet<u32> = out.iter().map(|(t, _)| *t).collect();
        assert_eq!(tokens.len(), 3, "no duplicate token ids in top-k output");
    }

    /// `synthesize_lm_head_kquant` converts f16 embeddings to Q4_0 in RAM.
    ///
    /// Invariants:
    ///   - `has_lm_head_kquant` false before synthesis, true after.
    ///   - Output byte length = vocab × (hidden/32 × 18).
    ///   - Re-quantizing a row via CPU path gives dot-product scores that rank
    ///     the matching row first (round-trip correctness).
    #[test]
    fn synthesize_lm_head_q4_produces_correct_bytes() {
        let vocab: usize = 16;
        // Q4_K uses 256-element super-blocks; the synth path now matches
        // the on-disk `lm_head_q4.bin` writer (Q4_K) so hidden must be a
        // multiple of 256. Earlier this used 64 (Q4_0's 32-elem blocks)
        // and the synth emitted Q4_0, which silently corrupted logits
        // when `lm_head_knn_backend` dispatched `q4k_matvec` on it.
        let hidden: usize = 256;

        // Build a synthetic f16 embedding table: row i = constant (i+1) * 0.01
        let mut f16_bytes = vec![0u8; vocab * hidden * 2];
        for row in 0..vocab {
            let val = (row as f32 + 1.0) * 0.01;
            let bits = larql_models::quant::half::f32_to_f16(val);
            for col in 0..hidden {
                let off = (row * hidden + col) * 2;
                let b = bits.to_le_bytes();
                f16_bytes[off] = b[0];
                f16_bytes[off + 1] = b[1];
            }
        }

        // Minimal VectorIndex with the f16 mmap and known dims.
        let mmap = Arc::new({
            let mem = memmap2::MmapMut::map_anon(f16_bytes.len()).unwrap();
            let mut mem = mem;
            mem.copy_from_slice(&f16_bytes);
            mem.make_read_only().unwrap()
        });

        let mut index =
            crate::index::core::VectorIndex::new(vec![None; 1], vec![None; 1], 1, hidden);
        index.vocab_size = vocab;
        index.set_lm_head_f16_mmap(mmap);

        assert!(
            !index.has_lm_head_kquant(),
            "should not have Q4 before synthesis"
        );
        index.synthesize_lm_head_kquant();
        assert!(index.has_lm_head_kquant(), "should have Q4 after synthesis");

        // Byte length check uses canonical Q4_K block geometry from
        // `larql-models::quant::ggml` so the test fails immediately if the
        // writer ever switches blocks under us.
        let synth = index
            .storage
            .lm_head_kquant_view()
            .expect("synth must populate lm_head_q4");
        let super_blocks = (vocab * hidden) / Q4_K_BLOCK_ELEMS;
        assert_eq!(
            synth.len(),
            super_blocks * Q4_K_BLOCK_BYTES,
            "synthesized Q4_K byte length should be \
             (vocab × hidden / Q4_K_BLOCK_ELEMS) × Q4_K_BLOCK_BYTES — \
             a different rate (e.g. /Q4_0_BLOCK_ELEMS × Q4_0_BLOCK_BYTES) means \
             the synth path has drifted from the on-disk Q4_K writer and \
             `q4k_matvec` will read it as garbage. Same byte rate (0.5625 \
             B/elem) makes this regression silent without an explicit \
             super-block count check."
        );

        // Calling again should be a no-op (idempotent).
        let ptr_before = synth.as_ptr();
        index.synthesize_lm_head_kquant();
        let ptr_after = index
            .storage
            .lm_head_kquant_view()
            .expect("synth must remain populated")
            .as_ptr();
        assert_eq!(ptr_before, ptr_after, "second call should not reallocate");
    }

    /// Regression: a vindex shipping `lm_head_q4.bin` but no `lm_head.bin`
    /// (the post-2026-04-26 Q4_K writer's default) used to leave
    /// `vocab_size = 0`. The Q4 lm_head fast path then silently bailed
    /// (`if vocab > 0`), forcing a 4× slower fallback through the f32
    /// BLAS gemv on `weights.lm_head`. This test pins the fix:
    /// `load_lm_head_kquant` must populate `vocab_size` from the file size
    /// when no other source has set it.
    #[test]
    fn load_lm_head_q4_sets_vocab_size_from_file_size() {
        // Q4_K and Q4_0 both rate at `Q4_BYTES_PER_ELEM_NUM /
        // Q4_BYTES_PER_ELEM_DEN` (= 9/16 = 0.5625 B/elem), so the same
        // formula handles both. vocab=256 × hidden=128 → 18432 bytes.
        let hidden = 128usize;
        let vocab = 256usize;
        let bytes = vocab * hidden * Q4_BYTES_PER_ELEM_NUM / Q4_BYTES_PER_ELEM_DEN;
        let payload = vec![0u8; bytes];

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_KQUANT_BIN), &payload).unwrap();

        // Build a minimal index — vocab_size starts at 0.
        let mut index = VectorIndex::empty(1, hidden);
        assert_eq!(index.vocab_size, 0);

        index
            .load_lm_head_kquant(tmp.path())
            .expect("load lm_head_q4");

        assert_eq!(
            index.vocab_size, vocab,
            "load_lm_head_kquant must derive vocab_size from file size when it's 0"
        );
    }

    /// Companion: when `vocab_size` is *already* set (by index.json or
    /// `load_lm_head`), `load_lm_head_kquant` must not clobber it.
    #[test]
    fn load_lm_head_q4_does_not_overwrite_existing_vocab_size() {
        let hidden = 128usize;
        let bytes = 256 * hidden * Q4_BYTES_PER_ELEM_NUM / Q4_BYTES_PER_ELEM_DEN;
        let payload = vec![0u8; bytes];
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_KQUANT_BIN), &payload).unwrap();

        let mut index = VectorIndex::empty(1, hidden);
        index.vocab_size = 999; // pretend index.json already set this
        index.load_lm_head_kquant(tmp.path()).unwrap();

        assert_eq!(index.vocab_size, 999, "must not clobber preset vocab_size");
    }

    /// Companion: `load_lm_head_kquant` is a no-op for vocab_size when the
    /// hidden_size is 0 (avoid div-by-zero / nonsense vocab).
    #[test]
    fn load_lm_head_q4_skips_vocab_inference_when_hidden_size_zero() {
        let payload = vec![0u8; 100];
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_KQUANT_BIN), &payload).unwrap();

        let mut index = VectorIndex::empty(1, 0);
        index.load_lm_head_kquant(tmp.path()).unwrap();
        assert_eq!(
            index.vocab_size, 0,
            "no inference possible without hidden_size"
        );
    }

    /// Regression test for the gemma3-4b-v2 garbage-output bug (2026-04-27):
    /// `format/weights/write_kquant::write_model_weights_kquant` writes
    /// `lm_head_q4.bin` as **Q4_K** (144 B / 256 elems with sub-block
    /// scales/mins). `lm_head_knn_backend` previously dispatched
    /// `backend.q4_matvec` which is **Q4_0** (18 B / 32 elems with one f16
    /// scale): same byte rate, completely different layout, silent garbage.
    ///
    /// This pins the contract that the two ends of the pipeline agree on
    /// the format. Round-trip a known matrix through the writer's
    /// quantiser, run it through `lm_head_knn_backend`, and assert the
    /// top-1 token matches the f32 dot-product reference.
    #[test]
    fn lm_head_q4k_writer_reader_format_round_trip() {
        // Q4_K constraint: hidden must be a multiple of 256, vocab*hidden
        // must be a multiple of 256. 256×256 satisfies both with cheap
        // numerical work for a unit test.
        let vocab = 256usize;
        let hidden = 256usize;

        // Build a deterministic, well-conditioned [vocab, hidden] matrix.
        // Each row has a peak at one column so the f32 reference has an
        // unambiguous top-1 answer for any one-hot-ish query, while
        // sub-block scales/mins are non-trivial (Q4_K is structure-aware).
        let mut lm_head = vec![0.0f32; vocab * hidden];
        for v in 0..vocab {
            for h in 0..hidden {
                // Peak shaped like a smooth Gaussian centred at column v%hidden,
                // with a small ramp for off-diagonal values.
                let dist = ((h as f32) - (v as f32 % hidden as f32)).abs();
                lm_head[v * hidden + h] = (-dist * 0.05).exp() + 0.001 * (h as f32);
            }
        }

        // Quantise via the SAME writer the production extractor uses.
        let q4k_bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&lm_head);
        // Sanity: byte count matches the canonical Q4_K rate.
        assert_eq!(
            q4k_bytes.len(),
            vocab * hidden / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES,
            "Q4_K quant should produce Q4_K_BLOCK_BYTES per Q4_K_BLOCK_ELEMS-element super-block"
        );

        // Inject into a synthetic VectorIndex via the synth path.
        let mut index = VectorIndex::empty(1, hidden);
        index.vocab_size = vocab;
        Arc::make_mut(&mut index.storage).set_lm_head_kquant_synth(Arc::new(q4k_bytes));

        // Pick a query that points at a known peak — token 42's row peaks
        // at column 42, so the dot product is highest at row 42.
        let target_token = 42u32;
        let mut query = ndarray::Array1::<f32>::zeros(hidden);
        query[target_token as usize] = 1.0;

        // f32 reference: dot product of `query` against every row of `lm_head`.
        let ref_scores: Vec<f32> = (0..vocab)
            .map(|v| {
                (0..hidden)
                    .map(|h| lm_head[v * hidden + h] * query[h])
                    .sum()
            })
            .collect();
        let ref_top1 = ref_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        assert_eq!(
            ref_top1, target_token,
            "fixture sanity: f32 reference must pick token 42"
        );

        // Run through the production dispatch with a CPU backend.
        let cpu = larql_compute::CpuBackend;
        let hits = index.lm_head_knn_backend(&query, 5, &cpu);
        assert!(
            !hits.is_empty(),
            "lm_head_knn_backend returned empty — Q4_K dispatch silently failed; \
             this is exactly the format-collision bug the test exists to catch"
        );
        let (top_token, _) = hits[0];
        assert_eq!(
            top_token, target_token,
            "Q4_K-quantised lm_head must select the same top-1 token as the \
             f32 reference (within Q4_K noise on a Gaussian-peak fixture). \
             A mismatch here means the writer and reader disagree on the \
             quantisation format — most likely a regression of the \
             Q4_K-vs-Q4_0 dispatch confusion fixed in 2026-04-27. \
             ref_top1={ref_top1}, got={top_token}"
        );

        // Stronger: top-5 must include the target (ranking can shift by
        // ±1 from Q4_K noise on the smooth fixture, but not by hundreds).
        let top5_tokens: Vec<u32> = hits.iter().map(|(t, _)| *t).collect();
        assert!(
            top5_tokens.contains(&target_token),
            "top-5 must contain target token {target_token}, got {top5_tokens:?}"
        );
    }

    /// Companion: the synth path (`synthesize_lm_head_kquant`) must produce
    /// the same Q4_K format as the on-disk writer. Earlier the synth path
    /// emitted Q4_0 while the writer emitted Q4_K — both ended up routed
    /// through `q4k_matvec` after the dispatch fix, so a Q4_0 synth would
    /// silently corrupt logits for tied-embedding models that take the
    /// synth branch.
    #[test]
    fn synth_q4_lm_head_uses_q4k_format() {
        let vocab = 256usize;
        let hidden = 256usize;

        // Build an f16 mmap-shaped buffer (vocab × hidden × 2 bytes).
        // Use simple values so f16 conversion round-trips cleanly.
        let mut f16_buf = vec![0u8; vocab * hidden * 2];
        for v in 0..vocab {
            for h in 0..hidden {
                let val = if h == v { 1.0f32 } else { 0.01 };
                let bits = larql_models::quant::half::f32_to_f16(val);
                let off = (v * hidden + h) * 2;
                f16_buf[off] = (bits & 0xff) as u8;
                f16_buf[off + 1] = ((bits >> 8) & 0xff) as u8;
            }
        }

        let mut index = VectorIndex::empty(1, hidden);
        index.vocab_size = vocab;
        index.set_lm_head_f16_mmap(Arc::new(memmap_from_bytes(&f16_buf)));
        index.synthesize_lm_head_kquant();

        let synth = index
            .storage
            .lm_head_kquant_view()
            .expect("synth must populate lm_head_q4");
        // Q4_K size invariant: Q4_K_BLOCK_BYTES per Q4_K_BLOCK_ELEMS-element super-block.
        assert_eq!(
            synth.len(),
            vocab * hidden / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES,
            "synth must produce Q4_K-sized bytes \
             (Q4_K_BLOCK_BYTES B / Q4_K_BLOCK_ELEMS elems), not Q4_0-sized \
             (Q4_0_BLOCK_BYTES B / Q4_0_BLOCK_ELEMS elems). Same byte rate \
             per element makes this regression silent without this assert."
        );

        // Functional check: top-1 against an indicator query points at the
        // expected diagonal token.
        let target = 17u32;
        let mut query = ndarray::Array1::<f32>::zeros(hidden);
        query[target as usize] = 1.0;
        let cpu = larql_compute::CpuBackend;
        let hits = index.lm_head_knn_backend(&query, 5, &cpu);
        let top: Vec<u32> = hits.iter().map(|(t, _)| *t).collect();
        assert!(
            top.contains(&target),
            "synth Q4_K lm_head must rank target token {target} in top-5 \
             of an indicator query; got {top:?}"
        );
    }

    /// Helper: build a memmap2::Mmap-shaped byte source for tests. Writes
    /// to a tempfile and mmaps it back — the synth function holds an
    /// `Arc<Mmap>` so we can't fake it inline.
    fn memmap_from_bytes(bytes: &[u8]) -> memmap2::Mmap {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), bytes).unwrap();
        let f = std::fs::File::open(tmp.path()).unwrap();
        unsafe { memmap2::Mmap::map(&f).unwrap() }
    }

    /// Architectural regression test: when `weight_manifest.json` lists
    /// `lm_head.weight` with `kind != tensor_q4k`, `load_lm_head_kquant` must
    /// refuse to load. This is the bug class that produced silent garbage
    /// logits in gemma3-4b-v2.vindex (writer Q4_K, reader Q4_0 dispatch).
    #[test]
    fn load_lm_head_q4_rejects_manifest_kind_mismatch() {
        let hidden = 128usize;
        let vocab = 256usize;
        let bytes = vocab * hidden * Q4_BYTES_PER_ELEM_NUM / Q4_BYTES_PER_ELEM_DEN;

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_KQUANT_BIN), vec![0u8; bytes]).unwrap();

        // Manifest claims lm_head is f16 — incompatible with Q4_K dispatch.
        let manifest = serde_json::json!([{
            "key": "lm_head.weight",
            "kind": crate::format::weights::write_f32::kind::TENSOR_F16,
            "shape": [vocab, hidden],
            "offset": 0,
            "length": bytes,
            "file": "lm_head_q4.bin",
        }]);
        std::fs::write(
            tmp.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let mut index = VectorIndex::empty(1, hidden);
        let result = index.load_lm_head_kquant(tmp.path());
        assert!(
            result.is_err(),
            "load_lm_head_kquant must reject when manifest kind disagrees with TENSOR_Q4K"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("manifest mismatch"),
            "error must explain the mismatch, got: {err_msg}"
        );
    }

    /// Companion: when the manifest correctly tags lm_head as TENSOR_Q4K,
    /// loading proceeds normally.
    #[test]
    fn load_lm_head_q4_accepts_correct_manifest_kind() {
        let hidden = 128usize;
        let vocab = 256usize;
        let bytes = vocab * hidden * Q4_BYTES_PER_ELEM_NUM / Q4_BYTES_PER_ELEM_DEN;

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(LM_HEAD_KQUANT_BIN), vec![0u8; bytes]).unwrap();

        let manifest = serde_json::json!([{
            "key": "lm_head.weight",
            "kind": crate::format::weights::write_f32::kind::TENSOR_Q4K,
            "shape": [vocab, hidden],
            "offset": 0,
            "length": bytes,
            "file": "lm_head_q4.bin",
        }]);
        std::fs::write(
            tmp.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let mut index = VectorIndex::empty(1, hidden);
        index
            .load_lm_head_kquant(tmp.path())
            .expect("matching manifest kind should load");
        assert_eq!(index.vocab_size, vocab);
    }
}
