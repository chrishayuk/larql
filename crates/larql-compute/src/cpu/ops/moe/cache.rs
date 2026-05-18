//! Bounded LRU cache for dequantised MoE expert weights.
//!
//! Gemma 4 26B A4B has 128 experts × 30 MoE layers. Per-layer Q4_K storage:
//! ~24 MB f32 per expert (gate_up + down combined). The router picks
//! top-K=8 per layer, so a naive decode path runs ~5.7 GB of Q4_K → f32
//! per token. In practice prompts route consistently to the same experts;
//! a bounded LRU keyed by the mmap pointer lets repeat hits skip both
//! allocation and decode.
//!
//! Key = mmap pointer (the `&[u8]` byte slice for one expert's packed
//! tensor). The mmap is stable for the life of the process, so the pointer
//! uniquely identifies `(layer, expert, kind)` even after the per-expert
//! byte-table refactor — `experts_gate_up[ei]` is still backed by the same
//! mmap range across calls.
//!
//! Value = `Arc<Vec<f32>>`. Cloning on hit is O(1) — real allocation +
//! dequant runs exactly once per cached entry.
//!
//! Sizing: `LARQL_MOE_CACHE_ENTRIES` env var caps the entry count
//! (default 256). At Gemma 4 26B-A4B sizes (~24 MB per cached expert)
//! that's ~6 GB resident per shard at steady state.
//!
//! Why 256: per-token working set is `num_moe_layers × top_k` distinct
//! expert calls. On 26B-A4B that's 30 × 8 = 240. Cap=64 (the prior
//! default) thrashed at near-100% miss rate because every token visits
//! 240 experts but the cache only held 64 — by the time the next token
//! came back to layer 0, the experts had been evicted. Cap=256 gives
//! one full token's working set plus headroom, taking the steady-state
//! hit rate from ~0% to >90% for prompts with stable routing (most
//! chat-style workloads).
//!
//! For multi-prompt servers with high routing diversity, raise this
//! further (512 / 1024) — RSS scales linearly. Set to 0 to disable
//! caching entirely (right answer once the NEON-vectorised direct-Q4K
//! matvec lands; see compute ROADMAP).
//!
//! Format dispatch (BF16 / Q4_K / F32) is part of the cache key. The same
//! address can be interpreted differently across formats or expected padded
//! lengths, so the key includes pointer, length, format, and expected float
//! count.

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock, RwLock};

use crate::options;

/// LRU cache entry: dequantised expert weights.
pub(super) type ExpertF32 = Arc<Vec<f32>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DequantError {
    UnsupportedFormat(crate::QuantFormat),
}

impl std::fmt::Display for DequantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(format) => {
                write!(f, "CPU MoE dequant does not support {format:?}")
            }
        }
    }
}

impl std::error::Error for DequantError {}

/// Cache key — in production the byte slice's start pointer is stable across
/// the lifetime of the mmap, so different experts in the same packed tensor get
/// distinct keys via their offset. Length, format, and expected float count are
/// included so reused addresses or differently interpreted byte slices cannot
/// return a stale decode. Tests use short heap Vecs whose addresses can be
/// recycled between cases, so include a content fingerprint under `cfg(test)`.
#[cfg(not(test))]
type Key = (usize, usize, crate::QuantFormat, usize);

#[cfg(test)]
type Key = (usize, usize, crate::QuantFormat, usize, u64);

#[cfg(not(test))]
fn cache_key(bytes: &[u8], format: crate::QuantFormat, expected_floats: usize) -> Key {
    (
        bytes.as_ptr() as usize,
        bytes.len(),
        format,
        expected_floats,
    )
}

#[cfg(test)]
fn cache_key(bytes: &[u8], format: crate::QuantFormat, expected_floats: usize) -> Key {
    use std::hash::{Hash, Hasher};

    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    (
        bytes.as_ptr() as usize,
        bytes.len(),
        format,
        expected_floats,
        h.finish(),
    )
}

struct Inner {
    map: std::collections::HashMap<Key, ExpertF32>,
    /// Insertion order — used for FIFO eviction when `map.len() > cap`.
    /// Hits do NOT touch this (eviction is now FIFO, not LRU): preserving
    /// recency would force every read to take a write lock, which destroys
    /// the parallel-hit pattern that motivates the `RwLock` switch.
    /// For workloads sized so the working set fits in `cap`, no eviction
    /// happens and the policy difference is moot.
    order: VecDeque<Key>,
    cap: usize,
}

impl Inner {
    fn new(cap: usize) -> Self {
        Self {
            map: std::collections::HashMap::with_capacity(cap.saturating_add(1)),
            order: VecDeque::with_capacity(cap.saturating_add(1)),
            cap,
        }
    }

    /// Read-only lookup — no map mutation, no order update.  Suitable to
    /// run under a shared `RwLock` read guard so concurrent rayon threads
    /// hitting different (or the same) keys don't serialize.
    fn get(&self, key: Key) -> Option<ExpertF32> {
        self.map.get(&key).cloned()
    }

    fn insert(&mut self, key: Key, val: ExpertF32) {
        if self.cap == 0 {
            return;
        }
        if self.map.contains_key(&key) {
            // Already present (a concurrent inserter raced us); don't duplicate.
            return;
        }
        while self.map.len() >= self.cap {
            if let Some(victim) = self.order.pop_front() {
                self.map.remove(&victim);
            } else {
                break;
            }
        }
        self.order.push_back(key);
        self.map.insert(key, val);
    }
}

fn cell() -> &'static RwLock<Inner> {
    static CELL: OnceLock<RwLock<Inner>> = OnceLock::new();
    CELL.get_or_init(|| {
        // Default 256: covers one token's working set on Gemma 4 26B-A4B
        // (30 MoE layers × top_k=8 = 240 distinct experts per token).
        // Prior default of 64 thrashed at ~100% miss rate. See module doc.
        let cap = options::env_usize(options::ENV_MOE_CACHE_ENTRIES).unwrap_or(256);
        RwLock::new(Inner::new(cap))
    })
}

/// Return a cached Arc<Vec<f32>> for `bytes`, dequantising under `format` on
/// miss. `expected_floats` is required for block formats (Q4_K) where the
/// output length is not derivable from the input length without padding info;
/// it's ignored for raw BF16. On hit, no allocation happens.
///
/// Concurrency: the hot path (cache hit) takes a *read* lock so any number of
/// rayon threads can clone their Arcs in parallel.  Misses take a brief write
/// lock only at insert time; the dequant itself runs lock-free.
pub(super) fn try_cached_dequant(
    bytes: &[u8],
    format: crate::QuantFormat,
    expected_floats: usize,
) -> Result<ExpertF32, DequantError> {
    if !matches!(
        format,
        crate::QuantFormat::BF16 | crate::QuantFormat::Q4_K | crate::QuantFormat::F32
    ) {
        return Err(DequantError::UnsupportedFormat(format));
    }

    let key = cache_key(bytes, format, expected_floats);
    // Fast path: shared read lock — concurrent hits don't contend.
    if let Ok(inner) = cell().read() {
        if let Some(hit) = inner.get(key) {
            return Ok(hit);
        }
    }
    // Miss: dequantise OUTSIDE any lock, then take the write lock to insert.
    let decoded = match format {
        crate::QuantFormat::BF16 => super::math::bf16_to_f32(bytes),
        crate::QuantFormat::Q4_K => {
            crate::cpu::ops::q4_common::dequantize_q4_k(bytes, expected_floats)
        }
        crate::QuantFormat::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        _ => unreachable!("unsupported formats return before cache lookup"),
    };
    let arc = Arc::new(decoded);
    if let Ok(mut inner) = cell().write() {
        inner.insert(key, arc.clone());
    }
    Ok(arc)
}

#[cfg(test)]
mod cache_format_tests {
    use super::*;
    use crate::QuantFormat;

    #[cfg(all(target_arch = "wasm32", feature = "browser-tests"))]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    /// BF16 path: 2 bytes per float, no padding. Round-trip a fixed value.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn bf16_dispatch_round_trip() {
        // 4 BF16 values of 1.0 (0x3F80 little-endian = [0x80, 0x3F]).
        let bytes = vec![0x80u8, 0x3F, 0x80, 0x3F, 0x80, 0x3F, 0x80, 0x3F];
        let out = try_cached_dequant(&bytes, QuantFormat::BF16, 4).unwrap();
        assert_eq!(out.len(), 4);
        for v in out.iter() {
            assert!((v - 1.0).abs() < 1e-3, "BF16 1.0 round-trip got {v}");
        }
    }

    /// Q4_K path: 144 bytes per 256 floats. Quantise→dequantise round-trip
    /// must come back within Q4 quantisation noise.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn q4k_dispatch_round_trip() {
        // 256-element ramp [-1, 1] — same fixture used by q4_common tests.
        let data: Vec<f32> = (0..256).map(|i| (i as f32 / 255.0) * 2.0 - 1.0).collect();
        let bytes = crate::cpu::ops::q4_common::quantize_q4_k(&data);
        assert_eq!(bytes.len(), larql_models::quant::ggml::Q4_K_BLOCK_BYTES);

        let out = try_cached_dequant(&bytes, QuantFormat::Q4_K, 256).unwrap();
        assert_eq!(out.len(), 256);
        let max_err: f32 = data
            .iter()
            .zip(&*out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // Q4 nibble step ≈ 0.13 over 2.0 range; allow 2× for sub-block bias.
        assert!(max_err < 0.12, "Q4_K round-trip max error {max_err}");
    }

    /// F32 path: passthrough.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn f32_dispatch_passthrough() {
        let data: Vec<f32> = vec![1.0, -2.5, 3.125, 0.0];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = try_cached_dequant(&bytes, QuantFormat::F32, data.len()).unwrap();
        assert_eq!(out.len(), data.len());
        for (a, b) in data.iter().zip(&*out) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn cache_key_separates_format_and_expected_length() {
        let data: Vec<f32> = vec![1.0, 2.0];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let f32_out = try_cached_dequant(&bytes, QuantFormat::F32, data.len()).unwrap();
        assert_eq!(&*f32_out, &[1.0, 2.0]);

        let bf16_out = try_cached_dequant(&bytes, QuantFormat::BF16, bytes.len() / 2).unwrap();
        assert_eq!(bf16_out.len(), bytes.len() / 2);
        assert_ne!(
            bf16_out.len(),
            f32_out.len(),
            "BF16 lookup must not reuse the prior F32 cache entry"
        );
    }

    /// Unsupported formats fail explicitly instead of looking like a skipped expert.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn unsupported_format_returns_error() {
        let bytes = vec![0u8; 18];
        let err = try_cached_dequant(&bytes, QuantFormat::Q4_0, 32).unwrap_err();
        assert_eq!(err, DequantError::UnsupportedFormat(QuantFormat::Q4_0));
    }

    /// Out-of-bounds Q4_K input returns empty (no panic).
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn q4k_truncated_input_returns_empty() {
        let bytes = vec![0u8; 100]; // 100 < 144 = one super-block
        let out = try_cached_dequant(&bytes, QuantFormat::Q4_K, 256).unwrap();
        assert!(out.is_empty(), "truncated Q4_K → empty (caller skips)");
    }

    /// Q4_K with non-multiple-of-256 expected_floats returns empty.
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn q4k_misaligned_length_returns_empty() {
        let bytes = vec![0u8; larql_models::quant::ggml::Q4_K_BLOCK_BYTES];
        let out = try_cached_dequant(&bytes, QuantFormat::Q4_K, 200).unwrap();
        assert!(out.is_empty(), "expected_floats not a 256 multiple → empty");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn inner_zero_capacity_drops_inserts() {
        let mut inner = Inner::new(0);
        let bytes = vec![1u8, 2, 3, 4];
        let key = cache_key(&bytes, QuantFormat::BF16, 2);

        inner.insert(key, Arc::new(vec![42.0]));

        assert!(inner.get(key).is_none());
        assert!(inner.order.is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn inner_fifo_eviction_removes_oldest_entry() {
        let mut inner = Inner::new(2);
        let a = vec![1u8];
        let b = vec![2u8];
        let c = vec![3u8];
        let ka = cache_key(&a, QuantFormat::BF16, 1);
        let kb = cache_key(&b, QuantFormat::BF16, 1);
        let kc = cache_key(&c, QuantFormat::BF16, 1);

        inner.insert(ka, Arc::new(vec![1.0]));
        inner.insert(kb, Arc::new(vec![2.0]));
        inner.insert(kc, Arc::new(vec![3.0]));

        assert!(inner.get(ka).is_none());
        assert_eq!(&*inner.get(kb).unwrap(), &[2.0]);
        assert_eq!(&*inner.get(kc).unwrap(), &[3.0]);
        assert_eq!(inner.order.len(), 2);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn inner_duplicate_insert_keeps_original_value_and_order() {
        let mut inner = Inner::new(2);
        let bytes = vec![9u8];
        let key = cache_key(&bytes, QuantFormat::BF16, 1);

        inner.insert(key, Arc::new(vec![1.0]));
        inner.insert(key, Arc::new(vec![2.0]));

        assert_eq!(&*inner.get(key).unwrap(), &[1.0]);
        assert_eq!(inner.order.len(), 1);
    }

    /// Parallel cache hits don't deadlock or corrupt — exercises the
    /// `RwLock` read-side under contention.  Many threads request the same
    /// few keys; the cache must stably return the same `Arc` content for
    /// each key without serializing readers (the perf claim isn't
    /// asserted here, but the absence of deadlock and content-identity
    /// regression is).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_hits_do_not_deadlock_or_corrupt() {
        // Pre-warm: a few small BF16 entries.
        let entries: Vec<Vec<u8>> = (0..4)
            .map(|i| {
                let v = (i + 1) as f32;
                let bits = v.to_bits();
                let hi = (bits >> 16) as u16;
                hi.to_le_bytes().repeat(4) // 4 BF16 values per entry
            })
            .collect();
        for e in &entries {
            let _ = try_cached_dequant(e, QuantFormat::BF16, 4).unwrap();
        }

        // 16 threads × 1000 lookups each, all on the same 4 keys.
        // Each thread checks the returned Vec matches the known constant.
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for tid in 0..16 {
                let entries = &entries;
                handles.push(s.spawn(move || {
                    for i in 0..1000 {
                        let idx = (tid + i) & 3; // 0..=3
                        let out = try_cached_dequant(&entries[idx], QuantFormat::BF16, 4).unwrap();
                        let expected = (idx + 1) as f32;
                        assert!(
                            out.iter().all(|v| (v - expected).abs() < 1e-3),
                            "thread {tid}/iter {i}: got {out:?}, expected {expected}"
                        );
                    }
                }));
            }
            for h in handles {
                h.join().expect("thread panicked");
            }
        });
    }
}
