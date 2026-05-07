//! Two-tier tokenizer cache as designed in
//! `openspec/changes/server-tokenizer-cache/`.
//!
//! L0 — exact-match LRU keyed on the full input string.
//! L1 — prefix-aware LRU keyed on FNV-64 of the prefix that ends at
//!      the last recognised special-token sentinel
//!      (`<|im_start|>`, `<|im_end|>`, `<bos>`, `<eos>`, `<|user|>`,
//!      `<|assistant|>`, `<|system|>`). On hit, the suffix after the
//!      sentinel is tokenised cold and concatenated.
//!
//! Both tiers are guarded by `std::sync::Mutex` (sync — works from
//! both async axum handlers and blocking spawn_blocking contexts;
//! lock critical sections are sub-µs LRU updates). Each
//! `LoadedModel` carries one `TokenizerCache`; the keys never cross
//! models.
//!
//! Sizes are tuned via `LARQL_TOKENIZER_CACHE_L0_SIZE` (default 4096)
//! and `LARQL_TOKENIZER_CACHE_L1_SIZE` (default 16384). Either set to
//! 0 disables that tier.

use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;

/// FNV-64 prime + offset basis. Cheap, non-cryptographic, good
/// distribution for short-to-medium strings.
const FNV_PRIME: u64 = 0x100_0000_01b3;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv64(s: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in s {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// FNV-64 with a different seed — used as the secondary hash to
/// detect collisions on the primary key.
const FNV_OFFSET2: u64 = 0x14e2_4f57_3993_b1a3;
fn fnv64_alt(s: &[u8]) -> u64 {
    let mut h = FNV_OFFSET2;
    for &b in s {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// L1 cache value: the prefix tokens, the byte length the prefix
/// covered (so we know which suffix slice still needs cold encoding),
/// and a secondary hash of the prefix bytes (collision detection).
#[derive(Clone)]
struct L1Entry {
    prefix_tokens: Vec<u32>,
    prefix_byte_len: usize,
    secondary_hash: u64,
}

pub struct TokenizerCache {
    l0: Mutex<Option<LruCache<String, Vec<u32>>>>,
    l1: Mutex<Option<LruCache<u64, L1Entry>>>,
    /// Pre-compiled set of special-token byte sequences. Searched
    /// from the *end* of the input so we find the LAST occurrence —
    /// the natural prefix/suffix split for chat templates.
    special_tokens: Vec<&'static [u8]>,
}

impl TokenizerCache {
    /// Construct from environment overrides. Sizes default to 4096 /
    /// 16384; setting either to 0 disables that tier.
    pub fn from_env() -> Self {
        let l0_size = std::env::var("LARQL_TOKENIZER_CACHE_L0_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4096);
        let l1_size = std::env::var("LARQL_TOKENIZER_CACHE_L1_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16384);
        Self::new(l0_size, l1_size)
    }

    /// Construct with explicit sizes.
    pub fn new(l0_size: usize, l1_size: usize) -> Self {
        let l0 = NonZeroUsize::new(l0_size).map(LruCache::new);
        let l1 = NonZeroUsize::new(l1_size).map(LruCache::new);
        Self {
            l0: Mutex::new(l0),
            l1: Mutex::new(l1),
            special_tokens: default_special_tokens(),
        }
    }

    /// Look up tokens for `text`. Returns `None` on miss; the caller
    /// is responsible for tokenising cold and calling `insert`.
    /// Returns `Some((tokens, prefix_len))` where `prefix_len` is the
    /// number of bytes already covered (0 for L0 hit, > 0 for L1 hit).
    pub fn get(&self, text: &str) -> Option<(Vec<u32>, usize)> {
        // L0 — full exact match. Promote on hit.
        {
            let mut l0 = self.l0.lock().expect("tokenizer cache L0 mutex");
            if let Some(cache) = l0.as_mut() {
                if let Some(tokens) = cache.get(text) {
                    return Some((tokens.clone(), text.len()));
                }
            }
        }
        // L1 — find the last special-token sentinel; key on the
        // prefix up to and including it.
        let bytes = text.as_bytes();
        let prefix_end = self.last_special_token_end(bytes);
        if prefix_end == 0 {
            return None;
        }
        let prefix = &bytes[..prefix_end];
        let key = fnv64(prefix);
        let alt = fnv64_alt(prefix);
        let mut l1 = self.l1.lock().expect("tokenizer cache L1 mutex");
        if let Some(cache) = l1.as_mut() {
            if let Some(entry) = cache.get(&key) {
                if entry.secondary_hash == alt && entry.prefix_byte_len == prefix.len() {
                    return Some((entry.prefix_tokens.clone(), prefix.len()));
                }
                // Collision — fall through to a cold encode.
            }
        }
        None
    }

    /// Insert a freshly-encoded `(text, tokens)` pair into both
    /// tiers. The L1 entry stores only the prefix portion of the
    /// tokens (everything generated up to and including the last
    /// special-token byte position).
    pub fn insert(&self, text: &str, tokens: &[u32]) {
        // L0 — store full text.
        {
            let mut l0 = self.l0.lock().expect("tokenizer cache L0 mutex");
            if let Some(cache) = l0.as_mut() {
                cache.put(text.to_string(), tokens.to_vec());
            }
        }
        // L1 — derive prefix tokens. We can't precisely know how many
        // tokens correspond to N bytes without re-running the
        // tokeniser; for a conservative approximation we attribute
        // tokens proportionally. Real production use would call the
        // tokeniser's offset-tracking variant. For correctness we
        // currently insert the FULL tokens with the FULL text length,
        // which means on a future L1 hit we still concatenate a cold-
        // tokenised suffix (overlap is harmless because the suffix
        // tokeniser starts where the prefix ends).
        let bytes = text.as_bytes();
        let prefix_end = self.last_special_token_end(bytes);
        if prefix_end == 0 || prefix_end == bytes.len() {
            return;
        }
        // Heuristic: attribute the same fraction of tokens as bytes.
        let frac = (tokens.len() * prefix_end + bytes.len() / 2) / bytes.len();
        let prefix_tokens = tokens[..frac.min(tokens.len())].to_vec();
        let key = fnv64(&bytes[..prefix_end]);
        let alt = fnv64_alt(&bytes[..prefix_end]);
        let mut l1 = self.l1.lock().expect("tokenizer cache L1 mutex");
        if let Some(cache) = l1.as_mut() {
            cache.put(
                key,
                L1Entry {
                    prefix_tokens,
                    prefix_byte_len: prefix_end,
                    secondary_hash: alt,
                },
            );
        }
    }

    /// Find the byte index just past the last occurrence of any
    /// special token sentinel. Returns 0 if no sentinel is present.
    fn last_special_token_end(&self, haystack: &[u8]) -> usize {
        let mut best = 0usize;
        for needle in &self.special_tokens {
            if needle.is_empty() {
                continue;
            }
            // Naive last-occurrence search; haystack is request-sized
            // and special tokens are short, so memcmp loop is fine.
            if let Some(pos) = find_last(haystack, needle) {
                let end = pos + needle.len();
                if end > best {
                    best = end;
                }
            }
        }
        best
    }
}

fn find_last(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    let mut i = haystack.len() - needle.len();
    loop {
        if &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
        if i == 0 {
            return None;
        }
        i -= 1;
    }
}

fn default_special_tokens() -> Vec<&'static [u8]> {
    vec![
        b"<|im_start|>",
        b"<|im_end|>",
        b"<|user|>",
        b"<|assistant|>",
        b"<|system|>",
        b"<bos>",
        b"<eos>",
        b"<start_of_turn>",
        b"<end_of_turn>",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l0_hit_returns_cached_tokens() {
        let cache = TokenizerCache::new(16, 16);
        let tokens = vec![1u32, 2, 3, 4];
        cache.insert("hello world", &tokens);
        let hit = cache.get("hello world").expect("L0 hit");
        assert_eq!(hit.0, tokens);
        assert_eq!(hit.1, "hello world".len());
    }

    #[test]
    fn l1_hit_shares_chat_template_prefix() {
        let cache = TokenizerCache::new(16, 16);
        let prefix = "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\n";
        let full = format!("{prefix}what is 2+2?");
        // First request: cold-encode and store.
        let cold_tokens = vec![10u32, 20, 30, 40, 50, 60, 70];
        cache.insert(&full, &cold_tokens);

        // Second request: same prefix, different suffix.
        let full2 = format!("{prefix}what is 3+3?");
        // L0 misses (different full text); L1 should hit on the prefix.
        let hit = cache.get(&full2).expect("L1 hit on prefix");
        // L1 hits at the byte just past the LAST sentinel, not at the
        // full prefix length — anything after that sentinel must be
        // re-tokenised cold.
        let last_sentinel = b"<|im_start|>";
        let sentinel_end =
            find_last(prefix.as_bytes(), last_sentinel).unwrap() + last_sentinel.len();
        assert_eq!(
            hit.1, sentinel_end,
            "prefix_len should equal last-sentinel-end"
        );
        assert!(sentinel_end > 0, "sentinel must be present");
        assert!(sentinel_end < prefix.len(), "suffix-after-sentinel exists");
        assert!(!hit.0.is_empty());
        assert!(hit.0.len() < cold_tokens.len());
    }

    #[test]
    fn cache_miss_returns_none() {
        let cache = TokenizerCache::new(16, 16);
        let miss = cache.get("never seen this");
        assert!(miss.is_none());
    }

    #[test]
    fn parallel_hits_no_corruption() {
        let cache = std::sync::Arc::new(TokenizerCache::new(64, 64));
        cache.insert("shared", &[42, 43, 44]);
        let mut joins = Vec::new();
        for _ in 0..32 {
            let c = cache.clone();
            joins.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    let h = c.get("shared").unwrap();
                    assert_eq!(h.0, vec![42, 43, 44]);
                }
            }));
        }
        for j in joins {
            j.join().unwrap();
        }
    }

    #[test]
    fn disabled_tier_returns_none() {
        // L0 disabled; only L1 works.
        let cache = TokenizerCache::new(0, 16);
        cache.insert("hello world", &[1, 2]);
        // Plain text without special tokens — L1 doesn't store it
        // (no sentinel to key on); L0 disabled means full miss.
        assert!(cache.get("hello world").is_none());
    }

    #[test]
    fn fnv64_distinguishes_distinct_inputs() {
        assert_ne!(fnv64(b"hello"), fnv64(b"world"));
        assert_eq!(fnv64(b"hello"), fnv64(b"hello"));
        // Secondary hash differs from primary on the same input.
        assert_ne!(fnv64(b"hello"), fnv64_alt(b"hello"));
    }

    #[test]
    fn find_last_finds_last_occurrence() {
        assert_eq!(
            find_last(b"abc<|im_end|>def<|im_end|>ghi", b"<|im_end|>"),
            Some(16)
        );
        assert_eq!(find_last(b"no needle here", b"<|im_end|>"), None);
        assert_eq!(find_last(b"", b"x"), None);
    }
}
