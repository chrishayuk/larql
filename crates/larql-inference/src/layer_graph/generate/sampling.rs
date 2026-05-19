//! Token sampling — temperature, top-k, top-p, seedable.
//!
//! Pipeline applied left-to-right:
//!
//! ```text
//! logits  →  temperature scale  →  top-k truncate  →  top-p truncate
//!         →  softmax            →  multinomial draw
//! ```
//!
//! Each filter is independent. [`SamplingConfig::greedy`] (temperature=0,
//! no truncation) returns the argmax — bit-for-bit identical to the
//! pre-existing `argmax` paths so wiring this module in is a no-op for
//! callers that don't opt into sampling.
//!
//! Reproducibility: when [`SamplingConfig::seed`] is set, the same logit
//! vector produces the same token id every call. Useful for evals.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Numeric guard: `temperature <= EPS` is treated as greedy (avoids
/// dividing by zero in the temperature step).
pub const TEMPERATURE_GREEDY_EPS: f32 = 1e-6;

/// Configuration for the next-token sampler.
///
/// Default is greedy decoding — `SamplingConfig::default()` returns the
/// argmax with no RNG and no allocations beyond what was already there.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    /// Softmax temperature. `0.0` (or any value `<= TEMPERATURE_GREEDY_EPS`)
    /// means greedy decoding. Standard non-greedy values are `0.6`–`1.0`.
    pub temperature: f32,
    /// Restrict to the top-k highest-probability tokens (after temperature
    /// scaling). `None` = no top-k filter.
    pub top_k: Option<usize>,
    /// Nucleus threshold — keep the smallest set of tokens whose cumulative
    /// probability exceeds `top_p`. `None` = no top-p filter. Common: `0.9`.
    pub top_p: Option<f32>,
    /// Seed for the RNG. Same seed + same logits = same token. `None` =
    /// non-deterministic (entropy from the OS).
    pub seed: Option<u64>,
    /// OpenAI `frequency_penalty`: subtract `freq * count(token)` from
    /// each candidate's logit before softmax. Penalises tokens that
    /// appear often in the generated text so far. Range typically
    /// `[-2.0, 2.0]` (positive = discourage repetition).
    pub frequency_penalty: f32,
    /// OpenAI `presence_penalty`: subtract `presence * 1` from any
    /// token that's already appeared at least once. Penalises *any*
    /// repeated token regardless of frequency.
    pub presence_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::greedy()
    }
}

impl SamplingConfig {
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// Pure temperature sampling (no truncation).
    pub const fn temperature(t: f32) -> Self {
        Self {
            temperature: t,
            top_k: None,
            top_p: None,
            seed: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    pub fn with_frequency_penalty(mut self, p: f32) -> Self {
        self.frequency_penalty = p;
        self
    }

    pub fn with_presence_penalty(mut self, p: f32) -> Self {
        self.presence_penalty = p;
        self
    }

    /// True iff repetition penalties are active (either field non-zero).
    pub fn has_repetition_penalty(&self) -> bool {
        self.frequency_penalty != 0.0 || self.presence_penalty != 0.0
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// True iff this config does plain argmax (no RNG needed).
    pub fn is_greedy(&self) -> bool {
        self.temperature <= TEMPERATURE_GREEDY_EPS && self.top_k.is_none() && self.top_p.is_none()
    }
}

/// Stateful sampler. Owns RNG state when sampling is non-greedy; for
/// greedy configs `Sampler::new` skips RNG construction entirely so a
/// single sampler instance can be cloned across no-cost greedy decoders.
pub struct Sampler {
    cfg: SamplingConfig,
    rng: Option<StdRng>,
}

impl Sampler {
    pub fn new(cfg: SamplingConfig) -> Self {
        let rng = if cfg.is_greedy() {
            None
        } else {
            Some(match cfg.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            })
        };
        Self { cfg, rng }
    }

    pub fn config(&self) -> SamplingConfig {
        self.cfg
    }

    /// Pick a token id from full-vocab logits. Returns `None` only when
    /// every entry is non-finite or the input is empty.
    pub fn sample(&mut self, logits: &[f32]) -> Option<u32> {
        self.sample_with_history(logits, &[])
    }

    /// Sample with awareness of previously-generated token ids so
    /// repetition penalties (`frequency_penalty`, `presence_penalty`)
    /// can be applied before softmax. Pass `generated: &[]` when no
    /// history is relevant (equivalent to [`Self::sample`]).
    pub fn sample_with_history(&mut self, logits: &[f32], generated: &[u32]) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }
        let penalised: Vec<f32>;
        let logits_ref: &[f32] = if self.cfg.has_repetition_penalty() && !generated.is_empty() {
            penalised = apply_repetition_penalty(logits, generated, self.cfg);
            &penalised
        } else {
            logits
        };
        if self.cfg.is_greedy() {
            return argmax(logits_ref);
        }
        let probs = apply_filters(logits_ref, self.cfg);
        if probs.is_empty() {
            return None;
        }
        let rng = self.rng.as_mut()?;
        Some(multinomial(&probs, rng) as u32)
    }

    /// Pick from a sparse `(id, score)` top-K hit list, used when the
    /// LM-head returns vindex KNN truncated results. Top-k filter from
    /// `cfg.top_k` is clamped to `hits.len()` (the KNN already truncated);
    /// temperature and top-p still apply.
    pub fn sample_from_topk(&mut self, hits: &[(u32, f32)]) -> Option<u32> {
        self.sample_from_topk_with_history(hits, &[])
    }

    /// Top-k sample with repetition-penalty support. The penalty
    /// adjusts the per-hit score in-place before normal filtering, so
    /// hits whose token id has already been emitted slide down the
    /// distribution.
    pub fn sample_from_topk_with_history(
        &mut self,
        hits: &[(u32, f32)],
        generated: &[u32],
    ) -> Option<u32> {
        if hits.is_empty() {
            return None;
        }
        let scored: Vec<f32> = if self.cfg.has_repetition_penalty() && !generated.is_empty() {
            let counts = token_counts(generated);
            hits.iter()
                .map(|(id, s)| {
                    let c = *counts.get(id).unwrap_or(&0);
                    if c == 0 {
                        *s
                    } else {
                        s - self.cfg.frequency_penalty * (c as f32) - self.cfg.presence_penalty
                    }
                })
                .collect()
        } else {
            hits.iter().map(|(_, s)| *s).collect()
        };
        if self.cfg.is_greedy() {
            // Re-pick argmax in case the penalty shifted ordering.
            let (idx, _) = scored
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
            return Some(hits[idx].0);
        }
        let probs = apply_filters(&scored, self.cfg);
        if probs.is_empty() {
            return Some(hits[0].0);
        }
        let rng = self.rng.as_mut()?;
        let pick = multinomial(&probs, rng);
        Some(hits[pick].0)
    }
}

/// Build a `token_id → count` map. Tiny helper used by both penalty
/// paths; allocations dominate here only for very long histories.
fn token_counts(generated: &[u32]) -> std::collections::HashMap<u32, usize> {
    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &id in generated {
        *counts.entry(id).or_insert(0) += 1;
    }
    counts
}

/// Apply OpenAI-style repetition penalties to a full-vocab logit slice.
/// Returns a fresh `Vec<f32>` with the modified logits — leaves the
/// original intact for callers that want to compare or fall back.
fn apply_repetition_penalty(logits: &[f32], generated: &[u32], cfg: SamplingConfig) -> Vec<f32> {
    let counts = token_counts(generated);
    let freq = cfg.frequency_penalty;
    let pres = cfg.presence_penalty;
    let mut out = logits.to_vec();
    for (id, c) in counts {
        let i = id as usize;
        if i >= out.len() {
            continue;
        }
        if !out[i].is_finite() {
            continue;
        }
        out[i] -= freq * (c as f32) + pres;
    }
    out
}

// ── Internals ────────────────────────────────────────────────────────────

fn argmax(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
}

/// Apply temperature → top-k → top-p → softmax. Returns a probability
/// vector the same length as `logits` with filtered entries set to 0.
fn apply_filters(logits: &[f32], cfg: SamplingConfig) -> Vec<f32> {
    let temp = if cfg.temperature > TEMPERATURE_GREEDY_EPS {
        cfg.temperature
    } else {
        1.0
    };
    let mut scaled: Vec<f32> = logits
        .iter()
        .map(|&l| {
            if l.is_finite() {
                l / temp
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();

    if let Some(k) = cfg.top_k {
        keep_top_k(&mut scaled, k);
    }

    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Vec::new();
    }
    let mut probs: Vec<f32> = scaled
        .iter()
        .map(|s| if s.is_finite() { (s - max).exp() } else { 0.0 })
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Vec::new();
    }
    for p in &mut probs {
        *p /= sum;
    }

    if let Some(p_thr) = cfg.top_p {
        keep_top_p(&mut probs, p_thr);
    }
    probs
}

/// Mask all but the top-k entries to `-inf` in place. Cheap when k is
/// small relative to vocab — a single `select_nth_unstable`-equivalent
/// sort would also work but allocates more.
fn keep_top_k(scaled: &mut [f32], k: usize) {
    if k == 0 || k >= scaled.len() {
        return;
    }
    // Find the k-th largest threshold via partial sort.
    let mut copy: Vec<f32> = scaled.iter().copied().filter(|v| v.is_finite()).collect();
    if copy.len() <= k {
        return;
    }
    // Descending nth-element: place the k-th largest at index k-1.
    copy.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let thr = copy[k - 1];
    for v in scaled.iter_mut() {
        if !v.is_finite() || *v < thr {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Keep the smallest set of indices whose cumulative probability ≥ p.
fn keep_top_p(probs: &mut [f32], p_thr: f32) {
    if !(0.0..1.0).contains(&p_thr) {
        return;
    }
    // Sort indices by probability descending.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_unstable_by(|&i, &j| {
        probs[j]
            .partial_cmp(&probs[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f32;
    let mut last_kept = 0usize;
    for (rank, &i) in order.iter().enumerate() {
        cum += probs[i];
        last_kept = rank;
        if cum >= p_thr {
            break;
        }
    }
    let kept: std::collections::HashSet<usize> =
        order.iter().take(last_kept + 1).copied().collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !kept.contains(&i) {
            *p = 0.0;
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Multinomial draw via inverse-CDF on a normalised probability vector.
fn multinomial(probs: &[f32], rng: &mut StdRng) -> usize {
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i;
        }
    }
    // Floating-point sum drift can leave `cum` ~slightly less than 1.
    // Fall through to the last finite entry rather than panicking.
    probs
        .iter()
        .enumerate()
        .rfind(|(_, &p)| p > 0.0)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn logits_3() -> Vec<f32> {
        // argmax = 1 (score 5.0), then 0, then 2.
        vec![3.0, 5.0, 1.0]
    }

    #[test]
    fn greedy_returns_argmax() {
        let mut s = Sampler::new(SamplingConfig::greedy());
        assert_eq!(s.sample(&logits_3()), Some(1));
    }

    #[test]
    fn greedy_ignores_nonfinite() {
        let mut s = Sampler::new(SamplingConfig::greedy());
        let l = vec![f32::NEG_INFINITY, f32::NAN, 0.5, 0.7, f32::NEG_INFINITY];
        assert_eq!(s.sample(&l), Some(3));
    }

    #[test]
    fn empty_logits_returns_none() {
        let mut s = Sampler::new(SamplingConfig::greedy());
        assert_eq!(s.sample(&[]), None);
    }

    #[test]
    fn frequency_penalty_pushes_repeated_token_below_argmax() {
        // Without penalty: argmax = 1 (score 5.0). With a large
        // frequency penalty applied to id 1 after it's been emitted
        // twice, the next-best token (id 0, score 3.0) wins.
        let cfg = SamplingConfig::greedy().with_frequency_penalty(2.0);
        let mut s = Sampler::new(cfg);
        let history = [1u32, 1u32]; // id 1 has count = 2
                                    // Penalty: 5.0 - 2.0 * 2 = 1.0. Now id 0 (3.0) > id 1 (1.0).
        assert_eq!(s.sample_with_history(&logits_3(), &history), Some(0));
    }

    #[test]
    fn presence_penalty_pushes_any_repeated_token() {
        // Presence applies once per id regardless of count.
        let cfg = SamplingConfig::greedy().with_presence_penalty(3.0);
        let mut s = Sampler::new(cfg);
        let history = [1u32]; // id 1 seen once
                              // Penalty: 5.0 - 3.0 = 2.0. Id 0 (3.0) wins.
        assert_eq!(s.sample_with_history(&logits_3(), &history), Some(0));
    }

    #[test]
    fn no_penalty_when_history_is_empty() {
        // Penalty fields set, but history is empty → behaves as plain greedy.
        let cfg = SamplingConfig::greedy()
            .with_frequency_penalty(5.0)
            .with_presence_penalty(5.0);
        let mut s = Sampler::new(cfg);
        assert_eq!(s.sample_with_history(&logits_3(), &[]), Some(1));
    }

    #[test]
    fn topk_repetition_penalty_applies_to_hit_scores() {
        let hits = vec![(1u32, 5.0f32), (0u32, 3.0f32), (2u32, 1.0f32)];
        let cfg = SamplingConfig::greedy().with_frequency_penalty(2.0);
        let mut s = Sampler::new(cfg);
        // Without penalty argmax id is 1; with penalty applied twice
        // (history has two 1s), score 5.0 - 4.0 = 1.0; id 0 wins.
        assert_eq!(s.sample_from_topk_with_history(&hits, &[1, 1]), Some(0));
    }

    #[test]
    fn temperature_seeded_is_reproducible() {
        let cfg = SamplingConfig::temperature(0.8).with_seed(42);
        let mut a = Sampler::new(cfg);
        let mut b = Sampler::new(cfg);
        for _ in 0..32 {
            assert_eq!(a.sample(&logits_3()), b.sample(&logits_3()));
        }
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut s = Sampler::new(SamplingConfig::temperature(0.0).with_seed(1));
        assert_eq!(s.sample(&logits_3()), Some(1));
    }

    #[test]
    fn top_k_one_is_greedy_under_temperature() {
        let mut s = Sampler::new(SamplingConfig::temperature(2.0).with_top_k(1).with_seed(42));
        for _ in 0..16 {
            assert_eq!(s.sample(&logits_3()), Some(1));
        }
    }

    #[test]
    fn top_p_one_keeps_full_distribution() {
        // top_p=1.0 is a no-op (the loop hits cum >= 1.0 only at the last
        // element). Verify by sampling many draws and checking we hit >1
        // distinct token (probabilistic — seeded so deterministic).
        let mut s = Sampler::new(
            SamplingConfig::temperature(1.0)
                .with_top_p(0.999)
                .with_seed(7),
        );
        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            seen.insert(s.sample(&logits_3()).unwrap());
        }
        assert!(seen.len() >= 2);
    }

    #[test]
    fn top_p_low_collapses_to_argmax() {
        // top_p=0.01 keeps only the single highest-prob token, regardless
        // of temperature.
        let mut s = Sampler::new(
            SamplingConfig::temperature(2.0)
                .with_top_p(0.01)
                .with_seed(1),
        );
        for _ in 0..16 {
            assert_eq!(s.sample(&logits_3()), Some(1));
        }
    }

    #[test]
    fn top_k_truncates_choices() {
        // top_k=2 over [3.0, 5.0, 1.0] keeps {0, 1}; index 2 should never sample.
        let mut s = Sampler::new(SamplingConfig::temperature(1.0).with_top_k(2).with_seed(99));
        for _ in 0..200 {
            let id = s.sample(&logits_3()).unwrap();
            assert!(id == 0 || id == 1, "top_k=2 leaked id={id}");
        }
    }

    #[test]
    fn sample_from_topk_greedy() {
        let hits = vec![(7u32, 3.5), (12, 2.1), (3, 1.0)];
        let mut s = Sampler::new(SamplingConfig::greedy());
        assert_eq!(s.sample_from_topk(&hits), Some(7));
    }

    #[test]
    fn sample_from_topk_uses_all_when_no_filters() {
        let hits = vec![(7u32, 3.5), (12, 3.4), (3, 3.3)];
        let mut s = Sampler::new(SamplingConfig::temperature(1.0).with_seed(11));
        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            seen.insert(s.sample_from_topk(&hits).unwrap());
        }
        assert!(seen.len() >= 2);
    }

    #[test]
    fn sample_from_topk_empty() {
        let mut s = Sampler::new(SamplingConfig::greedy());
        assert_eq!(s.sample_from_topk(&[]), None);
    }

    #[test]
    fn config_is_greedy_predicate() {
        assert!(SamplingConfig::greedy().is_greedy());
        assert!(SamplingConfig::temperature(0.0).is_greedy());
        assert!(!SamplingConfig::temperature(0.5).is_greedy());
        assert!(!SamplingConfig::greedy().with_top_p(0.9).is_greedy());
    }
}
