//! Vindex A/B comparison — run the same forward pass against two
//! `VectorIndex` instances and report how much their final logits
//! diverge.
//!
//! Format-agnostic by construction. Works for any pair of loaded
//! vindexes: f32 vs FP4, FP4 vs FP6, Q4K vs FP4, etc. The only thing
//! that varies between runs is the `VectorIndex` the walk kernel
//! dispatches through — everything else (attention weights, lm_head,
//! embeddings, tokenizer) is shared. That isolates the measurement to
//! the storage-format delta.
//!
//! Primary consumer: exp 26 Q2 (FP4 end-to-end correctness) via the
//! `vindex_compare` example. But the library has no FP4-specific
//! behaviour and is ready for any future storage-format A/B.
//!
//! Lifted from the retired `kv-cache-benchmark` crate (2026-05-16); the
//! `real-model` feature gate is gone because all deps (`larql-inference`,
//! `larql-vindex`) are already unconditional in `larql-kv`.

use std::collections::HashMap;

use serde::Serialize;

use larql_inference::attention::SharedKV;
use larql_inference::forward::{embed_tokens_pub, hidden_to_raw_logits, run_layer_with_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::WalkFfn;
use larql_vindex::VectorIndex;

/// Per-comparison knobs. Kept minimal; future options added as fields.
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// K for top-K agreement measurement. `5` by default.
    pub top_k: usize,
    /// Cap prompt length to this many tokens (None = full).
    pub max_seq_len: Option<usize>,
    /// Stop at this many layers (None = all of them).
    pub max_layers: Option<usize>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            max_seq_len: None,
            max_layers: None,
        }
    }
}

/// Metrics for a single prompt comparison.
#[derive(Debug, Clone, Serialize)]
pub struct PromptReport {
    pub prompt: String,
    pub seq_len: usize,
    /// Cosine similarity between reference and candidate logit vectors
    /// at the final position.
    pub logit_cos: f64,
    /// Did argmax(logits_ref) == argmax(logits_cand)?
    pub argmax_match: bool,
    /// Jaccard index of the top-K token-id sets.
    pub top_k_jaccard: f64,
    /// KL(softmax(ref) || softmax(cand)). Symmetric reported separately.
    pub kl_forward: f64,
    /// KL(softmax(cand) || softmax(ref)).
    pub kl_reverse: f64,
    /// Symmetrised KL (mean of forward + reverse).
    pub kl_symmetric: f64,
    /// Argmax token id for each side.
    pub ref_top_token_id: u32,
    pub cand_top_token_id: u32,
    /// Optional human-readable decoded tokens (filled by the CLI, not
    /// the library — we don't want a tokenizer dep in the pure path).
    pub ref_top_token: Option<String>,
    pub cand_top_token: Option<String>,
}

/// Aggregate report across a prompt set.
#[derive(Debug, Clone, Serialize)]
pub struct AggregateReport {
    pub n_prompts: usize,
    pub reference_label: String,
    pub candidate_label: String,
    pub config: ComparisonConfigSerde,
    pub prompts: Vec<PromptReport>,
    /// Fraction of prompts where argmax matches.
    pub argmax_agreement: f64,
    /// Mean top-K Jaccard.
    pub top_k_agreement_mean: f64,
    /// Mean logit cosine similarity.
    pub logit_cos_mean: f64,
    /// Mean / 95th percentile / max symmetric KL.
    pub kl_mean: f64,
    pub kl_p95: f64,
    pub kl_max: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonConfigSerde {
    pub top_k: usize,
    pub max_seq_len: Option<usize>,
    pub max_layers: Option<usize>,
}

impl From<&ComparisonConfig> for ComparisonConfigSerde {
    fn from(c: &ComparisonConfig) -> Self {
        Self {
            top_k: c.top_k,
            max_seq_len: c.max_seq_len,
            max_layers: c.max_layers,
        }
    }
}

/// Run the same forward pass against two vindexes, one prompt per call.
///
/// Returns the final-position logits for each side. Shared model
/// weights, shared tokenisation, identical prefill through every layer
/// — the only axis of variation is which `VectorIndex` backs the walk
/// kernel during the FFN stage.
///
/// The function is entirely format-blind: `WalkFfn::new_unlimited`
/// uses the unified `GateIndex::ffn_row_*` dispatch we wired in the
/// trait refactor, so whichever backend the vindex carries (FP4, Q4K,
/// native f32) automatically fires.
pub fn forward_to_logits(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    config: &ComparisonConfig,
) -> Vec<f32> {
    forward_to_logits_traced(weights, index, token_ids, config).0
}

/// Same as `forward_to_logits` but also returns the per-layer walk-path
/// trace (one `(layer, path_name)` per layer). Enables the CLI
/// `--trace` flag and catches cases where a candidate vindex silently
/// falls through to an unexpected backend — the bug class exp 26 Q2
/// surfaced during development.
pub fn forward_to_logits_traced(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    config: &ComparisonConfig,
) -> (Vec<f32>, Vec<(usize, &'static str)>) {
    let mut h = embed_tokens_pub(weights, token_ids);

    let num_layers = config.max_layers.unwrap_or(weights.num_layers);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let mut trace: Vec<(usize, &'static str)> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        // WalkFfn with dispatch trace enabled. The trace is drained
        // per-layer so we can pin which path fired even when multiple
        // positions are processed.
        let walk_ffn = WalkFfn::new_unlimited(weights, index).with_dispatch_trace();

        if let Some((h_new, _, kv_out)) =
            run_layer_with_ffn(weights, &h, layer, &walk_ffn, false, None, shared_kv)
        {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
            // Surface the first trace entry for this layer (there are
            // seq_len entries at the serial sparse path, but they all
            // report the same name). Missing trace == cache hit or
            // zero-features-dense.
            let entries = walk_ffn.take_dispatch_trace();
            let path = entries.first().map(|e| e.path).unwrap_or("unknown");
            trace.push((layer, path));
        } else {
            break;
        }
    }

    let seq_len = h.shape()[0];
    let last_h = h.slice(ndarray::s![seq_len - 1..seq_len, ..]).to_owned();
    (hidden_to_raw_logits(weights, &last_h), trace)
}

/// Compare two vindexes on a single prompt. Computes logits via
/// `forward_to_logits` on each and then the full set of metrics.
pub fn compare_prompt(
    weights: &ModelWeights,
    reference: &VectorIndex,
    candidate: &VectorIndex,
    prompt: &str,
    token_ids: &[u32],
    config: &ComparisonConfig,
) -> PromptReport {
    let logits_ref = forward_to_logits(weights, reference, token_ids, config);
    let logits_cand = forward_to_logits(weights, candidate, token_ids, config);
    metrics_from_logits(
        prompt,
        token_ids.len(),
        &logits_ref,
        &logits_cand,
        config.top_k,
    )
}

/// Compare a whole prompt set. Returns an `AggregateReport`.
///
/// Tokenisation is the caller's job (pass `token_ids_per_prompt`
/// alongside the prompts). Keeps this library tokenizer-free.
pub fn compare_many(
    weights: &ModelWeights,
    reference: &VectorIndex,
    candidate: &VectorIndex,
    prompts_and_tokens: &[(&str, Vec<u32>)],
    reference_label: &str,
    candidate_label: &str,
    config: &ComparisonConfig,
) -> AggregateReport {
    let mut per_prompt = Vec::with_capacity(prompts_and_tokens.len());
    for (prompt, token_ids) in prompts_and_tokens {
        let mut ids = token_ids.clone();
        if let Some(cap) = config.max_seq_len {
            if ids.len() > cap {
                ids.truncate(cap);
            }
        }
        per_prompt.push(compare_prompt(
            weights, reference, candidate, prompt, &ids, config,
        ));
    }
    aggregate(per_prompt, reference_label, candidate_label, config)
}

// ── Metrics ────────────────────────────────────────────────────────────────

fn metrics_from_logits(
    prompt: &str,
    seq_len: usize,
    logits_ref: &[f32],
    logits_cand: &[f32],
    top_k: usize,
) -> PromptReport {
    assert_eq!(
        logits_ref.len(),
        logits_cand.len(),
        "logit vectors must have the same vocab size"
    );

    let argmax_ref = argmax(logits_ref);
    let argmax_cand = argmax(logits_cand);
    let cos = cosine(logits_ref, logits_cand);

    let top_ref = top_k_ids(logits_ref, top_k);
    let top_cand = top_k_ids(logits_cand, top_k);
    let jac = jaccard(&top_ref, &top_cand);

    let probs_ref = softmax(logits_ref);
    let probs_cand = softmax(logits_cand);
    let kl_forward = kl_divergence(&probs_ref, &probs_cand);
    let kl_reverse = kl_divergence(&probs_cand, &probs_ref);
    let kl_sym = 0.5 * (kl_forward + kl_reverse);

    PromptReport {
        prompt: prompt.to_string(),
        seq_len,
        logit_cos: cos,
        argmax_match: argmax_ref == argmax_cand,
        top_k_jaccard: jac,
        kl_forward,
        kl_reverse,
        kl_symmetric: kl_sym,
        ref_top_token_id: argmax_ref,
        cand_top_token_id: argmax_cand,
        ref_top_token: None,
        cand_top_token: None,
    }
}

fn aggregate(
    prompts: Vec<PromptReport>,
    reference_label: &str,
    candidate_label: &str,
    config: &ComparisonConfig,
) -> AggregateReport {
    let n = prompts.len();
    if n == 0 {
        return AggregateReport {
            n_prompts: 0,
            reference_label: reference_label.to_string(),
            candidate_label: candidate_label.to_string(),
            config: config.into(),
            prompts,
            argmax_agreement: f64::NAN,
            top_k_agreement_mean: f64::NAN,
            logit_cos_mean: f64::NAN,
            kl_mean: f64::NAN,
            kl_p95: f64::NAN,
            kl_max: f64::NAN,
        };
    }

    let argmax_hits = prompts.iter().filter(|p| p.argmax_match).count() as f64;
    let top_k_mean = prompts.iter().map(|p| p.top_k_jaccard).sum::<f64>() / n as f64;
    let cos_mean = prompts.iter().map(|p| p.logit_cos).sum::<f64>() / n as f64;

    let mut kls: Vec<f64> = prompts.iter().map(|p| p.kl_symmetric).collect();
    kls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let kl_mean = kls.iter().sum::<f64>() / n as f64;
    let kl_p95 = percentile(&kls, 0.95);
    let kl_max = *kls.last().unwrap_or(&f64::NAN);

    AggregateReport {
        n_prompts: n,
        reference_label: reference_label.to_string(),
        candidate_label: candidate_label.to_string(),
        config: config.into(),
        prompts,
        argmax_agreement: argmax_hits / n as f64,
        top_k_agreement_mean: top_k_mean,
        logit_cos_mean: cos_mean,
        kl_mean,
        kl_p95,
        kl_max,
    }
}

// ── Numeric helpers ────────────────────────────────────────────────────────

fn argmax(xs: &[f32]) -> u32 {
    let mut idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best {
            best = v;
            idx = i;
        }
    }
    idx as u32
}

fn top_k_ids(xs: &[f32], k: usize) -> Vec<u32> {
    let k = k.min(xs.len());
    let mut indexed: Vec<(usize, f32)> = xs.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<u32> = indexed[..k].iter().map(|(i, _)| *i as u32).collect();
    top.sort_unstable();
    top
}

fn jaccard(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let sa: std::collections::BTreeSet<u32> = a.iter().copied().collect();
    let sb: std::collections::BTreeSet<u32> = b.iter().copied().collect();
    let intersect = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        1.0
    } else {
        intersect / union
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        num += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let denom = (na.sqrt()) * (nb.sqrt());
    if denom == 0.0 {
        1.0
    } else {
        num / denom
    }
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&v| ((v - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    // KL(p || q) = Σ p_i * log(p_i / q_i). Skip p_i == 0 (by
    // convention 0 log 0 = 0). Guard against q_i == 0 with a floor.
    const EPS: f64 = 1e-12;
    let mut kl = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi <= 0.0 {
            continue;
        }
        let qi_safe = qi.max(EPS);
        kl += pi * (pi.ln() - qi_safe.ln());
    }
    kl
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_finds_max() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0, -5.0]), 1);
        assert_eq!(argmax(&[-1.0, -3.0, -2.0]), 0);
    }

    #[test]
    fn top_k_ids_returns_correct_indices() {
        // Top-3 by value: idx 1 (3.0), idx 2 (2.0), idx 0 (1.0).
        let logits = [1.0, 3.0, 2.0, -5.0, 0.5];
        let top = top_k_ids(&logits, 3);
        assert_eq!(top.len(), 3);
        // Results are sorted by id; set-equality with {0, 1, 2}.
        let expected: std::collections::BTreeSet<u32> = [0u32, 1, 2].into_iter().collect();
        let got: std::collections::BTreeSet<u32> = top.into_iter().collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn jaccard_full_overlap_equals_one() {
        assert_eq!(jaccard(&[1, 2, 3], &[1, 2, 3]), 1.0);
    }

    #[test]
    fn jaccard_no_overlap_equals_zero() {
        assert_eq!(jaccard(&[1, 2], &[3, 4]), 0.0);
    }

    #[test]
    fn jaccard_partial() {
        // {1,2,3} ∩ {2,3,4} = {2,3}; ∪ = {1,2,3,4}; jac = 2/4 = 0.5.
        assert!((jaccard(&[1, 2, 3], &[2, 3, 4]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!((cosine(&a, &b) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0f32, 2.0, 3.0]);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn kl_identical_is_zero() {
        let p = softmax(&[1.0f32, 2.0, 3.0]);
        assert!(kl_divergence(&p, &p).abs() < 1e-9);
    }

    #[test]
    fn kl_is_nonnegative() {
        let p = softmax(&[1.0f32, 2.0, 3.0]);
        let q = softmax(&[3.0f32, 1.0, 2.0]);
        let kl = kl_divergence(&p, &q);
        assert!(kl >= 0.0, "KL must be non-negative, got {kl}");
    }

    #[test]
    fn aggregate_handles_empty_gracefully() {
        let r = aggregate(vec![], "ref", "cand", &ComparisonConfig::default());
        assert_eq!(r.n_prompts, 0);
        assert!(r.argmax_agreement.is_nan());
    }

    #[test]
    fn aggregate_computes_means() {
        // Two prompts: one argmax match, one argmax miss. Expected
        // argmax_agreement = 0.5.
        let prompts = vec![
            PromptReport {
                prompt: "a".into(),
                seq_len: 1,
                logit_cos: 0.9,
                argmax_match: true,
                top_k_jaccard: 0.8,
                kl_forward: 0.01,
                kl_reverse: 0.01,
                kl_symmetric: 0.01,
                ref_top_token_id: 42,
                cand_top_token_id: 42,
                ref_top_token: None,
                cand_top_token: None,
            },
            PromptReport {
                prompt: "b".into(),
                seq_len: 2,
                logit_cos: 0.7,
                argmax_match: false,
                top_k_jaccard: 0.4,
                kl_forward: 0.05,
                kl_reverse: 0.05,
                kl_symmetric: 0.05,
                ref_top_token_id: 1,
                cand_top_token_id: 7,
                ref_top_token: None,
                cand_top_token: None,
            },
        ];
        let r = aggregate(prompts, "r", "c", &ComparisonConfig::default());
        assert_eq!(r.n_prompts, 2);
        assert!((r.argmax_agreement - 0.5).abs() < 1e-9);
        assert!((r.top_k_agreement_mean - 0.6).abs() < 1e-9);
        assert!((r.logit_cos_mean - 0.8).abs() < 1e-9);
        assert!((r.kl_mean - 0.03).abs() < 1e-9);
    }

    #[test]
    fn percentile_handles_edges() {
        let sorted = [0.1, 0.2, 0.3, 0.4, 0.5];
        assert_eq!(percentile(&sorted, 0.0), 0.1);
        assert_eq!(percentile(&sorted, 1.0), 0.5);
        // p95 on 5 elements → round((5-1)*0.95) = round(3.8) = 4 → sorted[4] = 0.5.
        assert_eq!(percentile(&sorted, 0.95), 0.5);
    }

    /// `forward_to_logits` against synthetic weights + vindex.
    /// Drives the layer loop body (lines 152-179) and the final
    /// logits projection (181-184).
    #[test]
    fn forward_to_logits_runs_against_synthetic_weights() {
        let weights = larql_inference::test_utils::make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let config = ComparisonConfig::default();
        let logits = forward_to_logits(&weights, &index, &[0u32, 1, 2], &config);
        assert_eq!(logits.len(), weights.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_to_logits_traced_returns_per_layer_dispatch_path() {
        let weights = larql_inference::test_utils::make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let config = ComparisonConfig::default();
        let (logits, trace) = forward_to_logits_traced(&weights, &index, &[0u32, 1], &config);
        assert_eq!(logits.len(), weights.vocab_size);
        assert!(!trace.is_empty(), "trace should record at least one layer");
        // Every layer entry pairs (layer, path); path string is non-empty.
        for (layer, path) in &trace {
            assert!(*layer < weights.num_layers);
            assert!(!path.is_empty());
        }
    }

    #[test]
    fn forward_to_logits_max_layers_truncates() {
        let weights = larql_inference::test_utils::make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let config = ComparisonConfig {
            max_layers: Some(1),
            ..ComparisonConfig::default()
        };
        let (_logits, trace) = forward_to_logits_traced(&weights, &index, &[0u32], &config);
        assert!(
            trace.len() <= 1,
            "max_layers=1 should bound trace, got {}",
            trace.len()
        );
    }

    #[test]
    fn compare_prompt_against_self_gives_perfect_agreement() {
        let weights = larql_inference::test_utils::make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let config = ComparisonConfig::default();
        let report = compare_prompt(&weights, &index, &index, "hello", &[0u32, 1], &config);
        assert!(report.argmax_match);
        assert!((report.logit_cos - 1.0).abs() < 1e-4);
        assert!((report.top_k_jaccard - 1.0).abs() < 1e-9);
        assert!(report.kl_symmetric < 1e-4);
        assert_eq!(report.ref_top_token_id, report.cand_top_token_id);
    }

    #[test]
    fn compare_many_against_self_aggregates_argmax_agreement() {
        let weights = larql_inference::test_utils::make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let config = ComparisonConfig::default();
        let prompts: &[(&str, Vec<u32>)] = &[("a", vec![0u32, 1]), ("b", vec![1u32, 2])];
        let agg = compare_many(&weights, &index, &index, prompts, "ref", "cand", &config);
        assert_eq!(agg.n_prompts, 2);
        assert!((agg.argmax_agreement - 1.0).abs() < 1e-9);
        assert!((agg.top_k_agreement_mean - 1.0).abs() < 1e-9);
        assert_eq!(agg.prompts.len(), 2);
    }

    #[test]
    fn compare_many_respects_max_seq_len_truncation() {
        let weights = larql_inference::test_utils::make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let config = ComparisonConfig {
            max_seq_len: Some(1),
            ..ComparisonConfig::default()
        };
        let prompts: &[(&str, Vec<u32>)] = &[("long", vec![0u32, 1, 2, 3])];
        let agg = compare_many(&weights, &index, &index, prompts, "ref", "cand", &config);
        // The first (and only) prompt should report seq_len=1 because
        // max_seq_len capped it to 1 token before forward_to_logits ran.
        assert_eq!(agg.prompts[0].seq_len, 1);
    }

    #[test]
    fn comparison_config_default_and_from_round_trip() {
        let cfg = ComparisonConfig::default();
        let summary: ComparisonConfigSerde = (&cfg).into();
        assert_eq!(summary.top_k, cfg.top_k);
        assert_eq!(summary.max_seq_len, cfg.max_seq_len);
        assert_eq!(summary.max_layers, cfg.max_layers);
    }
}
