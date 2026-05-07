//! Shared utilities for walker modules.

use super::weight_walker::ThresholdCounts;

/// Decode a single token ID to a trimmed string.
pub fn decode_token(tokenizer: &tokenizers::Tokenizer, id: u32) -> Option<String> {
    crate::tokenizer::decode_token(tokenizer, id)
}

/// Round to 4 decimal places.
pub fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Extract top-N entities by count, with average confidence.
pub fn top_entities(
    counts: &std::collections::HashMap<String, (usize, f64)>,
    n: usize,
) -> Vec<(String, usize, f64)> {
    let mut sorted: Vec<_> = counts
        .iter()
        .map(|(name, (count, sum_conf))| (name.clone(), *count, sum_conf / *count as f64))
        .collect();
    sorted.sort_by_key(|x| std::cmp::Reverse(x.1));
    sorted.truncate(n);
    sorted
}

/// Increment threshold counters for a normalized score.
pub fn count_threshold(t: &mut ThresholdCounts, v: f64) {
    if v >= 0.01 {
        t.t_01 += 1;
    }
    if v >= 0.05 {
        t.t_05 += 1;
    }
    if v >= 0.10 {
        t.t_10 += 1;
    }
    if v >= 0.25 {
        t.t_25 += 1;
    }
    if v >= 0.50 {
        t.t_50 += 1;
    }
    if v >= 0.75 {
        t.t_75 += 1;
    }
    if v >= 0.90 {
        t.t_90 += 1;
    }
}

/// Approximate current date without a chrono dependency.
pub fn current_date() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = now / 86400;
    let year = 1970 + (days / 365);
    let remaining = days % 365;
    let month = remaining / 30 + 1;
    let day = remaining % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
}

// ── Top-K utilities ──

/// Top-k (index, value) from a flat slice using partial sort.
pub fn partial_top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    let k = k.min(indexed.len());
    if k == 0 {
        return vec![];
    }
    if k >= indexed.len() {
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        return indexed;
    }
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed
}

/// Top-k from a matrix column.
pub fn partial_top_k_column(
    matrix: &ndarray::Array2<f32>,
    col: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    let nrows = matrix.shape()[0];
    let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(nrows);
    for i in 0..nrows {
        indexed.push((i, matrix[[i, col]]));
    }

    let k = k.min(indexed.len());
    if k == 0 {
        return vec![];
    }
    if k >= indexed.len() {
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        return indexed;
    }
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::collections::HashMap;

    // ── round4 ────────────────────────────────────────────────────────────────

    #[test]
    fn round4_rounds_to_four_decimal_places() {
        assert_eq!(round4(1.23456789), 1.2346);
        assert_eq!(round4(0.0), 0.0);
        assert_eq!(round4(1.0), 1.0);
    }

    #[test]
    fn round4_preserves_exact_values() {
        assert_eq!(round4(0.1234), 0.1234);
        assert_eq!(round4(-3.5678), -3.5678);
    }

    // ── top_entities ──────────────────────────────────────────────────────────

    #[test]
    fn top_entities_returns_top_n_by_count() {
        let mut counts: HashMap<String, (usize, f64)> = HashMap::new();
        counts.insert("a".into(), (5, 2.5)); // count=5, avg_conf=0.5
        counts.insert("b".into(), (10, 8.0)); // count=10, avg_conf=0.8
        counts.insert("c".into(), (2, 1.0)); // count=2, avg_conf=0.5
        let top = top_entities(&counts, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "b"); // highest count first
        assert_eq!(top[0].1, 10);
        assert_eq!(top[1].0, "a");
    }

    #[test]
    fn top_entities_averages_confidence_correctly() {
        let mut counts: HashMap<String, (usize, f64)> = HashMap::new();
        counts.insert("x".into(), (4, 2.0)); // avg = 0.5
        let top = top_entities(&counts, 1);
        assert!((top[0].2 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn top_entities_empty_map_returns_empty() {
        let counts: HashMap<String, (usize, f64)> = HashMap::new();
        assert!(top_entities(&counts, 5).is_empty());
    }

    #[test]
    fn top_entities_n_larger_than_map_returns_all() {
        let mut counts: HashMap<String, (usize, f64)> = HashMap::new();
        counts.insert("x".into(), (1, 1.0));
        counts.insert("y".into(), (2, 2.0));
        let top = top_entities(&counts, 100);
        assert_eq!(top.len(), 2);
    }

    // ── count_threshold ───────────────────────────────────────────────────────

    fn fresh() -> super::super::weight_walker::ThresholdCounts {
        super::super::weight_walker::ThresholdCounts::default()
    }

    #[test]
    fn count_threshold_increments_all_for_high_value() {
        let mut t = fresh();
        count_threshold(&mut t, 0.95);
        assert_eq!(t.t_01, 1);
        assert_eq!(t.t_05, 1);
        assert_eq!(t.t_10, 1);
        assert_eq!(t.t_25, 1);
        assert_eq!(t.t_50, 1);
        assert_eq!(t.t_75, 1);
        assert_eq!(t.t_90, 1);
    }

    #[test]
    fn count_threshold_increments_only_low_for_small_value() {
        let mut t = fresh();
        count_threshold(&mut t, 0.03);
        assert_eq!(t.t_01, 1);
        assert_eq!(t.t_05, 0);
        assert_eq!(t.t_10, 0);
    }

    #[test]
    fn count_threshold_none_for_zero() {
        let mut t = fresh();
        count_threshold(&mut t, 0.0);
        assert_eq!(t.t_01, 0);
    }

    // ── current_date ──────────────────────────────────────────────────────────

    #[test]
    fn current_date_has_yyyy_mm_dd_format() {
        let d = current_date();
        let parts: Vec<&str> = d.split('-').collect();
        assert_eq!(parts.len(), 3, "expected YYYY-MM-DD, got: {d}");
        assert_eq!(parts[0].len(), 4, "year should be 4 digits");
        assert_eq!(parts[1].len(), 2, "month should be 2 digits");
        assert_eq!(parts[2].len(), 2, "day should be 2 digits");
    }

    // ── partial_top_k ─────────────────────────────────────────────────────────

    #[test]
    fn partial_top_k_returns_k_items_in_desc_order() {
        let data = vec![0.1f32, 0.9, 0.3, 0.7, 0.5];
        let top = partial_top_k(&data, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1); // index of 0.9
        assert_eq!(top[1].0, 3); // index of 0.7
        assert!(top[0].1 >= top[1].1, "should be descending");
        assert!(top[1].1 >= top[2].1);
    }

    #[test]
    fn partial_top_k_zero_k_returns_empty() {
        let data = vec![1.0f32, 2.0, 3.0];
        assert!(partial_top_k(&data, 0).is_empty());
    }

    #[test]
    fn partial_top_k_k_larger_than_data_returns_all_sorted() {
        let data = vec![0.5f32, 0.1, 0.9];
        let top = partial_top_k(&data, 100);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 2); // 0.9 first
    }

    #[test]
    fn partial_top_k_empty_input_returns_empty() {
        assert!(partial_top_k(&[], 5).is_empty());
    }

    // ── partial_top_k_column ──────────────────────────────────────────────────

    #[test]
    fn partial_top_k_column_extracts_correct_column() {
        // 4×3 matrix; column 1 values are [2, 5, 1, 8]
        let data: Vec<f32> = vec![0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 8.0, 0.0];
        let m = Array2::from_shape_vec((4, 3), data).unwrap();
        let top = partial_top_k_column(&m, 1, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 3); // row 3 has value 8
        assert_eq!(top[1].0, 1); // row 1 has value 5
    }

    #[test]
    fn partial_top_k_column_k_zero_returns_empty() {
        let m = Array2::from_elem((4, 2), 1.0f32);
        assert!(partial_top_k_column(&m, 0, 0).is_empty());
    }
}
