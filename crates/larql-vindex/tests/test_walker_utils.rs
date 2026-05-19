#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

use larql_vindex::walker::utils::*;
use larql_vindex::walker::weight_walker::ThresholdCounts;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_round4() {
    assert_eq!(round4(0.123456789), 0.1235);
    assert_eq!(round4(1.0), 1.0);
    assert_eq!(round4(0.0), 0.0);
    assert_eq!(round4(0.00001), 0.0);
    assert_eq!(round4(0.99999), 1.0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_partial_top_k_basic() {
    let data = vec![1.0, 5.0, 3.0, 4.0, 2.0];
    let top3 = partial_top_k(&data, 3);
    assert_eq!(top3.len(), 3);
    assert_eq!(top3[0], (1, 5.0));
    assert_eq!(top3[1], (3, 4.0));
    assert_eq!(top3[2], (2, 3.0));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_partial_top_k_k_larger_than_data() {
    let data = vec![1.0, 2.0];
    let top5 = partial_top_k(&data, 5);
    assert_eq!(top5.len(), 2);
    assert_eq!(top5[0], (1, 2.0));
    assert_eq!(top5[1], (0, 1.0));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_partial_top_k_empty() {
    let data: Vec<f32> = vec![];
    let top = partial_top_k(&data, 3);
    assert!(top.is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_partial_top_k_k_zero() {
    let data = vec![1.0, 2.0, 3.0];
    let top = partial_top_k(&data, 0);
    assert!(top.is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_partial_top_k_column() {
    let matrix = ndarray::array![[1.0f32, 4.0], [3.0, 2.0], [5.0, 6.0]];
    // Column 0: [1, 3, 5] → top-2 = [(2, 5.0), (1, 3.0)]
    let top = partial_top_k_column(&matrix, 0, 2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0], (2, 5.0));
    assert_eq!(top[1], (1, 3.0));

    // Column 1: [4, 2, 6] → top-2 = [(2, 6.0), (0, 4.0)]
    let top = partial_top_k_column(&matrix, 1, 2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0], (2, 6.0));
    assert_eq!(top[1], (0, 4.0));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_top_entities() {
    let mut counts = std::collections::HashMap::new();
    counts.insert("France".to_string(), (10, 5.0)); // avg conf = 0.5
    counts.insert("Germany".to_string(), (20, 8.0)); // avg conf = 0.4
    counts.insert("Japan".to_string(), (5, 4.0)); // avg conf = 0.8

    let top2 = top_entities(&counts, 2);
    assert_eq!(top2.len(), 2);
    // Sorted by count descending
    assert_eq!(top2[0].0, "Germany");
    assert_eq!(top2[0].1, 20);
    assert!((top2[0].2 - 0.4).abs() < 0.001);
    assert_eq!(top2[1].0, "France");
    assert_eq!(top2[1].1, 10);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_top_entities_empty() {
    let counts = std::collections::HashMap::new();
    let top = top_entities(&counts, 5);
    assert!(top.is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn test_count_threshold() {
    let mut t = ThresholdCounts::default();

    count_threshold(&mut t, 0.5);
    assert_eq!(t.t_01, 1);
    assert_eq!(t.t_05, 1);
    assert_eq!(t.t_10, 1);
    assert_eq!(t.t_25, 1);
    assert_eq!(t.t_50, 1);
    assert_eq!(t.t_75, 0);
    assert_eq!(t.t_90, 0);

    count_threshold(&mut t, 0.95);
    assert_eq!(t.t_01, 2);
    assert_eq!(t.t_90, 1);

    count_threshold(&mut t, 0.005);
    assert_eq!(t.t_01, 2); // below 0.01
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_current_date_format() {
    let date = current_date();
    // Should be YYYY-MM-DD format
    assert_eq!(date.len(), 10);
    assert_eq!(date.chars().nth(4), Some('-'));
    assert_eq!(date.chars().nth(7), Some('-'));
    // Year should be reasonable
    let year: u32 = date[0..4].parse().unwrap();
    assert!((2024..=2030).contains(&year));
}
