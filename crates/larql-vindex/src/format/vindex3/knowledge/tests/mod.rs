//! Gates for the V3 query surface at its source: the roles resolve
//! the executed bytes, the annotation implements the V2 extractor's
//! contract, and the KNN statistic matches the V2 gate scan.

mod overlay;

use ndarray::Array1;

use crate::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, G_FFN, G_HIDDEN, G_LAYERS,
    G_VOCAB,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::plan_component_ops;
use crate::tokenizers::Tokenizer;

use super::KnowledgeView;

fn view_for(
    write: impl FnOnce(&std::path::Path),
    name: &str,
    vocab: usize,
) -> (tempfile::TempDir, KnowledgeView) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), container.path(), name);
    let tok_json = larql_inference_free_tokenizer(vocab);
    let tokenizer = Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let view = KnowledgeView::from_plan(&plan, &store, &tokenizer).unwrap();
    (container, view)
}

/// A pre-tokenizer-free WordLevel tokenizer, `[N]` ↔ id N — the same
/// shape `larql_inference::test_utils::synthetic_tokenizer_json`
/// builds, inlined here because larql-vindex cannot depend on
/// larql-inference.
fn larql_inference_free_tokenizer(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

#[test]
fn the_view_binds_the_plans_feature_space() {
    let (_c, view) = view_for(miniature_glimmer, "know-mini", G_VOCAB);
    assert_eq!(view.num_layers(), G_LAYERS);
    assert_eq!(view.loaded_layers(), vec![0, 1]);
    assert_eq!(view.num_features(0), G_FFN);
    assert_eq!(view.max_features(), G_FFN);
    assert_eq!(view.hidden_size(), G_HIDDEN);
    assert_eq!(view.vocab_size(), G_VOCAB);
    let (embed, scale) = view.embedding();
    assert_eq!(embed.shape(), &[G_VOCAB, G_HIDDEN]);
    assert!(scale.is_finite());
    // Out-of-range layers report an empty feature space, not a panic.
    assert_eq!(view.num_features(7), 0);
    assert!(view.feature_meta(7, 0).is_none());
    assert!(view.feature_metas(7).is_none());
    assert!(view.gate_knn(7, &Array1::zeros(G_HIDDEN), 3).is_empty());
}

/// The annotation contract, checked against the arithmetic directly:
/// scores are `embedding · feature_down` and `c_score` is the top
/// logit — the V2 extractor's statement.
#[test]
fn annotations_implement_the_v2_contract() {
    let (_c, view) = view_for(dense_f32_model, "know-dense", 128);
    let meta = view.feature_meta(0, 0).expect("feature 0 is annotated");
    assert_eq!(meta.top_k.len(), 8);
    assert_eq!(meta.top_token, meta.top_k[0].token);
    assert_eq!(meta.top_token_id, meta.top_k[0].token_id);
    assert_eq!(meta.c_score, meta.top_k[0].logit);
    // Descending logits.
    for pair in meta.top_k.windows(2) {
        assert!(pair[0].logit >= pair[1].logit);
    }
    // Token surfaces decode through the tokenizer.
    assert_eq!(meta.top_token, format!("[{}]", meta.top_token_id));
    // Every dense feature is annotated.
    let metas = view.feature_metas(0).unwrap();
    assert!(metas.iter().all(Option::is_some));
}

#[test]
fn batched_promotions_are_the_same_annotation_contract() {
    let (_c, view) = view_for(dense_f32_model, "know-promotion-batch", 128);
    let width = view.num_features(0);
    let features: Vec<_> = (0..width).rev().collect();
    let batched = view.feature_promotions(0, &features).unwrap();
    assert_eq!(batched.len(), features.len());
    for (&feature, actual) in features.iter().zip(batched) {
        let actual = actual.unwrap();
        let expected = view.feature_meta(0, feature).unwrap();
        assert_eq!(actual.top_token_id, expected.top_token_id);
        assert!((actual.c_score - expected.c_score).abs() < 1e-5);
        assert_eq!(
            actual
                .top_k
                .iter()
                .map(|item| item.token_id)
                .collect::<Vec<_>>(),
            expected
                .top_k
                .iter()
                .map(|item| item.token_id)
                .collect::<Vec<_>>()
        );
    }
    assert!(view.feature_promotions(0, &[width]).is_err());
}

/// The KNN statistic is the V2 gate scan's: dot product, ranked by
/// absolute magnitude descending.
#[test]
fn gate_knn_is_the_v2_statistic() {
    let (_c, view) = view_for(miniature_glimmer, "know-knn", G_VOCAB);
    let mut query = Array1::zeros(G_HIDDEN);
    query[0] = 1.0;
    query[3] = -0.5;
    let hits = view.gate_knn(0, &query, G_FFN);
    assert_eq!(hits.len(), G_FFN, "top_k above width returns everything");
    for pair in hits.windows(2) {
        assert!(
            pair[0].1.abs() >= pair[1].1.abs(),
            "ranking must be |score| desc"
        );
    }
    // Truncation keeps the head of the same ranking.
    assert_eq!(view.gate_knn(0, &query, 3), hits[..3].to_vec());

    // The walk is the annotated top-k over the requested layers.
    let trace = view.walk(&query, &[0, 1], 3);
    assert_eq!(trace.layers.len(), 2);
    assert_eq!(trace.layers[0].0, 0);
    assert_eq!(trace.layers[0].1.len(), 3);
    assert_eq!(trace.layers[0].1[0].feature, hits[0].0);
    assert_eq!(trace.layers[0].1[0].gate_score, hits[0].1);
    assert!(!trace.layers[0].1[0].meta.top_token.is_empty());
}

#[test]
fn postings_are_a_non_learned_maskable_access_path() {
    let (_c, view) = view_for(miniature_glimmer, "know-postings", G_VOCAB);
    let source_tokens = [2, 3, 2];
    let index = view
        .build_posting_index(&[0, 1], &source_tokens, 3, 2)
        .unwrap();
    assert_eq!(index.features_per_source(), 3);
    assert_eq!(index.target_top_k(), 2);
    assert_eq!(index.source_tokens(), 2, "source keys are deduplicated");
    assert_eq!(index.total_features(), 2 * G_FFN);
    assert!(index.logical_index_bytes() >= 2 * 2 * 3 * 12);

    let token = source_tokens[0];
    assert_eq!(index.source_postings(token).len(), 2 * 3);
    let unmasked = index.lookup(&[token, token], None);
    assert!(!unmasked.candidates.is_empty());
    assert_eq!(unmasked.total_features, 2 * G_FFN);
    assert_eq!(unmasked.eligible_features, 2 * G_FFN);
    assert!(unmasked.candidate_fraction() > 0.0);
    assert!(unmasked.candidate_fraction() <= 1.0);
    assert_eq!(
        unmasked.logical_bytes_touched,
        (unmasked.source_postings_touched + unmasked.target_postings_touched) * 12
    );
    let prefix = index.lookup_at_width(&[token], None, 1);
    assert_eq!(prefix.source_postings_touched, 2);
    assert!(prefix.candidates.len() <= 2);
    assert!(index.logical_index_bytes_at_width(1) <= index.logical_index_bytes());

    let masked = index.lookup(&[token], Some(&[1]));
    assert_eq!(masked.eligible_features, G_FFN);
    assert!(masked.candidates.iter().all(|address| address.layer == 1));
    assert!(masked.candidates.len() <= unmasked.candidates.len());
    assert_eq!(
        masked.candidate_fraction(),
        masked.candidates.len() as f64 / G_FFN as f64
    );

    let error = view
        .build_posting_index(&[0], &[G_VOCAB as u32], 1, 1)
        .unwrap_err();
    assert!(error.to_string().contains("outside vocabulary"));
}

/// `find_features` implements `VectorIndex::find_features`'s matching
/// rule: case-insensitive substring over the annotation surfaces, with
/// an optional layer filter — the WHERE-clause candidate scan.
#[test]
fn find_features_matches_annotation_surfaces() {
    let (_c, view) = view_for(miniature_glimmer, "know-find", G_VOCAB);

    // Every annotated slot matches an absent entity filter.
    let all = view.find_features(None, None);
    let annotated: usize = view
        .loaded_layers()
        .into_iter()
        .map(|l| {
            (0..view.num_features(l))
                .filter(|&f| view.feature_meta(l, f).is_some())
                .count()
        })
        .sum();
    assert_eq!(all.len(), annotated);
    assert!(!all.is_empty(), "the fixture must annotate features");

    // Entity filter: pick a real top token and find its slot again.
    let (layer, feature) = all[0];
    let token = view.feature_meta(layer, feature).unwrap().top_token;
    let hits = view.find_features(Some(&token), None);
    assert!(
        hits.contains(&(layer, feature)),
        "{token} must match its own slot"
    );

    // Layer filter restricts the scan.
    let layer_hits = view.find_features(None, Some(0));
    assert!(layer_hits.iter().all(|&(l, _)| l == 0));

    // A nonsense entity matches nothing.
    assert!(view
        .find_features(Some("zz-no-such-token"), None)
        .is_empty());
}
