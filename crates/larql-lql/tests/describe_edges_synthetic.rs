//! `larql_lql::describe` — the structured DESCRIBE a binding calls — on a
//! hand-built two-layer index whose gate scores are known exactly.

use larql_lql::describe::{describe_edges, layer_bands, DescribedEdge, LayerBand};
use larql_vindex::ndarray::{Array1, Array2};
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};

const NUM_LAYERS: usize = 2;
const HIDDEN: usize = 2;
const LAST_LAYER: usize = NUM_LAYERS - 1;
const ENTITY: &str = "France";

/// Gate scores are `gate · query` with `query = e₀`, so each feature's
/// score is the first component of its gate row.
const LONDON_GATE: f32 = 30.0;
const PARIS_L1_GATE: f32 = 20.0;
const PARIS_L0_GATE: f32 = 10.0;
/// Above DESCRIBE's gate threshold, but the token is the entity itself.
const SELF_GATE: f32 = 9.0;
/// Below DESCRIBE's gate threshold (5.0), so the walk hit is dropped.
const WEAK_GATE: f32 = 2.0;

const LONDON_CONFIDENCE: f32 = 0.9;
const PARIS_L1_CONFIDENCE: f32 = 0.7;

fn meta(token: &str, id: u32, confidence: f32) -> Option<FeatureMeta> {
    Some(FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: confidence,
        top_k: Vec::new(),
    })
}

fn gates(scores: &[f32]) -> Option<Array2<f32>> {
    let mut gate = Array2::zeros((scores.len(), HIDDEN));
    for (feature, score) in scores.iter().enumerate() {
        gate[[feature, 0]] = *score;
    }
    Some(gate)
}

/// Layer 0: Paris (10), London (30), Berlin (2, below threshold).
/// Layer 1: Paris (20), France (9, the entity itself).
fn index() -> VectorIndex {
    VectorIndex::new(
        vec![
            gates(&[PARIS_L0_GATE, LONDON_GATE, WEAK_GATE]),
            gates(&[PARIS_L1_GATE, SELF_GATE]),
        ],
        vec![
            Some(vec![
                meta("Paris", 1, 0.5),
                meta("London", 2, LONDON_CONFIDENCE),
                meta("Berlin", 3, 0.5),
            ]),
            Some(vec![
                meta("Paris", 1, PARIS_L1_CONFIDENCE),
                meta(ENTITY, 4, 0.5),
            ]),
        ],
        NUM_LAYERS,
        HIDDEN,
    )
}

fn config() -> VindexConfig {
    VindexConfig {
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        ..VindexConfig::default()
    }
}

fn query() -> Array1<f32> {
    Array1::from_vec(vec![1.0, 0.0])
}

fn targets(edges: &[DescribedEdge]) -> Vec<&str> {
    edges.iter().map(|e| e.target.as_str()).collect()
}

#[test]
fn edges_are_ranked_by_gate_and_filtered() {
    let edges = describe_edges(&index(), &config(), None, ENTITY, &query(), None);

    // Berlin is below the gate threshold and France is the entity itself.
    assert_eq!(targets(&edges), ["London", "Paris"]);

    let london = &edges[0];
    assert_eq!(london.gate_score, LONDON_GATE);
    assert_eq!((london.layer, london.feature), (0, 1));
    assert_eq!(london.confidence, LONDON_CONFIDENCE);
    assert_eq!((london.layers.as_slice(), london.count), (&[0][..], 1));

    // Paris fires on both layers; the strongest hit (layer 1) is reported.
    let paris = &edges[1];
    assert_eq!(paris.gate_score, PARIS_L1_GATE);
    assert_eq!((paris.layer, paris.feature), (1, 0));
    assert_eq!(paris.confidence, PARIS_L1_CONFIDENCE);
    assert_eq!((paris.layers.as_slice(), paris.count), (&[0, 1][..], 2));

    // No classifier: nothing is labelled.
    assert!(edges.iter().all(|e| e.relation.is_none() && !e.is_probe));
    assert!(edges.iter().all(|e| e.also.is_empty()));
}

#[test]
fn band_all_scans_every_layer_like_no_band() {
    let all = describe_edges(
        &index(),
        &config(),
        None,
        ENTITY,
        &query(),
        Some(LayerBand::All),
    );
    let unbanded = describe_edges(&index(), &config(), None, ENTITY, &query(), None);
    assert_eq!(all, unbanded);
}

#[test]
fn undeclared_family_bands_span_the_whole_model() {
    // No declared bands and no family bands: every band is (0, last).
    let bands = layer_bands(&config());
    let whole = (0, LAST_LAYER);
    assert_eq!(bands.syntax, whole);
    assert_eq!(bands.knowledge, whole);
    assert_eq!(bands.output, whole);
}
