//! Coverage for `impl GateIndex for PatchedVindex`.

use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, GateIndex, PatchedVindex, StorageBucket, VectorIndex};
use ndarray::array;

const LAYER: usize = 0;
const FEATURE: usize = 1;
const HIDDEN: usize = 2;
const TOP_K: usize = 2;

fn meta(token: &str, id: u32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.into(),
        top_token_id: id,
        c_score: 1.0,
        top_k: vec![TopKEntry {
            token: token.into(),
            token_id: id,
            logit: 1.0,
        }],
    }
}

fn patched_index() -> PatchedVindex {
    let base = VectorIndex::new(
        vec![Some(array![[1.0, 0.0], [0.0, 1.0]])],
        vec![Some(vec![Some(meta("base0", 0)), Some(meta("base1", 1))])],
        1,
        HIDDEN,
    );
    let mut patched = PatchedVindex::new(base);
    patched.insert_feature(LAYER, FEATURE, vec![2.0, 0.0], meta("patch", 2));
    patched.set_up_vector(LAYER, FEATURE, vec![3.0, 4.0]);
    patched.set_down_vector(LAYER, FEATURE, vec![5.0, 6.0]);
    patched
}

#[test]
fn patched_gate_index_trait_surfaces_overlay_and_base_overrides() {
    let patched = patched_index();
    let gate: &dyn GateIndex = &patched;

    assert_eq!(gate.num_features(LAYER), 2);
    assert_eq!(
        gate.feature_meta(LAYER, FEATURE).unwrap().top_token,
        "patch"
    );
    assert_eq!(gate.gate_override(LAYER, FEATURE).unwrap(), &[2.0, 0.0]);
    assert_eq!(gate.up_override(LAYER, FEATURE).unwrap(), &[3.0, 4.0]);
    assert_eq!(gate.down_override(LAYER, FEATURE).unwrap(), &[5.0, 6.0]);
    assert!(gate.has_overrides_at(LAYER));

    let hits = gate.gate_knn(LAYER, &array![1.0, 0.0], TOP_K);
    assert_eq!(hits[0].0, FEATURE);
    assert_eq!(
        gate.gate_knn_batch(LAYER, &array![[1.0, 0.0], [0.0, 1.0]], TOP_K),
        vec![0, FEATURE]
    );
}

#[test]
fn patched_gate_index_trait_forwards_empty_storage_capabilities() {
    let patched = patched_index();
    let gate: &dyn GateIndex = &patched;

    assert!(!gate.has_down_features());
    assert!(gate.down_feature_vector(LAYER, FEATURE).is_none());
    assert!(gate.down_layer_matrix(LAYER).is_none());
    let scores = gate.gate_scores_batch(LAYER, &array![[1.0, 0.0]]).unwrap();
    assert_eq!(scores.shape(), &[1, 2]);
    assert!(gate
        .gate_scores_batch_backend(LAYER, &array![[1.0, 0.0]], None)
        .is_some());
    assert!(gate.up_layer_matrix(LAYER).is_none());
    assert!(!gate.has_full_mmap_ffn());

    assert!(!gate.has_interleaved());
    assert!(gate.interleaved_gate(LAYER).is_none());
    assert!(gate.interleaved_up(LAYER).is_none());
    assert!(gate.interleaved_down(LAYER).is_none());
    gate.prefetch_interleaved_layer(LAYER);

    assert!(!gate.has_interleaved_q4());
    assert!(gate.interleaved_q4_gate(LAYER).is_none());
    assert!(gate.interleaved_q4_up(LAYER).is_none());
    assert!(gate.interleaved_q4_down(LAYER).is_none());
    assert!(gate.interleaved_q4_mmap_ref().is_none());
    gate.prefetch_interleaved_q4_layer(LAYER);

    assert!(!gate.has_interleaved_kquant());
    assert!(gate.interleaved_kquant_mmap_ref().is_none());
    assert!(gate.interleaved_kquant_layer_data(LAYER).is_none());
    gate.prefetch_interleaved_kquant_layer(LAYER);

    assert!(!gate.has_down_features_kquant());
    assert!(!gate.kquant_down_feature_scaled_add(LAYER, FEATURE, 1.0, &mut [0.0; HIDDEN]));
    assert!(gate.kquant_ffn_layer(LAYER, 0).is_none());
    assert!(!gate.kquant_ffn_row_into(LAYER, 0, FEATURE, &mut [0.0; HIDDEN]));
    assert!(gate
        .kquant_ffn_row_dot(LAYER, 0, FEATURE, &[1.0, 0.0])
        .is_none());
    assert!(!gate.kquant_ffn_row_scaled_add_via_cache(LAYER, 2, FEATURE, 1.0, &mut [0.0; HIDDEN]));
    assert!(!gate.kquant_ffn_row_scaled_add(LAYER, 0, FEATURE, 1.0, &mut [0.0; HIDDEN]));
    assert!(gate
        .kquant_matmul_transb(LAYER, 0, &[1.0, 0.0], 1, None)
        .is_none());

    assert!(!gate.has_fp4_storage());
    assert!(gate
        .fp4_ffn_row_dot(LAYER, 0, FEATURE, &[1.0, 0.0])
        .is_none());
    assert!(!gate.fp4_ffn_row_scaled_add(LAYER, 0, FEATURE, 1.0, &mut [0.0; HIDDEN]));
    assert!(!gate.fp4_ffn_row_into(LAYER, 0, FEATURE, &mut [0.0; HIDDEN]));
    assert_eq!(gate.primary_storage_bucket(), StorageBucket::Exact);
}
