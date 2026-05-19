//! Coverage for the default `FfnRowAccess` / `GateLookup` dispatch contract.

use larql_models::TopKEntry;
use larql_vindex::{
    FeatureMeta, FfnRowAccess, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides,
    QuantizedFfnAccess, StorageBucket,
};
use ndarray::{array, Array1, Array2};

const LAYER: usize = 0;
const GATE_COMPONENT: usize = 0;
const UP_COMPONENT: usize = 1;
const DOWN_COMPONENT: usize = 2;
const BAD_COMPONENT: usize = 99;
const FEATURE: usize = 1;
const WIDTH: usize = 2;
const Q4K_DOT: f32 = 42.0;
const FP4_DOT: f32 = 7.0;

#[derive(Default)]
struct DummyGateIndex {
    interleaved_gate: Option<Array2<f32>>,
    interleaved_up: Option<Array2<f32>>,
    interleaved_down: Option<Array2<f32>>,
    up_matrix: Option<Array2<f32>>,
    down_matrix: Option<Array2<f32>>,
    down_features: Vec<Vec<f32>>,
    has_full_mmap_ffn: bool,
    has_interleaved_q4: bool,
    has_interleaved_kquant: bool,
    has_fp4_storage: bool,
}

impl DummyGateIndex {
    fn native() -> Self {
        Self {
            interleaved_gate: Some(array![[1.0, 0.0], [2.0, 3.0]]),
            interleaved_up: Some(array![[0.0, 1.0], [4.0, 5.0]]),
            interleaved_down: Some(array![[1.0, 1.0], [6.0, 7.0]]),
            up_matrix: Some(array![[8.0, 9.0], [10.0, 11.0]]),
            down_matrix: Some(array![[12.0, 13.0], [14.0, 15.0]]),
            down_features: vec![vec![16.0, 17.0], vec![18.0, 19.0]],
            has_full_mmap_ffn: true,
            ..Self::default()
        }
    }

    fn q4k() -> Self {
        Self {
            has_interleaved_kquant: true,
            ..Self::default()
        }
    }

    fn fp4_with_native() -> Self {
        Self {
            has_fp4_storage: true,
            ..Self::native()
        }
    }
}

impl GateLookup for DummyGateIndex {
    fn gate_knn(&self, _layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        if top_k == 0 {
            return vec![];
        }
        if residual[0] >= 0.0 {
            vec![(1, residual[0])]
        } else {
            vec![(2, residual[0])]
        }
    }

    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        Some(FeatureMeta {
            top_token: format!("L{layer}F{feature}"),
            top_token_id: feature as u32,
            c_score: 1.0,
            top_k: vec![TopKEntry {
                token: "tok".into(),
                token_id: feature as u32,
                logit: 1.0,
            }],
        })
    }

    fn num_features(&self, _layer: usize) -> usize {
        3
    }
}

impl PatchOverrides for DummyGateIndex {}

impl NativeFfnAccess for DummyGateIndex {
    fn down_feature_vector(&self, _layer: usize, feature: usize) -> Option<&[f32]> {
        self.down_features.get(feature).map(Vec::as_slice)
    }

    fn has_down_features(&self) -> bool {
        !self.down_features.is_empty()
    }

    fn down_layer_matrix(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.down_matrix.as_ref().map(Array2::view)
    }

    fn up_layer_matrix(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.up_matrix.as_ref().map(Array2::view)
    }

    fn has_full_mmap_ffn(&self) -> bool {
        self.has_full_mmap_ffn
    }

    fn has_interleaved(&self) -> bool {
        self.interleaved_gate.is_some()
            || self.interleaved_up.is_some()
            || self.interleaved_down.is_some()
    }

    fn interleaved_gate(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_gate.as_ref().map(Array2::view)
    }

    fn interleaved_up(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_up.as_ref().map(Array2::view)
    }

    fn interleaved_down(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_down.as_ref().map(Array2::view)
    }
}

impl QuantizedFfnAccess for DummyGateIndex {
    fn has_interleaved_q4(&self) -> bool {
        self.has_interleaved_q4
    }

    fn has_interleaved_kquant(&self) -> bool {
        self.has_interleaved_kquant
    }

    fn kquant_ffn_row_dot(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _x: &[f32],
    ) -> Option<f32> {
        Some(Q4K_DOT)
    }

    fn kquant_ffn_row_into(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        out: &mut [f32],
    ) -> bool {
        out.copy_from_slice(&[21.0, 22.0]);
        true
    }

    fn kquant_ffn_row_scaled_add(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        out[0] += alpha * 23.0;
        out[1] += alpha * 24.0;
        true
    }

    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        out[0] += alpha * 25.0;
        out[1] += alpha * 26.0;
        true
    }
}

impl Fp4FfnAccess for DummyGateIndex {
    fn has_fp4_storage(&self) -> bool {
        self.has_fp4_storage
    }

    fn fp4_ffn_row_dot(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _x: &[f32],
    ) -> Option<f32> {
        Some(FP4_DOT)
    }

    fn fp4_ffn_row_scaled_add(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        out[0] += alpha * 2.0;
        out[1] += alpha * 3.0;
        true
    }

    fn fp4_ffn_row_into(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        out: &mut [f32],
    ) -> bool {
        out.copy_from_slice(&[4.0, 5.0]);
        true
    }
}

#[test]
fn default_gate_knn_batch_unions_per_row_hits() {
    let idx = DummyGateIndex::default();
    assert_eq!(
        idx.gate_knn_batch(LAYER, &array![[1.0, 0.0], [-1.0, 0.0]], 1),
        vec![1, 2]
    );
    assert!(idx.gate_knn_batch(LAYER, &array![[1.0, 0.0]], 0).is_empty());
}

#[test]
fn native_default_row_dispatch_covers_dot_scaled_add_and_into() {
    let idx = DummyGateIndex::native();
    let x = [1.0, 2.0];

    assert_eq!(
        idx.ffn_row_dot(LAYER, GATE_COMPONENT, FEATURE, &x),
        Some(8.0)
    );
    assert_eq!(
        idx.ffn_row_dot(LAYER, UP_COMPONENT, FEATURE, &x),
        Some(14.0)
    );
    assert_eq!(
        idx.ffn_row_dot(LAYER, DOWN_COMPONENT, FEATURE, &x),
        Some(56.0)
    );
    assert_eq!(idx.ffn_row_dot(LAYER, BAD_COMPONENT, FEATURE, &x), None);
    assert_eq!(
        idx.ffn_row_dot(LAYER, GATE_COMPONENT, FEATURE, &[1.0]),
        None
    );

    let mut out = [1.0, 1.0];
    assert!(idx.ffn_row_scaled_add(LAYER, GATE_COMPONENT, FEATURE, 0.5, &mut out));
    assert_eq!(out, [2.0, 2.5]);

    let mut out = [0.0, 0.0];
    assert!(idx.ffn_row_scaled_add(LAYER, UP_COMPONENT, FEATURE, 1.0, &mut out));
    assert_eq!(out, [4.0, 5.0]);

    let mut out = [0.0, 0.0];
    assert!(idx.ffn_row_scaled_add(LAYER, DOWN_COMPONENT, FEATURE, 1.0, &mut out));
    assert_eq!(out, [18.0, 19.0]);

    let mut row = [0.0; WIDTH];
    assert!(idx.ffn_row_into(LAYER, GATE_COMPONENT, FEATURE, &mut row));
    assert_eq!(row, [2.0, 3.0]);
    assert!(!idx.ffn_row_into(LAYER, BAD_COMPONENT, FEATURE, &mut row));
}

#[test]
fn q4k_default_fallback_covers_row_operations() {
    let idx = DummyGateIndex::q4k();
    let x = [1.0, 2.0];

    assert_eq!(
        idx.ffn_row_dot(LAYER, GATE_COMPONENT, FEATURE, &x),
        Some(Q4K_DOT)
    );

    let mut row = [0.0; WIDTH];
    assert!(idx.ffn_row_into(LAYER, UP_COMPONENT, FEATURE, &mut row));
    assert_eq!(row, [21.0, 22.0]);

    let mut gate_or_up = [0.0; WIDTH];
    assert!(idx.ffn_row_scaled_add(LAYER, UP_COMPONENT, FEATURE, 0.5, &mut gate_or_up));
    assert_eq!(gate_or_up, [11.5, 12.0]);

    let mut down = [0.0; WIDTH];
    assert!(idx.ffn_row_scaled_add(LAYER, DOWN_COMPONENT, FEATURE, 0.5, &mut down));
    assert_eq!(down, [12.5, 13.0]);
}

#[test]
fn fp4_default_dispatch_has_priority_over_native_storage() {
    let idx = DummyGateIndex::fp4_with_native();
    let x = [1.0, 2.0];

    assert_eq!(
        idx.ffn_row_dot(LAYER, GATE_COMPONENT, FEATURE, &x),
        Some(FP4_DOT)
    );

    let mut out = [0.0; WIDTH];
    assert!(idx.ffn_row_scaled_add(LAYER, DOWN_COMPONENT, FEATURE, 2.0, &mut out));
    assert_eq!(out, [4.0, 6.0]);

    let mut row = [0.0; WIDTH];
    assert!(idx.ffn_row_into(LAYER, DOWN_COMPONENT, FEATURE, &mut row));
    assert_eq!(row, [4.0, 5.0]);
}

#[test]
fn primary_storage_bucket_follows_dispatch_priority() {
    assert_eq!(
        DummyGateIndex::default().primary_storage_bucket(),
        StorageBucket::Exact
    );
    assert_eq!(
        DummyGateIndex {
            has_interleaved_q4: true,
            ..DummyGateIndex::default()
        }
        .primary_storage_bucket(),
        StorageBucket::Quantized
    );
    assert_eq!(
        DummyGateIndex::q4k().primary_storage_bucket(),
        StorageBucket::Quantized
    );
    assert_eq!(
        DummyGateIndex::native().primary_storage_bucket(),
        StorageBucket::Exact
    );
    assert_eq!(
        DummyGateIndex::fp4_with_native().primary_storage_bucket(),
        StorageBucket::Fp4
    );
}
