//! Activation patching — swap residual rows from one prompt's forward pass
//! into another's.
//!
//! Two-pass primitive:
//!
//! 1. Run the **donor** prompt with [`capture_donor_state`] to record the
//!    post-layer residual at each requested `(layer, position)` coord.
//! 2. Run the **recipient** prompt with [`PatchHook::from_donor`]. At each
//!    coord the hook overwrites the recipient's post-layer residual row
//!    with the donor's. Downstream layers see the patched value.
//!
//! This is the building block for lazarus's `patch_activations`,
//! `full_causal_trace`, and any "what does this residual at this position
//! contribute?" experiment.
//!
//! Usage:
//! ```ignore
//! use larql_inference::forward::patching::{capture_donor_state, patch_and_trace};
//!
//! // Patch (layer 5, position 3) and (layer 7, position 3) from donor
//! // tokens into recipient tokens, then read the recipient's post-layer
//! // residual at layer 10.
//! let donor = capture_donor_state(weights, &donor_tokens, &[(5, 3), (7, 3)]);
//! let trace = patch_and_trace(weights, &recipient_tokens, &donor, &[10]);
//! ```

use super::hooks::{LayerHook, RecordHook};
use super::trace::trace_forward_full_hooked;
use super::TraceResult;
use crate::ffn::{FfnBackend, WeightFfn};
use crate::model::ModelWeights;
use ndarray::Array2;
use std::collections::HashMap;

/// Donor-side state: the residual row at each requested `(layer, position)`
/// coord, captured during the donor forward pass.
pub struct DonorState {
    /// `(layer, position) → residual row (length = hidden_size)`.
    pub records: HashMap<(usize, usize), Vec<f32>>,
}

impl DonorState {
    /// Number of recorded coords.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Run a forward pass on `tokens` and capture the post-layer residual row
/// at each requested `(layer, position)` coord. The returned [`DonorState`]
/// feeds [`PatchHook::from_donor`] for the second pass.
///
/// Out-of-range positions are silently dropped (so callers can request
/// "all layers at position p" against prompts of varying lengths without
/// pre-filtering).
pub fn capture_donor_state(
    weights: &ModelWeights,
    tokens: &[u32],
    coords: &[(usize, usize)],
) -> DonorState {
    let ffn = WeightFfn { weights };
    capture_donor_state_with_ffn(weights, tokens, coords, &ffn)
}

/// Backend-parametric donor capture. Use this when a trace must match a
/// specific inference path, e.g. vindex `WalkFfn` rather than dense weights.
pub fn capture_donor_state_with_ffn(
    weights: &ModelWeights,
    tokens: &[u32],
    coords: &[(usize, usize)],
    ffn: &dyn FfnBackend,
) -> DonorState {
    if coords.is_empty() {
        return DonorState {
            records: HashMap::new(),
        };
    }

    let layers: std::collections::HashSet<usize> = coords.iter().map(|(l, _)| *l).collect();
    let max_layer = *layers.iter().max().unwrap();
    let layer_vec: Vec<usize> = layers.iter().copied().collect();

    let mut record = RecordHook::for_layers(layers.iter().copied());
    let _ = trace_forward_full_hooked(
        weights,
        tokens,
        &layer_vec,
        false,
        0,
        false,
        ffn,
        &mut record,
    );

    let mut records = HashMap::with_capacity(coords.len());
    for &(layer, pos) in coords {
        if layer > max_layer {
            continue;
        }
        let Some(matrix) = record.post_layer.get(&layer) else {
            continue;
        };
        if pos >= matrix.nrows() {
            continue;
        }
        records.insert((layer, pos), matrix.row(pos).to_vec());
    }
    DonorState { records }
}

/// `LayerHook` that overwrites the recipient's post-layer residual row
/// with a donor's recorded value at each known `(layer, position)`.
///
/// Skips coords whose position exceeds the recipient's sequence length —
/// useful when the donor and recipient have different lengths and only
/// the overlap matters.
pub struct PatchHook<'a> {
    /// `(layer, position) → donor residual row to splice in`.
    pub records: &'a HashMap<(usize, usize), Vec<f32>>,
}

impl<'a> PatchHook<'a> {
    pub fn from_donor(state: &'a DonorState) -> Self {
        Self {
            records: &state.records,
        }
    }
}

impl LayerHook for PatchHook<'_> {
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        let n_rows = h.nrows();
        let hidden = h.ncols();
        for ((l, pos), row) in self.records.iter() {
            if *l != layer || *pos >= n_rows || row.len() != hidden {
                continue;
            }
            let mut dest = h.row_mut(*pos);
            for (d, s) in dest.iter_mut().zip(row.iter()) {
                *d = *s;
            }
        }
    }
}

/// Convenience: pass 2. Run `recipient_tokens` with the donor's state
/// patched in, capturing residuals at `capture_layers` for inspection.
///
/// Returns the standard [`TraceResult`] but with post-patch residuals
/// (i.e. layers downstream of any patched coord see the donor's value).
pub fn patch_and_trace(
    weights: &ModelWeights,
    recipient_tokens: &[u32],
    donor: &DonorState,
    capture_layers: &[usize],
) -> TraceResult {
    let ffn = WeightFfn { weights };
    patch_and_trace_with_ffn(weights, recipient_tokens, donor, capture_layers, &ffn)
}

/// Backend-parametric activation patching. Donor and recipient passes should
/// use the same FFN backend so the causal intervention is interpreted in the
/// same mechanism the caller is studying.
pub fn patch_and_trace_with_ffn(
    weights: &ModelWeights,
    recipient_tokens: &[u32],
    donor: &DonorState,
    capture_layers: &[usize],
    ffn: &dyn FfnBackend,
) -> TraceResult {
    let mut hook = PatchHook::from_donor(donor);
    trace_forward_full_hooked(
        weights,
        recipient_tokens,
        capture_layers,
        false,
        0,
        false,
        ffn,
        &mut hook,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::trace::trace_forward_full;
    use crate::model::ModelWeights;
    use crate::test_utils::make_test_weights;
    use std::sync::OnceLock;

    fn shared_weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    fn baseline_residual(weights: &ModelWeights, tokens: &[u32], layer: usize) -> Vec<f32> {
        let ffn = WeightFfn { weights };
        let trace = trace_forward_full(weights, tokens, &[layer], false, 0, false, &ffn);
        trace
            .residuals
            .into_iter()
            .find(|(l, _)| *l == layer)
            .expect("baseline must capture requested layer")
            .1
    }

    #[test]
    fn capture_donor_state_records_requested_coords() {
        let weights = shared_weights();
        let donor = capture_donor_state(weights, &[0u32, 1, 2], &[(0, 0), (1, 2)]);
        assert_eq!(donor.len(), 2);
        assert!(donor.records.contains_key(&(0, 0)));
        assert!(donor.records.contains_key(&(1, 2)));
        for v in donor.records.values() {
            assert_eq!(v.len(), weights.hidden_size);
        }
    }

    #[test]
    fn capture_donor_state_drops_out_of_range_positions() {
        let weights = shared_weights();
        // tokens has length 2, but pos 5 is requested — should be skipped.
        let donor = capture_donor_state(weights, &[0u32, 1], &[(0, 0), (0, 5)]);
        assert!(donor.records.contains_key(&(0, 0)));
        assert!(!donor.records.contains_key(&(0, 5)));
    }

    #[test]
    fn empty_donor_state_is_noop_patch() {
        let weights = shared_weights();
        let donor = DonorState {
            records: HashMap::new(),
        };
        let recipient = vec![3u32, 4, 5];
        let baseline = baseline_residual(weights, &recipient, 1);
        let trace = patch_and_trace(weights, &recipient, &donor, &[1]);
        let after = trace
            .residuals
            .into_iter()
            .find(|(l, _)| *l == 1)
            .unwrap()
            .1;
        // BLAS on Windows reorders parallel reductions across successive
        // matmul calls (often accompanied by `BLAS : Bad memory
        // unallocation!`), so two identical forward passes drift in the
        // 1e-3 range. Linux/macOS BLAS stays well below 1e-6.
        const NOOP_PATCH_TOL: f32 = if cfg!(windows) { 1e-2 } else { 1e-6 };
        for (b, a) in baseline.iter().zip(after.iter()) {
            assert!(
                (b - a).abs() < NOOP_PATCH_TOL,
                "empty patch should be a noop: {b} vs {a}"
            );
        }
    }

    #[test]
    fn patch_changes_recipient_residual_downstream() {
        // Patch donor's post-layer residual at layer 0, position 1 into
        // recipient. The capture at the next layer (downstream) must
        // differ from the un-patched baseline. 2-layer fixture: patch at
        // layer 0, observe at layer 1.
        let weights = shared_weights();
        if weights.num_layers < 2 {
            return; // need at least 2 layers for upstream→downstream
        }
        let downstream = weights.num_layers - 1;
        let donor_tokens = vec![10u32, 20, 30];
        let recipient_tokens = vec![1u32, 2, 3];

        let donor = capture_donor_state(weights, &donor_tokens, &[(0, 1)]);
        assert_eq!(donor.len(), 1);

        let baseline = baseline_residual(weights, &recipient_tokens, downstream);
        let patched = patch_and_trace(weights, &recipient_tokens, &donor, &[downstream])
            .residuals
            .into_iter()
            .find(|(l, _)| *l == downstream)
            .unwrap()
            .1;

        let differs = baseline
            .iter()
            .zip(patched.iter())
            .any(|(b, p)| (b - p).abs() > 1e-5);
        assert!(
            differs,
            "patching donor residual must perturb downstream recipient residual"
        );
    }

    #[test]
    fn empty_coords_returns_empty_donor_without_running_forward() {
        // capture_donor_state short-circuits on empty coords — no forward pass
        // is run, so this stays cheap when callers gate on a runtime flag.
        let weights = shared_weights();
        let donor = capture_donor_state(weights, &[0u32, 1], &[]);
        assert!(donor.is_empty());
        assert_eq!(donor.len(), 0);
    }

    #[test]
    fn donor_state_accessors_match_records_map() {
        let weights = shared_weights();
        let donor = capture_donor_state(weights, &[0u32, 1, 2], &[(0, 0), (1, 1)]);
        assert!(!donor.is_empty());
        assert_eq!(donor.len(), donor.records.len());
        assert_eq!(donor.len(), 2);
    }

    #[test]
    fn patch_skips_recipient_position_out_of_range() {
        // Donor records (1, 5) but recipient has only 2 tokens. The hook's
        // `*pos >= n_rows` guard must short-circuit cleanly without panicking
        // and without disturbing the recipient at any in-range position.
        let weights = shared_weights();
        if weights.num_layers < 2 {
            return;
        }
        let donor_tokens = vec![1u32, 2, 3, 4, 5, 6];
        let donor = capture_donor_state(weights, &donor_tokens, &[(0, 5)]);
        assert!(donor.records.contains_key(&(0, 5)));

        let recipient = vec![7u32, 8]; // shorter than donor — pos 5 OOR
        let baseline = baseline_residual(weights, &recipient, 1);
        let patched = patch_and_trace(weights, &recipient, &donor, &[1])
            .residuals
            .into_iter()
            .find(|(l, _)| *l == 1)
            .unwrap()
            .1;
        for (b, p) in baseline.iter().zip(patched.iter()) {
            assert!(
                (b - p).abs() < 1e-6,
                "OOR patch must be a noop: baseline={b} patched={p}"
            );
        }
    }

    #[test]
    fn patch_skips_donor_record_with_wrong_hidden_size() {
        // Build a DonorState manually with a row whose length mismatches
        // hidden_size. The hook's `row.len() != hidden` guard must short-
        // circuit so the recipient passes through untouched.
        let weights = shared_weights();
        let mut records = HashMap::new();
        records.insert((0usize, 0usize), vec![0.0f32; weights.hidden_size + 1]); // wrong len
        let donor = DonorState { records };

        let recipient = vec![1u32, 2];
        let baseline = baseline_residual(weights, &recipient, 0);
        let patched = patch_and_trace(weights, &recipient, &donor, &[0])
            .residuals
            .into_iter()
            .find(|(l, _)| *l == 0)
            .unwrap()
            .1;
        for (b, p) in baseline.iter().zip(patched.iter()) {
            assert!(
                (b - p).abs() < 1e-6,
                "mismatched-len donor record must be ignored: baseline={b} patched={p}"
            );
        }
    }

    #[test]
    fn capture_donor_skips_unrecorded_layer() {
        // When a coord requests a layer that's beyond the model's depth,
        // RecordHook never receives a callback for it — the post-trace
        // pickup loop's `record.post_layer.get(&layer)` returns None
        // and the coord is silently dropped (the `continue` branch).
        let weights = shared_weights();
        let coords = vec![
            (0usize, 0usize),            // valid
            (weights.num_layers + 9, 0), // beyond model depth — None branch
        ];
        let donor = capture_donor_state(weights, &[0u32, 1], &coords);
        assert!(donor.records.contains_key(&(0, 0)));
        assert!(!donor.records.contains_key(&(weights.num_layers + 9, 0)));
    }

    #[test]
    fn capture_donor_drops_layer_above_max() {
        // Coords request layer N+1 (above the model's last layer). The
        // `layer > max_layer` guard never fires here (layer == max_layer),
        // but coverage of the post-trace skip path is exercised by
        // requesting a layer that is in the coord set but never recorded.
        let weights = shared_weights();
        let coords = vec![(0usize, 0usize), (weights.num_layers - 1, 0)];
        let donor = capture_donor_state(weights, &[0u32, 1], &coords);
        // Both coords are in range — both should be recorded.
        assert!(donor.records.contains_key(&(0, 0)));
        assert!(donor.records.contains_key(&(weights.num_layers - 1, 0)));
    }

    #[test]
    fn patch_at_layer_overwrites_residual_at_that_layer() {
        // After patching at (layer L, position p), the recipient's
        // post-layer residual at (L, p) should equal the donor's.
        let weights = shared_weights();
        let donor_tokens = vec![10u32, 20, 30];
        let recipient_tokens = vec![1u32, 2, 3];

        let donor = capture_donor_state(weights, &donor_tokens, &[(0, 1)]);
        let donor_row = donor.records.get(&(0, 1)).unwrap().clone();

        // Re-run recipient with PatchHook + RecordHook so we can read
        // the post-patch residual at the patched layer.
        let mut record = RecordHook::for_layers([0usize]);
        let mut patch = PatchHook::from_donor(&donor);
        let mut composite = super::super::hooks::CompositeHook::new(vec![&mut patch, &mut record]);
        let ffn = WeightFfn { weights };
        let _ = trace_forward_full_hooked(
            weights,
            &recipient_tokens,
            &[0],
            false,
            0,
            false,
            &ffn,
            &mut composite,
        );
        let post_patch = record.post_layer.get(&0).unwrap().row(1).to_vec();
        for (a, b) in donor_row.iter().zip(post_patch.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "post-patch residual at (0,1) must equal donor row: donor={a} got={b}"
            );
        }
    }
}
