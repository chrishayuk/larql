//! Layer-level intervention adapters.
//!
//! These helpers run the normal FFN, PLE, and layer-scalar tail after replacing
//! or removing one attention component. They are used by mechanistic
//! interpretability and OV/RD experiments without making the canonical layer
//! dispatcher carry every intervention variant.

use super::dot_proj;
use super::layer::{apply_layer_scalar, run_ffn};
use super::ple::apply_per_layer_embedding;
use crate::attention::SharedKV;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use ndarray::{s, Array2};

/// Shared FFN + PLE + layer-scalar tail used by every intervention helper.
///
/// Extracted because each `run_layer_with_*` adapter ran the same three
/// post-attention steps verbatim; threading it through one function keeps the
/// tail's contract (which dump hook fires, whether `run_ffn` captures activation)
/// in one place.
fn finish_layer_tail(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    ple_input: Option<&Array2<f32>>,
) -> Array2<f32> {
    let (h_post_ffn, _) = run_ffn(weights, h_post_attn, layer, ffn, false);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    h_out
}

/// Run a single transformer layer while zeroing selected pre-W_O attention heads.
///
/// This is intended for OV ablation diagnostics: the selected query-head slices
/// are zeroed after GQA and before W_O, then the normal FFN, PLE, and layer
/// scalar path runs unchanged.
#[allow(clippy::type_complexity)]
pub fn run_layer_with_zeroed_pre_o_heads(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    heads: &[usize],
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post_attn, kv_out) = crate::attention::run_attention_block_zero_pre_o_heads(
        weights, h, layer, heads, shared_kv,
    )?;
    if let Some(dir) = crate::forward::dump_config::DumpConfig::get().layer_dir() {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = crate::forward::dump_config::cpu_layer_h_post_attn_path(dir, layer);
        let _ = std::fs::write(&path, &bytes);
    }
    Some((
        finish_layer_tail(weights, &h_post_attn, layer, ffn, ple_input),
        kv_out,
    ))
}

/// Run a single transformer layer while replacing one pre-W_O attention head.
///
/// This supports static-injection gates: a head can be replaced by global,
/// position, prompt-type, or token-role means while the rest of the block runs
/// through the normal residual path.
#[allow(clippy::too_many_arguments)]
pub fn run_layer_with_replaced_pre_o_head(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    head: usize,
    replacement: &Array2<f32>,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post_attn, kv_out) = crate::attention::run_attention_block_replace_pre_o_head(
        weights,
        h,
        layer,
        head,
        replacement,
        shared_kv,
    )?;
    Some((
        finish_layer_tail(weights, &h_post_attn, layer, ffn, ple_input),
        kv_out,
    ))
}

/// Run a layer while first exposing one original pre-W_O head to a mapper, then
/// replacing that head with the mapper's returned value.
///
/// This is the reusable adapter for OV/RD-style experiments: callers can
/// inspect the original `(seq_len, head_dim)` pre-W_O slice and synthesize a
/// replacement, while the engine owns attention recomputation, FFN, PLE,
/// layer-scalar, and shared-KV handling.
#[allow(clippy::too_many_arguments)]
pub fn run_layer_with_mapped_pre_o_head<F>(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    head: usize,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    mut map_head: F,
) -> Option<(Array2<f32>, Option<SharedKV>)>
where
    F: FnMut(&Array2<f32>) -> Option<Array2<f32>>,
{
    let (_, pre_o) =
        crate::attention::run_attention_block_shared_with_pre_o(weights, h, layer, shared_kv)?;
    let head_dim = weights.arch.head_dim_for_layer(layer);
    let start = head.checked_mul(head_dim)?;
    let end = start.checked_add(head_dim)?;
    if end > pre_o.ncols() {
        return None;
    }
    let original_head = pre_o.slice(s![.., start..end]).to_owned();
    let replacement = map_head(&original_head)?;
    if replacement.nrows() != original_head.nrows() || replacement.ncols() != original_head.ncols()
    {
        return None;
    }
    run_layer_with_replaced_pre_o_head(
        weights,
        h,
        layer,
        ffn,
        head,
        &replacement,
        ple_input,
        shared_kv,
    )
}

/// Run a layer while exposing one original pre-W_O head to a mapper that
/// returns a replacement residual-space delta for that head.
///
/// This is the Mode D adapter: the mapper can replace W_O with a residual
/// lookup/add table while the engine still owns attention recomputation, FFN,
/// PLE, layer scalar, and shared-KV behavior.
#[allow(clippy::too_many_arguments)]
pub fn run_layer_with_mapped_head_residual_delta<F>(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    head: usize,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    mut map_head_delta: F,
) -> Option<(Array2<f32>, Option<SharedKV>)>
where
    F: FnMut(&Array2<f32>) -> Option<Array2<f32>>,
{
    let (_, pre_o) =
        crate::attention::run_attention_block_shared_with_pre_o(weights, h, layer, shared_kv)?;
    let head_dim = weights.arch.head_dim_for_layer(layer);
    let start = head.checked_mul(head_dim)?;
    let end = start.checked_add(head_dim)?;
    if end > pre_o.ncols() {
        return None;
    }
    let original_head = pre_o.slice(s![.., start..end]).to_owned();
    let replacement_delta = map_head_delta(&original_head)?;
    if replacement_delta.nrows() != original_head.nrows()
        || replacement_delta.ncols() != weights.hidden_size
    {
        return None;
    }
    run_layer_with_replaced_head_residual_delta(
        weights,
        h,
        layer,
        ffn,
        head,
        &replacement_delta,
        ple_input,
        shared_kv,
    )
}

/// Run a layer while replacing one head's residual-space contribution with the
/// original `pre_W_O @ W_O_head` contribution.
///
/// This is a no-op sanity path for residual-delta replacement: it exercises the
/// same bypass path as Mode D while preserving the original head contribution.
pub fn run_layer_with_original_head_residual_delta(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    head: usize,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (_, pre_o) =
        crate::attention::run_attention_block_shared_with_pre_o(weights, h, layer, shared_kv)?;
    let head_dim = weights.arch.head_dim_for_layer(layer);
    let start = head.checked_mul(head_dim)?;
    let end = start.checked_add(head_dim)?;
    if end > pre_o.ncols() {
        return None;
    }
    let head_out = pre_o.slice(s![.., start..end]);
    let w_o = weights.tensors.get(&weights.arch.attn_o_key(layer))?;
    let w_o_head = w_o.slice(s![.., start..end]);
    let replacement_delta = dot_proj(&head_out, &w_o_head);
    run_layer_with_replaced_head_residual_delta(
        weights,
        h,
        layer,
        ffn,
        head,
        &replacement_delta,
        ple_input,
        shared_kv,
    )
}

/// Run a single transformer layer while subtracting selected pre-W_O head
/// contributions after W_O projection and before the attention residual path.
///
/// This should match [`run_layer_with_zeroed_pre_o_heads`] up to numerical
/// noise, and is used as a diagnostic for W_O block indexing.
pub fn run_layer_with_subtracted_pre_o_heads(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    heads: &[usize],
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post_attn, kv_out) = crate::attention::run_attention_block_subtract_pre_o_heads(
        weights, h, layer, heads, shared_kv,
    )?;
    Some((
        finish_layer_tail(weights, &h_post_attn, layer, ffn, ple_input),
        kv_out,
    ))
}

/// Run a single transformer layer while replacing one attention head's
/// residual-space contribution after W_O projection.
///
/// This is the Mode D validation path: a precomputed lookup/add table can
/// provide `replacement_delta` directly in residual space, bypassing W_O while
/// preserving FFN, PLE, and layer scalar behavior.
#[allow(clippy::too_many_arguments)]
pub fn run_layer_with_replaced_head_residual_delta(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    head: usize,
    replacement_delta: &Array2<f32>,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post_attn, kv_out) = crate::attention::run_attention_block_replace_head_residual_delta(
        weights,
        h,
        layer,
        head,
        replacement_delta,
        shared_kv,
    )?;
    Some((
        finish_layer_tail(weights, &h_post_attn, layer, ffn, ple_input),
        kv_out,
    ))
}

#[cfg(test)]
mod tests {
    //! Coverage matrix:
    //!
    //! - `finish_layer_tail` is exercised transitively by every intervention
    //!   variant — it has no public surface so we don't test it directly.
    //! - The two "mapped" wrappers have happy-path tests (identity mapper)
    //!   and validation-failure tests (mapper returns wrong-shape array,
    //!   mapper returns None, head index out of range).
    //! - The three direct attention-mutation helpers (`zero`, `replaced`,
    //!   `subtracted`) get one happy-path test each plus a no-op test
    //!   (empty heads slice / preserved-residual replacement).
    //! - `original_head_residual_delta` is the no-op sanity path — verify
    //!   it agrees with the standard layer up to FP noise.
    use super::*;
    use crate::ffn::WeightFfn;
    use crate::forward::run_layer_with_ffn;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    fn h(rows: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (rows, hidden),
            (0..rows * hidden)
                .map(|i| (i as f32 + 1.0) * 0.02)
                .collect(),
        )
        .unwrap()
    }

    fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // ── Identity round-trips against the canonical layer dispatcher ──────────

    #[test]
    fn mapped_pre_o_identity_matches_standard_layer() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(3, weights.hidden_size);
        let (baseline, _, _) = run_layer_with_ffn(&weights, &input, 0, &ffn, false, None, None)
            .expect("baseline layer failed");
        let (mapped, _) =
            run_layer_with_mapped_pre_o_head(&weights, &input, 0, &ffn, 0, None, None, |head| {
                Some(head.clone())
            })
            .expect("mapped layer failed");
        assert_eq!(mapped.shape(), baseline.shape());
        assert!(
            max_abs_diff(&mapped, &baseline) < 1e-5,
            "identity pre-W_O mapping drifted"
        );
    }

    #[test]
    fn original_head_residual_delta_matches_standard_layer() {
        // The original-delta path computes pre_W_O[head] @ W_O[head] and
        // routes it through the residual-delta replacement — must equal
        // the unintervened layer up to FP noise.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (baseline, _, _) = run_layer_with_ffn(&weights, &input, 0, &ffn, false, None, None)
            .expect("baseline layer failed");
        let (intervened, _) =
            run_layer_with_original_head_residual_delta(&weights, &input, 0, &ffn, 0, None, None)
                .expect("original-delta layer failed");
        assert_eq!(intervened.shape(), baseline.shape());
        assert!(
            max_abs_diff(&intervened, &baseline) < 1e-4,
            "original residual-delta path drifted from baseline"
        );
    }

    // ── Direct attention-mutation happy paths ─────────────────────────────────

    #[test]
    fn zeroed_pre_o_heads_runs_to_completion_with_empty_list() {
        // Empty `heads` slice — the attention block runs unchanged, the
        // tail produces the same output as the standard dispatcher.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (baseline, _, _) = run_layer_with_ffn(&weights, &input, 0, &ffn, false, None, None)
            .expect("baseline layer failed");
        let (intervened, _) =
            run_layer_with_zeroed_pre_o_heads(&weights, &input, 0, &ffn, &[], None, None)
                .expect("zeroed layer failed");
        assert_eq!(intervened.shape(), baseline.shape());
        assert!(
            max_abs_diff(&intervened, &baseline) < 1e-5,
            "zeroing zero heads must be a no-op"
        );
    }

    #[test]
    fn zeroed_pre_o_heads_changes_output_when_head_targeted() {
        // Zeroing a real head should change the output.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (baseline, _, _) = run_layer_with_ffn(&weights, &input, 0, &ffn, false, None, None)
            .expect("baseline layer failed");
        let (intervened, _) =
            run_layer_with_zeroed_pre_o_heads(&weights, &input, 0, &ffn, &[0], None, None)
                .expect("zeroed layer failed");
        assert_eq!(intervened.shape(), baseline.shape());
        assert!(
            max_abs_diff(&intervened, &baseline) > 1e-5,
            "zeroing head 0 should change the output"
        );
    }

    #[test]
    fn replaced_pre_o_head_with_zero_matches_zeroed_path() {
        // Replacing head 0 with a zero array of the right shape should
        // match the zeroed-heads helper.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let head_dim = weights.arch.head_dim_for_layer(0);
        let replacement = Array2::<f32>::zeros((input.nrows(), head_dim));
        let (replaced, _) = run_layer_with_replaced_pre_o_head(
            &weights,
            &input,
            0,
            &ffn,
            0,
            &replacement,
            None,
            None,
        )
        .expect("replaced layer failed");
        let (zeroed, _) =
            run_layer_with_zeroed_pre_o_heads(&weights, &input, 0, &ffn, &[0], None, None)
                .expect("zeroed layer failed");
        assert!(
            max_abs_diff(&replaced, &zeroed) < 1e-5,
            "replace(head, 0) must equal zero(head)"
        );
    }

    #[test]
    fn subtracted_pre_o_heads_matches_zeroed_path() {
        // The subtract path is a diagnostic for W_O block indexing — it
        // should agree with zeroing the same heads up to FP noise.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (zeroed, _) =
            run_layer_with_zeroed_pre_o_heads(&weights, &input, 0, &ffn, &[0], None, None)
                .expect("zeroed layer failed");
        let (subtracted, _) =
            run_layer_with_subtracted_pre_o_heads(&weights, &input, 0, &ffn, &[0], None, None)
                .expect("subtracted layer failed");
        assert!(
            max_abs_diff(&zeroed, &subtracted) < 1e-4,
            "subtract(head) must agree with zero(head)"
        );
    }

    // ── Mapper validation branches ─────────────────────────────────────────────

    #[test]
    fn mapped_pre_o_head_mapper_returning_none_short_circuits() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let result =
            run_layer_with_mapped_pre_o_head(&weights, &input, 0, &ffn, 0, None, None, |_| None);
        assert!(result.is_none(), "mapper returning None must short-circuit");
    }

    #[test]
    fn mapped_pre_o_head_wrong_shape_returns_none() {
        // Mapper returns an array with the wrong column count — adapter
        // rejects it instead of running attention with a malformed slice.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let result =
            run_layer_with_mapped_pre_o_head(&weights, &input, 0, &ffn, 0, None, None, |_| {
                Some(Array2::<f32>::zeros((1, 1)))
            });
        assert!(result.is_none(), "wrong-shape replacement must be rejected");
    }

    #[test]
    fn mapped_pre_o_head_out_of_range_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        // make_test_weights has 2 q-heads — head 99 is out of range.
        let result =
            run_layer_with_mapped_pre_o_head(&weights, &input, 0, &ffn, 99, None, None, |head| {
                Some(head.clone())
            });
        assert!(result.is_none(), "out-of-range head must be rejected");
    }

    #[test]
    fn mapped_head_residual_delta_identity_matches_baseline() {
        // Identity mapper on the residual-delta path: replace W_O head
        // contribution with the original delta — should equal the
        // unintervened layer.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (baseline, _, _) = run_layer_with_ffn(&weights, &input, 0, &ffn, false, None, None)
            .expect("baseline layer failed");
        let (mapped, _) = run_layer_with_mapped_head_residual_delta(
            &weights,
            &input,
            0,
            &ffn,
            0,
            None,
            None,
            |head| {
                // head is (seq_len, head_dim); compute head @ W_O[head] manually.
                let w_o = weights
                    .tensors
                    .get(&weights.arch.attn_o_key(0))
                    .expect("W_O");
                let head_dim = weights.arch.head_dim_for_layer(0);
                let w_o_head = w_o.slice(s![.., 0..head_dim]);
                Some(super::super::dot_proj(&head.view(), &w_o_head))
            },
        )
        .expect("mapped residual-delta layer failed");
        assert!(
            max_abs_diff(&mapped, &baseline) < 1e-4,
            "identity residual-delta mapping drifted"
        );
    }

    #[test]
    fn mapped_head_residual_delta_mapper_returning_none_short_circuits() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let result = run_layer_with_mapped_head_residual_delta(
            &weights,
            &input,
            0,
            &ffn,
            0,
            None,
            None,
            |_| None,
        );
        assert!(result.is_none());
    }

    #[test]
    fn mapped_head_residual_delta_wrong_hidden_width_returns_none() {
        // Residual-delta replacement must have hidden_size columns.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let result = run_layer_with_mapped_head_residual_delta(
            &weights,
            &input,
            0,
            &ffn,
            0,
            None,
            None,
            |_| Some(Array2::<f32>::zeros((2, 1))), // wrong column count
        );
        assert!(result.is_none());
    }

    #[test]
    fn mapped_head_residual_delta_out_of_range_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let result = run_layer_with_mapped_head_residual_delta(
            &weights,
            &input,
            0,
            &ffn,
            99,
            None,
            None,
            |h| Some(Array2::<f32>::zeros((h.nrows(), weights.hidden_size))),
        );
        assert!(result.is_none());
    }
}
