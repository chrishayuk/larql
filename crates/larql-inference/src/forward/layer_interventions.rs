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
    let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, kv_out))
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
    let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, kv_out))
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
    let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, kv_out))
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
    let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, kv_out))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
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
        let max_abs = mapped
            .iter()
            .zip(baseline.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "identity pre-W_O mapping drifted by {max_abs}"
        );
    }
}
