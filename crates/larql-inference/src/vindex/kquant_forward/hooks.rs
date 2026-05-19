use std::collections::HashMap;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::SharedKV;
use crate::forward::embed_tokens_pub;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::{run_layer_with_capture_hooked, LayerHook};

use super::tensors::{insert_q4k_layer_tensors, remove_layer_tensors};

/// Compute final hidden states on a Q4_K/Q6_K vindex while firing a
/// [`LayerHook`] at each layer.
///
/// This is the Q4K/vindex-backed counterpart to
/// `forward::trace_forward_full_hooked`: it keeps the mmap/dequant layer-scope
/// behavior of `predict_kquant_hidden` while exposing pre-layer, post-attention,
/// optional attention-weight/FFN-activation, and post-layer hook points.
pub fn predict_kquant_hidden_hooked(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    capture_activation: bool,
    capture_attention: bool,
    hook: &mut dyn LayerHook,
) -> Result<Array2<f32>, String> {
    if weights.arch.is_hybrid_moe() {
        return Err(
            "predict_kquant_hidden_hooked currently supports dense FFN vindexes only".into(),
        );
    }

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..weights.num_layers {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let ffn_backend = crate::ffn::WeightFfn { weights };
        let step = run_layer_with_capture_hooked(
            weights,
            &h,
            layer,
            &ffn_backend,
            capture_activation,
            capture_attention,
            ple_inputs.get(layer),
            shared_kv,
            hook,
        );

        let Some((h_new, _, _, kv_out)) = step else {
            remove_layer_tensors(weights, inserted);
            return Err(format!("Q4K hooked forward failed at layer {layer}"));
        };
        h = h_new;
        if let Some(kv) = kv_out {
            kv_cache.insert(layer, kv);
        }
        remove_layer_tensors(weights, inserted);
    }

    Ok(h)
}
