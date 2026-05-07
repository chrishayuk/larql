use larql_inference::ModelWeights;
use larql_vindex::VectorIndex;

pub(super) fn insert_q4k_layer_tensors(
    weights: &mut ModelWeights,
    index: &VectorIndex,
    layer: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    larql_inference::vindex::insert_q4k_layer_tensors(weights, index, layer)
        .map_err(|err| Box::<dyn std::error::Error>::from(std::io::Error::other(err)))
}

pub(super) fn remove_layer_tensors(weights: &mut ModelWeights, keys: Vec<String>) {
    larql_inference::vindex::remove_layer_tensors(weights, keys);
}
