//! FFN tensor naming conventions and helpers for cloning tensors on demand.

use std::collections::HashMap;

use ndarray::ArcArray2;

pub fn detect_ffn_pattern(tensors: &HashMap<String, ArcArray2<f32>>, component: &str) -> String {
    // Multimodal checkpoints (Gemma 4 / Gemma 3 conditional generation) nest
    // the text stack under `language_model` and ALSO carry a vision tower with
    // its own `.layers.N.mlp.{up,down}_proj` and a per-layer-input `gate`.
    // Those must never be chosen: the untyped fallback below picked
    // `layers.N.per_layer_input_gate.weight` for gate and
    // `vision_tower.encoder.layers.N.mlp.up_proj.linear.weight` for up/down on
    // gemma-4-E2B-it, and only failed because the vision tower has no layer
    // 25 — on a layer it does have, compile would have selected the wrong
    // tensors and reported success. Explicit language-model patterns first.
    let patterns: &[&str] = match component {
        "gate" => &[
            "model.language_model.layers.{}.mlp.gate_proj.weight",
            "language_model.model.layers.{}.mlp.gate_proj.weight",
            "model.layers.{}.mlp.gate_proj.weight",
            "layers.{}.ffn.gate.weight",
            "model.layers.{}.feed_forward.gate_proj.weight",
        ],
        "up" => &[
            "model.language_model.layers.{}.mlp.up_proj.weight",
            "language_model.model.layers.{}.mlp.up_proj.weight",
            "model.layers.{}.mlp.up_proj.weight",
            "layers.{}.ffn.up.weight",
            "model.layers.{}.feed_forward.up_proj.weight",
        ],
        "down" => &[
            "model.language_model.layers.{}.mlp.down_proj.weight",
            "language_model.model.layers.{}.mlp.down_proj.weight",
            "model.layers.{}.mlp.down_proj.weight",
            "layers.{}.ffn.down.weight",
            "model.layers.{}.feed_forward.down_proj.weight",
        ],
        _ => &[],
    };

    for pat in patterns {
        let test = pat.replace("{}", "0");
        if tensors.contains_key(&test) {
            return pat.to_string();
        }
    }

    let search = match component {
        "gate" => "gate",
        "up" => "up",
        "down" => "down",
        _ => "",
    };
    // Fallback by substring — restricted to the TEXT FFN. A vision tower, an
    // audio encoder or a per-layer-input projection all contain "gate"/"up"/
    // "down" in their names and are not the FFN being edited.
    let excluded = |k: &str| {
        k.contains("vision") || k.contains("audio") || k.contains("per_layer") || k.contains("embed")
    };
    let mut candidates: Vec<&String> = tensors
        .keys()
        .filter(|k| k.contains(search) && k.contains(".0.") && k.contains("mlp") && !excluded(k))
        .collect();
    candidates.sort();
    if let Some(key) = candidates.first() {
        return key.replace(".0.", ".{}.");
    }

    format!("model.layers.{{}}.mlp.{}_proj.weight", component)
}

pub fn ensure_cloned(
    modified: &mut HashMap<String, ArcArray2<f32>>,
    originals: &HashMap<String, ArcArray2<f32>>,
    key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !modified.contains_key(key) {
        let original = originals
            .get(key)
            .ok_or_else(|| format!("tensor not found: {}", key))?;
        modified.insert(key.to_string(), original.to_owned().into());
    }
    Ok(())
}

pub fn decode_f32_b64(b64: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn tensors(names: &[&str]) -> HashMap<String, ArcArray2<f32>> {
        names
            .iter()
            .map(|n| (n.to_string(), Array2::<f32>::zeros((2, 2)).into_shared()))
            .collect()
    }

    #[test]
    fn gemma4_multimodal_names_resolve_to_the_language_model_ffn_not_vision_or_ple() {
        // The exact trap hit on gemma-4-E2B-it.
        let t = tensors(&[
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.per_layer_input_gate.weight",
            "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
            "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight",
        ]);
        assert_eq!(detect_ffn_pattern(&t, "gate"), "model.language_model.layers.{}.mlp.gate_proj.weight");
        assert_eq!(detect_ffn_pattern(&t, "up"), "model.language_model.layers.{}.mlp.up_proj.weight");
        assert_eq!(detect_ffn_pattern(&t, "down"), "model.language_model.layers.{}.mlp.down_proj.weight");
    }

    #[test]
    fn fallback_never_picks_vision_audio_or_per_layer_tensors() {
        let t = tensors(&[
            "model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
            "model.language_model.layers.0.per_layer_input_gate.weight",
            "odd.prefix.layers.0.mlp.up_proj.weight",
        ]);
        assert_eq!(detect_ffn_pattern(&t, "up"), "odd.prefix.layers.{}.mlp.up_proj.weight");
        // Nothing text-FFN-shaped for gate: the documented default, not the PLE gate.
        assert_eq!(detect_ffn_pattern(&t, "gate"), "model.layers.{}.mlp.gate_proj.weight");
    }

    #[test]
    fn plain_llama_names_still_resolve() {
        let t = tensors(&["model.layers.0.mlp.gate_proj.weight"]);
        assert_eq!(detect_ffn_pattern(&t, "gate"), "model.layers.{}.mlp.gate_proj.weight");
    }
}
