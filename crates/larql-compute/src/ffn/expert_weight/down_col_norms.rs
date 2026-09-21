//! Static per-intermediate-neuron down-projection norms — the `‖d_i‖` half
//! of R4's contribution formula `|φ(g)·u|·‖d_i‖` (BW12-0, static
//! sub-expert repackability). Unlike the activation half
//! ([`super::ExpertWeightFfn::run_expert_observed`]), this needs no
//! forward pass: `down`'s weights don't depend on the input, only on which
//! (layer, expert) is asked about, so each is computed once and reused
//! across every captured token.
//!
//! GPT-OSS's `down` matrix is `[hidden_out, intermediate_in]`, row-major —
//! one intermediate neuron's contribution vector is a COLUMN (strided
//! across every row), not a row. Computing every column's norm in one pass
//! over the matrix (touch each element exactly once, accumulate into the
//! matching column's running sum) is the cheap way to get all of them,
//! rather than one column at a time.

use larql_models::ModelWeights;

/// `‖down[:, i]‖` for every intermediate neuron `i`, for one (layer, expert).
///
/// `None` when the architecture/layer/expert combination has no resolvable
/// down-projection weight — mirrors `run_expert_observed`'s own `Option`
/// contract: a missing tensor is a refusal, never a zero-norm neuron.
pub fn down_col_norms(weights: &ModelWeights, layer: usize, expert: usize) -> Option<Vec<f32>> {
    let key = weights.arch.expert_ffn_down_key(layer, expert)?;
    let w_down = weights.tensors.get(&key)?;
    let (_hidden_out, intermediate) = w_down.dim();
    let mut sq_sums = vec![0.0f32; intermediate];
    for row in w_down.rows() {
        for (i, &v) in row.iter().enumerate() {
            sq_sums[i] += v * v;
        }
    }
    for s in &mut sq_sums {
        *s = s.sqrt();
    }
    Some(sq_sums)
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::WeightArray;

    const HIDDEN: usize = 3;
    const INTER: usize = 2;

    fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> WeightArray {
        WeightArray::from_shape_fn((rows, cols), |(r, c)| f(r, c))
    }

    /// A minimal GPT-OSS-shaped single-expert fixture carrying only a
    /// `down` tensor — `down_col_norms` reads nothing else, so this stays
    /// deliberately smaller than `expert_weight::mod`'s own
    /// router+gate+up+down fixture rather than reusing (or duplicating)
    /// its full setup.
    fn down_only_weights() -> larql_models::ModelWeights {
        let mut weights = larql_models::test_fixtures::make_test_weights();
        weights.arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "gpt_oss",
            "num_hidden_layers": 1,
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": HIDDEN,
            "num_local_experts": 1,
            "num_experts_per_tok": 1,
            "swiglu_limit": 7.0,
        }));
        weights.hidden_size = HIDDEN;
        weights.intermediate_size = INTER;
        weights.num_layers = 1;

        let down_key = weights.arch.expert_ffn_down_key(0, 0).unwrap();
        // [hidden_out=3, intermediate_in=2]: column 0 = (1,2,3), column 1 = (0,0,4).
        weights.tensors.insert(
            down_key,
            mat(HIDDEN, INTER, |r, c| match (r, c) {
                (0, 0) => 1.0,
                (1, 0) => 2.0,
                (2, 0) => 3.0,
                (2, 1) => 4.0,
                _ => 0.0,
            }),
        );
        weights
    }

    #[test]
    fn matches_hand_computed_norms_on_a_synthetic_expert() {
        let weights = down_only_weights();
        let norms = down_col_norms(&weights, 0, 0).expect("layer 0 expert 0 resolves");
        assert_eq!(norms.len(), INTER);
        // column 0: sqrt(1^2 + 2^2 + 3^2) = sqrt(14); column 1: sqrt(4^2) = 4.
        assert!((norms[0] - 14.0f32.sqrt()).abs() < 1e-5);
        assert!((norms[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn none_for_an_expert_with_no_down_tensor() {
        let weights = down_only_weights();
        assert!(down_col_norms(&weights, 0, 1).is_none());
    }
}
