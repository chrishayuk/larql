//! `larql shannon block-norms` — BW12-0's static half.
//!
//! Dumps [`larql_compute::ffn::expert_weight::down_col_norms::down_col_norms`]
//! for every `(layer, expert)` a loaded MoE model has, through the SAME
//! `LARQL_MOE_BLOCK_CONTRIB_TRACE` sink the activation captures use
//! (`block_contrib_trace::record`) — one chunk per `(layer, expert)`,
//! `n_tokens = 1`, the single row being that expert's down-column norms.
//!
//! Run once per model, with `LARQL_MOE_BLOCK_CONTRIB_TRACE` pointed at a
//! DIFFERENT path than any activation capture: this is a static precompute
//! (no forward pass, no tokens), not a token-indexed trace, and mixing the
//! two into one file would make a reader unable to tell a norms-chunk from
//! an activation-chunk apart (both are just `(layer, expert, n_tokens,
//! intermediate)` + raw floats).

use larql_compute::ffn::expert_weight::{block_contrib_trace, down_col_norms::down_col_norms};
use ndarray::Array2;

use super::super::shannon_cmd;
use super::BlockNormsArgs;

pub fn run_block_norms(args: BlockNormsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = shannon_cmd::load_model(&args.model)?;
    let weights = model.weights();
    if !weights.arch.is_moe() {
        return Err(format!(
            "{}: not a MoE architecture — no expert down-projections to norm",
            args.model
        )
        .into());
    }
    let num_experts = weights.arch.num_experts();

    let mut written = 0usize;
    for layer in 0..weights.num_layers {
        for expert in 0..num_experts {
            let Some(norms) = down_col_norms(weights, layer, expert) else {
                continue;
            };
            let intermediate = norms.len();
            let row = Array2::from_shape_vec((1, intermediate), norms)?;
            block_contrib_trace::record(layer, expert, &row);
            written += 1;
        }
    }

    if written == 0 {
        eprintln!(
            "warning: wrote nothing — is LARQL_MOE_BLOCK_CONTRIB_TRACE set, and does this \
             model resolve any expert down-projection?"
        );
    }
    println!(
        "wrote down-column norms for {written} (layer, expert) pair(s) across {} layers × {num_experts} experts",
        weights.num_layers
    );
    Ok(())
}
