use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{encode_prompt, hidden_to_raw_logits};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::input::{load_prompts, parse_head_spec};
use super::metrics::{kl_logp, log_softmax, max_abs_diff, mean};
use super::reports::{SanityCheckReport, SanityHeadReport, SanityPromptReport};
use super::types::HeadId;
use super::zero_ablate::forward_q4k_zero_pre_o_head;

#[derive(Args)]
pub(super) struct SanityCheckArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 0:4,0:6.
    #[arg(long)]
    heads: String,

    /// Limit prompts for bounded sanity runs.
    #[arg(long)]
    max_prompts: Option<usize>,
}

#[derive(Debug)]
struct SanityHeadAccumulator {
    prompts: Vec<SanityPromptReport>,
}

impl SanityHeadAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
        }
    }

    fn add(&mut self, prompt: SanityPromptReport) {
        self.prompts.push(prompt);
    }

    fn finish(self, head: HeadId) -> SanityHeadReport {
        let noop_kls: Vec<f64> = self.prompts.iter().map(|p| p.noop_kl).collect();
        let residual_delta_noop_kls: Vec<f64> = self
            .prompts
            .iter()
            .map(|p| p.residual_delta_noop_kl)
            .collect();
        let zero_subtract_kls: Vec<f64> = self.prompts.iter().map(|p| p.zero_subtract_kl).collect();
        SanityHeadReport {
            layer: head.layer,
            head: head.head,
            prompts: self.prompts.len(),
            noop_mean_kl: mean(&noop_kls),
            noop_max_kl: noop_kls.iter().copied().fold(0.0, f64::max),
            noop_max_abs_logit_diff: self
                .prompts
                .iter()
                .map(|p| p.noop_max_abs_logit_diff)
                .fold(0.0, f64::max),
            residual_delta_noop_mean_kl: mean(&residual_delta_noop_kls),
            residual_delta_noop_max_kl: residual_delta_noop_kls.iter().copied().fold(0.0, f64::max),
            residual_delta_noop_max_abs_logit_diff: self
                .prompts
                .iter()
                .map(|p| p.residual_delta_noop_max_abs_logit_diff)
                .fold(0.0, f64::max),
            zero_subtract_mean_kl: mean(&zero_subtract_kls),
            zero_subtract_max_kl: zero_subtract_kls.iter().copied().fold(0.0, f64::max),
            zero_subtract_max_abs_logit_diff: self
                .prompts
                .iter()
                .map(|p| p.zero_subtract_max_abs_logit_diff)
                .fold(0.0, f64::max),
            per_prompt: self.prompts,
        }
    }
}

pub(super) fn run_sanity_check(args: SanityCheckArgs) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    eprintln!("Loading vindex: {}", args.index.display());
    let start = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_kquant(&args.index)?;
    index.load_interleaved_kquant(&args.index)?;
    let mut weights = load_model_weights_kquant(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;
    if weights.arch.is_hybrid_moe() {
        return Err("ov-rd sanity-check currently supports dense FFN vindexes only".into());
    }
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );

    let selected_heads = parse_head_spec(&args.heads)?;
    if selected_heads.is_empty() {
        return Err("no heads selected for sanity check".into());
    }
    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("Prompts: {}", prompts.len());

    let mut accumulators: Vec<SanityHeadAccumulator> = selected_heads
        .iter()
        .map(|_| SanityHeadAccumulator::new())
        .collect();

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", prompt_idx + 1, prompts.len(), label);

        let token_ids = encode_prompt(&tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");

        let baseline_hidden =
            larql_inference::vindex::predict_kquant_hidden(&mut weights, &token_ids, &index, None);
        let baseline_logits = final_logits(&weights, &baseline_hidden);
        let baseline_logp = log_softmax(&baseline_logits);

        for (idx, head) in selected_heads.iter().copied().enumerate() {
            let noop_hidden =
                forward_q4k_noop_replace_pre_o_head(&mut weights, &token_ids, &index, head)?;
            let noop_logits = final_logits(&weights, &noop_hidden);
            let noop_logp = log_softmax(&noop_logits);

            let residual_delta_noop_hidden = forward_q4k_noop_replace_head_residual_delta(
                &mut weights,
                &token_ids,
                &index,
                head,
            )?;
            let residual_delta_noop_logits = final_logits(&weights, &residual_delta_noop_hidden);
            let residual_delta_noop_logp = log_softmax(&residual_delta_noop_logits);

            let zero_hidden = forward_q4k_zero_pre_o_head(&mut weights, &token_ids, &index, head)?;
            let zero_logits = final_logits(&weights, &zero_hidden);
            let zero_logp = log_softmax(&zero_logits);

            let subtract_hidden =
                forward_q4k_subtract_pre_o_head(&mut weights, &token_ids, &index, head)?;
            let subtract_logits = final_logits(&weights, &subtract_hidden);
            let subtract_logp = log_softmax(&subtract_logits);

            accumulators[idx].add(SanityPromptReport {
                id: label.to_string(),
                stratum: stratum.to_string(),
                noop_kl: kl_logp(&baseline_logp, &noop_logp),
                noop_max_abs_logit_diff: max_abs_diff(&baseline_logits, &noop_logits),
                residual_delta_noop_kl: kl_logp(&baseline_logp, &residual_delta_noop_logp),
                residual_delta_noop_max_abs_logit_diff: max_abs_diff(
                    &baseline_logits,
                    &residual_delta_noop_logits,
                ),
                zero_subtract_kl: kl_logp(&zero_logp, &subtract_logp),
                zero_subtract_max_abs_logit_diff: max_abs_diff(&zero_logits, &subtract_logits),
            });
        }
    }

    let heads = selected_heads
        .iter()
        .copied()
        .zip(accumulators)
        .map(|(head, acc)| acc.finish(head))
        .collect();
    let report = SanityCheckReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        selected_heads,
        heads,
    };

    let out_path = args.out.join("sanity_check.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn forward_q4k_noop_replace_pre_o_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    larql_inference::vindex::predict_kquant_hidden_with_mapped_pre_o_head(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original| Ok(original.clone()),
    )
    .map_err(Into::into)
}

fn forward_q4k_subtract_pre_o_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    larql_inference::vindex::predict_kquant_hidden_with_subtracted_pre_o_heads(
        weights,
        token_ids,
        index,
        head.layer,
        &[head.head],
    )
    .map_err(Into::into)
}

fn forward_q4k_noop_replace_head_residual_delta(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    larql_inference::vindex::predict_kquant_hidden_with_original_head_residual_delta(
        weights, token_ids, index, head.layer, head.head,
    )
    .map_err(Into::into)
}

fn final_logits(weights: &larql_inference::ModelWeights, h: &Array2<f32>) -> Vec<f32> {
    let last = h.nrows().saturating_sub(1);
    let h_last = h.slice(s![last..last + 1, ..]).to_owned();
    hidden_to_raw_logits(weights, &h_last)
}
