use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, ValueEnum};
use larql_inference::{encode_prompt, hidden_to_raw_logits};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::input::{load_prompts, parse_head_spec};
use super::metrics::{argmax, bool_rate, kl_logp, log_softmax, mean, percentile, top_k_indices};
use super::reports::{
    CaptureReport, ZeroAblationReport, ZeroHeadReport, ZeroPromptReport, ZeroStratumReport,
};
use super::types::HeadId;

#[derive(Args)]
pub(super) struct ZeroAblateArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 11:3,11:0,0:4.
    #[arg(long)]
    heads: Option<String>,

    /// Stage-0 stats JSON. Used with --top-heads when --heads is absent.
    #[arg(long)]
    stage0: Option<PathBuf>,

    /// Number of highest-variance Stage-0 heads to test.
    #[arg(long, default_value_t = 8)]
    top_heads: usize,

    /// Stage-0 statistic used to rank --top-heads.
    #[arg(long, value_enum, default_value_t = Stage0Rank::RawVariance)]
    stage0_rank: Stage0Rank,

    /// Limit prompts for bounded gate runs.
    #[arg(long)]
    max_prompts: Option<usize>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Stage0Rank {
    /// Rank by raw pre-W_O variance.
    RawVariance,
    /// Rank by W_O-visible residual contribution variance.
    WoVisibleVariance,
}

#[derive(Debug)]
struct ZeroHeadAccumulator {
    prompts: Vec<ZeroPromptReport>,
    by_stratum: HashMap<String, Vec<ZeroPromptReport>>,
}

impl ZeroHeadAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
            by_stratum: HashMap::new(),
        }
    }

    fn add(&mut self, prompt: ZeroPromptReport) {
        let stratum = prompt.stratum.clone();
        self.prompts.push(prompt.clone());
        self.by_stratum.entry(stratum).or_default().push(prompt);
    }

    fn finish(self, head: HeadId) -> ZeroHeadReport {
        let prompts_len = self.prompts.len();
        let kl_values: Vec<f64> = self.prompts.iter().map(|p| p.kl).collect();
        let mean_kl = mean(&kl_values);
        let p95_kl = percentile(kl_values.clone(), 0.95);
        let max_kl = kl_values.iter().copied().fold(0.0, f64::max);
        let mean_delta_cross_entropy_bits = mean(
            &self
                .prompts
                .iter()
                .map(|p| p.delta_cross_entropy_bits)
                .collect::<Vec<_>>(),
        );
        let top1_agreement = bool_rate(self.prompts.iter().map(|p| p.top1_agree));
        let top5_contains_baseline_top1 =
            bool_rate(self.prompts.iter().map(|p| p.baseline_top1_in_ablated_top5));
        let mut worst_examples = self.prompts.clone();
        worst_examples.sort_by(|a, b| b.kl.partial_cmp(&a.kl).unwrap_or(std::cmp::Ordering::Equal));
        worst_examples.truncate(10);

        let mut strata: Vec<_> = self
            .by_stratum
            .into_iter()
            .map(|(stratum, prompts)| {
                let values: Vec<f64> = prompts.iter().map(|p| p.kl).collect();
                ZeroStratumReport {
                    stratum,
                    prompts: prompts.len(),
                    mean_kl: mean(&values),
                    max_kl: values.iter().copied().fold(0.0, f64::max),
                    top1_agreement: bool_rate(prompts.iter().map(|p| p.top1_agree)),
                    top5_contains_baseline_top1: bool_rate(
                        prompts.iter().map(|p| p.baseline_top1_in_ablated_top5),
                    ),
                }
            })
            .collect();
        strata.sort_by(|a, b| a.stratum.cmp(&b.stratum));
        ZeroHeadReport {
            layer: head.layer,
            head: head.head,
            ablation_kind: "zero_pre_wo".to_string(),
            patch_location: "before_W_O".to_string(),
            preserved_components: vec![
                "FFN".to_string(),
                "PLE".to_string(),
                "layer_scalar".to_string(),
            ],
            bounded_vocab_size: None,
            prompts: prompts_len,
            mean_kl,
            p95_kl,
            max_kl,
            mean_delta_cross_entropy_bits,
            top1_agreement,
            top5_contains_baseline_top1,
            strata,
            worst_examples,
            per_prompt: self.prompts,
        }
    }
}

pub(super) fn run_zero_ablate(args: ZeroAblateArgs) -> Result<(), Box<dyn std::error::Error>> {
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
        return Err("ov-rd zero-ablate currently supports dense FFN vindexes only".into());
    }
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );

    let selected_heads = select_zero_ablation_heads(&args)?;
    if selected_heads.is_empty() {
        return Err("no heads selected for zero-ablation".into());
    }
    eprintln!("Selected heads: {:?}", selected_heads);

    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Prompts: {}", prompts.len());

    let mut accumulators: Vec<ZeroHeadAccumulator> = selected_heads
        .iter()
        .map(|_| ZeroHeadAccumulator::new())
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
        let baseline_top1 = argmax(&baseline_logits);

        for (idx, head) in selected_heads.iter().copied().enumerate() {
            let ablated_hidden =
                forward_q4k_zero_pre_o_head(&mut weights, &token_ids, &index, head)?;
            let ablated_logits = final_logits(&weights, &ablated_hidden);
            let ablated_logp = log_softmax(&ablated_logits);
            let kl = kl_logp(&baseline_logp, &ablated_logp);
            let ablated_top1 = argmax(&ablated_logits);
            let ablated_top5 = top_k_indices(&ablated_logits, 5);
            accumulators[idx].add(ZeroPromptReport {
                id: label.to_string(),
                stratum: stratum.to_string(),
                kl,
                delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
                baseline_top1,
                ablated_top1,
                top1_agree: baseline_top1 == ablated_top1,
                baseline_top1_in_ablated_top5: ablated_top5.contains(&baseline_top1),
            });
        }
    }

    let head_reports = selected_heads
        .iter()
        .copied()
        .zip(accumulators)
        .map(|(head, acc)| acc.finish(head))
        .collect();

    let report = ZeroAblationReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        selected_heads,
        heads: head_reports,
    };

    let out_path = args.out.join("gate1_zero_ablation.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn select_zero_ablation_heads(
    args: &ZeroAblateArgs,
) -> Result<Vec<HeadId>, Box<dyn std::error::Error>> {
    let mut heads = if let Some(spec) = &args.heads {
        parse_head_spec(spec)?
    } else {
        let stage0_path = args
            .stage0
            .as_ref()
            .ok_or("--heads or --stage0 must be provided")?;
        let file = std::fs::File::open(stage0_path)?;
        let report: CaptureReport = serde_json::from_reader(file)?;
        let mut candidates = report.heads;
        candidates.sort_by(|a, b| {
            stage0_rank_score(b, args.stage0_rank)
                .partial_cmp(&stage0_rank_score(a, args.stage0_rank))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
            .into_iter()
            .take(args.top_heads)
            .map(|h| HeadId {
                layer: h.layer,
                head: h.head,
            })
            .collect()
    };

    heads.sort_by_key(|h| (h.layer, h.head));
    heads.dedup();
    Ok(heads)
}

fn stage0_rank_score(head: &super::reports::HeadReport, rank: Stage0Rank) -> f64 {
    match rank {
        Stage0Rank::RawVariance => head.stats.variance,
        Stage0Rank::WoVisibleVariance => head
            .wo_visible_stats
            .as_ref()
            .map(|stats| stats.variance)
            .unwrap_or(f64::NEG_INFINITY),
    }
}

pub(super) fn forward_q4k_zero_pre_o_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    larql_inference::vindex::predict_kquant_hidden_with_zeroed_pre_o_heads(
        weights,
        token_ids,
        index,
        head.layer,
        &[head.head],
    )
    .map_err(Into::into)
}

fn final_logits(weights: &larql_inference::ModelWeights, h: &Array2<f32>) -> Vec<f32> {
    let last = h.nrows().saturating_sub(1);
    let h_last = h.slice(s![last..last + 1, ..]).to_owned();
    hidden_to_raw_logits(weights, &h_last)
}
