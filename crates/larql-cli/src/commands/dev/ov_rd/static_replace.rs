use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, hidden_to_raw_logits, WeightFfn};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::input::{load_prompts, parse_head_spec, split_prompt_records};
use super::metrics::{argmax, bool_rate, kl_logp, log_softmax, mean, percentile, top_k_indices};
use super::reports::{
    StaticHeadReport, StaticModeReport, StaticReplacementReport, ZeroPromptReport,
    ZeroStratumReport,
};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::{StaticHeadAccumulator, StaticHeadMeans};
use super::types::{HeadId, PromptRecord};

#[derive(Args)]
pub(super) struct StaticReplaceArgs {
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
    heads: String,

    /// Limit prompts for bounded gate runs.
    #[arg(long)]
    max_prompts: Option<usize>,

    /// Evaluate only prompts where prompt_index % eval_mod == eval_offset.
    /// The remaining prompts are used to fit static means. Omit for in-sample
    /// fit/eval on the same prompt set.
    #[arg(long)]
    eval_mod: Option<usize>,

    /// Held-out modulo offset used with --eval-mod.
    #[arg(long, default_value_t = 0)]
    eval_offset: usize,
}

#[derive(Debug, Clone, Copy)]
enum StaticReplacementKind {
    Zero,
    Global,
    Position,
    Stratum,
    PositionPlusStratum,
    PositionStratum,
}

impl StaticReplacementKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Global => "global_mean",
            Self::Position => "position_mean",
            Self::Stratum => "stratum_mean",
            Self::PositionPlusStratum => "position_plus_stratum_mean",
            Self::PositionStratum => "position_stratum_mean",
        }
    }
}

const STATIC_REPLACEMENT_KINDS: [StaticReplacementKind; 6] = [
    StaticReplacementKind::Zero,
    StaticReplacementKind::Global,
    StaticReplacementKind::Position,
    StaticReplacementKind::Stratum,
    StaticReplacementKind::PositionPlusStratum,
    StaticReplacementKind::PositionStratum,
];

#[derive(Debug)]
struct StaticModeAccumulator {
    prompts: Vec<ZeroPromptReport>,
    by_stratum: HashMap<String, Vec<ZeroPromptReport>>,
}

impl StaticModeAccumulator {
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

    fn finish(self, kind: StaticReplacementKind) -> StaticModeReport {
        let kl_values: Vec<f64> = self.prompts.iter().map(|p| p.kl).collect();
        let mean_delta_cross_entropy_bits = mean(
            &self
                .prompts
                .iter()
                .map(|p| p.delta_cross_entropy_bits)
                .collect::<Vec<_>>(),
        );
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
        StaticModeReport {
            replacement_kind: kind.as_str().to_string(),
            patch_location: "before_W_O".to_string(),
            runtime_class: match kind {
                StaticReplacementKind::Zero => "negligible_test",
                _ => "static_injection_lookup_add",
            }
            .to_string(),
            prompts: self.prompts.len(),
            mean_kl: mean(&kl_values),
            p95_kl: percentile(kl_values.clone(), 0.95),
            max_kl: kl_values.iter().copied().fold(0.0, f64::max),
            mean_delta_cross_entropy_bits,
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts.iter().map(|p| p.baseline_top1_in_ablated_top5),
            ),
            strata,
            worst_examples,
            per_prompt: self.prompts,
        }
    }
}

pub(super) fn run_static_replace(
    args: StaticReplaceArgs,
) -> Result<(), Box<dyn std::error::Error>> {
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
        return Err("ov-rd static-replace currently supports dense FFN vindexes only".into());
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
        return Err("no heads selected for static replacement".into());
    }
    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("Prompts: {}", prompts.len());
    let (fit_prompts, eval_prompts): (Vec<PromptRecord>, Vec<PromptRecord>) =
        if let Some(eval_mod) = args.eval_mod {
            split_prompt_records(&prompts, eval_mod, args.eval_offset)?
        } else {
            (prompts.clone(), prompts.clone())
        };

    eprintln!("Pass 1/2: fitting static pre-W_O means");
    let means = fit_static_means(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
    )?;

    eprintln!("Pass 2/2: evaluating static replacements");
    let mut accumulators: HashMap<(HeadId, &'static str), StaticModeAccumulator> = HashMap::new();
    for head in &selected_heads {
        for kind in STATIC_REPLACEMENT_KINDS {
            accumulators.insert((*head, kind.as_str()), StaticModeAccumulator::new());
        }
    }

    for (prompt_idx, record) in eval_prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", prompt_idx + 1, eval_prompts.len(), label);

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
        for head in &selected_heads {
            let head_means = means.get(head).ok_or_else(|| {
                format!("missing fitted means for L{} H{}", head.layer, head.head)
            })?;
            for kind in STATIC_REPLACEMENT_KINDS {
                let replacement =
                    build_static_replacement(kind, token_ids.len(), head_means, stratum)?;
                let replaced_hidden = forward_q4k_replace_pre_o_head(
                    &mut weights,
                    &token_ids,
                    &index,
                    *head,
                    &replacement,
                )?;
                let replaced_logits = final_logits(&weights, &replaced_hidden);
                let replaced_logp = log_softmax(&replaced_logits);
                let kl = kl_logp(&baseline_logp, &replaced_logp);
                let replaced_top1 = argmax(&replaced_logits);
                let replaced_top5 = top_k_indices(&replaced_logits, 5);
                accumulators
                    .get_mut(&(*head, kind.as_str()))
                    .expect("static accumulator missing")
                    .add(ZeroPromptReport {
                        id: label.to_string(),
                        stratum: stratum.to_string(),
                        kl,
                        delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
                        baseline_top1,
                        ablated_top1: replaced_top1,
                        top1_agree: baseline_top1 == replaced_top1,
                        baseline_top1_in_ablated_top5: replaced_top5.contains(&baseline_top1),
                    });
            }
        }
    }

    let mut head_reports = Vec::new();
    for head in &selected_heads {
        let mut modes = Vec::new();
        for kind in STATIC_REPLACEMENT_KINDS {
            let acc = accumulators
                .remove(&(*head, kind.as_str()))
                .expect("static accumulator missing at finish");
            modes.push(acc.finish(kind));
        }
        let train_samples = means.get(head).map(|m| m.count).unwrap_or(0);
        head_reports.push(StaticHeadReport {
            layer: head.layer,
            head: head.head,
            train_samples,
            modes,
        });
    }

    let report = StaticReplacementReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        train_prompts_seen: fit_prompts.len(),
        eval_prompts_seen: eval_prompts.len(),
        eval_mod: args.eval_mod,
        eval_offset: args.eval_offset,
        selected_heads,
        heads: head_reports,
    };

    let out_path = args.out.join("gate_static_replacement.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

pub(super) fn fit_static_means(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
) -> Result<HashMap<HeadId, StaticHeadMeans>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }

    let mut accumulators: HashMap<HeadId, StaticHeadAccumulator> = HashMap::new();
    for head in heads {
        let head_dim = weights.arch.head_dim_for_layer(head.layer);
        accumulators.insert(*head, StaticHeadAccumulator::new(head_dim));
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  fit [{}/{}] {}", prompt_idx + 1, prompts.len(), label);
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let acc = accumulators
                        .get_mut(head)
                        .expect("static mean accumulator missing");
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        if let Some(values) = row.as_slice() {
                            acc.add(pos, stratum, values);
                        }
                    }
                }
            }

            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) =
                    run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None)
                {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    Ok(accumulators
        .into_iter()
        .map(|(head, acc)| (head, acc.finish()))
        .collect())
}

fn build_static_replacement(
    kind: StaticReplacementKind,
    seq_len: usize,
    means: &StaticHeadMeans,
    stratum: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut values = Vec::with_capacity(seq_len * means.head_dim);
    for pos in 0..seq_len {
        let owned_row;
        let row = match kind {
            StaticReplacementKind::Zero => None,
            StaticReplacementKind::Global => Some(&means.global),
            StaticReplacementKind::Position => means.positions.get(pos).or(Some(&means.global)),
            StaticReplacementKind::Stratum => means.strata.get(stratum).or(Some(&means.global)),
            StaticReplacementKind::PositionPlusStratum => {
                let pos_row = means.positions.get(pos).unwrap_or(&means.global);
                let stratum_row = means.strata.get(stratum).unwrap_or(&means.global);
                owned_row = pos_row
                    .iter()
                    .zip(stratum_row.iter())
                    .zip(means.global.iter())
                    .map(|((&p, &s), &g)| p + s - g)
                    .collect::<Vec<_>>();
                Some(&owned_row)
            }
            StaticReplacementKind::PositionStratum => means
                .position_strata
                .get(stratum)
                .and_then(|rows| rows.get(pos))
                .or_else(|| means.positions.get(pos))
                .or(Some(&means.global)),
        };
        if let Some(row) = row {
            values.extend_from_slice(row);
        } else {
            values.extend(std::iter::repeat(0.0).take(means.head_dim));
        }
    }
    Ok(Array2::from_shape_vec((seq_len, means.head_dim), values)?)
}

fn forward_q4k_replace_pre_o_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    replacement: &Array2<f32>,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    larql_inference::vindex::predict_kquant_hidden_with_replaced_pre_o_head(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        replacement,
    )
    .map_err(Into::into)
}

fn final_logits(weights: &larql_inference::ModelWeights, h: &Array2<f32>) -> Vec<f32> {
    let last = h.nrows().saturating_sub(1);
    let h_last = h.slice(s![last..last + 1, ..]).to_owned();
    hidden_to_raw_logits(weights, &h_last)
}
