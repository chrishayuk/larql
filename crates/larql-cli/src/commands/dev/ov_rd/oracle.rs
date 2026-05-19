use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{encode_prompt, hidden_to_raw_logits};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::basis::{
    build_roundtrip_bases, fit_z_pca_bases, RoundtripPatchMetrics, WoRoundtripBasis, ZPcaBasis,
};
use super::input::{load_prompts, parse_head_spec, parse_usize_list};
use super::metrics::{
    argmax, bool_rate, kl_logp, log_softmax, max_abs_diff, mean, percentile, token_prob,
    top_k_indices,
};
use super::reports::{
    OracleLowrankHeadReport, OracleLowrankPointReport, OracleLowrankPromptReport,
    OracleLowrankReport, OracleRoundtripHeadReport, OracleRoundtripPromptReport,
    OracleRoundtripReport,
};
use super::static_replace::fit_static_means;
use super::stats::StaticHeadMeans;
use super::types::HeadId;

#[derive(Args)]
pub(super) struct OracleRoundtripArgs {
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

    /// Relative singular value cutoff for retained W_O-visible directions.
    #[arg(long, default_value_t = 1e-6)]
    sigma_rel_cutoff: f64,

    /// Limit prompts for bounded sanity runs.
    #[arg(long)]
    max_prompts: Option<usize>,
}

#[derive(Args)]
pub(super) struct OracleLowrankArgs {
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

    /// Comma-separated K values for the low-rank sweep.
    #[arg(long, default_value = "1,2,4,8,16,32")]
    ks: String,

    /// Relative singular value cutoff for retained W_O-visible directions.
    #[arg(long, default_value_t = 1e-6)]
    sigma_rel_cutoff: f64,

    /// Limit prompts for bounded sanity runs.
    #[arg(long)]
    max_prompts: Option<usize>,
}

#[derive(Debug)]
struct OracleLowrankPointAccumulator {
    prompts: Vec<OracleLowrankPromptReport>,
}

impl OracleLowrankPointAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
        }
    }

    fn add(&mut self, prompt: OracleLowrankPromptReport) {
        self.prompts.push(prompt);
    }

    fn finish(self, k: usize) -> OracleLowrankPointReport {
        let kls: Vec<f64> = self.prompts.iter().map(|p| p.kl).collect();
        let mean_delta_cross_entropy_bits = mean(
            &self
                .prompts
                .iter()
                .map(|p| p.delta_cross_entropy_bits)
                .collect::<Vec<_>>(),
        );
        OracleLowrankPointReport {
            k,
            prompts: self.prompts.len(),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            mean_delta_cross_entropy_bits,
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts.iter().map(|p| p.baseline_top1_in_lowrank_top5),
            ),
            mean_baseline_top1_prob: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_prob)
                    .collect::<Vec<_>>(),
            ),
            mean_lowrank_prob_of_baseline_top1: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.lowrank_prob_of_baseline_top1)
                    .collect::<Vec<_>>(),
            ),
            mean_baseline_top1_margin: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_margin)
                    .collect::<Vec<_>>(),
            ),
            mean_pre_wo_l2: mean(&self.prompts.iter().map(|p| p.pre_wo_l2).collect::<Vec<_>>()),
            mean_wo_visible_l2: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.wo_visible_l2)
                    .collect::<Vec<_>>(),
            ),
            per_prompt: self.prompts,
        }
    }
}

#[derive(Debug)]
struct OracleRoundtripAccumulator {
    prompts: Vec<OracleRoundtripPromptReport>,
}

impl OracleRoundtripAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
        }
    }

    fn add(&mut self, prompt: OracleRoundtripPromptReport) {
        self.prompts.push(prompt);
    }

    fn finish(self, head: HeadId, basis: &WoRoundtripBasis) -> OracleRoundtripHeadReport {
        let kls: Vec<f64> = self.prompts.iter().map(|p| p.kl).collect();
        let pre_l2: Vec<f64> = self.prompts.iter().map(|p| p.pre_wo_l2).collect();
        let visible_l2: Vec<f64> = self.prompts.iter().map(|p| p.wo_visible_l2).collect();
        OracleRoundtripHeadReport {
            layer: head.layer,
            head: head.head,
            head_dim: basis.head_dim,
            rank_retained: basis.rank_retained(),
            sigma_max: basis.sigma_max,
            sigma_min_retained: basis.sigma_min_retained,
            sigma_rel_cutoff: basis.sigma_rel_cutoff,
            prompts: self.prompts.len(),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            max_abs_logit_diff: self
                .prompts
                .iter()
                .map(|p| p.max_abs_logit_diff)
                .fold(0.0, f64::max),
            mean_pre_wo_l2: mean(&pre_l2),
            max_pre_wo_l2: pre_l2.iter().copied().fold(0.0, f64::max),
            mean_wo_visible_l2: mean(&visible_l2),
            max_wo_visible_l2: visible_l2.iter().copied().fold(0.0, f64::max),
            per_prompt: self.prompts,
        }
    }
}

pub(super) fn run_oracle_roundtrip(
    args: OracleRoundtripArgs,
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
        return Err("ov-rd oracle-roundtrip currently supports dense FFN vindexes only".into());
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
        return Err("no heads selected for oracle roundtrip".into());
    }
    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("Prompts: {}", prompts.len());

    eprintln!("Building W_O-visible roundtrip bases");
    let bases =
        build_roundtrip_bases(&mut weights, &index, &selected_heads, args.sigma_rel_cutoff)?;
    for head in &selected_heads {
        let basis = bases
            .get(head)
            .ok_or_else(|| format!("missing basis for L{} H{}", head.layer, head.head))?;
        eprintln!(
            "  L{}H{} rank={} sigma_max={:.6} sigma_min_retained={:.6}",
            head.layer,
            head.head,
            basis.rank_retained(),
            basis.sigma_max,
            basis.sigma_min_retained
        );
    }

    let mut accumulators: Vec<OracleRoundtripAccumulator> = selected_heads
        .iter()
        .map(|_| OracleRoundtripAccumulator::new())
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
            let basis = bases
                .get(&head)
                .ok_or_else(|| format!("missing basis for L{} H{}", head.layer, head.head))?;
            let (roundtrip_hidden, metrics) =
                forward_q4k_oracle_roundtrip_head(&mut weights, &token_ids, &index, head, basis)?;
            let roundtrip_logits = final_logits(&weights, &roundtrip_hidden);
            let roundtrip_logp = log_softmax(&roundtrip_logits);
            accumulators[idx].add(OracleRoundtripPromptReport {
                id: label.to_string(),
                stratum: stratum.to_string(),
                kl: kl_logp(&baseline_logp, &roundtrip_logp),
                max_abs_logit_diff: max_abs_diff(&baseline_logits, &roundtrip_logits),
                pre_wo_l2: metrics.pre_wo_l2,
                wo_visible_l2: metrics.wo_visible_l2,
            });
        }
    }

    let heads = selected_heads
        .iter()
        .copied()
        .zip(accumulators)
        .map(|(head, acc)| {
            let basis = bases
                .get(&head)
                .expect("basis existed during oracle roundtrip");
            acc.finish(head, basis)
        })
        .collect();
    let report = OracleRoundtripReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        sigma_rel_cutoff: args.sigma_rel_cutoff,
        selected_heads,
        heads,
    };

    let out_path = args.out.join("oracle_roundtrip.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

pub(super) fn run_oracle_lowrank(
    args: OracleLowrankArgs,
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
        return Err("ov-rd oracle-lowrank currently supports dense FFN vindexes only".into());
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
        return Err("no heads selected for oracle lowrank".into());
    }
    let mut ks = parse_usize_list(&args.ks)?;
    ks.sort_unstable();
    ks.dedup();
    if ks.is_empty() {
        return Err("no K values selected for oracle lowrank".into());
    }
    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("K sweep: {:?}", ks);
    eprintln!("Prompts: {}", prompts.len());

    eprintln!("Fitting position-mean static bases");
    let means = fit_static_means(&mut weights, &index, &tokenizer, &prompts, &selected_heads)?;

    eprintln!("Building W_O-visible bases");
    let bases =
        build_roundtrip_bases(&mut weights, &index, &selected_heads, args.sigma_rel_cutoff)?;
    for head in &selected_heads {
        let basis = bases
            .get(head)
            .ok_or_else(|| format!("missing basis for L{} H{}", head.layer, head.head))?;
        eprintln!(
            "  L{}H{} rank={} sigma_max={:.6} sigma_min_retained={:.6}",
            head.layer,
            head.head,
            basis.rank_retained(),
            basis.sigma_max,
            basis.sigma_min_retained
        );
    }

    eprintln!("Fitting empirical z-space PCA bases");
    let pca_bases = fit_z_pca_bases(
        &mut weights,
        &index,
        &tokenizer,
        &prompts,
        &selected_heads,
        &bases,
        &means,
    )?;

    let mut accumulators: HashMap<(HeadId, usize), OracleLowrankPointAccumulator> = HashMap::new();
    for head in &selected_heads {
        for &k in &ks {
            accumulators.insert((*head, k), OracleLowrankPointAccumulator::new());
        }
    }

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
        let baseline_top2 = top_k_indices(&baseline_logits, 2);
        let baseline_top2_token = baseline_top2.get(1).copied().unwrap_or(baseline_top1);
        let baseline_top1_prob = token_prob(&baseline_logp, baseline_top1);
        let baseline_top2_prob = token_prob(&baseline_logp, baseline_top2_token);
        let baseline_top1_margin = baseline_top1_prob - baseline_top2_prob;

        for head in &selected_heads {
            let basis = bases.get(head).ok_or_else(|| {
                format!(
                    "missing basis for oracle lowrank L{} H{}",
                    head.layer, head.head
                )
            })?;
            let head_means = means.get(head).ok_or_else(|| {
                format!(
                    "missing position means for oracle lowrank L{} H{}",
                    head.layer, head.head
                )
            })?;
            let pca_basis = pca_bases.get(head).ok_or_else(|| {
                format!(
                    "missing empirical PCA basis for oracle lowrank L{} H{}",
                    head.layer, head.head
                )
            })?;
            for &k in &ks {
                let (lowrank_hidden, metrics) = forward_q4k_oracle_lowrank_head(
                    &mut weights,
                    &token_ids,
                    &index,
                    *head,
                    basis,
                    pca_basis,
                    head_means,
                    k,
                )?;
                let lowrank_logits = final_logits(&weights, &lowrank_hidden);
                let lowrank_logp = log_softmax(&lowrank_logits);
                let kl = kl_logp(&baseline_logp, &lowrank_logp);
                let lowrank_top1 = argmax(&lowrank_logits);
                let lowrank_top5 = top_k_indices(&lowrank_logits, 5);
                let lowrank_top2 = top_k_indices(&lowrank_logits, 2);
                let lowrank_top2_token = lowrank_top2.get(1).copied().unwrap_or(lowrank_top1);
                let lowrank_top1_prob = token_prob(&lowrank_logp, lowrank_top1);
                let lowrank_top2_prob = token_prob(&lowrank_logp, lowrank_top2_token);
                let lowrank_top1_margin = lowrank_top1_prob - lowrank_top2_prob;
                let lowrank_prob_of_baseline_top1 = token_prob(&lowrank_logp, baseline_top1);
                accumulators
                    .get_mut(&(*head, k))
                    .expect("oracle lowrank accumulator missing")
                    .add(OracleLowrankPromptReport {
                        id: label.to_string(),
                        stratum: stratum.to_string(),
                        kl,
                        delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
                        baseline_top1,
                        lowrank_top1,
                        top1_agree: baseline_top1 == lowrank_top1,
                        baseline_top1_in_lowrank_top5: lowrank_top5.contains(&baseline_top1),
                        baseline_top1_prob,
                        baseline_top2: baseline_top2_token,
                        baseline_top2_prob,
                        baseline_top1_margin,
                        lowrank_top1_prob,
                        lowrank_prob_of_baseline_top1,
                        lowrank_top1_margin,
                        pre_wo_l2: metrics.pre_wo_l2,
                        wo_visible_l2: metrics.wo_visible_l2,
                    });
            }
        }
    }

    let mut head_reports = Vec::new();
    for head in &selected_heads {
        let basis = bases
            .get(head)
            .ok_or_else(|| format!("missing basis for L{} H{}", head.layer, head.head))?;
        let pca_basis = pca_bases
            .get(head)
            .ok_or_else(|| format!("missing PCA basis for L{} H{}", head.layer, head.head))?;
        let mut points = Vec::new();
        for &k in &ks {
            let acc = accumulators
                .remove(&(*head, k))
                .expect("oracle lowrank accumulator missing at finish");
            points.push(acc.finish(k));
        }
        let static_train_samples = means.get(head).map(|m| m.count).unwrap_or(0);
        head_reports.push(OracleLowrankHeadReport {
            layer: head.layer,
            head: head.head,
            head_dim: basis.head_dim,
            rank_retained: basis.rank_retained(),
            empirical_rank: pca_basis.rank(),
            sigma_max: basis.sigma_max,
            sigma_min_retained: basis.sigma_min_retained,
            static_train_samples,
            points,
        });
    }

    let report = OracleLowrankReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        static_base: "position_mean".to_string(),
        ks,
        sigma_rel_cutoff: args.sigma_rel_cutoff,
        selected_heads,
        heads: head_reports,
    };

    let out_path = args.out.join("oracle_lowrank.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn forward_q4k_oracle_roundtrip_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
) -> Result<(Array2<f32>, RoundtripPatchMetrics), Box<dyn std::error::Error>> {
    let mut metrics = None;

    let h = larql_inference::vindex::predict_kquant_hidden_with_mapped_pre_o_head(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original_head| {
            let mut replacement = Vec::with_capacity(original_head.len());
            let mut pre_sq = 0.0;
            let mut visible_sq = 0.0;
            let mut count = 0usize;
            for pos in 0..original_head.nrows() {
                let row = original_head.row(pos);
                let values = row
                    .as_slice()
                    .ok_or("pre-W_O head row was not contiguous during roundtrip")?;
                let projected = basis.project(values);
                for (&original, &recon) in values.iter().zip(projected.iter()) {
                    let delta = original as f64 - recon as f64;
                    pre_sq += delta * delta;
                }
                let delta = values
                    .iter()
                    .zip(projected.iter())
                    .map(|(&original, &recon)| original as f64 - recon as f64)
                    .collect::<Vec<_>>();
                visible_sq += basis.visible_sq_norm(&delta);
                count += 1;
                replacement.extend_from_slice(&projected);
            }
            metrics = Some(RoundtripPatchMetrics {
                pre_wo_l2: (pre_sq / count.max(1) as f64).sqrt(),
                wo_visible_l2: (visible_sq / count.max(1) as f64).sqrt(),
            });
            Array2::from_shape_vec((original_head.nrows(), original_head.ncols()), replacement)
                .map_err(|err| err.to_string())
        },
    )?;

    Ok((
        h,
        metrics.ok_or("oracle roundtrip did not visit target layer")?,
    ))
}

fn forward_q4k_oracle_lowrank_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    k: usize,
) -> Result<(Array2<f32>, RoundtripPatchMetrics), Box<dyn std::error::Error>> {
    let mut metrics = None;

    let h = larql_inference::vindex::predict_kquant_hidden_with_mapped_pre_o_head(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original_head| {
            let mut replacement = Vec::with_capacity(original_head.len());
            let mut pre_sq = 0.0;
            let mut visible_sq = 0.0;
            let mut count = 0usize;
            for pos in 0..original_head.nrows() {
                let row = original_head.row(pos);
                let values = row
                    .as_slice()
                    .ok_or("pre-W_O head row was not contiguous during lowrank")?;
                let base = means.positions.get(pos).unwrap_or(&means.global);
                let residual = values
                    .iter()
                    .zip(base.iter())
                    .map(|(&yi, &bi)| yi - bi)
                    .collect::<Vec<_>>();
                let z = basis.residual_to_z(&residual);
                let z_projected = pca_basis.project_with_rank(&z, k);
                let residual_projected = basis.z_to_residual(&z_projected);
                let projected = residual_projected
                    .into_iter()
                    .zip(base.iter())
                    .map(|(ri, &bi)| ri + bi)
                    .collect::<Vec<_>>();
                for (&original, &recon) in values.iter().zip(projected.iter()) {
                    let delta = original as f64 - recon as f64;
                    pre_sq += delta * delta;
                }
                let delta = values
                    .iter()
                    .zip(projected.iter())
                    .map(|(&original, &recon)| original as f64 - recon as f64)
                    .collect::<Vec<_>>();
                visible_sq += basis.visible_sq_norm(&delta);
                count += 1;
                replacement.extend_from_slice(&projected);
            }
            metrics = Some(RoundtripPatchMetrics {
                pre_wo_l2: (pre_sq / count.max(1) as f64).sqrt(),
                wo_visible_l2: (visible_sq / count.max(1) as f64).sqrt(),
            });
            Array2::from_shape_vec((original_head.nrows(), original_head.ncols()), replacement)
                .map_err(|err| err.to_string())
        },
    )?;

    Ok((
        h,
        metrics.ok_or("oracle lowrank did not visit target layer")?,
    ))
}

fn final_logits(weights: &larql_inference::ModelWeights, h: &Array2<f32>) -> Vec<f32> {
    let last = h.nrows().saturating_sub(1);
    let h_last = h.slice(s![last..last + 1, ..]).to_owned();
    hidden_to_raw_logits(weights, &h_last)
}
