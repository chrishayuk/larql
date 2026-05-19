use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{dot_proj, embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::input::{limit_prompts_per_stratum, load_prompts, parse_layer_spec};
use super::reports::{CaptureReport, HeadReport};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::RunningHeadStats;

#[derive(Args)]
pub(super) struct CaptureArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Layers to capture. Comma-separated or range. Default: all.
    #[arg(long)]
    layers: Option<String>,

    /// Limit prompts for smoke runs.
    #[arg(long)]
    max_prompts: Option<usize>,

    /// Limit prompts per stratum after loading the prompt file.
    #[arg(long)]
    max_per_stratum: Option<usize>,

    /// Limit token positions per prompt for smoke runs.
    #[arg(long)]
    max_positions: Option<usize>,

    /// Also compute W_O-visible residual-contribution statistics.
    ///
    /// This is slower than raw pre-W_O capture because it projects each head
    /// through its W_O block, but it gives the ranking the downstream residual
    /// actually sees.
    #[arg(long)]
    wo_visible: bool,
}

pub(super) fn run_capture(args: CaptureArgs) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    eprintln!("Loading vindex: {}", args.index.display());
    let start = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_kquant(&args.index)?;
    index.load_interleaved_kquant(&args.index)?;
    let mut weights = load_model_weights_kquant(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..weights.num_layers).collect(),
    };
    let capture_layer = |layer: usize| layers.contains(&layer);

    let mut prompts = load_prompts(&args.prompts, args.max_prompts)?;
    if let Some(max_per_stratum) = args.max_per_stratum {
        prompts = limit_prompts_per_stratum(prompts, max_per_stratum);
    }
    eprintln!("Prompts: {}", prompts.len());
    eprintln!("Layers: {:?}", layers);

    let mut stats: Vec<Vec<RunningHeadStats>> = (0..weights.num_layers)
        .map(|layer| {
            let heads = weights.arch.num_q_heads_for_layer(layer);
            let head_dim = weights.arch.head_dim_for_layer(layer);
            (0..heads)
                .map(|_| RunningHeadStats::new(head_dim))
                .collect()
        })
        .collect();
    let mut wo_visible_stats: Vec<Vec<Option<RunningHeadStats>>> = (0..weights.num_layers)
        .map(|layer| {
            let heads = weights.arch.num_q_heads_for_layer(layer);
            (0..heads)
                .map(|_| {
                    if args.wo_visible {
                        Some(RunningHeadStats::new(weights.hidden_size))
                    } else {
                        None
                    }
                })
                .collect()
        })
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

        let mut h = embed_tokens_pub(&weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(&weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(&mut weights, &index, layer)?;

            if capture_layer(layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(&weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                add_pre_o_stats(
                    &mut stats[layer],
                    &pre_o,
                    weights.arch.num_q_heads_for_layer(layer),
                    weights.arch.head_dim_for_layer(layer),
                    args.max_positions,
                );
                if args.wo_visible {
                    let w_o = weights
                        .tensors
                        .get(&weights.arch.attn_o_key(layer))
                        .ok_or_else(|| format!("missing W_O tensor at layer {layer}"))?;
                    add_pre_o_wo_visible_stats(
                        &mut wo_visible_stats[layer],
                        &pre_o,
                        w_o,
                        weights.arch.num_q_heads_for_layer(layer),
                        weights.arch.head_dim_for_layer(layer),
                        args.max_positions,
                    );
                }
            }

            {
                let ffn = WeightFfn { weights: &weights };
                if let Some((h_new, _, _)) = run_layer_with_ffn(
                    &weights,
                    &h,
                    layer,
                    &ffn,
                    false,
                    ple_inputs.get(layer),
                    None,
                ) {
                    h = h_new;
                }
            }

            remove_layer_tensors(&mut weights, inserted);
        }
    }

    let mut heads = Vec::new();
    for &layer in &layers {
        let head_dim = weights.arch.head_dim_for_layer(layer);
        for (head, stat) in stats[layer].iter().enumerate() {
            heads.push(HeadReport {
                layer,
                head,
                head_dim,
                stats: stat.finish(),
                wo_visible_stats: wo_visible_stats[layer][head]
                    .as_ref()
                    .map(RunningHeadStats::finish),
            });
        }
    }

    let report = CaptureReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen: prompts.len(),
        layers,
        max_positions: args.max_positions,
        wo_visible: args.wo_visible,
        heads,
    };

    let out_path = args.out.join("stage0_pre_o_stats.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn add_pre_o_stats(
    stats: &mut [RunningHeadStats],
    pre_o: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    max_positions: Option<usize>,
) {
    let positions = max_positions
        .map(|n| n.min(pre_o.nrows()))
        .unwrap_or_else(|| pre_o.nrows());
    for pos in 0..positions {
        for head in 0..num_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            let row = pre_o.slice(s![pos, start..end]);
            if let Some(values) = row.as_slice() {
                stats[head].add(values);
            }
        }
    }
}

fn add_pre_o_wo_visible_stats(
    stats: &mut [Option<RunningHeadStats>],
    pre_o: &Array2<f32>,
    w_o: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    num_heads: usize,
    head_dim: usize,
    max_positions: Option<usize>,
) {
    let positions = max_positions
        .map(|n| n.min(pre_o.nrows()))
        .unwrap_or_else(|| pre_o.nrows());
    for head in 0..num_heads {
        let Some(head_stats) = stats.get_mut(head).and_then(Option::as_mut) else {
            continue;
        };
        let start = head * head_dim;
        let end = start + head_dim;
        let head_out = pre_o.slice(s![0..positions, start..end]);
        let w_o_head = w_o.slice(s![.., start..end]);
        let contribution = dot_proj(&head_out, &w_o_head);
        for row in contribution.rows() {
            if let Some(values) = row.as_slice() {
                head_stats.add(values);
            }
        }
    }
}
