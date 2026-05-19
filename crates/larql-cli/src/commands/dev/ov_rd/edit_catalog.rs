use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};

use super::basis::{build_roundtrip_bases, fit_z_pca_bases, WoRoundtripBasis, ZPcaBasis};
use super::input::{
    limit_prompts_per_stratum, load_prompts, parse_head_spec, parse_usize_list,
    split_prompt_records,
};
use super::metrics::{
    argmax, bool_rate, kl_logp, log_softmax, mean, percentile, token_prob, top_k_indices,
};
use super::oracle_pq_forward::final_logits;
use super::pq::{kmeans_centroids, nearest_centroid_index};
use super::reports::{
    OracleEditCatalogHeadReport, OracleEditCatalogPointReport, OracleEditCatalogPromptReport,
    OracleEditCatalogReport,
};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::static_replace::fit_static_means;
use super::stats::StaticHeadMeans;
use super::types::{HeadId, PromptRecord};

#[derive(Args)]
pub(super) struct OracleEditCatalogArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 20:6.
    #[arg(long)]
    heads: String,

    /// Comma-separated full-edit catalogue sizes.
    #[arg(long, default_value = "32,64,128,256")]
    edit_counts: String,

    /// Comma-separated catalogue spaces: hidden,pca.
    #[arg(long, default_value = "hidden,pca")]
    spaces: String,

    /// PCA coordinate rank used by the pca catalogue space.
    #[arg(long, default_value_t = 192)]
    pca_rank: usize,

    /// Relative singular value cutoff for retained W_O-visible directions.
    #[arg(long, default_value_t = 1e-6)]
    sigma_rel_cutoff: f64,

    /// Lloyd iterations per full-edit catalogue.
    #[arg(long, default_value_t = 25)]
    kmeans_iters: usize,

    /// Limit prompts for bounded oracle runs.
    #[arg(long)]
    max_prompts: Option<usize>,

    /// Keep at most N prompts per stratum after loading.
    #[arg(long)]
    max_per_stratum: Option<usize>,

    /// Evaluate only prompts where prompt_index % eval_mod == eval_offset.
    /// The remaining prompts are used to fit static means, PCA, and catalogues.
    #[arg(long)]
    eval_mod: Option<usize>,

    /// Held-out modulo offset used with --eval-mod.
    #[arg(long, default_value_t = 0)]
    eval_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EditCatalogSpace {
    Hidden,
    Pca,
}

impl EditCatalogSpace {
    fn parse(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match name.trim() {
            "hidden" => Ok(Self::Hidden),
            "pca" => Ok(Self::Pca),
            other => {
                Err(format!("invalid edit-catalog space '{other}', expected hidden or pca").into())
            }
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Hidden => "hidden",
            Self::Pca => "pca",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EditCatalogKey {
    head: HeadId,
    space: EditCatalogSpace,
    edits: usize,
}

#[derive(Debug, Clone)]
struct EditCatalog {
    space: EditCatalogSpace,
    feature_centroids: Vec<Vec<f64>>,
    residual_table: Vec<Vec<f32>>,
}

pub(super) fn run_oracle_edit_catalog(
    args: OracleEditCatalogArgs,
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
        return Err("ov-rd oracle-edit-catalog currently supports dense FFN vindexes only".into());
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
        return Err("no heads selected for oracle edit catalogue".into());
    }
    let mut edit_counts = parse_usize_list(&args.edit_counts)?;
    edit_counts.sort_unstable();
    edit_counts.dedup();
    if edit_counts.is_empty() {
        return Err("no edit counts selected".into());
    }
    if edit_counts.iter().any(|&edits| edits == 0) {
        return Err("--edit-counts values must be greater than zero".into());
    }
    let mut spaces = parse_string_list(&args.spaces)
        .into_iter()
        .map(|space| EditCatalogSpace::parse(&space))
        .collect::<Result<Vec<_>, _>>()?;
    spaces.sort_by_key(|space| space.as_str());
    spaces.dedup();
    if spaces.is_empty() {
        return Err("no edit-catalog spaces selected".into());
    }

    let mut prompts = load_prompts(&args.prompts, args.max_prompts)?;
    if let Some(max_per_stratum) = args.max_per_stratum {
        prompts = limit_prompts_per_stratum(prompts, max_per_stratum);
    }
    let prompts_seen = prompts.len();
    let (fit_prompts, eval_prompts) = if let Some(eval_mod) = args.eval_mod {
        split_prompt_records(&prompts, eval_mod, args.eval_offset)?
    } else {
        (prompts.clone(), prompts)
    };

    eprintln!("Selected heads: {:?}", selected_heads);
    eprintln!("Edit counts: {:?}", edit_counts);
    eprintln!(
        "Edit spaces: {:?}",
        spaces
            .iter()
            .map(|space| space.as_str())
            .collect::<Vec<_>>()
    );
    eprintln!("Prompts: {}", prompts_seen);

    eprintln!("Fitting position-mean static bases");
    let means = fit_static_means(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
    )?;

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
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
    )?;

    eprintln!("Fitting full-edit catalogues");
    let catalogs = fit_edit_catalogs(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &spaces,
        &edit_counts,
        args.pca_rank,
        args.kmeans_iters,
    )?;

    let hidden_tables = build_static_hidden_tables(&mut weights, &index, &selected_heads, &means)?;
    let w_o_heads = copy_w_o_heads(&mut weights, &index, &selected_heads)?;

    let mut accumulators: HashMap<EditCatalogKey, EditCatalogAccumulator> = HashMap::new();
    for head in &selected_heads {
        for &space in &spaces {
            for &edits in &edit_counts {
                accumulators.insert(
                    EditCatalogKey {
                        head: *head,
                        space,
                        edits,
                    },
                    EditCatalogAccumulator::new(),
                );
            }
        }
    }

    for (prompt_idx, record) in eval_prompts.iter().enumerate() {
        let label = prompt_label(record);
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
        let baseline_top2 = top_k_indices(&baseline_logits, 2);
        let baseline_top2_token = baseline_top2.get(1).copied().unwrap_or(baseline_top1);
        let baseline_top1_prob = token_prob(&baseline_logp, baseline_top1);
        let baseline_top2_prob = token_prob(&baseline_logp, baseline_top2_token);
        let baseline_top1_margin = baseline_top1_prob - baseline_top2_prob;

        for head in &selected_heads {
            let basis = bases
                .get(head)
                .ok_or_else(|| format!("missing basis for L{}H{}", head.layer, head.head))?;
            let pca_basis = pca_bases
                .get(head)
                .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;
            let head_means = means
                .get(head)
                .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
            let static_hidden = hidden_tables.get(head).ok_or_else(|| {
                format!(
                    "missing static hidden table for L{}H{}",
                    head.layer, head.head
                )
            })?;
            let w_o_head = w_o_heads
                .get(head)
                .ok_or_else(|| format!("missing W_O head for L{}H{}", head.layer, head.head))?;

            for &space in &spaces {
                for &edits in &edit_counts {
                    let key = EditCatalogKey {
                        head: *head,
                        space,
                        edits,
                    };
                    let catalog = catalogs.get(&key).ok_or_else(|| {
                        format!(
                            "missing edit catalog for L{}H{} {} {edits}",
                            head.layer,
                            head.head,
                            space.as_str()
                        )
                    })?;
                    let catalog_hidden = forward_q4k_oracle_edit_catalog_head(
                        &mut weights,
                        &token_ids,
                        &index,
                        *head,
                        basis,
                        pca_basis,
                        head_means,
                        static_hidden,
                        w_o_head,
                        catalog,
                        args.pca_rank,
                    )?;
                    let catalog_logits = final_logits(&weights, &catalog_hidden);
                    let catalog_logp = log_softmax(&catalog_logits);
                    let kl = kl_logp(&baseline_logp, &catalog_logp);
                    let catalog_top1 = argmax(&catalog_logits);
                    let catalog_top5 = top_k_indices(&catalog_logits, 5);
                    let catalog_top2 = top_k_indices(&catalog_logits, 2);
                    let catalog_top2_token = catalog_top2.get(1).copied().unwrap_or(catalog_top1);
                    let catalog_top1_prob = token_prob(&catalog_logp, catalog_top1);
                    let catalog_top2_prob = token_prob(&catalog_logp, catalog_top2_token);
                    let catalog_top1_margin = catalog_top1_prob - catalog_top2_prob;
                    let catalog_prob_of_baseline_top1 = token_prob(&catalog_logp, baseline_top1);
                    accumulators
                        .get_mut(&key)
                        .expect("edit-catalog accumulator missing")
                        .add(OracleEditCatalogPromptReport {
                            id: label.to_string(),
                            stratum: stratum.to_string(),
                            kl,
                            delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
                            baseline_top1,
                            catalog_top1,
                            top1_agree: baseline_top1 == catalog_top1,
                            baseline_top1_in_catalog_top5: catalog_top5.contains(&baseline_top1),
                            baseline_top1_prob,
                            baseline_top2: baseline_top2_token,
                            baseline_top2_prob,
                            baseline_top1_margin,
                            catalog_top1_prob,
                            catalog_prob_of_baseline_top1,
                            catalog_top1_margin,
                        });
                }
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
        for &space in &spaces {
            for &edits in &edit_counts {
                let key = EditCatalogKey {
                    head: *head,
                    space,
                    edits,
                };
                let acc = accumulators
                    .remove(&key)
                    .expect("edit-catalog accumulator missing at finish");
                points.push(acc.finish(space, edits, weights.hidden_size));
            }
        }
        let static_train_samples = means.get(head).map(|m| m.count).unwrap_or(0);
        head_reports.push(OracleEditCatalogHeadReport {
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

    let report = OracleEditCatalogReport {
        index: args.index.display().to_string(),
        prompt_file: args.prompts.display().to_string(),
        prompts_seen,
        train_prompts_seen: fit_prompts.len(),
        eval_prompts_seen: eval_prompts.len(),
        max_per_stratum: args.max_per_stratum,
        eval_mod: args.eval_mod,
        eval_offset: args.eval_offset,
        static_base: "position_mean".to_string(),
        spaces: spaces
            .iter()
            .map(|space| space.as_str().to_string())
            .collect(),
        edit_counts,
        pca_rank: args.pca_rank,
        sigma_rel_cutoff: args.sigma_rel_cutoff,
        kmeans_iters: args.kmeans_iters,
        selected_heads,
        heads: head_reports,
    };

    let out_path = args.out.join("oracle_edit_catalog.json");
    let file = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(file, &report)?;
    eprintln!("Wrote {}", out_path.display());

    Ok(())
}

fn fit_edit_catalogs(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    spaces: &[EditCatalogSpace],
    edit_counts: &[usize],
    pca_rank: usize,
    iterations: usize,
) -> Result<HashMap<EditCatalogKey, EditCatalog>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let w_o_heads = copy_w_o_heads(weights, index, heads)?;

    let mut samples: HashMap<(HeadId, EditCatalogSpace), Vec<Vec<f64>>> = HashMap::new();
    for head in heads {
        for &space in spaces {
            samples.insert((*head, space), Vec::new());
        }
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = prompt_label(record);
        eprintln!(
            "  catalog-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(weights, &h, layer)
                    .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).expect("basis pre-created for edit catalog");
                    let head_means = means.get(head).expect("means pre-created for edit catalog");
                    let pca_basis = pca_bases
                        .get(head)
                        .expect("PCA pre-created for edit catalog");
                    if pca_basis.rank() < pca_rank && spaces.contains(&EditCatalogSpace::Pca) {
                        return Err(format!(
                            "PCA rank {} is below requested rank {} for L{}H{}",
                            pca_basis.rank(),
                            pca_rank,
                            head.layer,
                            head.head
                        )
                        .into());
                    }
                    let w_o_head = w_o_heads
                        .get(head)
                        .expect("W_O head pre-copied for edit catalog");
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during edit catalog fit")?;
                        let residual = head_residual(values, head_means, pos);
                        for &space in spaces {
                            let sample = match space {
                                EditCatalogSpace::Hidden => {
                                    project_head_vector_to_hidden(w_o_head, &residual)
                                        .into_iter()
                                        .map(|value| value as f64)
                                        .collect::<Vec<_>>()
                                }
                                EditCatalogSpace::Pca => {
                                    let z = basis.residual_to_z(&residual);
                                    pca_basis.coordinates_with_rank(&z, pca_rank)
                                }
                            };
                            samples
                                .get_mut(&(*head, space))
                                .expect("edit samples missing")
                                .push(sample);
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

    let mut catalogs = HashMap::new();
    for head in heads {
        let basis = bases
            .get(head)
            .ok_or_else(|| format!("missing basis for L{}H{}", head.layer, head.head))?;
        let pca_basis = pca_bases
            .get(head)
            .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;
        let w_o_head = w_o_heads
            .get(head)
            .ok_or_else(|| format!("missing W_O head for L{}H{}", head.layer, head.head))?;
        for &space in spaces {
            let head_samples = samples
                .get(&(*head, space))
                .ok_or_else(|| format!("missing edit samples for L{}H{}", head.layer, head.head))?;
            for &edits in edit_counts {
                let feature_centroids = kmeans_centroids(head_samples, edits, iterations);
                let residual_table = match space {
                    EditCatalogSpace::Hidden => feature_centroids
                        .iter()
                        .map(|centroid| centroid.iter().map(|&value| value as f32).collect())
                        .collect(),
                    EditCatalogSpace::Pca => feature_centroids
                        .iter()
                        .map(|centroid| {
                            let z = pca_basis.reconstruct_from_coordinates(centroid);
                            let residual = basis.z_to_residual(&z);
                            project_head_vector_to_hidden(w_o_head, &residual)
                        })
                        .collect(),
                };
                catalogs.insert(
                    EditCatalogKey {
                        head: *head,
                        space,
                        edits,
                    },
                    EditCatalog {
                        space,
                        feature_centroids,
                        residual_table,
                    },
                );
            }
        }
    }

    Ok(catalogs)
}

fn forward_q4k_oracle_edit_catalog_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    static_hidden: &StaticHiddenTable,
    w_o_head: &[Vec<f32>],
    catalog: &EditCatalog,
    pca_rank: usize,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let hidden_size = weights.hidden_size;
    larql_inference::vindex::predict_kquant_hidden_with_mapped_head_residual_delta(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original_head| {
            let mut replacement_delta = Vec::with_capacity(original_head.nrows() * hidden_size);
            for pos in 0..original_head.nrows() {
                let row = original_head.row(pos);
                let values = row
                    .as_slice()
                    .ok_or("pre-W_O head row was not contiguous during edit catalog eval")?;
                let residual = head_residual(values, means, pos);
                let feature = match catalog.space {
                    EditCatalogSpace::Hidden => project_head_vector_to_hidden(w_o_head, &residual)
                        .into_iter()
                        .map(|value| value as f64)
                        .collect::<Vec<_>>(),
                    EditCatalogSpace::Pca => {
                        let z = basis.residual_to_z(&residual);
                        pca_basis.coordinates_with_rank(&z, pca_rank)
                    }
                };
                let code = nearest_centroid_index(&feature, &catalog.feature_centroids);
                let static_delta = static_hidden.delta_for_position(pos);
                let edit_delta = &catalog.residual_table[code];
                for (&base, &edit) in static_delta.iter().zip(edit_delta.iter()) {
                    replacement_delta.push(base + edit);
                }
            }
            Array2::from_shape_vec((original_head.nrows(), hidden_size), replacement_delta)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(Into::into)
}

#[derive(Debug, Clone)]
struct StaticHiddenTable {
    by_position: Vec<Vec<f32>>,
    global: Vec<f32>,
}

impl StaticHiddenTable {
    fn delta_for_position(&self, position: usize) -> &[f32] {
        self.by_position
            .get(position)
            .map(|delta| delta.as_slice())
            .unwrap_or(&self.global)
    }
}

fn build_static_hidden_tables(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    heads: &[HeadId],
    means: &HashMap<HeadId, StaticHeadMeans>,
) -> Result<HashMap<HeadId, StaticHiddenTable>, Box<dyn std::error::Error>> {
    let w_o_heads = copy_w_o_heads(weights, index, heads)?;
    let mut tables = HashMap::new();
    for head in heads {
        let w_o_head = w_o_heads
            .get(head)
            .ok_or_else(|| format!("missing W_O head for L{}H{}", head.layer, head.head))?;
        let head_means = means
            .get(head)
            .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
        let global = project_head_vector_to_hidden(w_o_head, &head_means.global);
        let by_position = head_means
            .positions
            .iter()
            .map(|mean| project_head_vector_to_hidden(w_o_head, mean))
            .collect();
        tables.insert(
            *head,
            StaticHiddenTable {
                by_position,
                global,
            },
        );
    }
    Ok(tables)
}

fn copy_w_o_heads(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    heads: &[HeadId],
) -> Result<HashMap<HeadId, Vec<Vec<f32>>>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let mut out = HashMap::new();
    for (layer, layer_heads) in heads_by_layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let w_o = weights
            .tensors
            .get(&weights.arch.attn_o_key(layer))
            .ok_or_else(|| format!("missing W_O tensor at layer {layer}"))?;
        let head_dim = weights.arch.head_dim_for_layer(layer);
        for head in layer_heads {
            let start = head.head * head_dim;
            let end = start + head_dim;
            let w_o_head = w_o.slice(s![.., start..end]);
            let rows = (0..w_o_head.nrows())
                .map(|row| {
                    (0..w_o_head.ncols())
                        .map(|col| w_o_head[[row, col]])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            out.insert(head, rows);
        }
        remove_layer_tensors(weights, inserted);
    }
    Ok(out)
}

fn head_residual(values: &[f32], means: &StaticHeadMeans, position: usize) -> Vec<f32> {
    let base = means.positions.get(position).unwrap_or(&means.global);
    values
        .iter()
        .zip(base.iter())
        .map(|(&value, &mean)| value - mean)
        .collect()
}

fn project_head_vector_to_hidden(w_o_head: &[Vec<f32>], values: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; w_o_head.len()];
    for (row_idx, row) in w_o_head.iter().enumerate() {
        let mut sum = 0.0f32;
        for (&value, &weight) in values.iter().zip(row.iter()) {
            sum += value * weight;
        }
        out[row_idx] = sum;
    }
    out
}

#[derive(Debug)]
struct EditCatalogAccumulator {
    prompts: Vec<OracleEditCatalogPromptReport>,
}

impl EditCatalogAccumulator {
    fn new() -> Self {
        Self {
            prompts: Vec::new(),
        }
    }

    fn add(&mut self, prompt: OracleEditCatalogPromptReport) {
        self.prompts.push(prompt);
    }

    fn finish(
        self,
        space: EditCatalogSpace,
        edits: usize,
        hidden_dim: usize,
    ) -> OracleEditCatalogPointReport {
        let kls = self.prompts.iter().map(|p| p.kl).collect::<Vec<_>>();
        OracleEditCatalogPointReport {
            space: space.as_str().to_string(),
            edits,
            address_bits: edits.next_power_of_two().trailing_zeros() as usize,
            residual_table_bytes_bf16: edits * hidden_dim * 2,
            prompts: self.prompts.len(),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            mean_delta_cross_entropy_bits: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.delta_cross_entropy_bits)
                    .collect::<Vec<_>>(),
            ),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts.iter().map(|p| p.baseline_top1_in_catalog_top5),
            ),
            mean_baseline_top1_prob: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_prob)
                    .collect::<Vec<_>>(),
            ),
            mean_catalog_prob_of_baseline_top1: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.catalog_prob_of_baseline_top1)
                    .collect::<Vec<_>>(),
            ),
            mean_baseline_top1_margin: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_margin)
                    .collect::<Vec<_>>(),
            ),
            per_prompt: self.prompts,
        }
    }
}

fn prompt_label(record: &PromptRecord) -> &str {
    record
        .id
        .as_deref()
        .or(record.stratum.as_deref())
        .unwrap_or("prompt")
}

fn parse_string_list(spec: &str) -> Vec<String> {
    spec.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}
