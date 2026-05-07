use std::collections::HashMap;

use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::s;

use super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::metrics::{argmax_usize, code_mass, entropy_bits, js_divergence_bits};
use super::pq::PqCodebook;
use super::reports::{CodeStabilityReport, CodeStabilityStratumReport};
use super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::stats::StaticHeadMeans;
use super::types::{HeadId, PqConfig, PromptRecord};

#[derive(Debug, Clone)]
struct CodeDistributionCounts {
    group_counts: HashMap<usize, Vec<usize>>,
    stratum_group_counts: HashMap<String, HashMap<usize, Vec<usize>>>,
}

impl CodeDistributionCounts {
    fn new(selected_groups: &[usize], levels: usize) -> Self {
        Self {
            group_counts: selected_groups
                .iter()
                .map(|&group| (group, vec![0; levels]))
                .collect(),
            stratum_group_counts: HashMap::new(),
        }
    }

    fn add(&mut self, group: usize, code: usize, stratum: &str, levels: usize) {
        if let Some(counts) = self.group_counts.get_mut(&group) {
            counts[code] += 1;
        }
        self.stratum_group_counts
            .entry(stratum.to_string())
            .or_default()
            .entry(group)
            .or_insert_with(|| vec![0; levels])[code] += 1;
    }
}

pub(super) fn measure_code_stability(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    train_prompts: &[PromptRecord],
    eval_prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
) -> Result<HashMap<(HeadId, PqConfig), Vec<CodeStabilityReport>>, Box<dyn std::error::Error>> {
    let train = collect_code_distribution_counts(
        weights,
        index,
        tokenizer,
        train_prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        selected_groups,
        "code-stability-train",
    )?;
    let eval = collect_code_distribution_counts(
        weights,
        index,
        tokenizer,
        eval_prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        selected_groups,
        "code-stability-eval",
    )?;

    let mut reports = HashMap::new();
    for (head, config) in codebooks.keys() {
        let levels = 1usize << config.bits_per_group;
        let empty_counts = CodeDistributionCounts::new(selected_groups, levels);
        let train_counts = train.get(&(*head, *config)).unwrap_or(&empty_counts);
        let eval_counts = eval.get(&(*head, *config)).unwrap_or(&empty_counts);
        let mut group_reports = Vec::new();
        for &group in selected_groups {
            let train_group = train_counts
                .group_counts
                .get(&group)
                .cloned()
                .unwrap_or_else(|| vec![0; levels]);
            let eval_group = eval_counts
                .group_counts
                .get(&group)
                .cloned()
                .unwrap_or_else(|| vec![0; levels]);
            let train_top = argmax_usize(&train_group);
            let eval_top = argmax_usize(&eval_group);
            let mut stratum_names = train_counts
                .stratum_group_counts
                .keys()
                .chain(eval_counts.stratum_group_counts.keys())
                .cloned()
                .collect::<Vec<_>>();
            stratum_names.sort();
            stratum_names.dedup();
            let by_stratum = stratum_names
                .into_iter()
                .map(|stratum| {
                    let train_s = train_counts
                        .stratum_group_counts
                        .get(&stratum)
                        .and_then(|groups| groups.get(&group))
                        .cloned()
                        .unwrap_or_else(|| vec![0; levels]);
                    let eval_s = eval_counts
                        .stratum_group_counts
                        .get(&stratum)
                        .and_then(|groups| groups.get(&group))
                        .cloned()
                        .unwrap_or_else(|| vec![0; levels]);
                    let train_s_top = argmax_usize(&train_s);
                    let eval_s_top = argmax_usize(&eval_s);
                    CodeStabilityStratumReport {
                        stratum,
                        train_positions: train_s.iter().sum(),
                        eval_positions: eval_s.iter().sum(),
                        train_entropy_bits: entropy_bits(&train_s),
                        eval_entropy_bits: entropy_bits(&eval_s),
                        train_top_code: train_s_top,
                        train_top_code_mass: code_mass(&train_s, train_s_top),
                        eval_top_code: eval_s_top,
                        eval_top_code_mass: code_mass(&eval_s, eval_s_top),
                        train_eval_js_bits: js_divergence_bits(&train_s, &eval_s),
                    }
                })
                .collect();
            group_reports.push(CodeStabilityReport {
                group,
                train_positions: train_group.iter().sum(),
                eval_positions: eval_group.iter().sum(),
                train_entropy_bits: entropy_bits(&train_group),
                eval_entropy_bits: entropy_bits(&eval_group),
                train_top_code: train_top,
                train_top_code_mass: code_mass(&train_group, train_top),
                eval_top_code: eval_top,
                eval_top_code_mass: code_mass(&eval_group, eval_top),
                train_eval_js_bits: js_divergence_bits(&train_group, &eval_group),
                by_stratum,
            });
        }
        reports.insert((*head, *config), group_reports);
    }

    Ok(reports)
}

fn collect_code_distribution_counts(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
    label_prefix: &str,
) -> Result<HashMap<(HeadId, PqConfig), CodeDistributionCounts>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let mut counts = HashMap::new();
    for (head, config) in codebooks.keys() {
        counts.insert(
            (*head, *config),
            CodeDistributionCounts::new(selected_groups, 1usize << config.bits_per_group),
        );
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!(
            "  {label_prefix} [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
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
                    let basis = bases.get(head).ok_or_else(|| {
                        format!("missing basis for L{}H{}", head.layer, head.head)
                    })?;
                    let head_means = means.get(head).ok_or_else(|| {
                        format!("missing means for L{}H{}", head.layer, head.head)
                    })?;
                    let pca_basis = pca_bases.get(head).ok_or_else(|| {
                        format!("missing PCA basis for L{}H{}", head.layer, head.head)
                    })?;
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    let head_codebooks = codebooks
                        .iter()
                        .filter(|((codebook_head, _), _)| codebook_head == head)
                        .collect::<Vec<_>>();
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during code stability")?;
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual = values
                            .iter()
                            .zip(base.iter())
                            .map(|(&yi, &bi)| yi - bi)
                            .collect::<Vec<_>>();
                        let z = basis.residual_to_z(&residual);
                        for ((_, config), codebook) in &head_codebooks {
                            let coords = pca_basis.coordinates_with_rank(&z, config.k);
                            let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
                            let levels = 1usize << config.bits_per_group;
                            let point_counts =
                                counts.get_mut(&(*head, *config)).ok_or_else(|| {
                                    format!(
                                        "missing code stability counts for L{}H{} {:?}",
                                        head.layer, head.head, config
                                    )
                                })?;
                            for &group in selected_groups {
                                point_counts.add(group, codes[group], stratum, levels);
                            }
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

    Ok(counts)
}
