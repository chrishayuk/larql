use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use clap::Args;
use larql_inference::attention::{
    run_attention_block_with_pre_o_and_all_attention_weights, SharedKV,
};
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use ndarray::{s, Array2};
use serde::Serialize;

use super::address::attention_argmax;
use super::basis::{build_roundtrip_bases, fit_z_pca_bases};
use super::input::{
    limit_prompts_per_stratum, load_prompts, parse_head_spec, split_prompt_records,
};
use super::metrics::{argmax, bool_rate, kl_logp, log_softmax, mean, percentile, top_k_indices};
use super::oracle_pq_fit::fit_pq_codebooks;
use super::oracle_pq_forward::{
    final_logits, forward_q4k_oracle_pq_head, forward_q4k_predicted_address_mode_d_head,
};
use super::oracle_pq_mode_d::materialize_mode_d_tables;
use super::program::{PositionContext, Program};
use super::static_replace::fit_static_means;
use super::types::{HeadId, PqConfig, PromptRecord};

#[derive(Args)]
pub struct ProbeProgramClassArgs {
    #[arg(long)]
    pub index: PathBuf,

    #[arg(long)]
    pub program: PathBuf,

    #[arg(long)]
    pub prompts: PathBuf,

    #[arg(long)]
    pub out: PathBuf,

    /// Optional override/guard for the program head, formatted as layer:head.
    #[arg(long)]
    pub head: Option<String>,

    /// Optional override/guard for the program group.
    #[arg(long)]
    pub group: Option<usize>,

    /// Comma-separated source list: residual_input,pre_wo_head_output,symbolic.
    #[arg(long, default_value = "residual_input,pre_wo_head_output,symbolic")]
    pub sources: String,

    /// Maximum prompts per stratum. 0 = unlimited.
    #[arg(long, default_value_t = 0)]
    pub max_per_stratum: usize,

    #[arg(long, default_value_t = 1)]
    pub eval_mod: usize,

    #[arg(long, default_value_t = 0)]
    pub eval_offset: usize,

    #[arg(long, default_value_t = 1e-6)]
    pub sigma_rel_cutoff: f64,

    #[arg(long, default_value_t = 25)]
    pub pq_iters: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
enum ProbeSource {
    ResidualInput,
    PreWoHeadOutput,
    Symbolic,
}

impl ProbeSource {
    fn parse_list(spec: &str) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let mut out = Vec::new();
        for part in spec.split(',') {
            let source = match part.trim() {
                "" => continue,
                "residual_input" => Self::ResidualInput,
                "pre_wo_head_output" => Self::PreWoHeadOutput,
                "symbolic" => Self::Symbolic,
                other => return Err(format!("unknown probe source '{other}'").into()),
            };
            if !out.contains(&source) {
                out.push(source);
            }
        }
        if out.is_empty() {
            return Err("--sources must name at least one source".into());
        }
        Ok(out)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ResidualInput => "residual_input",
            Self::PreWoHeadOutput => "pre_wo_head_output",
            Self::Symbolic => "symbolic",
        }
    }
}

#[derive(Clone)]
struct ProbePrompt {
    id: String,
    stratum: String,
    token_ids: Vec<u32>,
    oracle_codes: Vec<Vec<usize>>,
    target_classes: Vec<usize>,
    features: BTreeMap<ProbeSource, Vec<Vec<f32>>>,
    baseline_logp: Vec<f64>,
    baseline_top1: u32,
}

#[derive(Default)]
struct FlatDataset {
    features: Vec<Vec<f32>>,
    labels: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct ClassMetrics {
    accuracy: f64,
    macro_f1: f64,
    per_class_f1: BTreeMap<usize, f64>,
    confusion: Vec<Vec<usize>>,
}

#[derive(Debug, Serialize)]
struct ReplacementMetrics {
    mean_kl: f64,
    p95_kl: f64,
    max_kl: f64,
    top1_agreement: f64,
    top5_retention: f64,
}

#[derive(Debug, Serialize)]
struct SourceReport {
    source: ProbeSource,
    classifier: &'static str,
    train_rows: usize,
    eval_rows: usize,
    input_dim: usize,
    class_metrics: ClassMetrics,
    replacement_metrics: ReplacementMetrics,
}

#[derive(Debug, Serialize)]
struct ProbeProgramClassReport {
    program_name: Option<String>,
    head: HeadId,
    group: usize,
    base_config_k: usize,
    base_config_groups: usize,
    base_config_bits_per_group: usize,
    fit_prompts: usize,
    eval_prompts: usize,
    classes: Vec<usize>,
    oracle_program_replacement: ReplacementMetrics,
    sources: Vec<SourceReport>,
}

#[derive(Clone)]
struct CentroidProbe {
    class_codes: Vec<usize>,
    mean: Vec<f64>,
    inv_std: Vec<f64>,
    centroids: Vec<Vec<f64>>,
    centroid_norms: Vec<f64>,
}

impl CentroidProbe {
    fn fit(rows: &[Vec<f32>], labels: &[usize], class_codes: &[usize]) -> Result<Self, String> {
        if rows.is_empty() {
            return Err("cannot fit probe on empty dataset".into());
        }
        let dim = rows[0].len();
        if dim == 0 {
            return Err("cannot fit probe with zero-dimensional rows".into());
        }
        if rows.iter().any(|row| row.len() != dim) {
            return Err("probe feature rows have inconsistent dimensions".into());
        }
        let mut mean = vec![0.0; dim];
        for row in rows {
            for (dst, &value) in mean.iter_mut().zip(row.iter()) {
                *dst += value as f64;
            }
        }
        let inv_n = 1.0 / rows.len() as f64;
        for value in &mut mean {
            *value *= inv_n;
        }
        let mut var = vec![0.0; dim];
        for row in rows {
            for (idx, &value) in row.iter().enumerate() {
                let d = value as f64 - mean[idx];
                var[idx] += d * d;
            }
        }
        let inv_std = var
            .into_iter()
            .map(|v| {
                let std = (v * inv_n).sqrt();
                if std > 1e-12 {
                    1.0 / std
                } else {
                    1.0
                }
            })
            .collect::<Vec<_>>();

        let class_to_idx = class_codes
            .iter()
            .enumerate()
            .map(|(idx, &code)| (code, idx))
            .collect::<HashMap<_, _>>();
        let mut centroids = vec![vec![0.0; dim]; class_codes.len()];
        let mut counts = vec![0usize; class_codes.len()];
        for (row, &label) in rows.iter().zip(labels.iter()) {
            let Some(&class_idx) = class_to_idx.get(&label) else {
                continue;
            };
            counts[class_idx] += 1;
            for idx in 0..dim {
                centroids[class_idx][idx] += ((row[idx] as f64) - mean[idx]) * inv_std[idx];
            }
        }
        for (centroid, &count) in centroids.iter_mut().zip(counts.iter()) {
            if count == 0 {
                continue;
            }
            let inv_count = 1.0 / count as f64;
            for value in centroid {
                *value *= inv_count;
            }
        }
        let centroid_norms = centroids
            .iter()
            .map(|c| c.iter().map(|v| v * v).sum::<f64>())
            .collect();

        Ok(Self {
            class_codes: class_codes.to_vec(),
            mean,
            inv_std,
            centroids,
            centroid_norms,
        })
    }

    fn predict(&self, row: &[f32]) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (class_idx, centroid) in self.centroids.iter().enumerate() {
            let mut dot = 0.0;
            for idx in 0..row.len() {
                dot += ((row[idx] as f64) - self.mean[idx]) * self.inv_std[idx] * centroid[idx];
            }
            let score = dot - 0.5 * self.centroid_norms[class_idx];
            if score > best_score {
                best_score = score;
                best_idx = class_idx;
            }
        }
        self.class_codes[best_idx]
    }

    fn predict_many(&self, rows: &[Vec<f32>]) -> Vec<usize> {
        rows.iter().map(|row| self.predict(row)).collect()
    }
}

pub(super) fn run_probe_program_class(
    args: ProbeProgramClassArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    let sources = ProbeSource::parse_list(&args.sources)?;
    let program_text = std::fs::read_to_string(&args.program)?;
    let mut program: Program = serde_json::from_str(&program_text)?;
    program.validate()?;
    program.normalize();

    let mut head = program.head;
    if let Some(spec) = args.head.as_deref() {
        let parsed = parse_head_spec(spec)?;
        head = parsed.into_iter().next().ok_or("--head was empty")?;
        if head != program.head {
            eprintln!(
                "WARNING: --head L{}H{} overrides program head L{}H{}",
                head.layer, head.head, program.head.layer, program.head.head
            );
        }
    }
    let group = args.group.unwrap_or(program.group);
    if group != program.group {
        eprintln!(
            "WARNING: --group {group} overrides program group {}",
            program.group
        );
    }

    let config = PqConfig::from(&program.base_config);
    eprintln!(
        "Probe program class: L{}H{} group {} {}:{}:{} sources={}",
        head.layer,
        head.head,
        group,
        config.k,
        config.groups,
        config.bits_per_group,
        sources
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_kquant(&args.index)?;
    index.load_interleaved_kquant(&args.index)?;
    let mut weights = load_model_weights_kquant(&args.index, &mut cb)?;
    if weights.arch.is_hybrid_moe() {
        return Err("ov-rd probe-program-class currently supports dense FFN vindexes only".into());
    }
    let tokenizer = load_vindex_tokenizer(&args.index)?;

    let mut all_records = load_prompts(&args.prompts, None)?;
    if args.max_per_stratum > 0 {
        all_records = limit_prompts_per_stratum(all_records, args.max_per_stratum);
    }
    let strata = strata_vocab(&all_records);
    let (fit_prompts, eval_prompts) =
        split_prompt_records(&all_records, args.eval_mod, args.eval_offset)?;
    eprintln!(
        "Prompts: {} fit, {} eval",
        fit_prompts.len(),
        eval_prompts.len()
    );

    let selected_heads = vec![head];
    let configs = vec![config];

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

    eprintln!("Fitting product quantizers");
    let codebooks = fit_pq_codebooks(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &configs,
        args.pq_iters,
        &[],
    )?;

    eprintln!("Materializing Mode D residual-space table");
    let mode_d_tables = materialize_mode_d_tables(
        &mut weights,
        &index,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &codebooks,
        &[],
    )?;
    let mode_d_table = mode_d_tables
        .get(&(head, config))
        .ok_or_else(|| format!("Mode D table missing for L{}H{}", head.layer, head.head))?;
    let basis = bases.get(&head).ok_or("W_O basis missing")?;
    let pca_basis = pca_bases.get(&head).ok_or("PCA basis missing")?;
    let head_means = means.get(&head).ok_or("Position means missing")?;
    let codebook = codebooks.get(&(head, config)).ok_or("Codebook missing")?;

    eprintln!("Capturing fit probe rows");
    let fit_captures = capture_probe_prompts(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        head,
        group,
        &program,
        basis,
        pca_basis,
        head_means,
        codebook,
        &strata,
    )?;
    eprintln!("Capturing eval probe rows");
    let eval_captures = capture_probe_prompts(
        &mut weights,
        &index,
        &tokenizer,
        &eval_prompts,
        head,
        group,
        &program,
        basis,
        pca_basis,
        head_means,
        codebook,
        &strata,
    )?;

    let classes = class_vocab(&program, &fit_captures, &eval_captures);
    validate_quotient_classes(&program, &classes)?;
    eprintln!("Behavioral classes: {:?}", classes);

    let oracle_program_replacement = evaluate_replacement(
        &mut weights,
        &index,
        head,
        group,
        mode_d_table,
        &eval_captures,
        |prompt, _| prompt.target_classes.clone(),
    )?;

    let mut source_reports = Vec::new();
    for &source in &sources {
        let train = flatten_source(&fit_captures, source)?;
        let eval = flatten_source(&eval_captures, source)?;
        let probe = CentroidProbe::fit(&train.features, &train.labels, &classes)?;
        let predictions = probe.predict_many(&eval.features);
        let class_metrics = class_metrics(&eval.labels, &predictions, &classes);
        let replacement_metrics = evaluate_replacement(
            &mut weights,
            &index,
            head,
            group,
            mode_d_table,
            &eval_captures,
            |_prompt, features| {
                features
                    .get(&source)
                    .expect("source features already validated")
                    .iter()
                    .map(|row| probe.predict(row))
                    .collect::<Vec<_>>()
            },
        )?;

        let input_dim = train.features.first().map(|row| row.len()).unwrap_or(0);
        eprintln!(
            "{}: acc {:.4} macro-F1 {:.4} replacement KL mean {:.6} p95 {:.6}",
            source.as_str(),
            class_metrics.accuracy,
            class_metrics.macro_f1,
            replacement_metrics.mean_kl,
            replacement_metrics.p95_kl
        );
        source_reports.push(SourceReport {
            source,
            classifier: "standardized_nearest_centroid_linear",
            train_rows: train.features.len(),
            eval_rows: eval.features.len(),
            input_dim,
            class_metrics,
            replacement_metrics,
        });
    }

    let report = ProbeProgramClassReport {
        program_name: program.name.clone(),
        head,
        group,
        base_config_k: config.k,
        base_config_groups: config.groups,
        base_config_bits_per_group: config.bits_per_group,
        fit_prompts: fit_captures.len(),
        eval_prompts: eval_captures.len(),
        classes,
        oracle_program_replacement,
        sources: source_reports,
    };

    let out_path = args.out.join("probe_program_class.json");
    serde_json::to_writer_pretty(std::fs::File::create(&out_path)?, &report)?;
    eprintln!("Wrote {}", out_path.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn capture_probe_prompts(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    records: &[PromptRecord],
    head: HeadId,
    group: usize,
    program: &Program,
    basis: &super::basis::WoRoundtripBasis,
    pca_basis: &super::basis::ZPcaBasis,
    head_means: &super::stats::StaticHeadMeans,
    codebook: &super::pq::PqCodebook,
    strata: &[String],
) -> Result<Vec<ProbePrompt>, Box<dyn std::error::Error>> {
    let mut captures = Vec::with_capacity(records.len());
    for (idx, record) in records.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", idx + 1, records.len(), label);

        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");

        let baseline_h =
            larql_inference::vindex::predict_kquant_hidden(weights, &token_ids, index, None);
        let baseline_logits = final_logits(weights, &baseline_h);
        let baseline_logp = log_softmax(&baseline_logits);
        let baseline_top1 = argmax(&baseline_logits);

        let (_, _, oracle_codes) = forward_q4k_oracle_pq_head(
            weights, &token_ids, index, head, basis, pca_basis, head_means, codebook, stratum,
        )?;
        let target_features = capture_target_features(weights, &token_ids, index, head)?;

        let mut target_classes = Vec::with_capacity(token_ids.len());
        let mut symbolic_rows = Vec::with_capacity(token_ids.len());
        for pos in 0..token_ids.len() {
            let original = oracle_codes
                .get(pos)
                .and_then(|codes| codes.get(group))
                .copied()
                .ok_or("oracle code capture missing target group")?;
            let attn_row = target_features
                .attention_rows
                .get(pos)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let attn_argmax = attention_argmax(attn_row, pos);
            let ctx = PositionContext {
                stratum: stratum.to_string(),
                position: pos,
                token_id: token_ids[pos],
                prev_token_id: (pos > 0).then(|| token_ids.get(pos - 1).copied()).flatten(),
                attends_bos: attn_argmax == 0,
                attends_prev: pos > 0 && attn_argmax + 1 == pos,
                original_code: original,
                current_code: original,
            };
            target_classes.push(program.apply_to_code(original, &ctx));
            symbolic_rows.push(symbolic_features(
                tokenizer, strata, stratum, &token_ids, pos, &ctx,
            ));
        }

        let mut features = BTreeMap::new();
        features.insert(
            ProbeSource::ResidualInput,
            array_rows(&target_features.residual_input),
        );
        features.insert(
            ProbeSource::PreWoHeadOutput,
            array_rows(&target_features.pre_wo_head_output),
        );
        features.insert(ProbeSource::Symbolic, symbolic_rows);

        captures.push(ProbePrompt {
            id: label.to_string(),
            stratum: stratum.to_string(),
            token_ids,
            oracle_codes,
            target_classes,
            features,
            baseline_logp,
            baseline_top1,
        });
    }
    Ok(captures)
}

struct TargetFeatures {
    residual_input: Array2<f32>,
    pre_wo_head_output: Array2<f32>,
    attention_rows: Vec<Vec<f32>>,
}

fn capture_target_features(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<TargetFeatures, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..=head.layer {
        let inserted = super::runtime::insert_q4k_layer_tensors(weights, index, layer)?;
        if layer == head.layer {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let (_, pre_o, all_weights) = run_attention_block_with_pre_o_and_all_attention_weights(
                weights, &h, layer, shared_kv,
            )
            .ok_or_else(|| {
                format!(
                    "probe feature capture failed at L{}H{}",
                    head.layer, head.head
                )
            })?;
            super::runtime::remove_layer_tensors(weights, inserted);

            let head_dim = weights.head_dim;
            let start = head.head * head_dim;
            let end = start + head_dim;
            if end > pre_o.ncols() {
                return Err(format!(
                    "head {} out of range for pre-W_O width {}",
                    head.head,
                    pre_o.ncols()
                )
                .into());
            }
            let pre_wo_head_output = pre_o.slice(s![.., start..end]).to_owned();
            let attention_rows = all_weights.heads.get(head.head).cloned().ok_or_else(|| {
                format!(
                    "attention weights missing for L{}H{}",
                    head.layer, head.head
                )
            })?;
            return Ok(TargetFeatures {
                residual_input: h,
                pre_wo_head_output,
                attention_rows,
            });
        }

        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            run_layer_with_ffn(
                weights,
                &h,
                layer,
                &ffn,
                false,
                ple_inputs.get(layer),
                shared_kv,
            )
            .map(|(h_new, _, kv_out)| (h_new, kv_out))
        };
        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            super::runtime::remove_layer_tensors(weights, inserted);
            return Err(format!("layer {layer} returned no output").into());
        }
        super::runtime::remove_layer_tensors(weights, inserted);
    }

    Err(format!("target layer {} was not reached", head.layer).into())
}

fn array_rows(array: &Array2<f32>) -> Vec<Vec<f32>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn symbolic_features(
    tokenizer: &tokenizers::Tokenizer,
    strata: &[String],
    stratum: &str,
    token_ids: &[u32],
    pos: usize,
    ctx: &PositionContext,
) -> Vec<f32> {
    let mut out = Vec::new();
    for known in strata {
        out.push((known == stratum) as u8 as f32);
    }
    let bucket = position_bucket(pos);
    for idx in 0..8 {
        out.push((idx == bucket) as u8 as f32);
    }
    out.push((pos == 0) as u8 as f32);
    out.push((pos + 1 == token_ids.len()) as u8 as f32);
    out.push(ctx.attends_bos as u8 as f32);
    out.push(ctx.attends_prev as u8 as f32);

    let token_text = tokenizer
        .decode(&[token_ids[pos]], false)
        .unwrap_or_default();
    out.push(token_text.chars().any(|c| c.is_ascii_digit()) as u8 as f32);
    out.push(token_text.chars().any(|c| c.is_ascii_alphabetic()) as u8 as f32);
    out.push(token_text.chars().any(|c| c.is_ascii_punctuation()) as u8 as f32);
    out.push(token_text.chars().any(|c| c.is_whitespace()) as u8 as f32);
    out.push((token_text.starts_with(' ') || token_text.starts_with('▁')) as u8 as f32);
    out.push((token_text.len() <= 1) as u8 as f32);
    out.push((token_ids[pos] as f32).ln_1p() / 16.0);
    if pos > 0 {
        out.push((token_ids[pos - 1] as f32).ln_1p() / 16.0);
    } else {
        out.push(0.0);
    }
    out
}

fn position_bucket(pos: usize) -> usize {
    match pos {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4..=7 => 4,
        8..=15 => 5,
        16..=31 => 6,
        _ => 7,
    }
}

fn flatten_source(
    prompts: &[ProbePrompt],
    source: ProbeSource,
) -> Result<FlatDataset, Box<dyn std::error::Error>> {
    let mut out = FlatDataset::default();
    for prompt in prompts {
        let rows = prompt
            .features
            .get(&source)
            .ok_or_else(|| format!("missing source {}", source.as_str()))?;
        if rows.len() != prompt.target_classes.len() {
            return Err(format!(
                "source {} row count mismatch for {}",
                source.as_str(),
                prompt.id
            )
            .into());
        }
        out.features.extend(rows.iter().cloned());
        out.labels.extend(prompt.target_classes.iter().copied());
    }
    Ok(out)
}

fn strata_vocab(records: &[PromptRecord]) -> Vec<String> {
    let mut set = BTreeSet::new();
    for record in records {
        set.insert(
            record
                .stratum
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        );
    }
    set.into_iter().collect()
}

fn class_vocab(program: &Program, fit: &[ProbePrompt], eval: &[ProbePrompt]) -> Vec<usize> {
    let mut set = BTreeSet::new();
    for tc in &program.terminal_classes {
        set.insert(tc.representative_code);
    }
    for prompt in fit.iter().chain(eval.iter()) {
        for &class_code in &prompt.target_classes {
            set.insert(class_code);
        }
    }
    set.into_iter().collect()
}

fn validate_quotient_classes(
    program: &Program,
    observed_classes: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let declared = program
        .terminal_classes
        .iter()
        .map(|tc| tc.representative_code)
        .collect::<BTreeSet<_>>();
    if declared.is_empty() {
        return Err(
            "program declares no terminal_classes; probe target would be raw PQ codes".into(),
        );
    }
    let leaked = observed_classes
        .iter()
        .copied()
        .filter(|code| !declared.contains(code))
        .collect::<Vec<_>>();
    if !leaked.is_empty() {
        return Err(format!(
            "program leaves raw PQ codes outside the behavioral quotient: {:?}; use a program that canonicalizes every observed code into terminal_classes",
            leaked
        )
        .into());
    }
    Ok(())
}

fn class_metrics(truth: &[usize], pred: &[usize], classes: &[usize]) -> ClassMetrics {
    let class_to_idx = classes
        .iter()
        .enumerate()
        .map(|(idx, &code)| (code, idx))
        .collect::<HashMap<_, _>>();
    let mut confusion = vec![vec![0usize; classes.len()]; classes.len()];
    let mut correct = 0usize;
    for (&t, &p) in truth.iter().zip(pred.iter()) {
        if t == p {
            correct += 1;
        }
        if let (Some(&ti), Some(&pi)) = (class_to_idx.get(&t), class_to_idx.get(&p)) {
            confusion[ti][pi] += 1;
        }
    }
    let mut per_class_f1 = BTreeMap::new();
    let mut f1s = Vec::new();
    for (idx, &class_code) in classes.iter().enumerate() {
        let tp = confusion[idx][idx] as f64;
        let row_sum = confusion[idx].iter().sum::<usize>() as f64;
        let col_sum = confusion.iter().map(|row| row[idx]).sum::<usize>() as f64;
        let precision = if col_sum > 0.0 { tp / col_sum } else { 0.0 };
        let recall = if row_sum > 0.0 { tp / row_sum } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        per_class_f1.insert(class_code, f1);
        f1s.push(f1);
    }
    ClassMetrics {
        accuracy: if truth.is_empty() {
            0.0
        } else {
            correct as f64 / truth.len() as f64
        },
        macro_f1: mean(&f1s),
        per_class_f1,
        confusion,
    }
}

fn evaluate_replacement<F>(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    head: HeadId,
    group: usize,
    mode_d_table: &super::pq::ModeDTable,
    prompts: &[ProbePrompt],
    mut predicted_classes: F,
) -> Result<ReplacementMetrics, Box<dyn std::error::Error>>
where
    F: FnMut(&ProbePrompt, &BTreeMap<ProbeSource, Vec<Vec<f32>>>) -> Vec<usize>,
{
    let mut prompt_kls = Vec::new();
    let mut top1_agree = Vec::new();
    let mut top5_keep = Vec::new();
    for prompt in prompts {
        let predicted = predicted_classes(prompt, &prompt.features);
        let mut remapped_codes = prompt.oracle_codes.clone();
        for (codes, &class_code) in remapped_codes.iter_mut().zip(predicted.iter()) {
            if group < codes.len() {
                codes[group] = class_code;
            }
        }
        let h = forward_q4k_predicted_address_mode_d_head(
            weights,
            &prompt.token_ids,
            index,
            head,
            mode_d_table,
            &remapped_codes,
            &prompt.stratum,
        )?;
        let logits = final_logits(weights, &h);
        let logp = log_softmax(&logits);
        let top1 = argmax(&logits);
        let top5 = top_k_indices(&logits, 5);
        prompt_kls.push(kl_logp(&prompt.baseline_logp, &logp));
        top1_agree.push(top1 == prompt.baseline_top1);
        top5_keep.push(top5.contains(&prompt.baseline_top1));
    }
    Ok(ReplacementMetrics {
        mean_kl: mean(&prompt_kls),
        p95_kl: percentile(prompt_kls.clone(), 0.95),
        max_kl: prompt_kls.iter().copied().fold(0.0_f64, f64::max),
        top1_agreement: bool_rate(top1_agree.into_iter()),
        top5_retention: bool_rate(top5_keep.into_iter()),
    })
}
