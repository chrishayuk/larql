mod args;
mod report;

pub(super) use args::EvalProgramArgs;

use std::collections::HashMap;

use larql_inference::encode_prompt;
use larql_vindex::{load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks};
use serde::Serialize;

use super::address::attention_argmax;
use super::basis::{build_roundtrip_bases, fit_z_pca_bases};
use super::input::{limit_prompts_per_stratum, load_prompts, split_prompt_records};
use super::metrics::{argmax, bool_rate, kl_logp, log_softmax, mean, percentile, top_k_indices};
use super::oracle_pq_fit::fit_pq_codebooks;
use super::oracle_pq_forward::{
    capture_attention_relation_rows, final_logits, forward_q4k_oracle_pq_head,
    forward_q4k_oracle_pq_mode_d_head, forward_q4k_predicted_address_mode_d_head,
};
use super::oracle_pq_mode_d::materialize_mode_d_tables;
use super::program::{
    fingerprint::codebook_fingerprint, BehaviorMetrics, PositionContext, Program,
};
use super::static_replace::fit_static_means;
use super::types::PqConfig;
use report::{EvalProgramReport, PromptReport, StratumReport};

// ────────────────────────────────────────────────────────────────────────────
// Diagnostic accumulators
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Serialize)]
struct CodeHistogram {
    counts: HashMap<usize, usize>,
}

impl CodeHistogram {
    fn add(&mut self, code: usize) {
        *self.counts.entry(code).or_default() += 1;
    }
    fn sorted_desc(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<_> = self.counts.iter().map(|(&k, &v)| (k, v)).collect();
        v.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        v
    }
    fn total(&self) -> usize {
        self.counts.values().sum()
    }
}

#[derive(Debug, Default)]
struct Diagnostics {
    original_codes: CodeHistogram,
    canonical_codes: CodeHistogram,
    final_classes: CodeHistogram,
    guard_fired: usize,
    guard_eligible: usize,
    oracle_mode_d_kls: Vec<f64>,
}

impl Diagnostics {
    fn oracle_metrics(&self) -> Option<(f64, f64, f64)> {
        if self.oracle_mode_d_kls.is_empty() {
            return None;
        }
        let kls = &self.oracle_mode_d_kls;
        Some((
            mean(kls),
            percentile(kls.clone(), 0.95),
            kls.iter().cloned().fold(0.0_f64, f64::max),
        ))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Entry point
// ────────────────────────────────────────────────────────────────────────────

pub(super) fn run_eval_program(args: EvalProgramArgs) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    let program_text = std::fs::read_to_string(&args.program)?;
    let mut program: Program = serde_json::from_str(&program_text)?;
    program.validate()?;
    program.normalize();

    let head = program.head;
    let config: PqConfig = PqConfig::from(&program.base_config);

    // Print identity + source so the first line of output is always traceable.
    eprintln!(
        "Program: {} — L{}H{} group {} {}:{}:{}",
        program.name.as_deref().unwrap_or("<unnamed>"),
        head.layer,
        head.head,
        program.group,
        config.k,
        config.groups,
        config.bits_per_group,
    );
    if let Some(src) = program
        .reference_metrics
        .as_ref()
        .and_then(|rm| rm.get("source"))
        .and_then(|v| v.as_str())
    {
        eprintln!("Reference: {src}");
    } else {
        eprintln!("Reference: (no source declared)");
    }

    let mut cb = SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_kquant(&args.index)?;
    index.load_interleaved_kquant(&args.index)?;
    let mut weights = load_model_weights_kquant(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;

    let metal_backend = super::metal_backend::init(args.metal);

    let mut all_records = load_prompts(&args.prompts, None)?;
    if args.max_per_stratum > 0 {
        all_records = limit_prompts_per_stratum(all_records, args.max_per_stratum);
    }
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

    // Compute and validate codebook fingerprint.
    let actual_fingerprint = codebooks
        .get(&(head, config))
        .map(|cb| codebook_fingerprint(&cb.centroids));
    let expected_fingerprint = program.codebook_fingerprint.clone();
    let fingerprint_match = match (&expected_fingerprint, &actual_fingerprint) {
        (Some(exp), Some(act)) => {
            let m = exp == act;
            if !m {
                eprintln!(
                    "WARNING: codebook fingerprint mismatch — program was authored against {exp}, got {act}"
                );
                eprintln!("         Program's numeric code labels may not match this codebook.");
            } else {
                eprintln!("Codebook fingerprint: {act} (matches program)");
            }
            Some(m)
        }
        (None, Some(act)) => {
            eprintln!("Codebook fingerprint: {act} (no reference in program file)");
            None
        }
        _ => None,
    };

    eprintln!("Materializing Mode D residual-space tables");
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
    let basis = bases
        .get(&head)
        .ok_or_else(|| format!("W_O basis missing for L{}H{}", head.layer, head.head))?;
    let pca_basis = pca_bases
        .get(&head)
        .ok_or_else(|| format!("PCA basis missing for L{}H{}", head.layer, head.head))?;
    let head_means = means
        .get(&head)
        .ok_or_else(|| format!("Position means missing for L{}H{}", head.layer, head.head))?;
    let codebook = codebooks
        .get(&(head, config))
        .ok_or_else(|| format!("Codebook missing for L{}H{}", head.layer, head.head))?;

    eprintln!("Evaluating program on {} prompts", eval_prompts.len());
    let mut prompt_reports: Vec<PromptReport> = Vec::new();
    let mut actual_intervention_backend: &'static str = "cpu_fallback";
    let mut diag = Diagnostics::default();
    let target_group = program.group;

    for (idx, record) in eval_prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", idx + 1, eval_prompts.len(), label);

        let token_ids = encode_prompt(&tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");

        // Pass 1: baseline — Metal if available, CPU fallback.
        let baseline_h = if let Some(ref b) = metal_backend {
            if let Some(h) =
                super::metal_backend::try_metal_baseline(&weights, &token_ids, &index, b)
            {
                h
            } else {
                larql_inference::vindex::predict_kquant_hidden(
                    &mut weights,
                    &token_ids,
                    &index,
                    None,
                )
            }
        } else {
            larql_inference::vindex::predict_kquant_hidden(&mut weights, &token_ids, &index, None)
        };
        let baseline_logits = final_logits(&weights, &baseline_h);
        let baseline_logp = log_softmax(&baseline_logits);
        let baseline_top1 = argmax(&baseline_logits);

        // Pass 2: oracle PQ codes — Metal capture if available, CPU fallback.
        // Metal: runs only layers 0..=target_layer on GPU, captures pre-W_O output,
        // then computes PQ codes on CPU. ~34× faster than full CPU forward pass for L0.
        let oracle_codes_by_position: Vec<Vec<usize>> = if let Some(ref b) = metal_backend {
            if let Some(pre_wo) = super::metal_backend::try_metal_capture_pre_wo(
                &weights, &token_ids, &index, head.layer, head.head, b,
            ) {
                // pre_wo: [seq_len × head_dim] f32 captured from attn_outs[target_layer]
                let head_dim = pre_wo.len() / token_ids.len();
                (0..token_ids.len())
                    .map(|pos| {
                        let values = &pre_wo[pos * head_dim..(pos + 1) * head_dim];
                        let base = head_means.positions.get(pos).unwrap_or(&head_means.global);
                        let residual: Vec<f32> =
                            values.iter().zip(base).map(|(yi, bi)| yi - bi).collect();
                        let z = basis.residual_to_z(&residual);
                        let coords = pca_basis.coordinates_with_rank(&z, codebook.config.k);
                        codebook.quantize_indices_for_stratum(&coords, stratum)
                    })
                    .collect()
            } else {
                // Fall back to CPU oracle PQ
                let (_, _, codes) = forward_q4k_oracle_pq_head(
                    &mut weights,
                    &token_ids,
                    &index,
                    head,
                    basis,
                    pca_basis,
                    head_means,
                    codebook,
                    stratum,
                )?;
                codes
            }
        } else {
            let (_, _, codes) = forward_q4k_oracle_pq_head(
                &mut weights,
                &token_ids,
                &index,
                head,
                basis,
                pca_basis,
                head_means,
                codebook,
                stratum,
            )?;
            codes
        };

        // Pass 3: oracle Mode D baseline — Metal if available, CPU fallback.
        let oracle_h = if let Some(ref b) = metal_backend {
            let oracle_delta_flat = build_replacement_delta(
                mode_d_table,
                &oracle_codes_by_position,
                stratum,
                weights.hidden_size,
            );
            let oracle_delta = ndarray::Array2::from_shape_vec(
                (token_ids.len(), weights.hidden_size),
                oracle_delta_flat,
            )
            .ok();
            oracle_delta.and_then(|d| {
                super::metal_backend::try_metal(
                    &weights, &token_ids, &index, head.layer, head.head, &d, b,
                )
            })
        } else {
            None
        };
        let oracle_h = match oracle_h {
            Some(h) => h,
            None => forward_q4k_oracle_pq_mode_d_head(
                &mut weights,
                &token_ids,
                &index,
                head,
                basis,
                pca_basis,
                head_means,
                codebook,
                mode_d_table,
                stratum,
            )?,
        };
        let oracle_logp = log_softmax(&final_logits(&weights, &oracle_h));
        diag.oracle_mode_d_kls
            .push(kl_logp(&baseline_logp, &oracle_logp));

        // Pass 4: attention rows for guard evaluation.
        let attention_rows =
            capture_attention_relation_rows(&mut weights, &token_ids, &index, head)?;

        // Apply program; accumulate diagnostics.
        let remapped_codes: Vec<Vec<usize>> = oracle_codes_by_position
            .iter()
            .enumerate()
            .map(|(pos, oracle_codes)| {
                let mut codes = oracle_codes.clone();
                let original = oracle_codes[target_group];

                let canonical = program
                    .stages
                    .first()
                    .map(|s| {
                        if s.fixed_point {
                            s.apply_fixed_point(original)
                        } else {
                            original
                        }
                    })
                    .unwrap_or(original);

                let attn_row = attention_rows.get(pos).map(Vec::as_slice).unwrap_or(&[]);
                let attn_argmax = attention_argmax(attn_row, pos);
                let ctx = PositionContext {
                    stratum: stratum.to_string(),
                    position: pos,
                    token_id: token_ids.get(pos).copied().unwrap_or(0),
                    prev_token_id: (pos > 0).then(|| token_ids.get(pos - 1).copied()).flatten(),
                    attends_bos: attn_argmax == 0,
                    attends_prev: pos > 0 && attn_argmax + 1 == pos,
                    original_code: original,
                    current_code: canonical,
                };

                let terminal = program.apply_to_code(original, &ctx);

                diag.original_codes.add(original);
                diag.canonical_codes.add(canonical);
                diag.final_classes.add(terminal);
                if canonical == 6 && stratum == "natural_prose" {
                    diag.guard_eligible += 1;
                    if terminal == 6 {
                        diag.guard_fired += 1;
                    }
                }

                codes[target_group] = terminal;
                codes
            })
            .collect();

        // Pass 5: program-mapped Mode D injection — Metal if available, CPU fallback.
        // Track which path ran so the report is self-describing.
        let delta_flat =
            build_replacement_delta(mode_d_table, &remapped_codes, stratum, weights.hidden_size);
        let replacement_delta =
            ndarray::Array2::from_shape_vec((token_ids.len(), weights.hidden_size), delta_flat)
                .map_err(|e| format!("delta shape: {e}"))?;
        let (program_h, used_metal) = if let Some(ref b) = metal_backend {
            if let Some(h) = super::metal_backend::try_metal(
                &weights,
                &token_ids,
                &index,
                head.layer,
                head.head,
                &replacement_delta,
                b,
            ) {
                (h, true)
            } else {
                let h = forward_q4k_predicted_address_mode_d_head(
                    &mut weights,
                    &token_ids,
                    &index,
                    head,
                    mode_d_table,
                    &remapped_codes,
                    stratum,
                )?;
                (h, false)
            }
        } else {
            let h = forward_q4k_predicted_address_mode_d_head(
                &mut weights,
                &token_ids,
                &index,
                head,
                mode_d_table,
                &remapped_codes,
                stratum,
            )?;
            (h, false)
        };
        // Log once on the first prompt which backend was actually used.
        if idx == 0 {
            actual_intervention_backend = if used_metal { "metal" } else { "cpu_fallback" };
            eprintln!("intervention_backend: {actual_intervention_backend}");
        }
        let program_logits = final_logits(&weights, &program_h);
        let program_logp = log_softmax(&program_logits);
        let program_top1 = argmax(&program_logits);
        let program_top5 = top_k_indices(&program_logits, 5);

        prompt_reports.push(PromptReport {
            id: label.to_string(),
            stratum: stratum.to_string(),
            positions: token_ids.len(),
            kl: kl_logp(&baseline_logp, &program_logp),
            top1_agree: baseline_top1 == program_top1,
            baseline_top1_in_top5: program_top5.contains(&baseline_top1),
        });
    }

    let measured = aggregate_metrics(&prompt_reports);
    let strata = build_strata_reports(&prompt_reports);

    let behavior_gate = if measured.passes_strict() {
        "strict_pass"
    } else if measured.passes_smoke() {
        "smoke_pass"
    } else {
        "fail"
    };

    let (metric_parity, metric_parity_failures) = match program.parity_check(&measured) {
        Ok(()) => ("PASS", None),
        Err(detail) => {
            if program.reference_metrics.is_some() {
                ("FAIL", Some(detail))
            } else {
                ("n/a", None)
            }
        }
    };

    let oracle_metrics = diag.oracle_metrics();
    print_summary(
        &measured,
        &diag,
        oracle_metrics,
        behavior_gate,
        metric_parity,
        metric_parity_failures.as_deref(),
        fingerprint_match,
        actual_fingerprint.as_deref(),
        expected_fingerprint.as_deref(),
    );

    let report = EvalProgramReport {
        program_name: program.name.clone(),
        reference_source: program
            .reference_metrics
            .as_ref()
            .and_then(|rm| rm.get("source"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        head,
        group: program.group,
        base_config_k: config.k,
        base_config_groups: config.groups,
        base_config_bits_per_group: config.bits_per_group,
        codebook_fingerprint_expected: expected_fingerprint,
        codebook_fingerprint_actual: actual_fingerprint.clone(),
        codebook_fingerprint_match: fingerprint_match,
        eval_prompts: prompt_reports.len(),
        oracle_mode_d_mean_kl: oracle_metrics.map(|(m, _, _)| m),
        oracle_mode_d_p95_kl: oracle_metrics.map(|(_, p, _)| p),
        oracle_mode_d_max_kl: oracle_metrics.map(|(_, _, x)| x),
        mean_kl: measured.mean_kl,
        p95_kl: measured.p95_kl,
        max_kl: measured.max_kl,
        top1_agreement: measured.top1,
        top5_retention: measured.top5,
        behavior_gate,
        metric_parity,
        metric_parity_failures,
        intervention_backend: actual_intervention_backend,
        strata,
        per_prompt: prompt_reports,
    };

    // Save diagnostics.
    let diag_json = serde_json::json!({
        "oracle_mode_d": oracle_metrics.map(|(m,p,x)| serde_json::json!({"mean_kl":m,"p95_kl":p,"max_kl":x})),
        "original_codes": diag.original_codes.sorted_desc().iter().take(8)
            .map(|&(c,n)| serde_json::json!({"code":c,"count":n,"pct":n as f64/diag.original_codes.total() as f64}))
            .collect::<Vec<_>>(),
        "final_classes": diag.final_classes.sorted_desc().iter()
            .map(|&(c,n)| serde_json::json!({"code":c,"count":n,"pct":n as f64/diag.final_classes.total() as f64}))
            .collect::<Vec<_>>(),
        "guard_eligible": diag.guard_eligible,
        "guard_fired": diag.guard_fired,
        "guard_fire_rate": if diag.guard_eligible > 0 { diag.guard_fired as f64/diag.guard_eligible as f64 } else { 0.0 },
        "codebook_fingerprint": actual_fingerprint,
    });
    serde_json::to_writer_pretty(
        std::fs::File::create(args.out.join("diagnostics.json"))?,
        &diag_json,
    )?;
    serde_json::to_writer_pretty(
        std::fs::File::create(args.out.join("eval_program.json"))?,
        &report,
    )?;
    eprintln!("Wrote {}", args.out.join("eval_program.json").display());
    eprintln!("Wrote {}", args.out.join("diagnostics.json").display());

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn aggregate_metrics(reports: &[PromptReport]) -> BehaviorMetrics {
    let kls: Vec<f64> = reports.iter().map(|r| r.kl).collect();
    BehaviorMetrics {
        mean_kl: mean(&kls),
        p95_kl: percentile(kls.clone(), 0.95),
        max_kl: kls.iter().cloned().fold(0.0_f64, f64::max),
        top1: bool_rate(reports.iter().map(|r| r.top1_agree)),
        top5: bool_rate(reports.iter().map(|r| r.baseline_top1_in_top5)),
    }
}

fn build_strata_reports(reports: &[PromptReport]) -> Vec<StratumReport> {
    let mut by_stratum: HashMap<String, Vec<&PromptReport>> = HashMap::new();
    for r in reports {
        by_stratum.entry(r.stratum.clone()).or_default().push(r);
    }
    let mut strata: Vec<StratumReport> = by_stratum
        .into_iter()
        .map(|(stratum, rs)| {
            let kls: Vec<f64> = rs.iter().map(|r| r.kl).collect();
            StratumReport {
                stratum,
                prompts: rs.len(),
                mean_kl: mean(&kls),
                p95_kl: percentile(kls.clone(), 0.95),
                max_kl: kls.iter().cloned().fold(0.0_f64, f64::max),
                top1_agreement: bool_rate(rs.iter().map(|r| r.top1_agree)),
                top5_retention: bool_rate(rs.iter().map(|r| r.baseline_top1_in_top5)),
            }
        })
        .collect();
    strata.sort_by(|a, b| a.stratum.cmp(&b.stratum));
    strata
}

#[allow(clippy::too_many_arguments)]
fn print_summary(
    m: &BehaviorMetrics,
    diag: &Diagnostics,
    oracle: Option<(f64, f64, f64)>,
    behavior_gate: &str,
    metric_parity: &str,
    metric_parity_failures: Option<&str>,
    fingerprint_match: Option<bool>,
    actual_fp: Option<&str>,
    expected_fp: Option<&str>,
) {
    eprintln!("─────────────────────────────────────────");
    if let Some((om, op, ox)) = oracle {
        eprintln!("oracle Mode D:  mean {om:.6}  p95 {op:.6}  max {ox:.6}");
    }
    eprintln!(
        "program:        mean {:.6}  p95 {:.6}  max {:.6}",
        m.mean_kl, m.p95_kl, m.max_kl
    );
    eprintln!(
        "top1 {:.4}  top5 {:.4}  behavior_gate: {behavior_gate}  metric_parity: {metric_parity}",
        m.top1, m.top5,
    );
    match fingerprint_match {
        Some(true) => eprintln!("fingerprint: match ({})", actual_fp.unwrap_or("")),
        Some(false) => eprintln!(
            "fingerprint: MISMATCH — expected {} got {}",
            expected_fp.unwrap_or("?"),
            actual_fp.unwrap_or("?")
        ),
        None => {}
    }
    eprintln!(
        "guard: {}/{} eligible fired  top-orig: {:?}  final: {:?}",
        diag.guard_fired,
        diag.guard_eligible,
        diag.original_codes
            .sorted_desc()
            .iter()
            .take(4)
            .map(|&(c, n)| (c, n))
            .collect::<Vec<_>>(),
        diag.final_classes
            .sorted_desc()
            .iter()
            .take(4)
            .map(|&(c, n)| (c, n))
            .collect::<Vec<_>>(),
    );
    if let Some(detail) = metric_parity_failures {
        eprintln!("metric parity FAIL:\n{detail}");
    }
}

fn build_replacement_delta(
    mode_d_table: &super::pq::ModeDTable,
    remapped_codes: &[Vec<usize>],
    stratum: &str,
    hidden_size: usize,
) -> Vec<f32> {
    let mut delta = Vec::with_capacity(remapped_codes.len() * hidden_size);
    for (pos, codes) in remapped_codes.iter().enumerate() {
        let d = mode_d_table.delta_for_position_codes_with_stratum(pos, codes, stratum);
        delta.extend_from_slice(&d);
    }
    delta
}
