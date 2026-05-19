mod args;
mod capture;
mod context;
mod evaluate;
mod guard_synth;
mod localize;
mod proposal;

pub(super) use args::InduceProgramArgs;

use larql_vindex::{load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks};

use super::input::{parse_head_spec, parse_pq_configs};
use super::program::{
    BaseConfig, BehaviorMetrics, CodeReference, ConstructionMode, GuardAnnotation, Program,
    ProgramRule, ProgramStage, TerminalClass,
};
use super::program_cache::ProgramCache;
use capture::build_fit_context;
use evaluate::evaluate_program_fast;
use guard_synth::synthesize_guard;
use localize::localize_failure;
use proposal::{identity_program, set_merge_program, single_merge_program};

pub(super) fn run_induce_program(
    args: InduceProgramArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    let heads = parse_head_spec(&args.head)?;
    let head = heads.into_iter().next().ok_or("--head required")?;
    let configs = parse_pq_configs(&args.base_config)?;
    let config = configs.into_iter().next().ok_or("--base-config required")?;
    let base_config = BaseConfig {
        k: config.k,
        groups: config.groups,
        bits_per_group: config.bits_per_group,
    };

    eprintln!(
        "Inducing program for L{}H{} group {} at {}:{}:{}",
        head.layer, head.head, args.group, config.k, config.groups, config.bits_per_group
    );

    let mut cb = SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_kquant(&args.index)?;
    index.load_interleaved_kquant(&args.index)?;
    let mut weights = load_model_weights_kquant(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;

    // Step 1: fit codebook + capture oracle codes, attention, baselines once.
    let fit = build_fit_context(&args, &mut weights, &index, &tokenizer, head, config)?;

    eprintln!("\n=== Oracle Mode D baseline ===");
    let (_, oracle_metrics) = evaluate_program_fast(
        &mut weights,
        &index,
        &fit,
        &identity_program(head, args.group, &base_config),
    )?;
    print_gate("oracle", &oracle_metrics);

    // Step 2: load cache if present.
    let cache_path = args.out.join("program_cache.json");
    let cache: Option<ProgramCache> = if cache_path.exists() {
        eprintln!("Loading cache from {}", cache_path.display());
        Some(serde_json::from_str(&std::fs::read_to_string(
            &cache_path,
        )?)?)
    } else {
        None
    };

    // Step 3: pairwise merge matrix — use cache for known pairs, eval the rest.
    eprintln!("\n=== Pairwise merge matrix ===");
    let num_codes = base_config.num_codes();
    let mut strict_pairs: Vec<(usize, usize, BehaviorMetrics)> = Vec::new();

    for source in 0..num_codes {
        for target in 0..num_codes {
            if source == target {
                continue;
            }
            let key = format!("{source}->{target}");

            if let Some(m) = cache.as_ref().and_then(|c| c.substitutions.get(&key)) {
                if m.passes_strict() {
                    strict_pairs.push((source, target, bm_from_cached(m)));
                }
                continue;
            }

            let prog = single_merge_program(head, args.group, &base_config, source, target);
            let (_, m) = evaluate_program_fast(&mut weights, &index, &fit, &prog)?;
            if m.passes_strict() {
                eprintln!("  {source}->{target}: strict_pass  {}", fmt_m(&m));
                strict_pairs.push((source, target, m));
            }
        }
    }

    // Step 4: dominant sink + merge candidates.
    let mut sink_votes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &(_, t, _) in &strict_pairs {
        *sink_votes.entry(t).or_default() += 1;
    }
    let sink = match sink_votes
        .into_iter()
        .max_by_key(|&(_, v)| v)
        .map(|(k, _)| k)
    {
        Some(s) => s,
        None => {
            eprintln!("No strict-pass pairwise edges — cannot proceed");
            return Ok(());
        }
    };
    let mut clique: Vec<usize> = strict_pairs
        .iter()
        .filter(|(_, t, _)| *t == sink)
        .map(|(s, _, _)| *s)
        .collect();
    clique.push(sink);
    clique.sort_unstable();
    clique.dedup();
    eprintln!("\ndominant sink={sink}  clique={clique:?}");

    // Step 5: evaluate set merge of the clique.
    let set_merge = set_merge_program(head, args.group, &base_config, clique.clone(), sink);
    let (_, sm_metrics) = evaluate_program_fast(&mut weights, &index, &fit, &set_merge)?;
    eprintln!(
        "set-merge {clique:?}->{sink}: {}  [{}]",
        fmt_m(&sm_metrics),
        gate_label(&sm_metrics)
    );

    if sm_metrics.passes_strict() {
        eprintln!("\n✓ Set merge passes strict — no guard needed");
        return save_and_report(&set_merge, &sm_metrics, &args.out);
    }
    if !sm_metrics.passes_smoke() {
        eprintln!("\n✗ Set merge fails smoke — cannot proceed with this clique");
        return Ok(());
    }

    // Step 6: CEGIS — localize the failure.
    eprintln!("\n=== Failure localization (§9.3) ===");
    let loc = localize_failure(&mut weights, &index, &fit, &clique, sink, sm_metrics.max_kl)?;

    // Step 7: guard synthesis (§9.4).
    eprintln!("\n=== Guard synthesis (§9.4) ===");
    let guard = match synthesize_guard(&fit, &loc, &clique, sink) {
        Some(g) => g,
        None => {
            eprintln!("✗ No separating predicate found at depth-2 — extend DSL (fallback A)");
            return Ok(());
        }
    };

    // Step 8: one oracle call — evaluate the guarded program.
    eprintln!("\n=== Oracle call: guarded program ===");
    let guarded = build_guarded_program(
        head,
        args.group,
        &base_config,
        sink,
        clique.iter().copied().filter(|&c| c != sink).collect(),
        loc.fragile_code,
        guard.predicate.clone(),
        &guard.label,
    );
    let (_, gm) = evaluate_program_fast(&mut weights, &index, &fit, &guarded)?;
    eprintln!(
        "guarded ({guard_label}): {metrics}  [{gate}]",
        guard_label = guard.label,
        metrics = fmt_m(&gm),
        gate = gate_label(&gm)
    );

    save_and_report(&guarded, &gm, &args.out)
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn gate_label(m: &BehaviorMetrics) -> &'static str {
    if m.passes_strict() {
        "strict_pass"
    } else if m.passes_smoke() {
        "smoke_pass"
    } else {
        "fail"
    }
}

fn fmt_m(m: &BehaviorMetrics) -> String {
    format!(
        "mean={:.6} max={:.6} top1={:.3}",
        m.mean_kl, m.max_kl, m.top1
    )
}

fn print_gate(label: &str, m: &BehaviorMetrics) {
    eprintln!(
        "{label}: mean={:.6} p95={:.6} max={:.6} top1={:.4}  [{}]",
        m.mean_kl,
        m.p95_kl,
        m.max_kl,
        m.top1,
        gate_label(m)
    );
}

fn bm_from_cached(c: &super::program_cache::CachedResult) -> BehaviorMetrics {
    BehaviorMetrics {
        mean_kl: c.mean_kl,
        p95_kl: c.p95_kl.unwrap_or(c.max_kl),
        max_kl: c.max_kl,
        top1: c.top1,
        top5: c.top5,
    }
}

fn build_guarded_program(
    head: super::types::HeadId,
    group: usize,
    config: &BaseConfig,
    sink: usize,
    merge_sources: Vec<usize>,
    guarded_source: usize,
    unless_predicate: super::program::Predicate,
    guard_label: &str,
) -> Program {
    // Stage 1: merge non-guarded sources into sink (fixed-point safe since
    // guarded_source is not included).
    let stage1_rules: Vec<ProgramRule> = if merge_sources.is_empty() {
        vec![]
    } else {
        let non_guarded: Vec<usize> = merge_sources
            .iter()
            .copied()
            .filter(|&c| c != guarded_source)
            .collect();
        if non_guarded.is_empty() {
            vec![]
        } else if non_guarded.len() == 1 {
            vec![ProgramRule::Map {
                source: non_guarded[0],
                target: sink,
            }]
        } else {
            // Use fixed_point: false to avoid chaining when the sink is in the source set.
            vec![ProgramRule::MapSet {
                source: non_guarded,
                target: sink,
            }]
        }
    };

    // Stage 2: guarded merge of the prose-boundary source.
    let stage2_rules = vec![ProgramRule::MapUnless {
        source: guarded_source,
        target: sink,
        code_reference: CodeReference::CurrentCode,
        unless: unless_predicate,
    }];

    Program {
        head,
        group,
        base_config: config.clone(),
        name: Some(format!("induced_{guard_label}")),
        stages: vec![
            ProgramStage {
                name: "canonicalize".to_string(),
                fixed_point: true,
                declared_rules: stage1_rules,
                effective_map: None,
                guards: vec![],
            },
            ProgramStage {
                name: "guarded_rewrite".to_string(),
                fixed_point: false,
                declared_rules: stage2_rules,
                effective_map: None,
                guards: vec![GuardAnnotation {
                    code: guarded_source,
                    preserves_when: guard_label.to_string(),
                }],
            },
        ],
        terminal_classes: vec![
            TerminalClass {
                class: sink,
                construction_mode: ConstructionMode::Representative,
                representative_code: sink,
            },
            TerminalClass {
                class: guarded_source,
                construction_mode: ConstructionMode::Representative,
                representative_code: guarded_source,
            },
        ],
        metrics: None,
        reference_metrics: None,
        tolerance: None,
        program_size: None,
        predicate_space_used: vec![],
        codebook_fingerprint: None,
    }
}

fn save_and_report(
    program: &Program,
    metrics: &BehaviorMetrics,
    out: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut p = program.clone();
    p.metrics = Some(metrics.clone());
    p.normalize();
    p.program_size = Some(p.compute_size());

    let path = out.join("induced_program.json");
    serde_json::to_writer_pretty(std::fs::File::create(&path)?, &p)?;
    eprintln!("\nWrote {}", path.display());
    Ok(())
}
