//! `larql accuracy` — split-axis accuracy suite for KV engines.
//!
//! Runs every selected engine through three corpora, splitting results
//! by [`KnowledgeSource`](larql_kv::accuracy_suite::prompts::KnowledgeSource)
//! so parametric correctness and in-context recall are reported
//! separately:
//!
//! - **Parametric** (`prompts::quick_20()` / `diverse_100()`): short
//!   factual completions. The answer lives in the model's weights —
//!   any K/V strategy should score near 100% here.
//! - **In-context** (`needle::needle_tests()`): needle-in-haystack at
//!   scaling context lengths. The answer is planted in the prompt;
//!   compressed engines (sliding window, residual replacement, quant
//!   K/V) may lose it as context grows.
//! - **Conflict** (`conflict::conflict_20()`): in-context premise
//!   contradicts pretraining. Score is `followed_context` vs
//!   `parametric_fallback` — the most engine-discriminating axis.
//!
//! Each cell reports both **top-1 match rate** (argmax verdict) and
//! **Shannon bits-per-token** (`-log2 P(expected_first_token | prompt)`,
//! lower = more confident).

use clap::Args;
use larql_inference::cpu_engine_backend;
use larql_inference::ffn::WeightFfn;
use larql_inference::InferenceModel;
use larql_kv::accuracy_suite::conflict::{conflict_20, conflict_quick};
use larql_kv::accuracy_suite::needle::{needle_tests, NeedleTest};
use larql_kv::accuracy_suite::prompts::{diverse_100, quick_20};
use larql_kv::accuracy_suite::runner::{
    compute_strategy_split, evaluate_conflict, evaluate_in_context, evaluate_parametric,
    format_strategy_split, ConflictScore, PromptScore, StrategySplit,
};
use larql_kv::EngineKind;
use std::path::PathBuf;
use std::time::Instant;

use crate::commands::primary::cache;

#[derive(Args)]
pub struct AccuracyArgs {
    /// Model: vindex directory, `hf://owner/name`, or a cache shorthand.
    pub model: String,

    /// Comma-separated engine specs (same syntax as `larql bench --engine`).
    /// Default: `standard,markov-rs,unlimited-context,turbo-quant`.
    #[arg(
        long,
        default_value = "standard,markov-rs,unlimited-context,turbo-quant"
    )]
    pub engines: String,

    /// Quick mode: 5-prompt parametric, 2 shortest needles, 5-prompt conflict.
    /// Off by default — full corpora are 101 parametric + 7 needles + 20 conflict.
    #[arg(long)]
    pub quick: bool,

    /// Override the parametric corpus size. Ignored when `--quick` is set.
    #[arg(long)]
    pub parametric_n: Option<usize>,

    /// Maximum needle context length in tokens. Default `8192` keeps the
    /// CI cost bounded; pass `32768` for the full sweep.
    #[arg(long, default_value = "8192")]
    pub needle_max_tokens: usize,

    /// Skip the conflict corpus.
    #[arg(long)]
    pub no_conflict: bool,

    /// Write a JSON report to this path. The split table still prints to stdout.
    #[arg(long, value_name = "PATH")]
    pub output_file: Option<PathBuf>,

    /// Verbose: log per-prompt scores as they arrive.
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, serde::Serialize)]
struct AccuracyReport {
    model: String,
    engines: Vec<String>,
    parametric_n: usize,
    needle_n: usize,
    conflict_n: usize,
    splits: Vec<StrategySplit>,
    per_prompt: Vec<PromptScore>,
    per_conflict: Vec<ConflictScore>,
}

pub fn run(args: AccuracyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = cache::resolve_model(&args.model)?;

    let engine_specs: Vec<String> = EngineKind::split_specs(&args.engines);
    if engine_specs.is_empty() {
        return Err("no engines selected: pass --engines standard,markov-rs,...".into());
    }
    let engine_kinds: Vec<(String, EngineKind)> = engine_specs
        .iter()
        .map(|spec| {
            EngineKind::from_name(spec)
                .map(|kind| (spec.to_string(), kind))
                .ok_or_else(|| format!("unknown engine spec: {spec}"))
        })
        .collect::<Result<_, _>>()?;

    eprintln!("larql accuracy: {}", model_path.display());
    let load_start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let ffn = WeightFfn { weights };
    eprintln!(
        "loaded weights in {:.1}s — vocab={}, layers={}, hidden={}",
        load_start.elapsed().as_secs_f64(),
        weights.vocab_size,
        weights.num_layers,
        weights.hidden_size,
    );

    // ── Choose corpora ────────────────────────────────────────────────
    let parametric_prompts = if args.quick {
        quick_20().into_iter().take(5).collect::<Vec<_>>()
    } else if let Some(n) = args.parametric_n {
        diverse_100().into_iter().take(n).collect()
    } else {
        diverse_100()
    };

    let needles: Vec<NeedleTest> = needle_tests()
        .into_iter()
        .filter(|n| n.context_tokens <= args.needle_max_tokens)
        .take(if args.quick { 2 } else { usize::MAX })
        .collect();

    let conflicts = if args.no_conflict {
        Vec::new()
    } else if args.quick {
        conflict_quick()
    } else {
        conflict_20()
    };

    eprintln!(
        "corpora: parametric={} needles={} conflict={}",
        parametric_prompts.len(),
        needles.len(),
        conflicts.len(),
    );

    // ── Drive each engine ─────────────────────────────────────────────
    let mut all_scores: Vec<PromptScore> = Vec::new();
    let mut all_conflicts: Vec<ConflictScore> = Vec::new();

    for (spec, kind) in &engine_kinds {
        eprintln!("\n── {spec} ──");
        let strategy_name = kind.display_name().to_string();

        let t0 = Instant::now();
        let param_scores = evaluate_parametric(
            || kind.clone().build(cpu_engine_backend()),
            weights,
            &ffn,
            tokenizer,
            &strategy_name,
            &parametric_prompts,
        );
        let p_match = param_scores.iter().filter(|s| s.top1_match).count();
        eprintln!(
            "  parametric: {}/{} top-1 in {:.1}s",
            p_match,
            param_scores.len(),
            t0.elapsed().as_secs_f64(),
        );
        if args.verbose {
            for s in &param_scores {
                eprintln!(
                    "    [{}] {} → {:?} (bits={:.2})",
                    if s.top1_match { "✓" } else { "✗" },
                    truncate(&s.prompt, 60),
                    s.predicted_top1,
                    s.bits_per_token,
                );
            }
        }
        all_scores.extend(param_scores);

        if !needles.is_empty() {
            let t0 = Instant::now();
            let needle_scores = evaluate_in_context(
                || kind.clone().build(cpu_engine_backend()),
                weights,
                &ffn,
                tokenizer,
                &strategy_name,
                &needles,
            );
            let n_match = needle_scores.iter().filter(|s| s.top1_match).count();
            eprintln!(
                "  in-context: {}/{} top-1 in {:.1}s",
                n_match,
                needle_scores.len(),
                t0.elapsed().as_secs_f64(),
            );
            if args.verbose {
                for s in &needle_scores {
                    eprintln!(
                        "    [{}] {} → {:?} (bits={:.2})",
                        if s.top1_match { "✓" } else { "✗" },
                        s.prompt,
                        s.predicted_top1,
                        s.bits_per_token,
                    );
                }
            }
            all_scores.extend(needle_scores);
        }

        if !conflicts.is_empty() {
            let t0 = Instant::now();
            let conflict_scores = evaluate_conflict(
                || kind.clone().build(cpu_engine_backend()),
                weights,
                &ffn,
                tokenizer,
                &strategy_name,
                &conflicts,
            );
            let followed = conflict_scores
                .iter()
                .filter(|s| s.followed_context)
                .count();
            let fallback = conflict_scores
                .iter()
                .filter(|s| s.parametric_fallback)
                .count();
            eprintln!(
                "  conflict: {} followed / {} fallback / {} other in {:.1}s",
                followed,
                fallback,
                conflict_scores.len() - followed - fallback,
                t0.elapsed().as_secs_f64(),
            );
            if args.verbose {
                for s in &conflict_scores {
                    let verdict = if s.followed_context {
                        "FOLLOW"
                    } else if s.parametric_fallback {
                        "FALLBACK"
                    } else {
                        "OTHER"
                    };
                    eprintln!(
                        "    [{verdict}] override={:?} param={:?} got={:?}",
                        s.override_answer, s.parametric_answer, s.predicted_top1,
                    );
                }
            }
            all_conflicts.extend(conflict_scores);
        }
    }

    // ── Render + emit ─────────────────────────────────────────────────
    let splits = compute_strategy_split(&all_scores, &all_conflicts);
    println!("{}", format_strategy_split(&splits));

    if let Some(path) = &args.output_file {
        let report = AccuracyReport {
            model: args.model.clone(),
            engines: engine_specs.iter().map(|s| s.to_string()).collect(),
            parametric_n: parametric_prompts.len(),
            needle_n: needles.len(),
            conflict_n: conflicts.len(),
            splits,
            per_prompt: all_scores,
            per_conflict: all_conflicts,
        };
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(path, &json)?;
        eprintln!("wrote {} bytes to {}", json.len(), path.display());
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let prefix: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{prefix}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_handles_short_strings() {
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn truncate_truncates_long_strings_with_ellipsis() {
        let s = truncate("0123456789abcdef", 6);
        assert_eq!(s.chars().count(), 6);
        assert!(s.ends_with('…'));
    }

    #[test]
    fn truncate_handles_unicode_safely() {
        // Verify that `truncate` slices by char-count, not byte-count,
        // so multi-byte UTF-8 in either the prompt or the ellipsis
        // doesn't panic.
        let s = truncate("αβγδεζηθικ", 5);
        assert_eq!(s.chars().count(), 5);
    }
}
