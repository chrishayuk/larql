//! Vindex A/B comparison runner. Format-agnostic — works for any pair
//! of VectorIndex instances sharing the same underlying model.
//!
//! Primary use: exp 26 Q2 (FP4 end-to-end correctness) via
//!
//!     cargo run --release -p larql-kv --example vindex_compare -- \
//!         --reference output/gemma3-4b-f16.vindex \
//!         --candidate output/gemma3-4b-fp4.vindex \
//!         --prompts   ~/chris-source/chris-experiments/routing/26_fp4_quantisation/prompts.txt \
//!         --out       ~/chris-source/chris-experiments/routing/26_fp4_quantisation/results/q2_fp4.json
//!
//! Any future storage-format comparison (FP6, NF4, Q4K regression
//! tests) reuses the same binary — nothing here is FP4-specific.

use std::path::PathBuf;

use larql_inference::InferenceModel;
use larql_kv::vindex_compare::{compare_many, forward_to_logits_traced, ComparisonConfig};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

struct Args {
    reference: PathBuf,
    candidate: PathBuf,
    prompts_path: Option<PathBuf>,
    model: String,
    out: Option<PathBuf>,
    top_k: usize,
    max_seq_len: Option<usize>,
    max_layers: Option<usize>,
    inline_prompts: Vec<String>,
    trace: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut a = Args {
        reference: PathBuf::new(),
        candidate: PathBuf::new(),
        prompts_path: None,
        model: "google/gemma-3-4b-it".into(),
        out: None,
        top_k: 5,
        max_seq_len: None,
        max_layers: None,
        inline_prompts: Vec::new(),
        trace: false,
    };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--reference" => {
                i += 1;
                a.reference = PathBuf::from(&argv[i]);
            }
            "--candidate" => {
                i += 1;
                a.candidate = PathBuf::from(&argv[i]);
            }
            "--prompts" => {
                i += 1;
                a.prompts_path = Some(PathBuf::from(&argv[i]));
            }
            "--model" => {
                i += 1;
                a.model = argv[i].clone();
            }
            "--out" => {
                i += 1;
                a.out = Some(PathBuf::from(&argv[i]));
            }
            "--top-k" => {
                i += 1;
                a.top_k = argv[i].parse().expect("int");
            }
            "--max-seq" => {
                i += 1;
                a.max_seq_len = Some(argv[i].parse().expect("int"));
            }
            "--max-layers" => {
                i += 1;
                a.max_layers = Some(argv[i].parse().expect("int"));
            }
            "--prompt" => {
                i += 1;
                a.inline_prompts.push(argv[i].clone());
            }
            "--trace" => {
                a.trace = true;
            }
            other => eprintln!("warn: ignored arg {other}"),
        }
        i += 1;
    }
    if a.reference.as_os_str().is_empty() || a.candidate.as_os_str().is_empty() {
        eprintln!(
            "usage: vindex_compare --reference PATH --candidate PATH \\
    [--prompts FILE] [--prompt 'inline text' ...] \\
    [--model NAME] [--out PATH] [--top-k K] [--max-seq N] [--max-layers L]

At least one of --prompts or --prompt must be provided."
        );
        std::process::exit(1);
    }
    a
}

fn load_prompts(args: &Args) -> Vec<String> {
    let mut prompts = args.inline_prompts.clone();
    if let Some(path) = &args.prompts_path {
        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            prompts.push(trimmed.to_string());
        }
    }
    if prompts.is_empty() {
        // Small default set so running with just --reference / --candidate
        // produces something on stdout. Real use cases should pass --prompts.
        prompts = default_prompt_set();
    }
    prompts
}

fn default_prompt_set() -> Vec<String> {
    vec![
        "The capital of France is".into(),
        "Two plus two equals".into(),
        "The quick brown fox".into(),
        "Once upon a time".into(),
        "The largest planet in the solar system is".into(),
        "Shakespeare wrote".into(),
        "In 1969, the first man to walk on the moon was".into(),
        "The chemical formula for water is".into(),
    ]
}

fn main() {
    let args = parse_args();

    println!("== vindex_compare ==");
    println!("  reference: {}", args.reference.display());
    println!("  candidate: {}", args.candidate.display());
    println!("  model    : {}", args.model);
    println!("  top-k    : {}", args.top_k);
    if let Some(cap) = args.max_seq_len {
        println!("  max_seq  : {cap}");
    }
    if let Some(l) = args.max_layers {
        println!("  max_layers: {l}");
    }
    println!();

    let t_load = std::time::Instant::now();
    eprintln!("Loading model weights ({})...", args.model);
    let model = InferenceModel::load(&args.model).unwrap_or_else(|e| panic!("load model: {e}"));
    let tokenizer = model.tokenizer().clone();

    eprintln!("Loading reference vindex...");
    let mut cb = SilentLoadCallbacks;
    let reference = VectorIndex::load_vindex(&args.reference, &mut cb)
        .unwrap_or_else(|e| panic!("load reference: {e:?}"));
    eprintln!("Loading candidate vindex...");
    let candidate = VectorIndex::load_vindex(&args.candidate, &mut cb)
        .unwrap_or_else(|e| panic!("load candidate: {e:?}"));
    eprintln!("  loaded in {:.1}s", t_load.elapsed().as_secs_f64());
    eprintln!(
        "  reference has_fp4_storage={}",
        reference.has_fp4_storage()
    );
    eprintln!(
        "  candidate has_fp4_storage={}",
        candidate.has_fp4_storage()
    );
    eprintln!();

    // Tokenise the prompt set.
    let prompts = load_prompts(&args);
    eprintln!("Prompt set: {} prompts", prompts.len());
    let prompts_and_tokens: Vec<(&str, Vec<u32>)> = prompts
        .iter()
        .map(|p| {
            let enc = tokenizer
                .encode(p.as_str(), true)
                .unwrap_or_else(|e| panic!("tokenize: {e}"));
            (p.as_str(), enc.get_ids().to_vec())
        })
        .collect();

    let config = ComparisonConfig {
        top_k: args.top_k,
        max_seq_len: args.max_seq_len,
        max_layers: args.max_layers,
    };

    let weights = model.weights();

    // Optional single-prompt dispatch trace — isolates which walk path
    // each vindex actually fires, per layer. Exp 26 Q2 surfaced a bug
    // where an FP4 vindex silently fell through to the safetensors-
    // weights path; --trace is the tool for catching that class again.
    if args.trace {
        let (prompt, tokens) = &prompts_and_tokens[0];
        eprintln!();
        eprintln!("── dispatch trace (prompt 0: {}) ──", prompt);
        let cfg = ComparisonConfig {
            top_k: args.top_k,
            max_seq_len: args.max_seq_len,
            max_layers: args.max_layers,
        };
        let (_logits, ref_trace) = forward_to_logits_traced(weights, &reference, tokens, &cfg);
        let (_logits, cand_trace) = forward_to_logits_traced(weights, &candidate, tokens, &cfg);
        eprintln!("  {:>3}  {:<32}  {:<32}", "L", "reference", "candidate");
        for (layer, (r_path, c_path)) in ref_trace.iter().zip(cand_trace.iter()).enumerate() {
            let flag = if r_path.1 == c_path.1 { " " } else { "≠" };
            eprintln!("  {:>3}  {:<32}  {:<32}  {flag}", layer, r_path.1, c_path.1);
        }
        eprintln!();
    }

    let t_run = std::time::Instant::now();
    let mut report = compare_many(
        weights,
        &reference,
        &candidate,
        &prompts_and_tokens,
        &args.reference.display().to_string(),
        &args.candidate.display().to_string(),
        &config,
    );
    eprintln!("Compared in {:.1}s", t_run.elapsed().as_secs_f64());

    // Decode top tokens for human-readable output (tokenizer-free library
    // keeps this in the CLI).
    for p in report.prompts.iter_mut() {
        p.ref_top_token = Some(decode_token(&tokenizer, p.ref_top_token_id));
        p.cand_top_token = Some(decode_token(&tokenizer, p.cand_top_token_id));
    }

    print_human_report(&report);

    if let Some(out_path) = &args.out {
        if let Some(parent) = out_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let json =
            serde_json::to_string_pretty(&report).unwrap_or_else(|e| panic!("serialise: {e}"));
        std::fs::write(out_path, json)
            .unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
        println!();
        println!("→ wrote {}", out_path.display());
    }
}

fn decode_token(tokenizer: &larql_inference::tokenizers::Tokenizer, id: u32) -> String {
    tokenizer
        .decode(&[id], false)
        .unwrap_or_else(|_| format!("<{id}>"))
}

fn print_human_report(report: &larql_kv::vindex_compare::AggregateReport) {
    println!("── per-prompt ──");
    for p in &report.prompts {
        let ref_t = p.ref_top_token.as_deref().unwrap_or("?");
        let cand_t = p.cand_top_token.as_deref().unwrap_or("?");
        let flag = if p.argmax_match { "✓" } else { "✗" };
        let short: String = p.prompt.chars().take(50).collect();
        println!(
            "  {flag} {short:<50}  ref={ref_t:<12}  cand={cand_t:<12}  cos={:.4}  jac={:.2}  KL={:.4}",
            p.logit_cos, p.top_k_jaccard, p.kl_symmetric
        );
    }
    println!();
    println!("── aggregate ──");
    println!("  n prompts             : {}", report.n_prompts);
    println!(
        "  argmax agreement      : {:.4}  ({}/{})",
        report.argmax_agreement,
        (report.argmax_agreement * report.n_prompts as f64).round() as usize,
        report.n_prompts
    );
    println!(
        "  top-{} Jaccard mean    : {:.4}",
        report.config.top_k, report.top_k_agreement_mean
    );
    println!("  logit cosine mean     : {:.4}", report.logit_cos_mean);
    println!("  symmetric KL mean     : {:.5}", report.kl_mean);
    println!("  symmetric KL p95      : {:.5}", report.kl_p95);
    println!("  symmetric KL max      : {:.5}", report.kl_max);
}
