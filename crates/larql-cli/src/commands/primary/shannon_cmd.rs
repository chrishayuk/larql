//! `larql shannon` — next-token bit measurements for scriptable demos.
//!
//! These commands put the existing dense transformer forward pass behind a
//! Shannon-style surface: score the true next token, report `-log2(p)`, and
//! optionally drive a real arithmetic coder from the model distribution.

use std::fs;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use larql_inference::attention::SharedKV;
use larql_inference::forward::{apply_norm, dot_proj};
use larql_inference::{encode_prompt, InferenceModel, ModelWeights, WeightFfn};
use ndarray::{s, Array2};

const LN_2: f64 = std::f64::consts::LN_2;
const DEFAULT_CONTEXT: usize = 512;
const DEFAULT_STRIDE: usize = 256;

// ── Engine identifiers used across `shannon verify` ─────────────────────
// Engines name themselves in the comparison table, in the --engines arg
// parser, and in the `RESULT {...}` JSON line each Python scorer emits.
// Keeping the literals here means a typo can't drift them apart.
const ENGINE_RUST: &str = "rust";
const ENGINE_MLX: &str = "mlx";
const ENGINE_HF: &str = "hf";

/// Prefix the Python reference scorers emit on their final JSON line when
/// invoked with `--json`. The verify subprocess parser greps for this. If
/// you change it, also update `scripts/shannon_score_{mlx,hf}.py` and the
/// `--json` flag's help text there.
const RESULT_PREFIX: &str = "RESULT ";
// Arithmetic coding must rebuild the exact same integer frequency table when
// decoding. The vindex/Metal path is fast but can produce tiny cross-run float
// drift, so keep this comfortably above Gemma's 262K vocab without making the
// table hypersensitive to low-order logit differences.
const FREQ_TOTAL: u32 = 1 << 19;
const CODE_BITS: u32 = 32;
const TOP_VALUE: u64 = (1u64 << CODE_BITS) - 1;
const FIRST_QTR: u64 = TOP_VALUE / 4 + 1;
const HALF: u64 = FIRST_QTR * 2;
const THIRD_QTR: u64 = FIRST_QTR * 3;
const VINDEX_BLOCK_TARGET_TOKENS: usize = 512;

#[derive(Subcommand)]
pub enum ShannonCommand {
    /// Score a corpus as model next-token bits.
    Score(ScoreArgs),

    /// Score an answer slot after a prefix, e.g. "The capital of France is " + "Paris".
    Slot(SlotArgs),

    /// Score repeated occurrences of a needle in a passage.
    Repeat(RepeatArgs),

    /// Per-layer Shannon bits via the final-norm logit lens.
    /// At every layer L (embed plus each post-block residual), project through
    /// `final_norm + lm_head` and report bits/token, KL-to-final, and the
    /// adjacent `bits_saved[L] = bits_via_lens[L-1] - bits_via_lens[L]` deltas.
    Layers(LayersArgs),

    /// Encode a short text file with model-driven arithmetic coding.
    Encode(EncodeArgs),

    /// Decode a file produced by `larql shannon encode`.
    Decode(DecodeArgs),

    /// Cross-engine bits/char comparison. Orchestrates `shannon score` (LARQL
    /// Rust, in-process) plus optional MLX and HF/PyTorch reference scorers
    /// (subprocesses); prints a delta table and exits non-zero if any pair-wise
    /// delta exceeds `--threshold`. See `scripts/README_shannon_score.md`.
    Verify(VerifyArgs),
}

#[derive(Args)]
pub struct ScoreArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// UTF-8 corpus file to score.
    #[arg(long, value_name = "FILE")]
    corpus: PathBuf,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    bytes: Option<usize>,

    /// Maximum tokens in each scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    context: usize,

    /// Newly-scored target tokens per forward window.
    #[arg(long, default_value_t = DEFAULT_STRIDE)]
    stride: usize,
}

#[derive(Args)]
pub struct SlotArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Prefix before the answer slot. Include boundary whitespace if needed.
    #[arg(long)]
    prefix: String,

    /// Slot text to score.
    #[arg(long)]
    answer: String,

    /// Maximum tokens in the scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    context: usize,

    /// Show top-k predictions before the first answer token.
    #[arg(long, default_value_t = 5)]
    top_k: usize,
}

#[derive(Args)]
pub struct RepeatArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// UTF-8 passage file.
    #[arg(long, value_name = "FILE")]
    text: PathBuf,

    /// String whose occurrences should be scored in context.
    #[arg(long)]
    needle: String,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    bytes: Option<usize>,

    /// Maximum tokens in the scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    context: usize,
}

#[derive(Args)]
pub struct LayersArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// UTF-8 corpus file to score.
    #[arg(long, value_name = "FILE")]
    corpus: PathBuf,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    bytes: Option<usize>,

    /// Maximum tokens in each scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    context: usize,

    /// Newly-scored target tokens per forward window.
    #[arg(long, default_value_t = DEFAULT_STRIDE)]
    stride: usize,
}

#[derive(Args)]
pub struct EncodeArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// UTF-8 input text.
    #[arg(long = "in", value_name = "FILE")]
    input: PathBuf,

    /// Compressed output file.
    #[arg(long, value_name = "FILE")]
    out: PathBuf,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    bytes: Option<usize>,

    /// Previous tokens visible to the model for each arithmetic-code step.
    /// Ignored when --vindex is used; the KV-cache path uses 512-token blocks.
    #[arg(long, default_value_t = 256)]
    context: usize,

    /// Use a Q4K vindex for KV-cached forced-token scoring instead of raw HF weights.
    #[arg(long, value_name = "DIR")]
    vindex: Option<PathBuf>,

    /// Use the best GPU backend for the vindex path. Required for the fast Q4K path.
    #[arg(long)]
    metal: bool,
}

#[derive(Args)]
pub struct DecodeArgs {
    /// Model path or HuggingFace model ID. Must match the encoder model.
    model: String,

    /// File produced by `larql shannon encode`.
    #[arg(long = "in", value_name = "FILE")]
    input: PathBuf,

    /// Recovered UTF-8 text output.
    #[arg(long, value_name = "FILE")]
    out: PathBuf,

    /// Use a Q4K vindex for KV-cached forced-token scoring instead of raw HF weights.
    #[arg(long, value_name = "DIR")]
    vindex: Option<PathBuf>,

    /// Use the best GPU backend for the vindex path. Required for the fast Q4K path.
    #[arg(long)]
    metal: bool,
}

#[derive(Args)]
pub struct VerifyArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// UTF-8 corpus file to score. CRLF is normalized to LF before scoring so
    /// the three engines agree on tokenization (Python text I/O strips \r,
    /// LARQL Rust doesn't — see scripts/README_shannon_score.md).
    #[arg(long, value_name = "FILE")]
    corpus: PathBuf,

    /// Limit input to the first N bytes, truncated on a UTF-8 boundary.
    #[arg(long)]
    bytes: Option<usize>,

    /// Maximum tokens in each scoring forward window.
    #[arg(long, default_value_t = DEFAULT_CONTEXT)]
    context: usize,

    /// Newly-scored target tokens per forward window.
    #[arg(long, default_value_t = DEFAULT_STRIDE)]
    stride: usize,

    /// Comma-separated reference engines to run alongside LARQL Rust.
    /// Available: `mlx`, `hf`. Default: both.
    #[arg(long, default_value = "mlx,hf", value_name = "LIST")]
    engines: String,

    /// Maximum acceptable pair-wise delta in percent. Exits non-zero if any
    /// pair of engines disagrees by more than this on total bits.
    #[arg(long, default_value_t = 0.5)]
    threshold: f64,

    /// Python interpreter used to invoke the MLX and HF reference scorers.
    #[arg(long, default_value = ".venv/bin/python")]
    python: PathBuf,

    /// Override the MLX scorer script location.
    #[arg(
        long,
        default_value = "scripts/shannon_score_mlx.py",
        value_name = "FILE"
    )]
    mlx_script: PathBuf,

    /// Override the HF scorer script location.
    #[arg(
        long,
        default_value = "scripts/shannon_score_hf.py",
        value_name = "FILE"
    )]
    hf_script: PathBuf,

    /// Device passed to the HF scorer. `cpu` is deterministic; `mps` is faster.
    #[arg(long, default_value = "cpu")]
    hf_device: String,

    /// Emit a final `RESULT {...}` JSON line on stdout in addition to the
    /// human-readable delta table. Mirrors the `--json` flag on the Python
    /// reference scorers and is what `scripts/diagnose_models.py` consumes
    /// when sweeping multiple architectures, so the multi-arch driver
    /// doesn't have to regex-parse the formatted table.
    #[arg(long)]
    json: bool,
}

pub fn run(cmd: ShannonCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        ShannonCommand::Score(args) => run_score(args),
        ShannonCommand::Slot(args) => run_slot(args),
        ShannonCommand::Repeat(args) => run_repeat(args),
        ShannonCommand::Layers(args) => run_layers(args),
        ShannonCommand::Encode(args) => run_encode(args),
        ShannonCommand::Decode(args) => run_decode(args),
        ShannonCommand::Verify(args) => run_verify(args),
    }
}

fn run_score(args: ScoreArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, args.stride)?;
    let text = read_text(&args.corpus, args.bytes)?;
    let model = load_model(&args.model)?;
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &text)?;
    if ids.len() < 2 {
        return Err("corpus must tokenize to at least one scored token".into());
    }

    eprintln!(
        "scoring {} target tokens over {} bytes...",
        ids.len() - 1,
        text.len()
    );
    let summary = score_token_range(
        model.weights(),
        &ids,
        1..ids.len(),
        args.context,
        args.stride,
        Some("scoring"),
    )?;

    print_score_summary(&summary, text.len(), text.chars().count());
    Ok(())
}

fn run_layers(args: LayersArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, args.stride)?;
    let text = read_text(&args.corpus, args.bytes)?;
    let model = load_model(&args.model)?;
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &text)?;
    if ids.len() < 2 {
        return Err("corpus must tokenize to at least one scored token".into());
    }
    let weights = model.weights();
    let n_layers = weights.num_layers;
    let n_captures = n_layers + 1;

    eprintln!(
        "scoring {} target tokens over {} bytes across {} layers...",
        ids.len() - 1,
        text.len(),
        n_layers,
    );

    let mut layer_summaries: Vec<LayerSummary> =
        (0..n_captures).map(|_| LayerSummary::default()).collect();

    let pb = progress_bar((ids.len() - 1) as u64, "layers");
    let mut target_start = 1usize;
    while target_start < ids.len() {
        let target_end = (target_start + args.stride).min(ids.len());
        let prefix_start = target_end
            .saturating_sub(args.context)
            .min(target_start.saturating_sub(1));
        let chunk_ids = &ids[prefix_start..target_end];

        let captures = forward_hidden_all_layers(weights, chunk_ids)?;
        if captures.len() != n_captures {
            return Err(format!("expected {} captures, got {}", n_captures, captures.len()).into());
        }

        let row_start = target_start - prefix_start - 1;
        let row_end = target_end - prefix_start - 1;
        let n_targets = target_end - target_start;

        // Final log-probs at scoring positions, used as the KL reference.
        let final_normed = final_norm(weights, captures.last().unwrap());
        let final_rows = final_normed.slice(s![row_start..row_end, ..]);
        let final_raw = dot_proj(&final_rows, &weights.lm_head);
        let final_log_probs: Vec<Vec<f32>> = (0..n_targets)
            .map(|t| compute_log_probs_row(weights, final_raw.row(t)))
            .collect();

        for (layer_idx, hidden) in captures.iter().enumerate() {
            let normed = final_norm(weights, hidden);
            let rows = normed.slice(s![row_start..row_end, ..]);
            let raw = dot_proj(&rows, &weights.lm_head);
            for offset in 0..n_targets {
                let target = ids[target_start + offset] as usize;
                let layer_lp = compute_log_probs_row(weights, raw.row(offset));
                if target >= layer_lp.len() {
                    return Err(format!("target token {target} out of vocab").into());
                }
                let bits = -(layer_lp[target] as f64) / LN_2;
                let final_lp = &final_log_probs[offset];
                let mut kl_nats = 0.0_f64;
                for v in 0..layer_lp.len() {
                    let lp_l = layer_lp[v] as f64;
                    if !lp_l.is_finite() {
                        continue;
                    }
                    let p_l = lp_l.exp();
                    if p_l <= 0.0 || !p_l.is_finite() {
                        continue;
                    }
                    let lp_f = final_lp[v] as f64;
                    if !lp_f.is_finite() {
                        continue;
                    }
                    kl_nats += p_l * (lp_l - lp_f);
                }
                layer_summaries[layer_idx].total_bits += bits;
                layer_summaries[layer_idx].total_kl_bits += kl_nats / LN_2;
                layer_summaries[layer_idx].n_tokens += 1;
            }
        }

        pb.inc(n_targets as u64);
        target_start = target_end;
    }
    pb.finish_and_clear();

    print_layers_summary(&layer_summaries, text.len(), text.chars().count());
    Ok(())
}

fn run_slot(args: SlotArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, 1)?;
    let model = load_model(&args.model)?;
    let full = format!("{}{}", args.prefix, args.answer);
    let prefix_ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &args.prefix)?;
    let full_ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &full)?;
    ensure_token_prefix(&prefix_ids, &full_ids)?;

    if prefix_ids.len() == full_ids.len() {
        return Err("answer did not add any tokens; check --prefix and --answer".into());
    }

    let range = prefix_ids.len()..full_ids.len();
    let summary = score_token_range(
        model.weights(),
        &full_ids,
        range.clone(),
        args.context,
        range.len().max(1),
        None,
    )?;

    println!("prefix bytes: {}", args.prefix.len());
    println!("answer: {:?}", args.answer);
    println!("answer tokens: {}", range.len());
    println!("bits: {:.3}", summary.total_bits);
    println!("bits/token: {:.3}", summary.bits_per_token());
    println!(
        "bits/char: {:.3}",
        summary.total_bits / args.answer.chars().count().max(1) as f64
    );

    let first_prefix_start = prefix_ids.len().saturating_sub(args.context);
    let prefix_window = &full_ids[first_prefix_start..prefix_ids.len()];
    let logits = logits_for_last_token(model.weights(), prefix_window)?;
    let target = full_ids[prefix_ids.len()];
    let prob = prob_for_target(&logits, target)?;
    let first_bits = -prob.log2();
    let target_text = decode_one(model.tokenizer(), target);
    println!(
        "first token: id={} text={:?} prob={:.6} bits={:.3}",
        target, target_text, prob, first_bits
    );
    print_top_k(model.tokenizer(), &logits, args.top_k);
    Ok(())
}

fn run_repeat(args: RepeatArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, 1)?;
    if args.needle.is_empty() {
        return Err("--needle must not be empty".into());
    }
    let text = read_text(&args.text, args.bytes)?;
    let matches: Vec<(usize, &str)> = text.match_indices(&args.needle).collect();
    if matches.is_empty() {
        return Err(format!("needle {:?} not found", args.needle).into());
    }

    let model = load_model(&args.model)?;
    println!(
        "{:<8} {:>10} {:>10} {:>12}  text",
        "occ", "byte", "tokens", "bits"
    );
    println!("{}", "-".repeat(60));
    for (i, (offset, matched)) in matches.iter().enumerate() {
        let prefix = &text[..*offset];
        let full = format!("{prefix}{matched}");
        let prefix_ids = encode_prompt(model.tokenizer(), &*model.weights().arch, prefix)?;
        let full_ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &full)?;
        ensure_token_prefix(&prefix_ids, &full_ids)?;
        let range = prefix_ids.len()..full_ids.len();
        let summary = score_token_range(
            model.weights(),
            &full_ids,
            range.clone(),
            args.context,
            range.len().max(1),
            None,
        )?;
        println!(
            "{:<8} {:>10} {:>10} {:>12.3}  {:?}",
            i + 1,
            offset,
            range.len(),
            summary.total_bits,
            matched
        );
    }
    Ok(())
}

fn run_encode(args: EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.vindex.is_some() {
        return run_encode_vindex(args);
    }
    if args.context < 1 {
        return Err("--context must be at least 1".into());
    }
    let text = read_text(&args.input, args.bytes)?;
    let model = load_model(&args.model)?;
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &text)?;
    if ids.len() < 2 {
        return Err("input must tokenize to at least one encoded token".into());
    }

    eprintln!(
        "encoding {} bytes as {} target tokens...",
        text.len(),
        ids.len() - 1
    );
    let pb = progress_bar((ids.len() - 1) as u64, "encoding");
    let mut encoder = ArithmeticEncoder::new();
    for pos in 1..ids.len() {
        let prefix_start = pos.saturating_sub(args.context);
        let logits = logits_for_last_token(model.weights(), &ids[prefix_start..pos])?;
        let counts = quantized_counts(&logits)?;
        let (low, high) = interval_for_symbol(&counts, ids[pos])?;
        encoder.encode(low, high, FREQ_TOTAL);
        pb.inc(1);
    }
    pb.finish_and_clear();

    let payload = encoder.finish();
    let blob = ShannonFile {
        context: args.context as u32,
        first_token: ids[0],
        target_tokens: (ids.len() - 1) as u64,
        original_bytes: text.len() as u64,
        payload,
    };
    let bytes = blob.to_bytes();
    fs::write(&args.out, &bytes)?;

    let chars = text.chars().count().max(1) as f64;
    println!("original:        {:>10} bytes", text.len());
    println!("payload:         {:>10} bytes", blob.payload.len());
    println!("file:            {:>10} bytes", bytes.len());
    println!("tokens:          {:>10}", ids.len() - 1);
    println!(
        "ratio(payload):  {:>10.2}x",
        text.len() as f64 / blob.payload.len().max(1) as f64
    );
    println!(
        "bits/char:       {:>10.3}",
        blob.payload.len() as f64 * 8.0 / chars
    );
    println!("wrote: {}", args.out.display());
    Ok(())
}

fn run_decode(args: DecodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.vindex.is_some() {
        return run_decode_vindex(args);
    }
    let mut raw = Vec::new();
    fs::File::open(&args.input)?.read_to_end(&mut raw)?;
    let blob = ShannonFile::from_bytes(&raw)?;
    if blob.context < 1 {
        return Err("compressed file has invalid context".into());
    }

    let model = load_model(&args.model)?;
    let mut decoder = ArithmeticDecoder::new(&blob.payload);
    let mut ids = Vec::with_capacity(blob.target_tokens as usize + 1);
    ids.push(blob.first_token);

    eprintln!("decoding {} target tokens...", blob.target_tokens);
    let pb = progress_bar(blob.target_tokens, "decoding");
    for _ in 0..blob.target_tokens {
        let prefix_start = ids.len().saturating_sub(blob.context as usize);
        let logits = logits_for_last_token(model.weights(), &ids[prefix_start..])?;
        let counts = quantized_counts(&logits)?;
        let value = decoder.scaled_value(FREQ_TOTAL);
        let (symbol, low, high) = symbol_for_value(&counts, value)?;
        decoder.decode(low, high, FREQ_TOTAL);
        ids.push(symbol);
        pb.inc(1);
    }
    pb.finish_and_clear();

    let text = model
        .tokenizer()
        .decode(&ids, true)
        .map_err(|e| format!("decode error: {e}"))?;
    fs::write(&args.out, text.as_bytes())?;
    println!("decoded:         {:>10} bytes", text.len());
    println!("expected:        {:>10} bytes", blob.original_bytes);
    println!("wrote: {}", args.out.display());
    Ok(())
}

struct TempFileGuard(PathBuf);
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(Debug)]
struct VerifyResult {
    engine: &'static str,
    tokens_scored: usize,
    total_bits: f64,
    bits_per_token: f64,
    bits_per_char: f64,
    elapsed_secs: f64,
}

fn run_verify(args: VerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, args.stride)?;

    let (normalized, normalized_chars, tmp_corpus, _cleanup) =
        normalize_corpus_to_tempfile(&args.corpus, args.bytes)?;
    let engines = parse_engine_list(&args.engines)?;

    let mut results = vec![run_rust_scorer(&args, &normalized, normalized_chars)?];
    if engines.contains(&ENGINE_MLX) {
        results.push(spawn_reference_scorer(
            ENGINE_MLX,
            &args.python,
            &args.mlx_script,
            &args.model,
            &tmp_corpus,
            args.context,
            args.stride,
            None,
        )?);
    }
    if engines.contains(&ENGINE_HF) {
        results.push(spawn_reference_scorer(
            ENGINE_HF,
            &args.python,
            &args.hf_script,
            &args.model,
            &tmp_corpus,
            args.context,
            args.stride,
            Some(&args.hf_device),
        )?);
    }

    let reference_idx = pick_reference_index(&results);
    print_delta_table(&results, reference_idx);
    let (max_delta_pct, max_pair) = max_pairwise_delta_pct(&results);
    let pass = max_delta_pct <= args.threshold;
    if args.json {
        emit_verify_json(
            &results,
            reference_idx,
            max_delta_pct,
            max_pair,
            args.threshold,
            pass,
        );
    }
    print_verdict(max_delta_pct, max_pair, args.threshold)
}

/// Emit the `RESULT {...}` line consumed by `scripts/diagnose_models.py`.
/// Schema is intentionally flat so the Python side doesn't need to know
/// about Rust's `VerifyResult` struct layout.
fn emit_verify_json(
    results: &[VerifyResult],
    reference_idx: usize,
    max_delta_pct: f64,
    max_pair: (&str, &str),
    threshold: f64,
    pass: bool,
) {
    let reference_engine = results.get(reference_idx).map(|r| r.engine).unwrap_or("");
    let engines_json: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "engine": r.engine,
                "tokens_scored": r.tokens_scored,
                "total_bits": r.total_bits,
                "bits_per_token": r.bits_per_token,
                "bits_per_char": r.bits_per_char,
                "elapsed_secs": r.elapsed_secs,
            })
        })
        .collect();
    let payload = serde_json::json!({
        "reference": reference_engine,
        "engines": engines_json,
        "max_delta_pct": max_delta_pct,
        "max_pair": [max_pair.0, max_pair.1],
        "threshold_pct": threshold,
        "pass": pass,
    });
    println!("{}{}", RESULT_PREFIX, payload);
}

/// Read the corpus, strip CRLF for cross-engine tokenization parity, and
/// write the result to a temp file the Python subprocesses can read.
/// Returns the in-memory string, its char count, the temp file path, and
/// a guard that deletes the file when dropped.
fn normalize_corpus_to_tempfile(
    corpus: &Path,
    bytes: Option<usize>,
) -> Result<(String, usize, PathBuf, TempFileGuard), Box<dyn std::error::Error>> {
    let raw_text = read_text(&corpus.to_path_buf(), bytes)?;
    let normalized: String = raw_text.chars().filter(|c| *c != '\r').collect();
    let normalized_bytes = normalized.len();
    let normalized_chars = normalized.chars().count();
    let stripped = raw_text.len() - normalized_bytes;
    if stripped > 0 {
        eprintln!(
            "normalized corpus: stripped {} CR bytes ({} -> {} bytes)",
            stripped,
            raw_text.len(),
            normalized_bytes
        );
    }
    let tmp_corpus =
        std::env::temp_dir().join(format!("larql_shannon_verify_{}.txt", std::process::id()));
    fs::write(&tmp_corpus, normalized.as_bytes())?;
    let guard = TempFileGuard(tmp_corpus.clone());
    Ok((normalized, normalized_chars, tmp_corpus, guard))
}

/// Load the model and score the corpus through the LARQL Rust forward
/// path in-process. The model load + tokenize are isolated here so the
/// caller doesn't double-encode the corpus.
fn run_rust_scorer(
    args: &VerifyArgs,
    normalized: &str,
    normalized_chars: usize,
) -> Result<VerifyResult, Box<dyn std::error::Error>> {
    eprintln!("[{ENGINE_RUST}] loading {}...", args.model);
    let start = Instant::now();
    let model = load_model(&args.model)?;
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, normalized)?;
    if ids.len() < 2 {
        return Err("corpus must tokenize to at least one scored token".into());
    }
    eprintln!("[{ENGINE_RUST}] scoring {} target tokens...", ids.len() - 1);
    let summary = score_token_range(
        model.weights(),
        &ids,
        1..ids.len(),
        args.context,
        args.stride,
        Some(ENGINE_RUST),
    )?;
    Ok(VerifyResult {
        engine: ENGINE_RUST,
        tokens_scored: summary.token_bits.len(),
        total_bits: summary.total_bits,
        bits_per_token: summary.bits_per_token(),
        bits_per_char: summary.total_bits / normalized_chars.max(1) as f64,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

/// Spawn a Python reference scorer and time the run end-to-end. Wraps
/// `run_python_scorer` so the timing isn't measured per-fork-per-launch
/// and the `[engine] launching` line is uniform across MLX and HF.
#[allow(clippy::too_many_arguments)]
fn spawn_reference_scorer(
    engine: &'static str,
    python: &Path,
    script: &Path,
    model: &str,
    corpus: &Path,
    context: usize,
    stride: usize,
    device: Option<&str>,
) -> Result<VerifyResult, Box<dyn std::error::Error>> {
    eprintln!("[{engine}] launching {} ...", script.display());
    let start = Instant::now();
    let mut r = run_python_scorer(
        engine, python, script, model, corpus, context, stride, device,
    )?;
    r.elapsed_secs = start.elapsed().as_secs_f64();
    Ok(r)
}

/// Reference = HF if present (canonical PyTorch-side number), else any
/// non-rust engine, else fall back to rust itself.
fn pick_reference_index(results: &[VerifyResult]) -> usize {
    results
        .iter()
        .position(|r| r.engine == ENGINE_HF)
        .or_else(|| results.iter().position(|r| r.engine != ENGINE_RUST))
        .unwrap_or(0)
}

fn print_delta_table(results: &[VerifyResult], reference_idx: usize) {
    let reference_bits = results[reference_idx].total_bits;
    println!();
    println!(
        "{:<8} {:>8} {:>12} {:>12} {:>14} {:>10} {:>10}",
        "engine", "tokens", "bits/token", "bits/char", "total bits", "Δ vs ref", "elapsed"
    );
    println!("{}", "-".repeat(80));
    for (i, r) in results.iter().enumerate() {
        let delta_pct = (r.total_bits - reference_bits) / reference_bits.max(1.0) * 100.0;
        let delta_str = if i == reference_idx {
            "—".to_string()
        } else if delta_pct >= 0.0 {
            format!("+{:.3}%", delta_pct)
        } else {
            format!("{:.3}%", delta_pct)
        };
        println!(
            "{:<8} {:>8} {:>12.4} {:>12.4} {:>14.3} {:>10} {:>9.1}s",
            r.engine,
            r.tokens_scored,
            r.bits_per_token,
            r.bits_per_char,
            r.total_bits,
            delta_str,
            r.elapsed_secs
        );
    }
}

/// Max pair-wise delta in percent (largest |Δ| relative to the larger
/// value in the pair). Returns 0.0 / ("", "") when fewer than two
/// engines ran.
fn max_pairwise_delta_pct(results: &[VerifyResult]) -> (f64, (&'static str, &'static str)) {
    let mut max_delta_pct = 0.0_f64;
    let mut max_pair: (&'static str, &'static str) = ("", "");
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let a = results[i].total_bits;
            let b = results[j].total_bits;
            let denom = a.max(b).max(1.0);
            let d = (a - b).abs() / denom * 100.0;
            if d > max_delta_pct {
                max_delta_pct = d;
                max_pair = (results[i].engine, results[j].engine);
            }
        }
    }
    (max_delta_pct, max_pair)
}

fn print_verdict(
    max_delta_pct: f64,
    max_pair: (&'static str, &'static str),
    threshold: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!(
        "max pair-wise delta: {:.3}% ({} vs {})",
        max_delta_pct, max_pair.0, max_pair.1
    );
    println!("threshold:           {:.3}%", threshold);
    if max_delta_pct > threshold {
        println!("FAIL");
        Err(format!(
            "engines disagree by {:.3}% (> {:.3}% threshold) between {} and {}",
            max_delta_pct, threshold, max_pair.0, max_pair.1
        )
        .into())
    } else {
        println!("PASS");
        Ok(())
    }
}

fn parse_engine_list(spec: &str) -> Result<Vec<&'static str>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for token in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match token {
            "mlx" => out.push(ENGINE_MLX),
            "hf" => out.push(ENGINE_HF),
            "rust" => {} // rust always runs; tolerate it in the list
            other => return Err(format!("unknown engine: {other} (expected mlx, hf, rust)").into()),
        }
    }
    Ok(out)
}

fn run_python_scorer(
    engine: &'static str,
    python: &Path,
    script: &Path,
    model: &str,
    corpus: &Path,
    context: usize,
    stride: usize,
    device: Option<&str>,
) -> Result<VerifyResult, Box<dyn std::error::Error>> {
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(python);
    cmd.arg(script)
        .arg(model)
        .arg("--corpus")
        .arg(corpus)
        .arg("--context")
        .arg(context.to_string())
        .arg("--stride")
        .arg(stride.to_string())
        .arg("--json");
    if let Some(dev) = device {
        cmd.arg("--device").arg(dev);
    }
    let output = cmd.stderr(Stdio::inherit()).output().map_err(|e| {
        format!(
            "failed to spawn {} scorer ({} {}): {e}",
            engine,
            python.display(),
            script.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!("{} scorer exited with status {}", engine, output.status).into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result_line = stdout
        .lines()
        .rev()
        .find(|l| l.starts_with(RESULT_PREFIX))
        .ok_or_else(|| {
            format!(
                "{engine} scorer did not emit a `{}` line; rerun without --json to inspect",
                RESULT_PREFIX.trim_end()
            )
        })?;
    let json_str = result_line.trim_start_matches(RESULT_PREFIX).trim();
    let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        format!(
            "{engine} scorer emitted unparseable `{}` line: {e}",
            RESULT_PREFIX.trim_end()
        )
    })?;

    Ok(VerifyResult {
        engine,
        tokens_scored: parsed["tokens_scored"].as_u64().unwrap_or(0) as usize,
        total_bits: parsed["total_bits"].as_f64().unwrap_or(0.0),
        bits_per_token: parsed["bits_per_token"].as_f64().unwrap_or(0.0),
        bits_per_char: parsed["bits_per_char"].as_f64().unwrap_or(0.0),
        elapsed_secs: 0.0, // filled in by caller
    })
}

struct VindexShannonRuntime {
    weights: larql_inference::ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    index: larql_vindex::VectorIndex,
    backend: Box<dyn larql_compute::ComputeBackend>,
}

fn load_vindex_runtime(
    vindex: &Path,
    metal: bool,
) -> Result<VindexShannonRuntime, Box<dyn std::error::Error>> {
    if !metal {
        return Err("--vindex Shannon encode/decode currently requires --metal".into());
    }

    eprintln!("loading vindex {}...", vindex.display());
    let start = Instant::now();
    let cfg = larql_vindex::load_vindex_config(vindex)?;
    if cfg.quant != larql_vindex::QuantFormat::Q4K {
        return Err(format!(
            "--vindex fast Shannon path requires Q4K, found {:?}",
            cfg.quant
        )
        .into());
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(vindex, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex, &mut cb)?;
    index.load_attn_kquant(vindex)?;
    index.load_interleaved_kquant(vindex)?;
    let _ = index.load_lm_head_kquant(vindex);
    let backend = larql_compute::default_backend();
    if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
        return Err("Metal/Q4 backend is not available".into());
    }
    eprintln!(
        "loaded vindex. {} layers, hidden_size={}, backend={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        backend.name(),
        start.elapsed().as_secs_f64()
    );

    Ok(VindexShannonRuntime {
        weights,
        tokenizer,
        index,
        backend,
    })
}

fn run_encode_vindex(args: EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex = args.vindex.as_ref().ok_or("--vindex missing")?;
    let text = read_text(&args.input, args.bytes)?;
    let mut rt = load_vindex_runtime(vindex, args.metal)?;
    let ids = encode_prompt(&rt.tokenizer, &*rt.weights.arch, &text)?;
    if ids.len() < 2 {
        return Err("input must tokenize to at least one encoded token".into());
    }

    eprintln!(
        "encoding {} bytes as {} target tokens with KV-cached vindex blocks...",
        text.len(),
        ids.len() - 1
    );
    let pb = progress_bar((ids.len() - 1) as u64, "encoding");
    let mut blocks = Vec::new();
    let mut prefill_ms = 0.0;
    let mut decode_ms = Vec::new();
    let mut start = 0usize;
    while start + 1 < ids.len() {
        let end = (start + VINDEX_BLOCK_TARGET_TOKENS + 1).min(ids.len());
        let block_ids = &ids[start..end];
        let mut encoder = ArithmeticEncoder::new();
        let forced = larql_inference::layer_graph::generate::stream_forced_full_logits(
            &mut rt.weights,
            block_ids[0],
            block_ids.len() - 1,
            &rt.index,
            rt.backend.as_ref(),
            |step, logits| {
                let target = block_ids[step + 1];
                let counts =
                    quantized_counts(logits).map_err(|e| format!("quantize logits: {e}"))?;
                let (low, high) =
                    interval_for_symbol(&counts, target).map_err(|e| format!("interval: {e}"))?;
                encoder.encode(low, high, FREQ_TOTAL);
                pb.inc(1);
                Ok(target)
            },
        )?;
        prefill_ms += forced.prefill_ms;
        decode_ms.extend(forced.decode_ms);
        blocks.push(VindexShannonBlock {
            first_token: block_ids[0],
            target_tokens: (block_ids.len() - 1) as u64,
            payload: encoder.finish(),
        });
        start = end - 1;
    }
    pb.finish_and_clear();

    let payload = encode_vindex_blocks(&blocks);
    let blob = ShannonFile {
        // The vindex fast path is full-context within the GPU KV cache. Use
        // u32::MAX so old CPU decode treats this as "effectively unlimited"
        // for normal demo-sized files.
        context: u32::MAX,
        first_token: ids[0],
        target_tokens: (ids.len() - 1) as u64,
        original_bytes: text.len() as u64,
        payload,
    };
    let bytes = blob.to_bytes();
    fs::write(&args.out, &bytes)?;

    let chars = text.chars().count().max(1) as f64;
    println!("original:        {:>10} bytes", text.len());
    println!("payload:         {:>10} bytes", blob.payload.len());
    println!("file:            {:>10} bytes", bytes.len());
    println!("tokens:          {:>10}", ids.len() - 1);
    println!(
        "ratio(payload):  {:>10.2}x",
        text.len() as f64 / blob.payload.len().max(1) as f64
    );
    println!(
        "bits/char:       {:>10.3}",
        blob.payload.len() as f64 * 8.0 / chars
    );
    println!("blocks:          {:>10}", blocks.len());
    println!("prefill total:   {:>10.1} ms", prefill_ms);
    if !decode_ms.is_empty() {
        let avg = decode_ms.iter().sum::<f64>() / decode_ms.len() as f64;
        println!("decode avg:      {:>10.1} ms/token", avg);
    }
    println!("wrote: {}", args.out.display());
    Ok(())
}

fn run_decode_vindex(args: DecodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex = args.vindex.as_ref().ok_or("--vindex missing")?;
    let mut raw = Vec::new();
    fs::File::open(&args.input)?.read_to_end(&mut raw)?;
    let blob = ShannonFile::from_bytes(&raw)?;
    let mut rt = load_vindex_runtime(vindex, args.metal)?;
    let blocks = parse_vindex_blocks(&blob.payload)?.unwrap_or_else(|| {
        vec![VindexShannonBlock {
            first_token: blob.first_token,
            target_tokens: blob.target_tokens,
            payload: blob.payload.clone(),
        }]
    });

    eprintln!(
        "decoding {} target tokens with KV-cached vindex blocks...",
        blob.target_tokens
    );
    let pb = progress_bar(blob.target_tokens, "decoding");
    let mut ids = Vec::with_capacity(blob.target_tokens as usize + 1);
    let mut prefill_ms = 0.0;
    let mut decode_ms = Vec::new();
    for (block_idx, block) in blocks.iter().enumerate() {
        let mut decoder = ArithmeticDecoder::new(&block.payload);
        let forced = larql_inference::layer_graph::generate::stream_forced_full_logits(
            &mut rt.weights,
            block.first_token,
            block.target_tokens as usize,
            &rt.index,
            rt.backend.as_ref(),
            |_step, logits| {
                let counts =
                    quantized_counts(logits).map_err(|e| format!("quantize logits: {e}"))?;
                let value = decoder.scaled_value(FREQ_TOTAL);
                let (symbol, low, high) =
                    symbol_for_value(&counts, value).map_err(|e| format!("decode symbol: {e}"))?;
                decoder.decode(low, high, FREQ_TOTAL);
                pb.inc(1);
                Ok(symbol)
            },
        )?;
        if block_idx == 0 {
            ids.push(block.first_token);
        }
        ids.extend_from_slice(&forced.forced_tokens);
        prefill_ms += forced.prefill_ms;
        decode_ms.extend(forced.decode_ms);
    }
    pb.finish_and_clear();

    let text = rt
        .tokenizer
        .decode(&ids, true)
        .map_err(|e| format!("decode error: {e}"))?;
    fs::write(&args.out, text.as_bytes())?;
    println!("decoded:         {:>10} bytes", text.len());
    println!("expected:        {:>10} bytes", blob.original_bytes);
    println!("blocks:          {:>10}", blocks.len());
    println!("prefill total:   {:>10.1} ms", prefill_ms);
    if !decode_ms.is_empty() {
        let avg = decode_ms.iter().sum::<f64>() / decode_ms.len() as f64;
        println!("decode avg:      {:>10.1} ms/token", avg);
    }
    println!("wrote: {}", args.out.display());
    Ok(())
}

fn load_model(model: &str) -> Result<InferenceModel, Box<dyn std::error::Error>> {
    eprintln!("loading {model}...");
    let start = Instant::now();
    let loaded = InferenceModel::load(model)?;
    eprintln!(
        "loaded. {} layers, hidden_size={} ({:.1}s)",
        loaded.num_layers(),
        loaded.hidden_size(),
        start.elapsed().as_secs_f64()
    );
    Ok(loaded)
}

fn read_text(
    path: &PathBuf,
    limit_bytes: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut text = fs::read_to_string(path)?;
    if let Some(limit) = limit_bytes {
        if text.len() > limit {
            let mut end = limit;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
        }
    }
    Ok(text)
}

fn validate_window(context: usize, stride: usize) -> Result<(), Box<dyn std::error::Error>> {
    if context < 2 {
        return Err("--context must be at least 2 for scoring".into());
    }
    if stride == 0 {
        return Err("--stride must be at least 1".into());
    }
    if stride >= context {
        return Err("--stride must be smaller than --context so every target has a prefix".into());
    }
    Ok(())
}

fn ensure_token_prefix(prefix: &[u32], full: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    if full.len() < prefix.len() || full[..prefix.len()] != *prefix {
        return Err(
            "answer did not tokenize as a suffix of prefix+answer; add explicit boundary whitespace"
                .into(),
        );
    }
    Ok(())
}

#[derive(Debug, Default)]
struct ScoreSummary {
    total_bits: f64,
    token_bits: Vec<f64>,
}

impl ScoreSummary {
    fn bits_per_token(&self) -> f64 {
        self.total_bits / self.token_bits.len().max(1) as f64
    }
}

fn score_token_range(
    weights: &ModelWeights,
    ids: &[u32],
    range: Range<usize>,
    context: usize,
    stride: usize,
    progress: Option<&str>,
) -> Result<ScoreSummary, Box<dyn std::error::Error>> {
    if range.start == 0 || range.end > ids.len() || range.start > range.end {
        return Err("invalid scoring token range".into());
    }
    let mut summary = ScoreSummary::default();
    let pb = progress.map(|label| progress_bar((range.end - range.start) as u64, label));
    let mut target_start = range.start;
    while target_start < range.end {
        let target_end = (target_start + stride).min(range.end);
        let prefix_start = target_end
            .saturating_sub(context)
            .min(target_start.saturating_sub(1));
        let chunk_ids = &ids[prefix_start..target_end];
        let hidden = forward_hidden(weights, chunk_ids)?;
        let hidden = final_norm(weights, &hidden);

        let row_start = target_start - prefix_start - 1;
        let row_end = target_end - prefix_start - 1;
        let rows = hidden.slice(s![row_start..row_end, ..]);
        let raw_logits = dot_proj(&rows, &weights.lm_head);
        for (offset, target_pos) in (target_start..target_end).enumerate() {
            let bits = bits_for_raw_row(weights, raw_logits.row(offset), ids[target_pos])?;
            summary.total_bits += bits;
            summary.token_bits.push(bits);
            if let Some(pb) = &pb {
                pb.inc(1);
            }
        }
        target_start = target_end;
    }
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    Ok(summary)
}

fn print_score_summary(summary: &ScoreSummary, bytes: usize, chars: usize) {
    let chars = chars.max(1) as f64;
    let bytes = bytes.max(1) as f64;
    println!("done.");
    println!("tokens scored:  {:>10}", summary.token_bits.len());
    println!("bits/token:     {:>10.3}", summary.bits_per_token());
    println!("bits/char:      {:>10.3}", summary.total_bits / chars);
    println!("bits/byte:      {:>10.3}", summary.total_bits / bytes);
    println!("total bits:     {:>10.1}", summary.total_bits);
}

fn forward_hidden(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    if token_ids.is_empty() {
        return Err("empty token window".into());
    }
    let ffn = WeightFfn { weights };
    let mut h = larql_inference::forward::embed_tokens_pub(weights, token_ids);
    let ple_inputs =
        larql_inference::forward::ple::precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();
    for layer in 0..weights.num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = larql_inference::forward::run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }
    }
    Ok(h)
}

fn forward_hidden_all_layers(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Vec<Array2<f32>>, Box<dyn std::error::Error>> {
    if token_ids.is_empty() {
        return Err("empty token window".into());
    }
    let ffn = WeightFfn { weights };
    let h0 = larql_inference::forward::embed_tokens_pub(weights, token_ids);
    let ple_inputs =
        larql_inference::forward::ple::precompute_per_layer_inputs(weights, &h0, token_ids);
    let mut captures: Vec<Array2<f32>> = Vec::with_capacity(weights.num_layers + 1);
    captures.push(h0.clone());
    let mut h = h0;
    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();
    for layer in 0..weights.num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = larql_inference::forward::run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }
        captures.push(h.clone());
    }
    Ok(captures)
}

fn final_norm(weights: &ModelWeights, h: &Array2<f32>) -> Array2<f32> {
    apply_norm(
        weights,
        h,
        weights.arch.final_norm_key(),
        weights.arch.norm_weight_offset(),
    )
}

fn logits_for_last_token(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let hidden = forward_hidden(weights, token_ids)?;
    let hidden = final_norm(weights, &hidden);
    logits_for_row(weights, &hidden, hidden.shape()[0] - 1)
}

fn logits_for_row(
    weights: &ModelWeights,
    final_hidden: &Array2<f32>,
    row_idx: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if row_idx >= final_hidden.shape()[0] {
        return Err("logit row out of range".into());
    }
    let row = final_hidden.slice(s![row_idx..row_idx + 1, ..]);
    let raw = dot_proj(&row, &weights.lm_head);
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    Ok(raw
        .row(0)
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit
        })
        .collect())
}

fn bits_for_target(logits: &[f32], target: u32) -> Result<f64, Box<dyn std::error::Error>> {
    let target = target as usize;
    if target >= logits.len() {
        return Err(format!("target token {target} out of vocab").into());
    }
    let max_logit = finite_max(logits)?;
    let exp_sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| ((v - max_logit) as f64).exp())
        .sum();
    let logsumexp = max_logit as f64 + exp_sum.ln();
    Ok((logsumexp - logits[target] as f64) / LN_2)
}

fn bits_for_raw_row(
    weights: &ModelWeights,
    row: ndarray::ArrayView1<'_, f32>,
    target: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let target = target as usize;
    if target >= row.len() {
        return Err(format!("target token {target} out of vocab").into());
    }

    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let transform = |v: f32| {
        let mut logit = v * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logit
    };

    let max_logit = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(transform)
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |m| m.max(v)))
        })
        .ok_or_else(|| "all logits were non-finite".to_string())?;

    let exp_sum: f64 = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| ((transform(v) - max_logit) as f64).exp())
        .sum();
    let target_logit = transform(row[target]);
    let logsumexp = max_logit as f64 + exp_sum.ln();
    Ok((logsumexp - target_logit as f64) / LN_2)
}

fn prob_for_target(logits: &[f32], target: u32) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(2.0_f64.powf(-bits_for_target(logits, target)?))
}

/// Apply per-arch logit scaling/softcap and return natural-log probabilities
/// over the full vocabulary for one position. Length matches the input row.
fn compute_log_probs_row(weights: &ModelWeights, row: ndarray::ArrayView1<'_, f32>) -> Vec<f32> {
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let transform = |v: f32| {
        if !v.is_finite() {
            return v;
        }
        let mut logit = v * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logit
    };
    let scaled: Vec<f32> = row.iter().copied().map(transform).collect();
    let max_logit = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| ((v - max_logit) as f64).exp())
        .sum();
    let logsumexp = (max_logit as f64) + exp_sum.ln();
    scaled
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v as f64) - logsumexp) as f32
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerSummary {
    total_bits: f64,
    total_kl_bits: f64,
    n_tokens: usize,
}

impl LayerSummary {
    fn bits_per_token(&self) -> f64 {
        if self.n_tokens == 0 {
            0.0
        } else {
            self.total_bits / self.n_tokens as f64
        }
    }

    fn kl_per_token(&self) -> f64 {
        if self.n_tokens == 0 {
            0.0
        } else {
            self.total_kl_bits / self.n_tokens as f64
        }
    }
}

fn layer_label(idx: usize) -> String {
    if idx == 0 {
        "embed".to_string()
    } else {
        format!("L{:02}", idx - 1)
    }
}

fn print_layers_summary(layer_summaries: &[LayerSummary], bytes: usize, chars: usize) {
    let n = layer_summaries.len();
    let scored = layer_summaries.first().map(|s| s.n_tokens).unwrap_or(0);
    println!("done.");
    println!("tokens scored:  {:>10}", scored);
    println!("bytes:          {:>10}", bytes);
    println!("chars:          {:>10}", chars);
    println!();
    println!("per-layer bit contribution (final-norm lens):");
    println!();
    println!(
        "  {:<6} {:<6}  {:>11}  {:>11}  {:>11}",
        "from", "to", "bits saved", "bits/token", "KL->final"
    );
    println!("  {:-<55}", "");

    let mut layers_only_total = 0.0_f64;
    for to_idx in 1..n {
        let from = &layer_summaries[to_idx - 1];
        let to = &layer_summaries[to_idx];
        let bits_saved = from.bits_per_token() - to.bits_per_token();
        let kl_reduction = from.kl_per_token() - to.kl_per_token();
        println!(
            "  {:<6} {:<6}  {:>11.3}  {:>11.3}  {:>11.3}",
            layer_label(to_idx - 1),
            layer_label(to_idx),
            bits_saved,
            to.bits_per_token(),
            kl_reduction,
        );
        if to_idx > 1 {
            // Skip the embed -> L0 transition: that's lens warm-up, not layer
            // labour. Match exp 34's `summary_layers_only` view.
            layers_only_total += bits_saved;
        }
    }

    if let (Some(first), Some(last)) = (layer_summaries.first(), layer_summaries.last()) {
        println!();
        println!(
            "embed bits/token: {:>10.3}    final bits/token: {:>10.3}",
            first.bits_per_token(),
            last.bits_per_token()
        );
    }
    println!(
        "total layers-only bits saved: {:>8.2} / token  (excludes embed -> L0)",
        layers_only_total
    );
}

fn finite_max(values: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |m| m.max(v)))
        })
        .ok_or_else(|| "all logits were non-finite".into())
}

fn print_top_k(tokenizer: &tokenizers::Tokenizer, logits: &[f32], top_k: usize) {
    let max_logit = match finite_max(logits) {
        Ok(v) => v,
        Err(_) => return,
    };
    let exp_sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| ((v - max_logit) as f64).exp())
        .sum();
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("top predictions before slot:");
    for (rank, (id, logit)) in indexed.into_iter().take(top_k).enumerate() {
        let prob = (((logit - max_logit) as f64).exp() / exp_sum).max(0.0);
        println!(
            "  {:>2}. id={:<8} text={:?} prob={:.6} bits={:.3}",
            rank + 1,
            id,
            decode_one(tokenizer, id as u32),
            prob,
            -prob.log2()
        );
    }
}

fn decode_one(tokenizer: &tokenizers::Tokenizer, id: u32) -> String {
    tokenizer
        .decode(&[id], true)
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| tokenizer.id_to_token(id))
        .unwrap_or_else(|| format!("[{id}]"))
}

fn quantized_counts(logits: &[f32]) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    if logits.len() >= FREQ_TOTAL as usize {
        return Err("vocab is too large for arithmetic coder frequency total".into());
    }
    let max_logit = finite_max(logits)?;
    let exp_values: Vec<f64> = logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v - max_logit) as f64).exp()
            } else {
                0.0
            }
        })
        .collect();
    let exp_sum: f64 = exp_values.iter().sum();
    if exp_sum <= 0.0 {
        return Err("invalid probability distribution".into());
    }
    let spare = FREQ_TOTAL as usize - logits.len();
    let mut max_idx = 0usize;
    let mut max_exp = f64::NEG_INFINITY;
    let mut sum = 0u32;
    let mut counts = Vec::with_capacity(logits.len());
    for (i, exp_v) in exp_values.iter().copied().enumerate() {
        if exp_v > max_exp {
            max_exp = exp_v;
            max_idx = i;
        }
        let count = 1 + (exp_v / exp_sum * spare as f64).floor() as u32;
        sum = sum.saturating_add(count);
        counts.push(count);
    }
    if sum > FREQ_TOTAL {
        return Err("frequency quantization overflowed".into());
    }
    counts[max_idx] += FREQ_TOTAL - sum;
    Ok(counts)
}

fn interval_for_symbol(
    counts: &[u32],
    symbol: u32,
) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    let symbol = symbol as usize;
    if symbol >= counts.len() {
        return Err(format!("symbol {symbol} out of frequency table").into());
    }
    let low: u32 = counts[..symbol].iter().sum();
    let high = low + counts[symbol];
    Ok((low, high))
}

fn symbol_for_value(
    counts: &[u32],
    value: u32,
) -> Result<(u32, u32, u32), Box<dyn std::error::Error>> {
    let mut low = 0u32;
    for (symbol, &count) in counts.iter().enumerate() {
        let high = low + count;
        if value < high {
            return Ok((symbol as u32, low, high));
        }
        low = high;
    }
    Err("arithmetic decoder value outside frequency table".into())
}

struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            used: 0,
        }
    }

    fn write(&mut self, bit: bool) {
        self.current = (self.current << 1) | u8::from(bit);
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used > 0 {
            self.current <<= 8 - self.used;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    byte_idx: usize,
    bit_idx: u8,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_idx: 0,
            bit_idx: 0,
        }
    }

    fn read(&mut self) -> bool {
        if self.byte_idx >= self.bytes.len() {
            return false;
        }
        let bit = (self.bytes[self.byte_idx] & (0x80 >> self.bit_idx)) != 0;
        self.bit_idx += 1;
        if self.bit_idx == 8 {
            self.bit_idx = 0;
            self.byte_idx += 1;
        }
        bit
    }
}

struct ArithmeticEncoder {
    low: u64,
    high: u64,
    pending: u64,
    bits: BitWriter,
}

impl ArithmeticEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            high: TOP_VALUE,
            pending: 0,
            bits: BitWriter::new(),
        }
    }

    fn encode(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * cum_high as u64) / total as u64 - 1;
        self.low += (range * cum_low as u64) / total as u64;

        loop {
            if self.high < HALF {
                self.output_bit_plus_follow(false);
            } else if self.low >= HALF {
                self.output_bit_plus_follow(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.pending += 1;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low *= 2;
            self.high = self.high * 2 + 1;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < FIRST_QTR {
            self.output_bit_plus_follow(false);
        } else {
            self.output_bit_plus_follow(true);
        }
        self.bits.finish()
    }

    fn output_bit_plus_follow(&mut self, bit: bool) {
        self.bits.write(bit);
        for _ in 0..self.pending {
            self.bits.write(!bit);
        }
        self.pending = 0;
    }
}

struct ArithmeticDecoder<'a> {
    low: u64,
    high: u64,
    code: u64,
    bits: BitReader<'a>,
}

impl<'a> ArithmeticDecoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        let mut bits = BitReader::new(bytes);
        let mut code = 0u64;
        for _ in 0..CODE_BITS {
            code = code * 2 + u64::from(bits.read());
        }
        Self {
            low: 0,
            high: TOP_VALUE,
            code,
            bits,
        }
    }

    fn scaled_value(&self, total: u32) -> u32 {
        let range = self.high - self.low + 1;
        ((((self.code - self.low + 1) * total as u64 - 1) / range) as u32).min(total - 1)
    }

    fn decode(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * cum_high as u64) / total as u64 - 1;
        self.low += (range * cum_low as u64) / total as u64;

        loop {
            if self.high < HALF {
                // nothing
            } else if self.low >= HALF {
                self.code -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.code -= FIRST_QTR;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low *= 2;
            self.high = self.high * 2 + 1;
            self.code = self.code * 2 + u64::from(self.bits.read());
        }
    }
}

struct ShannonFile {
    context: u32,
    first_token: u32,
    target_tokens: u64,
    original_bytes: u64,
    payload: Vec<u8>,
}

#[derive(Clone)]
struct VindexShannonBlock {
    first_token: u32,
    target_tokens: u64,
    payload: Vec<u8>,
}

impl ShannonFile {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(36 + self.payload.len());
        out.extend_from_slice(b"LSC1");
        out.extend_from_slice(&self.context.to_le_bytes());
        out.extend_from_slice(&self.first_token.to_le_bytes());
        out.extend_from_slice(&self.target_tokens.to_le_bytes());
        out.extend_from_slice(&self.original_bytes.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        if bytes.len() < 36 || &bytes[..4] != b"LSC1" {
            return Err("not a LARQL Shannon compressed file".into());
        }
        let context = u32::from_le_bytes(bytes[4..8].try_into()?);
        let first_token = u32::from_le_bytes(bytes[8..12].try_into()?);
        let target_tokens = u64::from_le_bytes(bytes[12..20].try_into()?);
        let original_bytes = u64::from_le_bytes(bytes[20..28].try_into()?);
        let payload_len = u64::from_le_bytes(bytes[28..36].try_into()?) as usize;
        if bytes.len() != 36 + payload_len {
            return Err("compressed file payload length mismatch".into());
        }
        Ok(Self {
            context,
            first_token,
            target_tokens,
            original_bytes,
            payload: bytes[36..].to_vec(),
        })
    }
}

fn encode_vindex_blocks(blocks: &[VindexShannonBlock]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LSB1");
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for block in blocks {
        out.extend_from_slice(&block.first_token.to_le_bytes());
        out.extend_from_slice(&block.target_tokens.to_le_bytes());
        out.extend_from_slice(&(block.payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&block.payload);
    }
    out
}

fn parse_vindex_blocks(
    bytes: &[u8],
) -> Result<Option<Vec<VindexShannonBlock>>, Box<dyn std::error::Error>> {
    if !bytes.starts_with(b"LSB1") {
        return Ok(None);
    }
    if bytes.len() < 8 {
        return Err("truncated vindex block payload".into());
    }
    let block_count = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let mut offset = 8usize;
    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        if bytes.len().saturating_sub(offset) < 20 {
            return Err("truncated vindex block header".into());
        }
        let first_token = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        let target_tokens = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        let payload_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?) as usize;
        offset += 8;
        if bytes.len().saturating_sub(offset) < payload_len {
            return Err("truncated vindex block payload".into());
        }
        blocks.push(VindexShannonBlock {
            first_token,
            target_tokens,
            payload: bytes[offset..offset + payload_len].to_vec(),
        });
        offset += payload_len;
    }
    if offset != bytes.len() {
        return Err("trailing bytes after vindex block payload".into());
    }
    Ok(Some(blocks))
}

fn progress_bar(len: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message(label.to_string());
    pb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_round_trip_fixed_counts() {
        let counts = vec![3, 1, 4, 2];
        let total: u32 = counts.iter().sum();
        let symbols = [0u32, 2, 2, 3, 1, 0, 2];

        let mut enc = ArithmeticEncoder::new();
        for &sym in &symbols {
            let (low, high) = interval_for_symbol(&counts, sym).unwrap();
            enc.encode(low, high, total);
        }
        let payload = enc.finish();
        let mut dec = ArithmeticDecoder::new(&payload);
        let mut out = Vec::new();
        for _ in 0..symbols.len() {
            let value = dec.scaled_value(total);
            let (sym, low, high) = symbol_for_value(&counts, value).unwrap();
            dec.decode(low, high, total);
            out.push(sym);
        }

        assert_eq!(out, symbols);
    }

    #[test]
    fn shannon_file_round_trip() {
        let file = ShannonFile {
            context: 128,
            first_token: 2,
            target_tokens: 42,
            original_bytes: 100,
            payload: vec![1, 2, 3, 4],
        };
        let parsed = ShannonFile::from_bytes(&file.to_bytes()).unwrap();
        assert_eq!(parsed.context, 128);
        assert_eq!(parsed.first_token, 2);
        assert_eq!(parsed.target_tokens, 42);
        assert_eq!(parsed.original_bytes, 100);
        assert_eq!(parsed.payload, vec![1, 2, 3, 4]);
    }

    #[test]
    fn vindex_blocks_round_trip() {
        let blocks = vec![
            VindexShannonBlock {
                first_token: 2,
                target_tokens: 3,
                payload: vec![1, 2, 3],
            },
            VindexShannonBlock {
                first_token: 5,
                target_tokens: 1,
                payload: vec![8, 13],
            },
        ];

        let encoded = encode_vindex_blocks(&blocks);
        let parsed = parse_vindex_blocks(&encoded).unwrap().unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].first_token, 2);
        assert_eq!(parsed[0].target_tokens, 3);
        assert_eq!(parsed[0].payload, vec![1, 2, 3]);
        assert_eq!(parsed[1].first_token, 5);
        assert_eq!(parsed[1].target_tokens, 1);
        assert_eq!(parsed[1].payload, vec![8, 13]);
        assert!(parse_vindex_blocks(&[1, 2, 3]).unwrap().is_none());
    }
}
