//! Pure helpers for `--bench-grid-lan` (LAN preregistration matrix).
//!
//! Mirrors the Exp 41 `run.py` orchestrator
//! (`experiments/41_residual_transport_grid/`): take a JSON config that
//! lists named bench runs (each one a self-contained `larql bench …`
//! invocation with optional env overrides), execute the matrix
//! repeatedly, and emit a JSONL manifest plus a summary table.
//!
//! This file is the **pure** layer: config types, template substitution,
//! bench-output parsing, byte estimation, and the coefficient-of-variation
//! retry rule from the Exp 41 spec. The subprocess driver + filesystem
//! writes live in `grid_lan_runtime.rs` (excluded from coverage like the
//! other `*_runtime.rs` files).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

// ── Config schema (matches run.py / config.example.json) ─────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GridLanConfig {
    #[serde(default = "default_bin")]
    pub larql_bin: String,
    #[serde(default)]
    pub defaults: Defaults,
    #[serde(default)]
    pub models: Models,
    pub runs: Vec<RunSpec>,
}

fn default_bin() -> String {
    "./target/release/larql".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Defaults {
    #[serde(default = "default_repeats")]
    pub repeats: u32,
    #[serde(default = "default_tokens")]
    pub tokens: u32,
    #[serde(default = "default_warmup")]
    pub warmup: u32,
    #[serde(default = "default_prompt")]
    pub prompt: String,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            repeats: default_repeats(),
            tokens: default_tokens(),
            warmup: default_warmup(),
            prompt: default_prompt(),
        }
    }
}

fn default_repeats() -> u32 {
    1
}
fn default_tokens() -> u32 {
    30
}
fn default_warmup() -> u32 {
    5
}
fn default_prompt() -> String {
    "The capital of France is".to_string()
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Models {
    #[serde(default)]
    pub dense: String,
    #[serde(default)]
    pub moe: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RunSpec {
    pub id: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub kind: String,
    pub command: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub vars: BTreeMap<String, String>,
    #[serde(default)]
    pub estimate: Option<Estimate>,
    /// Per-run repeat override. When `None`, the run inherits
    /// `defaults.repeats` from the top-level config.
    #[serde(default)]
    pub repeats: Option<u32>,
}

fn default_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Estimate {
    pub model_kind: String,
    #[serde(default = "default_dispatch")]
    pub dispatch: String,
    pub encoding: String,
    #[serde(default = "default_response_encoding")]
    pub response_encoding: String,
    pub hidden: u32,
    pub layers: u32,
    #[serde(default = "default_shards")]
    pub shards: u32,
    #[serde(default)]
    pub active_shards: Option<u32>,
}

fn default_dispatch() -> String {
    "streaming".to_string()
}
fn default_response_encoding() -> String {
    "f32".to_string()
}
fn default_shards() -> u32 {
    1
}

// ── Templating ───────────────────────────────────────────────────────────────

/// Build the literal argv for one run by substituting `{...}` placeholders
/// against a per-run context built from `config.larql_bin`, `models`,
/// `defaults`, and per-run `vars`.
///
/// Unrecognised placeholders are left in place — matches `str.format`
/// behaviour from `run.py` would have raised, but the experiment never
/// hits that path because configs are checked in lockstep with the
/// template keys.
pub fn command_for(run: &RunSpec, config: &GridLanConfig) -> Vec<String> {
    let mut ctx: BTreeMap<String, String> = BTreeMap::new();
    ctx.insert("larql_bin".into(), config.larql_bin.clone());
    ctx.insert("dense_model".into(), config.models.dense.clone());
    ctx.insert("moe_model".into(), config.models.moe.clone());
    ctx.insert("tokens".into(), config.defaults.tokens.to_string());
    ctx.insert("warmup".into(), config.defaults.warmup.to_string());
    ctx.insert("prompt".into(), config.defaults.prompt.clone());
    // Per-run vars override the defaults.
    for (k, v) in &run.vars {
        ctx.insert(k.clone(), v.clone());
    }
    run.command
        .iter()
        .map(|tok| substitute(tok, &ctx))
        .collect()
}

/// Lightweight `{name}` substitution. Unknown names are left literal.
/// Two consecutive `{{` / `}}` produce a single literal brace, matching
/// Python `str.format` so authors who know the run.py syntax aren't
/// surprised.
pub fn substitute(template: &str, ctx: &BTreeMap<String, String>) -> String {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut name = String::new();
                let mut closed = false;
                for nc in chars.by_ref() {
                    if nc == '}' {
                        closed = true;
                        break;
                    }
                    name.push(nc);
                }
                if !closed {
                    // Dangling opener — keep verbatim so it isn't lost.
                    out.push('{');
                    out.push_str(&name);
                    continue;
                }
                match ctx.get(&name) {
                    Some(v) => out.push_str(v),
                    None => {
                        out.push('{');
                        out.push_str(&name);
                        out.push('}');
                    }
                }
            }
            _ => out.push(c),
        }
    }
    out
}

// ── Bench output parsing ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedBench {
    pub bench_rows: Vec<ParsedRow>,
    pub remote_stage_ms: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedRow {
    pub backend: String,
    pub prefill_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub tok_per_s: f64,
    pub steps: u32,
    pub note: String,
}

/// Parse the `larql bench` stdout table. The shape (from
/// `bench/output.rs:format_data_row`) is:
///
/// ```text
///   Backend                  prefill       mean        p50      tok/s  steps  notes…
///   ──────────────…
///   metal                    123.4ms     12.34ms    11.23ms    81.2     50  note
/// ```
///
/// Plus the optional "Per-stage average" / "Remote FFN per token"
/// blocks the renderer prints after the table. Lines that don't fit
/// the row shape are ignored.
pub fn parse_bench_output(text: &str) -> ParsedBench {
    let mut bench_rows = Vec::new();
    let mut remote_stage_ms = BTreeMap::new();
    for line in text.lines() {
        if let Some(row) = parse_data_row(line) {
            bench_rows.push(row);
            continue;
        }
        if let Some((label, ms)) = parse_remote_stage(line) {
            remote_stage_ms.insert(label, ms);
        }
    }
    ParsedBench {
        bench_rows,
        remote_stage_ms,
    }
}

/// One data row of the bench table. Returns `None` for header /
/// separator / blank / mid-stage-breakdown lines.
fn parse_data_row(line: &str) -> Option<ParsedRow> {
    // Data rows always start with exactly two spaces in the renderer.
    // The header line also starts with two spaces; reject it by
    // peeking at the first non-blank word.
    if !line.starts_with("  ") {
        return None;
    }
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('─') || trimmed.starts_with("Backend") {
        return None;
    }

    // Split into whitespace tokens. We expect at least:
    //   backend_word(s) prefill_ms mean_ms p50_ms tok_s steps [wire] [note...]
    // Where backend is column-aligned to 24 chars but always one
    // identifier (no whitespace) in the existing renderer.
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.len() < 5 {
        return None;
    }
    // Find the first token that ends with `ms` and has a parseable
    // numeric prefix — that's prefill. Everything before it is the
    // backend name. Indented stage-breakdown lines also have `ms`
    // tokens, but they're rejected by the leading-spaces check below
    // (they start with 4+ spaces, not 2 followed by a non-space).
    if line.starts_with("   ") {
        // 3+ leading spaces => not a top-level row.
        return None;
    }
    let prefill_idx = tokens.iter().position(|t| parse_ms(t).is_some())?;
    let prefill = parse_ms(tokens[prefill_idx])?;
    let mean = tokens.get(prefill_idx + 1).and_then(|t| parse_ms(t))?;
    let p50 = tokens.get(prefill_idx + 2).and_then(|t| parse_ms(t))?;
    let tok_s = tokens
        .get(prefill_idx + 3)
        .and_then(|t| t.parse::<f64>().ok())?;
    let steps = tokens
        .get(prefill_idx + 4)
        .and_then(|t| t.parse::<u32>().ok())?;

    let backend = tokens[..prefill_idx].join(" ");
    let note_start = prefill_idx + 5;
    let note = if note_start < tokens.len() {
        // Skip the optional wire_KB/tok column when it looks numeric.
        let after = &tokens[note_start..];
        if !after.is_empty() && after[0].parse::<f64>().is_ok() {
            after[1..].join(" ")
        } else {
            after.join(" ")
        }
    } else {
        String::new()
    };

    Some(ParsedRow {
        backend,
        prefill_ms: prefill,
        mean_ms: mean,
        p50_ms: p50,
        tok_per_s: tok_s,
        steps,
        note,
    })
}

/// Parse a number that ends with the literal `ms` suffix (e.g.
/// `12.34ms`). Returns the numeric part as f64.
fn parse_ms(tok: &str) -> Option<f64> {
    let stripped = tok.strip_suffix("ms")?;
    stripped.parse::<f64>().ok()
}

/// Pick "attn+norm+lmhead", "ffn round-trips", "total/tok" lines out of
/// the post-table breakdown block (matches the regex from run.py).
fn parse_remote_stage(line: &str) -> Option<(String, f64)> {
    let trimmed = line.trim_start();
    let trim_count = line.len() - trimmed.len();
    if trim_count < 4 {
        // Stage rows always have 4+ leading spaces in the renderer.
        return None;
    }
    let labels = ["attn+norm+lmhead", "ffn round-trips", "total/tok"];
    for label in labels {
        if let Some(rest) = trimmed.strip_prefix(label) {
            let rest = rest.trim();
            let ms_tok = rest.split_whitespace().next()?;
            let value = parse_ms(ms_tok)?;
            let key = label.replace(['+', ' '], "_");
            return Some((key, value));
        }
    }
    None
}

// ── Byte estimate (mirrors run.py:estimate_bytes) ────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ByteEstimate {
    pub model_kind: String,
    pub dispatch: String,
    pub hidden: u32,
    pub layers: u32,
    pub shards: u32,
    pub active_shards_assumed: u32,
    pub encoding: String,
    pub response_encoding: String,
    pub upload_bytes_per_activation: u64,
    pub download_bytes_per_activation: u64,
    pub upload_bytes_per_token: u64,
    pub download_bytes_per_token: u64,
    pub total_bytes_per_token: u64,
    pub total_bytes_measured_tokens: u64,
    pub note: String,
}

/// Q8K layout: hidden one-byte quantised values + per-256-block scale
/// (f32) + 8 per-block sub-scales (f16 each).
///
/// Matches `run.py:q8k_bytes`. Used by `encoded_bytes` for the q8k
/// encoding.
pub fn q8k_bytes(hidden: u32) -> u64 {
    let blocks = hidden.div_ceil(256);
    hidden as u64 + (blocks as u64) * 4 + (blocks as u64) * 8 * 2
}

/// Bytes per activation under a given wire encoding.
pub fn encoded_bytes(hidden: u32, encoding: &str) -> Result<u64, String> {
    match encoding {
        "f32" => Ok(hidden as u64 * 4),
        "f16" => Ok(hidden as u64 * 2),
        "q8k" => Ok(q8k_bytes(hidden)),
        "none" => Ok(0),
        other => Err(format!("unknown encoding {other:?}")),
    }
}

/// Project upload/download bytes per token for a run, mirroring the
/// `run.py:estimate_bytes` math.
pub fn estimate_bytes(est: &Estimate, default_tokens: u32) -> Result<ByteEstimate, String> {
    let upload_per_activation = encoded_bytes(est.hidden, &est.encoding)?;
    let download_per_activation = encoded_bytes(est.hidden, &est.response_encoding)?;

    let fanout = if est.model_kind == "dense" {
        1
    } else if est.dispatch == "streaming" {
        est.shards
    } else {
        est.active_shards.unwrap_or(est.shards)
    };

    let upload_per_token = est.layers as u64 * fanout as u64 * upload_per_activation;
    let download_per_token = est.layers as u64 * fanout as u64 * download_per_activation;
    let total_per_token = upload_per_token + download_per_token;

    Ok(ByteEstimate {
        model_kind: est.model_kind.clone(),
        dispatch: est.dispatch.clone(),
        hidden: est.hidden,
        layers: est.layers,
        shards: est.shards,
        active_shards_assumed: fanout,
        encoding: est.encoding.clone(),
        response_encoding: est.response_encoding.clone(),
        upload_bytes_per_activation: upload_per_activation,
        download_bytes_per_activation: download_per_activation,
        upload_bytes_per_token: upload_per_token,
        download_bytes_per_token: download_per_token,
        total_bytes_per_token: total_per_token,
        total_bytes_measured_tokens: total_per_token * default_tokens as u64,
        note: "estimate from config, not captured from sockets".to_string(),
    })
}

// ── Stat helpers ─────────────────────────────────────────────────────────────

/// Sample mean. Returns `None` on an empty slice so the caller can
/// distinguish "no observations" from a true zero mean.
pub fn mean(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(samples.iter().sum::<f64>() / samples.len() as f64)
}

/// Coefficient of variation = stddev / mean. `None` when the sample
/// has fewer than 2 points or the mean is non-positive (CoV is
/// dimensionless and undefined there).
pub fn coefficient_of_variation(samples: &[f64]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let m = mean(samples)?;
    if m <= 0.0 {
        return None;
    }
    // Population stddev (Bessel correction not needed for the n=3..5
    // batches the Exp 41 spec actually runs).
    let var = samples.iter().map(|x| (x - m).powi(2)).sum::<f64>() / samples.len() as f64;
    Some(var.sqrt() / m)
}

/// Exp 41 §LAN Preregistration retry rule: if the spread across
/// repeats exceeds the threshold, run the bench up to `max_extra`
/// more times. Returns the number of *additional* repeats the caller
/// should issue (0 when the current batch already settles).
pub fn extra_repeats_needed(samples: &[f64], threshold: f64, max_extra: u32) -> u32 {
    match coefficient_of_variation(samples) {
        Some(cov) if cov > threshold => max_extra,
        _ => 0,
    }
}

// ── Filename sanitiser (matches run.py:safe_name) ────────────────────────────

/// Make a path-safe slug — keeps `[A-Za-z0-9_.-]`, collapses runs of
/// other characters to `_`, and trims leading/trailing underscores.
pub fn safe_name(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut last_underscore = false;
    for c in value.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' {
            out.push(c);
            last_underscore = c == '_';
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    trimmed
}

// ── Run selection helper ─────────────────────────────────────────────────────

/// Pick the subset of runs that match the CLI filters. When `only`
/// contains entries, only those `id` values are included. When
/// `include_disabled` is false, runs with `enabled = false` are
/// dropped.
pub fn selected_runs<'a>(
    config: &'a GridLanConfig,
    only: Option<&[String]>,
    include_disabled: bool,
) -> Vec<&'a RunSpec> {
    let only_set: Option<std::collections::HashSet<&str>> =
        only.map(|v| v.iter().map(String::as_str).collect());
    config
        .runs
        .iter()
        .filter(|r| match &only_set {
            Some(ids) => ids.contains(r.id.as_str()),
            None => true,
        })
        .filter(|r| include_disabled || r.enabled)
        .collect()
}

// ── JSONL record ─────────────────────────────────────────────────────────────

/// One JSONL line written to the run manifest. Matches the run.py
/// record shape so existing tooling consuming `runs.jsonl` keeps
/// working when the orchestration moves into the Rust CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub repeat_index: u32,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    pub command: Vec<String>,
    pub env_overrides: BTreeMap<String, String>,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_rev: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_dirty: Option<bool>,
    pub stdout_path: String,
    pub stderr_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_estimate: Option<ByteEstimate>,
    pub dry_run: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returncode: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parsed: Option<ParsedBench>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_from(pairs: &[(&'static str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.to_string()))
            .collect()
    }

    // ── substitute ──────────────────────────────────────────────────────────

    #[test]
    fn substitute_replaces_named_placeholders() {
        let ctx = ctx_from(&[("name", "alice"), ("count", "3")]);
        assert_eq!(
            substitute("hi {name}, take {count} steps", &ctx),
            "hi alice, take 3 steps"
        );
    }

    #[test]
    fn substitute_leaves_unknown_placeholders_literal() {
        let ctx = ctx_from(&[("name", "alice")]);
        assert_eq!(substitute("hi {name}, {age}", &ctx), "hi alice, {age}");
    }

    #[test]
    fn substitute_handles_double_braces_as_literal_brace() {
        let ctx = ctx_from(&[("x", "y")]);
        assert_eq!(
            substitute("{{not a placeholder}}", &ctx),
            "{not a placeholder}"
        );
    }

    #[test]
    fn substitute_keeps_dangling_opener_verbatim() {
        let ctx = ctx_from(&[]);
        assert_eq!(substitute("oops {unfinished", &ctx), "oops {unfinished");
    }

    // ── command_for ─────────────────────────────────────────────────────────

    fn sample_config() -> GridLanConfig {
        GridLanConfig {
            larql_bin: "./bin/larql".into(),
            defaults: Defaults {
                repeats: 3,
                tokens: 60,
                warmup: 5,
                prompt: "Hello".into(),
            },
            models: Models {
                dense: "models/dense.vindex".into(),
                moe: "models/moe.vindex".into(),
            },
            runs: vec![RunSpec {
                id: "dense-stream".into(),
                enabled: true,
                kind: "dense".into(),
                command: vec![
                    "{larql_bin}".into(),
                    "bench".into(),
                    "{dense_model}".into(),
                    "--tokens".into(),
                    "{tokens}".into(),
                    "--warmup".into(),
                    "{warmup}".into(),
                ],
                env: BTreeMap::new(),
                vars: BTreeMap::new(),
                estimate: None,
                repeats: None,
            }],
        }
    }

    #[test]
    fn command_for_substitutes_all_default_keys() {
        let cfg = sample_config();
        let argv = command_for(&cfg.runs[0], &cfg);
        assert_eq!(
            argv,
            vec![
                "./bin/larql".to_string(),
                "bench".into(),
                "models/dense.vindex".into(),
                "--tokens".into(),
                "60".into(),
                "--warmup".into(),
                "5".into(),
            ]
        );
    }

    #[test]
    fn command_for_per_run_vars_override_defaults() {
        let mut cfg = sample_config();
        cfg.runs[0].vars.insert("tokens".into(), "120".into());
        let argv = command_for(&cfg.runs[0], &cfg);
        assert!(argv.iter().any(|a| a == "120"));
        assert!(!argv.iter().any(|a| a == "60"));
    }

    // ── parse_bench_output ──────────────────────────────────────────────────

    #[test]
    fn parse_bench_output_picks_up_data_rows() {
        // Matches the renderer in `bench/output.rs:format_data_row`.
        let stdout = "\
  Backend                    prefill       mean        p50      tok/s  steps  notes
  ────────────────────────────────────────────────────────────────────────────────
  metal                       125.2ms     12.34ms    11.05ms       81.0      50  ok
  cpu                          80.4ms     30.12ms    28.90ms       33.2      50  warm
";
        let parsed = parse_bench_output(stdout);
        assert_eq!(parsed.bench_rows.len(), 2);
        let row = &parsed.bench_rows[0];
        assert_eq!(row.backend, "metal");
        assert!((row.prefill_ms - 125.2).abs() < 1e-6);
        assert!((row.mean_ms - 12.34).abs() < 1e-6);
        assert!((row.p50_ms - 11.05).abs() < 1e-6);
        assert!((row.tok_per_s - 81.0).abs() < 1e-6);
        assert_eq!(row.steps, 50);
        assert_eq!(row.note, "ok");
    }

    #[test]
    fn parse_bench_output_skips_header_and_separator() {
        let stdout = "\
  Backend                    prefill       mean        p50      tok/s  steps  notes
  ────────────────────────────────────────────────────────────────────────────────
";
        let parsed = parse_bench_output(stdout);
        assert!(parsed.bench_rows.is_empty());
    }

    #[test]
    fn parse_bench_output_extracts_remote_stage_breakdown() {
        let stdout = "\
  Backend                    prefill       mean        p50      tok/s  steps  notes
  ────────────────────────────────────────────────────────────────────────────────
  remote-ffn                  100.0ms     12.00ms    11.50ms       83.3      50  http
    attn+norm+lmhead       3.20ms
    ffn round-trips        9.10ms
    total/tok             12.30ms
";
        let parsed = parse_bench_output(stdout);
        assert_eq!(parsed.bench_rows.len(), 1);
        // Keys mirror run.py: `+` and ` ` become `_`; other chars (here `/`) stay.
        assert!((parsed.remote_stage_ms["attn_norm_lmhead"] - 3.20).abs() < 1e-6);
        assert!((parsed.remote_stage_ms["ffn_round-trips"] - 9.10).abs() < 1e-6);
        assert!((parsed.remote_stage_ms["total/tok"] - 12.30).abs() < 1e-6);
    }

    #[test]
    fn parse_bench_output_ignores_malformed_lines() {
        let stdout = "\
  Backend                    prefill
  not a row at all
   indented_subline 4.00ms
";
        let parsed = parse_bench_output(stdout);
        assert!(parsed.bench_rows.is_empty());
        assert!(parsed.remote_stage_ms.is_empty());
    }

    // ── encoded_bytes / q8k_bytes ────────────────────────────────────────────

    #[test]
    fn q8k_bytes_layout_matches_python_reference() {
        // hidden=2816, blocks=11 → 2816 + 11*4 + 11*8*2 = 3036
        assert_eq!(q8k_bytes(2816), 3036);
        // hidden=256 → 1 block → 256 + 4 + 16 = 276
        assert_eq!(q8k_bytes(256), 276);
    }

    #[test]
    fn encoded_bytes_known_formats() {
        assert_eq!(encoded_bytes(2816, "f32").unwrap(), 11264);
        assert_eq!(encoded_bytes(2816, "f16").unwrap(), 5632);
        assert_eq!(encoded_bytes(2816, "q8k").unwrap(), 3036);
        assert_eq!(encoded_bytes(2816, "none").unwrap(), 0);
        assert!(encoded_bytes(2816, "bogus").is_err());
    }

    // ── estimate_bytes ───────────────────────────────────────────────────────

    #[test]
    fn estimate_bytes_dense_stream_assumes_fanout_one() {
        let est = Estimate {
            model_kind: "dense".into(),
            dispatch: "streaming".into(),
            encoding: "f32".into(),
            response_encoding: "f32".into(),
            hidden: 2816,
            layers: 60,
            shards: 2,
            active_shards: None,
        };
        let out = estimate_bytes(&est, 30).unwrap();
        // 60 layers × 1 fanout × 11264 bytes = 675840
        assert_eq!(out.upload_bytes_per_token, 675840);
        assert_eq!(out.active_shards_assumed, 1);
    }

    #[test]
    fn estimate_bytes_moe_streaming_fanout_equals_shards() {
        let est = Estimate {
            model_kind: "moe".into(),
            dispatch: "streaming".into(),
            encoding: "f32".into(),
            response_encoding: "f32".into(),
            hidden: 2816,
            layers: 30,
            shards: 4,
            active_shards: None,
        };
        let out = estimate_bytes(&est, 30).unwrap();
        assert_eq!(out.active_shards_assumed, 4);
        // 30 × 4 × 11264 = 1351680
        assert_eq!(out.upload_bytes_per_token, 1351680);
    }

    #[test]
    fn estimate_bytes_moe_batch_prefers_active_shards_override() {
        let est = Estimate {
            model_kind: "moe".into(),
            dispatch: "batch".into(),
            encoding: "q8k".into(),
            response_encoding: "f32".into(),
            hidden: 2816,
            layers: 30,
            shards: 4,
            active_shards: Some(2),
        };
        let out = estimate_bytes(&est, 30).unwrap();
        assert_eq!(out.active_shards_assumed, 2);
        // Upload: 30 layers × 2 fanout × q8k(2816)=3036 = 182160
        assert_eq!(out.upload_bytes_per_token, 182160);
    }

    // ── stats / repeat decision ─────────────────────────────────────────────

    #[test]
    fn mean_and_cov_basic() {
        assert!(mean(&[]).is_none());
        assert!((mean(&[1.0, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-6);
        assert!(coefficient_of_variation(&[5.0]).is_none(), "1 sample");
        assert!(coefficient_of_variation(&[0.0, 0.0]).is_none(), "mean 0");
        // 90/100/110 → mean 100, stddev sqrt(200/3) ≈ 8.165 → cov ≈ 0.0816
        let cov = coefficient_of_variation(&[90.0, 100.0, 110.0]).unwrap();
        assert!((cov - 0.0816).abs() < 1e-3, "got {cov}");
    }

    #[test]
    fn extra_repeats_needed_triggers_only_above_threshold() {
        // CoV ~0.08 < 0.15 → no extra repeats.
        assert_eq!(extra_repeats_needed(&[90.0, 100.0, 110.0], 0.15, 2), 0);
        // CoV ~0.45 > 0.15 → 2 more.
        assert_eq!(extra_repeats_needed(&[50.0, 100.0, 150.0], 0.15, 2), 2);
        // Fewer than 2 samples → cannot decide, default to 0.
        assert_eq!(extra_repeats_needed(&[100.0], 0.15, 2), 0);
    }

    // ── safe_name ───────────────────────────────────────────────────────────

    #[test]
    fn safe_name_collapses_unsafe_chars() {
        assert_eq!(safe_name("dense-http stream f32"), "dense-http_stream_f32");
        assert_eq!(safe_name("/leading/and trailing/"), "leading_and_trailing");
        assert_eq!(safe_name("ok.name_42"), "ok.name_42");
    }

    // ── selected_runs ───────────────────────────────────────────────────────

    fn cfg_with_runs(specs: Vec<(&str, bool)>) -> GridLanConfig {
        GridLanConfig {
            larql_bin: "/bin".into(),
            defaults: Defaults::default(),
            models: Models::default(),
            runs: specs
                .into_iter()
                .map(|(id, enabled)| RunSpec {
                    id: id.into(),
                    enabled,
                    kind: String::new(),
                    command: vec![],
                    env: BTreeMap::new(),
                    vars: BTreeMap::new(),
                    estimate: None,
                    repeats: None,
                })
                .collect(),
        }
    }

    #[test]
    fn selected_runs_filters_disabled_by_default() {
        let cfg = cfg_with_runs(vec![("a", true), ("b", false), ("c", true)]);
        let picked: Vec<&str> = selected_runs(&cfg, None, false)
            .iter()
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(picked, vec!["a", "c"]);
    }

    #[test]
    fn selected_runs_include_disabled_returns_all() {
        let cfg = cfg_with_runs(vec![("a", true), ("b", false)]);
        let picked: Vec<&str> = selected_runs(&cfg, None, true)
            .iter()
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(picked, vec!["a", "b"]);
    }

    #[test]
    fn selected_runs_only_filters_to_named_ids() {
        let cfg = cfg_with_runs(vec![("a", true), ("b", true), ("c", true)]);
        let only = vec!["b".to_string(), "c".to_string()];
        let picked: Vec<&str> = selected_runs(&cfg, Some(&only), false)
            .iter()
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(picked, vec!["b", "c"]);
    }

    // ── config deserialization smoke test ────────────────────────────────────

    #[test]
    fn config_deserializes_minimal_run_list() {
        let json = r#"{
            "runs": [
                {
                    "id": "minimal",
                    "command": ["./larql", "bench", "{prompt}"]
                }
            ]
        }"#;
        let cfg: GridLanConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.runs.len(), 1);
        assert_eq!(cfg.runs[0].id, "minimal");
        assert!(cfg.runs[0].enabled);
        assert_eq!(cfg.larql_bin, "./target/release/larql");
        assert_eq!(cfg.defaults.tokens, 30);
    }

    #[test]
    fn config_round_trips_estimate_block() {
        let json = r#"{
            "runs": [{
                "id": "with-est",
                "command": [],
                "estimate": {
                    "model_kind": "moe",
                    "encoding": "f16",
                    "hidden": 2816,
                    "layers": 30
                }
            }]
        }"#;
        let cfg: GridLanConfig = serde_json::from_str(json).unwrap();
        let est = cfg.runs[0].estimate.as_ref().unwrap();
        assert_eq!(est.model_kind, "moe");
        assert_eq!(est.dispatch, "streaming"); // default
        assert_eq!(est.shards, 1); // default
        let bytes = estimate_bytes(est, 30).unwrap();
        // 30 × 1 × f16(2816)=5632 = 168960
        assert_eq!(bytes.upload_bytes_per_token, 168960);
    }
}
