//! Subprocess orchestrator for `--bench-grid-lan`. Excluded from coverage
//! like other `*_runtime.rs` files — every code path needs to spawn the
//! larql binary against real shard endpoints to exercise.
//!
//! The pure helpers (config parsing, command templating, bench-row
//! parsing, byte estimation, CoV-based retry) live in `grid_lan.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use super::grid_lan::{
    command_for, estimate_bytes, extra_repeats_needed, parse_bench_output, safe_name,
    selected_runs, ByteEstimate, GridLanConfig, ParsedBench, RunRecord, RunSpec,
};

/// Top-level options for one grid-lan invocation. Mirrors run.py's CLI
/// surface so users coming from the Python orchestrator have an obvious
/// translation.
pub struct GridLanOptions {
    pub config_path: PathBuf,
    pub out_dir: PathBuf,
    pub only: Option<Vec<String>>,
    pub include_disabled: bool,
    pub dry_run: bool,
    pub timeout_secs: Option<u64>,
    /// Exp 41 LAN preregistration retry rule: when the per-row CoV
    /// across `defaults.repeats` repeats exceeds this threshold, the
    /// orchestrator runs up to `cov_extra_repeats` additional repeats
    /// before giving up.
    pub cov_threshold: f64,
    pub cov_extra_repeats: u32,
}

/// Entry point invoked from `run.rs` when `--bench-grid-lan` is set.
pub fn run(opts: GridLanOptions) -> Result<(), Box<dyn std::error::Error>> {
    let config: GridLanConfig = serde_json::from_slice(&std::fs::read(&opts.config_path)?)
        .map_err(|e| {
            format!(
                "failed to parse grid-lan config {}: {e}",
                opts.config_path.display()
            )
        })?;

    std::fs::create_dir_all(&opts.out_dir)?;
    let manifest_path = opts.out_dir.join("runs.jsonl");
    let mut manifest = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&manifest_path)?;

    let runs = selected_runs(&config, opts.only.as_deref(), opts.include_disabled);
    if runs.is_empty() {
        return Err("no runs selected".into());
    }

    let git_rev = read_git("rev-parse", &["HEAD"]).map(|s| s.trim().to_string());
    let git_dirty = read_git("status", &["--short"]).map(|s| !s.trim().is_empty());

    use std::io::Write;
    for run in runs {
        let base_repeats = run.repeats.unwrap_or(config.defaults.repeats).max(1);
        let mut samples_ms_per_tok: Vec<f64> = Vec::new();
        let do_one = |repeat_index: u32,
                      manifest: &mut std::fs::File,
                      samples: &mut Vec<f64>|
         -> Result<(), Box<dyn std::error::Error>> {
            match run_one(
                run,
                &config,
                &opts,
                repeat_index,
                git_rev.clone(),
                git_dirty,
            ) {
                Ok(rec) => {
                    if let Some(parsed) = &rec.parsed {
                        if let Some(first) = parsed.bench_rows.first() {
                            samples.push(first.mean_ms);
                        }
                    }
                    manifest.write_all((serde_json::to_string(&rec)? + "\n").as_bytes())?;
                    manifest.flush()?;
                    print_status(&rec, repeat_index);
                }
                Err(e) => eprintln!("{} r{}: error: {e}", run.id, repeat_index),
            }
            Ok(())
        };

        // Pass 1: configured repeats.
        for repeat_index in 0..base_repeats {
            do_one(repeat_index, &mut manifest, &mut samples_ms_per_tok)?;
        }
        // Pass 2 (Exp 41 CoV rule): add up to `cov_extra_repeats` more when
        // the per-row spread exceeded `cov_threshold`.
        let extra = extra_repeats_needed(
            &samples_ms_per_tok,
            opts.cov_threshold,
            opts.cov_extra_repeats,
        );
        for i in 0..extra {
            do_one(base_repeats + i, &mut manifest, &mut samples_ms_per_tok)?;
        }
    }

    println!("wrote {}", manifest_path.display());
    Ok(())
}

fn run_one(
    run: &RunSpec,
    config: &GridLanConfig,
    opts: &GridLanOptions,
    repeat_index: u32,
    git_rev: Option<String>,
    git_dirty: Option<bool>,
) -> Result<RunRecord, String> {
    let cmd = command_for(run, config);
    let stem = format!("{}.r{repeat_index}", safe_name(&run.id));
    let stdout_path = opts.out_dir.join(format!("{stem}.stdout.txt"));
    let stderr_path = opts.out_dir.join(format!("{stem}.stderr.txt"));

    let byte_estimate: Option<ByteEstimate> = run
        .estimate
        .as_ref()
        .and_then(|est| estimate_bytes(est, config.defaults.tokens).ok());

    let started_at = now_utc();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    let mut record = RunRecord {
        run_id: run.id.clone(),
        repeat_index,
        started_at: started_at.clone(),
        finished_at: None,
        command: cmd.clone(),
        env_overrides: run.env.clone(),
        cwd,
        git_rev,
        git_dirty,
        stdout_path: display_path(&opts.out_dir, &stdout_path),
        stderr_path: display_path(&opts.out_dir, &stderr_path),
        byte_estimate,
        dry_run: opts.dry_run,
        returncode: None,
        elapsed_ms: None,
        parsed: None,
    };

    if opts.dry_run {
        return Ok(record);
    }

    let (program, args) = cmd
        .split_first()
        .ok_or_else(|| format!("run {:?}: empty command after substitution", run.id))?;
    let mut command = Command::new(program);
    command.args(args);
    apply_env(&mut command, &run.env);

    let t0 = std::time::Instant::now();
    let output = command
        .output()
        .map_err(|e| format!("spawn {program:?}: {e}"))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let _ = std::fs::write(&stdout_path, &output.stdout);
    let _ = std::fs::write(&stderr_path, &output.stderr);

    let parsed: ParsedBench = parse_bench_output(&String::from_utf8_lossy(&output.stdout));
    record.finished_at = Some(now_utc());
    record.returncode = output.status.code();
    record.elapsed_ms = Some(elapsed_ms);
    record.parsed = Some(parsed);
    let _ = opts.timeout_secs; // future: wire actual timeout via wait_timeout

    Ok(record)
}

fn apply_env(cmd: &mut Command, env: &BTreeMap<String, String>) {
    for (k, v) in env {
        cmd.env(k, v);
    }
}

fn print_status(rec: &RunRecord, repeat_index: u32) {
    let status = if rec.dry_run {
        "dry-run".to_string()
    } else {
        format!("rc={}", rec.returncode.unwrap_or(-1))
    };
    println!("{} r{}: {}", rec.run_id, repeat_index, status);
}

fn now_utc() -> String {
    // ISO-8601-ish; second resolution is sufficient for run metadata.
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

fn display_path(base: &Path, full: &Path) -> String {
    full.strip_prefix(base)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| full.display().to_string())
}

fn read_git(sub: &str, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg(sub).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}
