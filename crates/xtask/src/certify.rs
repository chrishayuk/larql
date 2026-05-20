//! wasm32 certification cascade.
//!
//! For each crate:
//!   Level 1  — `cargo check --target wasm32-unknown-unknown`
//!   Build    — `cargo test --no-run --target wasm32-unknown-unknown --lib`
//!              produces the wasm binary we analyze
//!   Closure  — wasmparser + ascent Datalog rules → CERTIFIED / REFUTED
//!   Level 2  — `wasm-pack test --node -- --lib` (runtime confirmation)
//!   Level 4  — cfg-gated test collector (boundary map, informational)
//!   Level 5/6 — `cargo mutants` on wasm32-accessible sources
//!
//! Exit code is non-zero only when a crate regresses below its claimed-level.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Per-crate certification outcome.
#[derive(Debug)]
pub struct CertResult {
    pub crate_name: String,
    /// true = call graph CERTIFIED (refuted set empty)
    pub call_graph_certified: Option<bool>,
    pub refuted_witnesses: Vec<String>,
    pub level1_pass: bool,
    pub level2_pass: Option<bool>,
    pub level4_unit_cws: usize,
    pub level4_integ_cws: usize,
    pub mutant_survivors: Option<usize>,
    /// Non-zero → regression below claimed level.
    pub regression: bool,
}

/// Run the certification cascade for one or all workspace members.
/// Returns the exit code (0 = no regression).
pub fn run(crate_name: Option<&str>) -> Result<()> {
    // Detect new uncertified crates in PRs first.
    crate::new_crate_detector::run()?;

    let meta = crate::status::workspace_meta()?;
    let mut any_regression = false;

    for pkg in &meta.packages {
        if !meta.workspace_members.contains(&pkg.id) {
            continue;
        }
        if let Some(name) = crate_name {
            if pkg.name != name {
                continue;
            }
        }
        let crate_root = pkg.manifest_path.parent().unwrap().as_std_path().to_path_buf();
        let claimed_level = pkg
            .metadata
            .as_object()
            .and_then(|m| m.get("wasm-cert"))
            .and_then(|w| w.get("claimed-level"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u8;

        let result = certify_crate(&pkg.name, &crate_root, claimed_level)?;
        print_result(&result);

        if result.regression {
            any_regression = true;
        }

        // Post GitHub Check Annotation
        let conclusion = if result.regression {
            "failure"
        } else if claimed_level == 0 {
            "neutral"
        } else {
            "success"
        };
        let annotations = result
            .refuted_witnesses
            .iter()
            .map(|w| (format!("crates/{}", pkg.name), 1u32, w.clone()))
            .collect::<Vec<_>>();
        crate::github::post_check(&pkg.name, conclusion, &annotations)?;
    }

    if any_regression {
        anyhow::bail!("one or more crates regressed below their claimed certification level");
    }
    Ok(())
}

fn certify_crate(crate_name: &str, crate_root: &Path, claimed_level: u8) -> Result<CertResult> {
    println!("\n──── {crate_name} (claimed-level {claimed_level}) ────");

    let mut result = CertResult {
        crate_name: crate_name.to_owned(),
        call_graph_certified: None,
        refuted_witnesses: vec![],
        level1_pass: false,
        level2_pass: None,
        level4_unit_cws: 0,
        level4_integ_cws: 0,
        mutant_survivors: None,
        regression: false,
    };

    // ── Level 1: compile check ────────────────────────────────────────────────
    let level1 = run_level1(crate_name)?;
    result.level1_pass = level1.is_empty();
    if !result.level1_pass {
        println!("  LEVEL-1 FAIL (compile errors):");
        for w in &level1 {
            println!("    {w}");
        }
        if claimed_level >= 1 {
            result.regression = true;
        }
        return Ok(result);
    }
    println!("  Level 1: PASS (compile-consistent)");

    // ── Build wasm test binary ────────────────────────────────────────────────
    let wasm_bin = build_wasm_test_binary(crate_name, crate_root)?;

    // ── Call-graph closure analysis ──────────────────────────────────────────
    if let Some(ref path) = wasm_bin {
        match analyze_call_graph(crate_name, path) {
            Ok((certified, witnesses)) => {
                result.call_graph_certified = Some(certified);
                result.refuted_witnesses = witnesses;
                if certified {
                    println!("  Call graph: CERTIFIED (closed under sandbox boundary)");
                } else {
                    println!("  Call graph: REFUTED ({} witness(es)):", result.refuted_witnesses.len());
                    for w in &result.refuted_witnesses {
                        println!("    REFUTED  {w}");
                    }
                    // Call graph refutation is a regression for level ≥ 3.
                    if claimed_level >= 3 && !result.refuted_witnesses.is_empty() {
                        result.regression = true;
                    }
                }
            }
            Err(e) => {
                eprintln!("  warning: call-graph analysis failed: {e}");
            }
        }
    }

    // ── Level 2: runtime confirmation ────────────────────────────────────────
    let level2 = run_level2(crate_root)?;
    result.level2_pass = Some(level2);
    if level2 {
        println!("  Level 2: PASS (runtime-consistent, Node.js)");
    } else {
        println!("  Level 2: FAIL (wasm-pack test --lib returned non-zero)");
        if claimed_level >= 2 {
            result.regression = true;
        }
    }

    // ── Level 4: boundary map ────────────────────────────────────────────────
    let audit = crate::audit::audit_crate(crate_name, crate_root)?;
    result.level4_unit_cws = audit.unit_counterwits.len();
    result.level4_integ_cws = audit.integ_counterwits.len();
    println!(
        "  Level 4: {}u + {}i counterwit­nesses (native-only boundary)",
        result.level4_unit_cws, result.level4_integ_cws
    );

    // ── Level 5/6: mutation testing ──────────────────────────────────────────
    let accessible_files = accessible_source_files(crate_root, &audit.accessible);
    if !accessible_files.is_empty() {
        match run_mutants(crate_root, &accessible_files) {
            Ok(survivors) => {
                result.mutant_survivors = Some(survivors);
                if survivors == 0 {
                    println!("  Level 5/6: PASS (0 surviving mutants — runtime-sound)");
                } else {
                    println!("  Level 5/6: {survivors} surviving mutant(s) — not yet runtime-sound");
                    if claimed_level >= 6 {
                        result.regression = true;
                    }
                }
            }
            Err(e) => {
                eprintln!("  warning: mutation testing failed: {e}");
            }
        }
    }

    Ok(result)
}

/// Run `cargo check --target wasm32-unknown-unknown --message-format json`.
/// Returns a list of error diagnostic messages (empty = pass).
fn run_level1(crate_name: &str) -> Result<Vec<String>> {
    let output = Command::new("cargo")
        .args([
            "check",
            "--target",
            "wasm32-unknown-unknown",
            "--message-format",
            "json",
            "-p",
            crate_name,
            "--lib",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .context("cargo check")?;

    let mut errors = vec![];
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg["reason"] == "compiler-message"
                && msg["message"]["level"] == "error"
            {
                let text = msg["message"]["rendered"]
                    .as_str()
                    .unwrap_or("")
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_owned();
                errors.push(text);
            }
        }
    }
    Ok(errors)
}

/// Build the wasm test binary via `cargo test --no-run`.
/// Returns the path to the `.wasm` artifact, or None if the build fails.
fn build_wasm_test_binary(crate_name: &str, crate_root: &Path) -> Result<Option<PathBuf>> {
    let output = Command::new("cargo")
        .args([
            "test",
            "--no-run",
            "--target",
            "wasm32-unknown-unknown",
            "--message-format",
            "json",
            "-p",
            crate_name,
            "--lib",
        ])
        .current_dir(crate_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .context("cargo test --no-run")?;

    // Parse JSON output for the compiler-artifact with a .wasm executable.
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg["reason"] == "compiler-artifact" {
                if let Some(exec) = msg["executable"].as_str() {
                    if exec.ends_with(".wasm") {
                        return Ok(Some(PathBuf::from(exec)));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// Analyze the call graph of the wasm binary and run Datalog rules.
/// Returns `(certified, witnesses)`.
fn analyze_call_graph(
    _crate_name: &str,
    wasm_path: &Path,
) -> Result<(bool, Vec<String>)> {
    let bytes = std::fs::read(wasm_path).context("read wasm binary")?;
    let mut facts = crate::wasm_facts::extract(&bytes)?;

    let non_intrinsic_indices: Vec<u32> = facts.non_intrinsic_imports.iter().map(|(_, _, idx)| *idx).collect();
    let roots: Vec<u32> = facts.roots.iter().map(|(_, idx)| *idx).collect();
    // take() extracts ownership without partial-moving the struct; facts stays
    // alive for label() below.
    let calls = std::mem::take(&mut facts.calls);
    let indirect_calls = std::mem::take(&mut facts.indirect_calls);
    let indirect_set: std::collections::HashSet<u32> = indirect_calls.iter().copied().collect();
    let result = crate::rules::analyze(calls, non_intrinsic_indices, indirect_calls, roots);

    let witnesses: Vec<String> = result
        .refuted_indices()
        .iter()
        .map(|idx| {
            let label = crate::wasm_facts::label(&facts, *idx);
            let reason = if indirect_set.contains(idx) {
                "indirect-call (unresolved dispatch)"
            } else {
                "non-intrinsic-import (containment violation)"
            };
            format!("fn {label}  [{reason}]")
        })
        .collect();

    Ok((result.is_certified(), witnesses))
}

fn run_level2(crate_root: &Path) -> Result<bool> {
    let status = Command::new("wasm-pack")
        .args(["test", "--node", "--", "--lib"])
        .current_dir(crate_root)
        .status()
        .context("wasm-pack test --node")?;
    Ok(status.success())
}

fn run_mutants(
    crate_root: &Path,
    accessible_files: &[PathBuf],
) -> Result<usize> {
    let mut cmd = Command::new("cargo");
    cmd.arg("mutants").arg("--no-shuffle");
    for f in accessible_files {
        cmd.args(["--file", &f.display().to_string()]);
    }
    cmd.args([
        "--test-tool",
        "cargo",
        "--",
        "--target",
        "wasm32-unknown-unknown",
        "--lib",
    ]);
    cmd.arg("--timeout").arg("300");
    cmd.current_dir(crate_root);

    let output = cmd.output().context("cargo mutants")?;

    // cargo mutants exits with 2 when there are survivors.
    // Parse its output to count survivors.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut survivors = 0;
    for line in stdout.lines() {
        if line.contains("mutant survived") || line.contains("SURVIVED") {
            survivors += 1;
        }
    }
    Ok(survivors)
}

fn accessible_source_files(crate_root: &Path, accessible: &[String]) -> Vec<PathBuf> {
    let src_dir = crate_root.join("src");
    accessible
        .iter()
        .flat_map(|m| crate::audit::module_paths(&src_dir, m))
        .collect()
}

fn print_result(r: &CertResult) {
    if r.regression {
        println!("  !! REGRESSION: {} regressed below claimed level", r.crate_name);
    } else {
        println!("  OK: {}", r.crate_name);
    }
}
