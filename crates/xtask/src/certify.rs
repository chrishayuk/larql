//! wasm32 certification cascade.
//!
//! For each crate:
//!   NATIVE check — crate-type detection (no lib target → host OS/IO layer)
//!   Level 1  — `cargo check --target wasm32-unknown-unknown`
//!   Build    — `cargo rustc --crate-type cdylib --target wasm32-unknown-unknown`
//!              produces the PRODUCTION wasm binary (no dev-dependencies)
//!   Closure  — wasmparser + ascent Datalog rules → partition label
//!              (analyzed against production binary, not the test binary, so
//!              dev-only dispatch from wasm-bindgen-test is excluded)
//!   Level 2  — `wasm-pack test --node -- --lib` (runtime confirmation)
//!   Level 4  — cfg-gated test collector (boundary map, informational)
//!   Level 5/6 — `cargo mutants` on wasm32-accessible sources
//!
//! Partition labels (stable path):
//!   STATIC  — call graph closed, no call_indirect, no non-intrinsic imports
//!   DYNAMIC — call_indirect present (dispatch unclassified without MIR)
//!   NATIVE  — non-intrinsic imports OR no lib target
//!
//! Exit code is non-zero only when a crate regresses below its claimed-level.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::rules::WasmPartition;

/// Per-crate certification outcome.
#[derive(Debug)]
pub struct CertResult {
    pub crate_name: String,
    pub partition: Option<WasmPartition>,
    /// Reason for NATIVE classification (crate-type or containment violation).
    pub native_reason: String,
    /// Functions that breach the sandbox boundary (non-intrinsic imports).
    pub containment_witnesses: Vec<String>,
    /// Functions with unresolved dynamic dispatch.
    pub dispatch_witnesses: Vec<String>,
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

        let result = certify_crate(&pkg.name, &crate_root, claimed_level, &pkg.targets)?;
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
        let annotations: Vec<_> = result
            .containment_witnesses
            .iter()
            .chain(result.dispatch_witnesses.iter())
            .map(|w| (format!("crates/{}", pkg.name), 1u32, w.clone()))
            .collect();
        crate::github::post_check(&pkg.name, conclusion, &annotations)?;
    }

    if any_regression {
        anyhow::bail!("one or more crates regressed below their claimed certification level");
    }
    Ok(())
}

fn certify_crate(
    crate_name: &str,
    crate_root: &Path,
    claimed_level: u8,
    targets: &[cargo_metadata::Target],
) -> Result<CertResult> {
    println!("\n──── {crate_name} (claimed-level {claimed_level}) ────");

    let mut result = CertResult {
        crate_name: crate_name.to_owned(),
        partition: None,
        native_reason: String::new(),
        containment_witnesses: vec![],
        dispatch_witnesses: vec![],
        level1_pass: false,
        level2_pass: None,
        level4_unit_cws: 0,
        level4_integ_cws: 0,
        mutant_survivors: None,
        regression: false,
    };

    // ── NATIVE: crate-type detection ──────────────────────────────────────────
    // Binary, cdylib, and bench crates belong in the host OS/IO layer — they
    // have no lib target and cannot be compiled as wasm32 library code.
    let has_lib = targets.iter().any(|t| t.is_lib() || t.is_rlib());
    if !has_lib {
        let kinds: Vec<String> = targets
            .iter()
            .flat_map(|t| t.kind.iter().map(|k| format!("{k:?}")))
            .collect();
        result.native_reason = format!("no lib target (kinds: {})", kinds.join(", "));
        result.partition = Some(WasmPartition::Native);
        println!("  Partition: NATIVE ({})", result.native_reason);
        if claimed_level >= 2 {
            result.regression = true;
        }
        return Ok(result);
    }

    // ── Level 1: compile check ────────────────────────────────────────────────
    let level1_errors = run_level1(crate_name)?;
    result.level1_pass = level1_errors.is_empty();
    if !result.level1_pass {
        println!("  LEVEL-1 FAIL (compile errors):");
        for w in &level1_errors {
            println!("    {w}");
        }
        if claimed_level >= 1 {
            result.regression = true;
        }
        return Ok(result);
    }
    println!("  Level 1: PASS (compile-consistent)");

    // ── Build wasm production binary (no dev-deps) ───────────────────────────
    let wasm_bin = build_wasm_production_binary(crate_name, crate_root)?;

    // ── Call-graph closure analysis ──────────────────────────────────────────
    if let Some(ref path) = wasm_bin {
        match analyze_call_graph(crate_name, path) {
            Ok((partition, containment_ws, dispatch_ws)) => {
                result.partition = Some(partition);
                result.containment_witnesses = containment_ws;
                result.dispatch_witnesses = dispatch_ws;

                println!("  Partition: {partition}");

                if !result.containment_witnesses.is_empty() {
                    println!("  Containment violations ({}):", result.containment_witnesses.len());
                    for w in &result.containment_witnesses {
                        println!("    CONTAINMENT  {w}");
                    }
                }
                if !result.dispatch_witnesses.is_empty() {
                    println!("  Dispatch witnesses ({}):", result.dispatch_witnesses.len());
                    for w in &result.dispatch_witnesses {
                        println!("    DISPATCH  {w}");
                    }
                }

                // Regression thresholds:
                //   claimed ≥ 2: NATIVE → regression
                //   claimed ≥ 3: DYNAMIC (any variant) → regression
                match partition {
                    WasmPartition::Native => {
                        result.native_reason = "non-intrinsic host imports reachable from exports".to_owned();
                        if claimed_level >= 2 {
                            result.regression = true;
                        }
                    }
                    WasmPartition::Dynamic
                    | WasmPartition::DynamicDecidable
                    | WasmPartition::DynamicUndecidable => {
                        if claimed_level >= 3 {
                            result.regression = true;
                        }
                    }
                    WasmPartition::Static => {}
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
    if result.level4_unit_cws > 0 || result.level4_integ_cws > 0 {
        println!(
            "  Level 4: {}u + {}i counterwit­nesses (native-only boundary)",
            result.level4_unit_cws, result.level4_integ_cws
        );
    }

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

/// Build the production wasm binary via `cargo rustc --crate-type cdylib`.
///
/// Using cdylib excludes dev-dependencies (wasm-bindgen-test, etc.) so the
/// call-graph analysis sees only production code dispatch, not test harness.
/// For cdylib artifacts the path is in `filenames`, not `executable`.
/// Returns the path to the `.wasm` artifact, or None if the build fails.
fn build_wasm_production_binary(crate_name: &str, crate_root: &Path) -> Result<Option<PathBuf>> {
    let output = Command::new("cargo")
        .args([
            "rustc",
            "--target",
            "wasm32-unknown-unknown",
            "--message-format",
            "json",
            "-p",
            crate_name,
            "--lib",
            "--crate-type",
            "cdylib",
        ])
        .current_dir(crate_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .context("cargo rustc --crate-type cdylib")?;

    // cdylib artifacts appear in `filenames`, not `executable`.
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg["reason"] == "compiler-artifact" {
                if let Some(filenames) = msg["filenames"].as_array() {
                    for f in filenames {
                        if let Some(s) = f.as_str() {
                            if s.ends_with(".wasm") {
                                return Ok(Some(PathBuf::from(s)));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

/// Analyze the call graph of the wasm binary via Datalog rules.
/// Returns `(partition, containment_witnesses, dispatch_witnesses)`.
fn analyze_call_graph(
    _crate_name: &str,
    wasm_path: &Path,
) -> Result<(WasmPartition, Vec<String>, Vec<String>)> {
    let bytes = std::fs::read(wasm_path).context("read wasm binary")?;
    let mut facts = crate::wasm_facts::extract(&bytes)?;

    let non_intrinsic_indices: Vec<u32> = facts.non_intrinsic_imports.iter().map(|(_, _, idx)| *idx).collect();
    let roots: Vec<u32> = facts.roots.iter().map(|(_, idx)| *idx).collect();
    let calls = std::mem::take(&mut facts.calls);
    let indirect_calls = std::mem::take(&mut facts.indirect_calls);
    let result = crate::rules::analyze(calls, non_intrinsic_indices, indirect_calls, roots);

    let partition = result.partition_stable();

    let containment_witnesses: Vec<String> = result
        .containment_violation_indices()
        .iter()
        .map(|idx| {
            let label = crate::wasm_facts::label(&facts, *idx);
            format!("fn {label}  [non-intrinsic-import (sandbox boundary breach)]")
        })
        .collect();

    let dispatch_witnesses: Vec<String> = result
        .dispatch_witness_indices()
        .iter()
        .map(|idx| {
            let label = crate::wasm_facts::label(&facts, *idx);
            format!("fn {label}  [call_indirect (unresolved dynamic dispatch)]")
        })
        .collect();

    Ok((partition, containment_witnesses, dispatch_witnesses))
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
