//! Surface auditor: wasm32-accessible module classification + Level-4 boundary map.
//!
//! Scans `src/lib.rs` to determine which `pub mod` declarations are wasm32-
//! accessible (not immediately preceded by `#[cfg(not(target_arch = "wasm32"))]`).
//! Then greps those modules for runtime-trap patterns and collects cfg-gated
//! tests as Level-4 compactness counterwit­nesses.
//!
//! Always exits 0 — purely informational.

use anyhow::Result;
use std::path::Path;

/// Audit result for a single crate.
#[derive(Debug, Default)]
pub struct AuditResult {
    pub crate_name: String,
    pub accessible: Vec<String>,
    pub native_only: Vec<String>,
    /// (file, line, pattern) for runtime-trap candidates in accessible modules.
    pub trap_candidates: Vec<(String, usize, String)>,
    /// Level-4 unit counterwit­nesses: (file, line, fn_name).
    pub unit_counterwits: Vec<(String, usize, String)>,
    /// Level-4 integration counterwit­nesses: file paths.
    pub integ_counterwits: Vec<String>,
}

/// Public entry point for `certify.rs` — returns the audit result for a crate.
pub fn audit_crate_pub(crate_name: &str, crate_root: &Path) -> Result<AuditResult> {
    audit_crate(crate_name, crate_root)
}

pub fn run(crate_name: Option<&str>) -> Result<()> {
    let meta = crate::status::workspace_meta()?;
    for pkg in &meta.packages {
        if let Some(name) = crate_name {
            if pkg.name != name {
                continue;
            }
        }
        // Only workspace members
        if !meta
            .workspace_members
            .contains(&pkg.id)
        {
            continue;
        }
        let result = audit_crate(&pkg.name, pkg.manifest_path.parent().unwrap().as_std_path())?;
        print_audit(&result);
    }
    Ok(())
}

fn audit_crate(crate_name: &str, crate_root: &Path) -> Result<AuditResult> {
    let mut result = AuditResult {
        crate_name: crate_name.to_owned(),
        ..Default::default()
    };

    // ── Module classification ─────────────────────────────────────────────────
    let lib_rs = crate_root.join("src/lib.rs");
    if !lib_rs.exists() {
        return Ok(result);
    }
    let src = std::fs::read_to_string(&lib_rs)?;
    classify_modules(&src, &mut result);

    let src_dir = crate_root.join("src");

    // ── Runtime-trap candidates in wasm32-accessible modules ──────────────────
    let trap_patterns = [
        "std::time::Instant",
        "std::thread::",
        "std::fs::",
        "std::net::",
        "std::process::",
    ];
    for mod_name in &result.accessible {
        let mod_paths = module_paths(&src_dir, mod_name);
        for path in &mod_paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                for (line_no, line) in content.lines().enumerate() {
                    for pat in &trap_patterns {
                        if line.contains(pat) {
                            result.trap_candidates.push((
                                path.display().to_string(),
                                line_no + 1,
                                pat.to_string(),
                            ));
                        }
                    }
                }
            }
        }
    }

    // ── Level-4: unit counterwit­nesses (cfg-gated test fns in src/) ──────────
    collect_unit_counterwits(&src_dir, &result.accessible, &mut result.unit_counterwits)?;

    // ── Level-4: integration test counterwit­nesses (tests/ top-level cfg) ────
    let tests_dir = crate_root.join("tests");
    if tests_dir.exists() {
        collect_integ_counterwits(&tests_dir, &mut result.integ_counterwits)?;
    }

    Ok(result)
}

/// Scan `src/lib.rs` and classify `pub mod` declarations.
fn classify_modules(src: &str, result: &mut AuditResult) {
    let mut prev_was_cfg_gate = false;
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed == "#[cfg(not(target_arch = \"wasm32\"))]" {
            prev_was_cfg_gate = true;
            continue;
        }
        if let Some(mod_name) = trimmed
            .strip_prefix("pub mod ")
            .and_then(|s| s.strip_suffix(';'))
        {
            if prev_was_cfg_gate {
                result.native_only.push(mod_name.to_owned());
            } else {
                result.accessible.push(mod_name.to_owned());
            }
        }
        // Reset gate tracker on any non-blank, non-cfg-attr, non-comment line
        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            prev_was_cfg_gate = false;
        }
    }
}

/// Return file paths that could contain module `name` (file or directory).
fn module_paths(src_dir: &Path, name: &str) -> Vec<std::path::PathBuf> {
    let mut paths = vec![];
    let file = src_dir.join(format!("{name}.rs"));
    if file.exists() {
        paths.push(file);
    }
    let dir = src_dir.join(name);
    if dir.is_dir() {
        // Recursively collect all .rs files in the module directory.
        collect_rs_files(&dir, &mut paths);
    }
    paths
}

fn collect_rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
}

/// Collect unit-test counterwit­nesses: `#[cfg(not(target_arch = "wasm32"))]`
/// immediately followed by `#[test]` or `fn ...` inside accessible modules.
fn collect_unit_counterwits(
    src_dir: &Path,
    accessible: &[String],
    out: &mut Vec<(String, usize, String)>,
) -> Result<()> {
    for mod_name in accessible {
        let paths = module_paths(src_dir, mod_name);
        for path in &paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                let lines: Vec<&str> = content.lines().collect();
                for (i, line) in lines.iter().enumerate() {
                    if line.trim() == "#[cfg(not(target_arch = \"wasm32\"))]" {
                        if let Some(next) = lines.get(i + 1) {
                            let nt = next.trim();
                            if nt == "#[test]" || nt.starts_with("fn ") || nt.starts_with("pub fn ") {
                                // Try to extract the fn name from this or next line
                                let fn_name = extract_fn_name(nt)
                                    .or_else(|| lines.get(i + 2).and_then(|l| extract_fn_name(l.trim())))
                                    .unwrap_or_else(|| "<unknown>".to_owned());
                                out.push((path.display().to_string(), i + 1, fn_name));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn extract_fn_name(line: &str) -> Option<String> {
    // Match: `fn foo(` or `pub fn foo(`
    let rest = line.strip_prefix("pub fn ").or_else(|| line.strip_prefix("fn "))?;
    let name = rest.split('(').next()?;
    Some(name.trim().to_owned())
}

/// Collect integration-test counterwit­nesses: `.rs` files in `tests/` that
/// begin with `#![cfg(not(target_arch = "wasm32"))]`.
fn collect_integ_counterwits(tests_dir: &Path, out: &mut Vec<String>) -> Result<()> {
    if let Ok(entries) = std::fs::read_dir(tests_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines().take(5) {
                    if line.trim().contains("cfg(not(target_arch = \"wasm32\"))")
                        && line.trim().starts_with("#!")
                    {
                        out.push(path.display().to_string());
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

fn print_audit(r: &AuditResult) {
    println!("\n=== {} ===", r.crate_name);
    if r.accessible.is_empty() && r.native_only.is_empty() {
        println!("  (no lib.rs found or no pub mod declarations)");
        return;
    }
    println!("  WASM32-ACCESSIBLE: {}", r.accessible.join(", ").or_empty());
    println!("  NATIVE-ONLY:       {}", r.native_only.join(", ").or_empty());
    if r.trap_candidates.is_empty() {
        println!("  RUNTIME-TRAP CANDIDATES: (none)");
    } else {
        println!("  RUNTIME-TRAP CANDIDATES:");
        for (f, l, p) in &r.trap_candidates {
            println!("    {f}:{l}  {p}");
        }
    }
    println!(
        "  LEVEL-4 COUNTERWIT­NESSES: {}u + {}i",
        r.unit_counterwits.len(),
        r.integ_counterwits.len()
    );
    for (f, l, name) in &r.unit_counterwits {
        println!("    {f}:{l}  fn {name}  [unit]");
    }
    for f in &r.integ_counterwits {
        println!("    {f}  [integration]");
    }
}

trait OrEmpty {
    fn or_empty(&self) -> &str;
}
impl OrEmpty for String {
    fn or_empty(&self) -> &str {
        if self.is_empty() { "(none)" } else { self }
    }
}
