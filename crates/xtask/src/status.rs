//! Workspace certification status table.
//!
//! Reads `[package.metadata.wasm-cert]` from each crate's `Cargo.toml` via
//! `cargo_metadata` and renders a Markdown table.  Writes to
//! `$GITHUB_STEP_SUMMARY` when set (GitHub Actions job summary).

use anyhow::Result;
use cargo_metadata::{Metadata, MetadataCommand};

/// Certification manifest read from `[package.metadata.wasm-cert]`.
#[derive(Debug, Default)]
pub struct WasmCert {
    pub claimed_level: u8,
    pub diagonalization_gate: String,
    pub notes: String,
}

impl WasmCert {
    fn from_metadata(meta: &serde_json::Map<String, serde_json::Value>) -> Option<Self> {
        let cert = meta.get("wasm-cert")?;
        Some(WasmCert {
            claimed_level: cert
                .get("claimed-level")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8,
            diagonalization_gate: cert
                .get("diagonalization-gate")
                .and_then(|v| v.as_str())
                .unwrap_or("uncertified")
                .to_owned(),
            notes: cert
                .get("notes")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
        })
    }
}

pub fn workspace_meta() -> Result<Metadata> {
    Ok(MetadataCommand::new().exec()?)
}

pub fn run(json: bool) -> Result<()> {
    let meta = workspace_meta()?;
    let mut rows: Vec<(String, WasmCert)> = vec![];

    for pkg in &meta.packages {
        if !meta.workspace_members.contains(&pkg.id) {
            continue;
        }
        let cert = pkg
            .metadata
            .as_object()
            .and_then(|m| WasmCert::from_metadata(m))
            .unwrap_or_default();
        rows.push((pkg.name.clone(), cert));
    }

    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let output = if json {
        render_json(&rows)
    } else {
        render_markdown(&rows)
    };

    println!("{output}");

    // Also write to GitHub Actions job summary if available.
    if let Ok(summary_path) = std::env::var("GITHUB_STEP_SUMMARY") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(summary_path)?;
        writeln!(f, "{output}")?;
    }

    Ok(())
}

fn render_markdown(rows: &[(String, WasmCert)]) -> String {
    let mut out = String::new();
    out.push_str("## wasm32 Certification Status\n\n");
    out.push_str("| Crate | Claimed level | Diag gate | Notes |\n");
    out.push_str("|-------|--------------|-----------|-------|\n");
    for (name, cert) in rows {
        let level_str = if cert.claimed_level == 0 {
            "0 ⚠ uncertified".to_owned()
        } else {
            cert.claimed_level.to_string()
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            name,
            level_str,
            cert.diagonalization_gate,
            cert.notes,
        ));
    }
    out
}

fn render_json(rows: &[(String, WasmCert)]) -> String {
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .map(|(name, cert)| {
            serde_json::json!({
                "crate": name,
                "claimed_level": cert.claimed_level,
                "diagonalization_gate": cert.diagonalization_gate,
                "notes": cert.notes,
            })
        })
        .collect();
    serde_json::to_string_pretty(&arr).unwrap_or_default()
}
