//! Minimal GitHub API helpers — used only in CI context when GITHUB_TOKEN is set.

use anyhow::{Context, Result};
use serde_json::Value;
use std::process::Command;

/// Returns true when running inside GitHub Actions.
pub fn in_ci() -> bool {
    std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
}

/// `OWNER/REPO` extracted from `GITHUB_REPOSITORY`.
pub fn repo() -> Option<String> {
    std::env::var("GITHUB_REPOSITORY").ok()
}

pub fn sha() -> Option<String> {
    std::env::var("GITHUB_SHA").ok()
}

pub fn pr_number() -> Option<u64> {
    // GITHUB_REF looks like refs/pull/123/merge for PRs.
    let refs = std::env::var("GITHUB_REF").ok()?;
    let mut parts = refs.split('/');
    // refs / pull / <number> / merge
    if parts.next()? == "refs" && parts.next()? == "pull" {
        parts.next()?.parse().ok()
    } else {
        None
    }
}

/// List files changed in a pull request.
///
/// Returns an empty Vec when not in a PR or when the API call fails.
pub fn pr_files(pr: u64) -> Vec<String> {
    let Some(repo) = repo() else { return vec![] };
    let out = Command::new("gh")
        .args([
            "api",
            &format!("repos/{repo}/pulls/{pr}/files"),
            "--jq",
            ".[].filename",
        ])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(str::to_owned)
            .collect(),
        _ => vec![],
    }
}

/// Post a GitHub Check Annotation for a crate certification result.
///
/// `conclusion`: `"success"`, `"failure"`, or `"neutral"`.
/// `annotations`: list of `(path, line, message)` triples.
pub fn post_check(
    crate_name: &str,
    conclusion: &str,
    annotations: &[(String, u32, String)],
) -> Result<()> {
    let Some(repo) = repo() else { return Ok(()) };
    let Some(sha) = sha() else { return Ok(()) };
    if std::env::var("GITHUB_TOKEN").is_err() {
        return Ok(());
    }

    let ann_json: Vec<Value> = annotations
        .iter()
        .map(|(path, line, msg)| {
            serde_json::json!({
                "path": path,
                "start_line": line,
                "end_line": line,
                "annotation_level": if conclusion == "failure" { "failure" } else { "notice" },
                "message": msg,
            })
        })
        .collect();

    let body = serde_json::json!({
        "name": format!("wasm-certify/{crate_name}"),
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {
            "title": format!("wasm-certify: {crate_name}"),
            "summary": format!("Certification conclusion: {conclusion}"),
            "annotations": ann_json,
        }
    });

    let body_str = body.to_string();
    let status = Command::new("gh")
        .args([
            "api",
            &format!("repos/{repo}/check-runs"),
            "--method",
            "POST",
            "--input",
            "-",
        ])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(body_str.as_bytes())?;
            }
            child.wait()
        })
        .context("gh api check-runs")?;

    if !status.success() {
        eprintln!("warning: gh api check-runs failed for {crate_name}");
    }
    Ok(())
}
