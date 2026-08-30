//! `vindex update` — the explicit self-updater.
//!
//! The doctrine, stated because it matters for a verification tool:
//! vindex NEVER checks for updates on its own. No verb phones home,
//! no background check runs, nothing changes unless the user types
//! `update`. A tool whose job is proving artifacts unchanged must be
//! boring about changing itself.
//!
//! `update --check` asks GitHub for the latest `vindex-v*` release
//! and reports. `update` additionally downloads the platform tarball
//! when one exists, sanity-runs the new binary, and swaps it into
//! place — or prints the exact `cargo install` command when no
//! prebuilt asset covers this platform. Network and archive work go
//! through the system's own `curl` and `tar`, keeping the crate's
//! dependency tree exactly as small as it was.

use std::path::Path;
use std::process::Command;

const REPO: &str = "chrishayuk/larql";
const TAG_PREFIX: &str = "vindex-v";

fn semver(v: &str) -> Option<(u64, u64, u64)> {
    let mut it = v.trim().split('.');
    let a = it.next()?.parse().ok()?;
    let b = it.next()?.parse().ok()?;
    let c = it.next()?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some((a, b, c))
}

/// The newest version among `tags` (each like `vindex-v0.2.1`) that is
/// strictly newer than `current`. Pure, so it is testable without a
/// network.
pub fn newer_version(current: &str, tags: &[String]) -> Option<String> {
    let cur = semver(current)?;
    tags.iter()
        .filter_map(|t| t.strip_prefix(TAG_PREFIX))
        .filter_map(|v| semver(v).map(|s| (s, v.to_string())))
        .filter(|(s, _)| *s > cur)
        .max_by_key(|(s, _)| *s)
        .map(|(_, v)| v)
}

fn curl(args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("curl")
        .args(["-fsSL", "--proto", "=https"])
        .args(args)
        .output()
        .map_err(|e| format!("curl: {e} — is curl installed?"))?;
    if !out.status.success() {
        return Err(format!(
            "curl failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

fn release_tags() -> Result<Vec<String>, String> {
    let body = curl(&[&format!(
        "https://api.github.com/repos/{REPO}/releases?per_page=30"
    )])?;
    let json: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| format!("parse releases: {e}"))?;
    Ok(json
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|r| r["tag_name"].as_str())
        .filter(|t| t.starts_with(TAG_PREFIX))
        .map(String::from)
        .collect())
}

/// The release-asset suffix for this platform, when a prebuilt one is
/// published.
fn platform_asset(version: &str) -> Option<String> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some(format!("vindex-{version}-macos-arm64.tar.gz"))
    } else {
        None
    }
}

fn install_hint() -> String {
    format!("cargo install --git https://github.com/{REPO} vindex-cli --force")
}

fn swap_in(new_binary: &Path) -> Result<(), String> {
    let current = std::env::current_exe().map_err(|e| format!("current exe: {e}"))?;
    let backup = current.with_extension("old");
    let _ = std::fs::remove_file(&backup);
    std::fs::rename(&current, &backup).map_err(|e| {
        format!(
            "cannot replace {} ({e}) — rerun with permission to write it, or: {}",
            current.display(),
            install_hint()
        )
    })?;
    match std::fs::copy(new_binary, &current) {
        Ok(_) => {
            let _ = std::fs::remove_file(&backup);
            Ok(())
        }
        Err(e) => {
            // Put the old binary back rather than leaving nothing.
            let _ = std::fs::rename(&backup, &current);
            Err(format!("install new binary: {e}"))
        }
    }
}

/// Run the update. Returns a human line describing what happened;
/// `check_only` never touches the filesystem.
pub fn run(check_only: bool) -> Result<String, String> {
    let current = env!("CARGO_PKG_VERSION");
    let tags = release_tags()?;
    let Some(latest) = newer_version(current, &tags) else {
        return Ok(format!(
            "vindex {current} is current — no newer release than {TAG_PREFIX}{current}"
        ));
    };
    if check_only {
        return Ok(format!(
            "vindex {latest} is available (running {current}) — `vindex update` installs it"
        ));
    }
    let Some(asset) = platform_asset(&latest) else {
        return Ok(format!(
            "vindex {latest} is available (running {current}), but no prebuilt binary covers this platform — install with:\n  {}",
            install_hint()
        ));
    };
    let url = format!("https://github.com/{REPO}/releases/download/{TAG_PREFIX}{latest}/{asset}");
    let dir = std::env::temp_dir().join(format!("vindex-update-{latest}"));
    std::fs::create_dir_all(&dir).map_err(|e| format!("temp dir: {e}"))?;
    let tarball = dir.join(&asset);
    std::fs::write(&tarball, curl(&[&url])?).map_err(|e| format!("write download: {e}"))?;
    let status = Command::new("tar")
        .args(["xzf"])
        .arg(&tarball)
        .args(["-C"])
        .arg(&dir)
        .status()
        .map_err(|e| format!("tar: {e}"))?;
    if !status.success() {
        return Err("tar extract failed".to_string());
    }
    let new_binary = dir.join("vindex");
    // Sanity: the downloaded binary must run and say the version we asked for.
    let out = Command::new(&new_binary)
        .arg("--version")
        .output()
        .map_err(|e| format!("run downloaded binary: {e}"))?;
    let said = String::from_utf8_lossy(&out.stdout);
    if !said.contains(&latest) {
        return Err(format!(
            "downloaded binary reports `{}` — expected {latest}; not installing",
            said.trim()
        ));
    }
    swap_in(&new_binary)?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(format!("vindex {current} → {latest} — installed"))
}

#[cfg(test)]
mod tests {
    use super::newer_version;

    fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn picks_the_newest_strictly_newer_release() {
        let t = tags(&["vindex-v0.2.0", "vindex-v0.2.1", "vindex-v0.3.0", "v1.9.9"]);
        assert_eq!(newer_version("0.2.0", &t).as_deref(), Some("0.3.0"));
        assert_eq!(newer_version("0.3.0", &t), None);
    }

    #[test]
    fn ignores_tags_that_are_not_vindex_releases_or_not_semver() {
        let t = tags(&["v9.9.9", "vindex-vnext", "vindex-v0.2"]);
        assert_eq!(newer_version("0.1.0", &t), None);
    }
}
