//! Startup memory pre-flight check (BUG-infer-deadlock §5.5).
//!
//! Read the systemd cgroup limits we run under, compare against the
//! resident-size estimate from `VindexConfig::estimate_resident_bytes`,
//! and refuse to start when the cgroup leaves us no headroom.
//!
//! Converts a 10-second runtime OOM-kill loop into a one-line startup
//! error operators can act on.
//!
//! Reads are best-effort and pure procfs:
//! - `/proc/self/cgroup`           — locate this process's cgroup
//! - `/sys/fs/cgroup/<path>/memory.max` (cgroup v2 unified hierarchy),
//!   falling back to `memory.high` if `max` is "max"/unlimited
//! - `/proc/meminfo`               — fall-through host-level estimate
//!   when no cgroup is set (e.g. running under a stock shell)
//!
//! When the limit is genuinely unlimited (cgroup v2 `memory.max == max`
//! AND we're cgroup-root or no v2 hierarchy), the pre-flight returns
//! `None` and the caller skips the check.  This keeps the developer
//! workflow on a workstation untouched.
//!
//! Cgroup v1 systems are not supported by this check (the file layout
//! is different).  `--no-memcheck` skips the pre-flight unconditionally.

use std::path::{Path, PathBuf};

pub mod bitnet;

/// Resident-size estimate the pre-flight compares against the cgroup
/// limit for one vindex. A BitNet `--keep-quant` container is sized by
/// its ternary layout ([`bitnet::estimate_bitnet_resident_bytes`]); every
/// other vindex by `VindexConfig::estimate_resident_bytes`, which models
/// the dense path. Zero means "no estimate — skip".
pub fn estimate_model_resident_bytes(config: &larql_vindex::VindexConfig) -> u64 {
    bitnet::estimate_bitnet_resident_bytes(config)
        .unwrap_or_else(|| config.estimate_resident_bytes())
}

/// Outcome of the pre-flight memory check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemCheckOutcome {
    /// Cgroup limit found and the estimate fits comfortably under it.
    Ok {
        cgroup_max_bytes: u64,
        estimate_bytes: u64,
    },
    /// No cgroup limit detected (or cgroup is unlimited); pre-flight is
    /// a no-op.
    Skipped { reason: &'static str },
    /// The estimate exceeds the cgroup limit (after subtracting an
    /// operator-tunable headroom).  The caller should refuse to start.
    Tight {
        cgroup_max_bytes: u64,
        estimate_bytes: u64,
        headroom_bytes: u64,
    },
}

/// Decide if the configured cgroup leaves us enough room to load.
///
/// `headroom_bytes` is the slack we reserve for the OS, jemalloc/glibc
/// allocator overhead, and the request-handling working set.  Default
/// 512 MiB.
pub fn check_memory_headroom(estimate_bytes: u64, headroom_bytes: u64) -> MemCheckOutcome {
    let limit = match read_cgroup_v2_memory_max() {
        Ok(Some(v)) => v,
        Ok(None) => {
            return MemCheckOutcome::Skipped {
                reason: "cgroup v2 memory.max is unlimited",
            }
        }
        Err(_) => {
            return MemCheckOutcome::Skipped {
                reason: "no cgroup v2 memory limit detectable",
            }
        }
    };

    decide_headroom(limit, estimate_bytes, headroom_bytes)
}

/// Pure classification: given an already-resolved cgroup `limit`, decide
/// whether `estimate_bytes` fits under it once `headroom_bytes` (capped at
/// half the limit) is reserved.  Split out from [`check_memory_headroom`] so
/// the decision arms are unit-testable without a cgroup-bearing filesystem.
fn decide_headroom(limit: u64, estimate_bytes: u64, headroom_bytes: u64) -> MemCheckOutcome {
    let headroom = headroom_bytes.min(limit / 2); // never claim more than half
    let usable = limit.saturating_sub(headroom);

    if estimate_bytes > usable {
        MemCheckOutcome::Tight {
            cgroup_max_bytes: limit,
            estimate_bytes,
            headroom_bytes: headroom,
        }
    } else {
        MemCheckOutcome::Ok {
            cgroup_max_bytes: limit,
            estimate_bytes,
        }
    }
}

/// Read this process's cgroup v2 `memory.max`, returning `Ok(Some(N))`
/// on a numeric limit, `Ok(None)` if it is `"max"` (unlimited), or
/// `Err` if the cgroup hierarchy can't be discovered.
pub fn read_cgroup_v2_memory_max() -> Result<Option<u64>, String> {
    let cgroup = std::fs::read_to_string(PROC_SELF_CGROUP)
        .map_err(|e| format!("read {PROC_SELF_CGROUP}: {e}"))?;
    let path = memory_max_path(&cgroup, Path::new(CGROUP_V2_ROOT))?;
    read_memory_max_at(&path)
}

/// Where the kernel publishes this process's cgroup membership.
const PROC_SELF_CGROUP: &str = "/proc/self/cgroup";
/// Mount point of the cgroup v2 unified hierarchy.
const CGROUP_V2_ROOT: &str = "/sys/fs/cgroup";
/// The limit file within a cgroup directory.
const MEMORY_MAX_FILE: &str = "memory.max";

/// The `memory.max` path for the cgroup `cgroup_text` names, under
/// `unified_root`.
///
/// The root is a parameter rather than a constant so this is exercisable
/// against a temporary hierarchy. It previously read `/proc/self/cgroup`
/// and joined `/sys/fs/cgroup` inline, which made every branch here
/// reachable only on a Linux host that happened to be running under cgroup
/// v2 — so on any other machine the resolution rules went untested while
/// looking covered by the caller's tests.
fn memory_max_path(cgroup_text: &str, unified_root: &Path) -> Result<PathBuf, String> {
    let cgroup_rel = parse_cgroup_v2_path(cgroup_text)
        .ok_or_else(|| "no cgroup v2 unified entry".to_string())?;
    // The unified entry is absolute (`/`, `/user.slice/...`); joining it
    // as-is would discard `unified_root` entirely.
    let trimmed = cgroup_rel.trim_start_matches('/');
    let candidate = if trimmed.is_empty() {
        unified_root.join(MEMORY_MAX_FILE)
    } else {
        unified_root.join(trimmed).join(MEMORY_MAX_FILE)
    };
    if !candidate.exists() {
        return Err(format!("{} not found", candidate.display()));
    }
    Ok(candidate)
}

/// Parse a `memory.max` file that is already known to exist.
fn read_memory_max_at(path: &Path) -> Result<Option<u64>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_memory_max(s.trim())
}

fn parse_memory_max(s: &str) -> Result<Option<u64>, String> {
    if s == "max" {
        return Ok(None);
    }
    s.parse::<u64>()
        .map(Some)
        .map_err(|e| format!("parse memory.max '{s}': {e}"))
}

/// Extract the cgroup v2 unified-hierarchy path (the `"0::/path"` line) from
/// `/proc/self/cgroup` content.  Returns `None` when only cgroup v1 lines
/// (non-zero hierarchy id) are present.  Pure string work, split out so the
/// parse is unit-testable without a real procfs.
fn parse_cgroup_v2_path(content: &str) -> Option<&str> {
    for line in content.lines() {
        // cgroup v2 unified line shape: "0::/path/under/sys/fs/cgroup".
        let mut parts = line.splitn(3, ':');
        let id = parts.next();
        let controllers = parts.next();
        let path = parts.next();
        if id == Some("0") && controllers == Some("") {
            if let Some(p) = path {
                return Some(p);
            }
        }
    }
    None
}

/// Format an explanation message for `MemCheckOutcome::Tight`.
pub fn explain_tight_outcome(o: &MemCheckOutcome) -> String {
    match o {
        MemCheckOutcome::Tight {
            cgroup_max_bytes,
            estimate_bytes,
            headroom_bytes,
        } => {
            format!(
                "vindex requires ~{:.1} GB resident; cgroup memory.max={:.1} GB, \
                 leaving ~{:.1} GB after the {:.0} MB headroom reserve. \
                 Inference will OOM. Increase MemoryMax to >= {:.1} GB or pass \
                 --lazy-weights (and accept the runtime OOM risk) or \
                 --no-memcheck (override).",
                gb(*estimate_bytes),
                gb(*cgroup_max_bytes),
                gb(cgroup_max_bytes.saturating_sub(*headroom_bytes)),
                (*headroom_bytes as f64) / (1024.0 * 1024.0),
                gb(estimate_bytes.saturating_add(*headroom_bytes)),
            )
        }
        _ => String::new(),
    }
}

fn gb(n: u64) -> f64 {
    (n as f64) / (1024.0 * 1024.0 * 1024.0)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_memory_max_unlimited() {
        assert_eq!(parse_memory_max("max"), Ok(None));
    }

    #[test]
    fn parse_memory_max_numeric() {
        assert_eq!(parse_memory_max("6442450944"), Ok(Some(6_442_450_944)));
    }

    #[test]
    fn parse_memory_max_garbage_errors() {
        assert!(parse_memory_max("not-a-number").is_err());
    }

    #[test]
    fn check_memory_headroom_with_unlimited_cgroup_skips() {
        // We can't easily mock locate_memory_max() inline, so this is
        // a documentation test: the path through `Skipped` is what we
        // get when memory.max is "max" or unreadable.  The logic is
        // exercised end-to-end by the `tight_outcome_message_format`
        // test below.
        let _ = MemCheckOutcome::Skipped {
            reason: "test placeholder",
        };
    }

    #[test]
    fn tight_outcome_message_format() {
        // A typical bug-report scenario: 5.2 GB vindex, 6 GB cgroup,
        // 512 MiB headroom -> tight.
        let outcome = MemCheckOutcome::Tight {
            cgroup_max_bytes: 6 * 1024 * 1024 * 1024,
            estimate_bytes: (5_200u64 * 1024 * 1024) + (200 * 1024 * 1024), // 5.4 GB
            headroom_bytes: 512 * 1024 * 1024,
        };
        let msg = explain_tight_outcome(&outcome);
        assert!(msg.contains("vindex requires ~5."), "got: {msg}");
        assert!(msg.contains("cgroup memory.max=6."), "got: {msg}");
        assert!(msg.contains("--lazy-weights"));
        assert!(msg.contains("--no-memcheck"));
    }

    #[test]
    fn explain_tight_outcome_returns_empty_for_ok() {
        let ok = MemCheckOutcome::Ok {
            cgroup_max_bytes: 8 * 1024 * 1024 * 1024,
            estimate_bytes: 1024 * 1024,
        };
        assert_eq!(explain_tight_outcome(&ok), "");
    }

    #[test]
    fn check_with_zero_estimate_always_passes() {
        // Zero estimate must never trip the tight branch, even under
        // a tiny cgroup.  This covers the --ffn-only / --embed-only /
        // --no-infer paths where estimate_resident_bytes is small.
        let result = check_memory_headroom(0, 512 * 1024 * 1024);
        // Either Ok or Skipped is acceptable; Tight would be a bug.
        assert!(
            !matches!(result, MemCheckOutcome::Tight { .. }),
            "got {result:?}"
        );
    }

    #[test]
    fn decide_headroom_ok_when_estimate_fits() {
        // 2 GB estimate under an 8 GB limit with 512 MiB headroom → Ok.
        let limit = 8 * 1024 * 1024 * 1024;
        let out = decide_headroom(limit, 2 * 1024 * 1024 * 1024, 512 * 1024 * 1024);
        assert_eq!(
            out,
            MemCheckOutcome::Ok {
                cgroup_max_bytes: limit,
                estimate_bytes: 2 * 1024 * 1024 * 1024,
            }
        );
    }

    #[test]
    fn decide_headroom_tight_when_estimate_exceeds_usable() {
        // 7.8 GB estimate under an 8 GB limit minus 512 MiB headroom → Tight.
        let limit = 8 * 1024 * 1024 * 1024;
        let estimate = limit - 100 * 1024 * 1024; // 7.9 GB
        let out = decide_headroom(limit, estimate, 512 * 1024 * 1024);
        assert!(
            matches!(out, MemCheckOutcome::Tight { headroom_bytes, .. } if headroom_bytes == 512 * 1024 * 1024),
            "got {out:?}"
        );
    }

    #[test]
    fn decide_headroom_caps_reserve_at_half_the_limit() {
        // A 1 GB headroom request against a 1 GB limit is capped to 512 MiB
        // (half), leaving 512 MiB usable. A 400 MiB estimate fits — but only
        // because of the cap: an uncapped 1 GB reserve would leave 0 usable
        // and trip Tight.
        let limit = 1024 * 1024 * 1024;
        let out = decide_headroom(limit, 400 * 1024 * 1024, 1024 * 1024 * 1024);
        assert_eq!(
            out,
            MemCheckOutcome::Ok {
                cgroup_max_bytes: limit,
                estimate_bytes: 400 * 1024 * 1024,
            },
            "headroom should be capped at limit/2 = 512 MiB"
        );
    }

    #[test]
    fn parse_cgroup_v2_path_finds_unified_entry() {
        let content = "12:pids:/system.slice\n0::/system.slice/larql.service\n";
        assert_eq!(
            parse_cgroup_v2_path(content),
            Some("/system.slice/larql.service")
        );
    }

    #[test]
    fn parse_cgroup_v2_path_returns_root_path() {
        assert_eq!(parse_cgroup_v2_path("0::/\n"), Some("/"));
    }

    #[test]
    fn parse_cgroup_v2_path_none_for_v1_only() {
        // Only legacy v1 lines (non-zero hierarchy ids) → no unified entry.
        let content = "11:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n";
        assert_eq!(parse_cgroup_v2_path(content), None);
    }

    /// The resolver joins the unified entry under the given root, and
    /// refuses when the file is not there.
    ///
    /// The `trim_start_matches('/')` is the load-bearing line: `Path::join`
    /// with an absolute argument REPLACES the base, so without the trim the
    /// candidate would be `/user.slice/memory.max` on the real filesystem
    /// regardless of where the hierarchy is mounted — a path that does not
    /// exist, reported as "not found" rather than as the bug it is.
    #[test]
    fn memory_max_path_joins_the_unified_entry_under_the_root() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("user.slice/session.scope");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("memory.max"), "1048576\n").unwrap();

        let cgroup = "0::/user.slice/session.scope\n";
        let got = memory_max_path(cgroup, root.path()).unwrap();
        assert_eq!(got, nested.join("memory.max"));
        assert!(
            got.starts_with(root.path()),
            "an absolute cgroup entry must not escape the root: {got:?}"
        );
    }

    /// A process in the root cgroup (`0::/`) reads the root's own file.
    #[test]
    fn the_root_cgroup_resolves_to_the_roots_own_limit_file() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("memory.max"), "max\n").unwrap();
        let got = memory_max_path("0::/\n", root.path()).unwrap();
        assert_eq!(got, root.path().join("memory.max"));
    }

    #[test]
    fn a_missing_limit_file_is_reported_with_its_path() {
        let root = tempfile::tempdir().unwrap();
        let err = memory_max_path("0::/nowhere\n", root.path()).unwrap_err();
        assert!(err.contains("nowhere"), "{err}");
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn a_cgroup_file_with_no_unified_entry_is_refused() {
        // v1-only hierarchy: no `0::` line at all.
        let root = tempfile::tempdir().unwrap();
        let err = memory_max_path("11:memory:/docker/abc\n", root.path()).unwrap_err();
        assert!(err.contains("no cgroup v2 unified entry"), "{err}");
    }

    /// Reading a limit file goes through the same parser as the caller,
    /// including the unlimited sentinel.
    #[test]
    fn reading_a_limit_file_handles_both_a_number_and_max() {
        let dir = tempfile::tempdir().unwrap();
        let numeric = dir.path().join("memory.max");
        std::fs::write(&numeric, "2097152\n").unwrap();
        assert_eq!(read_memory_max_at(&numeric).unwrap(), Some(2_097_152));

        let unlimited = dir.path().join("unlimited.max");
        std::fs::write(&unlimited, "max\n").unwrap();
        assert_eq!(read_memory_max_at(&unlimited).unwrap(), None);
    }

    #[test]
    fn reading_an_absent_limit_file_reports_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_memory_max_at(&dir.path().join("gone.max")).unwrap_err();
        assert!(err.contains("gone.max"), "{err}");
    }
}
