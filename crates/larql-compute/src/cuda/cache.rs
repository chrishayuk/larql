//! Kernel cache directory layout.
//!
//! Phase-1 (cuda-f32-baseline): create the directory + write a
//! `.version` marker so future PTX caches have a stable home.
//! Phase-2 (cuda-q4-matvec) will start populating it.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;

use super::error::CudaInitError;

/// Resolve the cache directory honouring `XDG_CACHE_HOME`. Defaults
/// to `$HOME/.cache/larql/cudarc/`.
pub fn cache_dir() -> PathBuf {
    if let Ok(xdg) = env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg).join("larql").join("cudarc")
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("larql")
            .join("cudarc")
    } else {
        PathBuf::from(".cache").join("larql").join("cudarc")
    }
}

/// Create the cache directory and write a `.version` marker. Idempotent.
pub fn ensure_initialised(_ctx: &Arc<CudaContext>) -> Result<(), CudaInitError> {
    let dir = cache_dir();
    fs::create_dir_all(&dir)
        .map_err(|e| CudaInitError::DriverMissing(format!("cache mkdir {dir:?}: {e}")))?;
    let marker = dir.join(".version");
    let mut f = fs::File::create(&marker)
        .map_err(|e| CudaInitError::DriverMissing(format!("cache marker {marker:?}: {e}")))?;
    // The marker is informational. Future schema changes bump the
    // larql-compute version; consumers that find a stale marker can
    // wipe the directory.
    writeln!(
        f,
        "larql-compute={}\ncudarc=0.19\nformat=v1",
        env!("CARGO_PKG_VERSION")
    )
    .ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_respects_xdg_cache_home() {
        // Set XDG_CACHE_HOME, read back path.
        // SAFETY: this test isolates env mutation per invocation.
        let prev = env::var("XDG_CACHE_HOME").ok();
        // SAFETY: env::set_var is unsafe in newer rustc; this is a test
        // and we restore at end. If multiple test threads collide, the
        // assertion still holds because both threads observe the same
        // value.
        unsafe {
            env::set_var("XDG_CACHE_HOME", "/tmp/test-xdg-cache");
        }
        let dir = cache_dir();
        assert!(dir.starts_with("/tmp/test-xdg-cache/larql/cudarc"));
        unsafe {
            match prev {
                Some(v) => env::set_var("XDG_CACHE_HOME", v),
                None => env::remove_var("XDG_CACHE_HOME"),
            }
        }
    }

    #[test]
    fn cache_dir_falls_back_to_home() {
        let prev_xdg = env::var("XDG_CACHE_HOME").ok();
        let prev_home = env::var("HOME").ok();
        unsafe {
            env::remove_var("XDG_CACHE_HOME");
            env::set_var("HOME", "/tmp/test-home");
        }
        let dir = cache_dir();
        assert_eq!(dir, PathBuf::from("/tmp/test-home/.cache/larql/cudarc"));
        unsafe {
            match prev_xdg {
                Some(v) => env::set_var("XDG_CACHE_HOME", v),
                None => env::remove_var("XDG_CACHE_HOME"),
            }
            match prev_home {
                Some(v) => env::set_var("HOME", v),
                None => env::remove_var("HOME"),
            }
        }
    }
}
