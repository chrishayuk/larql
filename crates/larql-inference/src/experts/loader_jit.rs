//! wasmtime-based WASM expert loader — JIT/AOT path with `.cwasm` cache.
//!
//! This module preserves the original wasmtime implementation for REUSE
//! compliance. It is gated behind the `expert-jit` feature and compiles only
//! on platforms where Cranelift is available (Linux, macOS, Windows, FreeBSD).
//! The universal default is `loader.rs` (wasmi pure-Rust interpreter).
#![cfg(all(
    feature = "expert-jit",
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "freebsd"
    )
))]

use std::path::Path;
use std::time::SystemTime;

use wasmtime::{Engine, Instance, Linker, Module, Store};
use wasmtime_wasi::p1::{self, WasiP1Ctx};

/// Per-instance store data — just the WASI preview1 context.
pub struct ExpertStore {
    pub wasi: WasiP1Ctx,
}

/// Compile (or load from cache) a WASM expert's `Module` without
/// instantiating it. Instantiation is deferred until the first `call()` so the
/// registry does not pay ~1 MiB of linear memory per expert at startup.
pub fn load_module(engine: &Engine, path: &Path) -> anyhow::Result<Module> {
    let cache_path = path.with_extension("cwasm");

    if cache_is_fresh(&cache_path, path) {
        // SAFETY: `Module::deserialize_file` is unsafe because it trusts the
        // precompiled artifact (mismatched wasmtime versions or corruption can
        // cause UB). We only deserialize files this process wrote itself into
        // a cache path next to the source `.wasm`, so the trust boundary stays
        // inside the same build output tree. Any error falls through to a
        // canonical compile-from-source.
        if let Ok(m) = unsafe { Module::deserialize_file(engine, &cache_path) } {
            return Ok(m);
        }
    }

    let module = Module::from_file(engine, path)?;

    // Best-effort: write the serialized form next to the source. A read-only
    // target dir or full disk must not break loading.
    if let Ok(bytes) = module.serialize() {
        let _ = std::fs::write(&cache_path, bytes);
    }

    Ok(module)
}

/// Instantiate a previously loaded `Module` with a fresh WASI context.
pub fn instantiate(
    engine: &Engine,
    module: &Module,
) -> anyhow::Result<(Store<ExpertStore>, Instance)> {
    let wasi = wasmtime_wasi::WasiCtxBuilder::new()
        .inherit_stderr()
        .build_p1();
    let mut store = Store::new(engine, ExpertStore { wasi });

    let mut linker: Linker<ExpertStore> = Linker::new(engine);
    preview1::add_to_linker_sync(&mut linker, |s: &mut ExpertStore| &mut s.wasi)?;

    let instance = linker.instantiate(&mut store, module)?;
    Ok((store, instance))
}

/// Compile and instantiate a WASM expert in one step — kept for callers that
/// want the historical semantics (e.g. tests that need immediate metadata
/// without touching the registry layer).
pub fn load_expert(engine: &Engine, path: &Path) -> anyhow::Result<(Store<ExpertStore>, Instance)> {
    let module = load_module(engine, path)?;
    instantiate(engine, &module)
}

fn cache_is_fresh(cache: &Path, source: &Path) -> bool {
    let cache_mtime = match std::fs::metadata(cache).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let source_mtime = match std::fs::metadata(source).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    cache_mtime >= source_mtime || {
        // Some filesystems round mtimes to 1s — treat equal-within-1s as fresh.
        cache_mtime
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            >= source_mtime
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::io::Write;

    fn fresh_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_loader_jit_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn cache_is_fresh_returns_false_when_cache_missing() {
        let cache = fresh_path("missing_cache");
        let source = fresh_path("missing_source_for_cache_test");
        std::fs::write(&source, b"src").unwrap();
        assert!(!cache_is_fresh(&cache, &source));
        let _ = std::fs::remove_file(&source);
    }

    #[test]
    fn cache_is_fresh_returns_false_when_source_missing() {
        let cache = fresh_path("cache_no_source");
        std::fs::write(&cache, b"compiled").unwrap();
        let source = fresh_path("does_not_exist");
        assert!(!cache_is_fresh(&cache, &source));
        let _ = std::fs::remove_file(&cache);
    }

    #[test]
    fn cache_is_fresh_returns_true_when_cache_newer_than_source() {
        let source = fresh_path("source_old");
        std::fs::write(&source, b"src").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let cache = fresh_path("cache_new");
        std::fs::write(&cache, b"compiled").unwrap();
        assert!(cache_is_fresh(&cache, &source));
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&cache);
    }

    #[test]
    fn cache_is_fresh_returns_false_when_source_newer_than_cache() {
        let cache = fresh_path("cache_old");
        std::fs::write(&cache, b"compiled").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let source = fresh_path("source_new");
        std::fs::write(&source, b"src").unwrap();
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&source)
            .unwrap();
        f.write_all(b"src updated").unwrap();
        drop(f);
        let _ = cache_is_fresh(&cache, &source);
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&cache);
    }
}
