use std::path::Path;

use wasmi::{Engine, Instance, Linker, Module, Store};

use super::wasi_shim;

/// Per-instance store data. Empty for now; WASI state is handled statelessly
/// by the shim closures (they write directly to host stdout/stderr).
pub struct ExpertStore {}

/// Compile a WASM expert's `Module` from the given `.wasm` file.
pub fn load_module(engine: &Engine, path: &Path) -> anyhow::Result<Module> {
    let bytes = std::fs::read(path)?;
    Module::new(engine, &bytes).map_err(Into::into)
}

/// Instantiate a previously compiled `Module` with a fresh WASI context.
pub fn instantiate(
    engine: &Engine,
    module: &Module,
) -> anyhow::Result<(Store<ExpertStore>, Instance)> {
    let mut store = Store::new(engine, ExpertStore {});
    let mut linker: Linker<ExpertStore> = Linker::new(engine);
    wasi_shim::add_to_linker(&mut linker)?;
    let instance = linker.instantiate_and_start(&mut store, module)?;
    Ok((store, instance))
}

/// Compile and instantiate a WASM expert in one step.
pub fn load_expert(engine: &Engine, path: &Path) -> anyhow::Result<(Store<ExpertStore>, Instance)> {
    let module = load_module(engine, path)?;
    instantiate(engine, &module)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {

    fn fresh_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larql_loader_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    // The .cwasm cache is gone; keep a trivial smoke test so the module
    // compiles and the loader functions are reachable.
    #[test]
    fn fresh_path_is_unique() {
        let a = fresh_path("a");
        let b = fresh_path("b");
        assert_ne!(a, b);
    }
}
