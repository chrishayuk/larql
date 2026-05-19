//! larql-browser-cli — in-browser WASM CLI for LARQL.
//!
//! Exposes LQL parsing and in-memory session execution via wasm-bindgen.
//! The OS-side binary (`larql-cli`) is a separate crate and is unaffected.

use wasm_bindgen::prelude::*;

/// Re-export the thread-pool initialiser for the parallel variant.
/// JS callers must `await initThreadPool(n)` before using parallel operations.
#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Parse an LQL statement string.
///
/// Returns the debug representation of the parsed AST on success, or
/// a human-readable parse error on failure.
#[wasm_bindgen]
pub fn parse_lql(input: &str) -> Result<String, JsValue> {
    larql_lql::parse(input)
        .map(|stmt| format!("{stmt:?}"))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// In-browser LQL session backed by an in-memory state machine.
///
/// On construction the session has no backend (`Backend::None`). Statements
/// that require a loaded vindex (WALK, DESCRIBE, SELECT, INFER, …) will
/// return an error until a backend is installed.
///
/// File-IO commands (USE, EXTRACT, COMPILE, SAVE PATCH) are not available in
/// this environment; they will return a meaningful error rather than panic.
#[wasm_bindgen]
pub struct LqlSession {
    inner: larql_lql::Session,
}

#[wasm_bindgen]
impl LqlSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: larql_lql::Session::new(),
        }
    }

    /// Parse and execute an LQL statement string.
    ///
    /// Returns the output lines as a JSON array of strings, or a JS error
    /// string on parse or execution failure.
    pub fn execute(&mut self, lql: &str) -> Result<String, JsValue> {
        let stmt = larql_lql::parse(lql).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let lines = self
            .inner
            .execute(&stmt)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_json::to_string(&lines).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for LqlSession {
    fn default() -> Self {
        Self::new()
    }
}
