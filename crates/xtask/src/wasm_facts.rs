//! Extract Datalog facts from a wasm32 binary using wasmparser.
//!
//! The compiled wasm binary is already fully monomorphized.  We walk:
//!   - ImportSection  → non-intrinsic imports (containment violations)
//!   - ExportSection  → root functions (call graph entry points)
//!   - CodeSection    → Call + CallIndirect instructions (call graph edges)
//!
//! The resulting facts feed the ascent rules in `rules.rs`.

use anyhow::Result;
use std::collections::HashMap;
use wasmparser::{BinaryReaderError, Operator, Parser, Payload};

/// All Datalog facts extracted from a single wasm binary.
#[derive(Default, Debug)]
pub struct WasmFacts {
    /// (caller_idx, callee_idx) for static Call instructions.
    pub calls: Vec<(u32, u32)>,
    /// Function indices that contain at least one call_indirect.
    pub indirect_calls: Vec<u32>,
    /// Function indices whose bodies contain inline asm (wasm `unreachable`
    /// emitted by `asm!` blocks is not detectable here; asm! compiles to LLVM
    /// which is then lowered — we detect via import names instead).
    pub inline_asm: Vec<u32>,
    /// Import entries that are NOT in the intrinsic whitelist.
    /// `(module, name, func_index)`
    pub non_intrinsic_imports: Vec<(String, String, u32)>,
    /// Export entries that are functions — the call graph roots.
    /// `(name, func_index)`
    pub roots: Vec<(String, u32)>,
    /// `func_index → human-readable name` (from the name section, if present).
    pub names: HashMap<u32, String>,
    /// Total number of imported functions (to offset local function indices).
    pub num_imports: u32,
}

/// Module/name patterns that are part of the wasm-bindgen + getrandom intrinsic
/// set.  Anything outside this list is a containment-violation witness.
fn is_intrinsic(module: &str, name: &str) -> bool {
    matches!(
        module,
        "__wbindgen_placeholder__" | "__wbindgen_externref_xform__" | "wbg"
    ) || name.starts_with("__wbg_")
        || name.starts_with("__wbindgen_")
        || module == "__wbg_getrandomvalues"
}

/// Record one function import, bumping the count and noting non-intrinsics.
fn register_import(
    facts: &mut WasmFacts,
    count: &mut u32,
    module: &str,
    name: &str,
    ty: wasmparser::TypeRef,
) {
    if let wasmparser::TypeRef::Func(_) = ty {
        let idx = *count;
        *count += 1;
        if !is_intrinsic(module, name) {
            facts
                .non_intrinsic_imports
                .push((module.to_owned(), name.to_owned(), idx));
        }
    }
}

pub fn extract(wasm_bytes: &[u8]) -> Result<WasmFacts> {
    let mut facts = WasmFacts::default();

    // We need to number functions ourselves: imported functions come first,
    // then locally defined functions in order of their code section entries.
    let mut import_func_count: u32 = 0;
    let mut local_func_idx: u32 = 0; // incremented as we process CodeSectionEntry

    for payload in Parser::new(0).parse_all(wasm_bytes) {
        let payload = payload.map_err(|e: BinaryReaderError| anyhow::anyhow!("{e}"))?;
        match payload {
            Payload::ImportSection(reader) => {
                for item in reader {
                    let item = item.map_err(|e: BinaryReaderError| anyhow::anyhow!("{e}"))?;
                    match item {
                        wasmparser::Imports::Single(_offset, import) => {
                            register_import(&mut facts, &mut import_func_count, import.module, import.name, import.ty);
                        }
                        wasmparser::Imports::Compact1 { module, items } => {
                            for compact_item in items {
                                let ci = compact_item.map_err(|e: BinaryReaderError| anyhow::anyhow!("{e}"))?;
                                register_import(&mut facts, &mut import_func_count, module, ci.name, ci.ty);
                            }
                        }
                        wasmparser::Imports::Compact2 { module, ty, names } => {
                            for name in names {
                                let name = name.map_err(|e: BinaryReaderError| anyhow::anyhow!("{e}"))?;
                                register_import(&mut facts, &mut import_func_count, module, name, ty);
                            }
                        }
                    }
                }
                facts.num_imports = import_func_count;
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export =
                        export.map_err(|e: BinaryReaderError| anyhow::anyhow!("{e}"))?;
                    if let wasmparser::ExternalKind::Func = export.kind {
                        facts.roots.push((export.name.to_owned(), export.index));
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                // Local functions are numbered import_func_count + local_func_idx.
                // We don't know import_func_count yet if the import section
                // hasn't been parsed — but ImportSection always comes first in
                // a valid wasm binary per the spec.
                let func_idx = facts.num_imports + local_func_idx;
                local_func_idx += 1;

                let mut ops = body
                    .get_operators_reader()
                    .map_err(|e: BinaryReaderError| anyhow::anyhow!("{e}"))?;
                loop {
                    let op = ops
                        .read()
                        .map_err(|e: BinaryReaderError| anyhow::anyhow!("{e}"))?;
                    match op {
                        Operator::Call { function_index } => {
                            facts.calls.push((func_idx, function_index));
                        }
                        Operator::CallIndirect { .. } => {
                            facts.indirect_calls.push(func_idx);
                        }
                        Operator::End => break,
                        _ => {}
                    }
                }
            }
            // Name section is optional and its API varies across wasmparser
            // versions — skip it; we fall back to func#N labels.
            _ => {}
        }
    }

    Ok(facts)
}

/// Build a human-readable label for a function index.
pub fn label(facts: &WasmFacts, idx: u32) -> String {
    if let Some(name) = facts.names.get(&idx) {
        return name.clone();
    }
    if idx < facts.num_imports {
        // Find import by index
        for (module, name, func_idx) in &facts.non_intrinsic_imports {
            if *func_idx == idx {
                return format!("{module}::{name}");
            }
        }
        format!("import#{idx}")
    } else {
        format!("func#{idx}")
    }
}
