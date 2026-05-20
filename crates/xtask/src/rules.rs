//! Datalog rules for wasm32 call-graph closure verification.
//!
//! Three-layer partition:
//!   STATIC  — call graph closed, no call_indirect, no non-intrinsic imports
//!   DYNAMIC — call_indirect present, but no containment violation
//!             (stable path: unclassified; nightly: DECIDABLE or UNDECIDABLE)
//!   NATIVE  — non-intrinsic host imports reachable from exports (sandbox breach)
//!
//! The `ascent` macro IS the resolution system — Horn clause resolution over a
//! finite closed-world fact base.  Input facts from wasm_facts.rs; optional
//! MIR-level facts from mir_facts.rs (nightly `mir-analysis` feature).

use ascent::ascent;

ascent! {
    // ── Input facts (populated by certify.rs from wasm_facts) ────────────────

    /// Static call edge: caller calls callee (both are function indices).
    relation calls(u32, u32);

    /// Function index that imports a non-intrinsic host symbol.
    relation is_non_intrinsic_import(u32);

    /// Function index that contains at least one call_indirect.
    relation has_indirect_call(u32);

    /// Wasm export (call graph root): func_index.
    relation is_root(u32);

    // ── Derived: transitive call closure ─────────────────────────────────────

    /// Transitive call: a can reach b through any number of static calls.
    relation calls_tc(u32, u32);
    calls_tc(a, b) <-- calls(a, b);
    calls_tc(a, c) <-- calls_tc(a, b), calls(b, c);

    // ── Containment violation ─────────────────────────────────────────────────
    // A non-intrinsic host import IS a sandbox breach — the only true source of
    // undecidability in this analysis (OS/IO resources outside the wasm sandbox).

    relation violates_containment(u32);
    violates_containment(f) <-- is_non_intrinsic_import(f);
    violates_containment(caller) <--
        calls_tc(caller, callee),
        violates_containment(callee);

    // ── Unresolved dispatch ───────────────────────────────────────────────────
    // call_indirect = dynamic dispatch.  Stable path: unclassified (neither
    // bounded nor unbounded proven).  Nightly mir-analysis subdivides these.

    relation unresolved_dispatch(u32);
    unresolved_dispatch(f) <-- has_indirect_call(f);

    // ── Containment violation witnesses (reachable from exports) ─────────────
    // These belong in the NATIVE layer: they breach the wasm sandbox boundary.

    relation containment_violation(u32);
    containment_violation(f) <--
        is_root(root),
        calls_tc(root, f),
        violates_containment(f);
    containment_violation(root) <--
        is_root(root),
        violates_containment(root);

    // ── Dispatch witnesses (reachable from exports) ───────────────────────────
    // These belong in the DYNAMIC layer: dispatch not statically resolved.
    // Separate from containment violations — dynamic dispatch within the wasm
    // sandbox is still valid wasm, just not STATIC-partition wasm.

    relation dispatch_witness(u32);
    dispatch_witness(f) <--
        is_root(root),
        calls_tc(root, f),
        unresolved_dispatch(f);
    dispatch_witness(root) <--
        is_root(root),
        unresolved_dispatch(root);
}

/// Partition label assigned to a wasm32 crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmPartition {
    /// Call graph closed: no call_indirect, no non-intrinsic imports.
    /// The strict wasm32-unknown-unknown static runtime.
    Static,
    /// call_indirect present, all dispatch proven bounded within partition (MIR/RTA).
    /// Part of the wasm32 dynamic runtime; still fully sandbox-contained.
    DynamicDecidable,
    /// call_indirect present, fn-ptr dispatch or unresolvable dyn (MIR/RTA).
    /// Dispatch targets cannot be fully bounded; may extend beyond the partition.
    DynamicUndecidable,
    /// call_indirect present, nightly MIR analysis not run — dispatch unclassified.
    Dynamic,
    /// Non-intrinsic host imports reachable from exports, or no lib target.
    /// Belongs in the host OS/IO layer, not in any wasm runtime layer.
    Native,
}

impl std::fmt::Display for WasmPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmPartition::Static => write!(f, "STATIC"),
            WasmPartition::DynamicDecidable => write!(f, "DYNAMIC.DECIDABLE"),
            WasmPartition::DynamicUndecidable => write!(f, "DYNAMIC.UNDECIDABLE"),
            WasmPartition::Dynamic => write!(f, "DYNAMIC"),
            WasmPartition::Native => write!(f, "NATIVE"),
        }
    }
}

/// Result of running the Datalog analysis.
pub struct AnalysisResult {
    pub prog: AscentProgram,
}

impl AnalysisResult {
    /// True iff no containment violations are reachable from exports.
    pub fn is_sandbox_contained(&self) -> bool {
        self.prog.containment_violation.is_empty()
    }

    /// True iff no dispatch witnesses are reachable from exports.
    pub fn is_statically_resolved(&self) -> bool {
        self.prog.dispatch_witness.is_empty()
    }

    pub fn containment_violation_indices(&self) -> Vec<u32> {
        self.prog.containment_violation.iter().map(|(idx,)| *idx).collect()
    }

    pub fn dispatch_witness_indices(&self) -> Vec<u32> {
        self.prog.dispatch_witness.iter().map(|(idx,)| *idx).collect()
    }

    /// Assign the stable-path partition label (no MIR facts available).
    pub fn partition_stable(&self) -> WasmPartition {
        if !self.is_sandbox_contained() {
            WasmPartition::Native
        } else if !self.is_statically_resolved() {
            WasmPartition::Dynamic
        } else {
            WasmPartition::Static
        }
    }
}

/// Run the Datalog rules given populated input facts.
pub fn analyze(
    calls: Vec<(u32, u32)>,
    non_intrinsic_imports: Vec<u32>,
    indirect_calls: Vec<u32>,
    roots: Vec<u32>,
) -> AnalysisResult {
    let mut prog = AscentProgram::default();
    prog.calls = calls;
    prog.is_non_intrinsic_import = non_intrinsic_imports.into_iter().map(|x| (x,)).collect();
    prog.has_indirect_call = indirect_calls.into_iter().map(|x| (x,)).collect();
    prog.is_root = roots.into_iter().map(|x| (x,)).collect();
    prog.run();
    AnalysisResult { prog }
}
