//! Datalog rules for wasm32 call-graph closure verification.
//!
//! The certification criterion: the call graph of the wasm32 artifact must
//! be statically resolvable, fully monomorphized, and closed under the
//! sandbox boundary.
//!
//! Violations:
//!   - `violates_containment`: function transitively calls a non-intrinsic
//!     host import or contains inline asm
//!   - `unresolved_dispatch`: function contains a call_indirect instruction
//!     (dynamic dispatch unresolvable at this analysis level)
//!
//! Both are immediate refutation conditions — undecidability IS the
//! refutation condition, not an edge case.

use ascent::ascent;

ascent! {
    // ── Input facts (populated by certify.rs from wasm_facts) ────────────────

    /// Static call edge: caller calls callee (both are function indices).
    relation calls(u32, u32);

    /// Function index that imports a non-intrinsic host symbol.
    relation is_non_intrinsic_import(u32);

    /// Function index that contains at least one call_indirect.
    relation has_indirect_call(u32);

    /// Wasm export (call graph root): (export_name, func_index).
    relation is_root(u32);

    // ── Derived: transitive call closure ─────────────────────────────────────

    /// Transitive call: a can reach b through any number of static calls.
    relation calls_tc(u32, u32);
    calls_tc(a, b) <-- calls(a, b);
    calls_tc(a, c) <-- calls_tc(a, b), calls(b, c);

    // ── Containment violation (non-intrinsic import, transitively reachable) ─

    relation violates_containment(u32);
    violates_containment(f) <-- is_non_intrinsic_import(f);
    violates_containment(caller) <--
        calls_tc(caller, callee),
        violates_containment(callee);

    // ── Unresolved dispatch (dynamic dispatch = refutation condition) ─────────

    relation unresolved_dispatch(u32);
    unresolved_dispatch(f) <-- has_indirect_call(f);

    // ── Refuted: witness set reachable from wasm exports ─────────────────────

    /// A function is refuted if it is reachable from any root AND it either
    /// violates containment or uses unresolved dynamic dispatch.
    relation refuted(u32);
    refuted(f) <--
        is_root(root),
        calls_tc(root, f),
        violates_containment(f);
    refuted(f) <--
        is_root(root),
        calls_tc(root, f),
        unresolved_dispatch(f);
    // Roots themselves can be refuted too.
    refuted(root) <--
        is_root(root),
        violates_containment(root);
    refuted(root) <--
        is_root(root),
        unresolved_dispatch(root);
}

/// Result of running the Datalog analysis.
pub struct AnalysisResult {
    pub prog: AscentProgram,
}

impl AnalysisResult {
    /// True iff the call graph is closed and certified.
    pub fn is_certified(&self) -> bool {
        self.prog.refuted.is_empty()
    }

    /// Refuted function indices.
    pub fn refuted_indices(&self) -> Vec<u32> {
        self.prog.refuted.iter().map(|(idx,)| *idx).collect()
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
