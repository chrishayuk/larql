//! MIR-level fact extractor for nightly RTA analysis.
//!
//! Feature-gated: `--features mir-analysis` (requires nightly Rust).
//!
//! Walks MIR terminators to classify each `call_indirect` site as:
//!   - `dyn Trait` dispatch  → `rta_candidate(call_site, trait_id)`
//!   - `fn ptr` dispatch     → `unresolvable_dispatch(call_site)`
//!
//! Then enumerates impl blocks within the wasm32 partition:
//!   - `rta_concrete_type(trait_id, type_id)` — concrete implementing types
//!
//! These facts are fed back into the Datalog rules to subdivide DYNAMIC into
//! DYNAMIC.DECIDABLE (all dispatch RTA-resolved within partition) and
//! DYNAMIC.UNDECIDABLE (fn-ptr or unresolvable dyn).
//!
//! On stable (feature absent): this module is empty; certify.rs reports DYNAMIC.

#[cfg(feature = "mir-analysis")]
pub use nightly::extract_mir_facts;

#[cfg(feature = "mir-analysis")]
mod nightly {
    use anyhow::Result;

    /// MIR-level dispatch facts for a single crate.
    pub struct MirFacts {
        /// (call_site_func_idx, trait_id): dyn Trait call sites and their trait.
        pub rta_candidates: Vec<(u32, u32)>,
        /// fn-pointer call sites: dispatch target cannot be bounded.
        pub unresolvable_dispatches: Vec<u32>,
        /// (trait_id, type_id): concrete types implementing a trait, within partition.
        pub rta_concrete_types: Vec<(u32, u32)>,
    }

    /// Extract MIR-level dispatch facts for the named crate.
    ///
    /// Uses `stable_mir` to walk MIR terminators.  Requires nightly Rust and the
    /// `mir-analysis` cargo feature.
    pub fn extract_mir_facts(_crate_name: &str) -> Result<MirFacts> {
        // stable_mir integration: walk each MonoItem::Fn, inspect its
        // TerminatorKind::Call — if the callee is a Ty::Dynamic (dyn Trait),
        // emit rta_candidate; if it is a Ty::FnPtr, emit unresolvable_dispatch.
        // For each impl block, emit rta_concrete_type.
        //
        // This is a placeholder until stable_mir lands on stable channel.
        // The nightly CI matrix cell enables this feature flag.
        anyhow::bail!("mir-analysis not yet implemented — nightly stable_mir integration pending")
    }
}
